"""Stream training batches from dataset shards.

Each worker process reads whole shards (fast, sequential disk reads), decodes the
PNGs, and turns labels into recognizer targets. Then, to keep the GPU busy:

- **Shuffling**: samples go into a buffer that is shuffled before use, and shard
  order is shuffled every epoch, so consecutive batches are not from the same shard.
- **Width bucketing**: images in a batch are padded to the widest one. Batching
  lines of similar width wastes much less work on padding.
- **Width budget**: a batch holds at most `width_budget` padded pixels of width in
  total (e.g. 128 lines x 512 px). Wide lines -> fewer per batch, so GPU memory use
  stays roughly constant.

Targets are the label in *reading order* (right to left; see arabic_ocr.bidi),
encoded as charset ids. A sample is skipped if its image is too narrow to hold its
target (CTC needs at least one output step per character); the count is reported.
"""

from __future__ import annotations

import random
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from arabic_ocr import charset
from arabic_ocr.bidi import reading_order
from training.model import INPUT_HEIGHT, WIDTH_STRIDE, min_steps_for, output_length

# Pad widths up to a multiple of this, so the GPU sees fewer distinct shapes.
WIDTH_ROUND = 32
assert WIDTH_ROUND % WIDTH_STRIDE == 0
MIN_WIDTH = 32


@dataclass
class Batch:
    images: torch.Tensor  # uint8 (B, 1, 64, W)
    targets: torch.Tensor  # int32, all targets concatenated
    target_lengths: torch.Tensor  # int32 (B,)
    input_lengths: torch.Tensor  # int32 (B,) valid output steps per line
    labels: list[str]  # logical-order labels, for measuring accuracy

    def pin_memory(self) -> Batch:
        self.images = self.images.pin_memory()
        return self


def iter_shard_samples(path: Path) -> Iterator[tuple[np.ndarray, str]]:
    """(image, label) pairs from one shard, decoded."""
    pending: dict[str, dict[str, bytes]] = {}
    with tarfile.open(path) as tar:
        for m in tar:
            key, _, ext = m.name.rpartition(".")
            if ext not in ("png", "txt"):
                continue
            f = tar.extractfile(m)
            if f is None:
                continue
            parts = pending.setdefault(key, {})
            parts[ext] = f.read()
            if len(parts) == 2:
                del pending[key]
                img = cv2.imdecode(np.frombuffer(parts["png"], np.uint8), cv2.IMREAD_GRAYSCALE)
                yield img, parts["txt"].decode("utf-8")


def make_target(label: str) -> list[int]:
    return charset.encode(reading_order(label))


def collate(items: list[tuple[np.ndarray, list[int], str]]) -> Batch:
    widths = [max(img.shape[1], MIN_WIDTH) for img, _, _ in items]
    max_w = -(-max(widths) // WIDTH_ROUND) * WIDTH_ROUND
    images = np.empty((len(items), 1, INPUT_HEIGHT, max_w), np.uint8)
    for b, (img, _, _) in enumerate(items):
        # Pad on the LEFT: the model reads right to left, so padding comes after the
        # end of the line, where the valid steps (input_lengths) have already ended.
        pad = max_w - img.shape[1]
        images[b, 0] = np.pad(img, ((0, 0), (pad, 0)), mode="edge")
    targets = [t for _, t, _ in items]
    return Batch(
        images=torch.from_numpy(images),
        targets=torch.tensor([i for t in targets for i in t], dtype=torch.int32),
        target_lengths=torch.tensor([len(t) for t in targets], dtype=torch.int32),
        input_lengths=torch.tensor([output_length(w) for w in widths], dtype=torch.int32),
        labels=[label for _, _, label in items],
    )


class ShardBatches(IterableDataset):
    """Yields ready-made `Batch`es. Use with DataLoader(batch_size=None)."""

    def __init__(
        self,
        shards: list[Path],
        width_budget: int,
        max_batch: int,
        shuffle: bool,
        seed: int,
        epoch: int = 0,
        buffer_size: int = 4096,
    ) -> None:
        self.shards = list(shards)
        self.width_budget = width_budget
        self.max_batch = max_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch
        self.buffer_size = buffer_size

    def _my_shards(self) -> list[Path]:
        shards = list(self.shards)
        if self.shuffle:
            random.Random(self.seed * 1000 + self.epoch).shuffle(shards)
        info = get_worker_info()
        if info is None:
            return shards
        return shards[info.id :: info.num_workers]  # each worker gets its own shards

    def _batches_from(self, items: list) -> Iterator[Batch]:
        items.sort(key=lambda it: it[0].shape[1])  # similar widths together
        batches, cur, cur_max = [], [], 0
        for it in items:
            w = -(-max(it[0].shape[1], MIN_WIDTH) // WIDTH_ROUND) * WIDTH_ROUND
            if cur and (
                len(cur) >= self.max_batch or max(cur_max, w) * (len(cur) + 1) > self.width_budget
            ):
                batches.append(cur)
                cur, cur_max = [], 0
            cur.append(it)
            cur_max = max(cur_max, w)
        if cur:
            batches.append(cur)
        if self.shuffle:
            self._rng.shuffle(batches)
        for b in batches:
            yield collate(b)

    def __iter__(self) -> Iterator[Batch]:
        info = get_worker_info()
        wid = info.id if info else 0
        self._rng = random.Random(self.seed * 1000 + self.epoch * 100 + wid)
        buffer: list = []
        self.skipped = 0
        for shard in self._my_shards():
            for img, label in iter_shard_samples(shard):
                target = make_target(label)
                if output_length(max(img.shape[1], MIN_WIDTH)) < min_steps_for(target):
                    self.skipped += 1
                    continue
                buffer.append((img, target, label))
                if len(buffer) >= self.buffer_size:
                    if self.shuffle:
                        self._rng.shuffle(buffer)
                    half = self.buffer_size // 2  # keep half for better mixing
                    yield from self._batches_from(buffer[:half])
                    buffer = buffer[half:]
        if buffer:
            if self.shuffle:
                self._rng.shuffle(buffer)
            yield from self._batches_from(buffer)
