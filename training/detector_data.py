"""Training examples for the detector: random page crops and their core maps.

For every page in the shards:
1. zoom by a random factor, so the detector sees text from tiny to large;
2. cut a random `crop` x `crop` window (padding with the page's border colour
   if the page is smaller);
3. draw the target: 1 inside every line's *core* (arabic_ocr/detect/boxes.py), 0
   elsewhere;
4. mark lines that became too small to read (under MIN_LINE_HEIGHT pixels) as
   "ignore": the loss doesn't count those pixels either way.

Shards are split between DataLoader workers; pages and shards are shuffled.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from arabic_ocr.detect.boxes import box_to_core, quad_to_rect
from training.make_pages import read_pages

MIN_LINE_HEIGHT = 6


def targets_for(lines: list[np.ndarray], h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """(core map, loss mask) for line quads already in crop coordinates."""
    gt = np.zeros((h, w), np.float32)
    mask = np.ones((h, w), np.float32)
    for quad in lines:
        if quad_to_rect(quad).h < MIN_LINE_HEIGHT:
            cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 0.0)
        else:
            cv2.fillPoly(gt, [np.round(box_to_core(quad)).astype(np.int32)], 1.0)
    return gt, mask


def random_crop(img, quads, rng: random.Random, crop: int, scale_range):
    s = rng.uniform(*scale_range)
    img = cv2.resize(
        img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    )
    quads = [q * s for q in quads]
    h, w = img.shape
    border = int(np.median(np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])))
    if h < crop or w < crop:
        img = cv2.copyMakeBorder(
            img, 0, max(0, crop - h), 0, max(0, crop - w), cv2.BORDER_CONSTANT, value=border
        )
        h, w = img.shape
    # Prefer windows that contain text: aim at a random line most of the time.
    if quads and rng.random() < 0.8:
        cx, cy = rng.choice(quads).mean(axis=0)
        x = int(np.clip(cx - rng.uniform(0.1, 0.9) * crop, 0, w - crop))
        y = int(np.clip(cy - rng.uniform(0.1, 0.9) * crop, 0, h - crop))
    else:
        x, y = rng.randint(0, w - crop), rng.randint(0, h - crop)
    window = img[y : y + crop, x : x + crop]
    shifted = [q - [x, y] for q in quads]
    return window, shifted


class PageCrops(IterableDataset):
    def __init__(
        self,
        shards: list[Path],
        crop: int = 640,
        scale_range=(0.45, 1.3),
        crops_per_page: int = 2,
        seed: int = 0,
        epoch: int = 0,
    ) -> None:
        self.shards = list(shards)
        self.crop = crop
        self.scale_range = scale_range
        self.crops_per_page = crops_per_page
        self.seed = seed
        self.epoch = epoch

    def __iter__(self):
        info = get_worker_info()
        wid, n = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 1000 + self.epoch * 100 + wid)
        shards = list(self.shards)
        random.Random(self.seed + self.epoch).shuffle(shards)
        buffer = []
        for shard in shards[wid::n]:
            for _key, img, meta in read_pages(shard):
                quads = [np.float32(ln["quad"]) for ln in meta["lines"]]
                for _ in range(self.crops_per_page):
                    window, q = random_crop(img, quads, rng, self.crop, self.scale_range)
                    gt, mask = targets_for(q, self.crop, self.crop)
                    buffer.append((window, gt, mask))
                if len(buffer) >= 64:  # mix crops from different pages
                    rng.shuffle(buffer)
                    while len(buffer) > 32:
                        yield self._tensors(*buffer.pop())
        rng.shuffle(buffer)
        for item in buffer:
            yield self._tensors(*item)

    @staticmethod
    def _tensors(window, gt, mask):
        return (
            torch.from_numpy(np.ascontiguousarray(window))[None],
            torch.from_numpy(gt)[None],
            torch.from_numpy(mask)[None],
        )
