"""Dataset files: how samples are stored, read back, and split.

**Format: tar shards (the "WebDataset" layout).** A dataset is a folder like

    data/datasets/<name>/
        manifest.json               # settings, counts, charset fingerprint
        train/shard-00000.tar
        train/shard-00001.tar
        val/shard-00000.tar
        test/shard-00000.tar

Each shard is an ordinary .tar file. Inside, every sample is three files sharing one
key:

    train-00000-00042.png    the grayscale line image
    train-00000-00042.txt    the label (UTF-8, logical order)
    train-00000-00042.json   extra info: font, size, ...

Why shards: copying or streaming a few hundred big files is far faster than millions
of tiny ones, and any tool that opens .tar files can inspect them.

**Splits without leaks.** A sentence must never appear in more than one split, or
the test score would be measuring memory, not reading. `split_of` assigns each
*source sentence* (before lines are cut from it) to train/val/test using a hash of its
text with diacritics removed, so the diacritized and plain versions of a sentence
always land in the same split. The hash makes the assignment stable: the same
sentence gets the same split on every machine, every run.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from arabic_ocr import charset

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def split_of(sentence: str, val_frac: float, test_frac: float) -> str:
    """Which split a source sentence belongs to ("train", "val", or "test")."""
    key = charset.strip_diacritics(sentence).encode("utf-8")
    # First 8 bytes of SHA-1 as a number in [0, 1): a stable, well-mixed "coin".
    bucket = int.from_bytes(hashlib.sha1(key).digest()[:8], "big") / 2**64
    if bucket < test_frac:
        return "test"
    if bucket < test_frac + val_frac:
        return "val"
    return "train"


def charset_fingerprint() -> str:
    """Short hash of the symbol list. A model must be trained and run with the same one."""
    return hashlib.sha1("\n".join(charset.SYMBOLS).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0  # fixed timestamp: the same inputs give byte-identical shards
    tar.addfile(info, io.BytesIO(data))


def encode_png(image: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()


class ShardWriter:
    """Writes one shard. Use as a context manager.

    The shard is written to `<name>.tar.tmp` and only renamed to `<name>.tar` when it
    is complete, so a crash never leaves a half-written shard that looks finished.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.tmp = path.with_suffix(".tar.tmp")
        self.count = 0
        self._tar: tarfile.TarFile | None = None

    def __enter__(self) -> ShardWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tar = tarfile.open(self.tmp, "w")
        return self

    def write(self, key: str, image: np.ndarray, label: str, meta: dict) -> None:
        assert self._tar is not None
        if image.dtype != np.uint8 or image.ndim != 2:
            raise ValueError("image must be a 2-D uint8 grayscale array")
        if charset.normalize(label) != label:
            raise ValueError(f"label is not normalized: {label!r}")
        charset.encode(label)  # raises if any character is outside the charset
        _add_bytes(self._tar, f"{key}.png", encode_png(image))
        _add_bytes(self._tar, f"{key}.txt", label.encode("utf-8"))
        _add_bytes(self._tar, f"{key}.json", json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        self.count += 1

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._tar is not None
        self._tar.close()
        if exc_type is None:
            self.tmp.replace(self.path)
        else:
            self.tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    key: str
    image: np.ndarray  # uint8, (H, W), white = 255
    label: str
    meta: dict


def read_shard(path: Path) -> Iterator[Sample]:
    """Yield the samples of one shard, in the order they were written."""
    parts: dict[str, dict[str, bytes]] = {}
    order: list[str] = []
    with tarfile.open(path, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            key, _, ext = member.name.rpartition(".")
            f = tar.extractfile(member)
            assert f is not None
            if key not in parts:
                parts[key] = {}
                order.append(key)
            parts[key][ext] = f.read()
    for key in order:
        p = parts[key]
        image = np.array(Image.open(io.BytesIO(p["png"])).convert("L"))
        yield Sample(
            key=key,
            image=image,
            label=p["txt"].decode("utf-8"),
            meta=json.loads(p["json"].decode("utf-8")),
        )


def shard_paths(dataset_dir: Path, split: str) -> list[Path]:
    return sorted((dataset_dir / split).glob("shard-*.tar"))


def read_split(dataset_dir: Path, split: str) -> Iterator[Sample]:
    for path in shard_paths(dataset_dir, split):
        yield from read_shard(path)
