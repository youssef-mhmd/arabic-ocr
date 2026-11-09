"""Generate a synthetic line-image dataset from a config file.

Usage:
    uv run python -m training.make_dataset training/configs/data_sample.yaml
    uv run python -m training.make_dataset CONFIG --workers 8     # use 8 CPU cores
    uv run python -m training.make_dataset CONFIG --overwrite     # start over

Steps:
1. Text comes from prepared corpora (`uv run python -m training.corpora prepare`),
   which are already cleaned and split into train/val/test paragraphs, so no
   paragraph is in two splits. `text_sources` says which corpora to use and how
   often to pick each one.
2. For each split, make `samples[split]` images in shards of `shard_size`. Each shard
   has its own seed derived from the config seed, so shards can be made in any order,
   in parallel, and re-made identically.
3. Write manifest.json with the settings, counts, and charset fingerprint.

Resuming: finished shards are kept and skipped, so after a crash or Ctrl+C just run
the same command again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml

from arabic_ocr import charset
from training.augment import AugmentConfig, augment
from training.corpora import CORPORA_DIR, LineFile, split_file
from training.dataset import SPLITS, ShardWriter, charset_fingerprint
from training.fonts import MANIFEST as FONTS_MANIFEST
from training.fonts import FontPicker, load_fonts
from training.render import MissingGlyphError, render_line
from training.text import LineSampler, SamplerConfig

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "data" / "datasets"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "name",
        "seed",
        "text_sources",
        "samples",
        "shard_size",
        "image_height",
        "font_size_px",
        "sampler",
        "augment",
    }
    if missing := required - cfg.keys():
        raise ValueError(f"config is missing: {', '.join(sorted(missing))}")
    if unknown := cfg.keys() - required:
        raise ValueError(f"config has unknown keys: {', '.join(sorted(unknown))}")
    # Fail early on typos in the nested settings (dataclasses reject unknown fields).
    SamplerConfig(**cfg["sampler"])
    AugmentConfig(**{k: v for k, v in cfg["augment"].items() if k != "enabled"})
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def shard_seed(base_seed: int, split: str, index: int) -> int:
    """A different, reproducible seed for every shard."""
    digest = hashlib.sha256(f"{base_seed}/{split}/{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


class TextPicker:
    """Picks a random paragraph: first a corpus (by weight), then a line in it."""

    def __init__(self, sources: list[dict], split: str, corpora_dir: Path) -> None:
        self.files, self.weights = [], []
        for src in sources:
            path = split_file(src["name"], split, corpora_dir)
            if path.exists() and path.stat().st_size > 0:
                self.files.append(LineFile(path))
                self.weights.append(src["weight"])
        if not self.files:
            raise FileNotFoundError(f"no prepared text for split '{split}' in {corpora_dir}")

    def pick(self, rng: random.Random) -> str:
        return rng.choices(self.files, weights=self.weights)[0].random_line(rng)


# Each worker process opens the corpus files once and reuses them for every shard.
_PICKERS: dict[tuple, TextPicker] = {}


def text_picker(cfg: dict, split: str, corpora_dir: Path) -> TextPicker:
    key = (json.dumps(cfg["text_sources"], sort_keys=True), split, str(corpora_dir))
    if key not in _PICKERS:
        _PICKERS[key] = TextPicker(cfg["text_sources"], split, corpora_dir)
    return _PICKERS[key]


def text_counts(cfg: dict, corpora_dir: Path) -> dict[str, dict[str, int]]:
    """Paragraphs available per split and source (0 if not prepared)."""
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts[split] = {}
        for src in cfg["text_sources"]:
            path = split_file(src["name"], split, corpora_dir)
            n = 0
            if path.exists():
                with open(path, "rb") as f:
                    n = sum(1 for _ in f)
            counts[split][src["name"]] = n
    return counts


# ---------------------------------------------------------------------------
# One shard (runs in a worker process)
# ---------------------------------------------------------------------------


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    h, w = image.shape
    new_w = max(1, round(w * height / h))
    interp = cv2.INTER_AREA if h > height else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, height), interpolation=interp)


def make_shard(
    cfg: dict, split: str, index: int, count: int, out: Path, corpora_dir: Path = CORPORA_DIR
) -> int:
    seed = shard_seed(cfg["seed"], split, index)
    rng = random.Random(seed)
    texts = text_picker(cfg, split, corpora_dir)
    np_rng = np.random.default_rng(seed)
    sampler = LineSampler(SamplerConfig(**cfg["sampler"]), seed=seed)
    picker = FontPicker(load_fonts(), rng)
    aug_settings = {k: v for k, v in cfg["augment"].items() if k != "enabled"}
    aug_cfg = AugmentConfig(**aug_settings)
    aug_on = cfg["augment"].get("enabled", True)
    lo, hi = cfg["font_size_px"]

    with ShardWriter(out) as writer:
        attempts = 0
        while writer.count < count:
            attempts += 1
            if attempts > count * 20:
                raise RuntimeError(f"{split} shard {index}: too many failed samples")
            line = sampler.sample(texts.pick(rng))
            if line is None:
                continue
            font = picker.pick(line)
            if font is None:
                continue
            size = rng.randint(lo, hi)
            try:
                ink = render_line(line, font.path, size_px=size)
            except MissingGlyphError:
                continue
            image = augment(ink, np_rng, aug_cfg) if aug_on else 255 - ink
            image = resize_to_height(image, cfg["image_height"])
            meta = {
                "font": font.path.name,
                "family": font.family,
                "style": font.style,
                "role": font.role,
                "size_px": size,
            }
            writer.write(f"{split}-{index:05d}-{writer.count:05d}", image, line, meta)
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic line-image dataset.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--overwrite", action="store_true", help="delete existing shards first")
    parser.add_argument("--out-dir", type=Path, default=DATASETS_DIR)
    parser.add_argument("--corpora-dir", type=Path, default=CORPORA_DIR)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = args.out_dir / cfg["name"]
    manifest_path = out_dir / "manifest.json"
    chash = config_hash(cfg)

    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["config_hash"] != chash or old["charset"] != charset_fingerprint():
            if not args.overwrite:
                print(
                    f"{out_dir} was made with different settings. "
                    "Use --overwrite to start over, or change `name`."
                )
                return 1
    if args.overwrite:
        for split in SPLITS:
            for p in (out_dir / split).glob("shard-*.tar*"):
                p.unlink()

    paragraphs = text_counts(cfg, args.corpora_dir)
    for split in SPLITS:
        if cfg["samples"][split] > 0 and not any(paragraphs[split].values()):
            print(
                f"no prepared text for '{split}'. Run: uv run python -m training.corpora "
                f"prepare {' '.join(s['name'] for s in cfg['text_sources'])}"
            )
            return 1
    for split, per_source in paragraphs.items():
        if missing := [n for n, c in per_source.items() if c == 0]:
            print(f"  note: no {split} text from {', '.join(missing)} (not prepared?)")

    manifest = {
        "name": cfg["name"],
        "status": "incomplete",
        "config": cfg,
        "config_hash": chash,
        "charset": charset_fingerprint(),
        "num_classes": charset.NUM_CLASSES,
        "fonts_commit": json.loads(FONTS_MANIFEST.read_text(encoding="utf-8"))["source_commit"],
        "paragraphs": paragraphs,
        "samples": {},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    jobs = []
    for split in SPLITS:
        total, size = cfg["samples"][split], cfg["shard_size"]
        for i in range(math.ceil(total / size)):
            path = out_dir / split / f"shard-{i:05d}.tar"
            count = min(size, total - i * size)
            if not path.exists():
                jobs.append((split, i, count, path))
    print(f"{cfg['name']}: {len(jobs)} shards to make with {args.workers} workers")

    start, done_samples = time.perf_counter(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(make_shard, cfg, split, i, count, path, args.corpora_dir): (split, i)
            for split, i, count, path in jobs
        }
        for n, fut in enumerate(as_completed(futures), 1):
            done_samples += fut.result()
            elapsed = time.perf_counter() - start
            rate = done_samples / elapsed
            print(
                f"  [{n}/{len(jobs)}] {futures[fut][0]} shard {futures[fut][1]} done "
                f"({rate:.0f} images/s)",
                flush=True,
            )

    manifest["samples"] = {s: cfg["samples"][s] for s in SPLITS}
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
