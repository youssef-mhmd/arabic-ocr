"""Generate a synthetic *page* dataset for the text-line detector.

Usage:
    uv run python -m training.make_pages training/configs/pages_v1.yaml
    ... --workers 8 / --overwrite      (same as make_dataset.py)
    ... --splits val test              (only these splits, e.g. for a quick check)

Each sample is a whole page (see training/page.py) stored in tar shards:

    data/datasets/<name>/train/shard-00000.tar
        train-00000-00007.jpg    the page, grayscale JPEG (quality 92)
        train-00000-00007.json   {"kind", "width", "height",
                                  "lines": [{"quad": [[x, y] x4], "text", "block", "order"}]}

Text for each split comes from that split's corpus text only, so no sentence is in
two splits. Finished shards are kept, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml

from training.corpora import CORPORA_DIR
from training.dataset import SPLITS
from training.fonts import FontPicker
from training.make_dataset import DATASETS_DIR, TextPicker, shard_seed, weighted_fonts
from training.page import PageConfig, make_page


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    tar.addfile(info, io.BytesIO(data))


def make_shard(cfg: dict, split: str, index: int, count: int, out: Path) -> int:
    seed = shard_seed(cfg["seed"], split, index)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    texts = TextPicker(cfg["text_sources"], split, CORPORA_DIR)
    picker = FontPicker(weighted_fonts(cfg.get("style_weights", {})), rng)
    page_cfg = PageConfig(**cfg.get("page", {}))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tar.tmp")
    written = 0
    with tarfile.open(tmp, "w") as tar:
        while written < count:
            img, lines, kind = make_page(texts, picker, rng, np_rng, page_cfg)
            if len(lines) < 3:
                continue  # an almost empty page teaches little
            key = f"{split}-{index:05d}-{written:05d}"
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            assert ok
            meta = {
                "kind": kind,
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
                "lines": [
                    {
                        "quad": np.round(ln.quad, 1).tolist(),
                        "text": ln.text,
                        "block": ln.block,
                        "order": ln.order,
                    }
                    for ln in lines
                ],
            }
            _add(tar, f"{key}.jpg", jpg.tobytes())
            _add(tar, f"{key}.json", json.dumps(meta, ensure_ascii=False).encode("utf-8"))
            written += 1
    tmp.replace(out)
    return written


def read_pages(path: Path):
    """Yield (key, grayscale image, meta) from one page shard."""
    parts: dict[str, dict[str, bytes]] = {}
    with tarfile.open(path) as tar:
        for m in tar:
            key, _, ext = m.name.rpartition(".")
            f = tar.extractfile(m)
            if f is None:
                continue
            parts.setdefault(key, {})[ext] = f.read()
            if len(parts[key]) == 2:
                p = parts.pop(key)
                img = cv2.imdecode(np.frombuffer(p["jpg"], np.uint8), cv2.IMREAD_GRAYSCALE)
                yield key, img, json.loads(p["json"].decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic page dataset.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=list(SPLITS), help="only make these"
    )
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    PageConfig(**cfg.get("page", {}))  # fail early on typos
    out_dir = DATASETS_DIR / cfg["name"]
    chash = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["config_hash"] != chash:
            print(f"{out_dir} was made with other settings; use --overwrite or a new name.")
            return 1
    if args.overwrite:
        for p in out_dir.glob("*/shard-*.tar*"):
            p.unlink()

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"name": cfg["name"], "status": "incomplete", "config": cfg, "config_hash": chash}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    jobs = []
    for split in args.splits:
        total, size = cfg["samples"][split], cfg["shard_size"]
        for i in range(math.ceil(total / size)):
            path = out_dir / split / f"shard-{i:05d}.tar"
            if not path.exists():
                jobs.append((split, i, min(size, total - i * size), path))
    print(f"{cfg['name']}: {len(jobs)} shards to make with {args.workers} workers")
    start, done = time.perf_counter(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(make_shard, cfg, *job): job for job in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            done += fut.result()
            rate = done / (time.perf_counter() - start)
            split, i = futures[fut][:2]
            print(f"  [{n}/{len(jobs)}] {split} shard {i} ({rate:.1f} pages/s)", flush=True)

    manifest["samples"] = cfg["samples"]
    if set(args.splits) == set(SPLITS):
        manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
