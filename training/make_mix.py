"""Assemble a fine-tuning dataset from real lines + synthetic lines (M4).

    uv run python -m training.make_mix --name mix_v6 --synthetic v5 --synthetic-shards 40 \\
        --real real_v1 --real-repeat 4

The new dataset's shards are *hard links* to existing shards (no copying, no extra
disk space):
- train: the first N synthetic training shards (so the model keeps what it knows)
  plus every real training shard, repeated R times (real lines are far fewer, and
  they are what we want it to learn);
- val: the real validation lines only, so "best model" means best on real scans.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from training.dataset import charset_fingerprint, read_shard, shard_paths

DATASETS = Path(__file__).resolve().parent.parent / "data" / "datasets"


def count(paths: list[Path]) -> int:
    return sum(1 for p in paths for _ in read_shard(p))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mix real and synthetic line shards.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--synthetic", required=True, help="synthetic dataset name, e.g. v5")
    parser.add_argument("--synthetic-shards", type=int, default=40)
    parser.add_argument("--real", required=True, help="real dataset folder, e.g. real_v1")
    parser.add_argument("--real-repeat", type=int, default=4)
    args = parser.parse_args(argv)

    out = DATASETS / args.name
    if out.exists():
        shutil.rmtree(out)
    (out / "train").mkdir(parents=True)
    (out / "val").mkdir()
    synth = shard_paths(DATASETS / args.synthetic, "train")[: args.synthetic_shards]
    real_root = DATASETS / args.real
    real_train = sorted(real_root.glob("*/train/shard-*.tar"))
    real_val = sorted(real_root.glob("*/val/shard-*.tar"))

    k = 0
    for src in synth + real_train * args.real_repeat:
        os.link(src, out / "train" / f"shard-{k:05d}.tar")
        k += 1
    for i, src in enumerate(real_val):
        os.link(src, out / "val" / f"shard-{i:05d}.tar")

    n_synth, n_real = count(synth), count(real_train)
    manifest = {
        "name": args.name,
        "charset": charset_fingerprint(),
        "samples": {
            "train": n_synth + n_real * args.real_repeat,
            "val": count(real_val),
            "test": 0,
        },
        "parts": {
            "synthetic": {"dataset": args.synthetic, "shards": len(synth), "lines": n_synth},
            "real": {"dataset": args.real, "lines": n_real, "repeat": args.real_repeat},
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
