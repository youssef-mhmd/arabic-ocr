"""Train the recognizer on a free cloud GPU (Kaggle or Google Colab).

Use this when your own GPU is busy or missing. It wraps train_recognizer.py and
adapts the settings to whatever GPU the notebook gives you.

Notebook cells (Kaggle: turn on "GPU" and "Internet" in the settings panel;
Colab: Runtime -> Change runtime type -> GPU):

    !curl -LsSf https://astral.sh/uv/install.sh | sh
    !git clone <your arabic-ocr repo URL> && cd arabic-ocr && ~/.local/bin/uv sync --group train
    %cd arabic-ocr

    # Option A (fast): you uploaded data/datasets/v1 as a private Kaggle dataset
    # or copied it to Google Drive. Point --dataset at that folder:
    !~/.local/bin/uv run --group train python -m training.cloud_train \
        training/configs/recognizer_v1.yaml --dataset /kaggle/input/arabic-ocr-v1 \
        --runs-dir /kaggle/working/runs

    # Option B (slow, several hours of CPU): build the dataset in the notebook first.
    !~/.local/bin/uv run --group train python -m training.cloud_train \
        training/configs/recognizer_v1.yaml --generate training/configs/data_v1.yaml

Checkpoints go to --runs-dir. Notebooks are stopped after a time limit (Kaggle: 12 h,
Colab free: a few hours); run the same cell again in a new session with the same
--runs-dir (copy last.pt back first if it was saved as notebook output) to resume.

What it adapts:
- **Mixed precision:** bf16 needs an Ampere-or-newer GPU (A100, L4, RTX 30xx+). On
  older ones (Kaggle's T4/P100, Colab's free T4) it falls back to full float32:
  slower, but numerically safe.
- **Data workers:** never more than the notebook has CPU cores (usually 2-4).

Note: training PyTorch comes from the CUDA 13.2 build (see pyproject.toml), which
needs a recent NVIDIA driver. If `torch.cuda.is_available()` is False in the
notebook, the driver is too old for that build. Not yet tested on Kaggle/Colab.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from training import train_recognizer

ROOT = Path(__file__).resolve().parent.parent


def run_step(*args: str) -> None:
    print(f"\n$ python -m {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-m", *args], cwd=ROOT, check=True)


def generate_dataset(data_config: Path) -> Path:
    run_step("training.download_fonts")
    run_step("training.corpora", "download")
    run_step("training.corpora", "prepare")
    run_step("training.make_dataset", str(data_config))
    name = yaml.safe_load(data_config.read_text(encoding="utf-8"))["name"]
    dataset = ROOT / "data" / "datasets" / name
    run_step("training.check_dataset", str(dataset))
    return dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the recognizer on Kaggle/Colab.")
    parser.add_argument("config", type=Path, help="recognizer training config (YAML)")
    parser.add_argument("--dataset", type=Path, help="folder of an existing dataset")
    parser.add_argument("--generate", type=Path, help="data config: build the dataset first")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--max-samples", type=int, help="stop early (quick test)")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("No usable CUDA GPU. Turn on the GPU in the notebook settings (see the top of")
        print("this file); if it is on, the NVIDIA driver may be too old for this PyTorch.")
        return 1
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory / 2**30:.0f} GB")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.generate:
        cfg["dataset"] = str(generate_dataset(args.generate))
    elif args.dataset:
        cfg["dataset"] = str(args.dataset.resolve())
    if not (Path(ROOT / cfg["dataset"]) / "manifest.json").exists():
        print(f"No dataset at {cfg['dataset']}: pass --dataset or --generate.")
        return 1

    if cfg.get("amp") == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        print("This GPU has no fast bf16: training in float32 instead (slower, but safe).")
        cfg["amp"] = None
    cores = os.cpu_count() or 2
    cfg["num_workers"] = min(cfg.get("num_workers", 4), cores)
    print(f"Data workers: {cfg['num_workers']} ({cores} CPU cores)")

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    cloud_cfg = args.runs_dir / f"{cfg['name']}.cloud.yaml"
    cloud_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    train_args = [str(cloud_cfg), "--runs-dir", str(args.runs_dir)]
    if args.max_samples:
        train_args += ["--max-samples", str(args.max_samples)]
    return train_recognizer.main(train_args)


if __name__ == "__main__":
    sys.exit(main())
