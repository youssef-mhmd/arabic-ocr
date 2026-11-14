"""Train the text-line detector.

Usage:
    uv run --group train python -m training.train_detector training/configs/detector_v1.yaml
    ... --max-samples 5000   # stop early (quick test / speed measurement)
    ... --fresh              # ignore an existing checkpoint

One training step:
1. a batch of 640x640 page crops and their target "core" maps (detector_data.py);
2. the model paints a core map for each crop;
3. the loss compares the two (below);
4. AdamW adjusts the weights.

**Loss** = balanced BCE + Dice:
- *BCE* (binary cross-entropy) scores every pixel: "core or not?". Most pixels are
  not core, so we only count the 3 hardest non-core pixels for every core pixel
  ("hard negative mining"); otherwise "nothing is text" would already look good.
- *Dice* measures the overlap of the predicted and true core regions as a whole,
  which keeps thin cores from fading.

Every `val_every_samples` crops it runs the detector on validation pages, exactly
as the shipped app does (core map -> boxes), and reports precision/recall/F1 at
IoU >= 0.5 (detect_eval.py). Checkpoints and resume work as in train_recognizer.py.
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

from arabic_ocr.detect.detector import boxes_from_prob, prepare_page
from training.detect_eval import Counts, match, split_truth
from training.detector_data import PageCrops
from training.detector_model import Detector, count_parameters
from training.make_pages import read_pages
from training.train_recognizer import append_csv, save_atomic, set_seed

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"


def detection_loss(logits: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits, gt, mask = logits.float(), gt.float(), mask.float()
    bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    pos = (gt > 0.5) & (mask > 0)
    neg = (gt <= 0.5) & (mask > 0)
    n_pos = int(pos.sum())
    n_neg = min(int(neg.sum()), max(3 * n_pos, 1000))
    hard_neg = torch.topk(bce[neg], n_neg).values if n_neg else bce.new_zeros(0)
    balanced = (bce[pos].sum() + hard_neg.sum()) / (n_pos + n_neg + 1e-6)
    prob = torch.sigmoid(logits) * mask
    inter = (prob * gt).sum()
    dice = 1 - 2 * inter / (prob.sum() + (gt * mask).sum() + 1e-6)
    return balanced + dice


def learning_rate(cfg: dict, seen: int, total: int) -> float:
    o = cfg["optimizer"]
    if seen < o["warmup_samples"]:
        return o["lr"] * max(seen, 1) / o["warmup_samples"]
    progress = min(1.0, (seen - o["warmup_samples"]) / max(1, total - o["warmup_samples"]))
    floor = o["lr"] * o["min_lr_fraction"]
    return floor + (o["lr"] - floor) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def validate_pages(model, pages, device, max_side: int, amp) -> Counts:
    model.eval()
    total = Counts()
    for img, lines in pages:
        page = prepare_page(img, max_side)
        x = torch.from_numpy(page.image)[None, None].to(device)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            prob = torch.sigmoid(model(x).float())[0, 0].cpu().numpy()
        preds = [d.quad for d in boxes_from_prob(prob, page)]
        truth, tiny = split_truth(lines, page.scale)
        total.add(match(preds, truth, tiny))
    model.train()
    return total


def main(argv: list[str] | None = None) -> int:
    # Windows Python uses a legacy codepage when output goes to a file.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Train the text-line detector.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = args.runs_dir / cfg["name"]
    if args.fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    if not torch.cuda.is_available():
        print("No CUDA GPU found. Install the train group: uv sync --group train")
        return 1
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    set_seed(cfg["seed"])

    dataset_dir = ROOT / cfg["dataset"]
    train_shards = sorted((dataset_dir / "train").glob("shard-*.tar"))
    pages_per_epoch = sum(1 for _ in train_shards) * cfg["shard_size"]
    per_epoch = pages_per_epoch * cfg["crops_per_page"]
    total = cfg["epochs"] * per_epoch

    model = Detector(**cfg.get("model", {})).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"]
    )
    amp = {"bf16": torch.bfloat16, None: None}[cfg.get("amp")]

    state = {"epoch": 0, "seen": 0, "step": 0, "best_f1": -1.0}
    last = run_dir / "last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        state = ckpt["state"]
        print(f"Resumed from {last}: epoch {state['epoch']}, {state['seen']:,} crops seen")
    elif cfg.get("init_from"):
        # Fine-tuning: start from another run's weights (not its optimizer or progress).
        src = ROOT / cfg["init_from"]
        model.load_state_dict(torch.load(src, map_location=device, weights_only=False)["model"])
        print(f"Starting {cfg['name']} from the weights of {src}")
    else:
        print(f"Starting {cfg['name']}: {count_parameters(model):,} weights")

    def checkpoint(path: Path) -> None:
        save_atomic(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "state": state,
                "config": cfg,
            },
            path,
        )

    val_pages = []
    for shard in sorted((dataset_dir / "val").glob("shard-*.tar")):
        for _k, img, meta in read_pages(shard):
            val_pages.append((img, meta["lines"]))
    val_pages = val_pages[: cfg["val_pages"]]

    def validate() -> None:
        t = time.perf_counter()
        c = validate_pages(model, val_pages, device, cfg["max_side"], amp)
        append_csv(
            run_dir / "val.csv",
            {
                "seen": state["seen"],
                "epoch": state["epoch"],
                "precision": round(c.precision, 4),
                "recall": round(c.recall, 4),
                "f1": round(c.f1, 4),
            },
        )
        print(
            f"  VAL  precision {c.precision:.1%}  recall {c.recall:.1%}  F1 {c.f1:.1%}  "
            f"({len(val_pages)} pages, {time.perf_counter() - t:.0f}s)",
            flush=True,
        )
        checkpoint(last)
        if c.f1 > state["best_f1"]:
            state["best_f1"] = c.f1
            shutil.copyfile(last, run_dir / "best.pt")
            print(f"    new best F1 {c.f1:.1%} -> best.pt", flush=True)

    model.train()
    started, limit = state["seen"], args.max_samples
    next_val = (state["seen"] // cfg["val_every_samples"] + 1) * cfg["val_every_samples"]
    window_n, window_t = 0, time.perf_counter()
    try:
        while state["epoch"] < cfg["epochs"]:
            data = PageCrops(
                train_shards,
                crop=cfg["crop"],
                scale_range=tuple(cfg["scale_range"]),
                crops_per_page=cfg["crops_per_page"],
                seed=cfg["seed"],
                epoch=state["epoch"],
            )
            loader = DataLoader(
                data,
                batch_size=cfg["batch_size"],
                num_workers=cfg["num_workers"],
                pin_memory=True,
                prefetch_factor=4,
                drop_last=True,
            )
            for images, gt, mask in loader:
                lr = learning_rate(cfg, state["seen"], total)
                for g in opt.param_groups:
                    g["lr"] = lr
                images = images.to(device, non_blocking=True)
                gt, mask = gt.to(device, non_blocking=True), mask.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                    logits = model(images)
                loss = detection_loss(logits, gt, mask)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["optimizer"]["grad_clip"]
                )
                if torch.isfinite(loss) and torch.isfinite(grad_norm):
                    opt.step()
                else:
                    state["skipped_steps"] = state.get("skipped_steps", 0) + 1
                n = images.shape[0]
                state["seen"] += n
                state["step"] += 1
                window_n += n
                if state["step"] % cfg["log_every_steps"] == 0:
                    now = time.perf_counter()
                    speed = window_n / (now - window_t)
                    window_n, window_t = 0, now
                    left = (total - state["seen"]) / speed / 3600
                    append_csv(
                        run_dir / "log.csv",
                        {
                            "step": state["step"],
                            "seen": state["seen"],
                            "loss": round(loss.item(), 4),
                            "lr": f"{lr:.6f}",
                            "crops_per_s": round(speed, 1),
                        },
                    )
                    print(
                        f"epoch {state['epoch'] + 1}/{cfg['epochs']}  {state['seen']:>9,} crops  "
                        f"loss {loss.item():.3f}  lr {lr:.2e}  {speed:.0f} crops/s  "
                        f"~{left:.1f} h left",
                        flush=True,
                    )
                if state["seen"] >= next_val:
                    validate()
                    next_val += cfg["val_every_samples"]
                if limit and state["seen"] - started >= limit:
                    raise KeyboardInterrupt
            state["epoch"] += 1
        validate()
        print(f"Finished. Best F1 {state['best_f1']:.1%} -> {run_dir / 'best.pt'}")
    except KeyboardInterrupt:
        checkpoint(last)
        print(f"Stopped after {state['seen']:,} crops. Saved {last}; run again to resume.")
    return 0


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    sys.exit(main())
