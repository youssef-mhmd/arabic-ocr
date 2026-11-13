"""Train the line recognizer.

Usage:
    uv run --group train python -m training.train_recognizer training/configs/recognizer_v1.yaml
    ... --max-samples 50000      # stop early (quick tests / speed measurement)
    ... --fresh                  # ignore an existing checkpoint and start over

What one training step does (see docs/learning/01-ml-basics.md):
1. take a batch of line images and their targets (text in reading order);
2. run the model: per-slice scores for every symbol ("forward pass");
3. CTC loss: how unlikely the correct text is under those scores;
4. backpropagate and let AdamW nudge every weight to make it more likely.

Every `val_every_samples` images it measures CER/WER on the validation set, writes a
checkpoint (`last.pt`), and keeps the best one so far (`best.pt`, lowest CER).

Resuming: if runs/<name>/last.pt exists, training continues from it: same weights,
optimizer state, learning-rate position and epoch. (The interrupted epoch restarts
its data from the beginning with the same shuffle.)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from arabic_ocr import charset
from arabic_ocr.recognize.ctc import greedy_decode_batch
from training.data import ShardBatches
from training.dataset import charset_fingerprint, shard_paths
from training.metrics import score
from training.model import Recognizer, count_parameters

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def learning_rate(cfg: dict, samples_seen: int, total_samples: int) -> float:
    """Warm up linearly, then follow a cosine curve down to min_lr_fraction * lr.

    Based on images seen (not steps), because batch sizes vary with line width.
    """
    o = cfg["optimizer"]
    peak, warm = o["lr"], o["warmup_samples"]
    if samples_seen < warm:
        return peak * max(samples_seen, 1) / warm
    progress = min(1.0, (samples_seen - warm) / max(1, total_samples - warm))
    floor = peak * o["min_lr_fraction"]
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def save_atomic(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)  # never leaves a half-written checkpoint


def append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp) -> tuple:
    model.eval()
    refs, hyps = [], []
    for batch in loader:
        images = batch.images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            log_probs = model(images, batch.input_lengths)
        decoded = greedy_decode_batch(log_probs.float().cpu().numpy(), batch.input_lengths.numpy())
        refs += batch.labels
        hyps += [d.text for d in decoded]
    model.train()
    return score(refs, hyps), list(zip(refs, hyps, strict=True))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the line recognizer.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--max-samples", type=int, help="stop after this many images")
    parser.add_argument("--fresh", action="store_true", help="ignore existing checkpoint")
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
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["charset"] != charset_fingerprint():
        print("The dataset was made with a different charset; regenerate it.")
        return 1
    train_size = manifest["samples"]["train"]
    total_samples = cfg["epochs"] * train_size

    model = Recognizer(**cfg["model"]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"]
    )
    ctc = nn.CTCLoss(blank=charset.BLANK_ID, zero_infinity=True)
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, None: None}[cfg.get("amp")]

    # --- resume ---------------------------------------------------------------
    state = {"epoch": 0, "samples_seen": 0, "step": 0, "best_cer": float("inf")}
    last = run_dir / "last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        state = ckpt["state"]
        print(f"Resumed from {last}: epoch {state['epoch']}, {state['samples_seen']:,} images seen")
    elif cfg.get("init_from"):
        # Fine-tuning: start from another run's weights instead of random ones, with
        # a fresh optimizer and learning-rate schedule. The model keeps what it
        # learned and adapts to the new data (much faster than starting over).
        src = ROOT / cfg["init_from"]
        init = torch.load(src, map_location=device, weights_only=False)
        if init["charset"] != charset_fingerprint():
            print(f"{src} was trained with a different charset.")
            return 1
        model.load_state_dict(init["model"])
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
                "charset": charset_fingerprint(),
                "num_classes": charset.NUM_CLASSES,
            },
            path,
        )

    val_loader = DataLoader(
        ShardBatches(
            shard_paths(dataset_dir, "val"),
            cfg["width_budget"],
            cfg["max_batch"],
            shuffle=False,
            seed=cfg["seed"],
        ),
        batch_size=None,
        num_workers=2,
    )

    def validate() -> None:
        t = time.perf_counter()
        s, pairs = evaluate(model, val_loader, device, amp)
        row = {
            "samples_seen": state["samples_seen"],
            "epoch": state["epoch"],
            **{k: round(v, 5) for k, v in s.as_dict().items()},
        }
        append_csv(run_dir / "val.csv", row)
        print(
            f"  VAL  CER {s.cer:.2%}  (no diacritics {s.cer_no_diacritics:.2%})  "
            f"WER {s.wer:.2%}  exact lines {s.exact:.1%}  [{time.perf_counter() - t:.0f}s]"
        )
        for ref, hyp in pairs[:: max(1, len(pairs) // 3)][:3]:
            print(f"    want: {ref}\n    got:  {hyp}")
        checkpoint(last)
        if s.cer < state["best_cer"]:
            state["best_cer"] = s.cer
            shutil.copyfile(last, run_dir / "best.pt")
            print(f"    new best CER {s.cer:.2%} -> best.pt")

    # --- training loop --------------------------------------------------------
    model.train()
    limit = args.max_samples
    started = state["samples_seen"]
    next_val = (state["samples_seen"] // cfg["val_every_samples"] + 1) * cfg["val_every_samples"]
    t0, window_samples, window_t = time.perf_counter(), 0, time.perf_counter()
    try:
        while state["epoch"] < cfg["epochs"]:
            loader = DataLoader(
                ShardBatches(
                    shard_paths(dataset_dir, "train"),
                    cfg["width_budget"],
                    cfg["max_batch"],
                    shuffle=True,
                    seed=cfg["seed"],
                    epoch=state["epoch"],
                ),
                batch_size=None,
                num_workers=cfg["num_workers"],
                pin_memory=True,
                prefetch_factor=4,
                persistent_workers=False,
            )
            for batch in loader:
                lr = learning_rate(cfg, state["samples_seen"], total_samples)
                for g in opt.param_groups:
                    g["lr"] = lr
                images = batch.images.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                    log_probs = model(images, batch.input_lengths)
                loss = ctc(
                    log_probs.float(),
                    batch.targets.to(device, non_blocking=True),
                    batch.input_lengths.to(device, non_blocking=True),
                    batch.target_lengths.to(device, non_blocking=True),
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["optimizer"]["grad_clip"]
                )
                # A single non-finite loss or gradient (NaN/inf) would poison every
                # weight for good (seen in the first test run). Skip that update instead,
                # count it, and keep the first bad batch on disk for investigation.
                if torch.isfinite(loss) and torch.isfinite(grad_norm):
                    opt.step()
                else:
                    state["skipped_steps"] = state.get("skipped_steps", 0) + 1
                    print(
                        f"  skipped step {state['step'] + 1}: loss {loss.item()}, "
                        f"grad norm {grad_norm.item()} (total skipped: {state['skipped_steps']})"
                    )
                    bad = run_dir / "first_bad_batch.pt"
                    if not bad.exists():
                        torch.save(
                            {
                                "images": batch.images,
                                "targets": batch.targets,
                                "input_lengths": batch.input_lengths,
                                "target_lengths": batch.target_lengths,
                                "labels": batch.labels,
                                "step": state["step"] + 1,
                            },
                            bad,
                        )

                n = images.shape[0]
                state["samples_seen"] += n
                state["step"] += 1
                window_samples += n
                if state["step"] % cfg["log_every_steps"] == 0:
                    now = time.perf_counter()
                    speed = window_samples / (now - window_t)
                    window_samples, window_t = 0, now
                    remaining = (total_samples - state["samples_seen"]) / speed / 3600
                    mem = torch.cuda.max_memory_allocated() / 2**30
                    append_csv(
                        run_dir / "log.csv",
                        {
                            "step": state["step"],
                            "samples_seen": state["samples_seen"],
                            "epoch": state["epoch"],
                            "loss": round(loss.item(), 4),
                            "lr": f"{lr:.6f}",
                            "images_per_s": round(speed),
                            "gpu_mem_gb": round(mem, 2),
                        },
                    )
                    print(
                        f"epoch {state['epoch'] + 1}/{cfg['epochs']}  "
                        f"{state['samples_seen']:>10,} imgs  loss {loss.item():.3f}  "
                        f"lr {lr:.2e}  {speed:.0f} img/s  GPU {mem:.1f} GB  "
                        f"~{remaining:.1f} h left",
                        flush=True,
                    )
                if state["samples_seen"] >= next_val:
                    validate()
                    next_val += cfg["val_every_samples"]
                if limit and state["samples_seen"] - started >= limit:
                    raise KeyboardInterrupt  # handled below: save and stop
            state["epoch"] += 1
        validate()
        print(f"Finished. Best CER {state['best_cer']:.2%} -> {run_dir / 'best.pt'}")
    except KeyboardInterrupt:
        checkpoint(last)
        print(
            f"Stopped after {state['samples_seen']:,} images "
            f"({time.perf_counter() - t0:.0f}s this session). Saved {last}; "
            "run the same command to resume."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
