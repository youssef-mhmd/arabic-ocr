"""Measure how well a trained recognizer reads: CER/WER overall, per font style and family.

Usage:
    uv run --group train python -m training.evaluate_recognizer runs/recognizer_v1/best.pt
    uv run python -m training.evaluate_recognizer models/recognizer-v1.onnx   # no torch needed
    ... --split val          # which split to use (default: test)
    ... --limit 2000         # only the first N lines (quick look)

A `.pt` checkpoint runs with PyTorch (GPU if there is one); an `.onnx` model runs
with ONNX Runtime on the CPU, exactly like the shipped app. Both should score
about the same; that is a good final check of the export.

Scores (see training/metrics.py): CER and WER, each with and without diacritics,
plus the share of lines read exactly right. They are compared against the M2 exit
targets from the PRD: CER <= 2% ignoring diacritics, <= 6% counting them.

Writes next to the model:
    <model>.eval-<split>.json     every number printed here
    <model>.errors-<split>.tsv    the worst-read lines (want / got / font), to look at
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from training.dataset import Sample, read_split
from training.metrics import Scores, edit_distance, score

ROOT = Path(__file__).resolve().parent.parent

# M2 exit criteria (docs/milestones.md, PRD §9), on the synthetic test set.
TARGET_CER_NODIAC = 0.02
TARGET_CER = 0.06


# ---------------------------------------------------------------------------
# Readers: list of samples -> list of predicted texts
# ---------------------------------------------------------------------------


def read_with_onnx(model_path: Path, samples: list[Sample]) -> list[str]:
    from arabic_ocr.recognize import LineRecognizer

    rec = LineRecognizer(model_path)
    return [d.text for d in rec.recognize_batch([s.image for s in samples], max_batch=32)]


def read_with_torch(ckpt_path: Path, samples: list[Sample]) -> list[str]:
    import torch

    from arabic_ocr.recognize.ctc import greedy_decode_batch
    from training.data import collate, make_target
    from training.model import Recognizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = Recognizer(num_classes=ckpt["num_classes"], **ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # Same batching as validation during training: similar widths together.
    order = sorted(range(len(samples)), key=lambda i: samples[i].image.shape[1])
    texts: list[str] = [""] * len(samples)
    with torch.no_grad():
        for k in range(0, len(order), 64):
            idx = order[k : k + 64]
            batch = collate([(samples[i].image, make_target(samples[i].label), "") for i in idx])
            log_probs = model(batch.images.to(device), batch.input_lengths)
            decoded = greedy_decode_batch(log_probs.float().cpu().numpy(), batch.input_lengths)
            for i, d in zip(idx, decoded, strict=True):
                texts[i] = d.text
    return texts


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def group_scores(samples: list[Sample], hyps: list[str], key: str) -> dict[str, Scores]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        groups[str(s.meta.get(key, "?"))].append(i)
    return {
        name: score([samples[i].label for i in idx], [hyps[i] for i in idx])
        for name, idx in groups.items()
    }


def print_table(title: str, rows: dict[str, Scores], limit: int | None = None) -> None:
    print(f"\n{title}")
    print(f"  {'':<28} {'lines':>6} {'CER':>7} {'CER-nd':>7} {'WER':>7} {'exact':>7}")
    ordered = sorted(rows.items(), key=lambda kv: -kv[1].cer)
    for name, s in ordered[:limit]:
        print(
            f"  {name[:28]:<28} {s.count:>6} {s.cer:>7.2%} {s.cer_no_diacritics:>7.2%} "
            f"{s.wer:>7.2%} {s.exact:>7.1%}"
        )
    if limit and len(ordered) > limit:
        print(f"  ... {len(ordered) - limit} more in the JSON file")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained line recognizer.")
    parser.add_argument("model", type=Path, help=".pt checkpoint or .onnx model")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/datasets/v1")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    samples: list[Sample] = []
    for s in read_split(args.dataset, args.split):
        samples.append(s)
        if args.limit and len(samples) >= args.limit:
            break
    print(f"Reading {len(samples):,} {args.split} lines with {args.model.name} ...")

    t = time.perf_counter()
    if args.model.suffix == ".onnx":
        hyps = read_with_onnx(args.model, samples)
    else:
        hyps = read_with_torch(args.model, samples)
    elapsed = time.perf_counter() - t
    print(f"  {elapsed:.0f}s ({len(samples) / elapsed:.0f} lines/s)")

    refs = [s.label for s in samples]
    overall = score(refs, hyps)
    by_style = group_scores(samples, hyps, "style")
    by_family = group_scores(samples, hyps, "family")

    print_table("Per font style (worst first)", by_style)
    print_table("Per font family (worst 15)", by_family, limit=15)

    ok_nd = overall.cer_no_diacritics <= TARGET_CER_NODIAC
    ok = overall.cer <= TARGET_CER
    print(f"\nOverall on {overall.count:,} lines:")
    print(f"  CER  {overall.cer:6.2%}   (target <= {TARGET_CER:.0%})  {'PASS' if ok else 'FAIL'}")
    print(
        f"  CER without diacritics {overall.cer_no_diacritics:6.2%}   "
        f"(target <= {TARGET_CER_NODIAC:.0%})  {'PASS' if ok_nd else 'FAIL'}"
    )
    print(f"  WER  {overall.wer:6.2%}   WER without diacritics {overall.wer_no_diacritics:6.2%}")
    print(f"  lines read exactly right: {overall.exact:.1%}")

    stem = args.model.with_suffix("")
    report = {
        "model": str(args.model),
        "split": args.split,
        "lines": overall.count,
        "lines_per_second": round(len(samples) / elapsed, 1),
        "overall": overall.as_dict(),
        "targets": {"cer_nodiac": TARGET_CER_NODIAC, "cer": TARGET_CER},
        "pass": ok and ok_nd,
        "by_style": {k: {**v.as_dict(), "lines": v.count} for k, v in by_style.items()},
        "by_family": {k: {**v.as_dict(), "lines": v.count} for k, v in by_family.items()},
    }
    json_path = Path(f"{stem}.eval-{args.split}.json")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    worst = sorted(
        range(len(samples)),
        key=lambda i: -edit_distance(refs[i], hyps[i]) / max(len(refs[i]), 1),
    )[:300]
    tsv_path = Path(f"{stem}.errors-{args.split}.tsv")
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        f.write("key\tfont\terrors\twant\tgot\n")
        for i in worst:
            s = samples[i]
            errs = edit_distance(refs[i], hyps[i])
            if errs == 0:
                break
            f.write(f"{s.key}\t{s.meta.get('font', '?')}\t{errs}\t{refs[i]}\t{hyps[i]}\n")
    print(f"\nWrote {json_path} and {tsv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
