"""Score a line detector: precision and recall at IoU >= 0.5.

- **IoU** (intersection over union): how much a predicted box and a true box
  overlap, from 0 (not at all) to 1 (identical). IoU >= 0.5 counts as "found".
- **Recall**: share of true lines that were found (M3 target: >= 95%).
- **Precision**: share of predicted boxes that are real lines.
- **F1**: one number combining both (their harmonic mean).

Each true line can be matched by at most one predicted box (best overlaps first).

Usage (ONNX model, CPU, as shipped):
    uv run python -m training.detect_eval models/detector-v1.onnx
    ... --dataset data/datasets/pages_v1 --split test --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arabic_ocr.detect.boxes import quad_iou, quad_to_rect

ROOT = Path(__file__).resolve().parent.parent
#: True lines shorter than this (pixels, at the detector's working size) are too
#: small to read; they are neither required nor counted as false alarms.
MIN_HEIGHT = 6


@dataclass
class Counts:
    tp: int = 0  # predicted boxes matching a true line
    n_pred: int = 0
    n_true: int = 0

    def add(self, other: Counts) -> None:
        self.tp += other.tp
        self.n_pred += other.n_pred
        self.n_true += other.n_true

    @property
    def precision(self) -> float:
        return self.tp / self.n_pred if self.n_pred else 1.0

    @property
    def recall(self) -> float:
        return self.tp / self.n_true if self.n_true else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def match(preds: list[np.ndarray], truth: list[np.ndarray], tiny: list[np.ndarray]) -> Counts:
    """Greedy one-to-one matching at IoU >= 0.5. Predictions that only match a
    `tiny` (too small to require) true line are ignored."""
    pairs = sorted(
        ((quad_iou(p, t), i, j) for i, p in enumerate(preds) for j, t in enumerate(truth)),
        reverse=True,
    )
    used_p, used_t = set(), set()
    for iou, i, j in pairs:
        if iou < 0.5:
            break
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
    ignored = {
        i
        for i, p in enumerate(preds)
        if i not in used_p and any(quad_iou(p, t) >= 0.5 for t in tiny)
    }
    return Counts(len(used_p), len(preds) - len(ignored), len(truth))


def split_truth(lines: list[dict], scale: float) -> tuple[list, list]:
    """True quads -> (required lines, tiny lines), by height at the working size."""
    req, tiny = [], []
    for ln in lines:
        q = np.float32(ln["quad"])
        (req if quad_to_rect(q).h * scale >= MIN_HEIGHT else tiny).append(q)
    return req, tiny


def main(argv: list[str] | None = None) -> int:
    from arabic_ocr.detect.detector import LineDetector, prepare_page
    from training.make_pages import read_pages

    parser = argparse.ArgumentParser(description="Evaluate a line detector (ONNX).")
    parser.add_argument("model", type=Path)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/datasets/pages_v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args(argv)

    det = LineDetector(args.model)
    total, by_kind = Counts(), defaultdict(Counts)
    n = 0
    for shard in sorted((args.dataset / args.split).glob("shard-*.tar")):
        for _key, img, meta in read_pages(shard):
            page = prepare_page(img, det.max_side)
            truth, tiny = split_truth(meta["lines"], page.scale)
            preds = [d.quad for d in det.detect(img)]
            c = match(preds, truth, tiny)
            total.add(c)
            by_kind[meta["kind"]].add(c)
            n += 1
            if n >= args.limit:
                break
        if n >= args.limit:
            break
    print(f"{n} pages")
    for kind, c in sorted(by_kind.items()):
        print(f"  {kind:8s} precision {c.precision:6.1%}  recall {c.recall:6.1%}  F1 {c.f1:6.1%}")
    print(
        f"  overall  precision {total.precision:6.1%}  recall {total.recall:6.1%}  "
        f"F1 {total.f1:6.1%}"
    )
    print(f"  (M3 target: recall >= 95%)  {'PASS' if total.recall >= 0.95 else 'FAIL'}")
    out = args.model.with_suffix(f".eval-{args.dataset.name}-{args.split}.json")
    out.write_text(
        json.dumps(
            {
                "pages": n,
                "precision": total.precision,
                "recall": total.recall,
                "f1": total.f1,
                "by_kind": {k: vars(c) for k, c in by_kind.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
