"""End-to-end score on synthetic pages: detector + recognizer + reading order.

    uv run python -m training.eval_pages                    # pages_v1 test split
    uv run python -m training.eval_pages --limit 100 --kind screen

For every page, the full pipeline (arabic_ocr.pipeline.OCR, ONNX on the CPU) reads
the page, and its lines are matched to the true lines by box overlap (IoU >= 0.5).

- **Page CER**: character errors over all true lines. A missed line counts as all
  its characters wrong; a predicted line that matches no true line counts its
  characters as extra errors. So both detection and recognition mistakes count.
  Reported with and without diacritics, like the recognizer's scores.
- **Order**: share of pages whose matched lines come out in exactly the true
  reading order.

Runtime code only (ONNX Runtime): no torch needed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from arabic_ocr import charset
from arabic_ocr.detect.boxes import quad_iou, quad_to_rect
from arabic_ocr.pipeline import OCR
from training.detect_eval import MIN_HEIGHT
from training.make_pages import read_pages
from training.metrics import edit_distance


def _clean(text: str, diacritics: bool) -> str:
    text = charset.strip_tatweel(charset.normalize(text))
    return text if diacritics else charset.strip_diacritics(text)


class Tally:
    def __init__(self) -> None:
        self.err = defaultdict(int)  # by "diac" / "nodiac"
        self.total = defaultdict(int)
        self.pages = 0
        self.ordered = 0

    def cer(self, key: str) -> float:
        return self.err[key] / max(self.total[key], 1)


def score_page(ocr: OCR, img: np.ndarray, meta: dict, tally: Tally) -> None:
    page = ocr.read_image(img)
    preds = page.lines
    # Output boxes are TL, TR, BR, BL; the dataset's are TR, TL, BL, BR.
    pquads = [ln.box[[1, 0, 3, 2]] for ln in preds]
    truth = [t for t in meta["lines"] if quad_to_rect(np.float32(t["quad"])).h >= MIN_HEIGHT]
    tiny = [t for t in meta["lines"] if t not in truth]
    tquads = [np.float32(t["quad"]) for t in truth]

    pairs = sorted(
        ((quad_iou(p, t), i, j) for i, p in enumerate(pquads) for j, t in enumerate(tquads)),
        reverse=True,
    )
    pred_of: dict[int, int] = {}  # true line -> predicted line
    used = set()
    for iou, i, j in pairs:
        if iou < 0.5:
            break
        if i not in used and j not in pred_of:
            used.add(i)
            pred_of[j] = i
    extra = [
        i
        for i, p in enumerate(pquads)
        if i not in used and not any(quad_iou(p, np.float32(t["quad"])) >= 0.5 for t in tiny)
    ]

    for key, diac in (("diac", True), ("nodiac", False)):
        for j, t in enumerate(truth):
            ref = _clean(t["text"], diac)
            hyp = _clean(preds[pred_of[j]].text, diac) if j in pred_of else ""
            tally.err[key] += edit_distance(ref, hyp)
            tally.total[key] += len(ref)
        tally.err[key] += sum(len(_clean(preds[i].text, diac)) for i in extra)

    # Reading order: true order numbers of the matched lines, in predicted order.
    true_of = {i: j for j, i in pred_of.items()}
    seq = [truth[true_of[i]]["order"] for i in range(len(preds)) if i in true_of]
    tally.pages += 1
    tally.ordered += seq == sorted(seq)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end score on synthetic pages.")
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets/pages_v1"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--kind", choices=["book", "columns", "screen"])
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--recognizer", type=Path)
    parser.add_argument("--out", type=Path, help="write the scores as JSON here")
    args = parser.parse_args(argv)

    ocr = OCR(args.detector, args.recognizer)
    tallies: dict[str, Tally] = defaultdict(Tally)
    n = 0
    for shard in sorted((args.dataset / args.split).glob("*.tar")):
        for _key, img, meta in read_pages(shard):
            if args.kind and meta["kind"] != args.kind:
                continue
            score_page(ocr, img, meta, tallies[meta["kind"]])
            n += 1
            if n % 25 == 0:
                print(f"  {n} pages", flush=True)
            if n >= args.limit:
                break
        if n >= args.limit:
            break

    overall = Tally()
    for t in list(tallies.values()):
        for key in ("diac", "nodiac"):
            overall.err[key] += t.err[key]
            overall.total[key] += t.total[key]
        overall.pages += t.pages
        overall.ordered += t.ordered
    tallies["overall"] = overall

    print(f"{n} pages")
    report = {}
    for name in ("book", "columns", "screen", "overall"):
        t = tallies.get(name)
        if not t or not t.pages:
            continue
        report[name] = {
            "pages": t.pages,
            "cer": t.cer("diac"),
            "cer_nodiac": t.cer("nodiac"),
            "order_ok": t.ordered / t.pages,
        }
        print(
            f"  {name:8} CER {t.cer('diac'):6.2%}  no-diacritics {t.cer('nodiac'):6.2%}  "
            f"order correct {t.ordered / t.pages:6.1%}  ({t.pages} pages)"
        )
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
