"""End-to-end score on whole pages: detector + recognizer + reading order.

    uv run python -m training.eval_pages                              # synthetic pages_v2 test
    uv run python -m training.eval_pages --real data/real/pages_screens
    uv run python -m training.eval_pages --real data/real/pages_printed --out scores.json

The full pipeline (arabic_ocr.pipeline.OCR, ONNX on the CPU) reads every page.

**Pages with line boxes** (synthetic pages, screenshots):
- *CER*: true lines and predicted lines that overlap (by at least half of the
  smaller one) are grouped; within a group, the texts are joined in reading order
  and compared. So a line the detector split in two, or two menu items it merged,
  costs only the characters actually misread. A missed line counts as all its
  characters wrong; a predicted line on no text counts its characters as extra
  errors.
- *Detection* precision/recall at IoU >= 0.5 (strict one-to-one matching).
- *Order*: share of pages whose matched lines come out in exactly the true order.

**Pages without boxes** (scanned book pages): the whole page text is compared with
the transcription (line breaks treated as spaces), so detection misses, extra
lines and reading-order mistakes all count as character errors.

Every score is reported as given, with diacritics removed, and (for real pages)
also ignoring transcription conventions (digit style, spaces before punctuation;
see training.metrics.ignore_conventions).

Runtime code only (ONNX Runtime): no torch needed.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from arabic_ocr import charset
from arabic_ocr.detect.boxes import quad_iou, quad_to_rect
from arabic_ocr.layout import row_order
from arabic_ocr.pipeline import OCR, load_image
from training.detect_eval import MIN_HEIGHT, match
from training.make_pages import read_pages
from training.metrics import fast_edit_distance, ignore_conventions

#: name -> (strip diacritics?, ignore conventions?)
VARIANTS = {"cer": (False, False), "nodiac": (True, False), "conv_nodiac": (True, True)}


def _clean(text: str, nodiac: bool, conventions: bool = False) -> str:
    text = charset.strip_tatweel(charset.normalize(text))
    if nodiac:
        text = charset.strip_diacritics(text)
    if conventions:
        text = ignore_conventions(text)
    return " ".join(text.split())


class Tally:
    def __init__(self) -> None:
        self.err: dict[str, int] = defaultdict(int)
        self.total: dict[str, int] = defaultdict(int)
        self.pages = self.ordered = self.boxed = 0
        self.tp = self.n_pred = self.n_true = 0
        self.seconds = 0.0

    def add(self, other: Tally) -> None:
        for k in VARIANTS:
            self.err[k] += other.err[k]
            self.total[k] += other.total[k]
        for f in ("pages", "ordered", "boxed", "tp", "n_pred", "n_true", "seconds"):
            setattr(self, f, getattr(self, f) + getattr(other, f))

    def rate(self, key: str) -> float:
        return self.err[key] / max(self.total[key], 1)

    def report(self) -> dict:
        out = {"pages": self.pages, "seconds_per_page": self.seconds / max(self.pages, 1)}
        out |= {k: self.rate(k) for k in VARIANTS}
        if self.boxed:
            out["precision"] = self.tp / max(self.n_pred, 1)
            out["recall"] = self.tp / max(self.n_true, 1)
            out["order_ok"] = self.ordered / self.boxed
        return out


def _aabb(quad: np.ndarray) -> tuple[float, float, float, float]:
    return quad[:, 0].min(), quad[:, 1].min(), quad[:, 0].max(), quad[:, 1].max()


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over the smaller area."""
    a32, b32 = np.float32(a), np.float32(b)
    inter, _ = cv2.intersectConvexConvex(a32, b32)
    small = min(cv2.contourArea(a32), cv2.contourArea(b32))
    return float(inter / small) if small > 0 else 0.0


def score_boxed(page, meta: dict, t: Tally) -> None:
    """Score a page whose labels have line boxes (see module docstring)."""
    preds = page.lines
    pquads = [ln.box[[1, 0, 3, 2]] for ln in preds]  # TL,TR,BR,BL -> TR,TL,BL,BR
    truth = [x for x in meta["lines"] if quad_to_rect(np.float32(x["quad"])).h >= MIN_HEIGHT]
    tiny = [np.float32(x["quad"]) for x in meta["lines"] if x not in truth]
    tquads = [np.float32(x["quad"]) for x in truth]

    # Detection, strict: one-to-one at IoU >= 0.5.
    c = match(pquads, tquads, tiny)
    t.tp, t.n_pred, t.n_true = t.tp + c.tp, t.n_pred + c.n_pred, t.n_true + c.n_true

    # Groups of overlapping true/predicted lines (union-find over both lists).
    n_t = len(truth)
    parent = list(range(n_t + len(preds)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for j, tq in enumerate(tquads):
        for i, pq in enumerate(pquads):
            if _overlap(tq, pq) >= 0.5:
                parent[find(n_t + i)] = find(j)
    ignored = {i for i, pq in enumerate(pquads) if any(_overlap(q, pq) >= 0.5 for q in tiny)}
    groups: dict[int, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for j in range(n_t):
        groups[find(j)][0].append(j)
    for i in range(len(preds)):
        if i not in ignored or find(n_t + i) != n_t + i:
            groups[find(n_t + i)][1].append(i)

    tboxes = [_aabb(q) for q in tquads]
    pboxes = [_aabb(q) for q in pquads]
    for tj, pi in groups.values():
        ref_raw = " ".join(truth[j]["text"] for j in row_order(tboxes, tj))
        hyp_raw = " ".join(preds[i].text for i in row_order(pboxes, pi))
        for key, (nodiac, conv) in VARIANTS.items():
            ref, hyp = _clean(ref_raw, nodiac, conv), _clean(hyp_raw, nodiac, conv)
            t.err[key] += fast_edit_distance(ref, hyp)
            t.total[key] += len(ref)

    # Reading order of the one-to-one IoU matches.
    pairs = sorted(
        ((quad_iou(p, q), i, j) for i, p in enumerate(pquads) for j, q in enumerate(tquads)),
        reverse=True,
    )
    true_of: dict[int, int] = {}
    used: set[int] = set()
    for iou, i, j in pairs:
        if iou < 0.5:
            break
        if i not in true_of and j not in used:
            true_of[i] = j
            used.add(j)
    seq = [truth[true_of[i]]["order"] for i in range(len(preds)) if i in true_of]
    t.boxed += 1
    t.ordered += seq == sorted(seq)


def score_text(page, meta: dict, t: Tally) -> None:
    """Score a page that only has its full text (see module docstring)."""
    for key, (nodiac, conv) in VARIANTS.items():
        ref = _clean(meta["text"].replace("\n", " "), nodiac, conv)
        hyp = _clean(page.text.replace("\n", " "), nodiac, conv)
        t.err[key] += fast_edit_distance(ref, hyp)
        t.total[key] += len(ref)


def iter_synthetic(dataset: Path, split: str, kind: str | None):
    for shard in sorted((dataset / split).glob("*.tar")):
        for key, img, meta in read_pages(shard):
            if kind is None or meta["kind"] == kind:
                yield key, img, meta


def iter_real(folder: Path):
    for label in sorted((folder / "labels").glob("*.json")):
        meta = json.loads(label.read_text(encoding="utf-8"))
        image = next((folder / "images").glob(label.stem + ".*"))
        yield label.stem, load_image(image), meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end score on whole pages.")
    parser.add_argument("--real", type=Path, help="a real page set folder (images/ + labels/)")
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets/pages_v2"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--kind", choices=["book", "columns", "screen"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--recognizer", type=Path)
    parser.add_argument("--out", type=Path, help="write the scores (and per-page) as JSON")
    args = parser.parse_args(argv)

    ocr = OCR(args.detector, args.recognizer)
    pages = (
        iter_real(args.real) if args.real else iter_synthetic(args.dataset, args.split, args.kind)
    )
    tallies: dict[str, Tally] = defaultdict(Tally)
    per_page = {}
    for n, (key, img, meta) in enumerate(pages, 1):
        page = ocr.read_image(img)
        t = Tally()
        t.pages, t.seconds = 1, page.seconds
        if meta.get("lines") is not None:
            score_boxed(page, meta, t)
        else:
            score_text(page, meta, t)
        tallies[meta.get("layout") or meta.get("style") or meta["kind"]].add(t)
        per_page[key] = t.report() | {"hyp": page.text} if args.real else t.report()
        if n % 25 == 0:
            print(f"  {n} pages", flush=True)
        if n >= args.limit:
            break

    overall = Tally()
    for t in list(tallies.values()):
        overall.add(t)
    tallies["overall"] = overall

    print(f"{overall.pages} pages ({args.real or args.dataset / args.split})")
    for name, t in tallies.items():
        line = (
            f"  {name:9} CER {t.rate('cer'):6.2%}  no-diac {t.rate('nodiac'):6.2%}  "
            f"no-diac+conventions {t.rate('conv_nodiac'):6.2%}"
        )
        if t.boxed:
            r = t.report()
            line += f"  P {r['precision']:.1%} R {r['recall']:.1%}  order {r['order_ok']:.0%}"
        print(line + f"  ({t.pages} pages, {t.seconds / max(t.pages, 1):.2f} s/page)")
    if args.out:
        report = {"summary": {k: t.report() for k, t in tallies.items()}, "pages": per_page}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    t0 = time.perf_counter()
    code = main()
    print(f"({time.perf_counter() - t0:.0f} s)")
    raise SystemExit(code)
