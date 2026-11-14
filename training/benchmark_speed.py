"""Speed benchmark: how long does the full pipeline take per page on the CPU?

    uv run python -m training.benchmark_speed                      # synthetic A4 pages
    uv run python -m training.benchmark_speed scans/*.png          # your own pages
    uv run python -m training.benchmark_speed --threads 4 --repeat 3

The M3 target (PRD): an A4 page at 300 DPI (2480 x 3508 px) in <= 3 s on a 4-core
CPU. `--threads 4` limits ONNX Runtime to 4 threads to imitate that machine.

Without image arguments it takes book pages from the pages_v2 test split and
enlarges them to A4 at 300 DPI, which is the size a scanner produces.

`--truth-boxes` replaces the detector with the dataset's true line boxes (synthetic
pages only). It measures everything except detection, so it works before a
detector model exists.

Runtime code only (ONNX Runtime): no torch needed.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import cv2
import numpy as np

from arabic_ocr.detect.boxes import Detection
from arabic_ocr.pipeline import OCR, load_image
from arabic_ocr.recognize.recognizer import LineRecognizer
from training.make_pages import read_pages

PAGES = Path("data/datasets/pages_v2/test")
A4_WIDTH = 2480  # pixels at 300 DPI
TARGET_SECONDS = 3.0


def synthetic_a4(count: int) -> list[tuple[str, np.ndarray, list[np.ndarray]]]:
    """(name, A4-sized page, true line quads) for the first `count` book pages."""
    out = []
    for shard in sorted(PAGES.glob("*.tar")):
        for key, img, meta in read_pages(shard):
            if meta["kind"] == "screen":
                continue
            s = A4_WIDTH / img.shape[1]
            big = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            quads = [np.float32(ln["quad"]) * s for ln in meta["lines"]]
            out.append((key, big, quads))
            if len(out) == count:
                return out
    return out


class TruthDetector:
    """Stand-in detector that returns given boxes (for --truth-boxes)."""

    def __init__(self) -> None:
        self.quads: list[np.ndarray] = []

    def detect(self, gray: np.ndarray) -> list[Detection]:
        return [Detection(q, 1.0) for q in self.quads]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Time the OCR pipeline per page.")
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--pages", type=int, default=5, help="synthetic pages to use")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--recognizer", type=Path)
    parser.add_argument("--truth-boxes", action="store_true")
    args = parser.parse_args(argv)

    if args.images:
        if args.truth_boxes:
            parser.error("--truth-boxes only works with synthetic pages")
        pages = [(p.name, load_image(p), []) for p in args.images]
    else:
        pages = synthetic_a4(args.pages)
        if not pages:
            parser.error(f"no pages found in {PAGES}")

    if args.truth_boxes:
        ocr = OCR.__new__(OCR)
        ocr.detector = TruthDetector()
        rec = args.recognizer or Path("models/recognizer-v3.onnx")
        ocr.recognizer = LineRecognizer(rec, args.threads)
    else:
        ocr = OCR(args.detector, args.recognizer, threads=args.threads)

    print(f"{len(pages)} pages, {args.threads} threads, best of {args.repeat} runs each")
    totals = []
    for name, image, quads in pages:
        if args.truth_boxes:
            ocr.detector.quads = quads
        ocr.read_image(image)  # warm-up
        runs = [ocr.read_image(image) for _ in range(args.repeat)]
        best = min(runs, key=lambda p: p.seconds)
        totals.append(best.seconds)
        steps = "  ".join(f"{k} {v:.2f}" for k, v in best.timings.items())
        h, w = image.shape[:2]
        print(f"{name}  {w}x{h}  {len(best.lines)} lines  {best.seconds:.2f} s  ({steps})")

    med = statistics.median(totals)
    verdict = "PASS" if med <= TARGET_SECONDS else "FAIL"
    print(f"median {med:.2f} s per page (target <= {TARGET_SECONDS:.0f} s): {verdict}")
    if args.truth_boxes:
        print("note: --truth-boxes skips the detector; real runs will be slower")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
