"""Build the *printed* real-world test set from OpenITI's gold-standard OCR data.

Usage:
    uv run python -m training.make_printed_set                    # dev set: 24 lines x 5 books
    uv run python -m training.make_printed_set --set heldout      # final check: 40 x 2 books

Two sets from *different* books:
- `printed` (development): used while improving the model. We look at its mistakes
  and change things because of them, so over time it stops being a fair test.
- `printed_heldout` (final check): other books, scored only when a model is
  finished, never used to make decisions. This is the honest real-world number.

Source: https://github.com/OpenITI/OCR_GS_Data (CC BY-NC-SA 4.0, see data/LICENSES.md).
Real scans of printed classical Arabic books, cut into lines, with transcriptions
that two people checked ("gold standard"). We use them **for evaluation only**:
the license forbids using them for a model we ship.

Each book's `7_final` folder has `000123.png` (the line image) and `000123.gt.txt`
(its text). This script picks lines at random (fixed seed, so the same set every
time), downloads just those files, and writes `data/real/printed/labels.tsv` +
`images/`, the format `evaluate_recognizer.py --real` reads.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

from arabic_ocr import charset

ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = ROOT / "data" / "real"
REPO = "OpenITI/OCR_GS_Data"
# set name -> (output folder, lines per source, [(source folder, subfolder with lines)])
SETS = {
    "dev": (
        "printed",
        24,
        [
            ("book_IbnAthir.Kamil", "7_final"),
            ("book_IbnFaqihHamadhani.Buldan", "7_final"),
            ("book_IbnQutayba.Adab", "7_final"),
            ("book_Jahiz.Hayawan", "7_final"),
            ("book_Yacqubi.Tarikh", "7_final"),
        ],
    ),
    "heldout": (
        "printed_heldout",
        40,
        [
            ("lq_Dhahabi.Tarikh", "7_final"),
            ("lq_IbnJawzi.Muntazam", "7_final"),
            # Not used: ara/ibncabdzahir. Its .gt.txt files turned out to be
            # uncorrected machine OCR (commas typed as » ؛ +, wrong letters, garbled
            # digits on most lines; checked by eye 2026-11-14), so it can't serve
            # as a test of accuracy.
        ],
    ),
}
HEADERS = {"User-Agent": "arabic-ocr-dev"}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def line_ids(folder: str) -> list[str]:
    """Line ids that have both an image and a transcription (first 1000 files)."""
    url = f"https://api.github.com/repos/{REPO}/contents/ara/{folder}"
    names = {e["name"] for e in json.loads(fetch(url))}
    return sorted(
        n.removesuffix(".gt.txt")
        for n in names
        if n.endswith(".gt.txt") and n.removesuffix(".gt.txt") + ".png" in names
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the printed real-world test set.")
    parser.add_argument("--set", choices=sorted(SETS), default="dev")
    parser.add_argument("--per-book", type=int, help="lines per source (default: per set)")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)

    folder_name, per_book, sources = SETS[args.set]
    per_book = args.per_book or per_book
    out = REAL_DIR / folder_name
    rng = random.Random(args.seed)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows, unsupported = [], set()
    for book, sub in sources:
        folder = f"{book}/{sub}" if sub else book
        ids = line_ids(folder)
        chosen = sorted(rng.sample(ids, min(per_book, len(ids))))
        short = book.removeprefix("book_").removeprefix("lq_")
        for i in chosen:
            raw = f"https://raw.githubusercontent.com/{REPO}/master/ara/{folder}/{i}"
            text = charset.normalize(fetch(raw + ".gt.txt").decode("utf-8").strip())
            if not text:
                continue
            name = f"{short}_{i[:12]}.png"
            (out / "images" / name).write_bytes(fetch(raw + ".png"))
            rows.append(f"{name}\t{text}\t{short}")
            unsupported |= charset.unsupported_chars(text)
        print(f"{short}: {len(chosen)} lines (of {len(ids)} listed)")

    (out / "labels.tsv").write_text("file\ttext\tnote\n" + "\n".join(rows) + "\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in (out / "images").iterdir()) / 2**20
    print(f"Wrote {len(rows)} lines to {out} ({size:.1f} MB).")
    if unsupported:
        shown = " ".join(f"U+{ord(c):04X}" for c in sorted(unsupported))
        print(f"Characters outside our charset (will count as errors): {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
