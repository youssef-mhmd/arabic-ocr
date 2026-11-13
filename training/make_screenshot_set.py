"""Build the *screenshot* real-world test set: lines drawn by a real browser.

Usage:
    uv run python -m training.make_screenshot_set            # writes the HTML page
    (then capture each line, see below)
    uv run python -m training.make_screenshot_set --check    # verify the PNGs exist

Why: our training images are all drawn by our own renderer (HarfBuzz + FreeType)
with Google fonts. Real screenshots look different: the operating system's fonts
(Segoe UI, Tahoma, Traditional Arabic…), ClearType-style smoothing, small UI text
sizes, coloured and dark-mode backgrounds. None of these fonts are in training.

How: this script picks sentences from the dataset's *test* split (never trained on),
gives each a font, size and colour scheme, and writes one HTML page with one line
per element. A browser draws the page and saves a screenshot of each line element
to `data/real/screenshots/images/<id>.png`. Because we wrote the text ourselves, the
labels (`labels.tsv`) are exact: no hand typing.

The capture step uses a browser (Chromium via Playwright). The page also works in
any browser: open it, and screenshot the lines yourself if you like.

Fonts: Windows system fonts with Arabic. On macOS/Linux the browser falls back to
other fonts; the set is meant to be built once on the owner's Windows laptop.
"""

from __future__ import annotations

import argparse
import html
import random
import sys
from pathlib import Path

from arabic_ocr import charset
from training.dataset import read_split

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "real" / "screenshots"

FONTS = [
    "Segoe UI",
    "Tahoma",
    "Arial",
    "Times New Roman",
    "Traditional Arabic",
    "Simplified Arabic",
    "Sakkal Majalla",
    "Arabic Typesetting",
    "Microsoft Sans Serif",
    "Calibri",
    "Courier New",
    "Andalus",
    "Microsoft Uighur",
    "Urdu Typesetting",
    "Aldhabi",
]
# (text colour, background): light, grey, dark mode, coloured.
SCHEMES = [
    ("#000000", "#ffffff"),
    ("#202124", "#ffffff"),
    ("#333333", "#f3f3f3"),
    ("#444444", "#fafafa"),
    ("#e8e8e8", "#1e1e1e"),
    ("#ffffff", "#2b2b2b"),
    ("#d4d4d4", "#000000"),
    ("#1a0dab", "#ffffff"),
    ("#ffffff", "#0b57d0"),
    ("#0f5132", "#d1e7dd"),
    ("#664d03", "#fff3cd"),
]
SIZES_PX = [13, 14, 15, 16, 18, 20, 22, 26]  # typical UI and web body sizes
WEIGHTS = ["normal", "normal", "normal", "bold"]


def pick_sentences(n: int, seed: int) -> list[str]:
    """Unseen test-split labels: 8-60 characters, at least one Arabic letter."""
    pool = []
    for s in read_split(ROOT / "data/datasets/v1", "test"):
        text = s.label.strip()
        if 8 <= len(text) <= 60 and any("ء" <= c <= "ي" for c in text):
            pool.append(text)
    rng = random.Random(seed)
    rng.shuffle(pool)
    seen, out = set(), []
    for t in pool:  # no duplicate sentences
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) == n:
            break
    return out


def build(n: int, seed: int) -> None:
    rng = random.Random(seed)
    texts = pick_sentences(n, seed)
    rows, divs = [], []
    for i, text in enumerate(texts):
        font = FONTS[i % len(FONTS)]
        fg, bg = rng.choice(SCHEMES)
        size = rng.choice(SIZES_PX)
        weight = rng.choice(WEIGHTS)
        line_id = f"s{i:03d}"
        style = (
            f"font-family:'{font}';font-size:{size}px;font-weight:{weight};"
            f"color:{fg};background:{bg}"
        )
        divs.append(f'<div class="line" id="{line_id}" style="{style}">{html.escape(text)}</div>')
        dark = int(bg[1:3], 16) < 128
        mode = "dark" if dark else "light"
        rows.append(f"{line_id}.png\t{charset.normalize(text)}\t{font} | {size}px {weight} {mode}")

    OUT.mkdir(parents=True, exist_ok=True)
    page = f"""<!doctype html>
<html lang="ar" dir="rtl"><head><meta charset="utf-8"><title>screenshot lines</title>
<style>
  body {{ margin: 16px; background: #888; }}
  .line {{ display: inline-block; clear: both; float: right; white-space: nowrap;
           padding: 4px 6px; margin: 0 0 10px 0; line-height: 1.5; }}
</style></head><body>
{chr(10).join(divs)}
</body></html>
"""
    (OUT / "page.html").write_text(page, encoding="utf-8")
    (OUT / "labels.tsv").write_text("file\ttext\tnote\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {OUT / 'page.html'} and labels.tsv ({len(rows)} lines).")
    print("Next: capture each .line element to images/<id>.png (see the docstring).")


def check() -> int:
    lines = (OUT / "labels.tsv").read_text(encoding="utf-8").splitlines()[1:]
    missing = [
        ln.split("\t")[0] for ln in lines if not (OUT / "images" / ln.split("\t")[0]).exists()
    ]
    print(f"{len(lines) - len(missing)}/{len(lines)} images present.")
    return 1 if missing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the screenshot test set page.")
    parser.add_argument("-n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        return check()
    build(args.n, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
