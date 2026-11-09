"""Render random sample lines and write an HTML page for checking them by eye.

Usage:
    uv run python -m training.inspect_samples                    # 60 lines, all fonts
    uv run python -m training.inspect_samples -n 200 --seed 3
    uv run python -m training.inspect_samples --each-font        # one line per font file

Open data/inspect/index.html in a browser. Each row shows the rendered image and,
next to it, the label as text drawn by the browser (a different shaping engine), so
you can compare them: same letters, same diacritics, same order, numbers the same way
round. Anything that differs is a bug in the renderer or the label.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import random
from pathlib import Path

from PIL import Image

from training.fonts import FontPicker, load_fonts
from training.render import render_line
from training.text import LineSampler, SamplerConfig, clean_text

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_TEXT = ROOT / "training" / "samples" / "sample_text.txt"
OUT_DIR = ROOT / "data" / "inspect"


def png_data_uri(ink) -> str:
    buf = io.BytesIO()
    Image.fromarray(255 - ink).save(buf, format="PNG")  # black ink on white paper
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


ROW = """<div class="row">
  <div class="meta">#{n} · {font} · {size}px · {style}</div>
  <img src="{src}" alt="">
  <div class="label" dir="rtl" lang="ar">{label}</div>
</div>"""

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Render check</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 16px; background: #fff; color: #111; }}
  .row {{ border-bottom: 1px solid #ddd; padding: 8px 0; max-width: 720px; }}
  img {{ max-width: 100%; height: auto; display: block; margin-left: auto; }}
  .label {{ font-family: "Segoe UI", "Geeza Pro", "Noto Naskh Arabic", sans-serif;
            font-size: 22px; text-align: right; background: #f3f6ff; padding: 2px 6px; }}
  .meta {{ color: #777; font-size: 12px; }}
</style></head><body>
<h1>Render check ({count} lines, seed {seed})</h1>
<p>Under each image is its label, drawn by the browser. Compare letters, diacritics,
word order, and numbers.</p>
{rows}
</body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render sample lines for visual checking.")
    parser.add_argument("-n", type=int, default=60, help="number of lines")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--each-font", action="store_true", help="one line per font file")
    parser.add_argument("--text", type=Path, default=SAMPLE_TEXT, help="UTF-8 text file")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    segments = clean_text(args.text.read_text(encoding="utf-8"))
    sampler = LineSampler(SamplerConfig(), seed=args.seed)
    fonts = load_fonts()
    picker = FontPicker(fonts, rng)

    rows = []
    jobs = fonts if args.each_font else [None] * args.n
    for font in jobs:
        for _ in range(100):  # find a line this font can draw
            line = sampler.sample(rng.choice(segments))
            chosen = font if font is not None else (picker.pick(line) if line else None)
            if line and chosen and chosen.supports(line):
                break
        else:
            continue
        size = rng.randint(24, 64)
        ink = render_line(line, chosen.path, size_px=size)
        rows.append(
            ROW.format(
                n=len(rows) + 1,
                src=png_data_uri(ink),
                label=html.escape(line),
                font=html.escape(f"{chosen.family} — {chosen.path.name}"),
                size=size,
                style=chosen.style,
            )
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "index.html"
    out.write_text(
        PAGE.format(count=len(rows), seed=args.seed, rows="\n".join(rows)), encoding="utf-8"
    )
    print(f"{len(rows)} lines -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
