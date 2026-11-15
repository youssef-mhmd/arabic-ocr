"""Make the app's small image files (run once; the results are committed).

    uv run python -m installers.make_assets

- src/arabic_ocr/gui/sample.png + sample.txt: a short Arabic text, rendered with
  HarfBuzz shaping (training/render.py). "Try a sample" in the app and the
  installers' self-test read it.
- src/arabic_ocr/gui/icon.png and installers/icons/: the app icon (the letter ع)
  as PNG, Windows .ico and macOS .icns.

Needs the training fonts: uv run python -m training.download_fonts
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from training.render import render_line

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "data" / "fonts"
GUI = ROOT / "src" / "arabic_ocr" / "gui"
ICONS = ROOT / "installers" / "icons"

SAMPLE = [
    "اللغة العربية من أكثر اللغات انتشارًا في العالم،",
    "ويتحدث بها أكثر من ٤٠٠ مليون إنسان.",
    "يقرأ هذا البرنامج النصوص العربية والفارسية والأردية",
    "من الصور والملفات الممسوحة دون اتصال بالإنترنت.",
]


def make_sample() -> None:
    font = FONTS / "notonaskharabic" / "NotoNaskhArabic[wght].ttf"
    ink = [render_line(text, font, size_px=46, padding=0) for text in SAMPLE]
    margin, gap = 80, 34
    width = max(i.shape[1] for i in ink) + 2 * margin
    height = sum(i.shape[0] for i in ink) + gap * (len(ink) - 1) + 2 * margin
    page = np.zeros((height, width), np.float32)  # ink, 0-1
    y = margin
    for line in ink:  # right-aligned, as Arabic text is
        h, w = line.shape
        x = width - margin - w
        page[y : y + h, x : x + w] = np.maximum(page[y : y + h, x : x + w], line / 255)
        y += h + gap
    gray = (255 - page * 230).clip(0, 255).astype(np.uint8)  # dark gray ink on white
    Image.fromarray(gray).save(GUI / "sample.png", optimize=True)
    (GUI / "sample.txt").write_text("\n".join(SAMPLE) + "\n", encoding="utf-8")
    print(f"sample.png {width}x{height}")


def make_icon(size: int = 1024) -> Image.Image:
    scale = size / 1024
    # A rounded square with a top-to-bottom teal gradient
    top, bottom = np.array([20, 140, 130]), np.array([10, 90, 110])
    t = np.linspace(0, 1, size)[:, None, None]
    grad = (top * (1 - t) + bottom * t).astype(np.uint8).repeat(size, axis=1)
    bg = Image.fromarray(grad, "RGB").convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    pad = int(64 * scale)
    ImageDraw.Draw(mask).rounded_rectangle(
        (pad, pad, size - pad, size - pad), radius=int(200 * scale), fill=255
    )
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    icon.paste(bg, (0, 0), mask)

    # The letter ع in white, centred on its ink
    font = FONTS / "notokufiarabic" / "NotoKufiArabic[wght].ttf"
    ink = render_line("ع", font, size_px=int(900 * scale), padding=0)
    ys, xs = np.nonzero(ink > 12)
    ink = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    target = int(560 * scale)
    h, w = ink.shape
    k = target / max(h, w)
    glyph = Image.fromarray(ink).resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)
    white = Image.new("RGBA", glyph.size, (255, 255, 255, 255))
    icon.paste(white, ((size - glyph.width) // 2, (size - glyph.height) // 2), glyph)
    return icon


def main() -> None:
    make_sample()
    ICONS.mkdir(parents=True, exist_ok=True)
    icon = make_icon()
    icon.resize((256, 256), Image.LANCZOS).save(GUI / "icon.png", optimize=True)
    icon.resize((512, 512), Image.LANCZOS).save(ICONS / "arabic-ocr.png", optimize=True)
    icon.save(ICONS / "arabic-ocr.ico", sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    icon.save(ICONS / "arabic-ocr.icns")
    print("icons ->", ICONS)


if __name__ == "__main__":
    main()
