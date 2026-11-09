"""Render one line of Arabic-script text to an image, with correct shaping.

How it works:
1. **Bidi runs.** The line's base direction is right-to-left. Numbers inside it are
   drawn left-to-right (so "سنة 2026" shows 2026, not 6202). Our charset has no Latin
   letters, so numbers are the only left-to-right runs. `split_runs` finds them with
   the Unicode Bidirectional Algorithm, so lines look the way browsers and word
   processors display them.
2. **Shaping (HarfBuzz).** For each run, HarfBuzz picks the right glyph for every
   letter (initial/medial/final/isolated forms, ligatures like لا) and positions the
   diacritics. It returns glyph ids and x/y positions.
3. **Rasterizing (FreeType).** FreeType draws each glyph as a grayscale bitmap; we
   paste the bitmaps at HarfBuzz's positions.

The result is an "ink" image: 0 = paper, 255 = full ink. Augmentation later turns it
into realistic paper/screen images. The label is always the input text, unchanged,
in logical order.
"""

from __future__ import annotations

import functools
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import freetype
import numpy as np
import uharfbuzz as hb


class MissingGlyphError(ValueError):
    """The font has no glyph for a character in the text."""


# ---------------------------------------------------------------------------
# Bidi: which characters are drawn left-to-right inside a right-to-left line
# ---------------------------------------------------------------------------
#
# This is the Unicode Bidirectional Algorithm (UAX #9), reduced to what our charset
# can contain: a right-to-left paragraph, Arabic letters (type AL), no Latin letters
# (no type L), and no explicit direction controls. Under those conditions the full
# algorithm gives exactly this result:
#   - numbers end up left-to-right (level 2); everything else right-to-left (level 1).
# The W-rules below decide which characters count as part of a number.


def _bidi_type(ch: str) -> str:
    return unicodedata.bidirectional(ch) or "ON"


def ltr_mask(text: str) -> list[bool]:
    """For each character, True if it is drawn left-to-right (part of a number)."""
    types = [_bidi_type(c) for c in text]
    n = len(types)

    # W1: a combining mark takes the type of the character before it.
    for i, t in enumerate(types):
        if t == "NSM":
            types[i] = types[i - 1] if i > 0 else "R"

    # W2: a European number after Arabic letters becomes an Arabic number.
    # W3: Arabic letters (AL) are then treated as plain right-to-left (R).
    last_strong = "R"  # start of a right-to-left paragraph
    for i, t in enumerate(types):
        if t in ("R", "L", "AL"):
            last_strong = t
        elif t == "EN" and last_strong == "AL":
            types[i] = "AN"
    types = ["R" if t == "AL" else t for t in types]

    # W4: one separator between two numbers of the same kind joins them.
    #     + and - (ES) join only European numbers; , . : / (CS) join either kind.
    for i in range(1, n - 1):
        left, mid, right = types[i - 1], types[i], types[i + 1]
        if mid == "ES" and left == right == "EN":
            types[i] = "EN"
        elif mid == "CS" and left == right and left in ("EN", "AN"):
            types[i] = left

    # W5: % signs (ET) next to a European number become part of it.
    i = 0
    while i < n:
        if types[i] == "ET":
            j = i
            while j < n and types[j] == "ET":
                j += 1
            if (i > 0 and types[i - 1] == "EN") or (j < n and types[j] == "EN"):
                types[i:j] = ["EN"] * (j - i)
            i = j
        else:
            i += 1

    # W6, W7, N1, N2, I2: with no left-to-right letters anywhere, every remaining
    # neutral resolves to right-to-left, and numbers (EN/AN) get level 2 (LTR).
    return [t in ("EN", "AN") for t in types]


@dataclass(frozen=True)
class Run:
    text: str
    rtl: bool


def split_runs(text: str) -> list[Run]:
    """Split a line into direction runs, in **visual order, left to right**.

    The paragraph is right-to-left, so the logically first run is drawn rightmost
    (Unicode rule L2: reverse the order of runs). Inside each run, HarfBuzz handles
    the glyph order.
    """
    runs: list[Run] = []
    for ch, ltr in zip(text, ltr_mask(text), strict=True):
        if runs and runs[-1].rtl == (not ltr):
            runs[-1] = Run(runs[-1].text + ch, runs[-1].rtl)
        else:
            runs.append(Run(ch, rtl=not ltr))
    return list(reversed(runs))


@dataclass
class LoadedFont:
    """A font opened once in both HarfBuzz and FreeType, at a given pixel size."""

    path: Path
    size_px: int
    hb_font: hb.Font
    ft_face: freetype.Face
    ascender: float  # pixels above the baseline
    descender: float  # pixels below the baseline (positive number)


@functools.lru_cache(maxsize=64)
def load_font(path: str | Path, size_px: int) -> LoadedFont:
    path = Path(path)
    blob = hb.Blob.from_file_path(str(path))
    hb_font = hb.Font(hb.Face(blob))
    # Positions come back in 1/64 pixel ("26.6 fixed point"), same as FreeType.
    hb_font.scale = (size_px * 64, size_px * 64)

    ft_face = freetype.Face(str(path))
    ft_face.set_pixel_sizes(0, size_px)

    extents = hb_font.get_font_extents("ltr")
    return LoadedFont(
        path=path,
        size_px=size_px,
        hb_font=hb_font,
        ft_face=ft_face,
        ascender=extents.ascender / 64,
        descender=-extents.descender / 64,
    )


def _shape(font: LoadedFont, run: Run) -> list[tuple[int, float, float, float]]:
    """Shape one run. Returns (glyph id, x offset, y offset, x advance) in pixels,
    in visual left-to-right order."""
    buf = hb.Buffer()
    buf.add_str(run.text)
    buf.direction = "rtl" if run.rtl else "ltr"
    buf.script = "Arab"
    buf.language = "ar"
    hb.shape(font.hb_font, buf)
    glyphs = []
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions, strict=True):
        if info.codepoint == 0:
            ch = run.text[info.cluster]
            raise MissingGlyphError(f"{font.path.name} has no glyph for {ch!r} (U+{ord(ch):04X})")
        glyphs.append((info.codepoint, pos.x_offset / 64, pos.y_offset / 64, pos.x_advance / 64))
    return glyphs


def render_line(
    text: str, font_path: str | Path, size_px: int = 48, padding: int = 4
) -> np.ndarray:
    """Render `text` with the given font. Returns a uint8 ink image (H, W).

    The image height covers the font's ascender-to-descender box plus any ink that
    sticks out of it (e.g. stacked diacritics), plus `padding` on every side.
    Raises MissingGlyphError if the font can't draw a character.
    """
    if not text:
        raise ValueError("cannot render an empty line")
    font = load_font(str(font_path), size_px)
    face = font.ft_face

    # 1. Shape every run and lay the glyphs out left to right on one baseline (y = 0).
    placed = []  # (glyph id, x, y) with y pointing up from the baseline
    pen_x = 0.0
    for run in split_runs(text):
        for gid, x_off, y_off, x_adv in _shape(font, run):
            placed.append((gid, pen_x + x_off, y_off))
            pen_x += x_adv

    # 2. Rasterize each glyph and remember its bitmap and top-left corner.
    bitmaps = []  # (array, left, top) in a y-down coordinate system, baseline at y = 0
    for gid, x, y in placed:
        face.load_glyph(gid, freetype.FT_LOAD_NO_HINTING | freetype.FT_LOAD_RENDER)
        slot = face.glyph
        bmp = slot.bitmap
        if bmp.width == 0 or bmp.rows == 0:
            continue  # e.g. a space
        arr = np.array(bmp.buffer, dtype=np.uint8).reshape(bmp.rows, bmp.pitch)[:, : bmp.width]
        left = round(x) + slot.bitmap_left
        top = -round(y) - slot.bitmap_top
        bitmaps.append((arr, left, top))

    # 3. Work out the canvas: union of the font's line box and all the ink.
    x0, x1 = 0, max(1, round(pen_x))
    y0, y1 = -round(font.ascender), round(font.descender)
    for arr, left, top in bitmaps:
        x0, x1 = min(x0, left), max(x1, left + arr.shape[1])
        y0, y1 = min(y0, top), max(y1, top + arr.shape[0])

    height = y1 - y0 + 2 * padding
    width = x1 - x0 + 2 * padding
    canvas = np.zeros((height, width), dtype=np.uint8)
    for arr, left, top in bitmaps:
        r, c = top - y0 + padding, left - x0 + padding
        region = canvas[r : r + arr.shape[0], c : c + arr.shape[1]]
        np.maximum(region, arr, out=region)  # overlapping glyphs: keep the darker ink
    return canvas


def font_supports(text: str, missing: str) -> bool:
    """True if a font whose missing symbols are `missing` can draw `text`."""
    return not (set(text) & set(missing))
