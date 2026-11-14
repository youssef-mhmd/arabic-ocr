"""Compose synthetic full pages for training the text-line detector.

The detector's job: look at a whole page and mark where each line of text is. To
learn it, it needs many pages where we know every line's position. We make them by
laying out lines rendered with our own renderer (training/render.py), the same way a
word processor or a browser would:

- **book**: one column of justified paragraphs, sometimes a header, a page number
  and footnotes under a short rule;
- **columns**: two columns (the first one on the right, as Arabic is read);
- **screen**: blocks of UI-like text (titles, menu items, short paragraphs) on
  coloured or dark backgrounds, at screen sizes.

Labels: for every line, the box around its ink (a 4-point polygon, so it can rotate
with the page), its text, its block, and its reading order.

Then the page is made to look printed/scanned or screen-captured, reusing the
building blocks of training/augment.py (paper texture, shadows, blur, noise, JPEG,
black-and-white thresholding, slight rotation).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import cv2
import numpy as np

from arabic_ocr import charset
from training import augment as A
from training.fonts import FontPicker
from training.render import MissingGlyphError, advance_width, render_line


@dataclass
class PageConfig:
    kind_weights: dict[str, float] = field(
        default_factory=lambda: {"book": 0.45, "columns": 0.25, "screen": 0.30}
    )
    page_width: tuple[int, int] = (900, 1600)  # pixels (detector scale)
    body_size: tuple[int, int] = (16, 42)  # font size in pixels
    line_spacing: tuple[float, float] = (1.4, 2.1)  # line pitch / font size
    strip_diacritics_prob: float = 0.6  # per page
    rotate_prob: float = 0.4
    max_rotate_deg: float = 3.0


@dataclass
class Line:
    quad: np.ndarray  # (4, 2) float32: top-right, top-left, bottom-left, bottom-right
    text: str
    block: int
    order: int  # reading order on the page (0 = first)


@dataclass
class Block:
    x0: int
    y0: int
    x1: int
    y1: int
    bg: float = 255.0  # background gray level (screen pages)
    fg: float = 0.0  # text gray level


# ---------------------------------------------------------------------------
# Text layout
# ---------------------------------------------------------------------------


def break_lines(words: list[str], font_path, size: int, width: float) -> list[str]:
    """Greedy line breaking: as many words per line as fit in `width` pixels."""
    lines, cur = [], ""
    for w in words:
        trial = f"{cur} {w}" if cur else w
        if cur and advance_width(trial, font_path, size) > width:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def justify_scale(line: str, font_path, size: int, width: float) -> float:
    """Word-space scale that makes `line` exactly `width` wide (1.0 if it can't)."""
    n_spaces = line.count(" ")
    if not n_spaces:
        return 1.0
    w1 = advance_width(line, font_path, size, 1.0)
    w2 = advance_width(line, font_path, size, 2.0)
    per_unit = (w2 - w1) or 1.0
    scale = 1.0 + (width - w1) / per_unit
    return scale if 1.0 <= scale <= 4.0 else 1.0


class PageComposer:
    """Lays out text on an "ink" canvas (0 = paper, 255 = ink) and records lines."""

    def __init__(self, width: int, height: int) -> None:
        self.ink = np.zeros((height, width), np.uint8)
        self.lines: list[Line] = []
        self.blocks: list[Block] = []

    def place(
        self, text: str, font_path, size: int, right: int, y: int, max_y: int | None = None, **kw
    ) -> int | None:
        """Draw one line with its right edge at x = `right` and its top at y.

        Returns the line's height in pixels, or None if it doesn't fit / can't draw.
        A line whose ink would reach below `max_y` is not drawn at all (every line
        that is drawn must get a label, or the detector learns to ignore text).
        """
        try:
            canvas = render_line(text, font_path, size_px=size, padding=0, **kw)
        except MissingGlyphError:
            return None
        h, w = canvas.shape
        x = right - w
        if x < 0 or y < 0 or y + h > self.ink.shape[0]:
            return None
        rows = np.where(canvas.max(axis=1) > 64)[0]
        cols = np.where(canvas.max(axis=0) > 64)[0]
        if len(rows) == 0:
            return None
        if max_y is not None and y + rows[-1] + 1 > max_y:
            return None
        region = self.ink[y : y + h, x : x + w]
        np.maximum(region, canvas, out=region)
        x0, x1 = x + cols[0], x + cols[-1] + 1
        y0, y1 = y + rows[0], y + rows[-1] + 1
        quad = np.float32([[x1, y0], [x0, y0], [x0, y1], [x1, y1]])
        self.lines.append(Line(quad, text, len(self.blocks) - 1, len(self.lines)))
        return h

    def paragraph(self, text, font, size, x0, x1, y, pitch, justify=True, max_y=None) -> int:
        """Lay out a paragraph between x0 and x1, starting at y (right-aligned or
        justified), stopping before any line would reach below `max_y`.
        Returns the y just below its last line."""
        width = x1 - x0
        lines = break_lines(text.split(), font.path, size, width)
        for i, ln in enumerate(lines):
            last = i == len(lines) - 1
            scale = justify_scale(ln, font.path, size, width) if justify and not last else 1.0
            if y + pitch > self.ink.shape[0]:
                break
            if self.place(ln, font.path, size, x1, y, max_y, space_scale=scale) is None:
                break
            y += pitch
        return y


# ---------------------------------------------------------------------------
# Page types
# ---------------------------------------------------------------------------


def _text(texts, rng: random.Random, strip: bool, min_words: int = 8) -> str:
    """A paragraph of at least `min_words` words from the corpora."""
    words: list[str] = []
    for _ in range(10):
        para = texts.pick_source(rng).random_line(rng)
        if strip:
            para = charset.strip_diacritics(para)
        words += para.split()
        if len(words) >= min_words:
            break
    return " ".join(words)


def _pick_font(picker: FontPicker, text: str, core_only: bool = True):
    for _ in range(5):
        f = picker.pick(text)
        if f is not None and (f.role == "core" or not core_only):
            return f
    return picker.pick(text)


def compose_book(comp, rng, texts, picker, cfg, columns: int) -> None:
    H, W = comp.ink.shape
    margin = int(W * rng.uniform(0.06, 0.12))
    strip = rng.random() < cfg.strip_diacritics_prob
    size = rng.randint(*cfg.body_size)
    pitch = int(size * rng.uniform(*cfg.line_spacing))
    y = int(H * rng.uniform(0.04, 0.09))

    if rng.random() < 0.6:  # header (title) across the full width
        title = " ".join(_text(texts, rng, strip, 3).split()[: rng.randint(2, 7)])
        font = _pick_font(picker, title, core_only=False)
        if font is not None:
            tsize = int(size * rng.uniform(1.3, 2.2))
            comp.blocks.append(Block(margin, y, W - margin, y))
            tw = advance_width(title, font.path, tsize)
            right = W - margin if rng.random() < 0.5 else int((W + tw) / 2)  # right or centred
            h = comp.place(title, font.path, tsize, min(W - margin, right), y)
            if h:
                comp.blocks[-1].y1 = y + h
                y += int(h * rng.uniform(1.4, 2.2))

    gutter = int(W * rng.uniform(0.03, 0.06))
    col_w = (W - 2 * margin - (columns - 1) * gutter) // columns
    # Real layouts fit at least ~6-8 words per line; cap the size to the column.
    size = min(size, max(cfg.body_size[0], col_w // 14))
    pitch = int(size * rng.uniform(*cfg.line_spacing))
    footer_h = int(H * rng.uniform(0.08, 0.2)) if rng.random() < 0.3 else 0
    bottom = H - int(H * 0.06) - footer_h
    for c in range(columns):  # c = 0 is the rightmost column (read first)
        right = W - margin - c * (col_w + gutter)
        cursor = y
        while cursor + pitch < bottom:
            text = _text(texts, rng, strip, rng.randint(15, 80))
            font = _pick_font(picker, text)
            if font is None:
                break
            block = Block(right - col_w, cursor, right, cursor)
            comp.blocks.append(block)
            justify = rng.random() < 0.7
            block.y1 = comp.paragraph(
                text, font, size, block.x0, right, cursor, pitch, justify, max_y=bottom
            )
            cursor = block.y1 + int(pitch * rng.uniform(0.2, 1.0))  # paragraph gap

    if footer_h:  # footnotes: short rule, then small text
        fy = bottom + int(footer_h * 0.15)
        comp.ink[fy : fy + 2, W - margin - W // 4 : W - margin] = 255
        fsize = max(12, int(size * 0.75))
        text = _text(texts, rng, strip, 30)
        font = _pick_font(picker, text)
        if font is not None:
            block = Block(margin, fy + int(fsize * 0.8), W - margin, fy + int(fsize * 0.8))
            comp.blocks.append(block)
            block.y1 = comp.paragraph(
                text, font, fsize, margin, W - margin, block.y0, int(fsize * 1.4)
            )

    if rng.random() < 0.6:  # page number
        num = str(rng.randint(1, 999))
        if rng.random() < 0.6:
            num = charset.convert_digits(num, "arabic")
        font = _pick_font(picker, num)
        if font is not None:
            nw = advance_width(num, font.path, size)
            ny = H - int(H * 0.045)
            comp.blocks.append(Block(int((W - nw) / 2), ny, int((W + nw) / 2), ny + size))
            comp.place(num, font.path, size, int((W + nw) / 2), ny)


def compose_screen(comp, rng, texts, picker, cfg) -> None:
    """UI-like page: a column of blocks (titles, menu rows, paragraphs), each with
    its own background and text colour."""
    H, W = comp.ink.shape
    dark = rng.random() < 0.35
    y = int(rng.uniform(0.01, 0.05) * H)
    strip = rng.random() < 0.85  # UI text rarely has diacritics
    while y < H * 0.92:
        kind = rng.choice(["title", "menu", "para", "para"])
        size = {
            "title": rng.randint(22, 44),
            "menu": rng.randint(13, 22),
            "para": rng.randint(13, 24),
        }[kind]
        pad = int(size * rng.uniform(0.4, 1.2))
        x0 = int(W * rng.uniform(0.0, 0.08))
        x1 = W - int(W * rng.uniform(0.0, 0.08))
        bg_level = rng.uniform(0, 70) if dark else rng.uniform(200, 255)
        if rng.random() < 0.25:  # a coloured/tinted block
            bg_level = rng.uniform(20, 120) if rng.random() < 0.5 else rng.uniform(150, 235)
        fg = (
            rng.uniform(190, 255)
            if bg_level < 128
            else rng.uniform(0, max(0.0, min(90, bg_level - 100)))
        )
        block = Block(x0, y, x1, y + pad, bg=bg_level, fg=fg)
        comp.blocks.append(block)
        pitch = int(size * rng.uniform(1.3, 1.8))
        if kind == "menu":
            n = rng.randint(2, 6)
            cy = y + pad
            for _ in range(n):
                item = " ".join(_text(texts, rng, strip, 3).split()[: rng.randint(1, 4)])
                font = _pick_font(picker, item)
                if font is None or comp.place(item, font.path, size, x1 - pad, cy) is None:
                    break
                cy += pitch
            end = cy
        else:
            n_words = rng.randint(2, 8) if kind == "title" else rng.randint(10, 60)
            text = " ".join(_text(texts, rng, strip, n_words).split()[:n_words])
            font = _pick_font(picker, text, core_only=kind != "title")
            if font is None:
                break
            end = comp.paragraph(text, font, size, x0 + pad, x1 - pad, y + pad, pitch, False)
        block.y1 = min(H, end + pad)
        y = block.y1 + int(size * rng.uniform(0.2, 1.5))
    comp.lines = [ln for ln in comp.lines if ln.quad[2, 1] <= H]


# ---------------------------------------------------------------------------
# Appearance
# ---------------------------------------------------------------------------


def rotate_page(img: np.ndarray, lines: list[Line], deg: float, fill: float):
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=float(fill))
    for ln in lines:
        pts = np.hstack([ln.quad, np.ones((4, 1), np.float32)])
        ln.quad = (pts @ m.T).astype(np.float32)
    return out


def finish_print(comp: PageComposer, rng: np.random.Generator, cfg: PageConfig) -> np.ndarray:
    h, w = comp.ink.shape
    acfg = A.AugmentConfig()
    bg = A.paper_background(h, w, rng, acfg)
    bg_level = float(np.median(bg))
    text = float(rng.uniform(0, min(80, bg_level - 100)))
    alpha = comp.ink.astype(np.float32) / 255.0 * rng.uniform(0.8, 1.0)
    img = bg * (1 - alpha) + text * alpha
    if rng.random() < 0.3:
        img = A.shadow(img, rng)
    if rng.random() < 0.15 and comp.lines:
        # Measure the paper level over about two line heights, not the whole page.
        line_h = float(np.median([ln.quad[2, 1] - ln.quad[1, 1] for ln in comp.lines]))
        img = A.binarize(img, rng, bg_level, text, window=int(2 * line_h))
    if rng.random() < 0.4:
        img = A.gaussian_blur(img, rng.uniform(0.3, 0.9))
    if rng.random() < 0.5:
        img = A.gaussian_noise(img, rng, rng.uniform(1, 8))
    if rng.random() < cfg.rotate_prob:
        deg = rng.uniform(-cfg.max_rotate_deg, cfg.max_rotate_deg)
        img = rotate_page(img, comp.lines, deg, fill=bg_level)
    if rng.random() < 0.5:
        img = A.jpeg(img, int(rng.integers(40, 95)))
    return img.clip(0, 255).astype(np.uint8)


def finish_screen(comp: PageComposer, rng: np.random.Generator) -> np.ndarray:
    h, w = comp.ink.shape
    page_bg = 255.0 if comp.blocks and comp.blocks[0].bg > 128 else 20.0
    bg = np.full((h, w), page_bg, np.float32)
    fg = np.zeros((h, w), np.float32)
    for b in comp.blocks:
        bg[b.y0 : b.y1, b.x0 : b.x1] = b.bg
        fg[b.y0 : b.y1, b.x0 : b.x1] = b.fg
    alpha = comp.ink.astype(np.float32) / 255.0
    img = bg * (1 - alpha) + fg * alpha
    if rng.random() < 0.4:
        img = A.jpeg(img, int(rng.integers(50, 95)))
    return img.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def make_page(texts, picker: FontPicker, rng: random.Random, np_rng, cfg: PageConfig):
    """One synthetic page: (grayscale image, list of Line, kind)."""
    kinds, weights = zip(*cfg.kind_weights.items(), strict=True)
    kind = rng.choices(kinds, weights=weights)[0]
    if kind == "screen":
        w = rng.randint(700, 1400)
        h = int(w * rng.uniform(0.55, 1.9))  # landscape window to phone screen
    else:
        w = rng.randint(*cfg.page_width)
        h = int(w * rng.uniform(1.3, 1.45))  # roughly A4 / letter
    comp = PageComposer(w, h)
    if kind == "screen":
        compose_screen(comp, rng, texts, picker, cfg)
        img = finish_screen(comp, np_rng)
    else:
        compose_book(comp, rng, texts, picker, cfg, columns=2 if kind == "columns" else 1)
        img = finish_print(comp, np_rng, cfg)
    for i, ln in enumerate(comp.lines):
        ln.order = i
    return img, comp.lines, kind
