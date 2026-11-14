"""The full OCR pipeline: page image -> structured text.

    from arabic_ocr.pipeline import OCR
    ocr = OCR()                          # loads the default models from models/
    page = ocr.read_image(image)         # numpy image (gray, BGR or BGRA)
    print(page.text)
    page = ocr.read_file("scan.png")

Steps:
1. **Preprocess** (preprocess.py): grayscale, straighten a tilted page.
2. **Detect** (detect/): find a box around every text line.
3. **Crop and rectify**: cut each line out of the full-resolution page, rotated
   so it is horizontal, with a small margin above and below (the recognizer was
   trained on lines with some white space around the text).
4. **Recognize** (recognize/): read each line with the CTC recognizer.
5. **Order** (layout.py): blocks and lines in reading order.
6. **Postprocess**: optional diacritics removal and digit conversion.

All boxes in the result are in the coordinates of the image you passed in.

ONNX Runtime only: the shipped runtime never imports torch.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from arabic_ocr import charset
from arabic_ocr.bidi import logical_order
from arabic_ocr.detect.boxes import RotRect, quad_to_rect
from arabic_ocr.detect.detector import LineDetector
from arabic_ocr.layout import reading_order
from arabic_ocr.preprocess import preprocess
from arabic_ocr.recognize.ctc import Decoded
from arabic_ocr.recognize.recognizer import LineRecognizer

#: Where models are looked for: $ARABIC_OCR_MODELS, else the repo's models/ folder.
_REPO_MODELS = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR = Path(os.environ.get("ARABIC_OCR_MODELS", _REPO_MODELS))
DETECTOR = "detector-v1.onnx"
RECOGNIZER = "recognizer-v3.onnx"

#: Margin added around a detected line before reading it, as a fraction of the
#: line's height: above/below, and at both ends. Detected boxes already hug the ink
#: (dots and diacritics included). Measured on 1,327 synthetic page lines with true
#: boxes, CER was 1.1% at PAD_Y 0.05 and rose to 1.6% / 2.9% at 0.15 / 0.35:
#: bigger margins pull in pieces of the neighbouring lines.
PAD_Y = 0.05
PAD_X = 0.2

Digits = Literal["keep", "western", "arabic"]


@dataclass
class Word:
    text: str
    confidence: float
    box: np.ndarray  # (4, 2) float32: top-left, top-right, bottom-right, bottom-left


@dataclass
class Line:
    text: str
    confidence: float
    box: np.ndarray  # (4, 2) float32: top-left, top-right, bottom-right, bottom-left
    words: list[Word] = field(default_factory=list)


@dataclass
class Block:
    lines: list[Line]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def box(self) -> np.ndarray:
        pts = np.concatenate([line.box for line in self.lines])
        (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
        return np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


@dataclass
class Page:
    width: int
    height: int
    angle: float  # deskew rotation applied, degrees
    blocks: list[Block]
    seconds: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)  # seconds per step

    @property
    def lines(self) -> list[Line]:
        return [line for block in self.blocks for line in block.lines]

    @property
    def text(self) -> str:
        """Blocks separated by an empty line, lines by a newline."""
        return "\n\n".join(block.text for block in self.blocks)


def load_image(path: str | Path) -> np.ndarray:
    """Read an image file (any path, including non-ASCII names on Windows)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    if image.dtype == np.uint16:  # 16-bit PNG/TIFF
        image = (image >> 8).astype(np.uint8)
    return image


def crop_line(gray: np.ndarray, rect: RotRect) -> tuple[np.ndarray, np.ndarray]:
    """Cut out a (possibly rotated) line, with margins, as a horizontal image.

    Returns the crop and the 2x3 matrix that maps crop pixels back to page pixels.
    """
    h = rect.h * (1 + 2 * PAD_Y)
    w = rect.w + 2 * PAD_X * rect.h
    big = RotRect(rect.cx, rect.cy, w, h, rect.angle)
    tr, tl, bl, _ = big.quad()
    cw, ch = max(1, round(w)), max(1, round(h))
    dst = np.float32([[cw, 0], [0, 0], [0, ch]])
    m = cv2.getAffineTransform(np.float32([tr, tl, bl]), dst)
    border = int(np.median(gray[:: max(1, gray.shape[0] // 64), :: max(1, gray.shape[1] // 64)]))
    crop = cv2.warpAffine(
        gray,
        m,
        (cw, ch),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    return crop, cv2.invertAffineTransform(m)


def _apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64)
    return (pts @ m[:, :2].T + m[:, 2]).astype(np.float32)


def _rect_pts(x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    """Corners of the rectangle x0..x1, y0..y1 in TL, TR, BR, BL order."""
    return np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def split_words(
    dec: Decoded, crop_w: int, crop_h: int, model_h: int, stride: int
) -> list[tuple[str, float, float, float]]:
    """Split a decoded line at its spaces into (text, confidence, x0, x1) in crop pixels.

    Each symbol's position comes from the slice where CTC emitted it (slice 0 = right
    edge). A word spans from the space before it to the space after it, so word boxes
    tile the line. Words come in reading order (right to left); each word's text is in
    logical order.
    """
    space = charset.encode(charset.SPACE)[0]
    px = stride * crop_h / model_h  # crop pixels per slice

    def x_of(frame: float) -> float:  # slice index -> x in the crop
        return float(np.clip(crop_w - frame * px, 0, crop_w))

    words = []
    start, edge = 0, 0.0  # first id of the current word; its right-edge slice
    ids = [*dec.ids, space]
    frames = [*dec.frames, crop_w / px]
    for k, i in enumerate(ids):
        if i != space:
            continue
        if k > start:
            text = logical_order(charset.decode(ids[start:k]))
            conf = float(np.mean(dec.probs[start:k])) if dec.probs else dec.confidence
            words.append((text, conf, x_of(frames[k]), x_of(edge)))
        start, edge = k + 1, frames[k]
    return words


def postprocess(text: str, diacritics: bool = True, digits: Digits = "keep") -> str:
    if not diacritics:
        text = charset.strip_diacritics(text)
    return charset.convert_digits(text, digits)


class OCR:
    """Loads the detector and recognizer once; then reads any number of pages."""

    def __init__(
        self,
        detector: str | Path | None = None,
        recognizer: str | Path | None = None,
        threads: int | None = None,
    ) -> None:
        det_path = Path(detector) if detector else MODELS_DIR / DETECTOR
        rec_path = Path(recognizer) if recognizer else MODELS_DIR / RECOGNIZER
        for p in (det_path, rec_path):
            if not p.is_file():
                raise FileNotFoundError(f"Model not found: {p}")
        self.detector = LineDetector(det_path, threads)
        self.recognizer = LineRecognizer(rec_path, threads)

    def read_file(self, path: str | Path, **kw) -> Page:
        return self.read_image(load_image(path), **kw)

    def read_image(
        self,
        image: np.ndarray,
        diacritics: bool = True,
        digits: Digits = "keep",
        deskew: bool = True,
        denoise: bool = False,
        binarize: bool = False,
    ) -> Page:
        clock = _Clock()
        prep = preprocess(image, deskew=deskew, denoise=denoise, binarize_page=binarize)
        gray = prep.image
        to_original = cv2.invertAffineTransform(prep.matrix)
        clock.lap("preprocess")

        rects = [quad_to_rect(d.quad) for d in self.detector.detect(gray)]
        clock.lap("detect")
        crops = [crop_line(gray, r) for r in rects]
        decoded = self.recognizer.recognize_batch([c for c, _ in crops])
        clock.lap("recognize")

        lines: list[Line | None] = []
        for (crop, m), dec in zip(crops, decoded, strict=True):
            text = postprocess(dec.text, diacritics, digits)
            if not text.strip():
                lines.append(None)
                continue
            m_full = _compose(to_original, m)  # crop pixels -> original image pixels
            ch, cw = crop.shape
            pad = PAD_Y * ch / (1 + 2 * PAD_Y)  # undo the vertical margin for output boxes
            words = [
                Word(
                    postprocess(t, diacritics, digits),
                    c,
                    _apply(m_full, _rect_pts(x0, x1, pad, ch - pad)),
                )
                for t, c, x0, x1 in split_words(
                    dec, cw, ch, self.recognizer.height, self.recognizer.stride
                )
            ]
            box = _apply(m_full, _rect_pts(0, cw, pad, ch - pad))
            lines.append(Line(text, dec.confidence, box, words))

        kept = [(i, ln) for i, ln in enumerate(lines) if ln is not None]
        aligned = []  # axis-aligned boxes in the straightened page, for reading order
        for i, _ in kept:
            q = rects[i].quad()
            aligned.append((q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()))
        blocks = [Block([kept[j][1] for j in blk]) for blk in reading_order(aligned)]
        clock.lap("layout")

        h, w = image.shape[:2]
        return Page(w, h, prep.angle, blocks, clock.total(), clock.laps)


class _Clock:
    def __init__(self) -> None:
        self.start = self.last = time.perf_counter()
        self.laps: dict[str, float] = {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.laps[name] = now - self.last
        self.last = now

    def total(self) -> float:
        return time.perf_counter() - self.start


def _compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """2x3 affine `outer` after `inner`."""
    a = np.vstack([outer, [0, 0, 1]])
    b = np.vstack([inner, [0, 0, 1]])
    return (a @ b)[:2]
