"""Clean up a page image before detection: grayscale, deskew, denoise, binarize.

    from arabic_ocr.preprocess import preprocess
    clean = preprocess(page_image)          # grayscale + deskew (the defaults)
    print(clean.angle)                      # how much the page was rotated

Each step is small and optional:

- **Grayscale**: the models only look at brightness, not colour.
- **Deskew**: a scanned page is often rotated by a degree or two. The detector
  copes with small angles, but straight lines give cleaner crops. We find the
  angle at which the rows of text line up best (the "projection profile" is
  sharpest) and rotate the page back by that angle.
- **Denoise** (off by default): a gentle median filter that removes salt-and-pepper
  speckles from noisy scans. It can blur tiny diacritics, so only use it on bad scans.
- **Binarize** (off by default): turn every pixel black or white (Otsu's method).
  The models were trained on grayscale, so this rarely helps; it exists for very
  uneven or faded scans.

NumPy + OpenCV only: the runtime never imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from arabic_ocr.recognize.recognizer import to_gray

#: Deskew searches angles in [-MAX_SKEW, +MAX_SKEW] degrees.
MAX_SKEW = 10.0


@dataclass
class Preprocessed:
    image: np.ndarray  # uint8 (H, W) grayscale page, ready for the detector
    angle: float  # degrees the original was rotated by to straighten it (0 = untouched)
    matrix: np.ndarray  # 2x3 affine: original pixel (x, y) -> pixel in `image`


def _dark_text(gray: np.ndarray) -> np.ndarray:
    """A 0/1 mask of "ink" pixels, whatever the polarity (dark-on-light or light-on-dark)."""
    _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if mask.mean() > 0.5:  # more "ink" than background: the page is light text on dark
        mask = 1 - mask
    return mask


def _profile_sharpness(mask: np.ndarray, angle: float) -> float:
    """How well the text rows line up after rotating `mask` by `angle` degrees.

    Sum the ink in every pixel row. When the text is straight, rows are either full
    of ink or empty, so the row sums vary a lot; a tilted page smears them out.
    """
    h, w = mask.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST)
    rows = rotated.sum(axis=1).astype(np.float64)
    return float(np.square(np.diff(rows)).sum())


def estimate_skew(gray: np.ndarray, max_angle: float = MAX_SKEW) -> float:
    """Angle (degrees) to rotate the page by so its text lines become horizontal.

    Coarse search every 1 degree, then a fine search every 0.1 degree around the
    best one. Works on a shrunken copy (at most 1000 px) for speed.
    """
    scale = min(1.0, 1000 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = _dark_text(small)
    if mask.sum() < 50:  # (nearly) empty page
        return 0.0

    def best(angles: np.ndarray) -> float:
        return float(max(angles, key=lambda a: _profile_sharpness(mask, a)))

    coarse = best(np.arange(-max_angle, max_angle + 0.5, 1.0))
    fine = best(np.arange(coarse - 1.0, coarse + 1.05, 0.1))
    return round(fine, 1)


def rotate(gray: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate around the centre by `angle` degrees (counter-clockwise), growing the
    canvas so no corner is cut off. New areas take the page's border colour.

    Returns the rotated image and the 2x3 matrix that was applied."""
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos + 0.5), int(h * cos + w * sin + 0.5)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    border = int(np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])))
    out = cv2.warpAffine(
        gray,
        m,
        (nw, nh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    return out, m


def binarize(gray: np.ndarray) -> np.ndarray:
    """Black text on white with Otsu's threshold (light-on-dark pages are inverted)."""
    mask = _dark_text(gray)
    return np.where(mask > 0, 0, 255).astype(np.uint8)


def preprocess(
    image: np.ndarray,
    deskew: bool = True,
    denoise: bool = False,
    binarize_page: bool = False,
) -> Preprocessed:
    """Grayscale, then optionally deskew, denoise and binarize a page image."""
    gray = to_gray(image)
    angle = 0.0
    matrix = np.float64([[1, 0, 0], [0, 1, 0]])
    if deskew:
        angle = estimate_skew(gray)
        if abs(angle) >= 0.2:  # smaller tilts are harmless; rotating blurs a little
            gray, matrix = rotate(gray, angle)
        else:
            angle = 0.0
    if denoise:
        gray = cv2.medianBlur(gray, 3)
    if binarize_page:
        gray = binarize(gray)
    return Preprocessed(gray, angle, matrix)
