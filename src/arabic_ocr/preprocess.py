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


#: Straight lines at least this long are rules or borders, not letters:
#: vertical ones as a share of the page height, horizontal ones of the page width.
RULE_V = 0.10
RULE_H = 0.25
#: ...and at least this many typical letter heights long.
RULE_LETTERS = 4
#: Only vertical rules are removed by default (see remove_rules).
VERTICAL_ONLY = True


def remove_rules(gray: np.ndarray) -> np.ndarray:
    """Paint long straight lines (page frames, column rules, table borders,
    separators) with the paper colour.

    Text lines are cut out with a small margin, and a frame line next to the text
    was read as an extra letter (a vertical rule looks like alef). A morphological
    "opening" with a long, 1-pixel-wide kernel keeps only ink that forms a straight
    run at least that long; no letter is anywhere near 10% of a page tall.

    Measured (CER, no diacritics): real Urdu pages in a printed frame 41.4% ->
    35.6%; Arabic printed pages, screenshots and synthetic pages unchanged. Removing
    horizontal rules too cost 0.06-0.07 points on screenshots and synthetic pages
    (and screenshot recall), so only vertical rules are removed (VERTICAL_ONLY).
    """
    h, w = gray.shape
    # Ink = darker than its *surroundings* (adaptive threshold), so shadows and
    # uneven lighting don't count; a global threshold marked shaded areas as ink
    # and erased the text in them. Light-on-dark pages are inverted first.
    g = 255 - gray if gray.mean() < 128 else gray
    block = max(15, int(0.02 * max(h, w)) | 1)
    mask = cv2.adaptiveThreshold(
        g, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 15
    )
    # A rule is long AND thin. Coloured panels and header bars on screenshots are
    # long too, but thick: remove the parts that are also this thick.
    thick = max(6, int(0.004 * max(h, w)))
    # ...and much taller than the letters: on a small image (one line of text) 10%
    # of the height is less than a tall alef. Typical letter height = median height
    # of the ink blobs.
    _n, _labels, stats, _c = cv2.connectedComponentsWithStats(mask, connectivity=8)
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    letter = float(np.median(heights[heights >= 3])) if (heights >= 3).any() else 10.0
    vk = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(25, int(h * RULE_V), int(RULE_LETTERS * letter)))
    )
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, int(w * RULE_H)), 1))
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vk)
    vert &= ~cv2.dilate(cv2.morphologyEx(vert, cv2.MORPH_OPEN, np.ones((1, thick), np.uint8)),
                        np.ones((3, 3), np.uint8))  # fmt: skip
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hk)
    horiz &= ~cv2.dilate(cv2.morphologyEx(horiz, cv2.MORPH_OPEN, np.ones((thick, 1), np.uint8)),
                         np.ones((3, 3), np.uint8))  # fmt: skip
    rules = vert if VERTICAL_ONLY else vert | horiz
    if not rules.any():
        return gray
    rules = cv2.dilate(rules, np.ones((3, 3), np.uint8))  # the rule's soft edges too
    paper = int(np.median(gray[mask == 0])) if (mask == 0).any() else 255
    out = gray.copy()
    out[rules > 0] = paper
    return out


def preprocess(
    image: np.ndarray,
    deskew: bool = True,
    denoise: bool = False,
    binarize_page: bool = False,
    rules: bool = True,
) -> Preprocessed:
    """Grayscale, then deskew, optionally denoise, remove rules/borders, and
    optionally binarize a page image."""
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
    if rules:
        gray = remove_rules(gray)
    if binarize_page:
        gray = binarize(gray)
    return Preprocessed(gray, angle, matrix)
