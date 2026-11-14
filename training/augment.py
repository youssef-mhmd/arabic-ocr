"""Turn a clean rendered "ink" image into a realistic, messy training image.

Why: a model trained only on perfect black-on-white text fails on real scans and
screenshots (the "domain gap"). By randomly degrading every training image, we force
the model to learn the letter shapes themselves, not the exact pixels of a clean
render.

Input:  an ink image from `training.render.render_line` (0 = no ink, 255 = full ink).
Output: a grayscale photo-like image (0 = black, 255 = white), uint8.

Every sample is either:
- **print**: paper texture, ink bleed/thinning, slight rotation, skew and perspective,
  uneven lighting, low scan resolution, blur, noise, JPEG; or
- **screen**: flat or gradient UI background (sometimes dark mode), UI rule lines,
  scaling blur, JPEG.

Rules that keep the label correct:
- Never flip or mirror (that would turn Arabic into nonsense with the same label).
- Never crop into the text; geometric changes enlarge the canvas instead.
- Always keep a minimum contrast between text and background.

Everything is driven by a `numpy.random.Generator`, so a fixed seed gives the same
images every time.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

#: Smallest height (pixels) the rendered text may be shrunk to by `low_resolution`.
MIN_TEXT_HEIGHT = 20


@dataclass
class AugmentConfig:
    #: share of samples that look like screenshots (the rest look like print/scans)
    screen_prob: float = 0.35

    # print-mode geometry and ink
    ink_change_prob: float = 0.3  # thicker (bleed) or thinner strokes
    rotate_prob: float = 0.4
    max_rotate_deg: float = 2.5
    shear_prob: float = 0.25
    max_shear: float = 0.15  # horizontal shift per pixel of height
    perspective_prob: float = 0.2
    max_perspective: float = 0.04  # corner shift as a fraction of image size

    # backgrounds and lighting
    paper_texture_prob: float = 0.7
    stain_prob: float = 0.15
    shadow_prob: float = 0.3
    dark_mode_prob: float = 0.25  # screen mode only
    ui_lines_prob: float = 0.2  # screen mode only
    min_contrast: int = 90  # gray levels between text and background (~3:1 or better)

    # image quality
    low_res_prob: float = 0.35
    min_low_res_scale: float = 0.35
    blur_prob: float = 0.4
    max_blur_sigma: float = 1.3
    motion_blur_prob: float = 0.05
    noise_prob: float = 0.5
    max_noise_sigma: float = 12.0
    jpeg_prob: float = 0.4
    min_jpeg_quality: int = 25
    #: at most this many lighting/quality effects on one image
    max_degradations: int = 3

    # margins around the text, as a fraction of the image height
    max_extra_margin: float = 0.25

    # --- added for v2 (all off by default, so v1 settings give v1 images) ---
    #: share of lines cropped close to the ink, as real line crops often are
    tight_crop_prob: float = 0.0
    tight_max_margin: float = 0.2
    #: share of lines with bits of the lines above/below poking in at the edges
    fragment_prob: float = 0.0
    #: print only: thresholded to two tones (black-and-white scans, photocopies)
    binarize_prob: float = 0.0
    #: print only: small holes in the strokes (worn type, faint print)
    broken_prob: float = 0.0
    #: print only: dust specks on the paper
    speckle_prob: float = 0.0
    #: stronger minimum contrast for very thin fonts (Thin/ExtraLight weights)
    thin_min_contrast: int = 90
    thin_max_degradations: int = 3


# ---------------------------------------------------------------------------
# Building blocks. Each takes and returns float32 arrays.
# ---------------------------------------------------------------------------


def change_ink(ink: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Make strokes thicker (ink bleed) or thinner (faded/light print)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    if rng.random() < 0.5:
        return cv2.dilate(ink, k)
    return cv2.erode(ink, k)


def _warp(ink: np.ndarray, matrix: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    if matrix.shape == (3, 3):
        return cv2.warpPerspective(ink, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR)
    return cv2.warpAffine(ink, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR)


def rotate(ink: np.ndarray, deg: float) -> np.ndarray:
    """Rotate around the center, growing the canvas so no ink is cut off."""
    h, w = ink.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w, new_h = int(np.ceil(w * cos + h * sin)), int(np.ceil(w * sin + h * cos))
    m[0, 2] += (new_w - w) / 2
    m[1, 2] += (new_h - h) / 2
    return _warp(ink, m, new_w, new_h)


def shear(ink: np.ndarray, amount: float) -> np.ndarray:
    """Slant the text (like italic or a skewed scan), growing the canvas to fit."""
    # x_new = x + amount * y: each row shifts sideways in proportion to its height.
    h, w = ink.shape
    extra = int(np.ceil(abs(amount) * h))
    offset = extra if amount < 0 else 0  # keep the most-shifted row inside the canvas
    m = np.float32([[1, amount, offset], [0, 1, 0]])
    return _warp(ink, m, w + extra, h)


def perspective(ink: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Move each corner a little (as if the page was photographed at an angle)."""
    h, w = ink.shape
    pad = np.float32([np.ceil(strength * w), np.ceil(strength * 2 * h)])
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # Each corner moves by up to `pad` in any direction; the canvas grows by 2 * pad
    # so every moved corner stays inside it.
    dst = src + pad + rng.uniform(-1, 1, size=(4, 2)).astype(np.float32) * pad
    m = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    return _warp(ink, m, w + 2 * int(pad[0]), h + 2 * int(pad[1]))


def add_margins(ink: np.ndarray, rng: np.random.Generator, max_frac: float) -> np.ndarray:
    """Random extra blank space on each side (real line crops are never exact)."""
    h = ink.shape[0]
    top, bottom, left, right = (int(rng.uniform(0, max_frac) * h) for _ in range(4))
    return cv2.copyMakeBorder(ink, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)


def crop_to_ink(ink: np.ndarray) -> np.ndarray:
    """Cut away the empty rows above and below the text (tight line crop)."""
    rows = np.where(ink.max(axis=1) > 32)[0]
    if len(rows) == 0:
        return ink
    return ink[rows[0] : rows[-1] + 1]


def add_fragments(ink: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bits of the neighbouring lines at the top and/or bottom edge.

    Real crops of a line often catch the descenders of the line above or the dots
    of the line below. We imitate them with a slice of this line's own ink, shifted
    sideways, so the strokes look like real text but aren't readable words.
    """
    h, w = ink.shape
    parts = [ink]
    sides = ["top", "bottom"] if rng.random() < 0.3 else [rng.choice(["top", "bottom"])]
    for side in sides:
        frac = rng.uniform(0.08, 0.3)
        n = max(1, int(frac * h))
        piece = ink[h - n :] if side == "top" else ink[:n]  # line above shows its bottom
        piece = np.roll(piece, int(rng.integers(0, w)), axis=1)
        gap = np.zeros((int(rng.uniform(0.02, 0.12) * h), w), ink.dtype)
        parts = [piece, gap, *parts] if side == "top" else [*parts, gap, piece]
    return np.vstack(parts)


def break_strokes(alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small holes in the ink (worn type). Kept sparse so dots never vanish entirely."""
    h, w = alpha.shape
    holes = smooth_noise(h, w, rng, scale=rng.uniform(1.5, 3)) > rng.uniform(0.75, 0.9)
    return np.where(holes, alpha * rng.uniform(0.0, 0.4), alpha)


def speckle(img: np.ndarray, rng: np.random.Generator, color: float) -> np.ndarray:
    """A few single-pixel dust specks, far fewer than the dots of the letters."""
    h, w = img.shape
    out = img.copy()
    n = int(h * w * rng.uniform(0.0002, 0.001))
    out[rng.integers(0, h, n), rng.integers(0, w, n)] = color
    return out


def binarize(
    img: np.ndarray, rng: np.random.Generator, bg: float, fg: float, window: int | None = None
) -> np.ndarray:
    """Two tones only, like a black-and-white scan: jagged edges, no gray.

    The threshold sits closer to the paper than to the ink (65-85% of the way from
    ink to paper), as on a scanner set to keep text: faded or slightly blurred dots
    still count as ink. A threshold halfway between erased whole dots in testing,
    which would make the image disagree with its label.

    The paper brightness is measured locally (like an adaptive scanner), so a
    shadow doesn't turn a whole region black. If thresholding would still lose (or
    invent) most of the text, the image is returned unchanged.

    `window`: size in pixels of the neighbourhood used for the paper level. The
    default (the image height) suits single line images; whole pages must pass
    about two line heights, or this is both slow and no longer local.
    """
    text_dark = fg < bg
    size = img.shape[0] if window is None else window
    k = max(3, size // 2 * 2 + 1)  # about half the line height
    kernel = np.ones((k, k), np.uint8)
    # Local paper level: the brightest (or darkest, in dark mode) nearby pixel.
    local = cv2.dilate(img, kernel) if text_dark else cv2.erode(img, kernel)
    local = cv2.GaussianBlur(local, (0, 0), k / 3)
    level = fg + (local - fg) * rng.uniform(0.65, 0.85)
    is_text = img < level if text_dark else img > level
    reference = img < (fg + local) / 2 if text_dark else img > (fg + local) / 2
    n, n_ref = int(is_text.sum()), int(reference.sum())
    if n_ref == 0 or n < 0.7 * n_ref or n > 3 * n_ref:
        return img
    return np.where(is_text, fg, bg).astype(np.float32)


def smooth_noise(h: int, w: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    """Soft, cloudy random pattern in [-1, 1] (for paper fibres and lighting)."""
    sh, sw = max(2, int(h / scale)), max(2, int(w / scale))
    small = rng.uniform(-1, 1, size=(sh, sw)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC).clip(-1, 1)


def paper_background(h: int, w: int, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    level = rng.uniform(185, 255)
    bg = np.full((h, w), level, np.float32)
    if rng.random() < cfg.paper_texture_prob:
        bg += smooth_noise(h, w, rng, scale=rng.uniform(2, 6)) * rng.uniform(2, 10)
        bg += smooth_noise(h, w, rng, scale=rng.uniform(20, 60)) * rng.uniform(3, 15)
    if rng.random() < cfg.stain_prob:
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        r = rng.uniform(0.3, 1.5) * h
        yy, xx = np.mgrid[0:h, 0:w]
        blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)))
        bg -= blob.astype(np.float32) * rng.uniform(15, 50)
    return bg


def screen_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    level = rng.uniform(0, 255)
    bg = np.full((h, w), level, np.float32)
    if rng.random() < 0.3:  # gentle gradient, as in many UI themes
        ramp = np.linspace(-1, 1, w if rng.random() < 0.5 else h, dtype=np.float32)
        ramp = ramp[None, :] if ramp.size == w else ramp[:, None]
        bg = bg + ramp * rng.uniform(5, 30)
    return bg


def add_ui_lines(img: np.ndarray, rng: np.random.Generator, color: float) -> np.ndarray:
    """A horizontal rule or box border near the top/bottom edge (table cells, inputs)."""
    h = img.shape[0]
    out = img.copy()
    for edge in ("top", "bottom"):
        if rng.random() < 0.5:
            y = (
                int(rng.uniform(0, 0.08) * h)
                if edge == "top"
                else h - 1 - int(rng.uniform(0, 0.08) * h)
            )
            thickness = int(rng.integers(1, 3))
            out[max(0, y - thickness + 1) : y + 1, :] = color
    return out


def shadow(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uneven lighting: darken part of the image smoothly."""
    h, w = img.shape
    light = 1 - (smooth_noise(h, w, rng, scale=max(h, w) / 2) + 1) / 2 * rng.uniform(0.1, 0.45)
    return img * light


def low_resolution(
    img: np.ndarray, rng: np.random.Generator, min_scale: float, text_height: int | None = None
) -> np.ndarray:
    """Shrink then enlarge back: loses detail like a low-DPI scan or zoomed screenshot.

    If `text_height` is given, never shrink the text below MIN_TEXT_HEIGHT pixels:
    below that, diacritics and dots vanish and the image no longer matches its label.
    """
    h, w = img.shape
    if text_height:
        min_scale = max(min_scale, MIN_TEXT_HEIGHT / text_height)
    if min_scale >= 0.85:
        return img
    s = rng.uniform(min_scale, 0.85)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    interp = cv2.INTER_NEAREST if rng.random() < 0.2 else cv2.INTER_LINEAR
    return cv2.resize(small, (w, h), interpolation=interp)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma)


def motion_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(3, 6))
    kernel = np.zeros((size, size), np.float32)
    kernel[size // 2, :] = 1 / size  # horizontal shake
    angle = rng.uniform(-30, 30)
    rot = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (size, size))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def gaussian_noise(img: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    return img + rng.normal(0, sigma, size=img.shape).astype(np.float32)


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(
        ".jpg", img.clip(0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def _pick_colors(
    bg_level: float, rng: np.random.Generator, min_contrast: int, dark_mode: bool
) -> float:
    """A text gray level at least `min_contrast` away from the background."""
    if dark_mode:  # light text on a dark background
        low = min(255.0, bg_level + min_contrast)
        return float(rng.uniform(low, 255)) if low < 255 else 255.0
    high = max(0.0, bg_level - min_contrast)
    return float(rng.uniform(0, min(high, 90))) if high > 0 else 0.0


def _chance(rng: np.random.Generator, p: float) -> bool:
    """True with probability p. Uses no random number when p is 0, so adding a new,
    disabled option never changes the images an existing config makes."""
    return p > 0 and rng.random() < p


def augment(
    ink: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig | None = None, thin: bool = False
) -> np.ndarray:
    """Turn an ink image into a realistic grayscale line image (uint8, white = 255).

    `thin`: the font is a very thin weight; use a stronger minimum contrast and fewer
    degradations, so the text stays readable (and matches its label).
    """
    cfg = cfg or AugmentConfig()
    min_contrast = max(cfg.min_contrast, cfg.thin_min_contrast) if thin else cfg.min_contrast
    max_degradations = (
        min(cfg.max_degradations, cfg.thin_max_degradations) if thin else cfg.max_degradations
    )
    a = ink.astype(np.float32)
    screen = rng.random() < cfg.screen_prob

    # 1. Shape of the ink (print only; screen text is crisp and straight).
    if not screen:
        if rng.random() < cfg.ink_change_prob:
            a = change_ink(a, rng)
        if rng.random() < cfg.shear_prob:
            a = shear(a, rng.uniform(-cfg.max_shear, cfg.max_shear))
        if rng.random() < cfg.rotate_prob:
            a = rotate(a, rng.uniform(-cfg.max_rotate_deg, cfg.max_rotate_deg))
        if rng.random() < cfg.perspective_prob:
            a = perspective(a, rng, rng.uniform(0, cfg.max_perspective))
    text_height = ink.shape[0]  # the rendered line before extra margins
    if _chance(rng, cfg.tight_crop_prob):
        a = crop_to_ink(a)
        text_height = a.shape[0]
        a = add_margins(a, rng, cfg.tight_max_margin)
    else:
        a = add_margins(a, rng, cfg.max_extra_margin)
    if _chance(rng, cfg.fragment_prob):
        a = add_fragments(a, rng)
    h, w = a.shape

    # 2. Background and text color.
    if screen:
        dark = rng.random() < cfg.dark_mode_prob
        bg = screen_background(h, w, rng)
        if dark:
            bg = np.minimum(bg, rng.uniform(0, 90))
        else:
            bg = np.maximum(bg, rng.uniform(150, 255))
    else:
        dark = False
        bg = paper_background(h, w, rng, cfg)
    bg_level = float(np.median(bg))
    text = _pick_colors(bg_level, rng, min_contrast, dark)

    # 3. Put the ink on the background. Ink opacity varies a little (faded print).
    alpha = (a / 255.0) * (rng.uniform(0.75, 1.0) if not screen else 1.0)
    if not screen and _chance(rng, cfg.broken_prob):
        alpha = break_strokes(alpha, rng)
    img = bg * (1 - alpha) + text * alpha
    if not screen and _chance(rng, cfg.speckle_prob):
        img = speckle(img, rng, text)

    if screen and rng.random() < cfg.ui_lines_prob:
        img = add_ui_lines(img, rng, color=text if rng.random() < 0.5 else bg_level * 0.6)

    # 4. Lighting and image quality. Each effect is realistic alone, but stacking all
    #    of them on small text erases dots and diacritics, so the image would no
    #    longer match its label. Pick them by probability, then keep at most
    #    `max_degradations`, in a fixed order (lighting -> resolution -> blur -> noise -> JPEG).
    chosen = [
        name
        for name, p in [
            ("shadow", 0.0 if screen else cfg.shadow_prob),
            ("low_res", cfg.low_res_prob),
            ("blur", cfg.blur_prob),
            ("binarize", 0.0 if screen else cfg.binarize_prob),
            ("motion", 0.0 if screen else cfg.motion_blur_prob),
            ("noise", cfg.noise_prob * (0.4 if screen else 1.0)),
            ("jpeg", cfg.jpeg_prob),
        ]
        # v1 effects always draw a random number (even at p = 0), as before; only
        # the new "binarize" skips it when disabled. Keeps v1 images reproducible.
        if (_chance(rng, p) if name == "binarize" else rng.random() < p)
    ]
    if len(chosen) > max_degradations:
        keep = set(rng.choice(chosen, size=max_degradations, replace=False).tolist())
        chosen = [c for c in chosen if c in keep]

    # Blur strength scales with text size: sigma 1.3 is mild on 60 px text but
    # wipes out the dots of 20 px text.
    max_sigma = min(cfg.max_blur_sigma if not screen else 0.7, max(0.4, text_height / 40))
    for step in chosen:
        if step == "shadow":
            img = shadow(img, rng)
        elif step == "low_res":
            img = low_resolution(img, rng, cfg.min_low_res_scale, text_height)
        elif step == "blur":
            img = gaussian_blur(img, rng.uniform(0.3, max(0.3, max_sigma)))
        elif step == "binarize":
            img = binarize(img, rng, bg_level, text)
        elif step == "motion":
            img = motion_blur(img, rng)
        elif step == "noise":
            img = gaussian_noise(img, rng, rng.uniform(1, cfg.max_noise_sigma))
        elif step == "jpeg":
            img = jpeg(img, int(rng.integers(cfg.min_jpeg_quality, 95)))

    return img.clip(0, 255).astype(np.uint8)
