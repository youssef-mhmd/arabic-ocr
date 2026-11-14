"""Line boxes <-> "core" maps, shared by detector training and the shipped runtime.

The detector does not draw boxes directly. It paints a map with a value between 0
and 1 for every pixel: "how sure am I that this pixel is in the *core* of a text
line?". The core of a line is a thin band through its middle: the full box shrunk
to KERNEL_FRACTION (40%) of its height, and by the same distance at both ends.

Why a thin core instead of the whole box: lines on a page are close together, and
dots and diacritics of neighbouring lines often touch. Whole boxes would merge into
one blob; thin cores stay apart. After finding each core, we grow it back to the
full line size (`core_to_box`). This is the idea behind DBNet and similar
"segmentation" detectors.

A box is a 4-point polygon, float32 (4, 2), in the order top-right, top-left,
bottom-left, bottom-right (Arabic lines start on the right). Lines are assumed to
be roughly horizontal (within about +/-30 degrees); the page is deskewed first.

NumPy + OpenCV only: the runtime never imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

#: Height of a line's core as a fraction of the full line height.
KERNEL_FRACTION = 0.4


@dataclass
class RotRect:
    """A rotated rectangle: centre, width (along the text), height, angle (degrees,
    0 = horizontal, positive = clockwise in image coordinates)."""

    cx: float
    cy: float
    w: float
    h: float
    angle: float

    def quad(self) -> np.ndarray:
        """Corners: top-right, top-left, bottom-left, bottom-right."""
        a = np.deg2rad(self.angle)
        u = np.array([np.cos(a), np.sin(a)]) * self.w / 2  # along the text, to the right
        v = np.array([-np.sin(a), np.cos(a)]) * self.h / 2  # down
        c = np.array([self.cx, self.cy])
        return np.float32([c + u - v, c - u - v, c - u + v, c + u + v])


def quad_to_rect(quad: np.ndarray) -> RotRect:
    """The rotated rectangle described by a (TR, TL, BL, BR) quad."""
    tr, tl, bl, br = np.asarray(quad, np.float64)
    top = tr - tl
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    c = (tr + tl + bl + br) / 4
    return RotRect(c[0], c[1], w, h, float(np.degrees(np.arctan2(top[1], top[0]))))


def box_to_core(quad: np.ndarray) -> np.ndarray:
    """Shrink a line box to its core (used to draw training targets)."""
    r = quad_to_rect(quad)
    d = r.h * (1 - KERNEL_FRACTION) / 2  # how far each edge moves in
    w = max(r.w - 2 * d, min(r.w, r.h * KERNEL_FRACTION))  # keep very short words visible
    return RotRect(r.cx, r.cy, w, r.h * KERNEL_FRACTION, r.angle).quad()


def core_to_box(r: RotRect) -> RotRect:
    """Grow a detected core back to the full line box (inverse of box_to_core)."""
    h = r.h / KERNEL_FRACTION
    d = (h - r.h) / 2
    return RotRect(r.cx, r.cy, r.w + 2 * d, h, r.angle)


def _horizontal_rect(contour: np.ndarray) -> RotRect:
    """minAreaRect, expressed so that `w` is the side closest to horizontal."""
    (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
    # OpenCV may describe the same rectangle with the sides swapped and the angle
    # shifted by 90 degrees. Choose the description whose angle is nearest 0.
    while angle > 45:
        angle -= 90
        w, h = h, w
    while angle < -45:
        angle += 90
        w, h = h, w
    return RotRect(cx, cy, w, h, angle)


@dataclass
class Detection:
    quad: np.ndarray  # (4, 2) float32, TR TL BL BR, in map coordinates
    score: float  # mean core probability inside the region


def map_to_boxes(
    prob: np.ndarray,
    threshold: float = 0.3,
    min_score: float = 0.5,
    min_height: float = 4.0,
) -> list[Detection]:
    """Turn a core-probability map (H, W) into line boxes.

    1. pixels above `threshold` are "core";
    2. each connected region of core pixels is one line;
    3. its rotated rectangle is grown back to the full line size.
    Regions whose average probability is below `min_score`, or whose full height
    would be under `min_height` pixels, are dropped as noise.
    """
    mask = (prob > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        region = np.zeros_like(mask)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(region[y : y + h, x : x + w], [cnt - [x, y]], -1, 1, -1)
        score = float(prob[y : y + h, x : x + w][region[y : y + h, x : x + w] > 0].mean())
        if score < min_score:
            continue
        box = core_to_box(_horizontal_rect(cnt))
        if box.h < min_height:
            continue
        out.append(Detection(box.quad(), score))
    return out


def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two convex quads (0 = no overlap, 1 = identical)."""
    a32, b32 = np.float32(a), np.float32(b)
    inter, _ = cv2.intersectConvexConvex(a32, b32)
    union = cv2.contourArea(a32) + cv2.contourArea(b32) - inter
    return float(inter / union) if union > 0 else 0.0
