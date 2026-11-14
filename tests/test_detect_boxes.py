"""Tests for line boxes <-> core maps (no model needed)."""

import cv2
import numpy as np
import pytest

from arabic_ocr.detect.boxes import (
    KERNEL_FRACTION,
    RotRect,
    box_to_core,
    map_to_boxes,
    quad_iou,
    quad_to_rect,
)


def test_quad_round_trip():
    r = RotRect(200, 100, 300, 40, 5.0)
    back = quad_to_rect(r.quad())
    assert back.w == pytest.approx(300, abs=1e-3)
    assert back.h == pytest.approx(40, abs=1e-3)
    assert back.angle == pytest.approx(5.0, abs=1e-3)


def test_quad_corner_order_is_tr_tl_bl_br():
    tr, tl, bl, br = RotRect(100, 50, 100, 20, 0).quad()
    assert tr[0] > tl[0] and tr[1] < br[1] and bl[0] < br[0]


def test_core_is_a_thin_band():
    quad = RotRect(300, 100, 400, 50, 0).quad()
    core = quad_to_rect(box_to_core(quad))
    assert core.h == pytest.approx(50 * KERNEL_FRACTION)
    assert core.w == pytest.approx(400 - 50 * (1 - KERNEL_FRACTION))


@pytest.mark.parametrize("angle", [-8.0, 0.0, 3.0])
def test_map_to_boxes_recovers_the_line(angle):
    """Paint the cores of three lines, then get the full boxes back."""
    truth = [RotRect(400, y, 600, 36, angle).quad() for y in (100, 160, 220)]
    prob = np.zeros((320, 800), np.float32)
    for q in truth:
        cv2.fillPoly(prob, [np.round(box_to_core(q)).astype(np.int32)], 1.0)
    found = map_to_boxes(prob)
    assert len(found) == 3
    for q in truth:
        assert max(quad_iou(q, d.quad) for d in found) > 0.85


def test_close_lines_stay_separate():
    """Boxes that overlap a little (touching diacritics) still give two lines."""
    a, b = RotRect(300, 100, 400, 40, 0).quad(), RotRect(300, 132, 400, 40, 0).quad()
    prob = np.zeros((250, 600), np.float32)
    for q in (a, b):
        cv2.fillPoly(prob, [np.round(box_to_core(q)).astype(np.int32)], 1.0)
    assert len(map_to_boxes(prob)) == 2


def test_weak_regions_are_dropped():
    prob = np.zeros((100, 300), np.float32)
    prob[40:50, 20:200] = 0.35  # above the pixel threshold, but a low average score
    assert map_to_boxes(prob) == []


def test_quad_iou():
    a = RotRect(100, 100, 100, 20, 0).quad()
    assert quad_iou(a, a) == pytest.approx(1.0)
    assert quad_iou(a, RotRect(400, 400, 100, 20, 0).quad()) == 0.0
    half = RotRect(150, 100, 100, 20, 0).quad()  # shifted by half its width
    assert quad_iou(a, half) == pytest.approx(1 / 3, abs=0.02)
