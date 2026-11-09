"""Tests for the augmentation pipeline (no fonts needed: uses a synthetic ink image)."""

import numpy as np
import pytest

from training import augment as A


def fake_ink(h=48, w=300) -> np.ndarray:
    """A few solid 'strokes' standing in for rendered text."""
    ink = np.zeros((h, w), np.uint8)
    ink[15:35, 20:60] = 255
    ink[20:30, 100:250] = 255
    ink[5:12, 150:160] = 255  # a 'diacritic' near the top
    return ink


def ink_mass(a: np.ndarray) -> float:
    return float(a.astype(np.float64).sum())


@pytest.mark.parametrize("seed", range(40))
def test_output_is_valid_grayscale(seed):
    out = A.augment(fake_ink(), np.random.default_rng(seed))
    assert out.dtype == np.uint8 and out.ndim == 2
    assert out.shape[0] >= 48 and out.shape[1] >= 300  # never smaller than the input


@pytest.mark.parametrize("seed", range(40))
def test_text_stays_visible(seed):
    ink = fake_ink()
    out = A.augment(ink, np.random.default_rng(seed)).astype(np.float32)
    # Text and background must still differ clearly: compare the darkest/lightest
    # parts of the image (the text is one of the two extremes).
    lo, hi = np.percentile(out, [5, 95])
    assert hi - lo > 40


def test_same_seed_same_image():
    a = A.augment(fake_ink(), np.random.default_rng(3))
    b = A.augment(fake_ink(), np.random.default_rng(3))
    c = A.augment(fake_ink(), np.random.default_rng(4))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("deg", [-3, -1, 1, 3])
def test_rotate_keeps_all_ink(deg):
    ink = fake_ink().astype(np.float32)
    out = A.rotate(ink, deg)
    assert out.shape[0] > ink.shape[0] and out.shape[1] >= ink.shape[1]
    assert ink_mass(out) == pytest.approx(ink_mass(ink), rel=0.03)


@pytest.mark.parametrize("amount", [-0.2, 0.2])
def test_shear_keeps_all_ink(amount):
    ink = fake_ink().astype(np.float32)
    out = A.shear(ink, amount)
    assert ink_mass(out) == pytest.approx(ink_mass(ink), rel=0.03)
    assert out[:, 0].max() == 0 or out[:, -1].max() == 0  # nothing touches both edges


def test_perspective_keeps_ink_inside():
    ink = fake_ink().astype(np.float32)
    for seed in range(10):
        out = A.perspective(ink, np.random.default_rng(seed), 0.04)
        assert ink_mass(out) > 0.85 * ink_mass(ink)
        assert out[0].max() == 0 and out[-1].max() == 0


def test_shear_direction_is_not_a_mirror():
    # A blob at the left must stay on the left (no flipping, ever).
    ink = np.zeros((40, 200), np.float32)
    ink[10:30, 5:25] = 255
    for out in (A.shear(ink, 0.15), A.rotate(ink, 2.0)):
        cols = np.nonzero(out.max(axis=0))[0]
        assert cols.mean() < out.shape[1] / 2


def test_margins_only_add_space():
    ink = fake_ink()
    out = A.add_margins(ink, np.random.default_rng(0), 0.25)
    assert out.shape[0] >= ink.shape[0] and out.shape[1] >= ink.shape[1]
    assert ink_mass(out) == ink_mass(ink)


def test_jpeg_and_low_res_keep_size():
    img = np.full((50, 200), 200, np.float32)
    img[20:30, 50:150] = 20
    rng = np.random.default_rng(0)
    assert A.jpeg(img, 30).shape == img.shape
    assert A.low_resolution(img, rng, 0.3).shape == img.shape
    assert A.motion_blur(img, rng).shape == img.shape


def test_degradation_budget_limits_stacked_effects(monkeypatch):
    calls = []
    for name in (
        "shadow",
        "low_resolution",
        "gaussian_blur",
        "motion_blur",
        "gaussian_noise",
        "jpeg",
    ):
        orig = getattr(A, name)
        monkeypatch.setattr(
            A, name, lambda *a, _o=orig, _n=name, **k: calls.append(_n) or _o(*a, **k)
        )
    cfg = A.AugmentConfig(
        screen_prob=0,
        shadow_prob=1,
        low_res_prob=1,
        blur_prob=1,
        motion_blur_prob=1,
        noise_prob=1,
        jpeg_prob=1,
        max_degradations=3,
    )
    for seed in range(10):
        calls.clear()
        A.augment(fake_ink(), np.random.default_rng(seed), cfg)
        assert len(calls) == 3


def test_low_resolution_never_shrinks_text_below_minimum():
    # Text 30 px tall may shrink to at most 20/30 of its size; a 20 px line not at all.
    img = np.random.default_rng(0).uniform(0, 255, (20, 200)).astype(np.float32)
    rng = np.random.default_rng(0)
    assert np.array_equal(A.low_resolution(img, rng, 0.3, text_height=20), img)


def test_dark_mode_text_is_lighter_than_background():
    cfg = A.AugmentConfig(
        screen_prob=1.0,
        dark_mode_prob=1.0,
        noise_prob=0,
        jpeg_prob=0,
        blur_prob=0,
        low_res_prob=0,
        ui_lines_prob=0,
        max_extra_margin=0,
    )
    ink = fake_ink()
    out = A.augment(ink, np.random.default_rng(0), cfg).astype(np.float32)
    assert out[ink == 255].mean() > out[ink == 0].mean() + 50


def test_print_text_is_darker_than_paper():
    cfg = A.AugmentConfig(
        screen_prob=0.0,
        noise_prob=0,
        jpeg_prob=0,
        blur_prob=0,
        low_res_prob=0,
        rotate_prob=0,
        shear_prob=0,
        perspective_prob=0,
        ink_change_prob=0,
        max_extra_margin=0,
    )
    ink = fake_ink()
    out = A.augment(ink, np.random.default_rng(0), cfg).astype(np.float32)
    assert out[ink == 255].mean() < out[ink == 0].mean() - 50
