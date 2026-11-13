"""Tests for the v2 data options: spacing, kashida, quotes, tight crops, scan effects."""

from pathlib import Path

import numpy as np
import pytest

from arabic_ocr import charset
from training import augment as A
from training import make_dataset
from training.render import RIGHT_JOINING, render_line
from training.text import DUAL_JOINING, LineSampler, SamplerConfig

ROOT = Path(__file__).resolve().parent.parent
NASKH = ROOT / "data" / "fonts" / "notonaskharabic" / "NotoNaskhArabic[wght].ttf"
needs_fonts = pytest.mark.skipif(not NASKH.exists(), reason="training fonts not downloaded")

PARAGRAPH = "قال الشيخ رحمه الله في كتابه المشهور عن أخبار البصرة والكوفة وما جرى فيهما"


def fake_ink(h=48, w=300) -> np.ndarray:
    ink = np.zeros((h, w), np.uint8)
    ink[15:35, 20:60] = 255
    ink[20:30, 100:250] = 255
    return ink


# --- render ------------------------------------------------------------------


@needs_fonts
def test_letter_gap_widens_only_lines_with_non_joining_letters():
    plain = render_line("ذات عيون", NASKH, 40)
    gapped = render_line("ذات عيون", NASKH, 40, letter_gap=10)
    assert gapped.shape[1] > plain.shape[1]
    # بيت has no right-joining letter followed by another letter: nothing changes.
    np.testing.assert_array_equal(
        render_line("بيت", NASKH, 40), render_line("بيت", NASKH, 40, letter_gap=10)
    )


@needs_fonts
def test_space_scale_changes_only_word_spaces():
    assert (
        render_line("في البيت", NASKH, 40, space_scale=2).shape[1]
        > (render_line("في البيت", NASKH, 40).shape[1])
    )
    np.testing.assert_array_equal(
        render_line("البيت", NASKH, 40), render_line("البيت", NASKH, 40, space_scale=2)
    )


def test_joining_sets_do_not_overlap():
    assert not (RIGHT_JOINING & DUAL_JOINING)
    assert "ا" in RIGHT_JOINING and "ب" in DUAL_JOINING and "ة" not in DUAL_JOINING


# --- text ----------------------------------------------------------------------


def test_v2_sampler_lines_are_valid_labels():
    s = LineSampler(SamplerConfig(quote_prob=1, footnote_prob=1, kashida_prob=1), seed=1)
    lines = [ln for ln in (s.sample(PARAGRAPH) for _ in range(200)) if ln]
    assert lines
    for ln in lines:
        assert charset.normalize(ln) == ln
        charset.encode(ln)  # every character is in the charset
        assert len(ln) <= 60
    assert any(charset.TATWEEL in ln for ln in lines)
    assert any("«" in ln or "»" in ln for ln in lines)
    assert any("(" in ln for ln in lines)


def test_kashida_only_follows_joining_letters():
    s = LineSampler(SamplerConfig(), seed=2)
    for _ in range(50):
        out = s.stretch("البصرة والكوفة")
        for i, ch in enumerate(out):
            if ch == charset.TATWEEL:
                prev = out[i - 1]
                assert prev == charset.TATWEEL or prev in DUAL_JOINING
        assert out.replace(charset.TATWEEL, "") == "البصرة والكوفة"


def test_new_options_off_keep_old_lines():
    """Switching the new options off must give exactly the v1 lines."""
    a = LineSampler(SamplerConfig(), seed=5)
    b = LineSampler(SamplerConfig(quote_prob=0, footnote_prob=0, kashida_prob=0), seed=5)
    assert [a.sample(PARAGRAPH) for _ in range(30)] == [b.sample(PARAGRAPH) for _ in range(30)]


# --- augment ---------------------------------------------------------------------


def test_crop_to_ink_removes_empty_rows():
    ink = np.zeros((60, 50), np.uint8)
    ink[20:30, 10:40] = 255
    assert A.crop_to_ink(ink).shape == (10, 50)


@pytest.mark.parametrize("seed", range(20))
def test_tight_crops_make_text_fill_more_of_the_image(seed):
    ink = np.zeros((100, 300), np.uint8)
    ink[40:60, 20:280] = 255  # text in the middle fifth, like a tall font box
    cfg = A.AugmentConfig(tight_crop_prob=1.0, screen_prob=1.0, ui_lines_prob=0)
    out = A.augment(ink, np.random.default_rng(seed), cfg)
    assert out.shape[0] <= 20 * (1 + 2 * cfg.tight_max_margin) + 1


@pytest.mark.parametrize("seed", range(20))
def test_scan_effects_keep_text_visible(seed):
    cfg = A.AugmentConfig(
        screen_prob=0,
        binarize_prob=1,
        broken_prob=1,
        speckle_prob=1,
        fragment_prob=1,
        tight_crop_prob=1,
    )
    out = A.augment(fake_ink(), np.random.default_rng(seed), cfg).astype(float)
    assert out.max() - out.min() >= 60


def test_binarize_gives_two_tones():
    img = np.linspace(0, 255, 100, dtype=np.float32).reshape(1, 100).repeat(5, axis=0)
    out = A.binarize(img, np.random.default_rng(0), bg=255.0, fg=0.0)
    assert set(np.unique(out)) <= {0.0, 255.0}


def test_thin_fonts_get_stronger_contrast():
    cfg = A.AugmentConfig(thin_min_contrast=200, screen_prob=1, dark_mode_prob=0)
    for seed in range(20):
        out = A.augment(fake_ink(), np.random.default_rng(seed), cfg, thin=True).astype(float)
        assert out.max() - out.min() >= 150


# --- config ----------------------------------------------------------------------


def test_v2_config_loads():
    cfg = make_dataset.load_config(ROOT / "training" / "configs" / "data_v2.yaml")
    assert cfg["style_weights"]["nastaliq"] > 1
    assert cfg["render"]["letter_gap_prob"] > 0


# --- v3: ligatures -----------------------------------------------------------------


def test_ligature_labels_normalize_to_the_phrase():
    from training.text import LIGATURES, ligate

    for phrase, glyph in LIGATURES.items():
        assert charset.normalize(glyph) == phrase  # the label rule (NFKC) agrees
        assert ligate(f"النبي {phrase} قال") == f"النبي {glyph} قال"


def test_honorific_sampler_inserts_whole_phrase():
    from training.text import LIGATURES

    s = LineSampler(SamplerConfig(honorific_prob=1, max_len=60), seed=4)
    lines = [ln for ln in (s.sample(PARAGRAPH) for _ in range(100)) if ln]
    assert any(any(p in ln for p in LIGATURES) for ln in lines)
    for ln in lines:
        charset.encode(ln)


@needs_fonts
def test_ligature_glyph_renders_narrower_than_phrase():
    phrase = "صلى الله عليه وسلم"
    assert render_line("ﷺ", NASKH, 40).shape[1] < render_line(phrase, NASKH, 40).shape[1]


def test_v3_config_loads():
    cfg = make_dataset.load_config(ROOT / "training" / "configs" / "data_v3.yaml")
    assert cfg["render"]["ligature_prob"] > 0 and cfg["sampler"]["honorific_prob"] > 0
