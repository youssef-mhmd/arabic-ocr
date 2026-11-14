"""Tests for the synthetic page generator and the detector's training targets."""

import random
from pathlib import Path

import numpy as np
import pytest

from arabic_ocr.detect.boxes import quad_to_rect
from training.detector_data import MIN_LINE_HEIGHT, targets_for
from training.render import advance_width

ROOT = Path(__file__).resolve().parent.parent
NASKH = ROOT / "data" / "fonts" / "notonaskharabic" / "NotoNaskhArabic[wght].ttf"
CORPORA = ROOT / "data" / "corpora" / "wikipedia_ar" / "train.txt"
needs_fonts = pytest.mark.skipif(not NASKH.exists(), reason="training fonts not downloaded")
needs_corpora = pytest.mark.skipif(not CORPORA.exists(), reason="corpora not prepared")

WORDS = "قال الشيخ رحمه الله في كتابه المشهور عن أخبار البصرة والكوفة وما جرى فيهما".split()


@needs_fonts
def test_break_lines_fits_the_width():
    from training.page import break_lines

    lines = break_lines(WORDS * 3, NASKH, 30, 300)
    assert len(lines) > 1
    assert " ".join(lines).split() == WORDS * 3  # no word lost or reordered
    for ln in lines:
        assert advance_width(ln, NASKH, 30) <= 300 or " " not in ln


@needs_fonts
def test_justify_scale_fills_the_width():
    from training.page import justify_scale

    line = " ".join(WORDS[:5])
    s = justify_scale(line, NASKH, 30, advance_width(line, NASKH, 30) + 40)
    assert s > 1
    assert advance_width(line, NASKH, 30, s) == pytest.approx(
        advance_width(line, NASKH, 30) + 40, abs=1
    )


@needs_fonts
@needs_corpora
@pytest.mark.parametrize("seed", range(6))
def test_pages_have_lines_inside_the_image(seed):
    from training.corpora import CORPORA_DIR
    from training.fonts import FontPicker
    from training.make_dataset import TextPicker, weighted_fonts
    from training.page import PageConfig, make_page

    texts = TextPicker([{"name": "wikipedia_ar", "weight": 1.0}], "train", CORPORA_DIR)
    rng = random.Random(seed)
    img, lines, kind = make_page(
        texts, FontPicker(weighted_fonts({}), rng), rng, np.random.default_rng(seed), PageConfig()
    )
    assert img.dtype == np.uint8 and img.ndim == 2
    assert kind in {"book", "columns", "screen"}
    assert len(lines) >= 1
    h, w = img.shape
    for ln in lines:
        assert ln.text.strip()
        assert ln.quad.shape == (4, 2)
        assert (ln.quad[:, 0] > -5).all() and (ln.quad[:, 0] < w + 5).all()
        assert (ln.quad[:, 1] > -5).all() and (ln.quad[:, 1] < h + 5).all()
    assert [ln.order for ln in lines] == list(range(len(lines)))


def test_targets_mark_cores_and_ignore_tiny_lines():
    big = np.float32([[300, 100], [100, 100], [100, 140], [300, 140]])
    tiny = np.float32([[300, 200], [250, 200], [250, 203], [300, 203]])
    gt, mask = targets_for([big, tiny], 320, 400)
    assert gt[120, 200] == 1  # middle of the big line: core
    assert gt[102, 200] == 0  # near its top edge: outside the core
    assert mask[201, 270] == 0  # the tiny line is ignored...
    assert gt[201, 270] == 0  # ...and not a target
    assert quad_to_rect(tiny).h < MIN_LINE_HEIGHT


@needs_fonts
@needs_corpora
@pytest.mark.parametrize("seed", range(8))
def test_every_drawn_line_has_a_label(seed, monkeypatch):
    """Regression (pages_v1): lines that ran past the bottom were drawn but their
    labels deleted, so the detector learned to ignore real text."""
    from training import page as P
    from training.corpora import CORPORA_DIR
    from training.fonts import FontPicker
    from training.make_dataset import TextPicker, weighted_fonts

    drawn = []
    place = P.PageComposer.place

    def counting_place(self, *a, **kw):
        h = place(self, *a, **kw)
        drawn.append(h is not None)
        return h

    monkeypatch.setattr(P.PageComposer, "place", counting_place)
    texts = TextPicker([{"name": "wikipedia_ar", "weight": 1.0}], "train", CORPORA_DIR)
    rng = random.Random(seed)
    cfg = P.PageConfig(kind_weights={"book": 0.5, "columns": 0.5, "screen": 0.0})
    _img, lines, _kind = P.make_page(
        texts, FontPicker(weighted_fonts({}), rng), rng, np.random.default_rng(seed), cfg
    )
    assert sum(drawn) == len(lines)
