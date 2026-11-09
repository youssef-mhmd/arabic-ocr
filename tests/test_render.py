"""Tests for bidi run splitting, line rendering, and the font manifest."""

import json
from pathlib import Path

import numpy as np
import pytest

from training.render import MissingGlyphError, font_supports, ltr_mask, render_line, split_runs

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "data" / "fonts"
NASKH = FONTS / "notonaskharabic" / "NotoNaskhArabic[wght].ttf"
VAZIR = FONTS / "vazirmatn" / "Vazirmatn[wght].ttf"

needs_fonts = pytest.mark.skipif(
    not (NASKH.exists() and VAZIR.exists()),
    reason="run: uv run python -m training.download_fonts --only notonaskharabic,vazirmatn",
)


def visual(text: str) -> str:
    """Characters in display order, left to right (ignoring glyph shapes)."""
    return "".join(r.text[::-1] if r.rtl else r.text for r in split_runs(text))


# --- bidi ------------------------------------------------------------------


def test_pure_arabic_is_one_rtl_run():
    runs = split_runs("بسم الله")
    assert len(runs) == 1 and runs[0].rtl


def test_number_keeps_its_digit_order():
    assert visual("سنة 2026") == "2026 ةنس"
    assert visual("سنة ٢٠٢٦") == "٢٠٢٦ ةنس"
    assert visual("سال ۱۴۰۳") == "۱۴۰۳ لاس"


def test_percent_after_arabic_detaches_like_in_browsers():
    # After Arabic letters a European number acts as an "Arabic number",
    # so % is not part of it and shows on its left.
    assert visual("نسبة 50%") == "%50 ةبسن"


def test_percent_at_line_start_stays_attached():
    assert visual("50% فقط") == "طقف 50%"


def test_separators_inside_numbers():
    assert visual("العدد 3.14") == "3.14 ددعلا"
    assert visual("12:30") == "12:30"
    assert ltr_mask("1-2") == [True, True, True]  # minus joins European numbers
    assert visual("رقم ٢٠٢٦-١٢") == "١٢-٢٠٢٦ مقر"  # but not Arabic-Indic ones


def test_diacritics_follow_their_letter():
    assert ltr_mask("بَ1") == [False, False, True]


# --- rendering --------------------------------------------------------------


@needs_fonts
def test_render_returns_ink_image():
    img = render_line("مرحبا بالعالم", NASKH, size_px=40)
    assert img.dtype == np.uint8 and img.ndim == 2
    assert img.shape[0] >= 40
    assert img.max() > 200  # there is solid ink
    assert img[0].max() == 0 and img[:, 0].max() == 0  # padding is blank


@needs_fonts
def test_longer_text_is_wider():
    short = render_line("كتاب", NASKH)
    long = render_line("كتاب كبير جدا", NASKH)
    assert long.shape[1] > short.shape[1]


@needs_fonts
def test_diacritics_are_drawn():
    plain = render_line("بسم", NASKH)
    marked = render_line("بِسْمِ", NASKH)
    assert int(marked.astype(int).sum()) > int(plain.astype(int).sum())


@needs_fonts
def test_joining_changes_shapes():
    # Joined letters are narrower than the same letters forced apart by spaces.
    joined = render_line("بببب", NASKH)
    apart = render_line("ب ب ب ب", NASKH)
    assert joined.shape[1] < apart.shape[1]


@needs_fonts
def test_persian_and_digits_render():
    img = render_line("این کتاب در ۱۴۰۳ چاپ شد، ٢٠٢٦ 2026", VAZIR)
    assert img.max() > 200


@needs_fonts
def test_missing_glyph_raises():
    with pytest.raises(MissingGlyphError):
        render_line("שלום", NASKH)  # Hebrew: not in an Arabic font


def test_empty_line_rejected():
    with pytest.raises(ValueError):
        render_line("", NASKH)


def test_font_supports():
    assert font_supports("سلام", missing="پچ")
    assert not font_supports("پاکستان", missing="پچ")


# --- font manifest and licenses ---------------------------------------------


def test_fonts_with_western_looking_arabic_digits_are_excluded():
    # Zain draws ٤ with the same shape as 4 (found during visual inspection).
    manifest = json.loads((ROOT / "training" / "fonts.json").read_text(encoding="utf-8"))
    zain = next(f for f in manifest["families"] if f["dir"] == "zain")
    regular = next(f for f in zain["files"] if f["filename"] == "Zain-Regular.ttf")
    assert "٤" in regular["missing"] and "4" in regular["missing"]


def test_no_font_draws_two_different_digits_the_same_way():
    """For every installed font, the digits it may be used for must look different
    after shaping. Found by the owner's manual review: Zain Italic drew ٢٠١٤ as 2014,
    and Qahiri / Reem Kufi / Baloo Bhaijaan drew some Arabic-Indic digits exactly like
    Persian ones."""
    from arabic_ocr import charset
    from training.fonts import load_fonts

    W, A, E = (
        charset.WESTERN_DIGITS,
        charset.ARABIC_INDIC_DIGITS,
        charset.EASTERN_ARABIC_INDIC_DIGITS,
    )
    fonts = load_fonts()
    if not fonts:
        pytest.skip("no fonts installed")
    for font in fonts:
        for i in range(10):
            pairs = [(W[i], A[i]), (W[i], E[i])] + ([(A[i], E[i])] if i in (4, 5, 6) else [])
            for a, b in pairs:
                if a in font.missing or b in font.missing:
                    continue
                ia = render_line("ب " + a, font.path, size_px=40, padding=0)
                ib = render_line("ب " + b, font.path, size_px=40, padding=0)
                same = ia.shape == ib.shape and np.array_equal(ia, ib)
                assert not same, f"{font.path.name} draws {a} and {b} identically"


def test_font_manifest_is_complete_and_licensed():
    manifest = json.loads((ROOT / "training" / "fonts.json").read_text(encoding="utf-8"))
    licenses = (ROOT / "data" / "LICENSES.md").read_text(encoding="utf-8")
    assert len(manifest["families"]) >= 50
    for fam in manifest["families"]:
        assert fam["license"] in ("OFL", "APACHE2")
        assert fam["role"] in ("core", "display")
        assert fam["files"], fam["family"]
        assert all(len(f["sha256"]) == 64 for f in fam["files"])
        assert f"| {fam['family']} |" in licenses
