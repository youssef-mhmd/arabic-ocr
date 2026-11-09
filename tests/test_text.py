"""Tests for corpus cleaning and the line sampler."""

from arabic_ocr import charset
from training.text import LineSampler, SamplerConfig, arabic_ratio, clean_text


def test_clean_text_cuts_out_latin_and_emoji():
    raw = "قال الرئيس (President) إن الاقتصاد 🙂 ينمو بسرعة"
    assert clean_text(raw) == ["قال الرئيس (", ") إن الاقتصاد", "ينمو بسرعة"]


def test_clean_text_keeps_lines_separate():
    assert clean_text("عنوان\nفقرة أولى") == ["عنوان", "فقرة أولى"]


def test_clean_text_drops_mostly_non_arabic_segments():
    assert clean_text("2026 - 12 - 01") == []


def test_clean_text_normalizes_presentation_forms():
    assert clean_text("ﻻﺑﻪ") == ["لابه"]


def test_clean_text_drops_orphan_diacritics():
    # a fatha after a space has no letter to sit on
    assert clean_text("كتب َالدرس") == ["كتب الدرس"]


def test_clean_text_drops_loose_tatweel_but_keeps_attached():
    assert clean_text("ـــ الفصل الأول ـــ") == ["الفصل الأول"]
    assert clean_text("مـــحمد") == ["مـــحمد"]


def test_lines_always_have_a_letter_or_digit():
    sampler = LineSampler(SamplerConfig(short_prob=1.0), seed=0)
    for _ in range(500):
        line = sampler.sample("« . ، » ؟ سلام")
        if line is not None:
            assert any(c.isalnum() for c in line)


def test_arabic_ratio():
    assert arabic_ratio("سلام") == 1.0
    assert arabic_ratio("١٢٣") == 0.0
    assert arabic_ratio("") == 0.0


SEGMENTS = clean_text(
    "اَلْعِلْمُ نُورٌ وَالْجَهْلُ ظَلَامٌ، وَمَنْ طَلَبَ الْعُلَا سَهِرَ اللَّيَالِي\n"
    "در این کتاب داستان‌های کوتاه فارسی گردآوری شده است\n"
    "پاکستان کی قومی زبان اردو ہے اور یہ بہت خوبصورت زبان ہے"
)


def _many_lines(seed=0, n=2000, **cfg):
    sampler = LineSampler(SamplerConfig(**cfg), seed=seed)
    return list(sampler.lines(SEGMENTS[i % len(SEGMENTS)] for i in range(n)))


def test_lines_are_valid_labels():
    for line in _many_lines():
        assert 1 <= len(line) <= 60
        assert line == line.strip()
        assert "  " not in line
        assert charset.normalize(line) == line
        assert charset.decode(charset.encode(line)) == line


def test_same_seed_same_lines():
    assert _many_lines(seed=7, n=200) == _many_lines(seed=7, n=200)
    assert _many_lines(seed=7, n=200) != _many_lines(seed=8, n=200)


def test_mix_of_diacritized_and_plain_lines():
    lines = _many_lines(n=600)
    # The first segment is fully diacritized, so its lines should come out both ways.
    first = [line for line in lines if "ل" in line and "ع" in line and "ک" not in line]
    has_marks = [any(c in charset.DIACRITIC_SET for c in line) for line in first]
    assert any(has_marks) and not all(has_marks)


def test_numbers_are_inserted_in_all_digit_styles():
    text = "".join(_many_lines(n=3000, number_prob=0.5))
    assert any(c in text for c in charset.WESTERN_DIGITS)
    assert any(c in text for c in charset.ARABIC_INDIC_DIGITS)
    assert any(c in text for c in charset.EASTERN_ARABIC_INDIC_DIGITS)


def test_lengths_cover_the_range():
    lengths = [len(line) for line in _many_lines(n=3000)]
    assert min(lengths) <= 3
    assert max(lengths) >= 50


def test_max_len_respected():
    assert all(len(line) <= 10 for line in _many_lines(max_len=10))
