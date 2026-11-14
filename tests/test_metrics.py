"""Tests for CER/WER."""

import pytest

from training.metrics import edit_distance, score


def test_edit_distance():
    assert edit_distance("كتاب", "كتاب") == 0
    assert edit_distance("كتاب", "كتب") == 1  # deletion
    assert edit_distance("كتب", "كتاب") == 1  # insertion
    assert edit_distance("كتاب", "كباب") == 1  # replacement
    assert edit_distance("", "abc") == 3
    assert edit_distance(["a", "b"], ["b"]) == 1  # works on word lists too


def test_perfect_prediction():
    s = score(["بِسْمِ اللَّهِ"], ["بِسْمِ اللَّهِ"])
    assert s.cer == 0 and s.wer == 0 and s.exact == 1


def test_missing_diacritic_counts_only_with_diacritics():
    s = score(["بِسْمِ"], ["بسْمِ"])  # kasra on ب missing
    assert s.cer == pytest.approx(1 / 6)
    assert s.cer_no_diacritics == 0
    assert s.wer == 1 and s.wer_no_diacritics == 0


def test_tatweel_is_never_an_error():
    s = score(["الخيل"], ["الخيـــل"])
    assert s.cer == 0 and s.exact == 1
    assert score(["فـإن"], ["فإن"]).cer == 0


def test_ignore_conventions_unifies_digits_and_punctuation_spacing():
    from training.metrics import ignore_conventions

    assert ignore_conventions("العجوز(٣) البصرة .") == ignore_conventions("العجوز(3) البصرة.")
    assert ignore_conventions("سنة ۱۹۸۸") == "سنة 1988"
    assert ignore_conventions("« قال »") == "«قال»"
    assert ignore_conventions("قال ثم") == "قال ثم"  # ordinary spaces are kept


def test_rates_are_per_character_of_reference():
    s = score(["سلام", "عليكم"], ["سلم", "عليكم"])  # 1 error in 9 characters
    assert s.cer == pytest.approx(1 / 9)
    assert s.wer == pytest.approx(1 / 2)
    assert s.exact == pytest.approx(0.5)


def test_fast_edit_distance_matches_reference():
    import random

    from training.metrics import edit_distance, fast_edit_distance

    rng = random.Random(0)
    alphabet = "ابتث سم"
    for _ in range(200):
        a = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))
        b = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))
        assert fast_edit_distance(a, b) == edit_distance(a, b)
