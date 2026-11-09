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


def test_rates_are_per_character_of_reference():
    s = score(["سلام", "عليكم"], ["سلم", "عليكم"])  # 1 error in 9 characters
    assert s.cer == pytest.approx(1 / 9)
    assert s.wer == pytest.approx(1 / 2)
    assert s.exact == pytest.approx(0.5)
