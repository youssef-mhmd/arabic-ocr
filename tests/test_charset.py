"""Tests for the character set and normalization rules."""

import pytest

from arabic_ocr import charset
from arabic_ocr.charset import (
    BLANK_ID,
    NUM_CLASSES,
    SYMBOLS,
    UnknownCharacterError,
    convert_digits,
    decode,
    encode,
    normalize,
    strip_diacritics,
    strip_tatweel,
)

# --- charset contents -------------------------------------------------------


def test_no_duplicate_symbols():
    assert len(SYMBOLS) == len(set(SYMBOLS))


def test_every_symbol_is_one_character():
    assert all(len(s) == 1 for s in SYMBOLS)


def test_blank_is_id_zero_and_counted():
    assert BLANK_ID == 0
    assert NUM_CLASSES == len(SYMBOLS) + 1
    assert charset.symbols_in_order()[0] == charset.BLANK


def test_size_is_about_150():
    assert 120 <= len(SYMBOLS) <= 180


@pytest.mark.parametrize(
    "chars",
    [
        "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي",  # letters + hamza forms
        "ًٌٍَُِّْٰ",  # tashkeel
        "ـ",  # tatweel
        "٠١٢٣٤٥٦٧٨٩",
        "۰۱۲۳۴۵۶۷۸۹",
        "0123456789",
        "،؛؟٪",
        "پچژگکیٹڈڑںےۀ",  # Persian/Urdu
    ],
)
def test_required_characters_present(chars):
    missing = [c for c in chars if not charset.is_supported(c)]
    assert missing == []


def test_latin_letters_not_supported():
    assert not charset.is_supported("a")
    assert charset.unsupported_chars("abc سلام") == {"a", "b", "c"}


def test_presentation_forms_not_in_charset():
    assert not any(charset._is_presentation_form(s) for s in SYMBOLS)


# --- encode / decode --------------------------------------------------------


def test_round_trip_with_diacritics_and_digits():
    text = normalize("قَالَ: «الْعِلْمُ نُورٌ» ٢٠٢٦، پاکستان؟")
    assert decode(encode(text)) == text


def test_decode_ignores_blank():
    ids = encode("سلام")
    with_blanks = [BLANK_ID, ids[0], BLANK_ID, *ids[1:], BLANK_ID]
    assert decode(with_blanks) == "سلام"


def test_encode_unknown_raises_or_skips():
    with pytest.raises(UnknownCharacterError):
        encode("سلام x")
    assert decode(encode("سلام x", unknown="skip")) == "سلام "


def test_ids_are_never_blank():
    assert BLANK_ID not in encode("".join(SYMBOLS))


# --- normalization ----------------------------------------------------------


def test_lam_alef_ligature_becomes_two_letters():
    assert normalize("ﻻ") == "لا"  # ﻻ


def test_contextual_letter_forms_become_base_letters():
    # ﺑ (beh initial), ﻬ (heh medial), ﺔ (teh marbuta final)
    assert normalize("ﺑﻬﺔ") == "بهة"


def test_isolated_tashkeel_form_becomes_mark_without_space():
    # U+FE7C is "shadda isolated form"; NFKC alone would give " ّ"
    assert normalize("بﹼ") == "بّ"


def test_persian_presentation_forms():
    assert normalize("ﮏ") == "ک"  # keheh final form
    assert normalize("ﯽ") == "ی"  # Farsi yeh isolated form


def test_decomposed_hamza_is_composed():
    assert normalize("أ") == "أ"
    assert normalize("إ") == "إ"
    assert normalize("ؤ") == "ؤ"


def test_stacked_diacritics_get_one_order():
    shadda_then_fatha = "بَّ"
    fatha_then_shadda = "بَّ"
    assert normalize(shadda_then_fatha) == normalize(fatha_then_shadda)


def test_invisible_characters_removed():
    text = "‏مرحبا‌ بك‫﻿"
    assert normalize(text) == "مرحبا بك"


def test_whitespace_collapsed():
    assert normalize("  سلام \t\n  عليكم  ") == "سلام عليكم"


def test_normalize_never_reverses_or_changes_letters():
    text = "أإآ ي ى ك ک ی"
    assert normalize(text) == text


def test_normalize_keeps_diacritics():
    text = "بِسْمِ"
    assert normalize(text) == text


def test_normalize_is_idempotent():
    text = "ﻻﺑ‏ بَّ  ٢٠٢٦"
    once = normalize(text)
    assert normalize(once) == once


def test_normalized_presentation_text_is_encodable():
    every_form = "".join(chr(cp) for cp in range(0xFE80, 0xFEFC + 1))
    assert charset.unsupported_chars(normalize(every_form)) == set()


# --- output options ---------------------------------------------------------


def test_strip_diacritics_keeps_letters_and_hamza():
    assert strip_diacritics("بِسْمِ اللَّهِ") == "بسم الله"
    assert strip_diacritics("ۂ") == "ۂ"
    assert strip_diacritics("ۂ") == "ۂ"


def test_strip_tatweel():
    assert strip_tatweel("مـــحمد") == "محمد"


def test_convert_digits():
    assert convert_digits("٢٠۲۶ 12", "western") == "2026 12"
    assert convert_digits("2026 ۱۲", "arabic") == "٢٠٢٦ ١٢"
    assert convert_digits("٢٠ 12", "keep") == "٢٠ 12"
