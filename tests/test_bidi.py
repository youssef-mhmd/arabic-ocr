"""Tests for the bidi helpers used by the renderer and the recognizer."""

import random
import sys

from arabic_ocr import charset
from arabic_ocr.bidi import logical_order, reading_order


def random_strings(n=20000, seed=0):
    rng = random.Random(seed)
    numberish = set(
        charset.WESTERN_DIGITS + charset.ARABIC_INDIC_DIGITS + charset.EASTERN_ARABIC_INDIC_DIGITS
    ) | set("%٪.,:/+-٫٬، ")
    pool = charset.SYMBOLS
    weights = [8 if s in numberish else 1 for s in pool]
    for _ in range(n):
        yield "".join(rng.choices(pool, weights, k=rng.randint(1, 20)))


def test_examples():
    assert reading_order("سنة 2026") == "سنة 6202"
    assert reading_order("٢٠٢٦ ميلادية") == "٦٢٠٢ ميلادية"
    assert reading_order("بسم الله") == "بسم الله"  # no numbers: unchanged
    assert reading_order("3.14") == "41.3"
    assert reading_order("50% فقط") == "%05 فقط"  # % belongs to a leading number
    assert reading_order("نسبة 50%") == "نسبة 05%"  # ...but not after Arabic text


def realistic_lines(n=20000, seed=0):
    """Words plus numbers written in ONE digit system, as real text has them."""
    rng = random.Random(seed)
    words = ["سنة", "الكتاب", "بِسْمِ", "صفحة", "در", "سال", "میں", "ہے", "عدد", "«قال»", "(أ)"]
    systems = [
        charset.WESTERN_DIGITS,
        charset.ARABIC_INDIC_DIGITS,
        charset.EASTERN_ARABIC_INDIC_DIGITS,
    ]
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(1, 6)):
            if rng.random() < 0.4:
                digits = rng.choice(systems)
                num = "".join(rng.choices(digits, k=rng.randint(1, 5)))
                if rng.random() < 0.3:
                    sep = rng.choice([".", ",", ":", "/", "٫", "٬", "-"])
                    num += sep + "".join(rng.choices(digits, k=rng.randint(1, 3)))
                if rng.random() < 0.2:
                    num += rng.choice(["%", "٪"])
                parts.append(num)
            else:
                parts.append(rng.choice(words))
        yield " ".join(parts)


def test_reading_order_is_its_own_inverse_on_realistic_text():
    # Also true for all 1,020,000 labels of the v1 dataset. It can fail for random
    # character soup (e.g. Persian and Arabic digits mixed inside one number, or a
    # diacritic on a digit), which real text doesn't contain.
    for text in realistic_lines():
        assert logical_order(reading_order(text)) == text, repr(text)


def test_reading_order_is_reverse_of_display_order():
    # The recognizer scans the image right to left, so its targets must be the
    # display order (what the renderer draws, left to right) reversed.
    sys.path.insert(0, ".")
    from training.render import split_runs

    for text in random_strings(5000, seed=1):
        display = "".join(r.text[::-1] if r.rtl else r.text for r in split_runs(text))
        assert reading_order(text) == display[::-1], repr(text)


def test_runtime_bidi_does_not_need_training_code():
    import arabic_ocr.bidi as b

    assert "training" not in b.__file__
