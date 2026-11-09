"""Tests for CTC greedy decoding (pure NumPy)."""

import numpy as np

from arabic_ocr import charset
from arabic_ocr.bidi import reading_order
from arabic_ocr.recognize.ctc import greedy_decode, greedy_decode_batch


def scores_for(path: list[int]) -> np.ndarray:
    """Log-probabilities where each time step strongly prefers the given id."""
    s = np.full((len(path), charset.NUM_CLASSES), np.log(0.01 / charset.NUM_CLASSES))
    for t, i in enumerate(path):
        s[t, i] = np.log(0.99)
    return s


def ids(text: str) -> list[int]:
    return charset.encode(text)


def test_repeats_merge_and_blanks_drop():
    b = charset.BLANK_ID
    seen, lam, meem = ids("سلم")
    out = greedy_decode(scores_for([b, seen, seen, b, lam, lam, lam, b, meem, b]))
    assert out.text == "سلم"
    assert 0.98 < out.confidence <= 1.0


def test_blank_keeps_doubled_letters_apart():
    b = charset.BLANK_ID
    (lam,) = ids("ل")
    assert greedy_decode(scores_for([lam, b, lam])).text == "لل"
    assert greedy_decode(scores_for([lam, lam])).text == "ل"


def test_output_is_converted_from_reading_order_to_logical():
    # The model reads right to left, so it emits "سنة 6202" for "سنة 2026".
    target = reading_order("سنة 2026")
    path = []
    for i in ids(target):
        path += [i, charset.BLANK_ID]
    assert greedy_decode(scores_for(path)).text == "سنة 2026"


def test_empty_prediction():
    out = greedy_decode(scores_for([charset.BLANK_ID] * 5))
    assert out.text == "" and out.confidence == 0.0


def test_batch_respects_lengths():
    b = charset.BLANK_ID
    (s,) = ids("س")
    (m,) = ids("م")
    batch = np.stack([scores_for([s, b, b, m]), scores_for([s, b, m, m])], axis=1)
    out = greedy_decode_batch(batch, lengths=np.array([2, 4]))
    assert [o.text for o in out] == ["س", "سم"]


def test_runtime_decoder_does_not_import_torch():
    import subprocess
    import sys

    code = "import sys, arabic_ocr.recognize.ctc; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
