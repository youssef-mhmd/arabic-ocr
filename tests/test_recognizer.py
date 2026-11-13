"""Tests for the ONNX runtime wrapper's image preparation (no model file needed)."""

import numpy as np

from arabic_ocr import charset
from arabic_ocr.recognize.recognizer import MIN_WIDTH, LineRecognizer, prepare_line, to_gray


def test_to_gray_handles_bgr_and_transparent_bgra():
    bgr = np.zeros((10, 20, 3), np.uint8)
    assert to_gray(bgr).shape == (10, 20)

    bgra = np.zeros((10, 20, 4), np.uint8)  # fully transparent black -> white
    assert to_gray(bgra).min() == 255


def test_prepare_line_keeps_aspect_ratio():
    line = np.full((32, 300), 255, np.uint8)
    out = prepare_line(line, 64)
    assert out.shape == (64, 600)
    assert out.dtype == np.uint8

    tall = np.full((128, 300), 255, np.uint8)
    assert prepare_line(tall, 64).shape == (64, 150)


def _bare_recognizer(stride: int = 4) -> LineRecognizer:
    rec = LineRecognizer.__new__(LineRecognizer)  # skip loading a model
    rec.stride = stride
    rec.height = 64
    return rec


def test_pad_is_on_the_left_and_less_than_one_step():
    rec = _bare_recognizer()
    line = np.full((64, 101), 255, np.uint8)
    line[:, -1] = 0  # dark right edge: the start of an Arabic line
    out = rec._pad(line)
    assert out.shape == (64, 104)
    assert out.shape[1] % rec.stride == 0
    assert (out[:, -1] == 0).all()  # right edge untouched
    np.testing.assert_array_equal(out[:, 3:], line)


def test_pad_reaches_min_width():
    rec = _bare_recognizer()
    assert rec._pad(np.full((64, 5), 255, np.uint8)).shape[1] == MIN_WIDTH


def test_charset_fingerprint_is_stable_and_short():
    fp = charset.fingerprint()
    assert len(fp) == 12
    assert fp == charset.fingerprint()
