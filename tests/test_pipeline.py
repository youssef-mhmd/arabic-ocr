"""Tests for preprocessing, reading order, line cropping, the pipeline and the CLI.

The end-to-end test uses the real recognizer and a stand-in detector that returns
the true line boxes, so it runs before the detector model exists.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from arabic_ocr import charset
from arabic_ocr.detect.boxes import Detection, RotRect
from arabic_ocr.export.to_json import SCHEMA, document_to_json
from arabic_ocr.layout import reading_order
from arabic_ocr.pipeline import OCR, crop_line, split_words
from arabic_ocr.preprocess import estimate_skew, preprocess, rotate
from arabic_ocr.recognize.ctc import Decoded
from arabic_ocr.recognize.recognizer import LineRecognizer

ROOT = Path(__file__).resolve().parent.parent
RECOGNIZER = ROOT / "models" / "recognizer-v3.onnx"
NASKH = ROOT / "data" / "fonts" / "notonaskharabic" / "NotoNaskhArabic[wght].ttf"


def _bars_page(angle: float = 0.0) -> np.ndarray:
    """White page with dark horizontal 'text lines', rotated by `angle` degrees."""
    page = np.full((800, 600), 255, np.uint8)
    for y in range(80, 720, 40):
        page[y : y + 18, 60:540] = 30
    if angle:
        page, _ = rotate(page, angle)
    return page


# --- preprocess --------------------------------------------------------------


@pytest.mark.parametrize("angle", [-4.0, 0.0, 2.5])
def test_estimate_skew_undoes_rotation(angle):
    assert estimate_skew(_bars_page(angle)) == pytest.approx(-angle, abs=0.3)


def test_preprocess_matrix_maps_original_to_straightened():
    tilted = _bars_page(3.0)
    out = preprocess(tilted)
    assert abs(out.angle + 3.0) < 0.3
    # The page centre stays at the centre of the (grown) output canvas.
    h, w = tilted.shape
    cx, cy = out.matrix @ [w / 2, h / 2, 1]
    oh, ow = out.image.shape
    assert (cx, cy) == pytest.approx((ow / 2, oh / 2), abs=1)


def test_preprocess_leaves_straight_page_alone():
    page = _bars_page()
    out = preprocess(page)
    assert out.angle == 0.0 and out.image.shape == page.shape


def test_binarize_gives_black_on_white():
    page = 255 - _bars_page()  # light text on dark
    out = preprocess(page, deskew=False, binarize_page=True).image
    assert set(np.unique(out)) == {0, 255}
    assert (out == 255).mean() > 0.5


# --- reading order -----------------------------------------------------------


def _flat(blocks):
    return [i for b in blocks for i in b]


def test_single_column_top_to_bottom():
    boxes = [(50, y, 500, y + 20) for y in (100, 40, 70)]
    assert _flat(reading_order(boxes)) == [1, 2, 0]


def test_two_columns_right_first_with_heading():
    heading = (100, 10, 500, 40)
    right = [(320, y, 560, y + 20) for y in (60, 90, 120)]
    left = [(40, y, 280, y + 20) for y in (61, 91, 121)]  # rows line up across columns
    boxes = [heading, *left, *right]
    blocks = reading_order(boxes)
    assert _flat(blocks) == [0, 4, 5, 6, 1, 2, 3]
    assert [4, 5, 6] in blocks and [1, 2, 3] in blocks


def test_big_gap_starts_new_block():
    boxes = [(50, 0, 500, 20), (50, 30, 500, 50), (50, 150, 500, 170)]
    assert reading_order(boxes) == [[0, 1], [2]]


def test_same_row_right_to_left():
    boxes = [(50, 0, 200, 20), (300, 2, 500, 22)]  # gap 100 > 0.8 * height -> columns
    assert _flat(reading_order(boxes)) == [1, 0]


def test_reading_order_empty():
    assert reading_order([]) == []


# --- crop + word boxes -------------------------------------------------------


def test_crop_line_rectifies_rotated_line(monkeypatch):
    from arabic_ocr import pipeline

    monkeypatch.setattr(pipeline, "PAD_Y", 0.3)
    page = np.full((300, 400), 255, np.uint8)
    rect = RotRect(200, 150, 200, 30, 10.0)
    cv2.fillPoly(page, [rect.quad().astype(np.int32)], 0)
    crop, back = crop_line(page, rect)
    ch, cw = crop.shape
    assert ch == round(30 * 1.6)
    pad = 9
    assert crop[pad + 3 : ch - pad - 3, 15:-15].max() < 60  # the line, now horizontal
    assert crop[:2].min() > 200 and crop[-2:].min() > 200  # white margins
    # The crop centre maps back to the line centre.
    assert back @ [cw / 2, ch / 2, 1] == pytest.approx([200, 150], abs=1)


def test_split_words_tiles_the_line():
    sp = charset.encode(" ")[0]
    a, b = charset.encode("بت")
    # reading order: word1 (2 chars) at slices 2,5; space at 10; word2 at 14,18.
    dec = Decoded("", 0.9, [a, b, sp, a, b], [0.9, 0.8, 1.0, 0.7, 0.6], [2, 5, 10, 14, 18])
    words = split_words(dec, crop_w=100, crop_h=64, model_h=64, stride=4)
    assert [w[0] for w in words] == ["بت", "بت"]
    (_, c1, x0a, x1a), (_, c2, x0b, x1b) = words
    assert (x1a, x0a) == (100, 60)  # right edge .. the space at slice 10
    assert (x1b, x0b) == (60, 0)
    assert c1 == pytest.approx(0.85) and c2 == pytest.approx(0.65)


# --- end to end with a stand-in detector -------------------------------------


class _TruthDetector:
    def __init__(self, quads):
        self.quads = quads

    def detect(self, gray):
        return [Detection(q, 1.0) for q in self.quads]


@pytest.mark.skipif(not (RECOGNIZER.exists() and NASKH.exists()), reason="model/fonts missing")
def test_pipeline_reads_a_two_line_page():
    from training.render import render_line

    texts = ["بسم الله الرحمن الرحيم", "قال الشيخ في كتابه"]
    page = np.full((260, 900), 255, np.uint8)
    quads = []
    for k, text in enumerate(texts):
        ink = render_line(text, NASKH, 40, padding=0)
        rows = np.where(ink.max(axis=1) > 32)[0]
        ink = ink[rows[0] : rows[-1] + 1]
        h, w = ink.shape
        x1, y0 = 850, 40 + 110 * k
        page[y0 : y0 + h, x1 - w : x1] = 255 - ink
        quads.append(np.float32([[x1, y0], [x1 - w, y0], [x1 - w, y0 + h], [x1, y0 + h]]))

    ocr = OCR.__new__(OCR)
    ocr.detector = _TruthDetector(quads[::-1])  # order must come from layout, not detector
    ocr.recognizer = LineRecognizer(RECOGNIZER)
    result = ocr.read_image(page, deskew=False)

    assert [ln.text for ln in result.lines] == texts
    first = result.lines[0]
    assert [w.text for w in first.words] == texts[0].split()
    # Word boxes lie inside the line, right to left.
    xs = [w.box[:, 0].mean() for w in first.words]
    assert xs == sorted(xs, reverse=True)
    assert first.box[:, 0].min() < 850 - 10 and first.box[:, 0].max() > 850

    doc = json.loads(document_to_json([result], "test.png"))
    assert doc["schema"] == SCHEMA
    assert doc["pages"][0]["text"] == result.text
    assert doc["pages"][0]["blocks"][0]["lines"][0]["words"][0]["text"] == "بسم"

    no_diac = ocr.read_image(page, deskew=False, diacritics=False)
    assert no_diac.text == result.text  # nothing to strip here, but the option works


# --- CLI ---------------------------------------------------------------------


def test_cli_help_and_missing_model(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from arabic_ocr import pipeline
    from arabic_ocr.cli import app

    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0

    img = tmp_path / "page.png"
    cv2.imwrite(str(img), _bars_page())
    monkeypatch.setattr(pipeline, "MODELS_DIR", tmp_path)
    res = runner.invoke(app, [str(img)])
    assert res.exit_code == 1
    assert "Model not found" in res.output


def test_two_columns_with_uneven_paragraph_gaps_and_page_number():
    h = 20
    right = [(320, y, 560, y + h) for y in (0, 30, 60, 120, 150)]  # gap after 3rd line
    left = [(40, y, 280, y + h) for y in (0, 30, 90, 120, 150)]  # gap after 2nd line
    page_no = [(290, 200, 310, 220)]  # centred, inside the gutter
    boxes = right + left + page_no
    assert _flat(reading_order(boxes)) == list(range(11))


def test_footnote_spanning_both_columns_comes_after_them():
    right = [(320, y, 560, y + 20) for y in (0, 30, 60)]
    left = [(40, y, 280, y + 20) for y in (0, 30, 60)]
    footnote = [(40, 120, 560, 140)]
    assert _flat(reading_order(right + left + footnote)) == list(range(7))


def test_poetry_like_rows_are_not_columns_when_not_side_by_side():
    # Short lines alternating right and left (one per row): no column structure.
    boxes = [(300, 0, 560, 20), (40, 40, 280, 60), (300, 80, 560, 100)]
    assert _flat(reading_order(boxes)) == [0, 1, 2]


def test_postprocess_options():
    from arabic_ocr.pipeline import postprocess

    text = "جـائزتـان ٢٠١٤ كَتَبَ"
    assert postprocess(text) == text
    assert postprocess(text, tatweel=False) == "جائزتان ٢٠١٤ كَتَبَ"
    assert postprocess(text, diacritics=False, tatweel=False, digits="western") == (
        "جائزتان 2014 كتب"
    )


def test_paragraphs_from_short_last_line_and_indent():
    h = 20
    # paragraph 1: two full lines + a short last line; paragraph 2: indented first line
    boxes = [
        (40, 0, 560, h),
        (40, 30, 560, 30 + h),
        (380, 60, 560, 60 + h),  # ends short (left end far from the column edge)
        (40, 90, 520, 90 + h),  # indented at the right -> also a new paragraph
        (40, 120, 560, 120 + h),
        (40, 150, 560, 150 + h),
    ]
    assert reading_order(boxes) == [[0, 1, 2], [3, 4, 5]]


def test_indent_needs_a_full_previous_line():
    # a list of short right-aligned items is not split at every line by the indent rule
    boxes = [(300, y, 520, y + 20) for y in (0, 30, 60)]
    boxes.append((40, 90, 560, 110))
    blocks = reading_order(boxes)
    assert [0, 1, 2] in blocks or blocks[0][:3] == [0, 1, 2]


def test_noise_lines_are_dropped():
    from arabic_ocr.pipeline import Line, is_noise

    def line(text, conf):
        return Line(text, conf, np.zeros((4, 2), np.float32))

    assert is_noise(line("* *************۵۵** **۵۵", 0.61))  # ornament border
    assert is_noise(line("(222222م", 0.33))
    assert is_noise(line("ط", 0.35))  # speck read as a letter
    assert not is_noise(line("- ٤١ -", 0.85))  # page number
    assert not is_noise(line("الاتحاد", 0.97))  # repeated letters are fine
    assert not is_noise(line("الـــجـبـل فــيـدرك", 0.88))  # tatweel runs are fine
    assert not is_noise(line("۶۸۰۰۰)،", 1.0))


def test_ragged_web_paragraph_stays_one_block():
    # right-aligned, unjustified lines ending at different places (as browsers wrap)
    boxes = [(x0, y, 560, y + 20) for x0, y in ((120, 0), (260, 30), (80, 60), (300, 90))]
    assert reading_order(boxes) == [[0, 1, 2, 3]]


def test_word_list_fixes_unsure_dot_slips_only():
    from arabic_ocr.correct import Lexicon

    lex = Lexicon({"الكتاب": 5000, "كتب": 900, "بيت": 800})
    assert lex.correct("الكتاث", 0.6) == "الكتاب"  # ث -> ب (same shape)
    assert lex.correct("بيث", 0.6) == "بيت"  # ث -> ت (same shape, dots differ)
    assert lex.correct("(بيث)،", 0.6) == "(بيت)،"  # punctuation kept
    assert lex.correct("بيث", 0.99) == "بيث"  # confident words are left alone
    assert lex.correct("كتب", 0.5) == "كتب"  # already a known word
    assert lex.correct("زيدون", 0.5) == "زيدون"  # unknown, no close known word: unchanged


def test_remove_rules_keeps_letters_and_shadows():
    from arabic_ocr.preprocess import remove_rules

    page = np.full((600, 400), 230, np.uint8)
    page[:, :200] -= 60  # a shadow over the left half
    page[50:550, 30:33] = 20  # a vertical frame line (83% of the page height)
    page[100:140, 200:204] = 20  # an alef-sized stroke
    for i in range(40):  # ordinary letters (they set the typical letter height)
        y, x = 160 + (i // 8) * 40, 220 + (i % 8) * 20
        page[y : y + 15, x : x + 8] = 20
    out = remove_rules(page)
    assert out[300, 31] > 150  # frame painted with the paper colour
    assert out[120, 202] == 20  # the letter stays
    assert abs(int(out[400, 100]) - 170) <= 2  # the shadow is not "ink"
