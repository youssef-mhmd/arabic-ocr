"""Searchable PDF: the hidden text must read back, search and copy as logical text.

Checked with pdfium (pypdfium2), the PDF engine of Chrome's built-in viewer.
"""

from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pytest

from arabic_ocr.export.searchable_pdf import SearchablePdf, image_dpi, ocr_to_pdf
from arabic_ocr.pipeline import Block, Line, Page, Word

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
NASKH = ROOT / "data" / "fonts" / "notonaskharabic" / "NotoNaskhArabic[wght].ttf"


def _box(x0, y0, x1, y1):
    return np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])  # TL TR BR BL


def _line(words: list[str], y: float, right: float = 560, word_w: float = 90) -> Line:
    """A line whose words run right to left, each word_w px wide, 30 px tall."""
    out = []
    for i, text in enumerate(words):
        x1 = right - i * (word_w + 15)
        out.append(Word(text, 0.9, _box(x1 - word_w, y, x1, y + 30)))
    first, last = out[0].box, out[-1].box
    return Line(" ".join(words), 0.9, _box(last[0, 0], y, first[1, 0], y + 30), out)


def _page() -> Page:
    lines = [
        _line(["الكتاب", "المفيد", "جدا"], 40),
        _line(["صدر", "عام", "2014", "م"], 100),
    ]
    return Page(600, 200, 0.0, [Block(lines)])


def _read_back(path: Path) -> tuple[str, pdfium.PdfTextPage]:
    pdf = pdfium.PdfDocument(str(path))
    tp = pdf[0].get_textpage()
    return " ".join(tp.get_text_range().split()), tp


def test_hidden_text_reads_back_in_logical_order(tmp_path):
    image = np.full((200, 600), 255, np.uint8)
    out = tmp_path / "page.pdf"
    pdf = SearchablePdf()
    pdf.add_page(image, _page(), dpi=150)
    pdf.save(out)

    text, tp = _read_back(out)
    # numbers keep their digit order; the one-letter last word is not lost
    assert text == "الكتاب المفيد جدا صدر عام 2014 م"
    for word in ("المفيد", "2014", "عام"):
        search = tp.search(word)
        assert search.get_next() is not None, word
        search.close()

    page = pdfium.PdfDocument(str(out))[0]
    assert page.get_size() == pytest.approx((600 * 72 / 150, 200 * 72 / 150))


def test_color_and_transparent_images(tmp_path):
    out = tmp_path / "color.pdf"
    pdf = SearchablePdf()
    pdf.add_page(np.zeros((200, 600, 3), np.uint8), _page(), jpeg_quality=None)
    pdf.add_page(np.zeros((200, 600, 4), np.uint8), _page(), jpeg_quality=85)
    pdf.save(out)
    assert len(pdfium.PdfDocument(str(out))) == 2


def test_image_dpi_reads_file_metadata(tmp_path):
    from PIL import Image

    Image.new("L", (10, 10)).save(tmp_path / "a.png", dpi=(200, 200))
    Image.new("L", (10, 10)).save(tmp_path / "b.png")
    assert image_dpi(tmp_path / "a.png") == pytest.approx(200, abs=1)
    assert image_dpi(tmp_path / "b.png") == 300


@pytest.mark.skipif(
    not (NASKH.exists() and (MODELS / "detector-v2.onnx").exists()), reason="fonts/models"
)
def test_ocr_to_pdf_end_to_end(tmp_path):
    from PIL import Image

    from arabic_ocr.pipeline import OCR
    from training.render import render_line

    ink = render_line("بسم الله الرحمن الرحيم", NASKH, 40)
    page = np.full((ink.shape[0] + 80, ink.shape[1] + 80), 255, np.uint8)
    page[40:-40, 40:-40] = 255 - ink
    src = tmp_path / "scan.png"
    Image.fromarray(page).save(src)

    out = tmp_path / "scan.pdf"
    doc = ocr_to_pdf(OCR(), src, out)
    text, tp = _read_back(out)
    assert text == " ".join(doc.text.split())
    search = tp.search("الرحمن")
    assert search.get_next() is not None
