"""Tests for reading inputs (images, multi-page TIFF, PDF, folders) and batch CLI."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arabic_ocr.cli import Format, output_names
from arabic_ocr.inputs import find_inputs, load_pages, page_count
from arabic_ocr.pipeline import PAGE_BREAK, Document, Page

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
NASKH = ROOT / "data" / "fonts" / "notonaskharabic" / "NotoNaskhArabic[wght].ttf"


def _pages(n: int, size=(300, 200)) -> list[Image.Image]:
    """n gray pages, each with a different shade so they can be told apart."""
    return [Image.new("L", size, color=60 * (i + 1)) for i in range(n)]


def test_find_inputs_file_folder_glob(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ("b.png", "a.jpg", "notes.txt", "sub/c.pdf"):
        (tmp_path / name).write_bytes(b"x")
    assert find_inputs(tmp_path / "a.jpg") == [tmp_path / "a.jpg"]
    assert find_inputs(tmp_path) == [tmp_path / "a.jpg", tmp_path / "b.png"]
    assert find_inputs(tmp_path, recursive=True)[-1] == tmp_path / "sub" / "c.pdf"
    assert find_inputs(str(tmp_path / "*.png")) == [tmp_path / "b.png"]
    assert find_inputs(tmp_path / "missing*.png") == []


def test_multipage_tiff(tmp_path):
    path = tmp_path / "scan.tif"
    first, *rest = _pages(3)
    first.save(path, save_all=True, append_images=rest)
    pages = list(load_pages(path))
    assert page_count(path) == 3
    assert [int(p.mean()) for p in pages] == [60, 120, 180]


def test_pdf_pages_rendered_at_dpi(tmp_path):
    path = tmp_path / "doc.pdf"
    first, *rest = _pages(2, size=(144, 72))
    first.save(path, save_all=True, append_images=rest, resolution=72)  # 2 x 1 inch pages
    assert page_count(path) == 2
    pages = list(load_pages(path, dpi=150))
    assert len(pages) == 2
    assert pages[0].shape == (150, 300)  # 1 x 2 inches at 150 DPI, one channel
    assert abs(int(pages[1].mean()) - 120) <= 2


def test_non_ascii_file_name(tmp_path):
    path = tmp_path / "صفحة.png"
    _pages(1)[0].save(path)
    assert next(load_pages(path)).shape == (200, 300)


def test_unsupported_type(tmp_path):
    path = tmp_path / "x.gif"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="Unsupported"):
        next(load_pages(path))


def test_output_names_never_collide():
    files = [Path("a.png"), Path("a.pdf"), Path("b.tif")]
    assert output_names(files, Format.txt) == {
        Path("a.png"): "a.png.txt",
        Path("a.pdf"): "a.pdf.txt",
        Path("b.tif"): "b.txt",
    }


def test_document_text_separates_pages():
    empty = Page(10, 10, 0.0, [])
    assert Document([empty, empty]).text == PAGE_BREAK


@pytest.mark.skipif(
    not (NASKH.exists() and (MODELS / "detector-v2.onnx").exists()), reason="fonts/models"
)
def test_cli_batch_folder(tmp_path):
    from typer.testing import CliRunner

    from arabic_ocr.cli import app
    from training.render import render_line

    ink = render_line("بسم الله الرحمن الرحيم", NASKH, 40)
    page = np.full((ink.shape[0] + 80, ink.shape[1] + 80), 255, np.uint8)
    page[40:-40, 40:-40] = 255 - ink
    src = tmp_path / "in"
    src.mkdir()
    Image.fromarray(page).save(src / "one.png")
    Image.fromarray(page).save(src / "two.pdf")

    out = tmp_path / "out"
    result = CliRunner().invoke(app, [str(src), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in out.iterdir()) == ["one.txt", "two.txt"]
    assert "الرحمن" in (out / "one.txt").read_text(encoding="utf-8")

    no_out = CliRunner().invoke(app, [str(src)])
    assert no_out.exit_code == 1 and "output folder" in no_out.output
