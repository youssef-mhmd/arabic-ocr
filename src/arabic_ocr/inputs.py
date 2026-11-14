"""Read input files as page images: images, multi-page TIFFs, PDFs, folders.

    from arabic_ocr.inputs import find_inputs, load_pages
    for path in find_inputs("scans/"):            # a file, a folder, or a glob
        for page in load_pages(path):             # one image per page
            ...

- Images (PNG, JPEG, BMP, WebP): one page. TIFF: every page of a multi-page file.
- PDF: every page rendered at 300 DPI by default (pypdfium2). Scanned and
  image-based PDFs are read this way; any text a PDF already has is ignored.
- Paths may contain any characters (Arabic file names on Windows included).

Pages come back as NumPy images (gray, BGR or BGRA, uint8), ready for OCR.read_image.
"""

from __future__ import annotations

import glob
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED = IMAGE_SUFFIXES | PDF_SUFFIXES
#: Resolution PDF pages are rendered at (PRD FR-2).
PDF_DPI = 300


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint16:  # 16-bit PNG/TIFF
        return (image >> 8).astype(np.uint8)
    if image.dtype != np.uint8:
        return np.clip(image, 0, 255).astype(np.uint8)
    return image


def load_image(path: str | Path) -> np.ndarray:
    """The first (or only) page of an image file."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return _to_uint8(image)


def _tiff_pages(path: Path) -> list[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    ok, pages = cv2.imdecodemulti(data, cv2.IMREAD_UNCHANGED)
    if not ok or not pages:
        raise ValueError(f"Cannot read TIFF: {path}")
    return [_to_uint8(p) for p in pages]


def _pdf_pages(path: Path, dpi: int) -> Iterator[np.ndarray]:
    import pypdfium2 as pdfium  # imported here: only needed for PDFs

    pdf = pdfium.PdfDocument(str(path))
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            # grayscale=True renders straight to one channel (less memory than RGB)
            bitmap = page.render(scale=dpi / 72, grayscale=True)
            yield bitmap.to_numpy().copy()
            bitmap.close()
            page.close()
    finally:
        pdf.close()


def page_count(path: str | Path) -> int:
    """Number of pages in a file, without rendering them."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        try:
            return len(pdf)
        finally:
            pdf.close()
    if suffix in {".tif", ".tiff"}:
        return len(_tiff_pages(path))
    return 1


def load_pages(path: str | Path, dpi: int = PDF_DPI) -> Iterator[np.ndarray]:
    """Every page of a file, one image at a time (PDF pages are rendered lazily)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type '{suffix}': {path}")
    if suffix in PDF_SUFFIXES:
        yield from _pdf_pages(path, dpi)
    elif suffix in {".tif", ".tiff"}:
        yield from _tiff_pages(path)
    else:
        yield load_image(path)


def find_inputs(target: str | Path, recursive: bool = False) -> list[Path]:
    """The supported files named by `target`: one file, a folder, or a glob pattern
    (e.g. "scans/*.png"). Sorted by name so batches always run in the same order."""
    target = Path(target)
    if target.is_file():
        return [target]
    if target.is_dir():
        pattern = "**/*" if recursive else "*"
        files = target.glob(pattern)
    else:
        files = (Path(p) for p in glob.glob(str(target), recursive=recursive))
    return sorted(p for p in files if p.is_file() and p.suffix.lower() in SUPPORTED)
