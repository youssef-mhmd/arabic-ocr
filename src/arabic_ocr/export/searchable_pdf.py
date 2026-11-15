"""Searchable PDF: the original page images with an invisible text layer.

    from arabic_ocr.export.searchable_pdf import SearchablePdf
    pdf = SearchablePdf()
    pdf.add_page(image, page, dpi=300)     # one call per page (image + its OCR result)
    pdf.save("out.pdf")

How it works (the method Tesseract and OCRmyPDF use):
- Each page shows the original image, at its real size (pixels / dpi inches).
- On top, every recognized word is written in text render mode 3 ("invisible"),
  stretched to exactly cover the word in the image. Viewers do not draw it, but
  they use it to select, search and copy text.
- The font is a "glyphless" font made on the fly: its glyphs are empty, and each
  character is stored by its Unicode number (CID = code point) with a ToUnicode
  map back to Unicode. So any Arabic, Persian or Urdu letter works without
  embedding a real Arabic font, and copied text comes out as Unicode.

Arabic order: PDF text always advances left to right. Each word's characters are
written in *display* order, left to right as they appear on the page: the logical
text reversed, except that numbers keep their left-to-right digits (bidi.py).
Viewers rebuild right-to-left text from those positions. Measured on a real
scanned page (192 words), text read back / words found by search:

    method                          pdfium (Chrome/Edge)  MuPDF (Sumatra...)  pdfminer
    display order (this file)       1.7% / 188            1.2% / 189          reversed
    logical order, mirrored matrix  7.4% / 189            reversed / 0        0.0%
      (Tesseract's method; pdfium also flipped brackets)
    logical order, normal matrix    reversed / 0          -                   -

Display order works in both viewer engines; pdfminer (a library, not a viewer)
applies no right-to-left rules and reverses Arabic from most PDFs, including ones
Edge prints itself. Adobe Reader and macOS Preview still need a manual check.

Memory: pages are added one at a time and stored compressed (JPEG or Flate), so
long PDFs don't keep every full-size page image in memory.
"""

from __future__ import annotations

import io
import math
import zlib
from pathlib import Path

import cv2
import numpy as np
import pikepdf
from pikepdf import Array, Dictionary, Name, String

from arabic_ocr.bidi import reading_order
from arabic_ocr.pipeline import Page

#: Glyph width in the glyphless font (1/1000 of the font size).
GLYPH_WIDTH = 500


def glyphless_font() -> bytes:
    """A tiny TrueType font with two empty glyphs, each GLYPH_WIDTH wide."""
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen

    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "glyph"])
    fb.setupCharacterMap({})
    empty = TTGlyphPen(None).glyph()
    fb.setupGlyf({".notdef": empty, "glyph": empty})
    fb.setupHorizontalMetrics({".notdef": (GLYPH_WIDTH, 0), "glyph": (GLYPH_WIDTH, 0)})
    fb.setupHorizontalHeader(ascent=1000, descent=0)
    fb.setupNameTable({"familyName": "GlyphLessFont", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=1000, sTypoDescender=0, usWinAscent=1000, usWinDescent=0)
    fb.setupPost()
    buf = io.BytesIO()
    fb.save(buf)
    return buf.getvalue()


TO_UNICODE = b"""/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 beginbfrange
<0000> <FFFF> <0000>
endbfrange
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"""


def _hex(text: str) -> str:
    """Characters as 2-byte CIDs (= Unicode code points), hex-encoded."""
    return "".join(f"{ord(c):04X}" for c in text if ord(c) <= 0xFFFF)


class SearchablePdf:
    def __init__(self) -> None:
        self.pdf = pikepdf.new()
        font_file = self.pdf.make_stream(glyphless_font())
        descriptor = Dictionary(
            Type=Name.FontDescriptor,
            FontName=Name.GlyphLessFont,
            Flags=5,
            FontBBox=Array([0, 0, GLYPH_WIDTH, 1000]),
            ItalicAngle=0,
            Ascent=1000,
            Descent=0,
            CapHeight=1000,
            StemV=80,
            FontFile2=font_file,
        )
        cid_font = Dictionary(
            Type=Name.Font,
            Subtype=Name.CIDFontType2,
            BaseFont=Name.GlyphLessFont,
            CIDSystemInfo=Dictionary(
                Registry=String("Adobe"), Ordering=String("Identity"), Supplement=0
            ),
            FontDescriptor=descriptor,
            DW=GLYPH_WIDTH,
            # every CID (Unicode code point) draws glyph 1, the empty one
            CIDToGIDMap=self.pdf.make_stream(b"\x00\x01" * 65536),
        )
        self.font = self.pdf.make_indirect(
            Dictionary(
                Type=Name.Font,
                Subtype=Name.Type0,
                BaseFont=Name.GlyphLessFont,
                Encoding=Name("/Identity-H"),
                DescendantFonts=Array([cid_font]),
                ToUnicode=self.pdf.make_stream(TO_UNICODE),
            )
        )

    def _image(self, image: np.ndarray, jpeg_quality: int | None) -> pikepdf.Stream:
        if image.ndim == 3 and image.shape[2] == 4:  # transparent -> on white
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            image = (image[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        gray = image.ndim == 2
        h, w = image.shape[:2]
        if jpeg_quality:
            ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            assert ok
            raw, filt = data.tobytes(), Name.DCTDecode
        else:
            pixels = image if gray else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            raw, filt = zlib.compress(np.ascontiguousarray(pixels).tobytes(), 6), Name.FlateDecode
        stream = self.pdf.make_stream(raw)
        stream.Type, stream.Subtype = Name.XObject, Name.Image
        stream.Width, stream.Height, stream.BitsPerComponent = w, h, 8
        stream.ColorSpace = Name.DeviceGray if gray else Name.DeviceRGB
        stream.Filter = filt
        return stream

    def _text_layer(self, page: Page, scale: float, page_h: float) -> bytes:
        """Invisible words: for each, font size = box height, stretched to box width."""
        ops = ["BT", "3 Tr"]
        for line in page.lines:
            words = line.words or []
            if not words:
                continue
            for word in words:
                # A real space after every word (also the last one on a line): it
                # separates words for copy and search, and pdfium silently drops a
                # text piece of a single character (one-letter words like و).
                text = word.text + " "
                cids = _hex(reading_order(text)[::-1])  # display order, left to right
                if not cids:
                    continue
                tl, tr, _br, bl = (np.asarray(p, np.float64) * scale for p in word.box)
                # PDF y points up: flip, then measure the box
                bl_x, bl_y = bl[0], page_h - bl[1]
                dx, dy = tr[0] - tl[0], -(tr[1] - tl[1])
                width = math.hypot(dx, dy)
                height = math.hypot(bl[0] - tl[0], bl[1] - tl[1])
                if width < 0.5 or height < 0.5:
                    continue
                cos, sin = dx / width, dy / width
                n = len(cids) // 4
                stretch = 100 * width / (n * GLYPH_WIDTH / 1000 * height)
                ops.append(
                    f"/F0 {height:.2f} Tf {stretch:.2f} Tz "
                    f"{cos:.5f} {sin:.5f} {-sin:.5f} {cos:.5f} {bl_x:.2f} {bl_y:.2f} Tm "
                    f"<{cids}> Tj"
                )
        ops.append("ET")
        return "\n".join(ops).encode("ascii")

    def add_page(
        self, image: np.ndarray, page: Page, dpi: float = 300, jpeg_quality: int | None = 85
    ) -> None:
        """Add one page: `image` as read (same pixel size the OCR saw) and its result.

        `jpeg_quality`: None stores the image losslessly (best for screenshots, but
        large for scans)."""
        h, w = image.shape[:2]
        scale = 72.0 / dpi  # pixels -> points
        page_w, page_h = w * scale, h * scale
        content = f"q {page_w:.2f} 0 0 {page_h:.2f} 0 0 cm /Im0 Do Q\n".encode(
            "ascii"
        ) + self._text_layer(page, scale, page_h)
        pdf_page = pikepdf.Page(
            Dictionary(
                Type=Name.Page,
                MediaBox=Array([0, 0, page_w, page_h]),
                Resources=Dictionary(
                    XObject=Dictionary(Im0=self._image(image, jpeg_quality)),
                    Font=Dictionary(F0=self.font),
                ),
                Contents=self.pdf.make_stream(content),
            )
        )
        self.pdf.pages.append(pdf_page)

    def add_text_to_page(self, original: pikepdf.Page, page: Page, dpi: float) -> bool:
        """Copy a page of an input PDF unchanged and lay the invisible text over it
        (smaller and lossless, unlike re-storing the rendered image). `page` is the
        OCR of that page rendered at `dpi`. Returns False (and adds nothing) for
        pages whose coordinates are rotated or cropped: the caller then falls back
        to add_page."""
        box = [float(v) for v in original.mediabox]
        crop = [float(v) for v in original.obj.get("/CropBox", original.mediabox)]
        if int(original.obj.get("/Rotate", 0)) % 360 != 0 or crop != box:
            return False
        x0, y0, _x1, y1 = box
        scale = 72.0 / dpi
        layer = self._text_layer(page, scale, y1 - y0)
        if x0 or y0:  # the page's origin is not at (0, 0): shift the text with it
            layer = f"1 0 0 1 {x0:.2f} {y0:.2f} cm\n".encode("ascii") + layer
        self.pdf.pages.append(original)
        new = self.pdf.pages[-1]
        # (add_resource also handles resources the page inherits from its parent)
        name = new.add_resource(self.font, Name.Font, Name("/FOCR"), replace_existing=True)
        layer = layer.replace(b"/F0 ", str(name).encode("ascii") + b" ")
        # Wrap the original content in q/Q so its graphics state can't leak into ours.
        new.contents_add(self.pdf.make_stream(b"q\n"), prepend=True)
        new.contents_add(self.pdf.make_stream(b"Q\nq\n" + layer + b"\nQ\n"))
        return True

    def save(self, path: str | Path) -> None:
        self.pdf.save(str(path), compress_streams=True)


#: Inputs stored losslessly in the PDF (screenshots, drawings); others as JPEG.
LOSSLESS_SUFFIXES = {".png", ".bmp", ".webp"}


def image_dpi(path: str | Path, default: float = 300) -> float:
    """The resolution saved in an image file, or `default` if it has none."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            dpi = im.info.get("dpi")
    except OSError:
        return default
    if dpi and dpi[0] and float(dpi[0]) >= 50:  # ignore missing/bogus values like 1 or 72x
        return float(dpi[0])
    return default


def ocr_to_pdf(ocr, source: str | Path, out: str | Path, dpi: int = 300, **options):
    """OCR a file into a searchable PDF, one page at a time. Returns the Document."""
    from arabic_ocr.pipeline import Document

    pages = []

    def read(image: np.ndarray) -> Page:
        page = ocr.read_image(image, **options)
        pages.append(page)
        return page

    _write(source, read, out, dpi)
    return Document(pages, Path(source).name)


def pages_to_pdf(
    source: str | Path | None,
    pages: list[Page],
    out: str | Path,
    dpi: int = 300,
    images: list[np.ndarray] | None = None,
) -> None:
    """Write a searchable PDF from results you already have (no OCR is run).

    `pages` must be the OCR of `source` read at `dpi` (one per page, in order), or,
    with `images` instead of a source, of those images (e.g. a pasted screenshot).
    Used by the desktop app, where the user may have corrected the text first.
    """
    if images is not None:
        pdf = SearchablePdf()
        for image, page in zip(images, pages, strict=True):
            pdf.add_page(image, page, dpi=dpi, jpeg_quality=None)
        pdf.save(out)
        return
    it = iter(pages)
    _write(source, lambda _image: next(it), out, dpi)


def _write(source: str | Path, page_for, out: str | Path, dpi: int) -> None:
    """Go through the pages of `source`; `page_for(image)` gives each one's result."""
    from arabic_ocr.inputs import PDF_SUFFIXES, load_pages

    source = Path(source)
    suffix = source.suffix.lower()
    page_dpi = dpi if suffix in PDF_SUFFIXES else image_dpi(source)
    quality = None if suffix in LOSSLESS_SUFFIXES else 85
    pdf = SearchablePdf()
    original = pikepdf.open(source) if suffix in PDF_SUFFIXES else None
    try:
        for i, image in enumerate(load_pages(source, dpi)):
            page = page_for(image)
            # PDF input: keep the original page, add only the text layer.
            if original is None or not pdf.add_text_to_page(original.pages[i], page, dpi):
                pdf.add_page(image, page, dpi=page_dpi, jpeg_quality=quality)
        pdf.save(out)
    finally:
        if original is not None:
            original.close()
