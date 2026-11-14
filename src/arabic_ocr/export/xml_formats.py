"""hOCR and ALTO XML: the two standard OCR exchange formats.

    from arabic_ocr.export.xml_formats import to_hocr, to_alto
    html = to_hocr(document)        # hOCR 1.2 (an HTML file, readable in a browser)
    xml = to_alto(document)         # ALTO 4 (used by libraries and digitisation tools)

Both describe page -> blocks -> lines -> words with pixel boxes and confidences.
Boxes are axis-aligned here (both formats only have rectangles): the bounding box
of each rotated quad. Text is in logical order; lines and words come in reading
order (words right to left), and pages/blocks are marked right-to-left.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from arabic_ocr import __version__
from arabic_ocr.pipeline import Document

HOCR_NS = "http://www.w3.org/1999/xhtml"
ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"
ALTO_SCHEMA = "http://www.loc.gov/standards/alto/v4/alto-4-2.xsd"


def _bbox(quad: np.ndarray) -> tuple[int, int, int, int]:
    q = np.asarray(quad)
    x0, y0 = np.floor(q.min(axis=0)).astype(int)
    x1, y1 = np.ceil(q.max(axis=0)).astype(int)
    return max(0, int(x0)), max(0, int(y0)), int(x1), int(y1)


def _hocr_title(box: tuple[int, int, int, int], conf: float | None = None, extra: str = ""):
    title = "bbox {} {} {} {}".format(*box)
    if conf is not None:
        title += f"; x_wconf {round(conf * 100)}"
    return title + extra


def to_hocr(doc: Document) -> str:
    """hOCR 1.2: XHTML with ocr_page / ocr_carea / ocr_line / ocrx_word elements."""
    html = ET.Element("html", {"xmlns": HOCR_NS, "lang": "ar", "dir": "rtl"})
    head = ET.SubElement(html, "head")
    ET.SubElement(head, "title").text = doc.source or "arabic-ocr"
    ET.SubElement(
        head, "meta", {"http-equiv": "Content-Type", "content": "text/html; charset=utf-8"}
    )
    ET.SubElement(head, "meta", {"name": "ocr-system", "content": f"arabic-ocr {__version__}"})
    caps = "ocr_page ocr_carea ocr_par ocr_line ocrx_word ocrp_wconf"
    ET.SubElement(head, "meta", {"name": "ocr-capabilities", "content": caps})
    body = ET.SubElement(html, "body")

    for p, page in enumerate(doc.pages, 1):
        title = _hocr_title((0, 0, page.width, page.height), extra=f"; ppageno {p - 1}")
        if doc.source:
            title = f'image "{doc.source}"; ' + title
        page_el = ET.SubElement(
            body, "div", {"class": "ocr_page", "id": f"page_{p}", "title": title}
        )
        for b, block in enumerate(page.blocks, 1):
            bid = f"block_{p}_{b}"
            area = ET.SubElement(
                page_el,
                "div",
                {"class": "ocr_carea", "id": bid, "title": _hocr_title(_bbox(block.box))},
            )
            par = ET.SubElement(
                area, "p", {"class": "ocr_par", "id": f"par_{p}_{b}", "dir": "rtl",
                            "title": _hocr_title(_bbox(block.box))},
            )  # fmt: skip
            for n, line in enumerate(block.lines, 1):
                line_el = ET.SubElement(
                    par, "span", {"class": "ocr_line", "id": f"line_{p}_{b}_{n}",
                                  "title": _hocr_title(_bbox(line.box), line.confidence)},
                )  # fmt: skip
                words = line.words or []
                for w, word in enumerate(words, 1):
                    word_el = ET.SubElement(
                        line_el, "span", {"class": "ocrx_word", "id": f"word_{p}_{b}_{n}_{w}",
                                          "title": _hocr_title(_bbox(word.box), word.confidence)},
                    )  # fmt: skip
                    word_el.text = word.text
                    word_el.tail = " " if w < len(words) else None
                if not words:
                    line_el.text = line.text

    ET.indent(html)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE html>\n'
        + ET.tostring(html, encoding="unicode")
        + "\n"
    )


def _alto_box(el: ET.Element, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    el.set("HPOS", str(x0))
    el.set("VPOS", str(y0))
    el.set("WIDTH", str(x1 - x0))
    el.set("HEIGHT", str(y1 - y0))


def to_alto(doc: Document) -> str:
    """ALTO 4.2 XML, measured in pixels."""
    ET.register_namespace("", ALTO_NS)
    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    ns = f"{{{ALTO_NS}}}"
    alto = ET.Element(
        f"{ns}alto",
        {"{http://www.w3.org/2001/XMLSchema-instance}schemaLocation": f"{ALTO_NS} {ALTO_SCHEMA}"},
    )
    desc = ET.SubElement(alto, f"{ns}Description")
    ET.SubElement(desc, f"{ns}MeasurementUnit").text = "pixel"
    if doc.source:
        src = ET.SubElement(desc, f"{ns}sourceImageInformation")
        ET.SubElement(src, f"{ns}fileName").text = doc.source
    proc = ET.SubElement(ET.SubElement(desc, f"{ns}OCRProcessing", {"ID": "ocr_1"}),
                         f"{ns}ocrProcessingStep")  # fmt: skip
    sw = ET.SubElement(proc, f"{ns}processingSoftware")
    ET.SubElement(sw, f"{ns}softwareName").text = "arabic-ocr"
    ET.SubElement(sw, f"{ns}softwareVersion").text = __version__

    layout = ET.SubElement(alto, f"{ns}Layout")
    for p, page in enumerate(doc.pages, 1):
        page_el = ET.SubElement(
            layout, f"{ns}Page",
            {"ID": f"page_{p}", "PHYSICAL_IMG_NR": str(p),
             "WIDTH": str(page.width), "HEIGHT": str(page.height)},
        )  # fmt: skip
        space = ET.SubElement(page_el, f"{ns}PrintSpace")
        _alto_box(space, (0, 0, page.width, page.height))
        for b, block in enumerate(page.blocks, 1):
            block_el = ET.SubElement(
                space, f"{ns}TextBlock", {"ID": f"block_{p}_{b}", "LANG": "ar"}
            )
            _alto_box(block_el, _bbox(block.box))
            for n, line in enumerate(block.lines, 1):
                line_el = ET.SubElement(block_el, f"{ns}TextLine", {"ID": f"line_{p}_{b}_{n}"})
                _alto_box(line_el, _bbox(line.box))
                words = line.words or []
                if not words:  # no word boxes: the line is one String
                    attrs = {"CONTENT": line.text, "WC": f"{line.confidence:.3f}"}
                    s = ET.SubElement(line_el, f"{ns}String", attrs)
                    _alto_box(s, _bbox(line.box))
                for w, word in enumerate(words, 1):
                    s = ET.SubElement(
                        line_el, f"{ns}String",
                        {"ID": f"word_{p}_{b}_{n}_{w}", "CONTENT": word.text,
                         "WC": f"{word.confidence:.3f}"},
                    )  # fmt: skip
                    _alto_box(s, _bbox(word.box))
                    if w < len(words):
                        ET.SubElement(line_el, f"{ns}SP")
    ET.indent(alto)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(alto, encoding="unicode") + "\n"
