"""hOCR and ALTO output: well-formed, complete, boxes and text correct."""

import xml.etree.ElementTree as ET

import numpy as np

from arabic_ocr.export.xml_formats import ALTO_NS, to_alto, to_hocr
from arabic_ocr.pipeline import Block, Document, Line, Page, Word


def _box(x0, y0, x1, y1):
    return np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def _doc() -> Document:
    w1 = Word("مرحبا", 0.97, _box(300, 10, 400, 40))
    w2 = Word("2026", 0.88, _box(200, 10, 290, 40))
    line = Line("مرحبا 2026", 0.92, _box(200, 10, 400, 40), [w1, w2])
    bare = Line("سطر & <بلا> كلمات", 0.5, _box(150, 60, 400, 90), [])
    page = Page(420, 120, 0.0, [Block([line]), Block([bare])])
    return Document([page, Page(420, 120, 0.0, [])], "صفحة.png")


def test_hocr():
    root = ET.fromstring(to_hocr(_doc()).split("\n", 2)[2])  # skip XML decl + doctype
    ns = {"h": "http://www.w3.org/1999/xhtml"}
    pages = root.findall(".//h:div[@class='ocr_page']", ns)
    assert len(pages) == 2
    assert 'image "صفحة.png"' in pages[0].get("title")
    words = root.findall(".//h:span[@class='ocrx_word']", ns)
    assert [w.text for w in words] == ["مرحبا", "2026"]
    assert words[0].get("title") == "bbox 300 10 400 40; x_wconf 97"
    lines = root.findall(".//h:span[@class='ocr_line']", ns)
    assert lines[1].text == "سطر & <بلا> كلمات"  # escaped and read back intact
    assert root.get("dir") == "rtl"


def test_alto():
    root = ET.fromstring(to_alto(_doc()).encode("utf-8"))
    ns = {"a": ALTO_NS}
    assert root.find("a:Description/a:MeasurementUnit", ns).text == "pixel"
    pages = root.findall(".//a:Page", ns)
    assert [p.get("WIDTH") for p in pages] == ["420", "420"]
    strings = root.findall(".//a:String", ns)
    assert [s.get("CONTENT") for s in strings] == ["مرحبا", "2026", "سطر & <بلا> كلمات"]
    s = strings[0]
    assert (s.get("HPOS"), s.get("VPOS"), s.get("WIDTH"), s.get("HEIGHT")) == (
        "300",
        "10",
        "100",
        "30",
    )
    assert s.get("WC") == "0.970"
    assert len(root.findall(".//a:SP", ns)) == 1  # one space between the two words
    assert len(root.findall(".//a:TextBlock", ns)) == 2
