"""Desktop app tests. They run without a screen (Qt's "offscreen" platform).

Needs the gui extra (`uv sync --extra gui`); skipped otherwise.
"""

import os
import string
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings, Qt

from arabic_ocr.gui import i18n
from arabic_ocr.gui.model import FORMATS, Options, Result, even_words, with_text
from arabic_ocr.pipeline import Block, Line, Page, Word

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
needs_models = pytest.mark.skipif(
    not (MODELS / "detector-v2.onnx").exists(), reason="models not in models/"
)


@pytest.fixture(scope="module")
def app():
    from arabic_ocr.gui.app import make_app

    return make_app(["test"])


def box(x0, y0, x1, y1):
    return np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def make_page() -> Page:
    """Two blocks: two lines, then one line. Words right to left."""

    def line(text, y):
        words = text.split()
        w = 100 / len(words)
        # first word (reading order) at the right
        ws = [
            Word(t, 0.9, box(100 - (i + 1) * w, y, 100 - i * w, y + 10))
            for i, t in enumerate(words)
        ]
        return Line(text, 0.9, box(0, y, 100, y + 10), ws)

    return Page(
        200,
        200,
        0.0,
        [Block([line("كَتَبَ الولدُ", 0), line("الدرس ١٢٣", 20)]), Block([line("ـــالسلام", 50)])],
    )


# ----------------------------------------------------------------- translations
def test_every_text_has_both_languages_with_the_same_placeholders():
    fields = string.Formatter()
    for key, entry in i18n.STRINGS.items():
        assert set(entry) == set(i18n.LANGUAGES), key
        names = [{f[1] for f in fields.parse(entry[lang]) if f[1]} for lang in i18n.LANGUAGES]
        assert all(n == names[0] for n in names), key


def test_language_switch():
    i18n.set_language("ar")
    assert i18n.is_rtl() and i18n.t("open") == "فتح…"
    assert i18n.t("page_of", i=2, n=5) == "صفحة 2 من 5"
    i18n.set_language("xx")  # unknown -> English
    assert i18n.language() == "en" and not i18n.is_rtl()


# ------------------------------------------------------------ options and edits
def test_options_apply_to_lines_and_words():
    result = Result("a.png", None, 300, [make_page()])
    opts = Options(diacritics=False, tatweel=False, digits="western")
    assert result.page_text(0, opts) == "كتب الولد\nالدرس 123\n\nالسلام"
    doc = result.document(opts)
    assert [w.text for w in doc.pages[0].lines[0].words] == ["كتب", "الولد"]
    # the raw result is unchanged: options can be switched back
    assert result.page_text(0, Options()).startswith("كَتَبَ")


def test_edits_keep_boxes_when_word_count_is_the_same():
    page = with_text(make_page(), "كتب الولد\nالدرس ٤٥٦\n\nالسلام عليكم")
    lines = page.lines
    assert [ln.text for ln in lines] == ["كتب الولد", "الدرس ٤٥٦", "السلام عليكم"]
    assert [w.text for w in lines[1].words] == ["الدرس", "٤٥٦"]
    np.testing.assert_array_equal(lines[1].words[1].box, make_page().lines[1].words[1].box)
    # 1 OCR word became 2: the line box is shared out, first word on the right
    first, second = lines[2].words
    assert first.box[:, 0].min() >= second.box[:, 0].max() - 1e-3
    assert len(page.blocks) == 2


def test_edits_with_fewer_or_more_lines():
    fewer = with_text(make_page(), "سطر واحد")
    assert [ln.text for ln in fewer.lines] == ["سطر واحد"]
    assert len(fewer.blocks) == 1
    more = with_text(make_page(), "أ\nب\nج\nد\nهـ")
    assert [ln.text for ln in more.lines] == ["أ", "ب", "ج د هـ"]  # nothing is lost


def test_even_words_cover_the_line():
    words = even_words(box(0, 0, 90, 10), ["ab", "c"], 0.5)
    xs = [(w.box[:, 0].min(), w.box[:, 0].max()) for w in words]
    # weights: len + 1 for the space -> "ab" 3, "c" 2 of 90 px
    assert xs[0] == pytest.approx((36, 90)) and xs[1] == pytest.approx((0, 36))


def test_result_saves_every_format_with_edits(tmp_path):
    image = np.full((200, 200), 255, np.uint8)
    result = Result("pasted-1.png", None, 300, [make_page()], images=[image])
    result.edits[0] = "مصحح"
    for fmt, (ext, _) in FORMATS.items():
        path = tmp_path / result.output_name(fmt)
        assert path.suffix == ext
        result.save(path, fmt, Options())
        assert path.stat().st_size > 0
    assert (tmp_path / "pasted-1.txt").read_text(encoding="utf-8") == "مصحح\n"
    assert "مصحح" in (tmp_path / "pasted-1.json").read_text(encoding="utf-8")


# --------------------------------------------------------------------- settings
def test_settings_round_trip(app, tmp_path):
    from arabic_ocr.gui.settings import Settings

    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    s = Settings(language="ar", options=Options(False, False, "arabic"), output_folder="x")
    s.export_format = "alto"
    s.save(store)
    again = Settings.load(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    assert again == s
    store.setValue("digits", "nonsense")
    store.setValue("export_format", "doc")
    bad = Settings.load(store)
    assert bad.options.digits == "keep" and bad.export_format == "txt"


# ------------------------------------------------------------ the whole window
@pytest.fixture
def window(app, tmp_path):
    from arabic_ocr.gui.main_window import MainWindow
    from arabic_ocr.gui.settings import Settings

    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    w = MainWindow(Settings(language="en", output_folder=str(tmp_path / "out")), store)
    w.show()
    yield w
    w.close()


@needs_models
def test_window_reads_edits_and_exports(window, tmp_path):
    from arabic_ocr.gui.app import SAMPLE_TEXT
    from arabic_ocr.gui.main_window import SAMPLE

    window.add_files([SAMPLE, tmp_path / "notes.docx"])  # .docx is skipped
    assert window.wait_idle()
    (entry,) = window.entries.values()
    assert entry.status == "done"
    truth = SAMPLE_TEXT.read_text(encoding="utf-8").split("\n")[0]
    assert window.text.toPlainText().startswith(truth[:20])
    assert len(window.view._items) == len(entry.result.pages[0].lines)

    # Clicking line 2's box puts the cursor on line 2 of the text
    window._on_line_clicked(1)
    assert window.text.textCursor().block().text().strip() == entry.result.pages[0].lines[1].text

    # Edit the text: the export uses it
    window.text.setPlainText("نص مصحح")
    assert entry.result.edits == {0: "نص مصحح"}
    window.settings.export_format = "txt"
    (path,) = window.export_all()
    assert path.read_text(encoding="utf-8") == "نص مصحح\n"

    # Options apply at once to unedited pages
    entry.result.edits.clear()
    window.settings.options = Options(diacritics=False, digits="western")
    window.refresh_text()
    assert "400" in window.text.toPlainText() and "٤٠٠" not in window.text.toPlainText()


@needs_models
def test_window_pasted_image_and_bad_file(window, tmp_path):
    from PySide6.QtGui import QGuiApplication, QImage

    from arabic_ocr.gui.app import SAMPLE_TEXT  # noqa: F401
    from arabic_ocr.gui.main_window import SAMPLE

    QGuiApplication.clipboard().setImage(QImage(str(SAMPLE)))
    window.paste_image()
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")
    window.add_files([bad])
    assert window.wait_idle()
    pasted, broken = window.entries.values()
    assert pasted.status == "done" and pasted.result.images is not None
    assert broken.status == "failed" and "Cannot read" in broken.error
    window.settings.export_format = "pdf"
    written = window.export_all()
    assert [p.name for p in written] == ["pasted-1.pdf"]


def test_window_language_and_cancel(window):
    window.toggle_language()
    assert window.layoutDirection() == Qt.LayoutDirection.RightToLeft
    assert window.act_open.text() == "فتح…"
    window.toggle_language()
    assert window.layoutDirection() == Qt.LayoutDirection.LeftToRight
    # Queue then cancel: nothing is left running
    window.add_image(np.full((50, 50), 255, np.uint8))
    window.cancel_all()
    assert not window._busy()


@needs_models
def test_self_test_passes(app, tmp_path):
    from arabic_ocr.gui.app import self_test

    report = tmp_path / "report.txt"
    assert self_test(report) == 0
    assert "SELF-TEST PASSED" in report.read_text(encoding="utf-8")
