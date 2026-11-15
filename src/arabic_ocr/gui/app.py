"""Start the desktop app.

    arabic-ocr-gui                      # open the window
    arabic-ocr-gui scan.pdf page.png    # ... and read these files
    arabic-ocr-gui --self-test [report.txt]

--self-test checks that an installed app works without anyone watching: it opens
the window, reads the bundled sample image in the background thread, exports
every format (txt, json, searchable PDF, hOCR, ALTO) and checks the results. It
prints a report (and writes it to report.txt if given; on Windows the app has no
console) and exits with code 0 if everything passed. The release workflow runs
it on fresh machines to test the installers.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import QEvent, QSettings
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

ICON = Path(__file__).with_name("icon.png")
#: What sample.png says (logical order), for the self-test.
SAMPLE_TEXT = Path(__file__).with_name("sample.txt")


class App(QApplication):
    """Also opens files sent by the system (macOS "Open With", dropping on the Dock)."""

    def __init__(self, argv) -> None:
        super().__init__(argv)
        self.window = None
        self._pending: list[str] = []

    def event(self, e) -> bool:
        if e.type() == QEvent.Type.FileOpen:
            path = e.file()
            if self.window is not None:
                self.window.add_files([path])
            else:
                self._pending.append(path)
            return True
        return super().event(e)


def make_app(argv=None) -> App:
    app = QApplication.instance() or App(argv if argv is not None else sys.argv)
    app.setApplicationName("arabic-ocr")
    app.setOrganizationName("arabic-ocr")
    app.setApplicationDisplayName("Arabic OCR")
    if ICON.exists():
        app.setWindowIcon(QIcon(str(ICON)))
    return app


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--self-test":
        return self_test(Path(args[1]) if len(args) > 1 else None)
    if args and args[0] in ("--version", "-V"):
        from arabic_ocr import __version__

        print(f"arabic-ocr-gui {__version__}")
        return 0

    from arabic_ocr.gui.main_window import MainWindow

    app = make_app()
    window = MainWindow()
    if isinstance(app, App):
        app.window = window
        args += app._pending
    window.show()
    if args:
        window.add_files(args)
    return app.exec()


def cer(truth: str, guess: str) -> float:
    """Character error rate: edits needed to turn guess into truth, per character."""
    prev = list(range(len(guess) + 1))
    for i, a in enumerate(truth, 1):
        cur = [i]
        for j, b in enumerate(guess, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / max(1, len(truth))


def self_test(report_path: Path | None = None, max_cer: float = 0.05) -> int:
    lines: list[str] = []
    ok = True

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        lines.append(f"{'PASS' if passed else 'FAIL'}  {name}{': ' + detail if detail else ''}")

    try:
        from arabic_ocr import __version__
        from arabic_ocr.gui.main_window import SAMPLE, MainWindow
        from arabic_ocr.gui.model import FORMATS
        from arabic_ocr.gui.settings import Settings

        lines.append(f"arabic-ocr-gui {__version__} self-test, {sys.platform}")
        app = make_app([sys.argv[0]])
        lines.append(f"Qt platform: {app.platformName()}")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # A throwaway settings file: the self-test never touches the user's settings.
            store = QSettings(str(tmp / "settings.ini"), QSettings.Format.IniFormat)
            settings = Settings(output_folder=str(tmp / "out"))
            window = MainWindow(settings, store)
            window.show()
            start = time.perf_counter()
            window.add_files([SAMPLE])
            finished = window.wait_idle(180_000)
            check("read the sample in the background", finished)
            entry = next(iter(window.entries.values()))
            check("sample read without error", entry.status == "done", entry.error)
            if entry.result is not None:
                text = entry.result.page_text(0, settings.options)
                truth = SAMPLE_TEXT.read_text(encoding="utf-8").strip()
                score = cer(truth, text.strip())
                check(
                    "recognized text", score <= max_cer,
                    f"CER {score:.2%} in {time.perf_counter() - start:.1f} s",
                )  # fmt: skip
                for fmt in FORMATS:
                    settings.export_format = fmt
                    files = window.export_all()
                    size = files[0].stat().st_size if files else 0
                    check(f"export {fmt}", size > 0, f"{files[0].name if files else '-'}, {size} B")
                pdf = tmp / "out" / entry.result.output_name("pdf")
                if pdf.exists():
                    import pypdfium2 as pdfium

                    doc = pdfium.PdfDocument(str(pdf))
                    found = doc[0].get_textpage().get_text_range()
                    doc.close()
                    check("searchable PDF has a text layer", len(found.strip()) > 0)
            window.close()
    except Exception as e:  # report anything unexpected as a failure
        import traceback

        check("no crash", False, f"{type(e).__name__}: {e}")
        lines.append(traceback.format_exc())

    lines.append("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    report = "\n".join(lines)
    if report_path is not None:
        report_path.write_text(report + "\n", encoding="utf-8")
    if sys.stdout is not None:  # None in a Windows app without a console
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            print(report)
        except (AttributeError, OSError):
            pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
