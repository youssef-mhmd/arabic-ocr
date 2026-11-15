"""The main window.

    ┌ toolbar: Open · Open folder · Paste image · Copy · Export · Export all · Settings ┐
    │ Files (queue)  │  page image with line boxes   │  text (editable)              │
    │ a.png   done   │                               │                               │
    │ b.pdf   40%    │                               │                               │
    └ status: what is being read · progress bar · Cancel ───────────────────────────────┘

Files are read in the background (worker.py) in the order they were added. Click
a file to see its result. Click a line box to jump to its text, or put the cursor
on a text line to see its box. Text edits are kept and used by every export.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QGuiApplication,
    QImage,
    QKeySequence,
    QTextCursor,
    QTextFormat,
    QTextOption,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStyle,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from arabic_ocr import __version__
from arabic_ocr.gui import i18n
from arabic_ocr.gui.i18n import LANGUAGES, t
from arabic_ocr.gui.image_view import ImageView, from_qimage
from arabic_ocr.gui.model import FORMATS, Result
from arabic_ocr.gui.settings import Settings, open_store
from arabic_ocr.gui.settings_dialog import SettingsDialog
from arabic_ocr.gui.worker import Job, start_worker, stop_worker
from arabic_ocr.inputs import SUPPORTED, find_inputs

#: The sample image shown by "Try a sample" (also used by the self-test).
SAMPLE = Path(__file__).with_name("sample.png")


@dataclass
class Entry:
    """One row of the file list."""

    job: Job
    item: QListWidgetItem
    status: str = "waiting"  # waiting | reading | done | failed | cancelled
    pages_done: int = 0
    pages_total: int = 0
    result: Result | None = None
    error: str = ""


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings | None = None, store=None) -> None:
        super().__init__()
        self.store = store if store is not None else open_store()
        self.settings = settings or Settings.load(self.store)
        i18n.set_language(self.settings.language)
        self.entries: dict[int, Entry] = {}
        self._ids = itertools.count(1)
        self._pasted = 0
        self._page = 0  # page shown, in the selected entry
        self._image_cache: tuple[tuple[int, int], np.ndarray] | None = None
        self._updating_text = False

        self.worker, self.thread = start_worker()
        self.worker.models_loading.connect(self._on_models_loading)
        self.worker.models_failed.connect(self._on_models_failed)
        self.worker.started.connect(self._on_started)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)

        self._build()
        self.setAcceptDrops(True)
        self.resize(1280, 800)
        self.retranslate()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        style = self.style()
        icon = style.standardIcon
        P = QStyle.StandardPixmap

        def action(icon_id, slot, shortcut=None) -> QAction:
            a = QAction(icon(icon_id), "", self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(shortcut)
            return a

        keys = QKeySequence.StandardKey
        self.act_open = action(P.SP_DialogOpenButton, self.open_files, keys.Open)
        self.act_folder = action(P.SP_DirOpenIcon, self.open_folder)
        self.act_paste = action(P.SP_FileDialogContentsView, self.paste_image)
        self.act_sample = action(P.SP_FileDialogInfoView, self.open_sample)
        self.act_copy = action(P.SP_FileDialogDetailedView, self.copy_text)
        self.act_export = action(P.SP_DialogSaveButton, self.export_current, keys.Save)
        self.act_export_all = action(P.SP_DriveHDIcon, self.export_all)
        self.act_settings = action(P.SP_FileDialogListView, self.open_settings)
        self.act_language = QAction("", self)
        self.act_language.triggered.connect(self.toggle_language)
        self.act_about = action(P.SP_MessageBoxInformation, self.about)

        # Ctrl+V pastes an image, unless the text box has focus (then it pastes text).
        paste = QAction(self)
        paste.setShortcut(QKeySequence.StandardKey.Paste)
        paste.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        paste.triggered.connect(self.paste_image)
        self.addAction(paste)

        bar = QToolBar(self)
        bar.setObjectName("main")
        bar.setMovable(False)
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        for a in (self.act_open, self.act_folder, self.act_paste):
            bar.addAction(a)
        bar.addSeparator()
        for a in (self.act_copy, self.act_export, self.act_export_all):
            bar.addAction(a)
        bar.addSeparator()
        for a in (self.act_settings, self.act_language, self.act_sample, self.act_about):
            bar.addAction(a)
        self.addToolBar(bar)
        self.toolbar = bar

        # Left: the file list
        self.files_label = QLabel()
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._on_select)
        self.btn_remove = QPushButton()
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear = QPushButton()
        self.btn_clear.clicked.connect(self.clear_list)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 0, 6, 0)
        lv.addWidget(self.files_label)
        lv.addWidget(self.list, 1)
        row = QHBoxLayout()
        row.addWidget(self.btn_remove)
        row.addWidget(self.btn_clear)
        lv.addLayout(row)

        # Middle: the image, with page navigation
        self.view = ImageView()
        self.view.line_clicked.connect(self._on_line_clicked)
        self.hint = QLabel()
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: gray; font-size: 15px; padding: 24px;")
        self.stack = QStackedWidget()
        self.stack.addWidget(self.hint)
        self.stack.addWidget(self.view)
        self.btn_prev = QToolButton()
        self.btn_prev.setIcon(icon(P.SP_ArrowBack))
        self.btn_prev.clicked.connect(lambda: self.show_page(self._page - 1))
        self.btn_next = QToolButton()
        self.btn_next.setIcon(icon(P.SP_ArrowForward))
        self.btn_next.clicked.connect(lambda: self.show_page(self._page + 1))
        self.page_label = QLabel()
        self.btn_fit = QToolButton()
        self.btn_fit.clicked.connect(self.view.fit)
        nav = QHBoxLayout()
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.page_label)
        nav.addWidget(self.btn_next)
        nav.addStretch(1)
        nav.addWidget(self.btn_fit)
        middle = QWidget()
        mv = QVBoxLayout(middle)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.addLayout(nav)
        mv.addWidget(self.stack, 1)

        # Right: the text. Always right-to-left: it is Arabic-script text.
        self.text = QPlainTextEdit()
        self.text.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        option = self.text.document().defaultTextOption()
        option.setTextDirection(Qt.LayoutDirection.RightToLeft)
        option.setAlignment(Qt.AlignmentFlag.AlignRight)
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.text.document().setDefaultTextOption(option)
        font = QFont(self.text.font())
        font.setPointSize(max(14, font.pointSize() + 4))
        self.text.setFont(font)
        self.text.textChanged.connect(self._on_text_edited)
        self.text.cursorPositionChanged.connect(self._on_cursor_moved)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(middle)
        split.addWidget(self.text)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 2)
        split.setSizes([220, 620, 440])
        self.setCentralWidget(split)

        # Status bar: message, progress, cancel
        self.status_label = QLabel()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        self.btn_cancel = QPushButton()
        self.btn_cancel.clicked.connect(self.cancel_all)
        self.btn_cancel.setVisible(False)
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.btn_cancel)

        # Drops on the image and text areas open files too.
        for w in (self.view.viewport(), self.text.viewport(), self.hint):
            w.setAcceptDrops(True)
            w.installEventFilter(self)

    def retranslate(self) -> None:
        """Set every visible word in the current language, and the layout direction."""
        rtl = i18n.is_rtl()
        QApplication.setLayoutDirection(
            Qt.LayoutDirection.RightToLeft if rtl else Qt.LayoutDirection.LeftToRight
        )
        self.setWindowTitle(f"{t('app_name')} {__version__}")
        for a, key, tip in (
            (self.act_open, "open", "open_tip"),
            (self.act_folder, "open_folder", None),
            (self.act_paste, "paste", "paste_tip"),
            (self.act_sample, "sample", None),
            (self.act_copy, "copy", "copy_tip"),
            (self.act_export, "export", "export_tip"),
            (self.act_export_all, "export_all", "export_all_tip"),
            (self.act_settings, "settings", None),
            (self.act_about, "about", None),
        ):
            a.setText(t(key))
            a.setToolTip(t(tip) if tip else t(key))
        other = "en" if i18n.language() == "ar" else "ar"
        self.act_language.setText(LANGUAGES[other])
        self.files_label.setText(t("files"))
        self.btn_remove.setText(t("remove"))
        self.btn_clear.setText(t("clear"))
        self.btn_cancel.setText(t("cancel"))
        self.btn_fit.setText(t("fit"))
        self.btn_prev.setToolTip(t("prev_page"))
        self.btn_next.setToolTip(t("next_page"))
        self.hint.setText(t("drop_hint"))
        for entry in self.entries.values():
            self._update_item(entry)
        self._update_nav()
        if not self._busy():
            self.status_label.setText(t("status_ready"))

    # --------------------------------------------------------------- adding files
    def add_files(self, paths) -> None:
        """Queue files and folders (folders: every supported file in them)."""
        files, skipped = [], []
        for p in paths:
            p = Path(p)
            if p.is_dir():
                files += find_inputs(p)
            elif p.suffix.lower() in SUPPORTED and p.is_file():
                files.append(p)
            else:
                skipped.append(p.name)
        for f in files:
            self._queue(Job(next(self._ids), f.name, path=f))
        if skipped:
            self.status_label.setText(t("unsupported", names=", ".join(skipped)))

    def add_image(self, image: np.ndarray, name: str | None = None) -> int:
        """Queue an image (e.g. pasted). Returns its job id."""
        if name is None:
            self._pasted += 1
            name = f"pasted-{self._pasted}.png"
        job = Job(next(self._ids), name, image=image)
        self._queue(job)
        return job.id

    def _queue(self, job: Job) -> None:
        item = QListWidgetItem()
        entry = Entry(job, item)
        self.entries[job.id] = entry
        self.list.addItem(item)
        self._update_item(entry)
        if self.list.currentRow() < 0:
            self.list.setCurrentRow(self.list.count() - 1)
        self.worker.submit.emit(job)
        self._update_progress()

    def open_files(self) -> None:
        exts = " ".join(f"*{s}" for s in sorted(SUPPORTED))
        paths, _ = QFileDialog.getOpenFileNames(
            self, t("open"), self.settings.last_folder, f"{t('open_tip')} ({exts})"
        )
        if paths:
            self.settings.last_folder = str(Path(paths[0]).parent)
            self.add_files(paths)

    def open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, t("open_folder"), self.settings.last_folder)
        if folder:
            self.settings.last_folder = folder
            self.add_files([folder])

    def open_sample(self) -> None:
        self.add_files([SAMPLE])

    def paste_image(self) -> None:
        clip = QGuiApplication.clipboard()
        mime = clip.mimeData()
        if mime is not None and mime.hasUrls():
            local = [u.toLocalFile() for u in mime.urls() if u.isLocalFile()]
            if local:
                self.add_files(local)
                return
        image = clip.image()
        if image.isNull():
            self.status_label.setText(t("no_image"))
            return
        self.add_image(from_qimage(image))

    # ---------------------------------------------------------------- drag & drop
    def _accepts(self, mime) -> bool:
        return mime.hasUrls() or mime.hasImage()

    def _drop(self, mime) -> None:
        if mime.hasUrls():
            local = [u.toLocalFile() for u in mime.urls() if u.isLocalFile()]
            if local:
                self.add_files(local)
                return
        if mime.hasImage():
            image = QImage(mime.imageData())
            if not image.isNull():
                self.add_image(from_qimage(image))

    def dragEnterEvent(self, event) -> None:
        if self._accepts(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        self._drop(event.mimeData())
        event.acceptProposedAction()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        kind = event.type()
        if kind in (QEvent.Type.DragEnter, QEvent.Type.DragMove, QEvent.Type.Drop):
            mime = event.mimeData()
            if mime.hasUrls() or (mime.hasImage() and not mime.hasText()):
                if kind == QEvent.Type.Drop:
                    self._drop(mime)
                event.acceptProposedAction()
                return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------ worker signals
    def _on_models_loading(self) -> None:
        self.status_label.setText(t("loading_models"))

    def _on_models_failed(self, error: str) -> None:
        QMessageBox.critical(self, t("error"), t("models_missing", error=error))

    def _on_started(self, job_id: int, pages: int) -> None:
        entry = self.entries.get(job_id)
        if entry is None:
            return
        entry.status, entry.pages_total = "reading", pages
        self._update_item(entry)
        self._update_progress()

    def _on_progress(self, job_id: int, page: int, pages: int) -> None:
        entry = self.entries.get(job_id)
        if entry is None:
            return
        entry.pages_done, entry.pages_total = page, pages
        self._update_item(entry)
        self._update_progress()

    def _on_finished(self, job_id: int, result: Result) -> None:
        entry = self.entries.get(job_id)
        if entry is None:
            return
        entry.status, entry.result = "done", result
        self._update_item(entry)
        if self.current() is entry:
            self.show_page(0)
        self._update_progress()
        if not self._busy():
            seconds = sum(p.seconds for p in result.pages)
            lines = sum(len(p.lines) for p in result.pages)
            self.status_label.setText(
                t("status_done", name=result.name, pages=len(result.pages), lines=lines,
                  seconds=seconds)
            )  # fmt: skip

    def _on_failed(self, job_id: int, error: str) -> None:
        entry = self.entries.get(job_id)
        if entry is None:
            return
        entry.status, entry.error = "failed", error
        self._update_item(entry)
        if self.current() is entry:
            self.show_page(0)
        self._update_progress()
        self.status_label.setText(f"{entry.job.name}: {error}")

    # ----------------------------------------------------------------- the queue
    def _busy(self) -> bool:
        return any(e.status in ("waiting", "reading") for e in self.entries.values())

    def _update_item(self, entry: Entry) -> None:
        if entry.status == "reading" and entry.pages_total:
            pct = int(100 * entry.pages_done / entry.pages_total)
            status = f"{t('reading')} {pct}%"
        else:
            status = t(entry.status)
        if entry.result and entry.result.edits:
            status += f" · {t('edited')}"
        name = entry.job.name
        if entry.job.path is None and name.startswith("pasted-"):
            name = t("pasted", n=name.removeprefix("pasted-").removesuffix(".png"))
        entry.item.setText(f"{name}\n{status}")
        entry.item.setToolTip(entry.error or str(entry.job.path or name))

    def _update_progress(self) -> None:
        active = [e for e in self.entries.values() if e.status in ("waiting", "reading")]
        self.progress.setVisible(bool(active))
        self.btn_cancel.setVisible(bool(active))
        reading = next((e for e in active if e.status == "reading"), None)
        if reading is None:
            if active:
                self.progress.setRange(0, 0)  # busy, no page count yet
            return
        n = max(1, reading.pages_total)
        self.progress.setRange(0, n)
        self.progress.setValue(reading.pages_done)
        self.status_label.setText(
            t("status_reading", name=reading.job.name, i=min(reading.pages_done + 1, n), n=n,
              left=len(active) - 1)
        )  # fmt: skip

    def cancel_all(self) -> None:
        ids = [i for i, e in self.entries.items() if e.status in ("waiting", "reading")]
        self.worker.cancel(ids)
        for i in ids:
            self.entries[i].status = "cancelled"
            self._update_item(self.entries[i])
        self._update_progress()
        self.status_label.setText(t("status_ready"))

    def remove_selected(self) -> None:
        entry = self.current()
        if entry is None:
            return
        self.worker.cancel([entry.job.id])
        del self.entries[entry.job.id]
        self.list.takeItem(self.list.row(entry.item))
        self._update_progress()
        self._on_select(self.list.currentRow())

    def clear_list(self) -> None:
        self.worker.cancel(list(self.entries))
        self.entries.clear()
        self.list.clear()
        self._update_progress()
        self._on_select(-1)
        self.status_label.setText(t("status_ready"))

    # -------------------------------------------------------------- showing results
    def current(self) -> Entry | None:
        item = self.list.currentItem()
        return next((e for e in self.entries.values() if e.item is item), None)

    def _on_select(self, _row: int) -> None:
        self.show_page(0)

    def page_image(self, entry: Entry, i: int) -> np.ndarray:
        """Page i's image: kept for pasted images, loaded again for files."""
        result = entry.result
        if result.images is not None:
            return result.images[i]
        key = (entry.job.id, i)
        if self._image_cache and self._image_cache[0] == key:
            return self._image_cache[1]
        from arabic_ocr.inputs import load_pages

        image = next(itertools.islice(load_pages(result.source, result.dpi), i, None))
        self._image_cache = (key, image)
        return image

    def show_page(self, i: int) -> None:
        entry = self.current()
        self._updating_text = True
        try:
            if entry is None or entry.result is None:
                self._page = 0
                self.view.show_page(None, [])
                self.stack.setCurrentWidget(self.hint)
                if entry is None:
                    self.hint.setText(t("drop_hint"))
                elif entry.status == "failed":
                    self.hint.setText(f"{t('failed')}: {entry.error}")
                else:
                    self.hint.setText(f"{entry.job.name}: {t(entry.status)}…")
                self.text.setPlainText("")
                self.text.setReadOnly(True)
                return
            result = entry.result
            self._page = max(0, min(i, len(result.pages) - 1))
            page = result.pages[self._page]
            self.view.show_page(
                self.page_image(entry, self._page), [line.box for line in page.lines]
            )
            self.stack.setCurrentWidget(self.view)
            self.text.setReadOnly(False)
            self.text.setPlainText(result.page_text(self._page, self.settings.options))
        finally:
            self.text.setExtraSelections([])
            self._updating_text = False
            self._update_nav()

    def _update_nav(self) -> None:
        entry = self.current()
        n = len(entry.result.pages) if entry and entry.result else 0
        self.page_label.setText(t("page_of", i=self._page + 1, n=n) if n else "")
        self.btn_prev.setEnabled(self._page > 0)
        self.btn_next.setEnabled(self._page < n - 1)
        for w in (self.btn_prev, self.btn_next, self.page_label):
            w.setVisible(n > 1)
        has_result = bool(entry and entry.result)
        self.act_copy.setEnabled(has_result)
        self.act_export.setEnabled(has_result)

    def refresh_text(self) -> None:
        """Show the current page's text again (after the options changed)."""
        entry = self.current()
        if entry and entry.result:
            self._updating_text = True
            self.text.setPlainText(entry.result.page_text(self._page, self.settings.options))
            self._updating_text = False

    def _on_text_edited(self) -> None:
        if self._updating_text:
            return
        entry = self.current()
        if entry is None or entry.result is None:
            return
        result = entry.result
        text = self.text.toPlainText()
        if text == self.settings.options.apply(result.pages[self._page].text):
            result.edits.pop(self._page, None)  # edited back to the original
        else:
            result.edits[self._page] = text
        self._update_item(entry)

    def _text_line_blocks(self) -> list[int]:
        """Block numbers of the non-empty lines of the text box, in order."""
        doc = self.text.document()
        return [b for b in range(doc.blockCount()) if doc.findBlockByNumber(b).text().strip()]

    def _highlight_block(self, number: int) -> None:
        """Shade one line of the text box (the line whose box is selected)."""
        selection = QTextEdit.ExtraSelection()
        color = QColor(255, 140, 0)
        color.setAlpha(50)
        selection.format.setBackground(color)
        selection.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
        cursor = QTextCursor(self.text.document().findBlockByNumber(number))
        selection.cursor = cursor
        self.text.setExtraSelections([selection])

    def _on_cursor_moved(self) -> None:
        if self._updating_text:
            return
        block = self.text.textCursor().blockNumber()
        blocks = self._text_line_blocks()
        if block in blocks:
            self.view.select(blocks.index(block))
            self._highlight_block(block)
        else:
            self.text.setExtraSelections([])

    def _on_line_clicked(self, index: int) -> None:
        blocks = self._text_line_blocks()
        if index < len(blocks):
            block = self.text.document().findBlockByNumber(blocks[index])
            cursor = self.text.textCursor()
            cursor.setPosition(block.position())
            self._updating_text = True
            self.text.setTextCursor(cursor)
            self._updating_text = False
            self.text.centerCursor()
            self._highlight_block(blocks[index])

    # ------------------------------------------------------------- copy & export
    def copy_text(self) -> None:
        entry = self.current()
        if entry is None or entry.result is None:
            return
        result = entry.result
        texts = [result.page_text(i, self.settings.options) for i in range(len(result.pages))]
        QGuiApplication.clipboard().setText("\n\n".join(texts))
        self.status_label.setText(t("copied"))

    def export_current(self) -> None:
        entry = self.current()
        if entry is None or entry.result is None:
            return
        filters = [f"{label} (*{ext})" for ext, label in FORMATS.values()]
        keys = list(FORMATS)
        start = Path(self.settings.output_folder) / entry.result.output_name("txt")
        path, chosen = QFileDialog.getSaveFileName(
            self, t("export"), str(start), ";;".join(filters)
        )
        if not path:
            return
        fmt = keys[filters.index(chosen)] if chosen in filters else "txt"
        suffix = Path(path).suffix.lower()
        by_ext = {ext: k for k, (ext, _) in FORMATS.items()}
        if suffix in by_ext:  # the typed extension wins over the filter
            fmt = by_ext[suffix]
        elif not suffix:
            path += FORMATS[fmt][0]
        try:
            entry.result.save(path, fmt, self.settings.options)
        except Exception as e:
            QMessageBox.warning(self, t("error"), t("export_failed", error=e))
            return
        self.status_label.setText(t("saved", path=path))

    def export_all(self) -> list[Path]:
        """Save every finished result to the output folder. Returns the files written."""
        done = [e.result for e in self.entries.values() if e.result is not None]
        if not done:
            QMessageBox.information(self, t("export_all"), t("nothing_to_export"))
            return []
        folder = Path(self.settings.output_folder)
        fmt = self.settings.export_format
        written = []
        try:
            folder.mkdir(parents=True, exist_ok=True)
            used: set[str] = set()
            for result in done:
                name = result.output_name(fmt)
                stem, ext = Path(name).stem, Path(name).suffix
                n = 2
                while name.lower() in used:  # a.png and a.pdf both -> a.txt
                    name, n = f"{stem}-{n}{ext}", n + 1
                used.add(name.lower())
                result.save(folder / name, fmt, self.settings.options)
                written.append(folder / name)
        except Exception as e:
            QMessageBox.warning(self, t("error"), t("export_failed", error=e))
            return written
        self.status_label.setText(t("saved_all", n=len(written), folder=folder))
        return written

    # -------------------------------------------------------------- settings etc.
    def open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            language = self.settings.language
            dialog.apply(self.settings)
            self.settings.save(self.store)
            if self.settings.language != language:
                i18n.set_language(self.settings.language)
                self.retranslate()
            self.refresh_text()

    def toggle_language(self) -> None:
        self.settings.language = "en" if i18n.language() == "ar" else "ar"
        i18n.set_language(self.settings.language)
        self.settings.save(self.store)
        self.retranslate()

    def about(self) -> None:
        QMessageBox.about(self, t("about"), t("about_text", version=__version__))

    def closeEvent(self, event) -> None:
        self.settings.save(self.store)
        self.worker.cancel(list(self.entries))
        stop_worker(self.worker, self.thread)
        super().closeEvent(event)

    def wait_idle(self, timeout_ms: int = 120_000) -> bool:
        """Process events until every queued file is read (for tests and the self-test)."""
        from PySide6.QtCore import QDeadlineTimer, QEventLoop

        deadline = QDeadlineTimer(timeout_ms)
        while self._busy():
            if deadline.hasExpired():
                return False
            loop = QEventLoop()
            QTimer.singleShot(20, loop.quit)
            loop.exec()
        return True
