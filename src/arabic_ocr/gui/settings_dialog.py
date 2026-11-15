"""The Settings dialog: language, output options, output folder."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from arabic_ocr.gui.i18n import LANGUAGES, t
from arabic_ocr.gui.model import FORMATS, Options
from arabic_ocr.gui.settings import Settings

DIGITS = ("keep", "western", "arabic")


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("settings_title"))
        self.setMinimumWidth(460)

        self.language = QComboBox()
        for code, name in LANGUAGES.items():
            self.language.addItem(name, code)
        self.language.setCurrentIndex(list(LANGUAGES).index(settings.language))

        self.diacritics = QCheckBox(t("keep_diacritics"))
        self.diacritics.setChecked(settings.options.diacritics)
        self.tatweel = QCheckBox(t("keep_tatweel"))
        self.tatweel.setChecked(settings.options.tatweel)
        self.digits = QComboBox()
        for d in DIGITS:
            self.digits.addItem(t(f"digits_{d}"), d)
        self.digits.setCurrentIndex(DIGITS.index(settings.options.digits))

        self.folder = QLineEdit(settings.output_folder)
        browse = QPushButton(t("browse"))
        browse.clicked.connect(self._browse)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(browse)

        self.format = QComboBox()
        for key, (ext, label) in FORMATS.items():
            self.format.addItem(f"{label} ({ext})", key)
        self.format.setCurrentIndex(list(FORMATS).index(settings.export_format))

        form = QFormLayout()
        form.addRow(t("language"), self.language)
        form.addRow("", self.diacritics)
        form.addRow("", self.tatweel)
        form.addRow(t("digits"), self.digits)
        form.addRow(t("output_folder"), folder_row)
        form.addRow(t("export_format"), self.format)

        note = QLabel(t("options_note"))
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, t("output_folder"), self.folder.text())
        if folder:
            self.folder.setText(folder)

    def apply(self, settings: Settings) -> None:
        """Copy the dialog's choices into `settings`."""
        settings.language = self.language.currentData()
        settings.options = Options(
            diacritics=self.diacritics.isChecked(),
            tatweel=self.tatweel.isChecked(),
            digits=self.digits.currentData(),
        )
        settings.output_folder = self.folder.text().strip() or settings.output_folder
        settings.export_format = self.format.currentData()
