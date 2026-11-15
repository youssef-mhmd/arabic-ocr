"""The app's settings, remembered between runs.

Stored with QSettings: the registry on Windows, a .plist file on macOS, an .ini
file on Linux (~/.config/arabic-ocr/arabic-ocr.conf).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QLocale, QSettings, QStandardPaths

from arabic_ocr.gui.i18n import LANGUAGES
from arabic_ocr.gui.model import FORMATS, Options


def default_language() -> str:
    """Arabic for computers set to Arabic, Persian or Urdu; otherwise English."""
    lang = QLocale.system().language()
    arabic = {QLocale.Language.Arabic, QLocale.Language.Persian, QLocale.Language.Urdu}
    return "ar" if lang in arabic else "en"


def default_output_folder() -> str:
    docs = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
    return str(Path(docs or Path.home()) / "Arabic OCR")


@dataclass
class Settings:
    language: str = field(default_factory=default_language)
    options: Options = field(default_factory=Options)
    output_folder: str = field(default_factory=default_output_folder)
    export_format: str = "txt"  # for "Export all"
    last_folder: str = ""  # where the Open dialog starts

    @classmethod
    def load(cls, store: QSettings) -> Settings:
        s = cls()

        def get(key: str, default, kind=str):
            value = store.value(key, default)
            if kind is bool:  # .ini files return "true"/"false" strings
                return value if isinstance(value, bool) else str(value).lower() == "true"
            return kind(value)

        s.language = get("language", s.language)
        if s.language not in LANGUAGES:
            s.language = "en"
        s.options = Options(
            diacritics=get("diacritics", True, bool),
            tatweel=get("tatweel", True, bool),
            digits=get("digits", "keep"),
        )
        if s.options.digits not in ("keep", "western", "arabic"):
            s.options.digits = "keep"
        s.output_folder = get("output_folder", s.output_folder)
        s.export_format = get("export_format", "txt")
        if s.export_format not in FORMATS:
            s.export_format = "txt"
        s.last_folder = get("last_folder", "")
        return s

    def save(self, store: QSettings) -> None:
        store.setValue("language", self.language)
        store.setValue("diacritics", self.options.diacritics)
        store.setValue("tatweel", self.options.tatweel)
        store.setValue("digits", self.options.digits)
        store.setValue("output_folder", self.output_folder)
        store.setValue("export_format", self.export_format)
        store.setValue("last_folder", self.last_folder)
        store.sync()


def open_store() -> QSettings:
    return QSettings("arabic-ocr", "arabic-ocr")
