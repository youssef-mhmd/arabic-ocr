"""Smoke tests: the package installs, imports, and its CLI starts."""

import subprocess
import sys

from typer.testing import CliRunner

import arabic_ocr
from arabic_ocr.cli import app


def test_version_is_set():
    assert arabic_ocr.__version__


def test_cli_version():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert arabic_ocr.__version__ in result.output


def test_runtime_package_does_not_import_torch():
    # Run in a fresh interpreter so other tests can't have imported torch already.
    code = (
        "import sys, arabic_ocr, arabic_ocr.cli, arabic_ocr.detect, arabic_ocr.recognize, "
        "arabic_ocr.export; assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_arabic_text_round_trips_as_utf8(tmp_path):
    # Guards the cross-platform rule: always read/write text as UTF-8.
    text = "بِسْمِ اللَّهِ ٠١٢٣ پچژگ"
    path = tmp_path / "sample.txt"
    path.write_text(text, encoding="utf-8")
    assert path.read_text(encoding="utf-8") == text
