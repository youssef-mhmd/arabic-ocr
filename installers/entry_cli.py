"""Start-up script of the packaged command-line tool (see arabic-ocr.spec)."""

from arabic_ocr.cli import app

if __name__ == "__main__":
    app(prog_name="arabic-ocr")
