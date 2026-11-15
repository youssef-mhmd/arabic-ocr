"""Start-up script of the packaged desktop app (see arabic-ocr.spec)."""

import sys

from arabic_ocr.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
