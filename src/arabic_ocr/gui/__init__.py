"""The desktop app (PySide6 / Qt 6). Start it with `arabic-ocr-gui`.

    app.py              entry point, --self-test
    main_window.py      the window: file list, page image, text
    image_view.py       page image with line boxes
    worker.py           OCR on a background thread
    model.py            results, output options, text edits, export (no Qt)
    settings.py         settings remembered between runs
    settings_dialog.py  the Settings dialog
    i18n.py             English and Arabic words

Needs the `gui` extra: `uv sync --extra gui` (or `pip install "arabic-ocr[gui]"`).
"""
