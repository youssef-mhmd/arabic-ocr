# PyInstaller recipe for the desktop app. Run through installers/build.py:
#
#     uv run --extra gui --group build python -m installers.build
#
# Makes one folder, dist/arabic-ocr/, with two programs that share the same
# libraries and models:
#   arabic-ocr-gui   the desktop app (no console window on Windows)
#   arabic-ocr       the command-line tool (same as `uv run arabic-ocr`)
# On macOS the folder becomes "dist/Arabic OCR.app".
#
# A Python interpreter, every library and the two ONNX models are inside, so
# the computer the app is installed on needs no Python at all.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent  # noqa: F821 (SPECPATH is set by PyInstaller)
sys.path.insert(0, str(ROOT / "src"))
from arabic_ocr.pipeline import DETECTOR, RECOGNIZER  # noqa: E402

ICONS = ROOT / "installers" / "icons"
GUI = ROOT / "src" / "arabic_ocr" / "gui"
VERSION = __import__("importlib.metadata").metadata.version("arabic-ocr")

datas = [
    # The shipped models only (models/ also holds older versions)
    (str(ROOT / "models" / DETECTOR), "models"),
    (str(ROOT / "models" / RECOGNIZER), "models"),
    (str(GUI / "sample.png"), "arabic_ocr/gui"),
    (str(GUI / "sample.txt"), "arabic_ocr/gui"),
    (str(GUI / "icon.png"), "arabic_ocr/gui"),
    (str(ROOT / "LICENSE"), "."),
    *copy_metadata("arabic-ocr"),  # for arabic_ocr.__version__
]

# Never bundled: training code and libraries the runtime does not use.
excludes = [
    "torch", "training", "tkinter", "matplotlib", "IPython", "pytest",
    "uharfbuzz", "freetype", "pyarrow", "yaml", "setuptools", "pkg_resources",
]  # fmt: skip

#: Large files the app never uses: OpenCV's video codecs (it only reads images),
#: and Qt's software OpenGL (the app draws with plain widgets, not OpenGL).
UNUSED = ("opencv_videoio_ffmpeg", "opengl32sw")


def trim(analysis):
    analysis.binaries = [b for b in analysis.binaries if not Path(b[0]).name.startswith(UNUSED)]

gui = Analysis(  # noqa: F821
    [str(ROOT / "installers" / "entry_gui.py")],
    pathex=[str(ROOT / "src")],
    datas=datas,
    excludes=excludes,
)
cli = Analysis(  # noqa: F821
    [str(ROOT / "installers" / "entry_cli.py")],
    pathex=[str(ROOT / "src")],
    datas=datas,
    # `arabic-ocr serve` imports these inside functions
    hiddenimports=[*collect_submodules("uvicorn"), "arabic_ocr.api", "multipart"],
    excludes=[*excludes, "PySide6", "shiboken6"],
)

trim(gui)
trim(cli)

icon = str(ICONS / ("arabic-ocr.ico" if sys.platform == "win32" else "arabic-ocr.icns"))
gui_exe = EXE(  # noqa: F821
    PYZ(gui.pure),  # noqa: F821
    gui.scripts,
    exclude_binaries=True,
    name="arabic-ocr-gui",
    console=False,
    icon=icon,
    upx=False,
)
cli_exe = EXE(  # noqa: F821
    PYZ(cli.pure),  # noqa: F821
    cli.scripts,
    exclude_binaries=True,
    name="arabic-ocr",
    console=True,
    icon=icon,
    upx=False,
)
coll = COLLECT(  # noqa: F821
    gui_exe,  # first: on macOS the first program is the one the app opens
    gui.binaries,
    gui.datas,
    cli_exe,
    cli.binaries,
    cli.datas,
    name="arabic-ocr",
    upx=False,
)

if sys.platform == "darwin":
    app = BUNDLE(  # noqa: F821
        coll,
        name="Arabic OCR.app",
        icon=str(ICONS / "arabic-ocr.icns"),
        bundle_identifier="io.github.youssef-mhmd.arabic-ocr",
        version=VERSION,
        info_plist={
            "CFBundleName": "Arabic OCR",
            "CFBundleDisplayName": "Arabic OCR",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "LSMinimumSystemVersion": "14.0",
            "NSHighResolutionCapable": True,
            "NSHumanReadableCopyright": "Apache-2.0",
            # "Open With > Arabic OCR" for images and PDFs
            "CFBundleDocumentTypes": [
                {
                    "CFBundleTypeName": "Image or PDF",
                    "CFBundleTypeRole": "Viewer",
                    "LSHandlerRank": "Alternate",
                    "LSItemContentTypes": [
                        "public.png", "public.jpeg", "public.tiff", "com.microsoft.bmp",
                        "org.webmproject.webp", "com.adobe.pdf",
                    ],  # fmt: skip
                }
            ],
        },
    )
