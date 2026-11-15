"""arabic-ocr: offline, cross-platform OCR for Arabic-script text.

    from arabic_ocr import OCR
    result = OCR().read("page.png")        # also PDFs, multi-page TIFFs, image arrays
    print(result.text)
    for line in result.lines:
        print(line.box, line.text, line.confidence)
    result.save("page.json")               # also .txt, .hocr, .xml (ALTO)
    OCR().to_pdf("scan.pdf", "searchable.pdf")

This is the runtime package. It must never import torch; trained models are
shipped as ONNX files and run with ONNX Runtime.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("arabic-ocr")
except PackageNotFoundError:  # a build that lost the package metadata
    __version__ = "0+unknown"

__all__ = ["OCR", "Block", "Document", "Line", "Page", "Word", "__version__"]


def __getattr__(name: str):
    # Loaded on first use, so `import arabic_ocr` (and `arabic-ocr --version`) stays
    # fast and doesn't load ONNX Runtime or OpenCV.
    if name in ("OCR", "Block", "Document", "Line", "Page", "Word"):
        from arabic_ocr import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module 'arabic_ocr' has no attribute {name!r}")
