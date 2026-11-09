"""arabic-ocr: offline, cross-platform OCR for Arabic-script text.

This is the runtime package. It must never import torch; trained models are
shipped as ONNX files and run with ONNX Runtime.
"""

from importlib.metadata import version

__version__ = version("arabic-ocr")

__all__ = ["__version__"]
