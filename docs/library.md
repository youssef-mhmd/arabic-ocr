# Python library reference

Use arabic-ocr from your own Python code. It needs Python 3.13 or newer. The
library runs on ONNX Runtime; it never needs PyTorch.

```bash
uv add "arabic-ocr @ git+https://github.com/youssef-mhmd/arabic-ocr"
# or: pip install "git+https://github.com/youssef-mhmd/arabic-ocr"
```

The models are the `.onnx` files in the repository's `models/` folder, and they
are also attached to every release. Installed from git, the package can't see
that folder, so download `detector-v2.onnx` and `recognizer-v6b-int8.onnx` from
the [latest release](https://github.com/youssef-mhmd/arabic-ocr/releases/latest)
into a folder, and point the library at it:

```bash
export ARABIC_OCR_MODELS=/path/to/models        # Windows: set ARABIC_OCR_MODELS=C:\path\to\models
```

or pass the paths: `OCR(detector="…/detector-v2.onnx", recognizer="…/recognizer-v6b-int8.onnx")`.
From a clone of the repository (`uv run …`), the models are found automatically.

## Quick start

```python
from arabic_ocr import OCR

ocr = OCR()                          # loads the models once (about a second)
doc = ocr.read("page.png")           # also PDFs, multi-page TIFFs, NumPy images
print(doc.text)

for line in doc.lines:
    print(line.box, line.text, line.confidence)

doc.save("page.json")                # .txt .json .hocr/.html .xml (ALTO)
ocr.to_pdf("scan.pdf", "searchable.pdf")
```

## `OCR`

```python
OCR(detector=None, recognizer=None, threads=None)
```

Loads the text-line detector and the line recognizer. Reuse one `OCR` object for
many files: loading is the slow part. `threads` limits CPU threads (default: all
cores). It raises `FileNotFoundError` if a model is missing.

| Method | Returns |
|---|---|
| `read(source, dpi=300, **options)` | `Document`. `source`: a file path (image, multi-page TIFF, PDF) or a NumPy image (gray, BGR or BGRA, `uint8`) |
| `read_image(image, **options)` | `Page` for one NumPy image |
| `to_pdf(source, out, dpi=300, **options)` | Writes a searchable PDF, returns the `Document` |

Options (all optional):

| Option | Default | Meaning |
|---|---|---|
| `diacritics` | `True` | `False` removes tashkeel |
| `tatweel` | `True` | `False` removes tatweel (ـ) |
| `digits` | `"keep"` | `"western"` → 0–9, `"arabic"` → ٠–٩ |
| `deskew` | `True` | Straighten tilted pages |
| `denoise` | `False` | Remove speckles |
| `binarize` | `False` | Pure black/white before reading |
| `drop_noise` | `True` | Drop "lines" that are page noise (borders, stains) |
| `correct_words` | `False` | Word-list correction of dot slips (needs `models/lexicon.tsv.gz`) |

## Results

All text is in **logical Unicode order** (the order it is typed). Boxes are in
pixels of the image you passed in. A box is 4 corner points `(x, y)`: top-left,
top-right, bottom-right, bottom-left (NumPy `float32`, shape `(4, 2)`). Boxes can
be slightly rotated when the page was straightened. `confidence` is 0–1.

| Class | Attributes |
|---|---|
| `Document` | `pages`, `source` (file name); `lines`, `text`; `to_json()`, `to_hocr()`, `to_alto()`, `save(path)` |
| `Page` | `width`, `height`, `angle` (deskew rotation, degrees), `blocks`, `seconds`, `timings`; `lines`, `text` |
| `Block` | `lines`; `text`, `box`. A paragraph, heading or list item, in reading order (columns right to left) |
| `Line` | `text`, `confidence`, `box`, `words` |
| `Word` | `text`, `confidence`, `box`. Words are in reading order, right to left |

`Document.text` separates blocks with an empty line and pages with a form feed
(`\f`) on its own line. `Page.text` has no form feeds.

## JSON format (`schema: arabic-ocr/1`)

```json
{
  "schema": "arabic-ocr/1",
  "file": "page.png",
  "pages": [{
    "index": 0, "width": 2480, "height": 3508, "angle": 0.4, "seconds": 1.2,
    "text": "…",
    "blocks": [{
      "box": [[x, y], [x, y], [x, y], [x, y]],
      "text": "…",
      "lines": [{
        "text": "…", "confidence": 0.98, "box": [[…]],
        "words": [{"text": "…", "confidence": 0.99, "box": [[…]]}]
      }]
    }]
  }]
}
```

## Local HTTP API

`arabic-ocr serve` runs the same engine behind a local web API. For other
languages and apps, see the [CLI reference](cli.md#arabic-ocr-serve-local-http-api).
From Python you can also build the app yourself:

```python
from arabic_ocr.api import create_app
app = create_app()        # a FastAPI app; run it with uvicorn
```
