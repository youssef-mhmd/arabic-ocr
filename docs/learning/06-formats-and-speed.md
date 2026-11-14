# 06: Output formats, the API, and making it fast

Note 05 covers how a page becomes text. This note covers M4's work on what happens around that: reading files, writing standard formats, serving the engine to other programs, and making it fast and light.

---

## 1. Inputs

`arabic_ocr/inputs.py` turns any input into page images:
- **images** (PNG, JPEG, BMP, WebP) are one page; **TIFF** files can hold many pages;
- **PDF** pages are rendered at 300 DPI with pdfium (the engine inside Chrome), one at a time, so a 500-page PDF never sits in memory all at once;
- **folders and patterns** (`scans/*.tif`) become a sorted list, so batches always run in the same order.

---

## 2. Searchable PDF: the right-to-left trap

A searchable PDF shows the original scan, with the recognized text written on top in **invisible** ink (PDF "text render mode 3"). Viewers don't draw it, but you can select, search and copy it.

The font is a tiny "glyphless" font we generate: every glyph is empty, and every character is stored as its Unicode number, with a table that maps it back to Unicode for copying. So any Arabic, Persian or Urdu letter works without embedding a real Arabic font.

The trap: **PDF text always runs left to right.** A viewer rebuilds right-to-left text from the *positions* of characters. We tried three ways of writing Arabic words and measured them with three different PDF engines on a real scanned page (192 words):

| how each word is written | pdfium (Chrome, Edge) | MuPDF | pdfminer |
|---|---|---|---|
| **display order** (as seen, left to right; numbers kept) | 1.7% off, 188 found | 1.2% off, 189 found | reversed |
| logical order with a mirrored text direction (Tesseract's way) | 7.4% off, 189 found, brackets flipped | reversed, 0 found | 0.0% off |
| logical order, normal direction | reversed, 0 found | | |

Display order wins in both *viewer* engines; pdfium, MuPDF and pdfminer all disagree with each other about Arabic, and pdfminer even reverses PDFs that Edge prints itself. Adobe Reader and macOS Preview still need a manual check.

A second surprise: pdfium silently **drops a word made of a single character** (و, م, ب). Writing a space after every word fixes it.

---

## 3. hOCR and ALTO

Two standard XML formats that libraries and digitisation tools read. Both describe pages → blocks → lines → words with boxes and confidences. Our ALTO output **validates against the official ALTO 4.2 schema** (checked once against loc.gov's XSD).

---

## 4. Library and HTTP API

```python
from arabic_ocr import OCR
result = OCR().read("scan.pdf")     # also images, TIFFs, NumPy arrays
result.save("scan.json")            # .txt .json .hocr .xml
OCR().to_pdf("scan.pdf", "searchable.pdf")
```

`arabic-ocr serve` starts a small web server on this computer only (127.0.0.1). Other programs send a file to `POST /ocr` and get JSON, text, hOCR, ALTO or a searchable PDF back. FastAPI also generates interactive documentation at `/docs`.

---

## 5. Memory: the arena that kept growing

ONNX Runtime keeps a memory **arena**: a pool of memory it re-uses between runs. That is great when every run has the same size. Ours never do (every page and every line is a different size), so the pool kept growing: 600+ MB after 20 pages. Turning the arena off keeps memory flat at ~145 MB for any number of pages, and costs only ~2% speed.

How it was found: memory after every 4 pages, with the arena on (592 → 676 MB) and off (139 → 146 MB).

---

## 6. Speed: measure fairly, then quantize the right part

**Measuring fairly.** The laptop's speed changed by ~70% over the day (other programs, power and heat): the same code measured 2.10 s per page in the morning and 3.6 s in the evening. A single timing comparison across runs is therefore meaningless. The fix: run both versions **in one process, alternating page by page** (A/B), so both see the same conditions.

**Profiling first.** ONNX Runtime's profiler showed where the recognizer spends its time: **convolutions 74%**, LSTM 25%. That decided what to optimise.

**Quantization** stores numbers as 8-bit integers instead of 32-bit floats; CPUs multiply those much faster.
- *Dynamic* quantization (the easy kind) mainly converted the LSTM, and ONNX Runtime's int8 LSTM was **14× slower**. A good reminder that "int8 = faster" is not automatic.
- *Static* quantization of the **convolutions only** (calibrated on 300 real line images) made the recognizer **1.8× faster** (3.61 → 1.91 s per page in the same A/B run) with no accuracy change (±0.03 points on every test set). It is now the default (`models/recognizer-v4-int8.onnx`, made by `training/quantize.py`).
- The same trick on the **detector** cost accuracy (screenshot recall 95.0% → 94.5%), even with its first and last layers left in float, so the detector stays float.

Rule kept throughout: never ship a faster model without re-measuring accuracy on the real test sets.
