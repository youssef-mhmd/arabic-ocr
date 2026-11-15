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

---

## 7. Real test sets for Persian and Urdu, and page borders

`training/make_real_pages.py printed --site fa|ur` builds Persian and Urdu sets from their Wikisources, the same way as the Arabic one. Each page is tagged by its style after looking at it: *naskh* (print) or *nastaliq* (the flowing Persian/Urdu script; handwritten-looking lithographs count here). Only naskh counts for the PRD's v1 target (≤ 4%).

| set | pages | CER (no diacritics) |
|---|---|---|
| Persian naskh | 12 | 9.2% |
| Urdu naskh (one 1920s letterpress book) | 11 | 35.6% |
| nastaliq (fa + ur) | 5 | 54–78% (not a v1 target) |

**Page frames.** The Urdu book prints a ruled frame around the text; the vertical frame line next to each text line was read as an extra alef. `preprocess.remove_rules` paints long, thin vertical lines with the paper colour before detection: Urdu 41.4% → 35.6%, nothing else changed. Three traps on the way, each caught by measuring:
- coloured panels on screenshots are long *and thick*: only thin lines may be removed;
- a page **shadow** made whole regions "ink" under one global threshold: a *local* (adaptive) threshold fixed it;
- on a small image "10% of the page height" is shorter than a tall alef: a rule must also be at least 4 typical letter heights long.

## 8. Real training lines, and the robot policy

The best way to close the gap to real (especially old) print is to train on real lines. `training/make_real_lines.py` does this from proofread Wikisource scans: run our OCR on a page, match each found line to the transcription line it reads most like, and keep clear matches with the human-checked text as the label.

It works (e.g. 15 of 17 lines kept on a page of «كليلة ودمنة»), but after a few thousand requests Wikimedia refused further downloads: its **User-Agent policy** requires automated clients to identify themselves with a contact address (e-mail or web page). Collection is paused until the project has such a contact to put there.

## 9. Verse layout

Persian and Arabic poetry is printed as rows of two half-lines (right half first). The column finder saw two columns and read all right halves, then all left halves; on the Shahnameh and Masnavi pages that made the page text 48–55% "wrong" although most words were read correctly. `layout.is_verse` now recognises verse: almost every line has a partner across the gap, both sides have as many lines, almost no line ends short, the two halves share a baseline almost exactly (median offset ≤ 0.07 line heights), and half-lines are short (≤ 10 line heights). Verse is then read row by row.

The baseline test was found by measuring: a first version without it also turned 5 of 20 synthetic two-column *prose* pages into "verse" (their rows drift by 0.08–0.29 line heights; verse rows by 0.03–0.06). Result: Persian naskh pages 19.2% → 10.1% CER; everything else unchanged, including two-column reading order.

## 10. Fine-tuning on real lines: the label-end trap

`training/make_real_lines.py` collected 18,592 real training lines (Arabic 9,265, Persian 9,323; Urdu only 4: its few non-test books are Nastaliq) from 267 Wikisource books, none of them in a test set. Recognizer-v6 was fine-tuned from v5 on these (repeated 4×) plus 200k synthetic lines.

v6 improved Persian (10.1% → 6.0%) and Urdu (35.6% → 29.1%) but made **Arabic printed pages twice as bad** (1.77% → 3.41%): it invented text at the *ends* of lines ("على وجه" → "على وا الا"). The cause was label noise. A printed line and a transcription line were accepted as a match at up to 30% CER, which also happens when the transcriber broke the line one word earlier or later than the print. Then the image and the label disagree exactly at the line end, and the model learns to make something up there. 41% of the lines had first or last words that disagreed with v5's reading.

`make_real_lines --clean` keeps a line only if v5 reads it within 20% CER *and* its first and last words agree (58% kept). Recognizer-v6b, trained on those, is better than v5 on every page-level set:

| real pages, CER (no diacritics) | v5 | v6 | **v6b** |
|---|---|---|---|
| Arabic printed (60) | 1.77% | 3.41% | **1.40%** |
| Persian naskh (32) | 10.1% | 6.0% | **6.8%** |
| Urdu naskh (11) | 35.6% | 29.1% | **33.9%** |
| screenshots, UI fonts (100) | 0.91% | 0.93% | **0.85%** |

One line set got worse: the older OpenITI *dev* lines (4.91% → 5.77%), while its *held-out* sister set improved (2.92% → 2.87%). v6b (int8) is the default model.

Lesson: with automatically matched labels, check the *edges* of each sample, not only the overall match.
