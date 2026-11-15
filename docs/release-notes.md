First release of **arabic-ocr**: offline OCR for Arabic, Persian and Urdu text in
images and PDFs, with its own detection and recognition models. It runs on your
computer's CPU, and nothing is sent over the internet.

## Download

| System | File |
|---|---|
| Windows 10/11 (64-bit) | `arabic-ocr-1.0.0-windows-x64-setup.exe` |
| macOS 14+ on Apple Silicon (M1 or newer) | `arabic-ocr-1.0.0-macos-arm64.dmg` |
| Ubuntu 22.04+ / Debian 12+ (64-bit) | `arabic-ocr_1.0.0_amd64.deb` (install with `sudo apt install ./arabic-ocr_1.0.0_amd64.deb`) |

No Python is needed. The installers are **not code-signed** yet: Windows
SmartScreen and macOS Gatekeeper warn the first time. The
[install guide](https://github.com/youssef-mhmd/arabic-ocr/blob/main/docs/install.md)
shows how to continue. Check downloads against `SHA256SUMS.txt`.

## What's in it

- **Desktop app** (English and Arabic interface): open files or folders, drag and
  drop, or paste a screenshot. See the page with a box around each line next to
  its text, correct the text, then copy or export. Files are read in the
  background, with progress and Cancel.
- **Exports:** plain text, JSON (lines and words with boxes and confidence),
  **searchable PDF** (the original pages with invisible, selectable text), hOCR,
  ALTO XML.
- **Options:** keep or remove diacritics (tashkeel) and tatweel; digits as
  printed, Western (0–9) or Arabic-Indic (٠–٩).
- **Inputs:** PNG, JPEG, TIFF (all pages), BMP, WebP, PDF.
- **Command line** (`arabic-ocr`), with batch mode for folders, and a **local HTTP
  API** (`arabic-ocr serve`). There is also a Python library (`from arabic_ocr import OCR`).
- Multi-column pages are read right to left; text comes out in logical order.

## Accuracy (character error rate, diacritics ignored)

| Content | CER |
|---|---|
| Printed Arabic, scanned books | 1.40% (1.50% with diacritics) |
| Screenshots, common UI and document fonts | 0.85% |
| Persian, modern print: scans / digital | 3.07% / 0.18% |
| Urdu, modern print (digital) | 0.27% |

An A4 page at 300 DPI takes about 1.5 s on 4 CPU cores. Memory use is about 150 MB.

## Known limits

- Not supported yet: Nastaliq (calligraphic Urdu), old lithographed prints,
  handwriting, and photos of signs.
- Calligraphic and monospace screen fonts (Aldhabi, Arabic Typesetting, Courier
  New) and very small text (≤ 14 px) are read less well.
- Not built for Intel Macs or Linux on ARM.

## Models and licenses

`detector-v2.onnx` (text-line detector, 14.6 MB) and `recognizer-v6b-int8.onnx`
(line recognizer, 12.7 MB) are attached for use with the Python library. The
models are released under **CC BY-SA 4.0**: they were trained on Wikipedia and
Wikisource text (by their contributors) and the Tanzil Quran text (tanzil.net).
The code is Apache-2.0. See `NOTICE` for details and third-party licenses.
