# PRD: Arabic OCR Engine (**arabic-ocr**)

| | |
|---|---|
| **Status** | Draft v0.1 |
| **Date** | 2026-11-09 |
| **Owner** | jousef |
| **License** | Open source (Apache-2.0 for code, see §11) |

---

## 1. Summary

arabic-ocr is a cross-platform (Windows, macOS, Linux) OCR engine that reads **only Arabic-script text**. We design and train our own neural network models with standard ML libraries. We do **not** use existing OCR engines (Tesseract, EasyOCR, PaddleOCR) or their pretrained weights. People can use it as a **command-line tool**, a **desktop app**, a **Python library**, or a **local HTTP API**.

Because the owner is new to machine learning, the project is built in small, testable phases. Each phase produces something that works, and the docs explain the ML concepts as they come up (see §13, Glossary).

---

## 2. Goals and non-goals

### Goals
1. Read Arabic text from images and PDFs and output it in correct right-to-left reading order.
2. Support **diacritics (tashkeel)**, **Arabic-Indic digits** (٠–٩), and **Persian/Urdu letters** (پ چ ژ گ ک ی ٹ ڈ ڑ ں ے ۀ …).
3. Handle printed documents and screenshots first (v1), then photos/scene text, then handwriting.
4. Run fully **offline on a normal computer with no GPU**.
5. Use **one language end to end** so a single developer can maintain it.

### Non-goals (for now)
- Latin, Hebrew, or other scripts. Latin text in an image is ignored or marked as unknown.
- Mobile apps (iOS/Android) and in-browser OCR. The ONNX model format leaves the door open to both later.
- Translation, spell-correction using large language models, or cloud services.

---

## 3. Users and use cases

| User | Use case |
|---|---|
| Student / researcher | Digitize scanned Arabic books and papers, including classical texts with tashkeel |
| Office worker | Turn scanned forms/PDFs into searchable PDFs |
| Developer | Call the library or local API from their own app |
| Everyday user | Drag a screenshot into the desktop app and copy the text |

---

## 4. Technology decision

### Chosen: **Python, managed with `uv`**

| Layer | Tool | Why |
|---|---|---|
| Project/env management | **uv** | Fast, reproducible, lockfile, same commands on all 3 OSes |
| Model training | **PyTorch** | Industry standard, best docs and tutorials for learners |
| Model runtime (inference) | **ONNX Runtime** | Small, fast on CPU, identical on Win/Mac/Linux; end users never need to install PyTorch |
| Image processing | **OpenCV**, **NumPy**, **Pillow** | Standard, cross-platform wheels |
| Arabic text rendering (synthetic data) | **uharfbuzz** + **freetype-py** | Correct Arabic shaping (joined letter forms, ligatures, diacritic placement) |
| Corpus files (synthetic data) | **pyarrow** | Reads the Parquet text extracts Wikimedia publishes (data generation only, not shipped) |
| PDF input / output | **pypdfium2** (read/render), **pikepdf** (write text layer) | Permissive licenses, cross-platform |
| CLI | **Typer** | Simple, auto-generated help |
| Local HTTP API | **FastAPI** + **uvicorn** | Simple, typed, auto docs |
| Desktop GUI | **PySide6 (Qt 6)** | Native look on all 3 OSes, good right-to-left text support, LGPL |
| Packaging | **PyInstaller**, built by **GitHub Actions** on each OS | One-click installers: `.exe`/`.msi`, `.dmg`, `.AppImage`/`.deb` |
| Tests / lint | **pytest**, **ruff** | Standard |

**Python version and platform notes (decided 2026-11-09, M0):**
- Python **3.13** (pinned in `.python-version`), chosen for the widest wheel coverage across all dependencies.
- ONNX Runtime stopped publishing macOS Intel wheels after 1.23.x. `pyproject.toml` gives Intel Macs onnxruntime 1.23.x and every other platform the latest. Exported models must use an ONNX opset that 1.23 supports.
- PyTorch has no macOS Intel wheels for Python 3.13, so training (`train` group) is not supported on Intel Macs. That's fine: training runs on GPU machines (Kaggle/Colab/Linux/Windows).

### Why not Rust, C++, or Go?
They would make a slightly smaller, faster app. But training must happen in Python anyway, so we would need **two languages**, two sets of build tools, and twice the debugging, which is too much for someone learning ML. ONNX Runtime already provides fast C++ execution under Python. If we ever need a tiny native binary, we can rewrite **only the runtime** in Rust later and keep the same `.onnx` model files.

---

## 5. Hardware: what you need and who can run it

### To **build/train** the models (developer)
- **Training needs a GPU.** You do **not** need to buy a high-end PC.
  - **Free options:** Kaggle Notebooks (about 30 GPU hours/week) and Google Colab (free tier) are enough for v1.
  - **Cheap paid option:** Rent GPUs on RunPod, Vast.ai, or Lambda (about $0.20–0.80/hour). A full v1 training run is estimated at 10–30 GPU hours.
  - A local NVIDIA GPU with 8 GB+ VRAM (RTX 3060 or better) is convenient but optional.
- **Your normal PC is fine for everything else:** writing code, generating data, running tests, and running the finished OCR.

### To **use** the finished tool (end users)
- **Works on any normal computer with no GPU:**
  - Windows 10/11 (64-bit)
  - macOS 12+ (Intel and Apple Silicon)
  - Linux (Ubuntu 22.04+ or equivalent, x86-64; ARM64 is a stretch goal)
- Minimum: 4 GB RAM and a dual-core CPU. Recommended: 8 GB RAM.
- Optional GPU acceleration (DirectML on Windows, CoreML on Mac, CUDA on NVIDIA) is a stretch goal.

---

## 6. How the OCR works (architecture)

```
 Image / PDF page
       │
       ▼
 ┌──────────────┐   deskew, denoise, binarize (classic image processing)
 │ Preprocess   │
 └──────┬───────┘
        ▼
 ┌──────────────┐   Neural net #1: finds where text lines are
 │ Text detector│   (DBNet-style segmentation, small CNN backbone)
 └──────┬───────┘
        ▼
 ┌──────────────┐   groups lines into blocks/columns,
 │ Layout       │   sorts them right-to-left, top-to-bottom
 └──────┬───────┘
        ▼
 ┌──────────────┐   Neural net #2: reads each line image → characters
 │ Recognizer   │   (CNN + Transformer/BiLSTM encoder + CTC decoding)
 └──────┬───────┘
        ▼
 ┌──────────────┐   Unicode normalization, digit handling,
 │ Postprocess  │   optional lexicon/language-model correction
 └──────┬───────┘
        ▼
 txt / JSON / searchable PDF / hOCR / ALTO
```

### Key Arabic-specific design choices
- **Line-level recognition, not letter-by-letter.** Arabic letters join and change shape by position, so cutting a word into letters is unreliable. The recognizer reads a whole line and outputs a character sequence (the CTC method).
- **Output in logical Unicode order.** The model outputs characters in the order they are typed, not their visual order. The text then displays correctly in any RTL-aware app.
- **Diacritics are separate output characters.** Each tashkeel mark (U+064B–U+0652, plus superscript alef U+0670) is its own symbol. Users can choose to keep or strip them in the output.
- **Character set (about 150 symbols):** Arabic letters, hamza forms, tashkeel, tatweel, Arabic-Indic and Eastern Arabic-Indic digits (٠–٩, ۰–۹), Western digits 0–9, Arabic punctuation (، ؛ ؟ ٪), and Persian/Urdu letters.
- **Mixed digits.** Numbers inside Arabic text run left to right. The text stays in logical order, so the Unicode bidirectional algorithm displays numbers correctly.

---

## 7. Data strategy

ML models learn from examples, so data is the biggest part of this project.

### 7.1 Synthetic data (main source for v1)
We generate millions of labeled line images by rendering real Arabic text with real fonts. The label is free because we know what text we rendered.
- **Text sources:** Arabic, Persian and Urdu Wikipedia (CC BY-SA 4.0), Arabic Wikisource for classical and diacritized text (CC BY-SA 4.0), and the Tanzil Quran text for fully diacritized text (CC BY 3.0 + Tanzil terms). All are listed in `training/configs/corpora.yaml` and `data/LICENSES.md`.
  - *Decided 2026-11-09 (M1):* Tashkeela (GPLv2) and OpenITI (CC BY-NC-SA 4.0) were dropped from training. Their licenses conflict with releasing the model for any use. They may only be used for private evaluation.
- **Fonts (open licenses, mostly SIL OFL):** Amiri, Noto Naskh Arabic, Noto Kufi Arabic, Noto Nastaliq Urdu, Scheherazade New, Lateef, Cairo, Tajawal, Almarai, Markazi Text, Reem Kufi, Vazirmatn, IBM Plex Sans Arabic, and others. Target: 50+ fonts.
- **Augmentations:** blur, noise, JPEG artifacts, rotation/skew, perspective, paper textures, shadows, low resolution, ink bleed, and screenshot-style UI backgrounds.

### 7.2 Real data (for testing, and fine-tuning where the license allows)
- Printed: scanned pages that we label ourselves (target: 200+ pages), plus public datasets if the license fits.
- Scene text (phase 5): EvArEST, ICDAR MLT (Arabic subset).
- Handwriting (phase 6): KHATT, IFN/ENIT.
- ⚠️ **License check required.** Many academic datasets are "research use only". Because the project is open source, we only **train shipped models** on data whose license allows redistribution of derived models. Restricted datasets may be used only for private evaluation. Every dataset's license is recorded in `data/LICENSES.md`.

---

## 8. Functional requirements

### 8.1 Inputs
- FR-1: Image files: PNG, JPEG, TIFF (multi-page), BMP, WebP.
- FR-2: PDF files (scanned or image-based). Pages are rendered at 300 DPI by default.
- FR-3: Clipboard image (GUI only).
- FR-4: Batch mode: a folder or glob of files.

### 8.2 Outputs
- FR-5: **Plain text** (UTF-8) in correct reading order, with paragraph breaks.
- FR-6: **JSON**: page → blocks → lines → words, each with bounding box, text, and confidence (0–1).
- FR-7: **Searchable PDF**: the original image with an invisible, selectable, searchable Arabic text layer.
- FR-8: **hOCR** and **ALTO XML**.
- FR-9: Options: keep/strip diacritics; output digits as-is, as Western, or as Arabic-Indic; strip tatweel.

### 8.3 Interfaces
- FR-10 **CLI:**
  ```
  arabic-ocr image.png                       # prints text
  arabic-ocr scan.pdf -f pdf -o out.pdf      # searchable PDF
  arabic-ocr ./folder -f json -o results/    # batch
  arabic-ocr --no-diacritics --digits western image.png
  ```
- FR-11 **Python library:**
  ```python
  from arabic_ocr import OCR

  result = OCR().read("page.png")
  print(result.text)
  for line in result.lines:
      print(line.box, line.text, line.confidence)
  ```
- FR-12 **Local HTTP API:** `arabic-ocr serve --port 8000` → `POST /ocr` (multipart upload) returns JSON. It binds to localhost by default.
- FR-13 **Desktop GUI:** drag and drop or open files; side-by-side image and text view with line boxes highlighted; edit text; copy; export in any format; language/diacritics options; Arabic and English UI.

### 8.4 Layout
- FR-14: Detect multi-column pages and output columns right to left.
- FR-15: Separate paragraphs; ignore page noise (borders, stains).
- FR-16 (stretch): Detect tables and keep cell order.

---

## 9. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | Platforms | Windows 10+, macOS 12+ (Intel + Apple Silicon), Ubuntu 22.04+ |
| NFR-2 | Offline | 100% offline, no telemetry |
| NFR-3 | Speed (CPU, 4 cores) | A4 page at 300 DPI in **≤ 2 s**; one screenshot in **≤ 0.5 s** |
| NFR-4 | Memory | **≤ 1 GB** RAM while processing one page |
| NFR-5 | Model size | Detector + recognizer **≤ 50 MB** total (v1) |
| NFR-6 | Installer size | **≤ 250 MB** per OS |
| NFR-7 | Reproducibility | `uv sync` + one command reproduces data generation and training; random seeds fixed |
| NFR-8 | Quality | CI runs tests on all 3 OSes on every push |

### Accuracy targets (CER = character error rate; lower is better)

| Content | v1 target | Later target |
|---|---|---|
| Clean printed text, diacritics ignored | CER ≤ 2% | ≤ 1% |
| Screenshots / UI text | CER ≤ 1% | ≤ 0.5% |
| Printed text, diacritics counted | CER ≤ 6% | ≤ 3% |
| Persian/Urdu printed (Naskh) | CER ≤ 4% | ≤ 2% |
| Nastaliq (Urdu style) | not a v1 target | ≤ 8% |
| Scene text photos (phase 5) | — | CER ≤ 15% |
| Handwriting (phase 6) | — | CER ≤ 20% |

We also report word error rate (WER) and detection precision/recall (IoU ≥ 0.5).

---

## 10. Roadmap

| Phase | Deliverable | Done when |
|---|---|---|
| **0 – Foundations** | uv project, folder structure, CI on 3 OSes, ML-basics notes | `uv run pytest` passes on Win/Mac/Linux in CI |
| **1 – Synthetic data** | Font + corpus pipeline that renders labeled Arabic line images with augmentations | 1M+ line images generated; visual spot-check shows correct shaping and diacritics |
| **2 – Recognizer** | Line recognizer (CNN + encoder + CTC), training script, ONNX export | Meets v1 CER on a synthetic test set and a small real printed set |
| **3 – Detector + pipeline** | Text-line detector, full image → text pipeline, CLI with txt/JSON | `arabic-ocr page.png` returns correct text on the real printed test pages |
| **4 – Layout + formats** | Multi-column reading order, PDF input, searchable PDF, hOCR/ALTO, library API, local HTTP API | All v1 output formats pass tests; v1 accuracy targets met |
| **5 – Desktop app + installers** | PySide6 GUI, installers for 3 OSes, **v1.0 release** | Installer works on clean Win/Mac/Linux machines with no Python installed |
| **6 – Scene text** | Retrain/fine-tune for photos (curved, colored, perspective text) | Phase 5 accuracy target met |
| **7 – Handwriting** | Handwriting recognizer variant | Phase 6 accuracy target met |
| **Later** | GPU acceleration, Nastaliq improvements, table detection, Rust runtime, mobile | — |

---

## 11. Licensing (open source)
- Code: **Apache-2.0** (permissive, includes a patent grant).
- Fonts used for training: OFL/Apache. The fonts are not shipped in the app, so their licenses only matter for training.
- Shipped model weights: trained only on redistributable data (§7.2); released under Apache-2.0 or CC BY-SA 4.0, depending on the training text licenses.
- Dependencies: all must be permissive or LGPL (PySide6 is LGPL, which is fine because we link to it dynamically).

---

## 12. Proposed repository layout

```
arabic-ocr/
├── pyproject.toml          # uv-managed
├── uv.lock
├── src/arabic_ocr/
│   ├── charset.py          # Arabic character set + normalization
│   ├── preprocess.py
│   ├── detect/             # detector model + inference
│   ├── recognize/          # recognizer model + CTC decoding
│   ├── layout.py           # reading order
│   ├── pipeline.py         # OCR().read()
│   ├── export/             # txt, json, pdf, hocr, alto
│   ├── cli.py
│   ├── api.py              # FastAPI server
│   └── gui/                # PySide6 app
├── training/               # dataset generation, training, evaluation, ONNX export
├── data/                   # (git-ignored) corpora, fonts, generated datasets; LICENSES.md
├── models/                 # released .onnx files
├── tests/
└── docs/                   # learning notes, glossary, architecture
```

---

## 13. Glossary (ML basics)
- **Model:** a function with millions of adjustable numbers (weights) that turns an input (an image) into an output (text).
- **Training:** showing the model many examples with known answers and adjusting the weights until it gets them right.
- **GPU:** a graphics card that does the math of training 10–100× faster than a CPU.
- **Detector:** a model that answers "where is the text?" by drawing boxes.
- **Recognizer:** a model that answers "what does this line say?"
- **CTC (Connectionist Temporal Classification):** a training method that lets the recognizer learn to read a whole line without being told where each letter is.
- **Synthetic data:** training images we generate ourselves by rendering text with fonts.
- **ONNX:** a standard file format for trained models that runs fast on any OS without PyTorch.
- **CER / WER:** the percentage of characters / words the OCR gets wrong.

---

## 14. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| The model works on synthetic data but fails on real scans (the "domain gap") | High | Heavy augmentation; label 200+ real pages early; fine-tune on real data |
| Diacritics are small and easy to miss | Medium | Use higher input resolution for lines; train a separate diacritics-heavy model if needed |
| Dataset licenses block open-source release | Medium | Track licenses from day one (§7.2) |
| No GPU access | Medium | Free Kaggle/Colab tiers; small models first |
| Scope is large for one learner | High | Strict phases; every phase ships something that works |
| macOS signing/notarization for installers | Low | Ship unsigned builds at first with instructions; add an Apple Developer ID later |

---

## 15. Open questions
1. ~~Final project name~~: decided, **arabic-ocr** (Python import name `arabic_ocr`).
2. **Training hardware:** start with free Kaggle/Colab, or rent cloud GPUs? This can be decided at phase 2; phases 0–1 need no GPU.
3. Do you have your own scanned Arabic documents to use as a real test set?
4. GUI language: Arabic + English from v1, or English first?
