# Milestones: arabic-ocr (Arabic OCR)

Derived from [PRD.md](PRD.md). Each milestone produces something that works and has clear exit criteria. Tick boxes as work lands. Don't start a milestone until the previous one's exit criteria are met, unless the item is marked *(parallel ok)*.

**Legend:** 🖥️ = runs on a normal PC · ⚡ = needs a GPU (Kaggle/Colab/cloud is fine)

---

## M0: Foundations 🖥️

**Goal:** an empty but correct project that builds and tests on all 3 OSes.

- [x] `uv init` the project with a `src/arabic_ocr/` package layout; Python version pinned in `.python-version`
- [x] `pyproject.toml` with dependency groups: runtime, `train`, `gui`, `dev`
- [x] Folder skeleton from PRD §12 (`src/arabic_ocr`, `training/`, `data/`, `models/`, `tests/`, `docs/`)
- [x] `.gitignore` (data/, generated datasets, checkpoints, `.venv`)
- [x] `git init`, first commit, Apache-2.0 `LICENSE`
- [x] ruff (lint + format) and pytest configured; one smoke test
- [x] GitHub Actions CI: lint + tests on `windows-latest`, `macos-latest`, `ubuntu-latest`
- [x] `docs/learning/01-ml-basics.md`: tensors, models, training loop, loss, train/val/test split

**Exit criteria:** `uv sync && uv run pytest` passes locally, and CI is green on all 3 OSes.

---

## M1: Character set and synthetic data 🖥️

**Goal:** generate millions of correctly shaped, labeled Arabic line images.

- [x] `charset.py`: full symbol set (Arabic letters, hamza forms, tashkeel U+064B–U+0652 + U+0670, tatweel, ٠–٩, ۰–۹, 0–9, Arabic punctuation, Persian/Urdu letters), with encode/decode
- [x] Unicode normalization rules (NFC, presentation forms → base letters, consistent hamza/alef handling); unit tests
- [x] Download scripts for corpora: Arabic Wikipedia, Arabic Wikisource (+ diacritized subset), Tanzil Quran, Persian/Urdu Wikipedia. *(Tashkeela and OpenITI dropped for license reasons; see PRD §7.1.)*
- [x] Text cleaning + line sampler (lengths 1–60 chars; mix of with/without diacritics; numbers; punctuation)
- [x] Font collection (50+ OFL/Apache Arabic fonts) with a `fonts.json` manifest recording each font's license
- [x] Renderer using uharfbuzz + freetype-py (correct joining, ligatures, mark positioning)
- [x] Augmentation pipeline: blur, noise, JPEG, skew/rotation, perspective, paper textures, shadows, low DPI, ink bleed, screenshot/UI backgrounds
- [x] Dataset writer (sharded files, e.g. LMDB or WebDataset tar) + train/val/test split
- [x] `data/LICENSES.md`: license for every corpus and font
- [x] Visual inspection notebook/script: render a grid of samples for spot-checking
- [x] `docs/learning/02-data.md`

**Exit criteria:** 1M+ line images generated; a manual check of 200 random samples shows correct shaping and diacritics, with labels matching the images.

✅ **Met 2026-11-09.** Dataset `v1`: 1,000,000 train / 10,000 val / 10,000 test line images (`training/configs/data_v1.yaml`); `training/check_dataset.py` passes on every sample. The owner's manual review of 200 random samples (plus 100 number lines) found one bug (fonts that swap digit shapes during shaping), which was fixed and the dataset regenerated before sign-off.

---

## M2: Line recognizer ⚡

**Goal:** a model that reads one line image → Arabic text.

- [ ] Recognizer model: CNN backbone + BiLSTM/Transformer encoder + CTC head (target ≤ 20 MB)
- [ ] Training script: config file, mixed precision, checkpointing, resume, fixed seeds, logs (TensorBoard or CSV)
- [ ] Notebook/script that runs training on Kaggle/Colab
- [ ] CTC greedy decoder (beam search later)
- [ ] Evaluation script: CER/WER, with and without diacritics, per font family
- [ ] Label a real test set: 50+ real printed line images and 50+ screenshot lines *(parallel ok)*
- [ ] ONNX export + check that ONNX output matches PyTorch output
- [ ] CPU inference wrapper using ONNX Runtime in `src/arabic_ocr/recognize/`
- [ ] `docs/learning/03-recognizer-and-ctc.md`

**Exit criteria:** On the synthetic test set, CER ≤ 2% with diacritics ignored and ≤ 6% with diacritics counted. On the real printed lines, CER ≤ 3% with diacritics ignored. The ONNX model runs on CPU.

---

## M3: Text detector and end-to-end pipeline ⚡🖥️

**Goal:** `arabic-ocr page.png` prints the page's text.

- [ ] Synthetic full-page generator (multi-line, multi-column, mixed sizes, headers) with line boxes as labels
- [ ] DBNet-style detector (small backbone, target ≤ 25 MB) + training + ONNX export
- [ ] Detection evaluation: precision/recall at IoU ≥ 0.5
- [ ] Preprocessing: grayscale, deskew, denoise, optional binarization
- [ ] `pipeline.py`: preprocess → detect → crop/rectify lines → recognize → postprocess
- [ ] Simple reading order (lines sorted top-to-bottom, right-to-left)
- [ ] CLI (Typer): `arabic-ocr <file>`; `-f txt|json`; `-o`; `--no-diacritics`; `--digits keep|western|arabic`
- [ ] JSON schema: page → blocks → lines → words, with box, text, confidence
- [ ] Label 50+ real printed pages + 50 screenshots for end-to-end testing *(parallel ok)*
- [ ] Speed benchmark script

**Exit criteria:** On the real test pages, end-to-end CER ≤ 3% for printed text and ≤ 1.5% for screenshots; detection recall ≥ 95%; an A4 page at 300 DPI takes ≤ 3 s on a 4-core CPU.

---

## M4: Layout, formats, library and API 🖥️

**Goal:** all v1 inputs/outputs work, and v1 accuracy targets are met.

- [ ] Layout analysis: blocks, multi-column detection (RTL column order), paragraphs, noise removal
- [ ] Input: PNG/JPEG/TIFF (multi-page)/BMP/WebP; PDF via pypdfium2 at 300 DPI; batch folders/globs
- [ ] Exporters: txt, JSON, **searchable PDF** (pikepdf invisible text layer), **hOCR**, **ALTO XML**
- [ ] Check the searchable PDF: text is selectable and searchable in common PDF viewers, Arabic copies correctly
- [ ] Public library API: `OCR().read(path_or_image)` → typed result objects; docstrings
- [ ] Local HTTP API: `arabic-ocr serve` (FastAPI, binds to 127.0.0.1 by default), `POST /ocr`
- [ ] Optional lexicon/word-list correction in postprocessing
- [ ] Fine-tune on real/augmented data to close the domain gap
- [ ] Performance tuning to reach NFR-3/4/5 (≤ 2 s/page, ≤ 1 GB RAM, models ≤ 50 MB)

**Exit criteria:** every v1 accuracy target in PRD §9 is met. All output formats pass tests on all 3 OSes.

---

## M5: Desktop app and v1.0 release 🖥️

**Goal:** non-technical users can install and use it.

- [ ] PySide6 GUI: open/drag-and-drop/paste from clipboard; side-by-side image and text; line boxes highlighted; edit text; copy; export in any format
- [ ] Settings: diacritics on/off, digit style, output folder
- [ ] UI languages: Arabic (RTL) + English
- [ ] Background processing with progress bar (UI never freezes); batch queue
- [ ] PyInstaller builds via GitHub Actions: Windows `.exe`/installer, macOS `.dmg` (Intel + Apple Silicon), Linux `.AppImage`/`.deb`
- [ ] Test on clean machines/VMs with no Python installed
- [ ] User docs: install guide per OS, CLI reference, library/API reference
- [ ] Publish models + release on GitHub with checksums

**Exit criteria:** **v1.0 tagged.** The installers work on clean Windows 10/11, macOS 12+, and Ubuntu 22.04 machines; installers are ≤ 250 MB.

---

## M6: Scene text (photos) ⚡

- [ ] Scene-text synthetic data (text on real photos, curved/colored text, perspective, lighting)
- [ ] Collect/label real Arabic sign and label photos; check dataset licenses
- [ ] Detector able to find rotated/curved text (polygon output)
- [ ] Fine-tune the recognizer for scene text; add a "photo" mode in the CLI/GUI

**Exit criteria:** CER ≤ 15% on the scene-text test set.

---

## M7: Handwriting ⚡

- [ ] Find handwriting datasets whose licenses allow a shipped model; otherwise collect our own
- [ ] Handwriting-style synthetic data (handwriting fonts + elastic distortions)
- [ ] Handwriting recognizer variant; "handwriting" mode in the CLI/GUI

**Exit criteria:** CER ≤ 20% on the handwriting test set.

---

## Later (unscheduled)
- GPU acceleration (DirectML / CoreML / CUDA execution providers)
- Better Nastaliq (Urdu) accuracy
- Table detection and cell ordering
- Native runtime in Rust; mobile/browser builds from the same ONNX models
