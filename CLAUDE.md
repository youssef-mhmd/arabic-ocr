# CLAUDE.md

arabic-ocr is an Arabic-script OCR engine built from scratch: we design and train our own detection and recognition models. It is cross-platform (Windows, macOS, Linux), runs offline on CPU, and is open source (Apache-2.0).

- Requirements: [docs/PRD.md](docs/PRD.md). This is the source of truth for scope, targets, and decisions.
- Naming: project, repo folder, and CLI command are `arabic-ocr`; the Python import package is `arabic_ocr` (Python names cannot contain hyphens).
- Plan and progress: [docs/milestones.md](docs/milestones.md). Tick checkboxes as items land; work on the current milestone only.

## The owner is new to machine learning
- Explain ML concepts in plain language when introducing them, and add or update the matching note in `docs/learning/`.
- Prefer simple, readable code over clever code. Keep every step small and runnable.
- Before any long or expensive step (dataset generation, GPU training), say what it does, how long it will take, and what hardware it needs.

## Stack (do not change without updating the PRD)
- **Python, managed with uv only.** Use `uv add`, `uv run`, `uv sync`. Never use pip, poetry, conda, or requirements.txt.
- Training: **PyTorch** (`train` dependency group). Shipped runtime: **ONNX Runtime**. The `arabic_ocr` runtime package must **never import torch**.
- Images: OpenCV, NumPy, Pillow. Arabic rendering: **uharfbuzz + freetype-py**. Never render Arabic text without HarfBuzz shaping. Corpus Parquet files: pyarrow (`data` group).
- PDF: pypdfium2 (read), pikepdf (write). CLI: Typer. API: FastAPI. GUI: PySide6 (`gui` extra, `pyside6-essentials`). Packaging: PyInstaller + Inno Setup (Windows) / hdiutil (macOS) / dpkg-deb (Linux), built and tested by GitHub Actions (`installers/`, `.github/workflows/release.yml`).
- Tests: pytest. Lint/format: ruff.

## Commands
```bash
uv sync --extra gui          # install deps (runtime + desktop app + dev + data generation)
uv sync --group train        # + training deps (CUDA PyTorch, ~3 GB; see [tool.uv.sources])
uv run --group train python -m training.<script>   # anything that imports torch
uv run python -m training.download_fonts      # fetch training fonts into data/fonts/
uv run python -m training.inspect_samples     # render samples -> data/inspect/index.html
uv run python -m training.corpora download && uv run python -m training.corpora prepare
uv run python -m training.make_dataset training/configs/data_sample.yaml   # build a dataset -> data/datasets/<name>/
uv run python -m training.check_dataset data/datasets/v1   # verify a dataset + stats
uv run python -m training.inspect_samples -n 200 --dataset data/datasets/v1   # review page
uv run pytest                # tests
uv run ruff check . && uv run ruff format .
uv run arabic-ocr <file>     # run the OCR CLI
uv run arabic-ocr-gui        # run the desktop app
uv sync --extra gui --group build && uv run python -m installers.build   # installer for this OS
```

## Layout
```
src/arabic_ocr/    runtime package (charset, preprocess, detect/, recognize/, layout, pipeline, export/, cli, api, gui/)
installers/        PyInstaller spec, installer build script, Inno Setup script, icons
training/          data generation, training, evaluation, ONNX export (may import torch)
data/              git-ignored corpora, fonts, generated datasets; data/LICENSES.md is tracked
models/            released .onnx files
tests/             pytest
docs/              PRD, milestones, learning notes
```

## Arabic-specific rules
- Text is always stored and output in **logical Unicode order** (the order it is typed), never in visual order. Never reverse strings to make them "look right".
- The recognizer reads **whole lines** with CTC; never segment into individual letters.
- Diacritics (U+064B–U+0652, U+0670) are separate symbols in the charset; stripping them is an output option, not a training-time default.
- The charset covers Arabic letters, hamza forms, tashkeel, tatweel, ٠–٩, ۰–۹, 0–9, Arabic punctuation (، ؛ ؟ ٪), and Persian/Urdu letters. `src/arabic_ocr/charset.py` is the single source of truth.
- Normalize with NFC, and map presentation forms (U+FB50–U+FDFF, U+FE70–U+FEFF) to base letters before training or comparing.
- Reading order: right to left, top to bottom; columns right to left.

## Cross-platform rules
- Use `pathlib` everywhere; no hard-coded separators or OS-specific paths.
- Always open text files with `encoding="utf-8"`. Windows defaults to a different encoding.
- No shell-specific scripts in the main workflow. Write helper scripts in Python and run them with `uv run`.
- Every dependency must ship wheels for Windows x64, macOS arm64 (macOS 14+) and Linux x86-64 (glibc 2.35, Ubuntu 22.04). Intel Macs are not supported.
- CI must pass on all 3 OSes.
- GUI: connect Qt signals to methods, never to lambdas that use `self` (crash on exit). Tests run Qt "offscreen".

## Data and licensing
- Every corpus, dataset, and font gets an entry in `data/LICENSES.md` before it is used.
- Shipped models may only be trained on data whose license allows redistribution. Research-only datasets (e.g. KHATT, IFN/ENIT) are for private evaluation only.
- Never commit datasets, fonts, checkpoints, or generated images. Only released `.onnx` models go in `models/`.

## Training conventions
- Configs live in files (YAML/TOML), not hard-coded. Fix random seeds. Save checkpoints and support resume.
- Report CER/WER both with and without diacritics. Compare against the PRD §9 targets.
- After export, verify the ONNX output matches PyTorch output before shipping a model.
