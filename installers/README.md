# Building the installers

The installers are built by GitHub Actions (`.github/workflows/release.yml`) on
all three systems. A build only works on the system it was made on, so you can't
make the macOS app on Windows. You can build the one for your own computer
locally to try a change.

## How it works

1. **PyInstaller** (`arabic-ocr.spec`) copies a Python interpreter, the
   `arabic_ocr` code, every library it uses (ONNX Runtime, OpenCV, Qt…) and the
   two models into one folder, with two small start-up programs:
   `arabic-ocr-gui` (the desktop app) and `arabic-ocr` (the command line). That
   is why users don't need Python. Training code and PyTorch are never included.
2. **Self-test** (`build.py`): runs the *built* programs, not the source. It
   reads the sample with the CLI, starts `arabic-ocr serve` and uploads the
   sample, and runs `arabic-ocr-gui --self-test`, which opens the window, reads
   the sample on the background thread and writes all five export formats.
3. **Package** it as each system expects:

   | System | Tool | Result |
   |---|---|---|
   | Windows | [Inno Setup 6](https://jrsoftware.org/isinfo.php) (`arabic-ocr.iss`) | `arabic-ocr-<v>-windows-x64-setup.exe` |
   | macOS | `hdiutil` (built in) | `arabic-ocr-<v>-macos-arm64.dmg` with `Arabic OCR.app` |
   | Linux | `dpkg-deb` | `arabic-ocr_<v>_amd64.deb` (installs to `/opt/arabic-ocr`) |

4. **Clean-machine tests** (release workflow only): each installer is installed
   on a fresh machine where Python is absent or hidden (Ubuntu 22.04 and 24.04
   containers, a Windows runner with Python removed from PATH, macOS 15 and 26),
   then the installed app runs its self-test. On Windows the test also uninstalls.

The Linux build runs inside an Ubuntu 22.04 container: a Linux program only runs
on systems at least as new as the one it was built on, and 22.04 is the oldest
Ubuntu we support.

## Build locally

```bash
uv sync --extra gui --group build
uv run python -m installers.build              # build + self-test + installer
uv run python -m installers.build --no-package # just dist/arabic-ocr/ (Windows/Linux)
```

It takes 2–5 minutes and needs no GPU. On Windows, install Inno Setup 6 first
(or set `INNO_SETUP` to its folder). The result goes in `dist/`.

## Making a release

1. Set `version` in `pyproject.toml` (e.g. `1.0.0`), run `uv lock`, and update
   `docs/release-notes.md`. Commit.
2. Tag and push: `git tag v1.0.0 && git push origin v1.0.0`.
3. The release workflow builds and tests all three installers. Only if every
   clean-machine test passes does it publish a GitHub release with the
   installers, the two `.onnx` models and `SHA256SUMS.txt`.

Pushes that change the app or its packaging run the same build and tests,
without publishing.

## Other files

- `make_assets.py`: makes the icon and the sample image (`icons/`,
  `src/arabic_ocr/gui/icon.png`, `sample.png`). The results are committed; run it
  again only to change them.
- `entry_gui.py`, `entry_cli.py`: the start-up scripts PyInstaller wraps.

## Not done yet

- **Code signing.** The Windows installer and the macOS app are unsigned, so
  Windows SmartScreen and macOS Gatekeeper warn on first start (`docs/install.md`
  explains how to get past it). Fixing this needs a Windows code-signing
  certificate and an Apple Developer ID (99 USD/year) for notarization.
- **Intel Macs and Linux ARM** are not built (see the PRD, §9 note 3).
