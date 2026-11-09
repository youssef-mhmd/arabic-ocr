# arabic-ocr

Offline, cross-platform (Windows, macOS, Linux) OCR engine for Arabic-script text, with its own detection and recognition models. Runs on CPU. Apache-2.0.

**Status:** early development (milestone M0, foundations). OCR does not work yet. See [docs/milestones.md](docs/milestones.md).

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # install runtime + dev dependencies
uv sync --group train        # + training dependencies (PyTorch; not available on Intel Macs)
uv run pytest                # run tests
uv run ruff check . && uv run ruff format .
uv run arabic-ocr --version
```

- Requirements: [docs/PRD.md](docs/PRD.md)
- ML learning notes: [docs/learning/](docs/learning/)

## License

Code is licensed under [Apache-2.0](LICENSE). Data and font licenses are tracked in [data/LICENSES.md](data/LICENSES.md).
