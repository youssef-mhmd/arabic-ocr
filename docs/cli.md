# Command-line reference: `arabic-ocr`

The `arabic-ocr` command comes with every installer (see [install.md](install.md))
and with the Python package (`uv run arabic-ocr` in this repository).

```text
arabic-ocr TARGET [options]
arabic-ocr serve [--host HOST] [--port PORT]
```

`TARGET` is an image or PDF file, a folder, or a glob pattern in quotes
(`"scans/*.tif"`). Supported inputs: PNG, JPEG, TIFF (every page), BMP, WebP,
PDF (every page, rendered at 300 DPI). Output text is UTF-8, in logical order
(the order Arabic is typed), so it displays correctly in any app that supports
right-to-left text.

## Examples

```bash
arabic-ocr page.png                          # print the text
arabic-ocr scan.pdf -o scan.txt              # every page of a PDF, to a file
arabic-ocr page.png -f json -o page.json     # lines and words with boxes and confidence
arabic-ocr scan.pdf -f pdf -o searchable.pdf # searchable PDF (invisible text layer)
arabic-ocr scan.tif -f alto -o scan.xml      # ALTO XML (also: -f hocr)
arabic-ocr ./scans -f json -o results/       # batch: every file in a folder
arabic-ocr "scans/**/*.png" -r -o results/   # batch: a glob, including sub-folders
arabic-ocr page.png --no-diacritics --no-tatweel --digits western
arabic-ocr serve --port 8000                 # local HTTP API
```

## Options

| Option | Meaning |
|---|---|
| `-f, --format` | `txt` (default), `json`, `pdf` (searchable PDF), `hocr`, `alto` |
| `-o, --output` | Output file (one input) or folder (several inputs). Without it, text is printed. A PDF always needs `-o`. |
| `-r, --recursive` | With a folder or glob: also read files in sub-folders |
| `--dpi N` | Resolution PDF pages are rendered at (default 300) |
| `--no-diacritics` | Remove tashkeel (ً ٌ ٍ َ ُ ِ ّ ْ ٰ) from the output |
| `--no-tatweel` | Remove tatweel/kashida (ـ) from the output |
| `--digits keep\|western\|arabic` | Digits as printed (default), as 0–9, or as ٠–٩ |
| `--correct-words` | Fix likely dot slips (ب/ت/ث/ن/ي…) with a word list. Source installs only: needs `models/lexicon.tsv.gz` (`uv run python -m training.make_lexicon`) |
| `--no-deskew` | Don't straighten tilted pages |
| `--denoise` | Remove speckles (noisy scans) |
| `--binarize` | Convert the page to pure black and white first |
| `--version` | Show the version |
| `--help` | Show the help |

### Batch mode

With a folder, a glob or several files, `-o` must be a folder. Each result is
written there as `<name>.<format>` (`.xml` for ALTO). If two inputs share a name
(`a.png`, `a.pdf`), their extension is kept (`a.png.txt`, `a.pdf.txt`). One
unreadable file doesn't stop the batch. It is reported, and the exit code is 1
at the end.

### Plain-text layout

Blocks (paragraphs, headings, columns) are separated by an empty line, and lines
by a newline. Pages of a PDF or TIFF are separated by a form feed (`\f`) on a
line of its own. Columns are read right to left.

## `arabic-ocr serve`: local HTTP API

```bash
arabic-ocr serve                 # http://127.0.0.1:8000, this computer only
arabic-ocr serve --port 9000
```

| Endpoint | Description |
|---|---|
| `GET /health` | `{"status": "ok", "version": "…"}` |
| `POST /ocr` | Upload one image or PDF as the multipart field `file`. Returns JSON by default. |
| `GET /docs` | Interactive documentation |

Query options for `POST /ocr`: `format=json|txt|hocr|alto|pdf`,
`diacritics=true|false`, `tatweel=true|false`, `digits=keep|western|arabic`,
`dpi=50…600`. Uploads up to 200 MB.

```bash
curl -F "file=@page.png" http://127.0.0.1:8000/ocr
curl -F "file=@scan.pdf" "http://127.0.0.1:8000/ocr?format=pdf" -o searchable.pdf
curl -F "file=@page.png" "http://127.0.0.1:8000/ocr?format=txt&diacritics=false"
```

The server listens on 127.0.0.1 by default, so only this computer can reach it.
`--host 0.0.0.0` opens it to your network (with a warning). It has no
passwords, so only do that on a network you trust.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Error: no input found, unreadable file, models missing, or a file in a batch failed |
| 2 | Wrong options (see `--help`) |
