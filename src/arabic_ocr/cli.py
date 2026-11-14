"""Command-line interface.

    arabic-ocr page.png                          # print the text
    arabic-ocr scan.pdf -o scan.txt              # a PDF (every page, 300 DPI)
    arabic-ocr page.png -f json -o out.json      # structured output with boxes
    arabic-ocr ./scans -f json -o results/       # batch: a folder ...
    arabic-ocr "scans/*.tif" -o results/         # ... or a glob pattern
    arabic-ocr scan.pdf -f pdf -o out.pdf        # searchable PDF
    arabic-ocr scan.tif -f alto -o out.xml       # also: -f hocr
    arabic-ocr page.png --no-diacritics --no-tatweel --digits western

Inputs: PNG, JPEG, TIFF (all pages), BMP, WebP, PDF. With several input files,
`-o` must be a folder; each result is written there as <name>.<format>.
"""

from __future__ import annotations

import sys
import time
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from arabic_ocr import __version__

app = typer.Typer(add_completion=False, help="Read Arabic-script text from images and PDFs.")


class Format(StrEnum):
    txt = "txt"
    json = "json"
    pdf = "pdf"  # searchable PDF: the page images with an invisible text layer
    hocr = "hocr"  # hOCR 1.2 (HTML)
    alto = "alto"  # ALTO 4 XML


#: File extension for each format in batch mode.
EXTENSIONS = {"txt": "txt", "json": "json", "pdf": "pdf", "hocr": "hocr", "alto": "xml"}


class DigitStyle(StrEnum):
    keep = "keep"
    western = "western"
    arabic = "arabic"


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"arabic-ocr {__version__}")
        raise typer.Exit()


def _utf8_console() -> None:
    # The Windows console may not default to UTF-8; Arabic must not turn into "?".
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def render(doc, fmt: Format) -> str:
    """A Document as text in the chosen format."""
    if fmt is Format.json:
        from arabic_ocr.export.to_json import document_to_json

        return document_to_json(doc.pages, doc.source)
    if fmt is Format.hocr:
        from arabic_ocr.export.xml_formats import to_hocr

        return to_hocr(doc)
    if fmt is Format.alto:
        from arabic_ocr.export.xml_formats import to_alto

        return to_alto(doc)
    return doc.text


def output_names(files: list[Path], fmt: Format) -> dict[Path, str]:
    """<name>.<format> for every input; inputs that share a name (a.png, a.pdf)
    keep their extension too (a.png.txt, a.pdf.txt) so nothing is overwritten."""
    stems = [f.stem for f in files]
    return {
        f: (f.name if stems.count(f.stem) > 1 else f.stem) + f".{EXTENSIONS[fmt.value]}"
        for f in files
    }


@app.command()
def main(
    target: Annotated[
        str, typer.Argument(help="An image or PDF file, a folder, or a glob like 'scans/*.png'.")
    ],
    fmt: Annotated[Format, typer.Option("--format", "-f", help="Output format.")] = Format.txt,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file (one input) or folder (several inputs)."),
    ] = None,
    recursive: Annotated[
        bool, typer.Option("--recursive", "-r", help="Also read files in sub-folders.")
    ] = False,
    dpi: Annotated[int, typer.Option("--dpi", help="Resolution for PDF pages.")] = 300,
    no_diacritics: Annotated[
        bool, typer.Option("--no-diacritics", help="Remove tashkeel from the output.")
    ] = False,
    no_tatweel: Annotated[
        bool, typer.Option("--no-tatweel", help="Remove tatweel/kashida (ـ) from the output.")
    ] = False,
    digits: Annotated[
        DigitStyle, typer.Option("--digits", help="Keep digits as printed, or convert them.")
    ] = DigitStyle.keep,
    no_deskew: Annotated[
        bool, typer.Option("--no-deskew", help="Do not straighten the page.")
    ] = False,
    denoise: Annotated[
        bool, typer.Option("--denoise", help="Remove speckles (noisy scans).")
    ] = False,
    binarize: Annotated[
        bool, typer.Option("--binarize", help="Convert to pure black/white.")
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_print_version, is_eager=True, help="Show version and exit."
        ),
    ] = False,
) -> None:
    """Read Arabic-script text from images and PDFs."""
    from arabic_ocr.inputs import find_inputs  # imported here so --help/--version stay fast
    from arabic_ocr.pipeline import OCR

    _utf8_console()
    files = find_inputs(target, recursive=recursive)
    if not files:
        typer.echo(f"Error: no supported image or PDF files found: {target}", err=True)
        raise typer.Exit(code=1)
    batch = len(files) > 1 or Path(target).is_dir()
    if batch and output is None:
        typer.echo("Error: several input files need an output folder: -o FOLDER", err=True)
        raise typer.Exit(code=1)

    options = {
        "diacritics": not no_diacritics,
        "tatweel": not no_tatweel,
        "digits": digits.value,
        "deskew": not no_deskew,
        "denoise": denoise,
        "binarize": binarize,
    }
    try:
        ocr = OCR()
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e

    if fmt is Format.pdf and output is None:
        typer.echo("Error: a PDF can't be shown on screen; give a file: -o out.pdf", err=True)
        raise typer.Exit(code=1)

    if not batch:
        try:
            if fmt is Format.pdf:
                from arabic_ocr.export.searchable_pdf import ocr_to_pdf

                ocr_to_pdf(ocr, files[0], output, dpi=dpi, **options)
                return
            result = render(ocr.read(files[0], dpi=dpi, **options), fmt)
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1) from e
        if output is None:
            typer.echo(result)
        else:
            output.write_text(result + "\n", encoding="utf-8")
        return

    output.mkdir(parents=True, exist_ok=True)
    names = output_names(files, fmt)
    failed = 0
    for i, path in enumerate(files, 1):
        start = time.perf_counter()
        try:
            if fmt is Format.pdf:
                from arabic_ocr.export.searchable_pdf import ocr_to_pdf

                doc = ocr_to_pdf(ocr, path, output / names[path], dpi=dpi, **options)
            else:
                doc = ocr.read(path, dpi=dpi, **options)
                (output / names[path]).write_text(render(doc, fmt) + "\n", encoding="utf-8")
        except Exception as e:  # keep going: one bad file must not stop a batch
            failed += 1
            typer.echo(f"[{i}/{len(files)}] {path}: FAILED ({e})", err=True)
            continue
        typer.echo(
            f"[{i}/{len(files)}] {path} -> {names[path]} "
            f"({len(doc.pages)} page(s), {time.perf_counter() - start:.1f} s)",
            err=True,
        )
    typer.echo(f"Done: {len(files) - failed} of {len(files)} files -> {output}", err=True)
    if failed:
        raise typer.Exit(code=1)
