"""Command-line interface.

    arabic-ocr page.png                     # print the text
    arabic-ocr page.png -f json -o out.json # structured output with boxes
    arabic-ocr page.png --no-diacritics --digits western

Images only for now (PNG, JPEG, TIFF, BMP, WebP); PDF support arrives in M4.
"""

from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from arabic_ocr import __version__

app = typer.Typer(add_completion=False, help="Read Arabic-script text from images.")


class Format(StrEnum):
    txt = "txt"
    json = "json"


class DigitStyle(StrEnum):
    keep = "keep"
    western = "western"
    arabic = "arabic"


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"arabic-ocr {__version__}")
        raise typer.Exit()


@app.command()
def main(
    file: Annotated[Path, typer.Argument(help="Image to read.", exists=True, dir_okay=False)],
    fmt: Annotated[Format, typer.Option("--format", "-f", help="Output format.")] = Format.txt,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write to this file instead of the screen."),
    ] = None,
    no_diacritics: Annotated[
        bool, typer.Option("--no-diacritics", help="Remove tashkeel from the output.")
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
    """Read Arabic-script text from FILE."""
    from arabic_ocr.pipeline import OCR  # imported here so --help and --version stay fast

    try:
        ocr = OCR()
        page = ocr.read_file(
            file,
            diacritics=not no_diacritics,
            digits=digits.value,
            deskew=not no_deskew,
            denoise=denoise,
            binarize=binarize,
        )
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e

    if fmt is Format.json:
        from arabic_ocr.export.to_json import document_to_json

        result = document_to_json([page], file.name)
    else:
        result = page.text

    if output is not None:
        output.write_text(result + "\n", encoding="utf-8")
    else:
        # The Windows console may not default to UTF-8; Arabic must not turn into "?".
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        typer.echo(result)
