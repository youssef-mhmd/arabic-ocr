"""Command-line interface: `arabic-ocr <file>`.

Only `--version` works in M0. Reading files arrives in M3.
"""

from pathlib import Path
from typing import Annotated

import typer

from arabic_ocr import __version__

app = typer.Typer(add_completion=False, help="Read Arabic-script text from images and PDFs.")


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"arabic-ocr {__version__}")
        raise typer.Exit()


@app.command()
def main(
    file: Annotated[Path | None, typer.Argument(help="Image or PDF to read.")] = None,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_print_version, is_eager=True, help="Show version and exit."
        ),
    ] = False,
) -> None:
    """Read Arabic-script text from FILE."""
    typer.echo("OCR is not implemented yet (planned for milestone M3).", err=True)
    raise typer.Exit(code=1)
