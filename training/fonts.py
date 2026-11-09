"""Pick training fonts from training/fonts.json.

Each font family has a sampling weight (see training/configs/fonts.yaml). A family's
weight is shared between its files, so a family with 8 weights (Thin…Black) is not
picked 8× more often than a family with one variable font file.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "training" / "fonts.json"
FONTS_DIR = ROOT / "data" / "fonts"


@dataclass(frozen=True)
class FontFile:
    path: Path
    family: str
    style: str  # naskh, sans, kufi, ...
    role: str  # core or display
    weight: float  # sampling weight
    missing: frozenset[str]  # charset symbols this font can't draw

    def supports(self, text: str) -> bool:
        return not (set(text) & self.missing)


def load_fonts(manifest: Path = MANIFEST, fonts_dir: Path = FONTS_DIR) -> list[FontFile]:
    """All font files from the manifest that exist on disk."""
    data = json.loads(manifest.read_text(encoding="utf-8"))
    fonts = []
    for fam in data["families"]:
        files = [f for f in fam["files"] if (fonts_dir / fam["dir"] / f["filename"]).exists()]
        for f in files:
            fonts.append(
                FontFile(
                    path=fonts_dir / fam["dir"] / f["filename"],
                    family=fam["family"],
                    style=fam["style"],
                    role=fam["role"],
                    weight=fam["weight"] / len(files),
                    missing=frozenset(f["missing"]),
                )
            )
    return fonts


class FontPicker:
    def __init__(self, fonts: list[FontFile], rng: random.Random) -> None:
        if not fonts:
            raise ValueError("no fonts found; run: uv run python -m training.download_fonts")
        self.fonts = fonts
        self.rng = rng

    def pick(self, text: str) -> FontFile | None:
        """A random font that can draw every character of `text`, or None."""
        usable = [f for f in self.fonts if f.supports(text)]
        if not usable:
            return None
        return self.rng.choices(usable, weights=[f.weight for f in usable])[0]
