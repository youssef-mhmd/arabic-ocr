"""Check a generated dataset end to end and print statistics.

Usage:
    uv run python -m training.check_dataset data/datasets/v1

Reads every shard (labels and metadata; images are only size-checked) and verifies:
- the manifest says "complete" and its charset fingerprint matches the code;
- every shard has the expected number of samples, each with png + txt + json;
- every label is normalized and uses only charset symbols;
- every label only uses characters its font draws correctly (per fonts.json);
- every image is a grayscale PNG of the configured height.

Then prints statistics that show what the model will learn from: line lengths,
how many lines have diacritics, digits, Persian/Urdu letters, and the font mix.
Exits with an error if any check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from arabic_ocr import charset
from training.dataset import SPLITS, charset_fingerprint
from training.fonts import MANIFEST as FONTS_MANIFEST

_DIGITS = set(
    charset.WESTERN_DIGITS + charset.ARABIC_INDIC_DIGITS + charset.EASTERN_ARABIC_INDIC_DIGITS
)
_PERSIAN_URDU = set(charset.PERSIAN_URDU_LETTERS)


def png_size(data: bytes) -> tuple[int, int, int]:
    """(width, height, color type) from a PNG header, without decoding the image."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    width, height, _depth, color_type = struct.unpack(">IIBB", data[16:26])
    return width, height, color_type


def check_shard(path: Path, height: int, font_missing: dict[str, frozenset[str]]) -> dict:
    errors: list[str] = []
    parts: dict[str, set[str]] = {}
    lengths: Counter = Counter()
    styles: Counter = Counter()
    widths: list[int] = []
    n_marks = n_digits = n_persian = 0
    labels: dict[str, str] = {}
    fonts: dict[str, str] = {}
    with tarfile.open(path) as tar:
        for m in tar:
            key, _, ext = m.name.rpartition(".")
            parts.setdefault(key, set()).add(ext)
            f = tar.extractfile(m)
            if f is None:
                continue
            data = f.read()
            if ext == "txt":
                label = data.decode("utf-8")
                labels[key] = label
                if charset.normalize(label) != label:
                    errors.append(f"{key}: label not normalized")
                if bad := charset.unsupported_chars(label):
                    errors.append(f"{key}: unsupported {sorted(bad)}")
                lengths[min(len(label) // 10 * 10, 60)] += 1
                chars = set(label)
                n_marks += bool(chars & charset.DIACRITIC_SET)
                n_digits += bool(chars & _DIGITS)
                n_persian += bool(chars & _PERSIAN_URDU)
            elif ext == "json":
                meta = json.loads(data)
                styles[meta["style"]] += 1
                fonts[key] = meta["font"]
            elif ext == "png":
                w, h, color = png_size(data)
                if h != height or color != 0:  # color type 0 = grayscale
                    errors.append(f"{key}: image {w}x{h} color type {color}")
                widths.append(w)
    for key, exts in parts.items():
        if exts != {"png", "txt", "json"}:
            errors.append(f"{key}: has {sorted(exts)}")
    # Every character must be one the sample's font draws correctly (per fonts.json).
    for key, label in labels.items():
        font = fonts.get(key)
        if font not in font_missing:
            errors.append(f"{key}: font {font} is not in fonts.json")
        elif bad := set(label) & font_missing[font]:
            errors.append(f"{key}: {font} can't draw {''.join(sorted(bad))} correctly")
    return {
        "count": len(parts),
        "errors": errors,
        "lengths": lengths,
        "styles": styles,
        "marks": n_marks,
        "digits": n_digits,
        "persian": n_persian,
        "widths": widths,
    }


def share(results: list[dict], key: str, total: int) -> str:
    """Percentage of samples counted under `key`, e.g. lines with diacritics."""
    return f"{100 * sum(r[key] for r in results) / max(total, 1):.1f}%"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a generated dataset.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    problems: list[str] = []
    if manifest["status"] != "complete":
        problems.append(f"manifest status is {manifest['status']!r}")
    if manifest["charset"] != charset_fingerprint():
        problems.append("charset fingerprint differs from the current charset.py")
    font_missing = {
        f["filename"]: frozenset(f["missing"])
        for fam in json.loads(FONTS_MANIFEST.read_text(encoding="utf-8"))["families"]
        for f in fam["files"]
    }

    for split in SPLITS:
        shards = sorted((args.dataset / split).glob("shard-*.tar"))
        if leftovers := list((args.dataset / split).glob("*.tmp")):
            problems.append(f"{split}: unfinished files {[p.name for p in leftovers]}")
        expected_shards = math.ceil(cfg["samples"][split] / cfg["shard_size"])
        if len(shards) != expected_shards:
            problems.append(f"{split}: {len(shards)} shards, expected {expected_shards}")

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            n = len(shards)
            results = list(
                pool.map(check_shard, shards, [cfg["image_height"]] * n, [font_missing] * n)
            )

        total = sum(r["count"] for r in results)
        if total != cfg["samples"][split]:
            problems.append(f"{split}: {total} samples, expected {cfg['samples'][split]}")
        for r in results:
            problems.extend(f"{split}: {e}" for e in r["errors"][:5])

        lengths, styles = Counter(), Counter()
        widths: list[int] = []
        for r in results:
            lengths.update(r["lengths"])
            styles.update(r["styles"])
            widths.extend(r["widths"])
        widths.sort()
        print(f"== {split}: {total:,} samples in {len(shards)} shards")
        print(
            f"   with diacritics {share(results, 'marks', total)}, "
            f"digits {share(results, 'digits', total)}, "
            f"Persian/Urdu letters {share(results, 'persian', total)}"
        )
        print(
            "   label length: "
            + ", ".join(
                f"{k}-{k + 9}: {100 * v / total:.0f}%" if k < 60 else f"60: {100 * v / total:.0f}%"
                for k, v in sorted(lengths.items())
            )
        )
        print(
            "   font style: "
            + ", ".join(f"{k} {100 * v / total:.0f}%" for k, v in styles.most_common())
        )
        if widths:
            print(
                f"   image width: median {widths[len(widths) // 2]} px, "
                f"max {widths[-1]} px (height {cfg['image_height']})"
            )

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems[:50]:
            print("  -", p)
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
