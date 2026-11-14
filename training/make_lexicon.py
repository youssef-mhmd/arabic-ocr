"""Build the word list (lexicon) used by the optional word correction (M4).

    uv run python -m training.make_lexicon            # -> models/lexicon.tsv.gz

Counts words in the *training* split of our corpora (Arabic, Persian and Urdu
Wikipedia, Arabic Wikisource; see data/LICENSES.md) and keeps the words seen at
least MIN_COUNT times, most frequent first: "word<TAB>count" per line, gzipped.
Words are normalized like OCR output (NFC, no diacritics, no tatweel).

Only the first MAX_MB megabytes of each corpus are read: word frequencies settle
long before the end, and it keeps this to a few minutes.
"""

from __future__ import annotations

import argparse
import gzip
import re
from collections import Counter
from pathlib import Path

from arabic_ocr import charset

ROOT = Path(__file__).resolve().parent.parent
CORPORA = ["wikipedia_ar", "wikisource_ar", "wikipedia_fa", "wikipedia_ur"]
MAX_MB = 150
MIN_COUNT = 3
MAX_WORDS = 400_000
#: A "word" for the lexicon: Arabic-script letters only (no digits or punctuation).
WORD = re.compile(r"[ء-يٮ-ۓەۮۯۺ-ۼ]+")


def normalize_word(word: str) -> str:
    return charset.strip_tatweel(charset.strip_diacritics(charset.normalize(word)))


def count_words(path: Path, max_bytes: int) -> Counter:
    counts: Counter = Counter()
    read = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            read += len(line.encode("utf-8"))
            counts.update(WORD.findall(normalize_word(line)))
            if read >= max_bytes:
                break
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the OCR word list.")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "lexicon.tsv.gz")
    args = parser.parse_args(argv)
    total: Counter = Counter()
    for name in CORPORA:
        path = ROOT / "data" / "corpora" / name / "train.txt"
        if not path.exists():
            print(f"  (missing {path.relative_to(ROOT)}, skipped)")
            continue
        c = count_words(path, MAX_MB * 2**20)
        print(f"  {name}: {sum(c.values()):,} words, {len(c):,} different", flush=True)
        total.update(c)
    words = [(w, n) for w, n in total.most_common(MAX_WORDS) if n >= MIN_COUNT]
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        for w, n in words:
            f.write(f"{w}\t{n}\n")
    print(f"Wrote {args.out} ({len(words):,} words, {args.out.stat().st_size / 2**20:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
