"""Accuracy measures for OCR: CER and WER.

- **CER** (character error rate): how many single-character edits (insert, delete,
  replace) turn the prediction into the correct text, divided by the length of the
  correct text. 0.02 = 2 mistakes per 100 characters. Lower is better.
- **WER** (word error rate): the same, counting whole words.

Both are reported twice (PRD §9, CLAUDE.md):
- *with diacritics*: every tashkeel mark counts;
- *without diacritics*: marks are removed from both texts first, so only the
  letters, digits and punctuation count.

Texts are compared in logical order after normalization.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from arabic_ocr import charset


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance: fewest inserts/deletes/replacements to turn a into b."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


@dataclass
class Scores:
    cer: float
    wer: float
    cer_no_diacritics: float
    wer_no_diacritics: float
    exact: float  # share of lines that are exactly right
    count: int

    def as_dict(self) -> dict[str, float]:
        return {
            "cer": self.cer,
            "wer": self.wer,
            "cer_nodiac": self.cer_no_diacritics,
            "wer_nodiac": self.wer_no_diacritics,
            "exact": self.exact,
        }


def _rates(refs: Sequence[str], hyps: Sequence[str]) -> tuple[float, float]:
    char_err = char_total = word_err = word_total = 0
    for r, h in zip(refs, hyps, strict=True):
        char_err += edit_distance(r, h)
        char_total += len(r)
        rw, hw = r.split(), h.split()
        word_err += edit_distance(rw, hw)
        word_total += len(rw)
    return char_err / max(char_total, 1), word_err / max(word_total, 1)


def score(refs: Sequence[str], hyps: Sequence[str]) -> Scores:
    refs = [charset.normalize(r) for r in refs]
    hyps = [charset.normalize(h) for h in hyps]
    cer, wer = _rates(refs, hyps)
    strip = charset.strip_diacritics
    cer_nd, wer_nd = _rates([strip(r) for r in refs], [strip(h) for h in hyps])
    exact = sum(r == h for r, h in zip(refs, hyps, strict=True)) / max(len(refs), 1)
    return Scores(cer, wer, cer_nd, wer_nd, exact, len(refs))
