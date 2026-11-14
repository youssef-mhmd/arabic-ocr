"""Accuracy measures for OCR: CER and WER.

- **CER** (character error rate): how many single-character edits (insert, delete,
  replace) turn the prediction into the correct text, divided by the length of the
  correct text. 0.02 = 2 mistakes per 100 characters. Lower is better.
- **WER** (word error rate): the same, counting whole words.

Both are reported twice (PRD §9, CLAUDE.md):
- *with diacritics*: every tashkeel mark counts;
- *without diacritics*: marks are removed from both texts first, so only the
  letters, digits and punctuation count.

Texts are compared in logical order after normalization. Tatweel (kashida, ـ) is
removed from both texts first: it only stretches a letter join and carries no
meaning, so writing it or not is never counted as a mistake.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

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


def fast_edit_distance(a: str, b: str) -> int:
    """Same result as edit_distance, for long strings (whole pages): NumPy, one row
    of the table at a time.

    Row update: substitutions/deletions come from the previous row (vectorised);
    insertions chain along the row, cur[j] = min over k <= j of (tmp[k] + j - k),
    which is a running minimum of (tmp[k] - k), plus j.
    """
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    bb = np.frombuffer(b.encode("utf-32-le"), dtype=np.uint32)
    j = np.arange(len(b) + 1)
    prev = j.copy()
    for i, ch in enumerate(a, 1):
        tmp = np.empty_like(prev)
        tmp[0] = i
        tmp[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (bb != ord(ch)))
        prev = np.minimum.accumulate(tmp - j) + j
    return int(prev[-1])


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


_ALL_DIGITS_TO_WESTERN = str.maketrans(
    "".join(charset.ARABIC_INDIC_DIGITS) + "".join(charset.EASTERN_ARABIC_INDIC_DIGITS),
    "0123456789" * 2,
)


def ignore_conventions(text: str) -> str:
    """Remove differences that are transcription *conventions*, not misreadings.

    Used for real-world sets, whose transcriptions were typed by people following
    their own rules (decided with the owner, 2026-11-13):
    - digit style: the page prints ٣ but the transcriber typed 3 (the PRD makes
      digit style an output option anyway);
    - spaces before punctuation: the page prints `البصرة .` but the transcriber
      typed `البصرة.` (and likewise after opening brackets).
    """
    text = text.translate(_ALL_DIGITS_TO_WESTERN)
    text = re.sub(r"\s+([.،,:؛!؟?)\]»])", r"\1", text)
    return re.sub(r"([(\[«])\s+", r"\1", text)


def score(refs: Sequence[str], hyps: Sequence[str]) -> Scores:
    refs = [charset.strip_tatweel(charset.normalize(r)) for r in refs]
    hyps = [charset.strip_tatweel(charset.normalize(h)) for h in hyps]
    cer, wer = _rates(refs, hyps)
    strip = charset.strip_diacritics
    cer_nd, wer_nd = _rates([strip(r) for r in refs], [strip(h) for h in hyps])
    exact = sum(r == h for r, h in zip(refs, hyps, strict=True)) / max(len(refs), 1)
    return Scores(cer, wer, cer_nd, wer_nd, exact, len(refs))
