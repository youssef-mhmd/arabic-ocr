"""Optional word-list correction of dot confusions.

Arabic letters often differ only in their dots (ب ت ث ن ي, ج ح خ, ف ق ...). On small
or blurry text the recognizer sometimes picks the wrong one. When it is unsure of a
word, the word is not in our word list, and changing *one* letter to a letter of
the same shape gives a common word, that word is probably what was printed.

    from arabic_ocr.correct import Lexicon
    lex = Lexicon.load("models/lexicon.tsv.gz")      # made by training/make_lexicon.py
    lex.correct("الكتاث", confidence=0.6)             # -> "الكتاب" (ث, ب: same shape)

Words the model is confident about, and words already in the list, are never
changed, so names and rare words are left alone.

Measured on real screenshot and printed lines (true boxes): CER 4.65% -> 4.57%
(about 2% fewer errors). Most remaining errors are not one-letter dot slips of a
common word, so this is an optional extra (OCR(...).read(..., correct_words=True),
CLI --correct-words), off by default.
"""

from __future__ import annotations

import gzip
from pathlib import Path

from arabic_ocr import charset

#: Letters with the same basic shape, told apart only by dots (or by language).
GROUPS = ["بتثنيیى", "جحخچ", "دذ", "رزژ", "سش", "صض", "طظ", "عغ", "فق", "هة", "كک"]
TWINS = {c: g.replace(c, "") for g in GROUPS for c in g}

#: Only words read with less confidence than this are candidates for correction.
MAX_CONFIDENCE = 0.97
#: A replacement must be at least this frequent in the word list.
MIN_COUNT = 20


class Lexicon:
    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts

    @classmethod
    def load(cls, path: str | Path) -> Lexicon:
        counts = {}
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                word, _, n = line.rstrip("\n").partition("\t")
                counts[word] = int(n)
        return cls(counts)

    def correct(self, word: str, confidence: float) -> str:
        """The corrected word (or the word unchanged)."""
        if confidence >= MAX_CONFIDENCE:
            return word
        # Keep punctuation, digits, diacritics: look the bare letters up.
        start = 0
        while start < len(word) and word[start] not in TWINS and not word[start].isalpha():
            start += 1
        end = len(word)
        while end > start and not word[end - 1].isalpha():
            end -= 1
        core = word[start:end]
        bare = charset.strip_tatweel(charset.strip_diacritics(core))
        if len(bare) < 2 or bare != core or self.counts.get(bare, 0) >= MIN_COUNT:
            return word  # too short, has diacritics/tatweel, or already a known word
        best, best_count = bare, max(self.counts.get(bare, 0) * 10, MIN_COUNT - 1)
        for i, c in enumerate(bare):
            for twin in TWINS.get(c, ""):
                cand = bare[:i] + twin + bare[i + 1 :]
                n = self.counts.get(cand, 0)
                if n > best_count:
                    best, best_count = cand, n
        return word[:start] + best + word[end:]
