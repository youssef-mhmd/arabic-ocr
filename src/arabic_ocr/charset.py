"""The Arabic-script character set and text normalization.

This module is the single source of truth for which symbols the recognizer can
output. Training and inference must both use it, so a label written today still
decodes to the same text after a retrain.

Key ideas:
- Every symbol gets an integer id. The model predicts ids; `decode` turns them back
  into text.
- Id 0 is reserved for the CTC "blank" (a "no new character here" signal used by
  the recognizer's loss; see docs/learning/03-recognizer-and-ctc.md later).
- Text is always in logical Unicode order (the order it is typed). Never reverse it.
- Diacritics (tashkeel) are separate symbols, not merged into letters.

**Never reorder or remove existing symbols** once a model has been trained: that
silently changes what each id means. Only append new symbols at the end of a group
and retrain.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Literal

# ---------------------------------------------------------------------------
# Symbol groups
# ---------------------------------------------------------------------------


def _chars(first: int, last: int) -> list[str]:
    """All characters from code point `first` to `last`, inclusive."""
    return [chr(cp) for cp in range(first, last + 1)]


#: Core Arabic letters, including hamza forms (ء آ أ ؤ إ ئ) and ta marbuta / alef
#: maksura. U+0621–U+063A and U+0641–U+064A.
ARABIC_LETTERS: list[str] = _chars(0x0621, 0x063A) + _chars(0x0641, 0x064A)

#: Extra letters seen in classical/Quranic Arabic.
ARABIC_EXTRA_LETTERS: list[str] = [
    "ٱ",  # ٱ alef wasla
]

#: Persian and Urdu letters that are not in the core Arabic block.
PERSIAN_URDU_LETTERS: list[str] = [
    "پ",  # پ peh
    "چ",  # چ tcheh
    "ژ",  # ژ jeh
    "ک",  # ک keheh (Persian/Urdu kaf)
    "گ",  # گ gaf
    "ی",  # ی Farsi yeh
    "ٹ",  # ٹ tteh
    "ڈ",  # ڈ ddal
    "ڑ",  # ڑ rreh
    "ں",  # ں noon ghunna
    "ھ",  # ھ heh doachashmee
    "ہ",  # ہ heh goal
    "ۂ",  # ۂ heh goal with hamza above
    "ۀ",  # ۀ heh with yeh above
    "ە",  # ە ae
    "ے",  # ے yeh barree
    "ۓ",  # ۓ yeh barree with hamza above
]

TATWEEL = "ـ"  # ـ kashida: stretches a joined letter; carries no meaning

#: Tashkeel (diacritics). U+064B–U+0652 plus superscript alef U+0670, and the
#: combining hamza above/below, which survive NFC when they follow a letter that
#: has no precomposed hamza form (common in Persian/Urdu text).
DIACRITICS: list[str] = [
    *_chars(0x064B, 0x0652),  # ً ٌ ٍ َ ُ ِ ّ ْ  fathatan … sukun
    "ٰ",  # ٰ superscript (dagger) alef
    "ٔ",  # ٔ hamza above
    "ٕ",  # ٕ hamza below
]

ARABIC_INDIC_DIGITS: list[str] = _chars(0x0660, 0x0669)  # ٠–٩
EASTERN_ARABIC_INDIC_DIGITS: list[str] = _chars(0x06F0, 0x06F9)  # ۰–۹ (Persian/Urdu)
WESTERN_DIGITS: list[str] = _chars(0x0030, 0x0039)  # 0–9

ARABIC_PUNCTUATION: list[str] = [
    "،",  # ، comma
    "؛",  # ؛ semicolon
    "؟",  # ؟ question mark
    "٪",  # ٪ percent
    "٫",  # ٫ decimal separator
    "٬",  # ٬ thousands separator
    "۔",  # ۔ Urdu full stop
]

#: Neutral punctuation that commonly appears inside Arabic text.
COMMON_PUNCTUATION: list[str] = list(".:!\"'()[]«»-/%*+=")

SPACE = " "

#: The CTC blank token. Not a real character; it never appears in text.
BLANK = "<blank>"
BLANK_ID = 0

# Order matters: an id is a position in this list. Append only.
_GROUPS: list[list[str]] = [
    [SPACE],
    ARABIC_LETTERS,
    ARABIC_EXTRA_LETTERS,
    PERSIAN_URDU_LETTERS,
    [TATWEEL],
    DIACRITICS,
    ARABIC_INDIC_DIGITS,
    EASTERN_ARABIC_INDIC_DIGITS,
    WESTERN_DIGITS,
    ARABIC_PUNCTUATION,
    COMMON_PUNCTUATION,
]

#: All real symbols, in id order (id = index + 1, because id 0 is the blank).
SYMBOLS: list[str] = [ch for group in _GROUPS for ch in group]

_CHAR_TO_ID: dict[str, int] = {ch: i + 1 for i, ch in enumerate(SYMBOLS)}
_ID_TO_CHAR: dict[int, str] = {i: ch for ch, i in _CHAR_TO_ID.items()}

#: Number of classes the recognizer must output (symbols + the blank).
NUM_CLASSES: int = len(SYMBOLS) + 1

DIACRITIC_SET: frozenset[str] = frozenset(DIACRITICS)

# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------


class UnknownCharacterError(ValueError):
    """Raised when text contains a character outside the charset."""


def is_supported(ch: str) -> bool:
    """True if `ch` is a single character in the charset."""
    return ch in _CHAR_TO_ID


def encode(text: str, unknown: Literal["error", "skip"] = "error") -> list[int]:
    """Turn text into a list of symbol ids (no blanks).

    `text` should already be normalized with `normalize`. With `unknown="skip"`,
    characters outside the charset are dropped instead of raising an error.
    """
    ids = []
    for ch in text:
        i = _CHAR_TO_ID.get(ch)
        if i is None:
            if unknown == "skip":
                continue
            raise UnknownCharacterError(f"character {ch!r} (U+{ord(ch):04X}) is not in the charset")
        ids.append(i)
    return ids


def decode(ids: Iterable[int]) -> str:
    """Turn symbol ids back into text. Blank ids (0) are ignored.

    This does not collapse repeats; that is the CTC decoder's job.
    """
    return "".join(_ID_TO_CHAR[i] for i in ids if i != BLANK_ID)


def unsupported_chars(text: str) -> set[str]:
    """Characters in `text` that are not in the charset (useful when cleaning corpora)."""
    return {ch for ch in text if ch not in _CHAR_TO_ID}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Invisible formatting characters that do not appear in an image, so the model can
# never learn to output them: zero-width space/joiners, direction marks and
# embeddings/isolates, the Arabic letter mark, and the byte-order mark.
# Note: ZWNJ (U+200C) is meaningful in Persian spelling, but its only visible effect
# is a break in letter joining; we drop it from labels.
_INVISIBLE = re.compile("[؜​-‏‪-‮⁠-⁩﻿]")

_WHITESPACE = re.compile(r"\s+")


def _is_presentation_form(ch: str) -> bool:
    cp = ord(ch)
    return 0xFB50 <= cp <= 0xFDFF or 0xFE70 <= cp <= 0xFEFF


def _unpresent(ch: str) -> str:
    """Map one Arabic presentation-form character to base letters (e.g. ﻻ → لا)."""
    base = unicodedata.normalize("NFKC", ch)
    # The isolated tashkeel forms (U+FE70–U+FE7F) decompose to "space + mark";
    # we only want the mark.
    if 0xFE70 <= ord(ch) <= 0xFE7F:
        base = base.lstrip(" ")
    return base


def normalize(text: str) -> str:
    """Normalize text so the same visible text always has the same label.

    Steps, in order:
    1. Map Arabic presentation forms (U+FB50–U+FDFF, U+FE70–U+FEFF) to base letters.
       These are legacy "already shaped" characters; e.g. ﻻ becomes ل + ا.
    2. Remove invisible formatting characters (direction marks, zero-width chars).
    3. Unicode NFC. This composes hamza forms (ا + ٔ → أ) and puts stacked
       diacritics in one standard order (e.g. shadda + fatha → fatha + shadda), so
       the two typing orders become identical.
    4. Turn every run of whitespace into a single space and trim the ends.

    It never reverses text, never removes diacritics, and never changes one letter
    into another (e.g. أ stays أ, ي stays ي).
    """
    text = "".join(_unpresent(ch) if _is_presentation_form(ch) else ch for ch in text)
    text = _INVISIBLE.sub("", text)
    text = unicodedata.normalize("NFC", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Output options (applied after recognition, or when comparing for evaluation)
# ---------------------------------------------------------------------------

_DIGIT_TABLES = {
    "western": str.maketrans(
        "".join(ARABIC_INDIC_DIGITS + EASTERN_ARABIC_INDIC_DIGITS), "0123456789" * 2
    ),
    "arabic": str.maketrans(
        "".join(WESTERN_DIGITS + EASTERN_ARABIC_INDIC_DIGITS), "".join(ARABIC_INDIC_DIGITS) * 2
    ),
}


def strip_diacritics(text: str) -> str:
    """Remove tashkeel marks (used for the "diacritics ignored" CER and the output option).

    The combining hamza marks are kept: they are part of the letter, not optional
    vowel marks.
    """
    hamza_marks = {"ٔ", "ٕ"}
    return "".join(ch for ch in text if ch not in DIACRITIC_SET or ch in hamza_marks)


def strip_tatweel(text: str) -> str:
    """Remove tatweel (kashida) stretching characters."""
    return text.replace(TATWEEL, "")


def convert_digits(text: str, style: Literal["keep", "western", "arabic"]) -> str:
    """Convert digits to Western (0–9) or Arabic-Indic (٠–٩), or leave them as they are."""
    if style == "keep":
        return text
    return text.translate(_DIGIT_TABLES[style])


def symbols_in_order() -> Sequence[str]:
    """The full class list, including the blank at index 0 (handy for model export)."""
    return [BLANK, *SYMBOLS]


def fingerprint() -> str:
    """Short hash of the symbol list. A model must be trained and run with the same one."""
    import hashlib

    return hashlib.sha1("\n".join(SYMBOLS).encode("utf-8")).hexdigest()[:12]
