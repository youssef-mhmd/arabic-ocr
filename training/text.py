"""Turn raw corpus text into clean, labeled text lines for rendering.

Two steps:
1. `clean_text` keeps only stretches of text that the charset can represent and that
   are mostly Arabic-script. Everything else (Latin words, emoji, wiki markup) is cut
   out.
2. `LineSampler` cuts clean text into lines of random length (1-60 characters by
   default), sometimes strips diacritics, and sometimes inserts numbers, so the
   training data covers the variety the recognizer will see.

Every line is in logical Unicode order and passes `charset.normalize`.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from arabic_ocr import charset
from training.render import RIGHT_JOINING

_ARABIC_LETTERS = frozenset(
    charset.ARABIC_LETTERS + charset.ARABIC_EXTRA_LETTERS + charset.PERSIAN_URDU_LETTERS
)
#: Phrases that print as a single glyph in many books and fonts. Our charset rule
#: maps these presentation forms back to their text (NFKC), so the *label* is always
#: the phrase; only the rendered image uses the glyph.
LIGATURES = {
    "صلى الله عليه وسلم": "ﷺ",  # ﷺ
    "جل جلاله": "ﷻ",  # ﷻ
}


def ligate(line: str) -> str:
    """The text to *draw*: honorific phrases replaced by their ligature glyph."""
    for phrase, glyph in LIGATURES.items():
        line = line.replace(phrase, glyph)
    return line


#: Letters that connect to the letter after them; only these can be stretched.
DUAL_JOINING = _ARABIC_LETTERS - RIGHT_JOINING
_LETTERS_AND_DIGITS = _ARABIC_LETTERS | frozenset(
    charset.WESTERN_DIGITS + charset.ARABIC_INDIC_DIGITS + charset.EASTERN_ARABIC_INDIC_DIGITS
)
# A run of tatweel with a space or line edge on both sides.
_LOOSE_TATWEEL = re.compile(rf"(?<!\S){charset.TATWEEL}+(?!\S)")


def _drop_orphan_marks(text: str) -> str:
    """Remove diacritics that are not attached to a letter (e.g. after a space).

    A mark with nothing to sit on cannot be rendered sensibly, so its label would
    be wrong.
    """
    out: list[str] = []
    for ch in text:
        if ch in charset.DIACRITIC_SET:
            # walk back over earlier marks to find the base character
            j = len(out) - 1
            while j >= 0 and out[j] in charset.DIACRITIC_SET:
                j -= 1
            if j < 0 or out[j] not in _ARABIC_LETTERS:
                continue
        out.append(ch)
    return "".join(out)


def arabic_ratio(text: str) -> float:
    """Share of non-space characters that are Arabic-script letters or diacritics."""
    chars = [c for c in text if c != " "]
    if not chars:
        return 0.0
    arabic = sum(c in _ARABIC_LETTERS or c in charset.DIACRITIC_SET for c in chars)
    return arabic / len(chars)


def clean_text(raw: str, min_arabic_ratio: float = 0.5) -> list[str]:
    """Split raw text into clean segments that contain only charset symbols.

    Unsupported characters (and line breaks) act as cut points. Segments that are
    mostly not Arabic (e.g. a run of digits and punctuation) are dropped.
    """
    segments: list[str] = []
    # Clean line by line: normalize() would turn line breaks into spaces and join
    # unrelated lines (e.g. a heading and the next paragraph).
    for raw_line in raw.splitlines():
        line = charset.normalize(raw_line)
        current: list[str] = []
        for ch in line:
            if charset.is_supported(ch):
                current.append(ch)
            else:
                segments.append("".join(current))
                current = []
        segments.append("".join(current))

    cleaned = []
    for seg in segments:
        seg = _drop_orphan_marks(seg)
        # Free-standing tatweel (e.g. the decoration in "ـــ الفصل ـــ") is not
        # attached to any letter; its length on the page has nothing to do with how
        # many tatweel characters were typed, so it can't be labeled reliably.
        seg = _LOOSE_TATWEEL.sub("", seg)
        seg = re.sub(" +", " ", seg).strip()
        if seg and arabic_ratio(seg) >= min_arabic_ratio:
            cleaned.append(seg)
    return cleaned


@dataclass
class SamplerConfig:
    min_len: int = 1
    max_len: int = 60
    #: chance of removing diacritics from a line that has them
    strip_diacritics_prob: float = 0.5
    #: chance of inserting a random number into a line
    number_prob: float = 0.1
    #: chance of the line being a short "fragment" (1-5 chars), e.g. a page number or label
    short_prob: float = 0.05
    # --- added for v2 (off by default) ---
    #: chance of putting «guillemets» around a few words (common in printed books)
    quote_prob: float = 0.0
    #: chance of adding a footnote mark like (٣) after a word
    footnote_prob: float = 0.0
    #: chance of stretching some letter joins with tatweel (kashida), as justified
    #: print does; the label contains the tatweel too
    kashida_prob: float = 0.0
    #: chance of adding an honorific phrase (e.g. صلى الله عليه وسلم) after a word;
    #: the dataset builder may then draw it as one ligature glyph (ﷺ), see LIGATURES
    honorific_prob: float = 0.0


class LineSampler:
    """Cuts clean text segments into training lines.

    Uses its own seeded random generator, so the same seed and input always give
    the same lines.
    """

    def __init__(self, config: SamplerConfig | None = None, seed: int = 0) -> None:
        self.config = config or SamplerConfig()
        self.rng = random.Random(seed)

    # --- pieces ---------------------------------------------------------------

    def random_number(self) -> str:
        """A number like 7, 2026, 3.5, or ١٤٤٥ in a random digit style."""
        rng = self.rng
        kind = rng.random()
        if kind < 0.6:
            num = str(rng.randint(0, 9999))
        elif kind < 0.85:
            num = str(rng.randint(1000, 9_999_999))
        else:
            num = f"{rng.randint(0, 999)}.{rng.randint(0, 99)}"
        style = rng.choice(["western", "arabic", "eastern"])
        if style == "western":
            return num
        if style == "arabic":
            return charset.convert_digits(num, "arabic").replace(".", "٫")
        eastern = str.maketrans("0123456789", "".join(charset.EASTERN_ARABIC_INDIC_DIGITS))
        return num.translate(eastern).replace(".", "٫")

    def footnote_mark(self) -> str:
        """A footnote reference like (٣) or (12), glued to the word before it."""
        n = str(self.rng.randint(1, 20))
        if self.rng.random() < 0.6:
            n = charset.convert_digits(n, "arabic")
        return f"({n})"

    def stretch(self, line: str) -> str:
        """Insert 1-3 tatweels after a few letters that join the next letter."""
        out = []
        for i, ch in enumerate(line):
            out.append(ch)
            nxt = line[i + 1] if i + 1 < len(line) else ""
            if ch in DUAL_JOINING and nxt.isalpha() and self.rng.random() < 0.15:
                out.append(charset.TATWEEL * self.rng.randint(1, 3))
        return "".join(out)

    def _take_words(self, words: list[str], start: int, target: int) -> str:
        line = words[start]
        for word in words[start + 1 :]:
            if len(line) + 1 + len(word) > target:
                break
            line += " " + word
        return line

    # --- main -----------------------------------------------------------------

    def _start_for(self, words: list[str], target: int) -> int:
        """A random start word that leaves enough text to reach `target` characters.

        Starting anywhere would often land near the end of a paragraph and give a
        line much shorter than intended.
        """
        remaining, total = [], -1
        for word in reversed(words):  # characters from each word to the end
            total += len(word) + 1
            remaining.append(total)
        remaining.reverse()
        good = [i for i, r in enumerate(remaining) if r >= target]
        return self.rng.choice(good) if good else 0

    def sample(self, segment: str, more: Callable[[], str] | None = None) -> str | None:
        """One line from a clean segment, or None if the segment can't give one.

        `more`, if given, returns another paragraph (from the same split). It is used
        when `segment` is too short for the chosen length, so that short paragraphs
        (titles, list items) don't make every line short.
        """
        cfg, rng = self.config, self.rng

        if rng.random() < cfg.short_prob:
            target = rng.randint(cfg.min_len, min(5, cfg.max_len))
        else:
            target = rng.randint(cfg.min_len, cfg.max_len)

        # Decide on diacritics first: stripping them shortens the text, so the
        # length target must be measured on the final text.
        strip = rng.random() < cfg.strip_diacritics_prob
        text = segment
        for _ in range(5):
            if strip:
                text = re.sub(" +", " ", charset.strip_diacritics(text)).strip()
            if len(text) >= target or more is None:
                break
            text = text + " " + more()
        if not text:
            return None
        words = text.split(" ")

        if rng.random() < cfg.number_prob:
            words.insert(rng.randint(0, len(words)), self.random_number())
        # (each new option only draws random numbers when it is switched on, so old
        # configs keep giving exactly the same lines)
        if cfg.quote_prob and rng.random() < cfg.quote_prob:
            i = rng.randrange(len(words))
            j = min(len(words) - 1, i + rng.randint(0, 4))
            words[i], words[j] = "«" + words[i], words[j] + "»"
        if cfg.honorific_prob and rng.random() < cfg.honorific_prob:
            phrase = rng.choice(list(LIGATURES))
            words.insert(rng.randint(1, len(words)), phrase)
        if cfg.footnote_prob and rng.random() < cfg.footnote_prob:
            mark = self.footnote_mark()
            i = rng.randrange(len(words))
            words[i] = words[i] + mark

        line = self._take_words(words, self._start_for(words, target), target)

        # A single word longer than max_len: cut it at the limit. The label stays
        # correct because we render exactly this cut string.
        if len(line) > cfg.max_len:
            line = line[: cfg.max_len].rstrip()
        if cfg.kashida_prob and rng.random() < cfg.kashida_prob:
            line = self.stretch(line)
        if not line or not (cfg.min_len <= len(line) <= cfg.max_len):
            return None
        # A line with no letters or digits (e.g. just "." or "«") is not worth
        # training on.
        if not any(c in _LETTERS_AND_DIGITS for c in line):
            return None
        return line

    def lines(self, segments: Iterable[str]) -> Iterator[str]:
        """Yield one line per segment (skipping segments that give none)."""
        for seg in segments:
            line = self.sample(seg)
            if line is not None:
                yield line
