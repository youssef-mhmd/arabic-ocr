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
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from arabic_ocr import charset

_ARABIC_LETTERS = frozenset(
    charset.ARABIC_LETTERS + charset.ARABIC_EXTRA_LETTERS + charset.PERSIAN_URDU_LETTERS
)


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

    def _take_words(self, words: list[str], start: int, target: int) -> str:
        line = words[start]
        for word in words[start + 1 :]:
            if len(line) + 1 + len(word) > target:
                break
            line += " " + word
        return line

    # --- main -----------------------------------------------------------------

    def sample(self, segment: str) -> str | None:
        """One line from a clean segment, or None if the segment can't give one."""
        cfg, rng = self.config, self.rng
        words = segment.split(" ")

        if rng.random() < cfg.short_prob:
            target = rng.randint(cfg.min_len, min(5, cfg.max_len))
        else:
            target = rng.randint(cfg.min_len, cfg.max_len)

        if rng.random() < cfg.number_prob:
            words = list(words)
            words.insert(rng.randint(0, len(words)), self.random_number())

        start = rng.randrange(len(words))
        line = self._take_words(words, start, target)

        if rng.random() < cfg.strip_diacritics_prob:
            line = charset.strip_diacritics(line)
            line = re.sub(" +", " ", line).strip()

        # A single word longer than max_len: cut it at the limit. The label stays
        # correct because we render exactly this cut string.
        if len(line) > cfg.max_len:
            line = line[: cfg.max_len].rstrip()
        if not line or not (cfg.min_len <= len(line) <= cfg.max_len):
            return None
        return line

    def lines(self, segments: Iterable[str]) -> Iterator[str]:
        """Yield one line per segment (skipping segments that give none)."""
        for seg in segments:
            line = self.sample(seg)
            if line is not None:
                yield line
