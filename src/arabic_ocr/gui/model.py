"""What the desktop app keeps for each file it read, and how it exports it.

No Qt here, so this part is easy to test.

The app runs OCR once with nothing removed (diacritics, tatweel and digits as
printed) and keeps that raw result. The output options are applied when the text
is shown or exported, so changing a setting updates every result instantly,
without reading the files again.

The user can also edit a page's text. Exports then use the edited text: each
edited line replaces the matching OCR line (same order), so JSON, hOCR, ALTO and
the searchable PDF keep the line boxes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from arabic_ocr.pipeline import Block, Document, Line, Page, Word, postprocess

#: Export formats: name -> (file extension, label shown in the save dialog).
FORMATS = {
    "txt": (".txt", "Text"),
    "json": (".json", "JSON"),
    "pdf": (".pdf", "Searchable PDF"),
    "hocr": (".hocr", "hOCR"),
    "alto": (".xml", "ALTO XML"),
}


@dataclass
class Options:
    """Output options (PRD FR-9). The OCR itself always keeps everything."""

    diacritics: bool = True
    tatweel: bool = True
    digits: str = "keep"  # keep | western | arabic

    def apply(self, text: str) -> str:
        return postprocess(text, self.diacritics, self.digits, self.tatweel)  # type: ignore[arg-type]


@dataclass
class Result:
    """One file (or pasted image) and what OCR found in it."""

    name: str
    source: Path | None  # None for a pasted image
    dpi: int  # PDF pages were rendered at this resolution
    pages: list[Page]  # raw OCR, one per page
    images: list[np.ndarray] | None = None  # kept only for pasted images
    edits: dict[int, str] = field(default_factory=dict)  # page index -> user's text

    def page_text(self, i: int, options: Options) -> str:
        """The text of page i as shown: the user's edit, or the OCR with options."""
        if i in self.edits:
            return self.edits[i]
        return options.apply(self.pages[i].text)

    def document(self, options: Options) -> Document:
        """A Document for export: options applied, the user's edits included."""
        pages = []
        for i, page in enumerate(self.pages):
            out = with_options(page, options)
            if i in self.edits:
                out = with_text(out, self.edits[i])
            pages.append(out)
        return Document(pages, self.name)

    def save(self, path: str | Path, fmt: str, options: Options) -> None:
        """Write the result in format `fmt` (a key of FORMATS)."""
        doc = self.document(options)
        if fmt == "pdf":
            from arabic_ocr.export.searchable_pdf import pages_to_pdf

            pages_to_pdf(self.source, doc.pages, path, dpi=self.dpi, images=self.images)
            return
        text = {
            "txt": lambda: doc.text,
            "json": doc.to_json,
            "hocr": doc.to_hocr,
            "alto": doc.to_alto,
        }[fmt]()
        Path(path).write_text(text + "\n", encoding="utf-8")

    def output_name(self, fmt: str) -> str:
        """File name for a batch export: the input's name with the format's extension."""
        stem = Path(self.name).stem or "page"
        return stem + FORMATS[fmt][0]


def with_options(page: Page, options: Options) -> Page:
    """A copy of `page` with the output options applied to every line and word."""
    page = copy.deepcopy(page)
    for line in page.lines:
        line.text = options.apply(line.text)
        for word in line.words:
            word.text = options.apply(word.text)
    return page


def with_text(page: Page, text: str) -> Page:
    """A copy of `page` whose lines say `text` (the user's edit).

    Edited lines are matched to OCR lines in order, ignoring empty lines. Extra
    edited lines are added to the last line; OCR lines with no edited line left
    are dropped. Words keep their boxes when the word count is unchanged;
    otherwise the line's box is shared out evenly between the new words.
    """
    page = copy.deepcopy(page)
    lines = page.lines
    edited = [t.strip() for t in text.splitlines() if t.strip()]
    if not lines:
        return page
    if len(edited) > len(lines):  # more lines than OCR found: keep all the text
        edited = [*edited[: len(lines) - 1], " ".join(edited[len(lines) - 1 :])]
    for line, new in zip(lines, edited, strict=False):
        if new != line.text:
            line.text = new
            line.words = _match_words(line, new.split())
    keep = {id(line) for line in lines[: len(edited)]}
    page.blocks = [Block([ln for ln in block.lines if id(ln) in keep]) for block in page.blocks]
    page.blocks = [b for b in page.blocks if b.lines]
    return page


def _match_words(line: Line, tokens: list[str]) -> list[Word]:
    if len(tokens) == len(line.words):
        return [Word(t, w.confidence, w.box) for t, w in zip(tokens, line.words, strict=True)]
    return even_words(line.box, tokens, line.confidence)


def even_words(box: np.ndarray, tokens: list[str], confidence: float) -> list[Word]:
    """Share a line box out between `tokens`, by length, right to left (Arabic)."""
    if not tokens:
        return []
    tl, tr, br, bl = np.asarray(box, np.float32)
    weights = np.array([len(t) + 1 for t in tokens], np.float32)  # +1: the space
    edges = np.concatenate([[0], np.cumsum(weights) / weights.sum()])
    words = []
    for t, a, b in zip(tokens, edges[:-1], edges[1:], strict=True):
        # a, b = fractions of the line measured from its right edge
        top_r, top_l = tr + (tl - tr) * a, tr + (tl - tr) * b
        bot_r, bot_l = br + (bl - br) * a, br + (bl - br) * b
        words.append(Word(t, confidence, np.float32([top_l, top_r, bot_r, bot_l])))
    return words
