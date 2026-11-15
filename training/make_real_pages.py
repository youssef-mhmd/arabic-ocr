"""Build the *real page* test sets for end-to-end evaluation (M3).

    uv run python -m training.make_real_pages printed     # ~60 scanned book pages
    uv run python -m training.make_real_pages printed --site fa -n 40   # Persian; also ur
    uv run python -m training.make_real_pages screens     # ~50 browser screenshots

Both write `data/real/<set>/images/<id>.(jpg|png)` + `labels/<id>.json`, read by
`training/eval_pages.py --real data/real/<set>`. Evaluation only: never train on them.

**printed** (`data/real/pages_printed`): pages from Arabic Wikisource
(ar.wikisource.org). Volunteers transcribe scanned public-domain books page by
page; we take pages marked "proofread" (صححت), with the scan from Wikimedia
Commons. The scans are real: old paper, show-through, book edges, old typefaces.
- Text (CC BY-SA) comes as wikitext. Pages with tables, images, footnotes or
  unknown templates are skipped; simple header templates (centred page numbers,
  running heads) become plain text.
- Our training corpus includes a 2023 Wikisource dump. A page is skipped if any
  of a few sample phrases from it appear in our training text, so the recognizer
  has not seen these sentences.
- At most 5 pages per book, so no single typeface dominates.
- Labels have no line boxes ("lines": null); scoring compares whole-page text.
  Transcriptions are proofread but not perfect, and sometimes follow their own
  conventions (e.g. typing ي where the print has ى); expect a little noise.

**screens** (`data/real/pages_screens_v2`; v1 = `pages_screens`, made with --mixed
--name pages_screens): web-like pages (articles, sidebars, search
results, chat, settings, cards) drawn by Microsoft Edge with Windows system fonts
at common screen scalings, captured as screenshots. The text comes from the
corpora's *test* split. A script inside each page lets the browser report every
line's words and position, so the labels (text, box, reading order) are exact.
Windows only (needs Edge and the Windows fonts); built once on the owner's laptop.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from arabic_ocr import charset

ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = ROOT / "data" / "real"


def user_agent() -> str:
    """Wikimedia's User-Agent policy (https://w.wiki/4wJS) requires automated
    clients to give a contact (e-mail or web page); without one, requests are
    refused. The contact comes from the ARABIC_OCR_CONTACT environment variable,
    so it is never written into the repository."""
    contact = os.environ.get("ARABIC_OCR_CONTACT", "").strip()
    if not contact:
        sys.exit(
            "Set ARABIC_OCR_CONTACT to an e-mail address or web page first: Wikimedia "
            "requires a contact in the User-Agent of automated downloads."
        )
    return f"arabic-ocr-dataset-builder/0.1 ({contact}; offline OCR research) python-urllib"


# --------------------------------------------------------------------------
# printed: Arabic Wikisource
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Site:
    """One Wikisource language edition."""

    lang: str  # wiki language code, also used in local upload URLs
    category: str  # the "proofread" quality category
    ns: int  # the Page: namespace number
    folder: str  # output folder under data/real
    corpora: tuple[str, ...]  # our training corpora to check for overlap

    @property
    def host(self) -> str:
        return f"https://{self.lang}.wikisource.org/w/"


SITES = {
    "ar": Site("ar", "صححت", 104, "pages_printed", ("wikisource_ar", "wikisource_ar_tashkeel")),
    "fa": Site("fa", "نمونه‌خوانی‌شده", 104, "pages_printed_fa", ("wikipedia_fa",)),
    "ur": Site("ur", "پروف خوانی شدہ", 250, "pages_printed_ur", ("wikipedia_ur",)),
}
THUMB_WIDTH = 1920  # largest standard Wikimedia thumbnail size that works for PDF pages
#: Templates that only place or style text: replaced by their text.
TEXT_TEMPLATES = {
    "وسط", "center", "c", "يمين", "يسار", "rh", "runningheader", "رأس", "رص",  # placement
    "ع", "ع2", "ع3", "ع4", "ع5", "ع6", "مض", "x-larger", "larger", "smaller",  # style
    "أبيات",  # verse; its two half-lines are separated by a double backslash
    # Persian / Urdu Wikisource
    "سرصفحه", "وسط‌چین", "زیرخط", "بزرگتر", "بزرگتر۱", "بزرگتر۲", "توخالی", "big",
    "سصم", "ود", "و", "ب", "کوچکتر", "اندازه قلم", "کذا",
}  # fmt: skip
#: Templates that only draw rules or spacing, or mark structure: dropped.
BLANK_TEMPLATES = {
    "rule", "dhr", "ص", "سطر", "فراغات", "nop", "خط‌کش", "خط کش",
    # Persian: rules, footnote dividers, spacing, poem/paragraph markers
    "خطکش", "خطکش-پانویس-کوچک", "پانویس-کوچک", "فاصله", "شعر", "پایان شعر", "بند", "ته",
}  # fmt: skip
#: Text templates that form lines of their own (headings, running heads, verses).
BLOCK_TEMPLATES = {
    "وسط", "center", "c", "يمين", "يسار", "rh", "runningheader", "رأس", "رص", "أبيات",
    "سرصفحه", "وسط‌چین", "سصم", "ود", "و", "ب",
}  # fmt: skip
#: Templates whose first argument is the printed text and the rest an editor's
#: note: {{کذا|بر|به خط الحاقی}} ("sic") prints only "بر".
FIRST_ARG_TEMPLATES = {"کذا"}
#: Arguments that are sizes, not text: {{اندازه قلم|200%|...}}.
_SIZE_ARG = re.compile(r"^\d+(\.\d+)?(%|em|px|pt)$")
#: Running headers keep their named values too, e.g. {{رص|وسط=( ٢٢٤ )}}.
NAMED_TEXT_TEMPLATES = {"رص", "rh", "runningheader", "رأس", "سرصفحه"}
#: Footnotes, tables, pictures, formulas: the page is skipped (the scan's footnote
#: area or picture would not match the text). {{حا}} is an inline footnote.
SKIP_MARKERS = [
    "{|", "<ref", "<math", "<table", "[[ملف:", "[[file:", "[[صورة:", "[[image:", "{{حا",
]  # fmt: skip


def _get(url: str, retries: int = 6) -> bytes:
    """Download with retries. On "429 Too Many Requests" wait as long as the server
    asks (Retry-After), or longer each time."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent()})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 10 * 2**attempt
        except OSError:
            if attempt == retries - 1:
                raise
            wait = 5 * 2**attempt
        print(f"  (server busy, waiting {wait} s)", flush=True)
        time.sleep(wait)
    raise AssertionError("unreachable")


def api(site: Site, **params) -> dict:
    params.update(format="json", formatversion=2)
    time.sleep(1.0)  # be polite: Wikimedia rate-limits fast anonymous clients
    return json.loads(_get(site.host + "api.php?" + urllib.parse.urlencode(params)))


def proofread_titles(cache: Path, site: Site) -> list[str]:
    """All page titles in the "proofread" category (cached after the first run)."""
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    titles, cont = [], {}
    while True:
        q = api(
            site, action="query", list="categorymembers", cmtitle="Category:" + site.category,
            cmnamespace=site.ns, cmlimit=500, **cont,
        )  # fmt: skip
        titles += [m["title"] for m in q["query"]["categorymembers"]]
        print(f"  {len(titles)} titles", flush=True)
        if "continue" not in q:
            break
        cont = {"cmcontinue": q["continue"]["cmcontinue"]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(titles, ensure_ascii=False), encoding="utf-8")
    return titles


def _template_text(match: re.Match) -> str:
    parts = [p.strip() for p in match.group(1).split("|")]
    name = parts[0].lower()
    if name in BLANK_TEMPLATES:
        return ""
    if name not in TEXT_TEMPLATES:
        raise ValueError(f"template {parts[0]}")
    args = []
    for p in parts[1:]:
        if "=" in p:
            if name not in NAMED_TEXT_TEMPLATES:
                continue  # a setting such as م=س, not text
            p = p.partition("=")[2].strip()
        if p and not _SIZE_ARG.match(p):
            args.append(re.sub(r"\\+", " ", p))  # verse half-line separator
    if name in FIRST_ARG_TEMPLATES:
        args = args[:1]
    if not args:
        return ""
    # Headings, running heads and verses are lines of their own; styling inside a
    # line (bold, sic, font size) must not break the line.
    sep = "\n" if name in BLOCK_TEMPLATES else ""
    return sep + " ".join(args) + sep


def wikitext_to_text(wikitext: str, drop_footnotes: bool = False) -> str | None:
    """Plain text of a proofread page, or None if the page is not simple enough.

    `drop_footnotes`: remove <ref> footnotes instead of rejecting the page. Right
    for training lines (lines are matched one by one; footnote lines just don't
    match), wrong for test pages (the scan's footnote area would be missing).
    """
    if drop_footnotes:
        wikitext = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", wikitext, flags=re.S | re.I)
        wikitext = re.sub(r"<references[^>]*/>|\{\{(?:پانویس|reflist)[^{}]*\}\}", "", wikitext)
    low = wikitext.lower()
    if any(m in low for m in SKIP_MARKERS):
        return None
    text = re.sub(r"<pagequality[^>]*/>", "", wikitext)
    text = re.sub(r"<section[^>]*/?>", "", text)  # labelled-section markers
    text = re.sub(r"</?noinclude>", "\n", text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"</?(poem|div|span|p|small|big|b|i|u|center|sup|sub)[^>]*>", "", text)
    if "<" in text and re.search(r"<[a-zA-Z/]", text):
        return None  # other markup we don't understand
    try:
        for _ in range(3):  # nested templates, innermost first
            text = re.sub(r"\{\{([^{}]*)\}\}", _template_text, text)
    except ValueError:
        return None
    if "{{" in text or "}}" in text:
        return None
    # Category links are page metadata, not printed text: drop them entirely.
    text = re.sub(r"\[\[\s*(?:category|تصنيف|رده|زمرہ)\s*:[^\]]*\]\]", "", text, flags=re.I)
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)  # [[target|label]] -> label
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    text = text.replace("'''", "").replace("''", "").replace("&nbsp;", " ")
    lines = [ln.strip() for ln in text.splitlines()]
    # List items (*, #) and indents (:) are real printed lines: drop only the
    # marker. Magic words like __NOTOC__ are not text.
    lines = [re.sub(r"^[*#:;]+\s*", "", ln) for ln in lines if not ln.startswith("__")]
    # normalize() also collapses whitespace, so apply it per line to keep the breaks.
    lines = [charset.normalize(ln) for ln in lines]
    return "\n".join(ln for ln in lines if ln) or None


def _arabic_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum("\u0600" <= c <= "\u06ff" for c in letters) / max(len(letters), 1)


class TrainingText:
    """Checks whether a phrase occurs in our training text."""

    def __init__(self, corpora: tuple[str, ...] = ("wikisource_ar", "wikisource_ar_tashkeel")):
        self.blobs = []
        for name in corpora:
            path = ROOT / "data" / "corpora" / name / "train.txt"
            if path.exists():
                print(f"  loading {path.relative_to(ROOT)} for the overlap check", flush=True)
                self.blobs.append(path.read_bytes())

    def seen(self, text: str, n: int = 4) -> bool:
        """True if any of `n` 6-word phrases from `text` is in the training text."""
        words = charset.strip_diacritics(text).split()
        if len(words) < 12:
            return False
        starts = np.linspace(0, len(words) - 6, n).astype(int)
        for s in starts:
            phrase = " ".join(words[s : s + 6]).encode("utf-8")
            if any(phrase in blob for blob in self.blobs):
                return True
        return False


def raw_wikitext(title: str, site: Site = SITES["ar"]) -> str:
    """A page's wikitext via the plain "raw" address (cached by Wikimedia; the API
    rate-limits anonymous clients much more strictly)."""
    time.sleep(1.0)
    url = site.host + "index.php?action=raw&title=" + urllib.parse.quote(title.replace(" ", "_"))
    return _get(url).decode("utf-8")


def scan_urls(page_title: str, site: Site = SITES["ar"]) -> list[str]:
    """Candidate URLs of the scan behind a page title: 'صفحة:<file>/<n>' (a page of a
    PDF/DjVu) or 'برگه:<image file>' (a single image, common on Persian Wikisource).

    Wikimedia stores a file under /<a>/<ab>/ where "ab" are the first two hex digits
    of the MD5 of its name, so the URL can be computed without an API call. Most
    scans are on Commons; some are uploaded to the Wikisource itself.
    """
    file_part, _, number = page_title.partition(":")[2].rpartition("/")
    # Persian Wikisource writes page numbers in Persian digits (.../۸): the URL needs 8.
    number = charset.convert_digits(number, "western")
    if not (number.isascii() and number.isdigit()):  # a single image file: no page number
        file_part, number = page_title.partition(":")[2], None
    name = file_part.replace(" ", "_")
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    q = urllib.parse.quote(name)
    bases = [
        "https://upload.wikimedia.org/wikipedia/commons",
        f"https://upload.wikimedia.org/wikisource/{site.lang}",
    ]
    if number is None:  # the original image (it may be smaller than a thumbnail)
        return [f"{b}/{h[0]}/{h[:2]}/{q}" for b in bases]
    tail = f"{h[0]}/{h[:2]}/{q}/page{number}-{THUMB_WIDTH}px-{q}.jpg"
    return [f"{b}/thumb/{tail}" for b in bases]


def download_scan(page_title: str, site: Site = SITES["ar"]) -> tuple[str, bytes] | None:
    for url in scan_urls(page_title, site):
        try:
            return url, _get(url)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
    return None


def build_printed(n_pages: int, per_book: int, seed: int, site: Site = SITES["ar"]) -> None:
    out = REAL_DIR / site.folder
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    titles = proofread_titles(out / "candidates.json", site)
    books: dict[str, list[str]] = {}
    for t in titles:
        books.setdefault(t.rpartition("/")[0], []).append(t)
    names = sorted(books)
    rng = random.Random(seed)
    rng.shuffle(names)
    print(f"{len(titles)} proofread pages in {len(names)} books")

    corpus = TrainingText(site.corpora)
    done = len(list((out / "labels").glob("*.json")))
    skipped: dict[str, int] = {}
    for book in names:
        if done >= n_pages:
            break
        pages = books[book][:]
        rng.shuffle(pages)
        taken = 0
        for title in pages[: per_book * 4]:  # look at a few candidates per book
            if taken >= per_book or done >= n_pages:
                break
            key = "p" + hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
            if (out / "labels" / f"{key}.json").exists():
                taken += 1
                continue
            wikitext = raw_wikitext(title, site)
            text = wikitext_to_text(wikitext)
            reason = None
            if text is None:
                reason = "markup"
            elif len(text) < 400 or text.count("\n") < 7:
                reason = "short"
            elif _arabic_share(text) < 0.9:
                reason = "not Arabic"
            elif charset.unsupported_chars(" ".join(text.splitlines())):
                reason = "unsupported characters"
            elif corpus.seen(text):
                reason = "in training text"
            if reason:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            got = download_scan(title, site)
            if got is None:
                skipped["no scan"] = skipped.get("no scan", 0) + 1
                continue
            url, data = got
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                skipped["bad image"] = skipped.get("bad image", 0) + 1
                continue
            (out / "images" / f"{key}.jpg").write_bytes(data)
            label = {
                "kind": "printed",
                "source": title,
                "url": f"https://{site.lang}.wikisource.org/wiki/" + urllib.parse.quote(title),
                "language": site.lang,
                "image_url": url,
                "license": "text CC BY-SA 4.0 (Wikisource); scan public domain (Commons)",
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
                "text": text,
                "lines": None,
            }
            (out / "labels" / f"{key}.json").write_text(
                json.dumps(label, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            taken += 1
            done += 1
            print(f"  [{done}/{n_pages}] {title}", flush=True)
    print(f"Done: {done} pages in {out}. Skipped: {skipped}")


# --------------------------------------------------------------------------
# screens: browser screenshots
# --------------------------------------------------------------------------

EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
#: Fonts apps, web pages and documents use (the screenshot target is measured on these).
UI_FONTS = [
    "Segoe UI", "Tahoma", "Arial", "Times New Roman", "Traditional Arabic",
    "Simplified Arabic", "Sakkal Majalla", "Microsoft Sans Serif", "Calibri",
]  # fmt: skip
#: Calligraphic, print-shop and monospace fonts, rare on screens: a stress test only.
RARE_FONTS = ["Courier New", "Urdu Typesetting", "Aldhabi", "Arabic Typesetting"]
FONTS = UI_FONTS + RARE_FONTS  # (v1 and v2 sets were made from all 13)
FONT_SETS = {"all": FONTS, "ui": UI_FONTS, "rare": RARE_FONTS}
# (text, muted text, background, panel, accent)
THEMES = [
    ("#202124", "#5f6368", "#ffffff", "#f1f3f4", "#1a0dab"),
    ("#111111", "#555555", "#fafafa", "#ffffff", "#0b57d0"),
    ("#e8eaed", "#9aa0a6", "#202124", "#303134", "#8ab4f8"),
    ("#d4d4d4", "#a0a0a0", "#1e1e1e", "#2b2b2b", "#4fc1ff"),
    ("#1f2937", "#6b7280", "#f9fafb", "#ffffff", "#047857"),
    ("#3c2f1e", "#7a6650", "#fdf6e3", "#f5ecd5", "#9c4221"),
]
# (CSS width, CSS height, device scale factor): laptop, desktop, scaled Windows, phone
VIEWPORTS = [
    (1366, 768, 1.0), (1536, 864, 1.25), (1920, 1080, 1.0), (1280, 720, 1.5),
    (1440, 900, 1.0), (412, 915, 2.625), (390, 844, 3.0),
]  # fmt: skip
KINDS = ["article", "sidebar", "search", "chat", "settings", "cards"]

LABEL_SCRIPT = """
<script>
// Group the word spans into visual lines and report each line's words and box,
// relative to the fixed-size #page frame. Lines that do not fit completely inside
// the frame are hidden, so the screenshot only shows labelled text.
(function () {
  const frame = document.getElementById('page').getBoundingClientRect();
  const W = frame.width, H = frame.height;
  const words = [...document.querySelectorAll('.w')];
  const lines = [];
  let cur = null;
  for (const w of words) {
    const b = w.getBoundingClientRect();
    const r = {left: b.left - frame.left, right: b.right - frame.left,
               top: b.top - frame.top, bottom: b.bottom - frame.top, height: b.height};
    // Same line: same element, and overlapping the line vertically by more than
    // half (words in a fallback font sit a little higher or lower).
    const overlap = cur ? Math.min(r.bottom, cur.box[3]) - Math.max(r.top, cur.box[1]) : 0;
    const same = cur && w.parentElement === cur.parent &&
      overlap > 0.5 * Math.min(r.height, cur.box[3] - cur.box[1]);
    if (!same) { cur = {parent: w.parentElement, top: r.top, words: [], spans: [],
                        box: [r.left, r.top, r.right, r.bottom]}; lines.push(cur); }
    cur.words.push(w.textContent); cur.spans.push(w);
    cur.box = [Math.min(cur.box[0], r.left), Math.min(cur.box[1], r.top),
               Math.max(cur.box[2], r.right), Math.max(cur.box[3], r.bottom)];
  }
  const out = [];
  for (const ln of lines) {
    const [x0, y0, x1, y1] = ln.box;
    if (x0 < 0 || y0 < 0 || x1 > W || y1 > H) {
      ln.spans.forEach(s => s.style.visibility = 'hidden');
      continue;
    }
    out.push({text: ln.words.join(' '), box: ln.box});
  }
  const pre = document.createElement('pre');
  pre.id = 'labels'; pre.style.display = 'none';
  pre.textContent = JSON.stringify({dpr: window.devicePixelRatio, lines: out});
  document.body.appendChild(pre);
})();
</script>
"""


def _words(text: str) -> str:
    return " ".join(f'<span class="w">{html.escape(w)}</span>' for w in text.split())


class Texts:
    """Unseen text: paragraphs from the corpora's test split."""

    def __init__(self, rng: random.Random) -> None:
        from training.corpora import CORPORA_DIR
        from training.make_dataset import TextPicker

        sources = [
            {"name": "wikipedia_ar", "weight": 0.6},
            {"name": "wikipedia_fa", "weight": 0.15},
            {"name": "wikipedia_ur", "weight": 0.1},
            {"name": "wikisource_ar", "weight": 0.15},
        ]
        self.picker = TextPicker(sources, "test", CORPORA_DIR)
        self.names = [src["name"] for src in sources if (CORPORA_DIR / src["name"]).exists()]
        self.rng = rng
        self.source = None  # one corpus for the whole page (set by new_page)

    def new_page(self, one_language: bool) -> str:
        """Start a page. With `one_language`, all its text comes from one corpus, as
        on a real screen (v1 mixed corpora word by word, which put Arabic, Persian
        and Urdu words side by side and made look-alike letters like ي/ی unguessable).
        Returns the corpus name (or "mixed")."""
        if not one_language:
            self.source = None
            return "mixed"
        i = self.picker.files.index(self.picker.pick_source(self.rng))
        self.source = self.picker.files[i]
        return self.names[i] if i < len(self.names) else str(i)

    def words(self, lo: int, hi: int) -> str:
        n = self.rng.randint(lo, hi)
        out: list[str] = []
        while len(out) < n:
            src = self.source or self.picker.pick_source(self.rng)
            para = src.random_line(self.rng)
            para = charset.normalize(charset.strip_diacritics(para))
            out += [w for w in para.split() if not charset.unsupported_chars(w)]
        return " ".join(out[:n])


def _page_html(
    kind: str, t: Texts, rng: random.Random, font: str, size: int, theme, w: int, h: int
) -> str:
    fg, muted, bg, panel, accent = theme
    p = lambda lo, hi, cls="": f'<p class="{cls}">{_words(t.words(lo, hi))}</p>'  # noqa: E731
    head = (
        f'<header><b class="logo">{_words(t.words(1, 2))}</b><nav>'
        + "".join(f"<a>{_words(t.words(1, 2))}</a>" for _ in range(rng.randint(3, 5)))
        + "</nav></header>"
    )
    if kind == "article":
        body = head + f"<main><h1>{_words(t.words(4, 9))}</h1>" + p(6, 12, "muted")
        body += "".join(p(30, 80) for _ in range(5)) + "</main>"
    elif kind == "sidebar":
        side = "".join(f'<div class="item">{_words(t.words(3, 9))}</div>' for _ in range(8))
        body = head + f'<div class="row"><main><h1>{_words(t.words(4, 9))}</h1>'
        body += "".join(p(30, 70) for _ in range(4)) + f"</main><aside>{side}</aside></div>"
    elif kind == "search":
        body = head + f'<main><div class="box">{_words(t.words(2, 5))}</div>'
        for _ in range(7):
            body += f'<div class="result"><a class="title">{_words(t.words(4, 10))}</a>'
            body += p(1, 3, "muted") + p(15, 35) + "</div>"
        body += "</main>"
    elif kind == "chat":
        body = f'<header><b class="logo">{_words(t.words(2, 3))}</b></header><main class="chat">'
        for _ in range(10):
            who = rng.choice(["me", "them"])
            body += f'<div class="bubble {who}">{_words(t.words(3, 30))}</div>'
        body += "</main>"
    elif kind == "settings":
        body = head + f"<main><h1>{_words(t.words(1, 3))}</h1>"
        for _ in range(9):
            body += f'<div class="setting"><div>{_words(t.words(2, 5))}</div>'
            body += f'<div class="muted small">{_words(t.words(4, 10))}</div></div>'
        body += "</main>"
    else:  # cards
        body = head + f"<main><h1>{_words(t.words(3, 7))}</h1>" + p(10, 25)
        body += '<div class="cards">'
        for _ in range(rng.randint(2, 3)):
            body += f'<div class="card"><h2>{_words(t.words(2, 5))}</h2>{p(12, 30)}</div>'
        body += "</div>" + p(20, 40) + "</main>"

    css = f"""
    html, body {{ margin: 0; background: {bg}; color: {fg}; }}
    body {{ font-family: '{font}'; font-size: {size}px; line-height: 1.6; }}
    /* A fixed-size frame at the top-left corner: the page lays out the same whatever
       the browser window size is (Edge's two capture modes use different ones). */
    #page {{ position: absolute; left: 0; top: 0; width: {w}px; height: {h}px;
             overflow: hidden; background: {bg}; }}
    header {{ display: flex; gap: 24px; align-items: center; padding: 10px 24px;
              background: {panel}; border-bottom: 1px solid {muted}55; }}
    header nav {{ display: flex; gap: 18px; }} header a {{ color: {accent}; }}
    .logo {{ font-size: 1.3em; }}
    main {{ padding: 12px 24px; max-width: 980px; }}
    h1 {{ font-size: 1.7em; margin: 10px 0; }} h2 {{ font-size: 1.2em; margin: 6px 0; }}
    p {{ margin: 0 0 12px 0; }} .muted {{ color: {muted}; }} .small {{ font-size: 0.85em; }}
    .row {{ display: flex; gap: 40px; }} .row main {{ flex: 3; }}
    aside {{ flex: 1; padding: 16px; background: {panel}; }} .item {{ margin-bottom: 14px; }}
    .box {{ display: inline-block; padding: 6px 16px; border: 1px solid {muted};
            border-radius: 20px; margin-bottom: 16px; }}
    .result {{ margin-bottom: 18px; }} .title {{ color: {accent}; font-size: 1.15em; }}
    .chat {{ display: flex; flex-direction: column; gap: 10px; }}
    .bubble {{ max-width: 60%; padding: 8px 14px; border-radius: 14px; }}
    .me {{ align-self: flex-start; background: {accent}; color: #fff; }}
    .them {{ align-self: flex-end; background: {panel}; }}
    .setting {{ padding: 10px 0; border-bottom: 1px solid {muted}44; }}
    .cards {{ display: flex; gap: 36px; margin: 16px 0; }}
    .card {{ flex: 1; padding: 12px 16px; background: {panel}; border-radius: 8px; }}
    """
    return (
        f'<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">'
        f'<style>{css}</style></head><body><div id="page">{body}</div>'
        f"{LABEL_SCRIPT}</body></html>"
    )


#: Fonts of modern digital documents (Word, PDFs, the web) for `digital` pages.
DOC_FONTS = [
    "Times New Roman", "Arial", "Tahoma", "Segoe UI", "Traditional Arabic",
    "Simplified Arabic", "Sakkal Majalla", "Calibri",
]  # fmt: skip


def build_digital(lang: str, n_pages: int, seed: int) -> None:
    """Modern digital document pages (Persian or Urdu): unseen test-split text laid
    out as a one-page A4 document (title + paragraphs) in a Windows document font,
    printed to PDF by Edge, then rendered at 300 DPI - like the born-digital law
    PDFs that make up most of the Arabic printed set. Labels are exact (we wrote the
    text); pages are scored on their whole text."""
    import pypdfium2 as pdfium

    from training.corpora import CORPORA_DIR
    from training.make_dataset import TextPicker

    if not EDGE.exists():
        sys.exit(f"Microsoft Edge not found at {EDGE} (this set is built on Windows).")
    picker = TextPicker([{"name": f"wikipedia_{lang}", "weight": 1.0}], "test", CORPORA_DIR)
    rng = random.Random(seed)
    out = REAL_DIR / f"pages_digital_{lang}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    def words(lo: int, hi: int) -> str:
        n, got = rng.randint(lo, hi), []
        while len(got) < n:
            para = charset.normalize(charset.strip_diacritics(picker.files[0].random_line(rng)))
            got += [w for w in para.split() if not charset.unsupported_chars(w)]
        return " ".join(got[:n])

    with tempfile.TemporaryDirectory() as tmp:
        made = 0
        while made < n_pages:
            font, size = rng.choice(DOC_FONTS), rng.choice([11, 12, 13, 14])
            title = words(3, 8)
            paras = [words(30, 70) for _ in range(rng.randint(3, 5))]
            align = rng.choice(["justify", "right"])
            body = f"<h1>{html.escape(title)}</h1>" + "".join(
                f"<p>{html.escape(p)}</p>" for p in paras
            )
            css = (
                "@page { size: A4; margin: 2cm; } "
                f"body {{ font-family: '{font}'; font-size: {size}pt; "
                f"line-height: {rng.uniform(1.3, 1.8):.2f}; text-align: {align}; }} "
                "h1 { font-size: 1.5em; text-align: center; } p { margin: 0 0 0.8em 0; }"
            )
            page = Path(tmp) / f"d{made}.html"
            page.write_text(
                f'<!doctype html><html lang="{lang}" dir="rtl"><head><meta charset="utf-8">'
                f"<style>{css}</style></head><body>{body}</body></html>",
                encoding="utf-8",
            )
            pdf_path = Path(tmp) / f"d{made}.pdf"
            _edge(["--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", page.as_uri()])
            if not pdf_path.exists():
                continue
            pdf = pdfium.PdfDocument(str(pdf_path))
            if len(pdf) != 1:  # text ran onto a second page: try another page
                pdf.close()
                continue
            image = pdf[0].render(scale=300 / 72, grayscale=True).to_numpy().copy()
            pdf.close()
            key = f"d{lang}{made:03d}"
            cv2.imwrite(str(out / "images" / f"{key}.png"), image)
            label = {
                "kind": "printed",
                "style": "naskh",
                "modern": True,
                "language": lang,
                "source": f"digital document ({font} {size}pt)",
                "font": font,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "text": "\n".join([title, *paras]),
                "lines": None,
            }
            (out / "labels" / f"{key}.json").write_text(
                json.dumps(label, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            made += 1
            print(f"  [{made}/{n_pages}] {key} {font} {size}pt", flush=True)
    print(f"Done -> {out}")


def _edge(args: list[str]) -> str:
    cmd = [str(EDGE), "--headless=new", "--disable-gpu", "--hide-scrollbars", *args]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=120).stdout


def _ink_box(gray: np.ndarray, box: list[float]) -> list[float] | None:
    """Shrink a browser box (device pixels) to the ink inside it."""
    x0, y0, x1, y1 = (round(v) for v in box)
    crop = gray[max(0, y0) : y1, max(0, x0) : x1].astype(np.int16)
    if crop.size == 0:
        return None
    edge = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    ink = np.abs(crop - int(np.median(edge))) > 40
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    if len(rows) == 0:
        return None
    return [x0 + cols[0], y0 + rows[0], x0 + cols[-1] + 1, y0 + rows[-1] + 1]


def build_screens(
    n_pages: int, seed: int, name: str, one_language: bool, fonts: list[str] = FONTS
) -> None:
    if not EDGE.exists():
        sys.exit(f"Microsoft Edge not found at {EDGE} (this set is built on Windows).")
    out = REAL_DIR / name
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    texts = Texts(rng)
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n_pages):
            kind = KINDS[i % len(KINDS)]
            font, size = rng.choice(fonts), rng.choice([13, 14, 15, 16, 18, 20])
            theme = rng.choice(THEMES)
            w, h, dpr = rng.choice(VIEWPORTS)
            key = f"s{i:03d}"
            language = texts.new_page(one_language)
            page = Path(tmp) / f"{key}.html"
            page.write_text(_page_html(kind, texts, rng, font, size, theme, w, h), encoding="utf-8")
            png = out / "images" / f"{key}.png"
            scale = f"--force-device-scale-factor={dpr}"
            # Screenshot mode: the window is exactly the frame. Dump mode keeps some
            # window chrome, so give it room; the frame lays out identically.
            _edge([f"--window-size={w},{h}", scale, f"--screenshot={png}", page.as_uri()])
            dom = _edge([f"--window-size={w + 300},{h + 300}", scale, "--dump-dom", page.as_uri()])
            m = re.search(r'<pre id="labels"[^>]*>(.*?)</pre>', dom, re.S)
            gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            if m is None or gray is None:
                print(f"  {key}: capture failed, skipped")
                continue
            info = json.loads(html.unescape(m.group(1)))
            lines = []
            for ln in info["lines"]:
                box = _ink_box(gray, [v * dpr for v in ln["box"]])
                if box is None:
                    continue
                x0, y0, x1, y1 = (float(v) for v in box)
                lines.append(
                    {
                        "quad": [[x1, y0], [x0, y0], [x0, y1], [x1, y1]],
                        "text": charset.normalize(ln["text"]),
                        "block": 0,
                        "order": len(lines),
                    }
                )
            label = {
                "kind": "screen",
                "layout": kind,
                "language": language,
                "font": font,
                "font_px": size,
                "viewport": [w, h, dpr],
                "width": int(gray.shape[1]),
                "height": int(gray.shape[0]),
                "text": "\n".join(ln["text"] for ln in lines),
                "lines": lines,
            }
            (out / "labels" / f"{key}.json").write_text(
                json.dumps(label, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            print(f"  [{i + 1}/{n_pages}] {key} {kind} {font} {size}px {w}x{h}@{dpr}", flush=True)
    print(f"Done -> {out}")


def relabel_printed(site: Site = SITES["ar"]) -> None:
    """Fetch the wikitext of every page already in the printed set again and redo
    its text with the current conversion rules (same pages, same scans)."""
    out = REAL_DIR / site.folder / "labels"
    for path in sorted(out.glob("*.json")):
        label = json.loads(path.read_text(encoding="utf-8"))
        text = wikitext_to_text(raw_wikitext(label["source"], site))
        if text is None:
            print(f"  {path.stem}: no longer passes the markup rules; left unchanged")
            continue
        if text != label["text"]:
            print(f"  {path.stem}: {len(label['text'])} -> {len(text)} characters")
            label["text"] = text
            path.write_text(json.dumps(label, ensure_ascii=False, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the real page test sets.")
    sub = parser.add_subparsers(dest="set", required=True)
    pr = sub.add_parser("printed", help="scanned book pages from Wikisource")
    pr.add_argument("--site", choices=sorted(SITES), default="ar", help="Wikisource language")
    pr.add_argument("-n", type=int, default=60)
    pr.add_argument("--per-book", type=int, default=5)
    pr.add_argument("--seed", type=int, default=11)
    dg = sub.add_parser("digital", help="modern digital document pages (Persian/Urdu)")
    dg.add_argument("--lang", choices=["fa", "ur", "ar"], required=True)
    dg.add_argument("-n", type=int, default=40)
    dg.add_argument("--seed", type=int, default=41)
    sc = sub.add_parser("screens", help="browser screenshots of web-like pages")
    sc.add_argument("-n", type=int, default=50)
    sc.add_argument("--seed", type=int, default=12)
    sc.add_argument("--name", default="pages_screens_v2", help="output folder in data/real")
    sc.add_argument("--fonts", choices=sorted(FONT_SETS), default="all")
    sc.add_argument(
        "--mixed", action="store_true", help="mix corpora word by word (how v1 was made)"
    )
    rl = sub.add_parser("relabel", help="re-make a printed set's text with the current rules")
    rl.add_argument("--site", choices=sorted(SITES), default="ar")
    args = parser.parse_args(argv)
    if args.set == "printed":
        build_printed(args.n, args.per_book, args.seed, SITES[args.site])
    elif args.set == "relabel":
        relabel_printed(SITES[args.site])
    elif args.set == "digital":
        build_digital(args.lang, args.n, args.seed)
    else:
        build_screens(args.n, args.seed, args.name, not args.mixed, FONT_SETS[args.fonts])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
