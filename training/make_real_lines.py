"""Real training lines from Wikisource scans (M4: close the gap to real print).

    uv run python -m training.make_real_lines --site ar --pages 1200
    uv run python -m training.make_real_lines --site fa --pages 800
    uv run python -m training.make_real_lines --site ur --pages 400
    uv run python -m training.make_real_lines --finish      # write the manifest

Synthetic lines can't teach everything real scans contain: old typefaces, worn
letterpress, uneven ink, paper texture. Wikisource has thousands of *proofread*
scanned pages whose text may be reused (scans public domain, text CC BY-SA; see
data/LICENSES.md). This script turns them into labelled line images:

1. pick proofread pages from books that are NOT in any real test set (and at most
   PER_BOOK pages per book, so no single book dominates);
2. run our OCR on the scan to find and read every line;
3. match each found line to the transcription line it reads most like
   (character error rate); keep a match only if it is clearly the same line
   (CER <= MAX_CER) - then the *transcription* becomes the label;
4. write the line crops (64 px high, like the synthetic data) as shards.

Lines go to data/datasets/real_v1/<site>/{train,val}/ (val: 1 in 20 books).
The labels are human-checked text; the model only chose *which* lines to keep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from arabic_ocr import charset
from arabic_ocr.detect.boxes import quad_to_rect
from arabic_ocr.pipeline import OCR, crop_line
from arabic_ocr.recognize.recognizer import to_gray
from training import make_real_pages as W
from training.dataset import ShardWriter, charset_fingerprint
from training.make_dataset import resize_to_height
from training.metrics import fast_edit_distance

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "datasets" / "real_v1"
PER_BOOK = 15
MAX_CER = 0.30
MIN_CHARS = 6
SHARD = 2000


def test_books() -> set[str]:
    """Books used by any real test set: never train on them."""
    books = set()
    for folder in (ROOT / "data" / "real").glob("pages_printed*"):
        for lab in (folder / "labels").glob("*.json"):
            books.add(json.loads(lab.read_text(encoding="utf-8"))["source"].rpartition("/")[0])
    return books


def _norm(text: str) -> str:
    return " ".join(charset.strip_tatweel(charset.strip_diacritics(text)).split())


def match_lines(found: list[str], reference: list[str]) -> list[tuple[int, int, float]]:
    """(found index, reference index, CER) for confident one-to-one matches."""
    pairs = []
    refs = [_norm(r) for r in reference]
    for i, f in enumerate(found):
        f = _norm(f)
        if len(f) < MIN_CHARS:
            continue
        for j, r in enumerate(refs):
            if len(r) < MIN_CHARS or abs(len(r) - len(f)) > 0.5 * len(r):
                continue
            cer = fast_edit_distance(r, f) / len(r)
            if cer <= MAX_CER:
                pairs.append((cer, i, j))
    used_f, used_r, out = set(), set(), []
    for cer, i, j in sorted(pairs):
        if i not in used_f and j not in used_r:
            used_f.add(i)
            used_r.add(j)
            out.append((i, j, cer))
    return out


def page_lines(ocr: OCR, image: np.ndarray, reference: str):
    """Yield (line image 64 px high, label) for the confident matches on one page."""
    gray = to_gray(image)
    # Find and read the lines exactly as the pipeline does (straighten, rules, detect).
    from arabic_ocr.preprocess import preprocess

    prep = preprocess(gray)
    rects = [quad_to_rect(d.quad) for d in ocr.detector.detect(prep.image)]
    crops = [crop_line(prep.image, r)[0] for r in rects]
    found = [d.text for d in ocr.recognizer.recognize_batch(crops)]
    ref_lines = [ln for ln in reference.splitlines() if ln.strip()]
    for i, j, _cer in match_lines(found, ref_lines):
        label = charset.normalize(ref_lines[j])
        if charset.unsupported_chars(label):
            continue
        yield resize_to_height(crops[i], 64), label


def build(site_name: str, n_pages: int, seed: int) -> None:
    site = W.SITES[site_name]
    titles = W.proofread_titles(ROOT / "data" / "real" / site.folder / "candidates.json", site)
    excluded = test_books()
    books: dict[str, list[str]] = {}
    for t in titles:
        book = t.rpartition("/")[0] or t
        if book not in excluded:
            books.setdefault(book, []).append(t)
    rng = random.Random(seed)
    names = sorted(books)
    rng.shuffle(names)
    plan = []
    for b in names:
        pages = books[b][:]
        rng.shuffle(pages)
        plan += [(b, t) for t in pages[:PER_BOOK]]
    plan = plan[: n_pages * 2]  # some pages are skipped (markup, no scan)
    print(f"{site_name}: {len(books)} books outside the test sets; trying up to {len(plan)} pages")

    ocr = OCR()
    out_dir = OUT / site_name
    done_path = out_dir / "done.json"
    done = set(json.loads(done_path.read_text(encoding="utf-8"))) if done_path.exists() else set()
    lines = {"train": [], "val": []}
    pages_ok = 0
    for book, title in plan:
        if pages_ok >= n_pages:
            break
        if title in done:
            continue
        done.add(title)
        try:
            text = W.wikitext_to_text(W.raw_wikitext(title, site))
            if not text or len(text) < 200:
                continue
            got = W.download_scan(title, site)
            if got is None:
                continue
            image = cv2.imdecode(np.frombuffer(got[1], np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            split = "val" if int(hashlib.sha1(book.encode()).hexdigest(), 16) % 20 == 0 else "train"
            new = list(page_lines(ocr, image, text))
        except Exception as e:  # network hiccups, odd files: skip the page
            print(f"  skipped {title[:60]}: {e}", flush=True)
            continue
        lines[split] += [(img, label, title) for img, label in new]
        pages_ok += 1
        if pages_ok % 25 == 0:
            n = len(lines["train"]) + len(lines["val"])
            print(f"  [{pages_ok}/{n_pages}] pages, {n} lines kept", flush=True)
            _flush(out_dir, site_name, lines, done, done_path)
    _flush(out_dir, site_name, lines, done, done_path)
    print(f"{site_name}: done, {pages_ok} pages")


def _flush(out_dir: Path, site_name: str, lines: dict, done: set, done_path: Path) -> None:
    """Write the collected lines as new shards (so an interrupted run keeps them)."""
    for split, items in lines.items():
        if not items:
            continue
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        index = len(list((out_dir / split).glob("shard-*.tar")))
        with ShardWriter(out_dir / split / f"shard-{index:05d}.tar") as w:
            for k, (img, label, title) in enumerate(items):
                key = f"{site_name}-{split}-{index:05d}-{k:05d}"
                w.write(key, img, label, {"source": title, "site": site_name})
        items.clear()
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path.write_text(json.dumps(sorted(done), ensure_ascii=False), encoding="utf-8")


def finish() -> None:
    """Count the lines per site and split, and write the dataset manifest."""
    from training.dataset import read_shard

    counts = {}
    for site_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        for split in ("train", "val"):
            n = sum(1 for s in (site_dir / split).glob("shard-*.tar") for _ in read_shard(s))
            counts[f"{site_dir.name}/{split}"] = n
    print(json.dumps(counts, indent=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real training lines from Wikisource.")
    parser.add_argument("--site", choices=sorted(W.SITES))
    parser.add_argument("--pages", type=int, default=500)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--finish", action="store_true", help="count lines per site/split")
    args = parser.parse_args(argv)
    if args.finish:
        finish()
        return 0
    if not args.site:
        parser.error("--site is required")
    print(f"charset {charset_fingerprint()}")
    build(args.site, args.pages, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
