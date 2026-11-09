"""Download and prepare text corpora (see training/configs/corpora.yaml).

Usage:
    uv run python -m training.corpora download [NAME ...] [--accept-tanzil-terms]
    uv run python -m training.corpora prepare  [NAME ...]
    uv run python -m training.corpora stats

`download` saves raw files under data/corpora/raw/<name>/ (skipping files already there).

`prepare` turns each source into clean, split text:

    data/corpora/<name>/train.txt   one cleaned paragraph per line
    data/corpora/<name>/val.txt
    data/corpora/<name>/test.txt
    data/corpora/<name>/stats.json  counts, and how much of the text is diacritized

Each paragraph is cleaned with `training.text.clean_text`, exact duplicates are
dropped, and the paragraph's split is decided by `training.dataset.split_of`, so a
paragraph (and every line later cut from it) belongs to exactly one split.

`LineFile` reads random lines from those files without loading them into memory,
which keeps dataset generation light even with hundreds of MB of text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import yaml

from arabic_ocr import charset
from training.dataset import SPLITS, split_of
from training.text import clean_text

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "training" / "configs" / "corpora.yaml"
CORPORA_DIR = ROOT / "data" / "corpora"
RAW_DIR = CORPORA_DIR / "raw"
USER_AGENT = "arabic-ocr-corpus-downloader"

# Quranic annotation marks (small high letters, pause marks, end-of-ayah ornaments).
# They are reading aids outside our charset; removing them leaves the words intact.
_QURAN_MARKS = re.compile("[ۖ-ۭ࣓-ࣿ]")


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def select(cfg: dict, names: list[str]) -> list[dict]:
    sources = cfg["sources"]
    if not names:
        return sources
    known = {s["name"] for s in sources}
    if unknown := set(names) - known:
        raise SystemExit(f"unknown source(s): {', '.join(sorted(unknown))}")
    return [s for s in sources if s["name"] in names]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _fetch_to(url: str, dest: Path) -> None:
    """Download `url` to `dest`, via a temporary file so a broken download never
    looks finished."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    # A dropped connection can end the stream early *without* an error. Never accept a
    # file shorter than the size the server announced.
    if total and done != total:
        tmp.unlink(missing_ok=True)
        raise OSError(f"incomplete download of {url}: got {done} of {total} bytes; run again")
    tmp.replace(dest)


def download(source: dict, accept_tanzil_terms: bool) -> None:
    out = RAW_DIR / source["name"]
    kind = source["kind"]
    if kind == "hf_parquet":
        for f in source["files"]:
            dest = out / Path(f).name
            if dest.exists():
                print(f"  {source['name']}: {dest.name} already downloaded")
                continue
            print(f"  {source['name']}: downloading {f}")
            _fetch_to(f"https://huggingface.co/datasets/{source['repo']}/resolve/main/{f}", dest)
    elif kind == "tanzil":
        dest = out / "quran-simple.txt"
        if dest.exists():
            print(f"  {source['name']}: already downloaded")
            return
        if not accept_tanzil_terms:
            print(
                f"  {source['name']}: skipped. The Tanzil text is CC BY 3.0 with extra terms\n"
                "    (verbatim copies only; credit Tanzil Project and link to tanzil.net;\n"
                "    see https://tanzil.net/docs/text_license). Re-run with\n"
                "    --accept-tanzil-terms if you accept them."
            )
            return
        print(f"  {source['name']}: downloading")
        _fetch_to(source["url"], dest)
    elif kind == "local":
        print(f"  {source['name']}: local file, nothing to download")
    elif kind == "filter":
        print(f"  {source['name']}: made from {source['from']} by `prepare`, nothing to download")
    else:
        raise ValueError(f"unknown kind: {kind}")


# ---------------------------------------------------------------------------
# Prepare
# ---------------------------------------------------------------------------


def raw_paragraphs(source: dict) -> Iterator[str]:
    """Yield raw paragraphs (one per line of text) from a downloaded source."""
    kind = source["kind"]
    raw = RAW_DIR / source["name"]
    if kind == "hf_parquet":
        import pyarrow.parquet as pq  # only needed here

        for f in source["files"]:
            path = raw / Path(f).name
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; run `download {source['name']}` first")
            for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=512):
                for article in batch.column(0).to_pylist():
                    yield from (article or "").splitlines()
    elif kind == "tanzil":
        path = raw / "quran-simple.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run `download {source['name']}` first")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            text = line.rsplit("|", 1)[-1]  # format: sura|ayah|text
            yield _QURAN_MARKS.sub("", text)
    elif kind == "local":
        yield from (ROOT / source["path"]).read_text(encoding="utf-8").splitlines()
    else:
        raise ValueError(f"unknown kind: {kind}")


def diacritics_per_letter(text: str) -> float:
    """~0 for plain text, ~0.5-0.9 for fully diacritized text."""
    letters = sum(c.isalpha() for c in text)
    return sum(c in charset.DIACRITIC_SET for c in text) / letters if letters else 0.0


def split_paragraphs(source: dict, split_cfg: dict, corpora_dir: Path) -> Iterator[tuple[str, str]]:
    """Yield (split, clean paragraph) pairs for a source."""
    if source["kind"] == "filter":
        # A subset of another prepared source: keep each paragraph in its original
        # split, so nothing can leak between splits.
        for split in SPLITS:
            path = split_file(source["from"], split, corpora_dir)
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; run `prepare {source['from']}` first")
            with open(path, encoding="utf-8") as f:
                for line in f:
                    para = line.rstrip("\r\n")
                    if diacritics_per_letter(para) >= source["min_diacritics_per_letter"]:
                        yield split, para
        return

    seen: set[bytes] = set()  # 8-byte hashes, so millions of paragraphs fit in memory
    for raw in raw_paragraphs(source):
        for para in clean_text(raw):
            if len(para) < 2:
                continue
            h = hashlib.blake2b(para.encode("utf-8"), digest_size=8).digest()
            if h in seen:
                continue
            seen.add(h)
            yield split_of(para, split_cfg["val_frac"], split_cfg["test_frac"]), para


def prepare(source: dict, default_split: dict, corpora_dir: Path = CORPORA_DIR) -> dict:
    split_cfg = {**default_split, **source.get("split", {})}
    out_dir = corpora_dir / source["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = {s: out_dir / f"{s}.txt.tmp" for s in SPLITS}
    files = {s: open(tmp[s], "w", encoding="utf-8", newline="\n") for s in SPLITS}

    stats = {s: {"paragraphs": 0, "chars": 0} for s in SPLITS}
    marks = letters = 0
    try:
        for split, para in split_paragraphs(source, split_cfg, corpora_dir):
            files[split].write(para + "\n")
            stats[split]["paragraphs"] += 1
            stats[split]["chars"] += len(para)
            letters += sum(c.isalpha() for c in para)
            marks += sum(c in charset.DIACRITIC_SET for c in para)
    finally:
        for f in files.values():
            f.close()
    for s in SPLITS:
        tmp[s].replace(out_dir / f"{s}.txt")

    result = {
        "name": source["name"],
        "license": source["license"],
        "language": source["language"],
        "split_fractions": split_cfg,
        "splits": stats,
        # marks per letter: ~0 for plain text, ~0.5-0.9 for fully diacritized text
        "diacritics_per_letter": round(marks / letters, 3) if letters else 0.0,
    }
    (out_dir / "stats.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


# ---------------------------------------------------------------------------
# Random access for dataset generation
# ---------------------------------------------------------------------------


class LineFile:
    """Random lines from a big text file, without loading it into memory.

    On open it records where every line starts (8 bytes per line); picking a line
    is then one seek + one read.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        offsets = [0]
        with open(path, "rb") as f:
            for line in f:
                offsets.append(offsets[-1] + len(line))
        self.offsets = np.array(offsets[:-1], dtype=np.int64)
        self._f = open(path, "rb")  # kept open for fast random reads

    def __len__(self) -> int:
        return len(self.offsets)

    def line(self, i: int) -> str:
        self._f.seek(int(self.offsets[i]))
        # Strip "\r" too: a file edited on Windows may end lines with "\r\n".
        return self._f.readline().decode("utf-8").rstrip("\r\n")

    def random_line(self, rng: random.Random) -> str:
        return self.line(rng.randrange(len(self.offsets)))


def split_file(name: str, split: str, corpora_dir: Path = CORPORA_DIR) -> Path:
    return corpora_dir / name / f"{split}.txt"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and prepare text corpora.")
    parser.add_argument("command", choices=["download", "prepare", "stats"])
    parser.add_argument("names", nargs="*", help="source names (default: all)")
    parser.add_argument("--accept-tanzil-terms", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config()
    sources = select(cfg, args.names)

    if args.command == "download":
        for s in sources:
            download(s, args.accept_tanzil_terms)
    elif args.command == "prepare":
        for s in sources:
            if s["kind"] not in ("local", "filter") and not (RAW_DIR / s["name"]).exists():
                print(f"  {s['name']}: not downloaded, skipped")
                continue
            print(f"  {s['name']}: preparing...", flush=True)
            r = prepare(s, cfg["split"])
            counts = {k: v["paragraphs"] for k, v in r["splits"].items()}
            mb = sum(v["chars"] for v in r["splits"].values()) / 1e6
            print(f"    {counts}, {mb:.1f}M chars, diacritics/letter {r['diacritics_per_letter']}")
    else:
        for s in sources:
            p = CORPORA_DIR / s["name"] / "stats.json"
            if p.exists():
                r = json.loads(p.read_text(encoding="utf-8"))
                counts = {k: v["paragraphs"] for k, v in r["splits"].items()}
                print(f"{s['name']:15} {counts} diacritics/letter {r['diacritics_per_letter']}")
            else:
                print(f"{s['name']:15} not prepared")
    return 0


if __name__ == "__main__":
    sys.exit(main())
