"""Tests for corpus preparation (no network: uses small fake raw files)."""

import io
import json
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arabic_ocr import charset
from training import corpora


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    monkeypatch.setattr(corpora, "RAW_DIR", raw)
    return raw, tmp_path / "corpora"


def all_lines(corpora_dir, name):
    return [
        line
        for split in ("train", "val", "test")
        for line in corpora.split_file(name, split, corpora_dir)
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def test_parquet_source_is_cleaned_deduplicated_and_split(dirs):
    raw, out = dirs
    articles = [
        "عنوان المقال\n"
        "هذه فقرة أولى (with Latin) عن التاريخ.\n"
        "هذه فقرة أولى (with Latin) عن التاريخ.",
        "مقال ثان\nنص عربي آخر مع أرقام ٢٠٢٦",
        "",
        None,
    ]
    (raw / "wiki").mkdir(parents=True)
    pq.write_table(pa.table({"text": articles}), raw / "wiki" / "part.parquet")
    source = {
        "name": "wiki",
        "kind": "hf_parquet",
        "files": ["x/part.parquet"],
        "license": "CC BY-SA 4.0",
        "language": "ar",
    }

    stats = corpora.prepare(source, {"val_frac": 0.0, "test_frac": 0.0}, out)

    lines = all_lines(out, "wiki")
    assert "هذه فقرة أولى (" in lines
    assert lines.count("هذه فقرة أولى (") == 1  # duplicate paragraph dropped
    assert all(charset.unsupported_chars(line) == set() for line in lines)
    assert stats["splits"]["train"]["paragraphs"] == len(lines)
    saved = json.loads((out / "wiki" / "stats.json").read_text(encoding="utf-8"))
    assert saved["license"] == "CC BY-SA 4.0"


def test_tanzil_source_strips_quran_marks_and_keeps_tashkeel(dirs):
    raw, out = dirs
    (raw / "q").mkdir(parents=True)
    (raw / "q" / "quran-simple.txt").write_text(
        "1|1|بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ\n"
        "2|2|ذَٰلِكَ الْكِتَابُ لَا رَيْبَ ۛ فِيهِ ۛ هُدًى لِلْمُتَّقِينَ\n"
        "# comment lines are ignored\n",
        encoding="utf-8",
    )
    source = {"name": "q", "kind": "tanzil", "url": "", "license": "CC BY 3.0", "language": "ar"}
    stats = corpora.prepare(source, {"val_frac": 0.0, "test_frac": 0.0}, out)
    lines = all_lines(out, "q")
    assert "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ" in lines
    assert "ذَٰلِكَ الْكِتَابُ لَا رَيْبَ فِيهِ هُدًى لِلْمُتَّقِينَ" in lines  # pause marks gone
    assert stats["diacritics_per_letter"] > 0.5


def test_no_paragraph_in_two_splits(dirs):
    _, out = dirs
    source = next(s for s in corpora.load_config()["sources"] if s["name"] == "sample")
    corpora.prepare(source, {"val_frac": 0.3, "test_frac": 0.3}, out)
    per_split = {
        s: set(corpora.split_file("sample", s, out).read_text(encoding="utf-8").splitlines())
        for s in ("train", "val", "test")
    }
    assert not (per_split["train"] & per_split["val"])
    assert not (per_split["train"] & per_split["test"])
    assert not (per_split["val"] & per_split["test"])
    assert all(per_split.values())


def test_filter_source_keeps_diacritized_paragraphs_in_their_split(dirs):
    _, out = dirs
    parent = out / "parent"
    parent.mkdir(parents=True)
    (parent / "train.txt").write_text("بِسْمِ اللَّهِ\nنص بلا تشكيل\n", encoding="utf-8")
    (parent / "val.txt").write_text("قَالَ الْمُعَلِّمُ\n", encoding="utf-8")
    (parent / "test.txt").write_text("", encoding="utf-8")
    source = {
        "name": "tash",
        "kind": "filter",
        "from": "parent",
        "min_diacritics_per_letter": 0.3,
        "license": "x",
        "language": "ar",
    }
    corpora.prepare(source, {"val_frac": 0.5, "test_frac": 0.5}, out)
    read = lambda s: corpora.split_file("tash", s, out).read_text(encoding="utf-8").splitlines()  # noqa: E731
    assert read("train") == ["بِسْمِ اللَّهِ"]
    assert read("val") == ["قَالَ الْمُعَلِّمُ"]  # split kept, not re-hashed
    assert read("test") == []


def test_line_file_random_access(tmp_path):
    path = tmp_path / "t.txt"
    lines = ["سطر أول", "سطر ثان ٢", "ثالث"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    lf = corpora.LineFile(path)
    assert len(lf) == 3
    assert [lf.line(i) for i in range(3)] == lines
    rng = random.Random(0)
    assert {lf.random_line(rng) for _ in range(50)} == set(lines)


class FakeResponse:
    """Announces `announced` bytes but only sends `data` (a dropped connection)."""

    def __init__(self, data: bytes, announced: int):
        self._data, self.headers = io.BytesIO(data), {"Content-Length": str(announced)}

    def read(self, n):
        return self._data.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_truncated_download_is_rejected(tmp_path, monkeypatch):
    # Found for real: a 172 MB file arrived as 10 MB with no error raised.
    monkeypatch.setattr(
        corpora.urllib.request, "urlopen", lambda *a, **k: FakeResponse(b"x" * 10, 100)
    )
    dest = tmp_path / "f.parquet"
    with pytest.raises(OSError, match="incomplete"):
        corpora._fetch_to("http://example.invalid/f", dest)
    assert not dest.exists() and not dest.with_name("f.parquet.part").exists()


def test_complete_download_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(
        corpora.urllib.request, "urlopen", lambda *a, **k: FakeResponse(b"x" * 100, 100)
    )
    dest = tmp_path / "f.parquet"
    corpora._fetch_to("http://example.invalid/f", dest)
    assert dest.read_bytes() == b"x" * 100


def test_tanzil_needs_explicit_acceptance(dirs, capsys):
    source = {"name": "q", "kind": "tanzil", "url": "http://invalid.example"}
    corpora.download(source, accept_tanzil_terms=False)  # must not touch the network
    assert "--accept-tanzil-terms" in capsys.readouterr().out


def test_every_source_is_in_licenses_md():
    from pathlib import Path

    licenses = (Path(corpora.ROOT) / "data" / "LICENSES.md").read_text(encoding="utf-8")
    for s in corpora.load_config()["sources"]:
        assert f"`{s['name']}`" in licenses, s["name"]
