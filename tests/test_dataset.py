"""Tests for the dataset format, split assignment, and the generator."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from arabic_ocr import charset
from training import corpora, make_dataset
from training.dataset import ShardWriter, read_shard, read_split, shard_paths, split_of

ROOT = Path(__file__).resolve().parent.parent
HAVE_FONTS = (ROOT / "data" / "fonts" / "notonaskharabic").exists()

# --- splits -----------------------------------------------------------------


def test_split_is_stable_and_ignores_diacritics():
    a = split_of("بِسْمِ اللَّهِ", 0.1, 0.1)
    assert a == split_of("بِسْمِ اللَّهِ", 0.1, 0.1)
    assert a == split_of("بسم الله", 0.1, 0.1)


def test_split_fractions_are_roughly_right():
    counts = Counter(split_of(f"جملة رقم {i}", 0.05, 0.1) for i in range(20000))
    assert counts["test"] / 20000 == pytest.approx(0.10, abs=0.01)
    assert counts["val"] / 20000 == pytest.approx(0.05, abs=0.01)


# --- shard format -----------------------------------------------------------


def test_shard_round_trip(tmp_path):
    path = tmp_path / "shard-00000.tar"
    img = np.random.default_rng(0).integers(0, 256, (64, 200), dtype=np.uint8)
    with ShardWriter(path) as w:
        w.write("k-0", img, "بِسْمِ ٢٠٢٦", {"font": "x.ttf"})
        w.write("k-1", img[:, :50], "سلام", {"font": "y.ttf"})
    samples = list(read_shard(path))
    assert [s.key for s in samples] == ["k-0", "k-1"]
    assert np.array_equal(samples[0].image, img)  # PNG is lossless
    assert samples[0].label == "بِسْمِ ٢٠٢٦"
    assert samples[1].meta == {"font": "y.ttf"}
    assert not path.with_suffix(".tar.tmp").exists()


def test_writer_rejects_bad_labels(tmp_path):
    img = np.zeros((10, 10), np.uint8)
    with pytest.raises(ValueError), ShardWriter(tmp_path / "a.tar") as w:
        w.write("k", img, "hello", {})  # Latin: not in the charset
    with pytest.raises(ValueError), ShardWriter(tmp_path / "b.tar") as w:
        w.write("k", img, "ﻻ", {})  # presentation form: not normalized


def test_failed_shard_leaves_no_finished_file(tmp_path):
    path = tmp_path / "shard-00000.tar"
    with pytest.raises(RuntimeError), ShardWriter(path):
        raise RuntimeError("crash")
    assert not path.exists() and not path.with_suffix(".tar.tmp").exists()


# --- generator --------------------------------------------------------------


def write_config(tmp_path, **changes) -> Path:
    cfg = yaml.safe_load((ROOT / "training/configs/data_sample.yaml").read_text(encoding="utf-8"))
    cfg.update(
        name="t",
        samples={"train": 30, "val": 10, "test": 10},
        shard_size=20,
        sampler={"number_prob": 0.0},  # lines are then exact substrings of their sentence
    )
    cfg.update(changes)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path


def prepare_sample_corpus(tmp_path) -> Path:
    corpora_dir = tmp_path / "corpora"
    cfg = corpora.load_config()
    sample = next(s for s in cfg["sources"] if s["name"] == "sample")
    corpora.prepare(sample, cfg["split"], corpora_dir)
    return corpora_dir


def run(cfg_path, tmp_path, *extra):
    return make_dataset.main(
        [
            str(cfg_path),
            "--workers",
            "1",
            "--out-dir",
            str(tmp_path / "datasets"),
            "--corpora-dir",
            str(tmp_path / "corpora"),
            *extra,
        ]
    )


def test_config_typos_are_rejected(tmp_path):
    with pytest.raises(TypeError):
        make_dataset.load_config(write_config(tmp_path, augment={"blur_probb": 0.5}))
    with pytest.raises(ValueError):
        make_dataset.load_config(write_config(tmp_path, extra_key=1))


@pytest.mark.skipif(not HAVE_FONTS, reason="needs data/fonts (see training/download_fonts.py)")
def test_generate_read_resume(tmp_path):
    cfg_path = write_config(tmp_path)
    corpora_dir = prepare_sample_corpus(tmp_path)
    assert run(cfg_path, tmp_path) == 0
    ds = tmp_path / "datasets" / "t"

    manifest = json.loads((ds / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["num_classes"] == charset.NUM_CLASSES

    # Right counts, shard sizes 20 + 10 for train.
    assert [len(list(read_shard(p))) for p in shard_paths(ds, "train")] == [20, 10]
    samples = {s: list(read_split(ds, s)) for s in ("train", "val", "test")}
    assert {s: len(v) for s, v in samples.items()} == {"train": 30, "val": 10, "test": 10}

    for s in samples["train"]:
        assert s.image.shape[0] == 64 and s.image.dtype == np.uint8
        assert charset.decode(charset.encode(s.label)) == s.label

    # No leaks: every word of every line comes from its own split's paragraphs. (A
    # line may join two paragraphs, so it isn't always a substring of one.)
    for split, items in samples.items():
        text = corpora.split_file("sample", split, corpora_dir).read_text(encoding="utf-8")
        vocab = set(charset.strip_diacritics(text).split())
        for s in items:
            words = charset.strip_diacritics(s.label).split()
            # the last word may be cut at max_len, so check whole words only
            assert all(w in vocab for w in words[:-1]), s.label

    # Resume: delete one shard; re-running re-makes it byte-for-byte identical.
    first = shard_paths(ds, "train")[0]
    original = first.read_bytes()
    first.unlink()
    assert run(cfg_path, tmp_path) == 0
    assert first.read_bytes() == original

    # Changed settings are refused without --overwrite.
    changed = write_config(tmp_path, seed=99)
    assert run(changed, tmp_path) == 1
    assert run(changed, tmp_path, "--overwrite") == 0
    assert first.read_bytes() != original
