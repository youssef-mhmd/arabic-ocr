"""Tests for the real page test-set builder (no network needed)."""

from training.make_real_pages import wikitext_to_text


def test_wikitext_keeps_lines_and_template_text():
    wikitext = (
        '<noinclude><pagequality level="3" user="x" />{{رص|وسط=( ٢٢٤ )}}</noinclude>'
        '<section begin="a" />{{ع4| الملك السعيد }} قال الشيخ\n'
        "{{أبيات|ليل يعارضه الزمان\\ما لي به إلا الأسى}}\n"
        "{{rule}}{{Nop}}<noinclude></noinclude>"
    )
    assert wikitext_to_text(wikitext).splitlines() == [
        "( ٢٢٤ )",
        "الملك السعيد قال الشيخ",  # bold text inside a line stays in that line
        "ليل يعارضه الزمان ما لي به إلا الأسى",
    ]


def test_pages_with_footnotes_tables_or_unknown_templates_are_skipped():
    assert wikitext_to_text("نص {{حا|حاشية}} آخر") is None
    assert wikitext_to_text("نص<ref>حاشية</ref>") is None
    assert wikitext_to_text("{|\n| خلية\n|}") is None
    assert wikitext_to_text("{{قالب_مجهول|نص}}") is None
