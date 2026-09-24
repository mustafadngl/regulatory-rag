from pathlib import Path

import pytest

from app.corpus.chunking import chunk_document, parse_articles

FIXTURE = Path(__file__).parent / "fixtures" / "sample_regulation.txt"


@pytest.fixture(scope="module")
def regulation() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_parses_every_article(regulation: str) -> None:
    articles = parse_articles(regulation)

    assert [a.number for a in articles] == ["1", "2", "6", "9"]


def test_captures_article_titles(regulation: str) -> None:
    titles = {a.number: a.title for a in parse_articles(regulation)}

    assert titles["6"] == "Classification rules for high-risk AI systems"


def test_attributes_articles_to_their_chapter(regulation: str) -> None:
    chapters = {a.number: a.chapter for a in parse_articles(regulation)}

    assert chapters["2"] == "I"
    assert chapters["6"] == "III"


def test_chunks_never_span_two_articles(regulation: str) -> None:
    for chunk in chunk_document(regulation, source="sample"):
        assert "Article" not in chunk.body.replace("paragraph", "")


def test_every_chunk_carries_a_breadcrumb_header(regulation: str) -> None:
    for chunk in chunk_document(regulation, source="sample"):
        assert chunk.text.startswith(f"Article {chunk.article}")
        assert chunk.body in chunk.text


def test_chunk_ids_are_unique(regulation: str) -> None:
    chunks = chunk_document(regulation, source="sample")

    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_respects_the_size_budget(regulation: str) -> None:
    for chunk in chunk_document(regulation, source="sample", max_chars=600):
        assert chunk.char_count <= 600


def test_long_paragraph_is_split_across_chunks(regulation: str) -> None:
    article_nine = [
        c for c in chunk_document(regulation, source="sample", max_chars=400) if c.article == "9"
    ]

    assert len(article_nine) > 1


def test_short_paragraphs_are_merged_rather_than_left_alone(regulation: str) -> None:
    article_two = [
        c for c in chunk_document(regulation, source="sample", max_chars=2000) if c.article == "2"
    ]

    assert len(article_two) == 1
    assert article_two[0].paragraph == "1-3"


def test_citation_reflects_article_and_paragraph(regulation: str) -> None:
    chunks = {c.chunk_id: c for c in chunk_document(regulation, source="sample", max_chars=500)}
    article_six = [c for c in chunks.values() if c.article == "6"]

    assert all(c.citation.startswith("Article 6(") for c in article_six)


def test_article_without_numbered_paragraphs_still_chunks(regulation: str) -> None:
    article_one = [c for c in chunk_document(regulation, source="sample") if c.article == "1"]

    assert len(article_one) == 1
    assert article_one[0].paragraph is None
    assert article_one[0].citation == "Article 1"


def test_empty_document_produces_no_chunks() -> None:
    assert chunk_document("", source="sample") == []
