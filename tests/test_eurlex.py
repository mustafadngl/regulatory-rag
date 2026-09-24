from pathlib import Path

import pytest

from app.corpus.chunking import chunk_articles
from app.corpus.eurlex import parse_xhtml
from app.corpus.models import Article

FIXTURE = Path(__file__).parent / "fixtures" / "sample_oj.xhtml"


@pytest.fixture(scope="module")
def articles() -> list[Article]:
    return parse_xhtml(FIXTURE.read_text(encoding="utf-8"))


def test_extracts_the_enacting_articles(articles: list[Article]) -> None:
    assert [a.number for a in articles] == ["1", "6"]


def test_strips_non_breaking_spaces_from_headings(articles: list[Article]) -> None:
    assert all("\u00a0" not in a.number for a in articles)


def test_excludes_recitals(articles: list[Article]) -> None:
    assert all("must not be indexed" not in a.body for a in articles)


def test_excludes_annexes(articles: list[Article]) -> None:
    assert "annex content" not in articles[-1].body.lower()


def test_assigns_chapter_and_title(articles: list[Article]) -> None:
    first = articles[0]

    assert first.chapter == "I"
    assert first.chapter_title == "GENERAL PROVISIONS"


def test_distinguishes_sections_from_chapters(articles: list[Article]) -> None:
    """Both levels share a CSS class, so only the heading text separates them."""
    article_six = articles[1]

    assert article_six.chapter == "III"
    assert article_six.chapter_title == "HIGH-RISK AI SYSTEMS"
    assert article_six.section == "1"
    assert article_six.section_title == "Classification of AI systems as high-risk"


def test_a_section_does_not_overwrite_its_chapter(articles: list[Article]) -> None:
    assert articles[1].chapter != "1"


def test_lettered_points_stay_with_their_paragraph(articles: list[Article]) -> None:
    body = articles[1].body

    assert "(a) the AI system is intended" in body
    assert "(b) the product is required" in body


def test_points_do_not_start_a_new_paragraph(articles: list[Article]) -> None:
    chunks = chunk_articles(articles, source="sample", max_chars=2000)
    article_six = [c for c in chunks if c.article == "6"]

    assert all(c.paragraph in {"1", "2", "1-2"} for c in article_six)


def test_chunks_carry_chapter_and_section(articles: list[Article]) -> None:
    chunks = chunk_articles(articles, source="sample", max_chars=400)
    article_six = [c for c in chunks if c.article == "6"]

    assert all(c.chapter == "III" and c.section == "1" for c in article_six)


def test_chunks_respect_the_budget_after_leftovers_are_merged(articles: list[Article]) -> None:
    """A short trailing paragraph must not be merged into a chunk that is already full."""
    for chunk in chunk_articles(articles, source="sample", max_chars=300):
        assert chunk.char_count <= 300
