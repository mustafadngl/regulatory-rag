"""Fetch and parse EU legislation from the Publications Office.

The public EUR-Lex web pages sit behind an AWS WAF JavaScript challenge, so they cannot be
retrieved by an HTTP client. The Publications Office Cellar service exposes the same documents
through content negotiation and is the supported machine-readable route.
"""

import warnings
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from app.corpus.models import Article

CELLAR_URL = "http://publications.europa.eu/resource/celex/{celex}"
XHTML = "application/xhtml+xml"

ARTICLE_NUMBER = "oj-ti-art"
ARTICLE_TITLE = "oj-sti-art"
BODY_TEXT = "oj-normal"
ANNEX_TITLE = "oj-doc-ti"

# The Official Journal reuses one class for chapter and section headings, and another for both
# of their titles, so the heading text is the only thing that distinguishes the two levels.
DIVISION_NUMBER = "oj-ti-section-1"
DIVISION_TITLE = "oj-ti-section-2"


def fetch_regulation(
    celex: str,
    cache_dir: Path,
    ca_bundle: str | None = None,
    timeout: float = 120.0,
) -> str:
    """Download a regulation as XHTML, caching it so repeat runs do not hit the network."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{celex}.xhtml"
    if cached.exists():
        return cached.read_text(encoding="utf-8")

    response = httpx.get(
        CELLAR_URL.format(celex=celex),
        headers={"Accept": XHTML, "Accept-Language": "eng"},
        follow_redirects=True,
        timeout=timeout,
        verify=ca_bundle or True,
    )
    response.raise_for_status()
    cached.write_text(response.text, encoding="utf-8")
    return response.text


def _normalise(text: str) -> str:
    return text.replace("\u00a0", " ").strip()


def _classes(tag: Tag) -> list[str]:
    return tag.get("class") or []


def _render(tag: Tag) -> str:
    """Render a content block, keeping lettered points such as '(a)' attached to their text.

    Points are marked up as two-cell tables, so joining the cells reproduces the reading order.
    """
    if tag.name != "table":
        return _normalise(tag.get_text(" ", strip=True))

    rows = []
    for row in tag.find_all("tr"):
        cells = [_normalise(cell.get_text(" ", strip=True)) for cell in row.find_all("td")]
        line = " ".join(cell for cell in cells if cell)
        if line:
            rows.append(line)
    return "\n".join(rows)


def _content_blocks(soup: BeautifulSoup) -> list[Tag]:
    """Top-level blocks only; paragraphs nested inside tables are rendered with their table."""
    return [
        tag
        for tag in soup.find_all(["p", "table"])
        if not (tag.name == "p" and tag.find_parent("table") is not None)
    ]


def parse_xhtml(html: str) -> list[Article]:
    """Extract articles from Cellar XHTML using the Official Journal's own semantic classes.

    Recitals precede the first article and annexes follow the last, so neither is included.
    """
    # The HTML parser is deliberate: the documents declare the XHTML namespace, and parsing them
    # as XML would force every lookup to be namespace-qualified for no practical gain.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")

    articles: list[Article] = []
    chapter: str | None = None
    chapter_title: str | None = None
    section: str | None = None
    section_title: str | None = None
    pending_division: str | None = None
    current: Article | None = None
    body: list[str] = []
    expecting_title = False

    def flush() -> None:
        if current is not None:
            current.body = "\n".join(body).strip()
            articles.append(current)

    for tag in _content_blocks(soup):
        classes = _classes(tag)
        text = _render(tag)
        if not text:
            continue

        if DIVISION_NUMBER in classes:
            if text.startswith("CHAPTER"):
                chapter = text.removeprefix("CHAPTER").strip() or None
                chapter_title = None
                section = None
                section_title = None
                pending_division = "chapter"
            elif text.startswith("SECTION"):
                section = text.removeprefix("SECTION").strip() or None
                section_title = None
                pending_division = "section"
            else:
                pending_division = None
            continue

        if DIVISION_TITLE in classes:
            if pending_division == "chapter":
                chapter_title = text
            elif pending_division == "section":
                section_title = text
            pending_division = None
            continue

        if ARTICLE_NUMBER in classes:
            flush()
            current = Article(
                number=text.removeprefix("Article").strip(),
                title="",
                body="",
                chapter=chapter,
                chapter_title=chapter_title,
                section=section,
                section_title=section_title,
            )
            body = []
            expecting_title = True
            continue

        if current is None:
            continue

        if ANNEX_TITLE in classes:
            flush()
            current = None
            body = []
            continue

        if expecting_title and ARTICLE_TITLE in classes:
            current.title = text
            expecting_title = False
            continue

        if BODY_TEXT in classes or tag.name == "table":
            expecting_title = False
            body.append(text)

    flush()
    return articles
