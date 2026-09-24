"""Structure-aware chunking for EU regulatory texts.

Regulations are already organised into chapters, articles and numbered paragraphs.
Splitting on that structure instead of on a fixed character window keeps each chunk
semantically self-contained and makes precise citation possible.
"""

import re
from dataclasses import dataclass, field

from app.corpus.models import Article, Chunk

CHAPTER_RE = re.compile(r"^CHAPTER\s+([IVXLCDM]+)\b(.*)$", re.MULTILINE)
ARTICLE_RE = re.compile(r"^Article\s+(\d+)\s*$", re.MULTILINE)
PARAGRAPH_RE = re.compile(r"^(\d+)\.\s+", re.MULTILINE)
SENTENCE_RE = re.compile(r"(?<=[.;:])\s+")

DEFAULT_MAX_CHARS = 1200
DEFAULT_MIN_CHARS = 300


@dataclass
class _Segment:
    paragraph: str | None
    text: str


@dataclass
class _Buffer:
    paragraphs: list[str | None] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return sum(len(t) for t in self.texts) + 2 * max(len(self.texts) - 1, 0)

    def add(self, segment: _Segment) -> None:
        self.paragraphs.append(segment.paragraph)
        self.texts.append(segment.text)

    def label(self) -> str | None:
        numbered = [p for p in self.paragraphs if p is not None]
        if not numbered:
            return None
        if len(numbered) == 1:
            return numbered[0]
        return f"{numbered[0]}-{numbered[-1]}"


def parse_articles(text: str) -> list[Article]:
    """Split a regulation into articles, tracking which chapter each one falls under."""
    chapters = [(m.start(), m.group(1)) for m in CHAPTER_RE.finditer(text)]
    matches = list(ARTICLE_RE.finditer(text))
    articles: list[Article] = []

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        remainder = text[match.end() : end].strip("\n")
        lines = [line for line in remainder.split("\n")]

        title = ""
        body_start = 0
        for position, line in enumerate(lines):
            if line.strip():
                title = line.strip()
                body_start = position + 1
                break

        chapter = None
        for offset, numeral in chapters:
            if offset < match.start():
                chapter = numeral
            else:
                break

        articles.append(
            Article(
                number=match.group(1),
                title=title,
                body="\n".join(lines[body_start:]).strip(),
                chapter=chapter,
            )
        )

    return articles


def _split_into_segments(body: str) -> list[_Segment]:
    """Break an article body into its numbered paragraphs, keeping any unnumbered lead-in."""
    markers = list(PARAGRAPH_RE.finditer(body))
    if not markers:
        return [_Segment(paragraph=None, text=body.strip())] if body.strip() else []

    segments: list[_Segment] = []
    lead = body[: markers[0].start()].strip()
    if lead:
        segments.append(_Segment(paragraph=None, text=lead))

    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(body)
        text = body[marker.end() : end].strip()
        if text:
            segments.append(_Segment(paragraph=marker.group(1), text=text))

    return segments


def _split_oversized(segment: _Segment, max_chars: int) -> list[_Segment]:
    """Break a paragraph that exceeds the budget on sentence boundaries."""
    if len(segment.text) <= max_chars:
        return [segment]

    pieces: list[str] = []
    current = ""
    for sentence in SENTENCE_RE.split(segment.text):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)

    return [_Segment(paragraph=segment.paragraph, text=piece) for piece in pieces]


def _breadcrumb(article: Article, paragraph: str | None) -> str:
    parts = [f"Article {article.number}"]
    if article.title:
        parts.append(article.title)
    if paragraph:
        parts.append(f"paragraph {paragraph}")
    return " > ".join(parts)


def chunk_article(
    article: Article,
    source: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """Chunk a single article, merging short paragraphs and splitting long ones.

    Chunks never span article boundaries, so a retrieved chunk always maps to one citation.
    """
    segments: list[_Segment] = []
    for segment in _split_into_segments(article.body):
        segments.extend(_split_oversized(segment, max_chars))

    buffers: list[_Buffer] = []
    current = _Buffer()

    for segment in segments:
        if current.texts and current.length + len(segment.text) > max_chars:
            buffers.append(current)
            current = _Buffer()
        current.add(segment)
        if current.length >= min_chars:
            buffers.append(current)
            current = _Buffer()

    if current.texts:
        fits = buffers and buffers[-1].length + current.length <= max_chars
        if fits and current.length < min_chars:
            for paragraph, text in zip(current.paragraphs, current.texts, strict=True):
                buffers[-1].add(_Segment(paragraph=paragraph, text=text))
        else:
            buffers.append(current)

    chunks: list[Chunk] = []
    for index, buffer in enumerate(buffers):
        paragraph = buffer.label()
        body = "\n\n".join(buffer.texts)
        header = _breadcrumb(article, paragraph)
        chunks.append(
            Chunk(
                chunk_id=f"{source}:art{article.number}:{index}",
                text=f"{header}\n\n{body}",
                body=body,
                source=source,
                chapter=article.chapter,
                section=article.section,
                article=article.number,
                article_title=article.title or None,
                paragraph=paragraph,
            )
        )

    return chunks


def chunk_articles(
    articles: list[Article],
    source: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in articles:
        chunks.extend(chunk_article(article, source, max_chars=max_chars, min_chars=min_chars))
    return chunks


def chunk_document(
    text: str,
    source: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    return chunk_articles(parse_articles(text), source, max_chars=max_chars, min_chars=min_chars)
