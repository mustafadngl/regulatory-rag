from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass
class Article:
    """One article of a regulation, before it is split into retrievable chunks."""

    number: str
    title: str
    body: str
    chapter: str | None = None
    chapter_title: str | None = None
    section: str | None = None
    section_title: str | None = None


class Chunk(BaseModel):
    """A retrievable unit of regulatory text, carrying enough structure to cite it."""

    chunk_id: str
    text: str = Field(description="Breadcrumb header followed by the source text.")
    body: str = Field(description="Source text without the breadcrumb header.")
    source: str
    chapter: str | None = None
    section: str | None = None
    article: str | None = None
    article_title: str | None = None
    paragraph: str | None = None

    @property
    def citation(self) -> str:
        if self.article is None:
            return self.source
        if self.paragraph is None:
            return f"Article {self.article}"
        return f"Article {self.article}({self.paragraph})"

    @property
    def char_count(self) -> int:
        return len(self.body)
