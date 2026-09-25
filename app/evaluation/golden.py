from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GoldenQuestion:
    id: str
    question: str
    expected_articles: list[str] = field(default_factory=list)
    answerable: bool = True

    def __post_init__(self) -> None:
        if self.answerable and not self.expected_articles:
            raise ValueError(f"{self.id}: an answerable question needs expected articles")
        if not self.answerable and self.expected_articles:
            raise ValueError(f"{self.id}: an unanswerable question cannot expect articles")


def load_golden_set(path: Path) -> list[GoldenQuestion]:
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    questions = [GoldenQuestion(**entry) for entry in entries]

    identifiers = [q.id for q in questions]
    duplicates = {i for i in identifiers if identifiers.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate question ids: {sorted(duplicates)}")

    return questions
