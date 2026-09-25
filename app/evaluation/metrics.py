"""Answer-quality metrics.

Everything here is a pure function of recorded outcomes, so the metrics can be tested without
calling a model. That matters: a quality gate whose own arithmetic is untested is not a gate.
"""

import re
from dataclasses import dataclass, field

# Models abbreviate freely: "Article 14", "Art 14(1)", "Art. 14". Matching only the long form
# scored correctly-cited answers as uncited, which is how this pattern earned its first bug.
ARTICLE_REFERENCE = re.compile(r"\bArt(?:icle)?\.?\s*(\d+)\b")


def cited_articles(answer: str) -> list[str]:
    """Article numbers referenced in an answer, in order of first appearance."""
    seen: list[str] = []
    for number in ARTICLE_REFERENCE.findall(answer):
        if number not in seen:
            seen.append(number)
    return seen


def first_hit_rank(expected: list[str], retrieved: list[str]) -> int | None:
    """One-based position of the first expected article, or None if absent."""
    for position, article in enumerate(retrieved, start=1):
        if article in expected:
            return position
    return None


@dataclass
class QuestionOutcome:
    id: str
    question: str
    answerable: bool
    expected_articles: list[str]
    retrieved_articles: list[str]
    answer: str
    grounded: bool
    cited: list[str] = field(default_factory=list)
    error: str | None = None
    """Set when the question could not be evaluated, e.g. the provider was unavailable.

    An error is not a quality failure. It is excluded from every metric, because scoring an
    outage as a regression would make the gate fire for reasons no code change can fix.
    """

    def __post_init__(self) -> None:
        if not self.cited:
            self.cited = cited_articles(self.answer)

    @property
    def rank(self) -> int | None:
        return first_hit_rank(self.expected_articles, self.retrieved_articles)

    @property
    def retrieved_expected(self) -> bool:
        return self.rank is not None

    @property
    def cited_expected(self) -> bool:
        return any(article in self.expected_articles for article in self.cited)

    @property
    def passed(self) -> bool:
        if self.error is not None:
            return False
        if not self.answerable:
            return not self.grounded
        return self.grounded and self.retrieved_expected and self.cited_expected

    @property
    def failure_reason(self) -> str | None:
        if self.error is not None:
            return f"not evaluated: {self.error}"
        if self.passed:
            return None
        if not self.answerable:
            return "answered a question the corpus cannot support"
        if not self.grounded:
            return "refused an answerable question"
        if not self.retrieved_expected:
            return f"expected {self.expected_articles}, retrieved {self.retrieved_articles}"
        return f"expected a citation to {self.expected_articles}, cited {self.cited or 'nothing'}"


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


@dataclass
class Report:
    outcomes: list[QuestionOutcome]

    @property
    def errored(self) -> list[QuestionOutcome]:
        return [o for o in self.outcomes if o.error is not None]

    @property
    def evaluated(self) -> list[QuestionOutcome]:
        return [o for o in self.outcomes if o.error is None]

    @property
    def error_rate(self) -> float:
        return _ratio(len(self.errored), len(self.outcomes))

    @property
    def answerable(self) -> list[QuestionOutcome]:
        return [o for o in self.evaluated if o.answerable]

    @property
    def unanswerable(self) -> list[QuestionOutcome]:
        return [o for o in self.evaluated if not o.answerable]

    @property
    def retrieval_recall(self) -> float:
        """Share of answerable questions where an expected article was retrieved."""
        return _ratio(sum(o.retrieved_expected for o in self.answerable), len(self.answerable))

    @property
    def mean_reciprocal_rank(self) -> float:
        if not self.answerable:
            return 0.0
        total = sum(1.0 / o.rank for o in self.answerable if o.rank)
        return round(total / len(self.answerable), 4)

    @property
    def citation_accuracy(self) -> float:
        """Share of answerable questions whose answer cited an expected article."""
        return _ratio(sum(o.cited_expected for o in self.answerable), len(self.answerable))

    @property
    def refusal_accuracy(self) -> float:
        """Share of out-of-scope questions correctly refused."""
        return _ratio(sum(not o.grounded for o in self.unanswerable), len(self.unanswerable))

    @property
    def false_refusal_rate(self) -> float:
        """Share of answerable questions wrongly refused."""
        return _ratio(sum(not o.grounded for o in self.answerable), len(self.answerable))

    @property
    def failures(self) -> list[QuestionOutcome]:
        return [o for o in self.evaluated if not o.passed]

    def as_dict(self) -> dict:
        return {
            "totals": {
                "questions": len(self.outcomes),
                "evaluated": len(self.evaluated),
                "errored": len(self.errored),
                "answerable": len(self.answerable),
                "unanswerable": len(self.unanswerable),
                "passed": len(self.evaluated) - len(self.failures),
            },
            "metrics": {
                "retrieval_recall": self.retrieval_recall,
                "mean_reciprocal_rank": self.mean_reciprocal_rank,
                "citation_accuracy": self.citation_accuracy,
                "refusal_accuracy": self.refusal_accuracy,
                "false_refusal_rate": self.false_refusal_rate,
            },
            "questions": [
                {
                    "id": o.id,
                    "question": o.question,
                    "answerable": o.answerable,
                    "expected_articles": o.expected_articles,
                    "retrieved_articles": o.retrieved_articles,
                    "cited": o.cited,
                    "grounded": o.grounded,
                    "passed": o.passed,
                    "error": o.error,
                    "failure_reason": o.failure_reason,
                    "answer": o.answer,
                }
                for o in self.outcomes
            ],
        }


@dataclass
class Thresholds:
    """Set just below the measured baseline of 1.00 / 1.00 / 1.00 / 0.00.

    Generation is not deterministic, so pinning a gate to its own best observation makes it
    fire on noise. The headroom is small enough that a genuine regression still trips it.
    """

    retrieval_recall: float = 0.90
    citation_accuracy: float = 0.85
    refusal_accuracy: float = 1.00
    max_false_refusal_rate: float = 0.10

    def violations(self, report: Report) -> list[str]:
        checks = [
            ("retrieval_recall", report.retrieval_recall, self.retrieval_recall, "at least"),
            ("citation_accuracy", report.citation_accuracy, self.citation_accuracy, "at least"),
            ("refusal_accuracy", report.refusal_accuracy, self.refusal_accuracy, "at least"),
        ]
        problems = [
            f"{name} {value:.2f} is below the required {limit:.2f}"
            for name, value, limit, _ in checks
            if value < limit
        ]
        if report.false_refusal_rate > self.max_false_refusal_rate:
            problems.append(
                f"false_refusal_rate {report.false_refusal_rate:.2f} exceeds "
                f"the permitted {self.max_false_refusal_rate:.2f}"
            )
        return problems
