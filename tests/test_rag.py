import pytest

from app.corpus.models import Chunk
from app.corpus.store import Retrieved
from app.generation import Completion
from app.rag import UNANSWERABLE, RagService, build_context


def hit(article: str, paragraph: str | None = None, body: str = "some text") -> Retrieved:
    return Retrieved(
        chunk=Chunk(
            chunk_id=f"REG:art{article}:0",
            text=f"header\n\n{body}",
            body=body,
            source="REG",
            chapter="III",
            article=article,
            article_title="Some title",
            paragraph=paragraph,
        ),
        score=0.75,
    )


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]


class FakeStore:
    def __init__(self, hits: list[Retrieved]) -> None:
        self._hits = hits
        self.calls: list[tuple[int, str | None]] = []

    def search(self, vector, *, limit=5, source=None) -> list[Retrieved]:
        self.calls.append((limit, source))
        return self._hits[:limit]


class FakeChat:
    model = "fake-model"

    def __init__(self, content: str) -> None:
        self._content = content
        self.messages: list[dict[str, str]] | None = None
        self.max_tokens: int | None = None
        self.call_count = 0

    def complete(self, messages, *, max_tokens=800, temperature=0.0) -> Completion:
        self.call_count += 1
        self.messages = messages
        self.max_tokens = max_tokens
        return Completion(
            content=self._content,
            reasoning="internal working that must never be returned",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        )


def build(hits: list[Retrieved], content: str, **kwargs) -> tuple[RagService, FakeChat, FakeStore]:
    store, chat = FakeStore(hits), FakeChat(content)
    return RagService(store, FakeEmbedder(), chat, **kwargs), chat, store


def test_context_labels_each_extract_with_its_citation() -> None:
    context = build_context([hit("6", "2"), hit("99")])

    assert "[1] Article 6(2) — Some title" in context
    assert "[2] Article 99 — Some title" in context


def test_a_grounded_answer_carries_its_citations() -> None:
    service, _, _ = build([hit("6", "2")], "High-risk systems are defined (Article 6(2)).")

    result = service.answer("what is high risk?")

    assert result.grounded is True
    assert [c.citation for c in result.citations] == ["Article 6(2)"]


def test_reasoning_is_never_returned_to_the_caller() -> None:
    service, _, _ = build([hit("6")], "The answer.")

    result = service.answer("q")

    assert "internal working" not in result.answer


def test_a_refusal_is_reported_as_ungrounded_without_citations() -> None:
    service, _, _ = build([hit("6")], UNANSWERABLE)

    result = service.answer("what is the capital of France?")

    assert result.grounded is False
    assert result.citations == []


def test_no_retrieval_skips_the_model_entirely() -> None:
    service, chat, _ = build([], "unused")

    result = service.answer("q")

    assert result.answer == UNANSWERABLE
    assert chat.call_count == 0


def test_typography_in_the_answer_is_normalised() -> None:
    service, _, _ = build(
        [hit("99", "4")], "Fines apply under Article\u202f99(4) for high\u2011risk."
    )

    result = service.answer("penalties?")

    assert result.answer.isascii()
    assert "Article 99(4)" in result.answer


def test_the_system_prefix_is_prepended_when_configured() -> None:
    service, chat, _ = build([hit("6")], "answer", system_prefix="detailed thinking off")

    service.answer("q")

    assert chat.messages[0]["content"].startswith("detailed thinking off")


def test_no_prefix_is_added_by_default() -> None:
    service, chat, _ = build([hit("6")], "answer")

    service.answer("q")

    assert chat.messages[0]["content"].startswith("You answer questions")


def test_the_answer_token_budget_is_applied() -> None:
    service, chat, _ = build([hit("6")], "answer", max_answer_tokens=123)

    service.answer("q")

    assert chat.max_tokens == 123


def test_top_k_can_be_overridden_per_request() -> None:
    service, _, store = build([hit("1"), hit("2"), hit("3")], "answer", top_k=5)

    service.answer("q", top_k=2)

    assert store.calls == [(2, None)]


def test_retrieval_can_be_scoped_to_a_source() -> None:
    service, _, store = build([hit("1")], "answer", source="32024R1689")

    service.answer("q")

    assert store.calls == [(5, "32024R1689")]


def test_timings_are_recorded() -> None:
    service, _, _ = build([hit("6")], "answer")

    result = service.answer("q")

    assert result.retrieval_ms >= 0
    assert result.generation_ms >= 0
    assert result.prompt_tokens == 100


@pytest.mark.parametrize("content", [UNANSWERABLE, UNANSWERABLE.upper()])
def test_refusal_detection_ignores_case(content: str) -> None:
    service, _, _ = build([hit("6")], content)

    assert service.answer("q").grounded is False
