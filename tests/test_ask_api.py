import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_rag_service
from app.main import create_app
from app.rag import UNANSWERABLE, Answer, Citation
from app.upstream import UpstreamError


def make_answer(**overrides) -> Answer:
    defaults = dict(
        question="what is high risk?",
        answer="High-risk systems are defined (Article 6(2)).",
        citations=[
            Citation(
                citation="Article 6(2)",
                chunk_id="REG:art6:0",
                article="6",
                article_title="Classification rules",
                chapter="III",
                score=0.81,
            )
        ],
        model="fake-model",
        grounded=True,
        prompt_tokens=100,
        completion_tokens=20,
        retrieval_ms=12.345,
        generation_ms=678.91,
    )
    return Answer(**{**defaults, **overrides})


class StubService:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error

    def answer(self, question: str, top_k: int | None = None) -> Answer:
        if self._error:
            raise self._error
        return self._result


def client_with(service) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_rag_service] = lambda: service
    return TestClient(app)


def test_a_question_returns_an_answer_with_citations() -> None:
    response = client_with(StubService(make_answer())).post("/ask", json={"question": "what?"})

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["citations"][0]["citation"] == "Article 6(2)"
    assert body["model"] == "fake-model"


def test_timings_are_rounded_in_the_response() -> None:
    response = client_with(StubService(make_answer())).post("/ask", json={"question": "what?"})

    usage = response.json()["usage"]
    assert usage["retrieval_ms"] == 12.3
    assert usage["generation_ms"] == 678.9


def test_an_ungrounded_answer_is_reported_as_such() -> None:
    answer = make_answer(answer=UNANSWERABLE, grounded=False, citations=[])

    body = client_with(StubService(answer)).post("/ask", json={"question": "capital?"}).json()

    assert body["grounded"] is False
    assert body["citations"] == []


def test_a_provider_outage_surfaces_as_bad_gateway() -> None:
    service = StubService(error=UpstreamError("HTTP 503: overloaded"))

    response = client_with(service).post("/ask", json={"question": "what?"})

    assert response.status_code == 502
    assert "unavailable" in response.json()["detail"].lower()


@pytest.mark.parametrize("payload", [{"question": "ab"}, {}, {"question": "ok?", "top_k": 0}])
def test_invalid_requests_are_rejected(payload: dict) -> None:
    response = client_with(StubService(make_answer())).post("/ask", json=payload)

    assert response.status_code == 422


def test_ask_is_unavailable_when_the_service_is_not_configured() -> None:
    """Without credentials the app still starts, but answering is switched off."""
    response = TestClient(create_app()).post("/ask", json={"question": "what?"})

    assert response.status_code == 503


def test_the_endpoint_is_documented() -> None:
    schema = TestClient(create_app()).get("/openapi.json").json()

    assert "/ask" in schema["paths"]
