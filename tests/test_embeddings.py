import json

import httpx
import pytest

from app.corpus.embeddings import EmbeddingClient, EmbeddingError


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries use exponential backoff; tests should not actually wait."""
    monkeypatch.setattr("app.corpus.embeddings.time.sleep", lambda _: None)


def build_client(handler, **kwargs) -> EmbeddingClient:
    return EmbeddingClient(
        "https://example.invalid/v1",
        "test-key",
        "test-model",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def embedding_response(request: httpx.Request) -> httpx.Response:
    texts = json.loads(request.content)["input"]
    return httpx.Response(
        200,
        json={"data": [{"index": i, "embedding": [0.1, 0.2]} for i in range(len(texts))]},
    )


def test_passages_are_embedded_with_the_passage_input_type() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return embedding_response(request)

    build_client(handler).embed_passages(["a", "b"])

    assert seen[0]["input_type"] == "passage"


def test_queries_are_embedded_with_the_query_input_type() -> None:
    """Asymmetric models encode questions differently from documents."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return embedding_response(request)

    build_client(handler).embed_query("what is high risk?")

    assert seen[0]["input_type"] == "query"


def test_large_inputs_are_split_into_batches() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(len(json.loads(request.content)["input"]))
        return embedding_response(request)

    vectors = build_client(handler, batch_size=2).embed_passages(["a", "b", "c", "d", "e"])

    assert calls == [2, 2, 1]
    assert len(vectors) == 5


def test_vectors_are_returned_in_request_order() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [9.0]},
                    {"index": 0, "embedding": [1.0]},
                ]
            },
        )

    vectors = build_client(handler).embed_passages(["first", "second"])

    assert vectors == [[1.0], [9.0]]


def test_transient_failures_are_retried() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503)
        return embedding_response(request)

    vectors = build_client(handler).embed_passages(["a"])

    assert attempts["n"] == 3
    assert len(vectors) == 1


def test_retries_are_eventually_given_up_on() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(EmbeddingError):
        build_client(handler, max_attempts=2).embed_passages(["a"])


def test_client_errors_are_not_retried() -> None:
    attempts = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(401, text="unauthorised")

    with pytest.raises(EmbeddingError, match="401"):
        build_client(handler).embed_passages(["a"])

    assert attempts["n"] == 1
