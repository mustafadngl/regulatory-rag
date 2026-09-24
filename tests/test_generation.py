import json

import httpx

from app.generation import ChatClient


def build_client(handler) -> ChatClient:
    return ChatClient(
        "https://example.invalid/v1",
        "test-key",
        "test-model",
        requests_per_minute=0,
        transport=httpx.MockTransport(handler),
    )


def respond(**message) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", **message}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        },
    )


def test_the_answer_is_taken_from_content() -> None:
    client = build_client(lambda _: respond(content="Article 6(2) applies."))

    assert client.complete([]).content == "Article 6(2) applies."


def test_reasoning_is_kept_separate_from_the_answer() -> None:
    """Chain of thought must never leak into what the caller shows a user."""
    handler = lambda _: respond(content="The answer.", reasoning_content="Let me think...")  # noqa: E731

    completion = build_client(handler).complete([])

    assert completion.content == "The answer."
    assert completion.reasoning == "Let me think..."


def test_surrounding_whitespace_is_trimmed() -> None:
    client = build_client(lambda _: respond(content="  spaced out  \n"))

    assert client.complete([]).content == "spaced out"


def test_a_null_content_becomes_an_empty_string() -> None:
    client = build_client(lambda _: respond(content=None))

    assert build_client(lambda _: respond(content=None)).complete([]).content == ""
    assert client.complete([]).reasoning is None


def test_token_usage_is_reported() -> None:
    completion = build_client(lambda _: respond(content="hi")).complete([])

    assert (completion.prompt_tokens, completion.completion_tokens) == (11, 7)
    assert completion.total_tokens == 18


def test_missing_usage_defaults_to_zero() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    completion = build_client(handler).complete([])

    assert completion.total_tokens == 0


def test_generation_parameters_are_sent() -> None:
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return respond(content="hi")

    build_client(handler).complete([{"role": "user", "content": "q"}], max_tokens=42)

    assert sent[0]["model"] == "test-model"
    assert sent[0]["max_tokens"] == 42
    assert sent[0]["temperature"] == 0.0


def test_the_model_name_is_exposed() -> None:
    assert build_client(lambda _: respond(content="hi")).model == "test-model"
