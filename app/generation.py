"""Chat completion client.

Reasoning models return their working in a separate `reasoning_content` field. Only `content`
is the answer; anything that leaks the reasoning into a response is a bug, not a feature.
"""

from dataclasses import dataclass

from app.upstream import UpstreamClient


@dataclass
class Completion:
    content: str
    reasoning: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatClient(UpstreamClient):
    def __init__(self, api_base: str, api_key: str, model: str, **kwargs):
        super().__init__(api_base, api_key, **kwargs)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> Completion:
        body = self.post(
            "/chat/completions",
            {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )

        message = body["choices"][0]["message"]
        usage = body.get("usage") or {}
        return Completion(
            content=(message.get("content") or "").strip(),
            reasoning=message.get("reasoning_content"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
