"""Embedding client for OpenAI-compatible endpoints.

Retrieval-tuned embedding models encode a question and a document differently, so the endpoint
takes an `input_type` telling it which side of the pair it is embedding. Getting this wrong is
silent: vectors are still returned, they just retrieve noticeably worse.
"""

import logging
import time
from collections.abc import Iterable, Sequence

import httpx

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class EmbeddingError(RuntimeError):
    pass


def _batched(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class EmbeddingClient:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        *,
        batch_size: int = 32,
        timeout: float = 120.0,
        ca_bundle: str | None = None,
        max_attempts: int = 4,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._model = model
        self._batch_size = batch_size
        self._max_attempts = max_attempts
        self._client = httpx.Client(
            base_url=api_base.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            verify=ca_bundle or True,
            transport=transport,
        )

    def __enter__(self) -> "EmbeddingClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents for indexing."""
        vectors: list[list[float]] = []
        for batch in _batched(texts, self._batch_size):
            vectors.extend(self._request(list(batch), input_type="passage"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a question for searching."""
        return self._request([text], input_type="query")[0]

    def _request(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        payload = {
            "input": texts,
            "model": self._model,
            "input_type": input_type,
            "encoding_format": "float",
            "truncate": "END",
        }

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.post("/embeddings", json=payload)
            except httpx.TransportError as exc:
                if attempt == self._max_attempts:
                    raise EmbeddingError(f"Transport failure after {attempt} attempts") from exc
                self._backoff(attempt, str(exc))
                continue

            if response.status_code in RETRYABLE_STATUS and attempt < self._max_attempts:
                self._backoff(attempt, f"HTTP {response.status_code}")
                continue

            if response.status_code != 200:
                raise EmbeddingError(f"HTTP {response.status_code}: {response.text[:300]}")

            data = response.json()["data"]
            return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]

        raise EmbeddingError(f"Giving up after {self._max_attempts} attempts")

    @staticmethod
    def _backoff(attempt: int, reason: str) -> None:
        delay = 2.0**attempt
        logger.warning("Embedding request failed (%s); retrying in %.0fs", reason, delay)
        time.sleep(delay)
