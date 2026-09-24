"""Embedding client for OpenAI-compatible endpoints.

Retrieval-tuned embedding models encode a question and a document differently, so the endpoint
takes an `input_type` telling it which side of the pair it is embedding. Getting this wrong is
silent: vectors are still returned, they just retrieve noticeably worse.
"""

from collections.abc import Iterable, Sequence

from app.upstream import UpstreamClient, UpstreamError

EmbeddingError = UpstreamError


def _batched(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class EmbeddingClient(UpstreamClient):
    def __init__(self, api_base: str, api_key: str, model: str, *, batch_size: int = 32, **kwargs):
        super().__init__(api_base, api_key, **kwargs)
        self._model = model
        self._batch_size = batch_size

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents for indexing."""
        vectors: list[list[float]] = []
        for batch in _batched(texts, self._batch_size):
            vectors.extend(self._embed(list(batch), input_type="passage"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a question for searching."""
        return self._embed([text], input_type="query")[0]

    def _embed(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        body = self.post(
            "/embeddings",
            {
                "input": texts,
                "model": self._model,
                "input_type": input_type,
                "encoding_format": "float",
                "truncate": "END",
            },
        )
        return [item["embedding"] for item in sorted(body["data"], key=lambda d: d["index"])]
