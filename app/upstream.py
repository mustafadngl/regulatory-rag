"""Shared HTTP behaviour for the model provider.

Embeddings and completions talk to the same endpoint under the same quota, so pacing,
retries and TLS configuration belong in one place rather than in each client.
"""

import logging
import time
from typing import Any

import httpx

from app.rate_limit import RateLimiter

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class UpstreamError(RuntimeError):
    pass


class UpstreamClient:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        *,
        timeout: float = 120.0,
        ca_bundle: str | None = None,
        max_attempts: int = 4,
        requests_per_minute: int = 40,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._max_attempts = max_attempts
        self._limiter = RateLimiter(requests_per_minute)
        self._client = httpx.Client(
            base_url=api_base.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            verify=ca_bundle or True,
            transport=transport,
        )

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(1, self._max_attempts + 1):
            self._limiter.acquire()

            try:
                response = self._client.post(path, json=payload)
            except httpx.TransportError as exc:
                if attempt == self._max_attempts:
                    raise UpstreamError(f"Transport failure after {attempt} attempts") from exc
                self._backoff(attempt, str(exc))
                continue

            if response.status_code in RETRYABLE_STATUS and attempt < self._max_attempts:
                self._backoff(attempt, f"HTTP {response.status_code}")
                continue

            if response.status_code != 200:
                raise UpstreamError(f"HTTP {response.status_code}: {response.text[:300]}")

            return response.json()

        raise UpstreamError(f"Giving up after {self._max_attempts} attempts")

    @staticmethod
    def _backoff(attempt: int, reason: str) -> None:
        delay = 2.0**attempt
        logger.warning("Upstream request failed (%s); retrying in %.0fs", reason, delay)
        time.sleep(delay)
