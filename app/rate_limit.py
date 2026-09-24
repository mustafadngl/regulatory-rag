"""Client-side request pacing.

Respecting a published quota locally is cheaper than discovering it through 429 responses:
a rejected request still costs a round trip, and the retry that follows waits far longer than
the delay that would have avoided it.
"""

import time
from collections.abc import Callable


class RateLimiter:
    """Spaces requests evenly so that no more than `max_per_minute` are issued."""

    def __init__(
        self,
        max_per_minute: int,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = 60.0 / max_per_minute if max_per_minute > 0 else 0.0
        self._monotonic = monotonic
        self._sleep = sleep
        self._next_allowed = 0.0

    def acquire(self) -> None:
        if self._interval <= 0:
            return

        now = self._monotonic()
        wait = self._next_allowed - now
        if wait > 0:
            self._sleep(wait)
            now = self._next_allowed

        self._next_allowed = now + self._interval
