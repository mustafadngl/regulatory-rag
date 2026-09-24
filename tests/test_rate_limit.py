from app.rate_limit import RateLimiter


class FakeClock:
    """A controllable clock, so pacing can be asserted without the tests actually waiting."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def build(max_per_minute: int) -> tuple[RateLimiter, FakeClock]:
    clock = FakeClock()
    return RateLimiter(max_per_minute, monotonic=clock.monotonic, sleep=clock.sleep), clock


def test_the_first_request_is_not_delayed() -> None:
    limiter, clock = build(60)

    limiter.acquire()

    assert clock.slept == []


def test_subsequent_requests_are_spaced_by_the_quota() -> None:
    limiter, clock = build(60)

    limiter.acquire()
    limiter.acquire()
    limiter.acquire()

    assert clock.slept == [1.0, 1.0]


def test_pacing_matches_a_forty_per_minute_quota() -> None:
    limiter, clock = build(40)

    limiter.acquire()
    limiter.acquire()

    assert clock.slept == [1.5]


def test_time_already_elapsed_counts_towards_the_interval() -> None:
    limiter, clock = build(60)

    limiter.acquire()
    clock.now += 2.0
    limiter.acquire()

    assert clock.slept == []


def test_a_zero_quota_disables_pacing() -> None:
    limiter, clock = build(0)

    limiter.acquire()
    limiter.acquire()

    assert clock.slept == []
