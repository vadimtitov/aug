"""Tests for aug/utils/ratelimit.py.

Behaviors under test:
  - a caller within its allowance is served; one over it is refused
  - the allowance refills over time
  - callers are isolated from each other
  - the bucket store is bounded, so many distinct callers cannot exhaust memory
"""

from aug.utils.ratelimit import RateLimiter


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_allows_up_to_the_limit_then_refuses():
    clock = FakeClock()
    limiter = RateLimiter(limit=3, per_seconds=60, now=clock)

    assert [limiter.allow("1.2.3.4") for _ in range(3)] == [True, True, True]
    assert limiter.allow("1.2.3.4") is False


def test_allowance_refills_over_time():
    clock = FakeClock()
    limiter = RateLimiter(limit=3, per_seconds=60, now=clock)

    for _ in range(3):
        limiter.allow("1.2.3.4")
    assert limiter.allow("1.2.3.4") is False

    clock.advance(20)  # one third of the window → one token back
    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is False


def test_callers_do_not_share_an_allowance():
    clock = FakeClock()
    limiter = RateLimiter(limit=1, per_seconds=60, now=clock)

    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is False
    assert limiter.allow("5.6.7.8") is True


def test_bucket_store_is_bounded():
    """A flood from many source addresses must not grow the limiter without bound.

    Otherwise the rate limiter is itself the memory-exhaustion vector it exists to
    prevent — every unseen IP would allocate a permanent entry.
    """
    clock = FakeClock()
    limiter = RateLimiter(limit=1, per_seconds=60, now=clock, max_tracked=100)

    for i in range(5000):
        limiter.allow(f"10.0.{i // 256}.{i % 256}")

    assert len(limiter) <= 100


def test_eviction_does_not_forgive_an_active_flooder():
    """The caller currently being limited must survive eviction of idle entries."""
    clock = FakeClock()
    limiter = RateLimiter(limit=2, per_seconds=60, now=clock, max_tracked=10)

    assert limiter.allow("attacker") is True
    assert limiter.allow("attacker") is True
    assert limiter.allow("attacker") is False

    for i in range(50):
        clock.advance(0.01)
        limiter.allow(f"other-{i}")

    assert limiter.allow("attacker") is False
