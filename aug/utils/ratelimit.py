"""In-process token-bucket rate limiting.

AUG runs as a single process, so a shared counter needs no coordination and no new
dependency.  If that ever changes this becomes a Redis or Postgres counter.
"""

import time
from collections.abc import Callable


class RateLimiter:
    """Allow ``limit`` requests per ``per_seconds`` per caller, refilling continuously.

    The bucket store is deliberately bounded.  An unbounded dict keyed by client IP
    turns the rate limiter into the memory-exhaustion vector it exists to prevent:
    every spoofed source address would allocate a permanent entry.
    """

    def __init__(
        self,
        limit: int,
        per_seconds: float,
        now: Callable[[], float] = time.monotonic,
        max_tracked: int = 10_000,
    ) -> None:
        self._limit = float(limit)
        self._rate = limit / per_seconds
        self._now = now
        self._max_tracked = max_tracked
        self._buckets: dict[str, tuple[float, float]] = {}  # key → (tokens, updated_at)

    def allow(self, key: str) -> bool:
        """Consume one token for ``key``.  False when the caller is over its allowance."""
        now = self._now()
        known = key in self._buckets
        tokens = self._tokens(key, now)

        if tokens < 1.0:
            # Keep tracking a refused caller — forgetting it would forgive the flood.
            self._buckets[key] = (tokens, now)
            return False

        if not known and len(self._buckets) >= self._max_tracked:
            self._evict(now)
        self._buckets[key] = (tokens - 1.0, now)
        return True

    def __len__(self) -> int:
        return len(self._buckets)

    def _tokens(self, key: str, now: float) -> float:
        """The caller's allowance right now, after continuous refill."""
        tokens, updated = self._buckets.get(key, (self._limit, now))
        return min(self._limit, tokens + (now - updated) * self._rate)

    def _evict(self, now: float) -> None:
        """Make room, dropping the callers that lose the least by being forgotten.

        A fully refilled bucket is indistinguishable from an untracked caller, so it
        goes first.  Never evict by age: the caller being actively limited is often
        the oldest entry, and dropping it would reset the attacker's allowance.
        """
        for key in [k for k in self._buckets if self._tokens(k, now) >= self._limit]:
            del self._buckets[key]

        while len(self._buckets) >= self._max_tracked:
            del self._buckets[max(self._buckets, key=lambda k: self._tokens(k, now))]
