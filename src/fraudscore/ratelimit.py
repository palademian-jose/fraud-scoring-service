"""Per-client token bucket."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    stamp: float


class TokenBucketLimiter:
    """`rate_per_minute` tokens refill continuously up to `burst`; each request costs one."""

    def __init__(
        self,
        rate_per_minute: int,
        burst: int,
        clock: Callable[[], float] = time.monotonic,
        max_clients: int = 10_000,
    ) -> None:
        self.rate = rate_per_minute / 60.0
        self.burst = float(burst)
        self._clock = clock
        self._max = max_clients
        self._buckets: dict[str, _Bucket] = {}

    def allow(self, key: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        now = self._clock()
        b = self._buckets.get(key)
        if b is None:
            if len(self._buckets) >= self._max:
                self._buckets.pop(next(iter(self._buckets)))
            b = self._buckets[key] = _Bucket(self.burst, now)
        b.tokens = min(self.burst, b.tokens + (now - b.stamp) * self.rate)
        b.stamp = now
        if b.tokens >= 1.0:
            b.tokens -= 1.0
            return True, 0.0
        return False, (1.0 - b.tokens) / self.rate
