"""Per-key request rate limiting with a Redis token bucket (atomic Lua script)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from redis.asyncio import Redis

# KEYS[1] = bucket; ARGV = capacity, refill_per_s, now_ms, cost
# Returns {allowed (0/1), remaining tokens (string), retry_after_ms}
TOKEN_BUCKET_LUA = """
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local data = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil or ts == nil then
  tokens = capacity
  ts = now
end
local elapsed = math.max(0, now - ts) / 1000.0
tokens = math.min(capacity, tokens + elapsed * rate)
local allowed = 0
local retry_ms = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry_ms = math.ceil((cost - tokens) / rate * 1000)
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('PEXPIRE', KEYS[1], math.ceil(capacity / rate * 1000) + 1000)
return {allowed, tostring(tokens), retry_ms}
"""


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after_s: float

    def headers(self) -> dict[str, str]:
        h = {
            "x-ratelimit-limit-requests": str(self.limit),
            "x-ratelimit-remaining-requests": str(self.remaining),
        }
        if not self.allowed:
            h["retry-after"] = str(max(1, round(self.retry_after_s + 0.499)))
        return h


class RateLimiter:
    """Bucket capacity = the per-minute limit (allows a full minute's burst), refilled
    continuously at ``rpm / 60`` tokens per second."""

    def __init__(self, redis: Redis, clock: Callable[[], float] = time.time) -> None:
        self.redis = redis
        self.clock = clock
        self._script = redis.register_script(TOKEN_BUCKET_LUA)

    async def hit(self, bucket: str, rpm: int) -> RateDecision:
        now_ms = int(self.clock() * 1000)
        allowed, remaining, retry_ms = await self._script(
            keys=[f"ratelimit:{bucket}"], args=[rpm, rpm / 60.0, now_ms, 1]
        )
        return RateDecision(
            allowed=bool(int(allowed)),
            limit=rpm,
            remaining=int(float(remaining)),
            retry_after_s=int(retry_ms) / 1000.0,
        )
