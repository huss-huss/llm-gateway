"""Per-key rate limiting with Redis token buckets (one atomic Lua script).

Two buckets per key, both refilled continuously with capacity = the per-minute limit:

- requests per minute (RPM): every request takes one token;
- tokens per minute (TPM): a request *reserves* an estimate (prompt + completion allowance)
  before the provider is called, then *settles* against actual usage afterwards (refunding or
  charging the difference). Settling may push the bucket below zero; later requests then wait
  until it refills, so under-estimates are paid back rather than forgiven.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from redis.asyncio import Redis

from llm_gateway.providers.base import estimate_tokens
from llm_gateway.schemas import ChatCompletionRequest

MESSAGE_OVERHEAD_TOKENS = 4

# KEYS[1] = bucket; ARGV = capacity, refill_per_s, now_ms, cost[, force]
# force=1 applies ``cost`` unconditionally (it may be negative, a refund, or leave the bucket
# in debt); otherwise ``cost`` is taken only if the bucket holds at least that many tokens.
# Returns {allowed (0/1), remaining tokens (string), retry_after_ms}
TOKEN_BUCKET_LUA = """
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local force = tonumber(ARGV[5] or '0')
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
if force == 1 then
  tokens = math.min(capacity, tokens - cost)
  allowed = 1
elseif tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry_ms = math.ceil((cost - tokens) / rate * 1000)
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
-- Expire once the bucket would be full again (a missing bucket reads as full).
redis.call('PEXPIRE', KEYS[1], math.ceil((capacity - math.min(tokens, 0)) / rate * 1000) + 1000)
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
    continuously at ``limit / 60`` tokens per second."""

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

    # ------------------------------------------------------------ tokens per minute

    async def _tokens(self, bucket: str, tpm: int, cost: int, force: bool) -> TokenDecision:
        now_ms = int(self.clock() * 1000)
        allowed, remaining, retry_ms = await self._script(
            keys=[f"ratelimit:tpm:{bucket}"],
            args=[tpm, tpm / 60.0, now_ms, cost, 1 if force else 0],
        )
        return TokenDecision(
            allowed=bool(int(allowed)),
            limit=tpm,
            remaining=float(remaining),
            retry_after_s=int(retry_ms) / 1000.0,
        )

    async def reserve_tokens(self, bucket: str, tpm: int, tokens: int) -> TokenDecision:
        """Atomically take ``tokens`` from the TPM bucket, or reject without taking any."""
        return await self._tokens(bucket, tpm, tokens, force=False)

    async def settle_tokens(self, bucket: str, tpm: int, delta: int) -> TokenDecision:
        """Apply ``actual - reserved``: a negative delta refunds, a positive one charges (and
        may leave the bucket in debt). Never rejects."""
        return await self._tokens(bucket, tpm, delta, force=True)


def format_reset(seconds: float) -> str:
    """Durations in OpenAI's ``x-ratelimit-reset-*`` style: ``"0s"``, ``"250ms"``, ``"6m0s"``."""
    if seconds <= 0:
        return "0s"
    if seconds < 1:
        return f"{max(1, math.ceil(seconds * 1000))}ms"
    whole = math.ceil(seconds)
    minutes, secs = divmod(whole, 60)
    return f"{minutes}m{secs}s" if minutes else f"{secs}s"


@dataclass(frozen=True)
class TokenDecision:
    allowed: bool
    limit: int
    remaining: float  # may be negative after a settle that charged more than was reserved
    retry_after_s: float

    @property
    def reset_s(self) -> float:
        """Seconds until the bucket is full again."""
        return max(0.0, (self.limit - self.remaining) / (self.limit / 60.0))

    def headers(self) -> dict[str, str]:
        h = {
            "x-ratelimit-limit-tokens": str(self.limit),
            "x-ratelimit-remaining-tokens": str(max(0, int(self.remaining))),
            "x-ratelimit-reset-tokens": format_reset(self.reset_s),
        }
        if not self.allowed:
            h["retry-after"] = str(max(1, math.ceil(self.retry_after_s)))
        return h


def estimate_prompt_tokens(req: ChatCompletionRequest) -> int:
    """Rough prompt size (~4 characters per token, as in ``estimate_tokens``), before routing.

    Counts message text, tool-call arguments, and tool definitions, plus a small per-message
    overhead for role/formatting tokens. Images and other non-text parts are not counted.
    """
    total = 0
    for m in req.messages:
        total += MESSAGE_OVERHEAD_TOKENS + estimate_tokens(m.text())
        for tc in m.tool_calls or []:
            total += estimate_tokens(tc.function.name + tc.function.arguments)
    if req.tools:
        total += estimate_tokens(json.dumps(req.tools, separators=(",", ":")))
    return total


@dataclass
class TokenReservation:
    """Tokens held against a key's TPM bucket for one in-flight request."""

    limiter: RateLimiter
    bucket: str
    limit: int
    reserved: int
    prompt_estimate: int
    decision: TokenDecision
    settled: bool = False

    async def settle(self, actual_tokens: int) -> TokenDecision:
        """Replace the reservation with ``actual_tokens``. Idempotent: only the first call
        touches Redis."""
        if not self.settled:
            self.settled = True
            self.decision = await self.limiter.settle_tokens(
                self.bucket, self.limit, actual_tokens - self.reserved
            )
        return self.decision
