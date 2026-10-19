"""Per-provider circuit breakers (closed -> open -> half-open -> closed).

A breaker opens after ``failure_threshold`` provider failures within ``window_s``, rejects calls
for ``cooldown_s``, then lets a single probe through (half-open). A successful probe closes it;
a failed one opens it again. The probe holds a lease (``probe_lease_s``, by default the
provider's timeout), so a probe whose outcome is never reported doesn't wedge the breaker.

Two backends, selected by ``routing.breaker.backend``:

- ``memory`` (default): state lives in the process. Each replica keeps its own view.
- ``redis``: state lives in one Redis hash per provider, updated by one atomic Lua script, so
  every replica sees the same state and only one replica at a time gets the half-open probe.
  If Redis errors or is slow, the replica falls back to in-process breakers (and says so in
  logs and metrics) until Redis answers again. Breaker storage never fails a request.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Iterable
from enum import StrEnum
from typing import Protocol

from redis.asyncio import Redis

from llm_gateway.config import GatewayConfig
from llm_gateway.observability import metrics

log = logging.getLogger(__name__)


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


Transition = Callable[[BreakerState, BreakerState], None]


class CircuitBreaker:
    """One in-process breaker. Not thread-safe; the gateway runs on a single event loop."""

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        *,
        window_s: float | None = None,
        probe_lease_s: float | None = None,
        on_transition: Transition | None = None,
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_s = cooldown_s
        self.window_s = window_s
        self.probe_lease_s = probe_lease_s
        self._clock = clock
        self._on_transition = on_transition
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._window_start = 0.0
        self._opened_at = 0.0
        self._probe_started: float | None = None

    @property
    def state(self) -> BreakerState:
        if self._state is BreakerState.OPEN and self._clock() - self._opened_at >= self.cooldown_s:
            self._move(BreakerState.HALF_OPEN)
            self._probe_started = None
        return self._state

    def allow(self) -> bool:
        """Whether a call may be attempted now. In half-open, exactly one probe is let through
        (another one only once the probe's lease has expired)."""
        state = self.state
        if state is BreakerState.CLOSED:
            return True
        if state is BreakerState.HALF_OPEN:
            now = self._clock()
            lease = self.probe_lease_s
            if self._probe_started is None or (
                lease is not None and now - self._probe_started >= lease
            ):
                self._probe_started = now
                return True
        return False

    def record_success(self) -> None:
        self._move(BreakerState.CLOSED)
        self._failures = 0
        self._probe_started = None

    def record_failure(self) -> None:
        state = self.state
        if state is BreakerState.HALF_OPEN:
            self._trip()
            return
        if state is BreakerState.OPEN:
            return  # a call that started before the circuit opened
        now = self._clock()
        if (
            self._failures
            and self.window_s is not None
            and now - self._window_start >= self.window_s
        ):
            self._failures = 0
        if not self._failures:
            self._window_start = now
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._trip()

    def release(self) -> None:
        """Give up the half-open probe without an outcome (e.g. the request itself was bad)."""
        self._probe_started = None

    def _trip(self) -> None:
        self._move(BreakerState.OPEN)
        self._opened_at = self._clock()
        self._failures = 0
        self._probe_started = None

    def _move(self, new: BreakerState) -> None:
        old, self._state = self._state, new
        if old is not new and self._on_transition is not None:
            self._on_transition(old, new)


def _transitioned(provider: str, backend: str, old: BreakerState, new: BreakerState) -> None:
    metrics.observe_breaker_transition(provider, old, new, backend)
    log.log(
        logging.WARNING if new is BreakerState.OPEN else logging.INFO,
        "circuit breaker %s -> %s",
        old,
        new,
        extra={
            "provider": provider,
            "from_state": str(old),
            "to_state": str(new),
            "backend": backend,
        },
    )


class Breakers(Protocol):
    """The router's view of the breakers: one per provider, addressed by provider name."""

    backend: str

    async def allow(self, provider: str) -> bool: ...
    async def record_success(self, provider: str) -> None: ...
    async def record_failure(self, provider: str) -> None: ...
    async def release(self, provider: str) -> None: ...
    async def state(self, provider: str) -> BreakerState: ...


class LocalBreakers:
    """In-process breakers (``backend: memory``)."""

    backend = "memory"

    def __init__(
        self,
        config: GatewayConfig,
        providers: Iterable[str],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        bc = config.routing.breaker
        self.breakers = {
            name: CircuitBreaker(
                bc.failure_threshold,
                bc.cooldown_s,
                clock,
                window_s=bc.window_s,
                probe_lease_s=config.probe_lease_for(name),
                on_transition=lambda old, new, name=name: _transitioned(
                    name, self.backend, old, new
                ),
            )
            for name in providers
        }

    def __getitem__(self, provider: str) -> CircuitBreaker:
        return self.breakers[provider]

    def _observe(self, provider: str) -> None:
        metrics.observe_breaker(provider, self.breakers[provider].state)

    async def allow(self, provider: str) -> bool:
        allowed = self.breakers[provider].allow()
        self._observe(provider)
        return allowed

    async def record_success(self, provider: str) -> None:
        self.breakers[provider].record_success()
        self._observe(provider)

    async def record_failure(self, provider: str) -> None:
        self.breakers[provider].record_failure()
        self._observe(provider)

    async def release(self, provider: str) -> None:
        self.breakers[provider].release()

    async def state(self, provider: str) -> BreakerState:
        self._observe(provider)
        return self.breakers[provider].state


# One hash per provider: state, failures + window_start (closed), opened_at, and the probe
# lease (probe_until, probe_owner). Times are epoch milliseconds from the caller.
# KEYS[1] = breaker hash
# ARGV = op (allow|success|failure|release|state), now_ms, failure_threshold, window_ms,
#        cooldown_ms, probe_lease_ms, state_ttl_ms, caller id
# Returns {allowed (0/1), state, from_1, to_1, from_2, to_2, ...} (the transitions made).
# TTLs: a closed hash with failures expires when its window ends; an open/half-open one
# expires state_ttl after its cooldown (refreshed on every write); a healthy provider has no
# hash at all (a missing hash reads as closed).
BREAKER_LUA = """
local key = KEYS[1]
local op = ARGV[1]
local now = tonumber(ARGV[2])
local threshold = tonumber(ARGV[3])
local window = tonumber(ARGV[4])
local cooldown = tonumber(ARGV[5])
local lease = tonumber(ARGV[6])
local ttl = tonumber(ARGV[7])
local me = ARGV[8]
local d = redis.call('HMGET', key, 'state', 'failures', 'window_start', 'opened_at',
                      'probe_until', 'probe_owner')
local state = d[1] or 'closed'
local failures = tonumber(d[2]) or 0
local window_start = tonumber(d[3]) or 0
local opened_at = tonumber(d[4]) or 0
local probe_until = tonumber(d[5]) or 0
local owner = d[6] or ''
local moves = {}
local dirty = false
local function move(to)
  if to ~= state then
    moves[#moves + 1] = state
    moves[#moves + 1] = to
    state = to
    dirty = true
  end
end
local function trip()
  move('open')
  opened_at = now
  failures = 0
  probe_until = 0
  owner = ''
  dirty = true
end
if state == 'open' and now - opened_at >= cooldown then
  move('half_open')
  probe_until = 0
  owner = ''
end
local allowed = 0
if op == 'allow' then
  if state == 'closed' then
    allowed = 1
  elseif state == 'half_open' and probe_until <= now then
    probe_until = now + lease
    owner = me
    allowed = 1
    dirty = true
  end
elseif op == 'success' then
  if state ~= 'closed' or failures > 0 then
    move('closed')
    failures = 0
    dirty = true
  end
elseif op == 'failure' then
  if state == 'half_open' then
    trip()
  elseif state == 'closed' then
    if failures == 0 or now - window_start >= window then
      failures = 0
      window_start = now
    end
    failures = failures + 1
    dirty = true
    if failures >= threshold then
      trip()
    end
  end
elseif op == 'release' then
  if state == 'half_open' and owner == me then
    probe_until = 0
    owner = ''
    dirty = true
  end
end
if dirty then
  if state == 'closed' and failures == 0 then
    redis.call('DEL', key)
  elseif state == 'closed' then
    redis.call('HSET', key, 'state', state, 'failures', failures, 'window_start', window_start)
    redis.call('PEXPIRE', key, math.max(1, window_start + window - now))
  else
    redis.call('HSET', key, 'state', state, 'failures', 0, 'opened_at', opened_at,
               'probe_until', probe_until, 'probe_owner', owner)
    redis.call('PEXPIRE', key, math.max(0, opened_at + cooldown - now) + ttl)
  end
end
local out = {allowed, state}
for i = 1, #moves do
  out[#out + 1] = moves[i]
end
return out
"""


def _ms(seconds: float) -> int:
    return max(1, round(seconds * 1000))


class RedisBreakers:
    """Breakers shared by every replica through Redis (``backend: redis``).

    Each operation is one Lua call. On any Redis error or timeout this replica switches to its
    own in-process breakers (starting closed) for ``store_retry_s``, then tries Redis again.
    """

    backend = "redis"

    def __init__(
        self,
        redis: Redis,
        config: GatewayConfig,
        providers: Iterable[str],
        *,
        clock: Callable[[], float] = time.time,
        local_clock: Callable[[], float] = time.monotonic,
        instance_id: str | None = None,
        key_prefix: str = "breaker:",
    ) -> None:
        self.redis = redis
        self.config = config
        self.providers = list(providers)
        self.instance_id = instance_id or uuid.uuid4().hex
        self.key_prefix = key_prefix
        self._clock = clock
        self._local_clock = local_clock
        self._script = redis.register_script(BREAKER_LUA)
        self.local = LocalBreakers(config, self.providers, local_clock)
        self.degraded = False
        self._retry_at = 0.0

    def key(self, provider: str) -> str:
        return f"{self.key_prefix}{provider}"

    async def _eval(self, op: str, provider: str) -> tuple[bool, BreakerState] | None:
        """Run ``op`` in Redis: (allowed, resulting state), or None if Redis is unavailable."""
        if self.degraded and self._local_clock() < self._retry_at:
            return None
        bc = self.config.routing.breaker
        args = [
            op,
            int(self._clock() * 1000),
            max(1, bc.failure_threshold),
            _ms(bc.window_s),
            _ms(bc.cooldown_s),
            _ms(self.config.probe_lease_for(provider)),
            _ms(bc.state_ttl_s),
            self.instance_id,
        ]
        try:
            async with asyncio.timeout(bc.store_timeout_s):
                allowed, state, *moves = await self._script(keys=[self.key(provider)], args=args)
        except Exception as exc:  # any store failure: never fail the request because of it
            metrics.BREAKER_STORE_ERRORS.labels(op).inc()
            self._degrade(exc)
            return None
        if self.degraded:
            self._recover()
        for old, new in zip(moves[::2], moves[1::2], strict=True):
            _transitioned(provider, self.backend, BreakerState(old), BreakerState(new))
        metrics.observe_breaker(provider, state)
        return bool(int(allowed)), BreakerState(state)

    def _degrade(self, exc: BaseException) -> None:
        self._retry_at = self._local_clock() + self.config.routing.breaker.store_retry_s
        if self.degraded:
            return
        self.degraded = True
        # Start the outage from a clean slate rather than from a previous outage's state.
        self.local = LocalBreakers(self.config, self.providers, self._local_clock)
        metrics.BREAKER_STORE_DEGRADED.set(1)
        log.warning(
            "circuit breaker store unavailable, falling back to in-process breakers",
            extra={"error": repr(exc), "retry_in_s": self.config.routing.breaker.store_retry_s},
        )

    def _recover(self) -> None:
        self.degraded = False
        metrics.BREAKER_STORE_DEGRADED.set(0)
        log.info("circuit breaker store reachable again, using shared breakers")

    async def allow(self, provider: str) -> bool:
        result = await self._eval("allow", provider)
        return await self.local.allow(provider) if result is None else result[0]

    async def record_success(self, provider: str) -> None:
        if await self._eval("success", provider) is None:
            await self.local.record_success(provider)

    async def record_failure(self, provider: str) -> None:
        if await self._eval("failure", provider) is None:
            await self.local.record_failure(provider)

    async def release(self, provider: str) -> None:
        if await self._eval("release", provider) is None:
            await self.local.release(provider)

    async def state(self, provider: str) -> BreakerState:
        result = await self._eval("state", provider)
        return await self.local.state(provider) if result is None else result[1]


def build_breakers(
    config: GatewayConfig, providers: Iterable[str], redis: Redis | None = None
) -> Breakers:
    """The breakers ``routing.breaker.backend`` asks for (memory when no Redis is given)."""
    if config.routing.breaker.backend == "redis" and redis is not None:
        return RedisBreakers(redis, config, providers)
    return LocalBreakers(config, providers)
