"""Circuit breakers shared across replicas through Redis (``routing.breaker.backend: redis``).

Each "gateway instance" here is its own Router with its own providers and its own Redis client,
as separate replicas would have; they share only the Redis server. With ``TEST_REDIS_URL`` set
(the CI services job) that is a real Redis, otherwise one fakeredis server.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis, FakeServer
from prometheus_client import REGISTRY
from redis.asyncio import Redis

from llm_gateway.breaker import BreakerState, LocalBreakers, RedisBreakers
from llm_gateway.config import GatewayConfig
from llm_gateway.providers.base import InvalidRequestError
from llm_gateway.router import Router
from llm_gateway.schemas import ChatCompletionRequest
from llm_gateway.state import build_providers, build_state


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def make_config(**breaker: Any) -> GatewayConfig:
    return GatewayConfig.model_validate(
        {
            "defaults": {"timeout_s": 1},
            "providers": {"primary": {"type": "fake"}, "backup": {"type": "fake"}},
            "aliases": {
                "failing": ["primary/fail-500", "backup/ok"],
                "healthy": ["primary/ok", "backup/ok"],
                "invalid": ["primary/fail-400", "backup/ok"],
            },
            "routing": {
                "retries": 0,
                "backoff_base_s": 0,
                "backoff_max_s": 0,
                "breaker": {
                    "backend": "redis",
                    "failure_threshold": 2,
                    "window_s": 60,
                    "cooldown_s": 10,
                    "probe_lease_s": 5,
                    **breaker,
                },
            },
        }
    )


def gateway(redis: Redis, cfg: GatewayConfig, clock: Callable[[], float]) -> Router:
    providers = build_providers(cfg)
    breakers = RedisBreakers(redis, cfg, providers, clock=clock, local_clock=clock)
    return Router(cfg, providers, breakers=breakers)


def req(model: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[{"role": "user", "content": "hi"}])


def outcomes(result) -> list[str]:
    return [a.outcome for a in result.attempts]


def sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.fixture
async def connect() -> AsyncIterator[Callable[[], Redis]]:
    """Open clients to one shared Redis server (a fresh client per simulated replica)."""
    url = os.environ.get("TEST_REDIS_URL")
    server = FakeServer()
    clients: list[Redis] = []

    def _connect() -> Redis:
        r = (
            Redis.from_url(url, decode_responses=True)
            if url
            else FakeAsyncRedis(server=server, decode_responses=True)
        )
        clients.append(r)
        return r

    await _connect().flushdb()
    yield _connect
    for c in clients:
        await c.aclose()


@pytest.fixture
def clock() -> Clock:
    return Clock()


async def trip(router: Router) -> None:
    for _ in range(router.config.routing.breaker.failure_threshold):
        await router.complete(req("failing"))


async def test_two_instances_share_open_state(connect, clock):
    cfg = make_config()
    a, b = gateway(connect(), cfg, clock), gateway(connect(), cfg, clock)
    await trip(a)
    assert await a.breakers.state("primary") is BreakerState.OPEN
    assert await b.breakers.state("primary") is BreakerState.OPEN

    result = await b.complete(req("failing"))
    assert outcomes(result) == ["circuit_open", "ok"]
    assert b.providers["primary"].calls == {}  # B never called the provider A saw failing


async def test_failures_from_both_instances_count_together(connect, clock):
    cfg = make_config()
    a, b = gateway(connect(), cfg, clock), gateway(connect(), cfg, clock)
    await a.complete(req("failing"))
    assert await b.breakers.state("primary") is BreakerState.CLOSED
    await b.complete(req("failing"))  # the second failure overall reaches the threshold
    assert await a.breakers.state("primary") is BreakerState.OPEN


async def test_failures_outside_the_window_are_forgotten(connect, clock):
    cfg = make_config(window_s=5)
    a, b = gateway(connect(), cfg, clock), gateway(connect(), cfg, clock)
    await a.complete(req("failing"))
    clock.t += 5
    await b.complete(req("failing"))  # starts a new window
    assert await a.breakers.state("primary") is BreakerState.CLOSED
    clock.t += 1
    await a.complete(req("failing"))
    assert await b.breakers.state("primary") is BreakerState.OPEN


async def test_single_half_open_probe_under_concurrency(connect, clock):
    cfg = make_config()
    replicas = [gateway(connect(), cfg, clock) for _ in range(3)]
    await trip(replicas[0])
    clock.t += 10  # cooldown over: half-open

    calls = [r.breakers.allow("primary") for r in replicas for _ in range(10)]
    granted = await asyncio.gather(*calls)
    assert sum(granted) == 1
    assert await replicas[1].breakers.state("primary") is BreakerState.HALF_OPEN

    clock.t += 4.9  # the probe's lease (probe_lease_s=5) still holds
    assert not any(await asyncio.gather(*(r.breakers.allow("primary") for r in replicas)))
    clock.t += 0.1  # lease expired without an outcome: exactly one new probe
    assert sum(await asyncio.gather(*(r.breakers.allow("primary") for r in replicas))) == 1


async def test_recovery_after_a_successful_probe(connect, clock):
    cfg = make_config()
    a, b = gateway(connect(), cfg, clock), gateway(connect(), cfg, clock)
    labels = {"provider": "primary", "backend": "redis"}
    before = {
        (f, t): sample("llm_gateway_circuit_transitions_total", from_state=f, to_state=t, **labels)
        for f, t in [("closed", "open"), ("open", "half_open"), ("half_open", "closed")]
    }
    await trip(a)
    clock.t += 10

    result = await b.complete(req("healthy"))  # B gets the probe, and it succeeds
    assert outcomes(result) == ["ok"] and result.target.id == "primary/ok"
    assert await a.breakers.state("primary") is BreakerState.CLOSED
    assert outcomes(await a.complete(req("healthy"))) == ["ok"]

    for (f, t), value in before.items():
        after = sample("llm_gateway_circuit_transitions_total", from_state=f, to_state=t, **labels)
        assert after == value + 1, (f, t)
    assert sample("llm_gateway_circuit_state", provider="primary") == 0


async def test_failed_probe_reopens_for_every_instance(connect, clock):
    cfg = make_config()
    a, b = gateway(connect(), cfg, clock), gateway(connect(), cfg, clock)
    await trip(a)
    clock.t += 10
    result = await a.complete(req("failing"))  # the probe fails
    assert outcomes(result) == ["unavailable", "ok"]
    assert await b.breakers.state("primary") is BreakerState.OPEN
    clock.t += 9.9  # a fresh cooldown, counted from the failed probe
    assert not await b.breakers.allow("primary")
    clock.t += 0.1
    assert await b.breakers.allow("primary")


async def test_invalid_request_hands_back_the_probe(connect, clock):
    cfg = make_config()
    a, b = gateway(connect(), cfg, clock), gateway(connect(), cfg, clock)
    await trip(a)
    clock.t += 10
    with pytest.raises(InvalidRequestError):
        await a.complete(req("invalid"))  # took the probe, but says nothing about the provider
    assert await b.breakers.allow("primary")  # released at once, not after the lease


async def test_ttl_cleanup(connect, clock):
    cfg = make_config(window_s=60, cooldown_s=10, state_ttl_s=100)
    a = gateway(connect(), cfg, clock)
    redis, key = a.breakers.redis, a.breakers.key("primary")

    await a.breakers.record_success("primary")
    assert not await redis.exists(key)  # a healthy provider has no state at all
    await a.breakers.record_failure("primary")
    assert 0 < await redis.pttl(key) <= 60_000  # failures expire with their window
    await a.breakers.record_success("primary")
    assert not await redis.exists(key)

    await trip(a)
    assert 100_000 < await redis.pttl(key) <= 110_000  # cooldown + state_ttl_s
    clock.t += 10
    assert await a.breakers.allow("primary")  # half-open writes refresh the TTL
    assert 99_000 < await redis.pttl(key) <= 100_000


async def test_state_expires_in_redis(connect):
    # Real time here: Redis expires keys on its own clock.
    cfg = make_config(failure_threshold=2, window_s=0.05, cooldown_s=0.05, state_ttl_s=0.05)
    a = gateway(connect(), cfg, time.time)
    key = a.breakers.key("primary")
    await a.breakers.record_failure("primary")
    assert await a.breakers.redis.exists(key)
    await asyncio.sleep(0.2)
    assert not await a.breakers.redis.exists(key)

    await a.breakers.record_failure("primary")
    await a.breakers.record_failure("primary")
    assert await a.breakers.state("primary") is BreakerState.OPEN
    await asyncio.sleep(0.3)
    assert not await a.breakers.redis.exists(key)
    assert await a.breakers.state("primary") is BreakerState.CLOSED


async def test_redis_outage_falls_back_to_in_process_breakers(clock, caplog):
    cfg = make_config(store_timeout_s=1)
    down = Redis.from_url("redis://127.0.0.1:1/0", decode_responses=True)
    a = gateway(down, cfg, clock)
    errors = sample("llm_gateway_circuit_store_errors_total", op="allow")
    try:
        with caplog.at_level(logging.WARNING, logger="llm_gateway.breaker"):
            for _ in range(2):
                result = await a.complete(req("failing"))  # requests still succeed
                assert result.target.id == "backup/ok"
        assert a.breakers.degraded
        assert sample("llm_gateway_circuit_store_degraded") == 1
        assert sample("llm_gateway_circuit_store_errors_total", op="allow") == errors + 1
        assert any("store unavailable" in r.getMessage() for r in caplog.records)
        # The in-process fallback still breaks the circuit.
        assert outcomes(await a.complete(req("failing"))) == ["circuit_open", "ok"]
        assert a.breakers.local["primary"].state is BreakerState.OPEN
    finally:
        await down.aclose()


async def test_store_is_retried_after_an_outage(clock):
    server = FakeServer()
    cfg = make_config(store_retry_s=5)
    a = gateway(FakeAsyncRedis(server=server, decode_responses=True), cfg, clock)
    other = gateway(FakeAsyncRedis(server=server, decode_responses=True), cfg, clock)
    await trip(other)  # shared state: open

    server.connected = False
    assert await a.breakers.allow("primary")  # local fallback starts closed
    assert a.breakers.degraded

    server.connected = True
    clock.t += 4.9
    assert await a.breakers.allow("primary")  # not retried yet: still local
    assert a.breakers.degraded
    clock.t += 0.1
    assert not await a.breakers.allow("primary")  # back on Redis, which says open
    assert not a.breakers.degraded
    assert sample("llm_gateway_circuit_store_degraded") == 0


class _HangingRedis:
    def register_script(self, script: str):
        async def run(**kwargs: Any) -> list[Any]:
            await asyncio.sleep(10)
            return [1, "closed"]

        return run


async def test_slow_store_times_out_instead_of_blocking(clock):
    cfg = make_config(store_timeout_s=0.05)
    providers = build_providers(cfg)
    breakers = RedisBreakers(_HangingRedis(), cfg, providers, clock=clock, local_clock=clock)
    router = Router(cfg, providers, breakers=breakers)
    started = time.perf_counter()
    result = await router.complete(req("healthy"))
    assert result.target.id == "primary/ok"
    assert time.perf_counter() - started < 1
    assert breakers.degraded


async def test_backend_switch(settings, gateway_config):
    st = await build_state(settings, gateway_config)
    try:
        assert isinstance(st.router.breakers, LocalBreakers)  # default: memory
    finally:
        await st.aclose()

    cfg = gateway_config.model_copy(deep=True)
    cfg.routing.breaker.backend = "redis"
    st = await build_state(settings, cfg)
    try:
        assert isinstance(st.router.breakers, RedisBreakers)
        assert st.router.breakers.redis is st.redis
    finally:
        await st.aclose()
