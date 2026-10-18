from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import select

from llm_gateway.auth import AuthContext
from llm_gateway.budget import BudgetTracker
from llm_gateway.db.models import UsageEvent
from llm_gateway.ratelimit import RateLimiter
from llm_gateway.redis_client import make_redis

CHAT = {"model": "fast", "messages": [{"role": "user", "content": "capital of France?"}]}


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


async def test_token_bucket_allows_burst_then_refills():
    redis = make_redis("memory://")
    clock = Clock()
    limiter = RateLimiter(redis, clock=clock)
    decisions = [await limiter.hit("k", rpm=3) for _ in range(4)]
    assert [d.allowed for d in decisions] == [True, True, True, False]
    assert [d.remaining for d in decisions[:3]] == [2, 1, 0]
    assert decisions[3].retry_after_s == pytest.approx(20.0, abs=0.01)  # 1 token / (3/60 per s)
    clock.t += 20
    assert (await limiter.hit("k", rpm=3)).allowed
    assert not (await limiter.hit("k", rpm=3)).allowed
    # Buckets are independent per key.
    assert (await limiter.hit("other", rpm=3)).allowed
    await redis.aclose()


async def test_rate_limit_enforced_over_http(state, anon_client):
    raw, _ = await state.keys.create("limited", "acme", rate_limit_rpm=2)
    headers = {"authorization": f"Bearer {raw}"}
    r1 = await anon_client.post("/v1/chat/completions", json=CHAT, headers=headers)
    r2 = await anon_client.post("/v1/chat/completions", json=CHAT, headers=headers)
    r3 = await anon_client.post("/v1/chat/completions", json=CHAT, headers=headers)
    assert [r.status_code for r in (r1, r2, r3)] == [200, 200, 429]
    assert r1.headers["x-ratelimit-limit-requests"] == "2"
    assert r2.headers["x-ratelimit-remaining-requests"] == "0"
    assert r3.json()["error"]["code"] == "rate_limit_exceeded"
    assert int(r3.headers["retry-after"]) >= 1
    # Streaming requests are limited by the same bucket.
    r4 = await anon_client.post(
        "/v1/chat/completions", json={**CHAT, "stream": True}, headers=headers
    )
    assert r4.status_code == 429


async def test_budget_blocks_once_exhausted(state, anon_client):
    # Fake pricing is $0.50 / $1.50 per 1M tokens, so ~1e-5 USD per request.
    raw, _row = await state.keys.create("budgeted", "acme", monthly_budget_usd=0.00001)
    headers = {"authorization": f"Bearer {raw}"}
    first = await anon_client.post("/v1/chat/completions", json=CHAT, headers=headers)
    assert first.status_code == 200
    assert float(first.headers["x-gateway-budget-remaining-usd"]) == pytest.approx(0.00001)
    cost = float(first.headers["x-gateway-cost-usd"])
    assert cost > 0

    statuses = []
    for _ in range(5):
        r = await anon_client.post("/v1/chat/completions", json=CHAT, headers=headers)
        statuses.append(r.status_code)
        if r.status_code == 402:
            assert r.json()["error"]["code"] == "monthly_budget_exceeded"
            break
    assert statuses[-1] == 402
    # Overshoot is bounded by one request's cost.
    spent = await state.budgets.spent_usd(await state.keys.authenticate(raw))
    assert spent < 0.00001 + cost + 1e-12


async def test_budget_counter_rebuilds_from_ledger(state):
    _raw, row = await state.keys.create("ledger", "acme", monthly_budget_usd=1)
    now = datetime(2026, 10, 15, tzinfo=UTC)
    async with state.sessionmaker() as s:
        for when, cost in ((now, "0.25"), (now, "0.5"), (datetime(2026, 9, 30, tzinfo=UTC), "9")):
            s.add(
                UsageEvent(
                    request_id="x",
                    api_key_id=row.id,
                    requested_model="fast",
                    status="ok",
                    http_status=200,
                    cost_usd=Decimal(cost),
                    created_at=when,
                )
            )
        await s.commit()
    tracker = BudgetTracker(state.redis, state.sessionmaker, clock=lambda: now)
    ctx = AuthContext(row.id, "ledger", "acme", monthly_budget_usd=Decimal(1))
    assert await tracker.spent_usd(ctx) == pytest.approx(0.75)  # September excluded
    await tracker.add(ctx, 0.1)
    assert await tracker.remaining_usd(ctx) == pytest.approx(0.15)
    await state.redis.flushall()
    assert await tracker.spent_usd(ctx) == pytest.approx(0.75)  # rebuilt from the ledger


async def test_rejected_requests_never_reach_provider(state, anon_client):
    raw, _ = await state.keys.create("zero", "acme", monthly_budget_usd=0)
    r = await anon_client.post(
        "/v1/chat/completions", json=CHAT, headers={"authorization": f"Bearer {raw}"}
    )
    assert r.status_code == 402
    assert state.providers["fake"].calls == {}
    async with state.sessionmaker() as s:
        assert (await s.scalars(select(UsageEvent))).all() == []
