from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import select

from llm_gateway.db.models import UsageEvent
from llm_gateway.ratelimit import RateLimiter, estimate_prompt_tokens, format_reset
from llm_gateway.schemas import ChatCompletionRequest, ProviderResponse, Usage
from llm_gateway.service import ChatService
from tests.test_streaming import parse_sse

ADMIN = {"authorization": "Bearer test-admin-key"}
# The gateway estimates this prompt at 4 (message overhead) + 5 (18 chars / 4) = 9 tokens.
# The fake provider reports 5 prompt + 8 completion tokens = 13 for its reply.
CHAT = {"model": "fast", "messages": [{"role": "user", "content": "capital of France?"}]}
PROMPT_ESTIMATE = 9
ACTUAL = 13
DEFAULT_MAX_TOKENS = 256  # tests/conftest.py BASE_CONFIG defaults.max_tokens


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(state) -> Clock:
    """Freeze the limiter's clock so no tokens refill while a test runs."""
    c = Clock()
    state.limiter.clock = c
    return c


async def make_key(state, tpm: int | None, **kw) -> tuple[str, dict[str, str], int]:
    raw, row = await state.keys.create("tpm", "acme", rate_limit_tpm=tpm, **kw)
    return raw, {"authorization": f"Bearer {raw}"}, row.id


async def remaining(state, key_id: int, tpm: int) -> float:
    """Current bucket level (a zero-cost reservation always succeeds and changes nothing)."""
    return (await state.limiter.reserve_tokens(str(key_id), tpm, 0)).remaining


# ---------------------------------------------------------------------- units


def test_estimate_prompt_tokens():
    req = ChatCompletionRequest.model_validate(CHAT)
    assert estimate_prompt_tokens(req) == PROMPT_ESTIMATE
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    with_tools = ChatCompletionRequest.model_validate({**CHAT, "tools": tools})
    assert estimate_prompt_tokens(with_tools) > PROMPT_ESTIMATE
    convo = ChatCompletionRequest.model_validate(
        {
            "model": "fast",
            "messages": [
                {"role": "system", "content": "x" * 400},
                {"role": "user", "content": [{"type": "text", "text": "y" * 40}]},
            ],
        }
    )
    assert estimate_prompt_tokens(convo) == 4 + 100 + 4 + 10


def test_format_reset_matches_openai_style():
    assert format_reset(0) == "0s"
    assert format_reset(-3) == "0s"
    assert format_reset(0.25) == "250ms"
    assert format_reset(0.0001) == "1ms"
    assert format_reset(7.8) == "8s"
    assert format_reset(60) == "1m0s"
    assert format_reset(90.2) == "1m31s"


async def test_reserve_and_settle_refund(state):
    clock = Clock()
    limiter = RateLimiter(state.redis, clock=clock)
    d = await limiter.reserve_tokens("k", 1000, 300)
    assert d.allowed and d.remaining == 700
    assert d.headers()["x-ratelimit-remaining-tokens"] == "700"
    assert d.headers()["x-ratelimit-reset-tokens"] == "18s"  # 300 tokens at 1000/60 per s
    assert "retry-after" not in d.headers()
    # Used only 120 of the 300 reserved: 180 are refunded.
    s = await limiter.settle_tokens("k", 1000, 120 - 300)
    assert s.allowed and s.remaining == 880
    # Refunds never push the bucket above its capacity.
    s = await limiter.settle_tokens("k", 1000, -5000)
    assert s.remaining == 1000


async def test_settle_charges_the_difference_into_debt(state):
    clock = Clock()
    limiter = RateLimiter(state.redis, clock=clock)
    assert (await limiter.reserve_tokens("k", 600, 100)).allowed
    # Used 900 instead of the 100 reserved: the bucket goes 300 into debt.
    s = await limiter.settle_tokens("k", 600, 800)
    assert s.remaining == -300
    assert s.headers()["x-ratelimit-remaining-tokens"] == "0"
    assert s.headers()["x-ratelimit-reset-tokens"] == "1m30s"  # 900 tokens at 10/s
    d = await limiter.reserve_tokens("k", 600, 50)
    assert not d.allowed
    assert d.retry_after_s == pytest.approx(35.0)  # 350 tokens at 10/s
    assert d.headers()["retry-after"] == "35"
    # The key stays in the bucket until it would be full again.
    assert await state.redis.pttl("ratelimit:tpm:k") > 90_000
    clock.t += 35
    assert (await limiter.reserve_tokens("k", 600, 50)).allowed


async def test_tpm_and_rpm_buckets_are_separate(state):
    limiter = RateLimiter(state.redis, clock=Clock())
    assert (await limiter.reserve_tokens("k", 10, 10)).allowed
    assert (await limiter.hit("k", rpm=1)).allowed
    assert not (await limiter.reserve_tokens("k", 10, 1)).allowed
    assert not (await limiter.hit("k", rpm=1)).allowed
    assert (await limiter.reserve_tokens("other", 10, 10)).allowed


async def test_concurrent_reservations_never_overshoot(state):
    limiter = RateLimiter(state.redis, clock=Clock())
    decisions = await asyncio.gather(*(limiter.reserve_tokens("k", 1000, 70) for _ in range(50)))
    assert sum(d.allowed for d in decisions) == 14  # floor(1000 / 70)
    assert min(d.remaining for d in decisions) == pytest.approx(20)
    assert (await limiter.reserve_tokens("k", 1000, 0)).remaining == pytest.approx(20)


# ---------------------------------------------------------------------- over HTTP


async def test_estimate_reserved_then_settled_to_actual_usage(state, anon_client, clock):
    _, headers, key_id = await make_key(state, 1000)
    r = await anon_client.post(
        "/v1/chat/completions", json={**CHAT, "max_tokens": 100}, headers=headers
    )
    assert r.status_code == 200
    assert r.json()["usage"]["total_tokens"] == ACTUAL
    assert r.headers["x-ratelimit-limit-tokens"] == "1000"
    # Headers report the bucket after settlement, not after the 109-token reservation.
    assert r.headers["x-ratelimit-remaining-tokens"] == str(1000 - ACTUAL)
    assert r.headers["x-ratelimit-reset-tokens"] == "780ms"  # 13 tokens at 1000/60 per s
    assert r.headers["x-ratelimit-limit-requests"]  # the request limit still applies
    assert await remaining(state, key_id, 1000) == 1000 - ACTUAL


async def test_reservation_without_max_tokens_uses_default_allowance(state, anon_client, clock):
    _, headers, _ = await make_key(state, 1000)
    reserved: list[int] = []
    original = state.limiter.reserve_tokens

    async def spy(bucket, tpm, tokens):
        reserved.append(tokens)
        return await original(bucket, tpm, tokens)

    state.limiter.reserve_tokens = spy
    await anon_client.post("/v1/chat/completions", json=CHAT, headers=headers)
    state.config.limits.completion_token_allowance = 20
    await anon_client.post("/v1/chat/completions", json=CHAT, headers=headers)
    await anon_client.post(
        "/v1/chat/completions", json={**CHAT, "max_completion_tokens": 7}, headers=headers
    )
    assert reserved == [
        PROMPT_ESTIMATE + DEFAULT_MAX_TOKENS,
        PROMPT_ESTIMATE + 20,
        PROMPT_ESTIMATE + 7,
    ]


async def test_tpm_exhausted_returns_429_with_headers(state, anon_client, clock):
    # tpm=100 < estimate (9 + 256), so each request reserves the whole bucket.
    _, headers, _ = await make_key(state, 100)
    first = await anon_client.post("/v1/chat/completions", json=CHAT, headers=headers)
    assert first.status_code == 200
    assert first.headers["x-ratelimit-remaining-tokens"] == str(100 - ACTUAL)

    r = await anon_client.post(
        "/v1/chat/completions", json={**CHAT, "temperature": 0.5}, headers=headers
    )
    assert r.status_code == 429
    err = r.json()["error"]
    assert err["code"] == "rate_limit_exceeded" and err["type"] == "rate_limit_error"
    assert "100 tokens/minute" in err["message"]
    assert r.headers["retry-after"] == "8"  # 13 missing tokens at 100/60 per s = 7.8 s
    assert r.headers["x-ratelimit-limit-tokens"] == "100"
    assert r.headers["x-ratelimit-remaining-tokens"] == str(100 - ACTUAL)
    assert r.headers["x-ratelimit-reset-tokens"] == "8s"
    assert r.headers["x-ratelimit-limit-requests"]
    # Rejected before the provider and kept out of the ledger.
    assert state.providers["fake"].calls["fake-fast"] == 1
    async with state.sessionmaker() as s:
        assert len((await s.scalars(select(UsageEvent))).all()) == 1

    clock.t += 8
    ok = await anon_client.post(
        "/v1/chat/completions", json={**CHAT, "temperature": 0.5}, headers=headers
    )
    assert ok.status_code == 200


async def test_streaming_is_limited_by_the_same_bucket(state, anon_client, clock):
    _, headers, _ = await make_key(state, 100)
    assert (await anon_client.post("/v1/chat/completions", json=CHAT, headers=headers)).is_success
    r = await anon_client.post(
        "/v1/chat/completions", json={**CHAT, "stream": True, "temperature": 1}, headers=headers
    )
    assert r.status_code == 429
    assert r.headers["retry-after"] == "8"


async def test_prompt_larger_than_limit_is_rejected_outright(state, anon_client, clock):
    _, headers, _ = await make_key(state, 5)
    r = await anon_client.post("/v1/chat/completions", json=CHAT, headers=headers)
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "request_too_large"
    assert "retry-after" not in r.headers  # waiting would not help
    assert r.headers["x-ratelimit-limit-tokens"] == "5"
    assert state.providers["fake"].calls == {}


async def test_provider_failure_refunds_the_reservation(state, anon_client, clock):
    _, headers, key_id = await make_key(state, 1000)
    r = await anon_client.post(
        "/v1/chat/completions", json={**CHAT, "model": "fake/fail-400"}, headers=headers
    )
    assert r.status_code == 400
    assert await remaining(state, key_id, 1000) == 1000
    r = await anon_client.post(
        "/v1/chat/completions",
        json={**CHAT, "model": "fake/fail-500", "stream": True},
        headers=headers,
    )
    assert r.status_code >= 500
    assert await remaining(state, key_id, 1000) == 1000


async def test_usage_above_the_estimate_is_charged(state, anon_client, clock):
    _, headers, key_id = await make_key(state, 2000)
    state.providers["fake"]._respond = lambda req: ProviderResponse("big", usage=Usage(900, 600))
    r = await anon_client.post(
        "/v1/chat/completions", json={**CHAT, "max_tokens": 10}, headers=headers
    )
    assert r.status_code == 200
    assert r.headers["x-ratelimit-remaining-tokens"] == "500"
    assert await remaining(state, key_id, 2000) == 500


async def test_usage_not_reported_falls_back_to_estimate(state, anon_client, clock):
    _, headers, key_id = await make_key(state, 1000)
    state.providers["fake"]._respond = lambda req: ProviderResponse("x" * 40, usage=Usage())
    r = await anon_client.post("/v1/chat/completions", json=CHAT, headers=headers)
    assert r.status_code == 200
    assert await remaining(state, key_id, 1000) == 1000 - (PROMPT_ESTIMATE + 10)


async def test_concurrent_requests_cannot_overshoot(state, anon_client, clock):
    # Each request reserves 9 + 100 = 109 tokens; 4 fit in 500, the 5th would not.
    _, headers, key_id = await make_key(state, 500)
    state.providers["fake"].latency_s = 0.05  # keep every request in flight together
    responses = await asyncio.gather(
        *(
            anon_client.post(
                "/v1/chat/completions",
                json={**CHAT, "max_tokens": 100, "temperature": 1},
                headers=headers,
            )
            for _ in range(12)
        )
    )
    statuses = sorted(r.status_code for r in responses)
    assert statuses == [200] * 4 + [429] * 8
    assert state.providers["fake"].calls["fake-fast"] == 4
    assert await remaining(state, key_id, 500) == 500 - 4 * ACTUAL


async def test_stream_settles_from_final_usage_chunk(state, anon_client, clock):
    _, headers, key_id = await make_key(state, 1000)
    r = await anon_client.post(
        "/v1/chat/completions", json={**CHAT, "stream": True}, headers=headers
    )
    assert r.status_code == 200
    assert parse_sse(r.text)[-1] == "[DONE]"
    # Headers go out before the body, so they show the bucket after the reservation.
    assert r.headers["x-ratelimit-limit-tokens"] == "1000"
    assert r.headers["x-ratelimit-remaining-tokens"] == str(
        1000 - PROMPT_ESTIMATE - DEFAULT_MAX_TOKENS
    )
    assert await remaining(state, key_id, 1000) == 1000 - ACTUAL


async def test_stream_client_disconnect_settles_with_what_is_known(state, clock):
    raw, _, key_id = await make_key(state, 1000)
    ctx = await state.keys.authenticate(raw)
    req = ChatCompletionRequest.model_validate({**CHAT, "stream": True})
    result = await ChatService(state).stream(ctx, req, "req_disconnect")
    events = result.events
    seen = [await anext(events) for _ in range(2)]  # role chunk + first content delta ("The ")
    assert '"content":"The "' in seen[1]
    assert await remaining(state, key_id, 1000) == 1000 - PROMPT_ESTIMATE - DEFAULT_MAX_TOKENS
    await events.aclose()  # the client goes away before the usage chunk arrives
    # No usage was reported, so the charge is the prompt estimate plus 1 token received.
    assert await remaining(state, key_id, 1000) == 1000 - (PROMPT_ESTIMATE + 1)
    async with state.sessionmaker() as s:
        event = (await s.scalars(select(UsageEvent))).one()
    assert event.status == "client_closed"


@pytest.mark.parametrize("stream", [False, True])
async def test_cache_hits_do_not_consume_tpm(state, anon_client, clock, stream):
    _, headers, key_id = await make_key(state, 100)
    body = {**CHAT, "temperature": 0, "stream": stream}
    first = await anon_client.post("/v1/chat/completions", json=body, headers=headers)
    assert first.status_code == 200 and first.headers["x-gateway-cache"] == "miss"
    # 87 tokens left is less than the next reservation would need (100), but a cache hit
    # makes no provider call, so it is neither reserved nor charged.
    for _ in range(3):
        hit = await anon_client.post("/v1/chat/completions", json=body, headers=headers)
        assert hit.status_code == 200 and hit.headers["x-gateway-cache"] == "hit"
        assert "x-ratelimit-limit-tokens" not in hit.headers
    assert await remaining(state, key_id, 100) == 100 - ACTUAL
    assert state.providers["fake"].calls["fake-fast"] == 1


async def test_default_tpm_from_config_and_unlimited_when_unset(state, anon_client, clock):
    _, headers, key_id = await make_key(state, None)
    r = await anon_client.post("/v1/chat/completions", json=CHAT, headers=headers)
    assert "x-ratelimit-limit-tokens" not in r.headers  # default_tpm is None: unlimited
    state.config.limits.default_tpm = 50
    r = await anon_client.post(
        "/v1/chat/completions", json={**CHAT, "temperature": 1}, headers=headers
    )
    assert r.headers["x-ratelimit-limit-tokens"] == "50"
    assert await remaining(state, key_id, 50) == 50 - ACTUAL
    # A key's own limit takes precedence over the default.
    _, own, _ = await make_key(state, 70)
    r = await anon_client.post("/v1/chat/completions", json={**CHAT, "temperature": 1}, headers=own)
    assert r.headers["x-ratelimit-limit-tokens"] == "70"


# ---------------------------------------------------------------------- admin API


async def test_admin_create_and_update_tpm(state, anon_client, clock):
    r = await anon_client.post(
        "/admin/keys",
        headers=ADMIN,
        json={"name": "svc", "tenant": "acme", "rate_limit_tpm": 5000},
    )
    assert r.status_code == 201
    created = r.json()
    assert created["rate_limit_tpm"] == 5000
    user = {"authorization": f"Bearer {created['key']}"}
    chat = await anon_client.post("/v1/chat/completions", json=CHAT, headers=user)
    assert chat.headers["x-ratelimit-limit-tokens"] == "5000"

    r = await anon_client.patch(
        f"/admin/keys/{created['id']}", headers=ADMIN, json={"rate_limit_tpm": 7}
    )
    assert r.status_code == 200
    view = r.json()
    assert view["rate_limit_tpm"] == 7
    assert view["rate_limit_rpm"] is None and view["monthly_budget_usd"] is None  # untouched
    # Takes effect immediately (the key cache is invalidated).
    chat = await anon_client.post("/v1/chat/completions", json=CHAT, headers=user)
    assert chat.status_code == 429 and chat.json()["error"]["code"] == "request_too_large"

    listed = (await anon_client.get("/admin/keys", headers=ADMIN)).json()["data"]
    assert next(k for k in listed if k["id"] == created["id"])["rate_limit_tpm"] == 7

    # null clears the key's limit (back to the config default, here unlimited).
    r = await anon_client.patch(
        f"/admin/keys/{created['id']}",
        headers=ADMIN,
        json={"rate_limit_tpm": None, "rate_limit_rpm": 30, "monthly_budget_usd": 2.5},
    )
    assert r.json()["rate_limit_tpm"] is None
    assert r.json()["rate_limit_rpm"] == 30 and r.json()["monthly_budget_usd"] == 2.5
    chat = await anon_client.post("/v1/chat/completions", json=CHAT, headers=user)
    assert chat.status_code == 200 and "x-ratelimit-limit-tokens" not in chat.headers
    assert chat.headers["x-ratelimit-limit-requests"] == "30"


async def test_admin_tpm_validation(anon_client):
    r = await anon_client.post(
        "/admin/keys", headers=ADMIN, json={"name": "a", "tenant": "t", "rate_limit_tpm": 0}
    )
    assert r.status_code == 400
    created = (
        await anon_client.post("/admin/keys", headers=ADMIN, json={"name": "a", "tenant": "t"})
    ).json()
    assert created["rate_limit_tpm"] is None
    url = f"/admin/keys/{created['id']}"
    assert (await anon_client.patch(url, json={"rate_limit_tpm": 5})).status_code == 401
    assert (
        await anon_client.patch(url, headers=ADMIN, json={"rate_limit_tpm": -1})
    ).status_code == 400
    assert (await anon_client.patch(url, headers=ADMIN, json={"tenant": "x"})).status_code == 400
    assert (
        await anon_client.patch("/admin/keys/99999", headers=ADMIN, json={"rate_limit_tpm": 5})
    ).status_code == 404
