from __future__ import annotations

from sqlalchemy import select

from llm_gateway.cache import cache_key, is_cacheable
from llm_gateway.db.models import UsageEvent
from llm_gateway.schemas import ChatCompletionRequest
from tests.test_streaming import parse_sse

CHAT = {
    "model": "fast",
    "temperature": 0,
    "messages": [{"role": "user", "content": "capital of France?"}],
}


def r(**kw) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate({**CHAT, **kw})


def test_only_temperature_zero_is_cacheable():
    assert is_cacheable(r())
    assert not is_cacheable(r(temperature=0.7))
    assert not is_cacheable(r(temperature=None))


def test_key_normalization():
    base = cache_key("t", r())
    # Transport-only fields and equivalent shapes produce the same key.
    assert cache_key("t", r(stream=True, user="u1", stream_options={"include_usage": True})) == base
    parts = [{"role": "user", "content": [{"type": "text", "text": "capital of France?"}]}]
    assert cache_key("t", r(messages=parts)) == base
    assert cache_key("t", r(max_tokens=50)) == cache_key("t", r(max_completion_tokens=50))
    assert cache_key("t", r(stop="x")) == cache_key("t", r(stop=["x"]))
    # Anything that changes the output changes the key; tenants are isolated.
    assert cache_key("t", r(model="smart")) != base
    assert cache_key("t", r(max_tokens=10)) != base
    assert cache_key("other-tenant", r()) != base


async def test_hit_after_miss(state, client):
    first = await client.post("/v1/chat/completions", json=CHAT)
    second = await client.post("/v1/chat/completions", json=CHAT)
    assert first.headers["x-gateway-cache"] == "miss"
    assert second.headers["x-gateway-cache"] == "hit"
    assert second.headers["x-gateway-cost-usd"] == "0.00000000"
    assert second.headers["x-gateway-provider"] == "fake"
    a, b = first.json(), second.json()
    assert a["choices"] == b["choices"] and a["usage"] == b["usage"]
    assert state.providers["fake"].calls["fake-fast"] == 1

    async with state.sessionmaker() as s:
        events = (await s.scalars(select(UsageEvent).order_by(UsageEvent.id))).all()
    assert [e.cache_hit for e in events] == [False, True]
    assert float(events[1].cost_usd) == 0 and events[1].prompt_tokens == 0


async def test_non_zero_temperature_bypasses(client):
    r1 = await client.post("/v1/chat/completions", json={**CHAT, "temperature": 0.5})
    r2 = await client.post("/v1/chat/completions", json={**CHAT, "temperature": 0.5})
    assert r1.headers["x-gateway-cache"] == r2.headers["x-gateway-cache"] == "bypass"


async def test_no_cache_header_skips_read(state, client):
    await client.post("/v1/chat/completions", json=CHAT)
    r = await client.post("/v1/chat/completions", json=CHAT, headers={"cache-control": "no-cache"})
    assert r.headers["x-gateway-cache"] == "miss"
    assert state.providers["fake"].calls["fake-fast"] == 2


async def test_streaming_populates_and_replays_cache(state, client):
    body = {**CHAT, "stream": True}
    s1 = await client.post("/v1/chat/completions", json=body)
    s2 = await client.post("/v1/chat/completions", json=body)
    assert s1.headers["x-gateway-cache"] == "miss"
    assert s2.headers["x-gateway-cache"] == "hit"

    def text(resp):
        return "".join(
            c["choices"][0]["delta"].get("content", "")
            for c in parse_sse(resp.text)[:-1]
            if c["choices"]
        )

    assert text(s1) == text(s2) == "The capital of France is Paris."
    # The streamed entry also serves non-streaming requests.
    n = await client.post("/v1/chat/completions", json=CHAT)
    assert n.headers["x-gateway-cache"] == "hit"
    assert state.providers["fake"].calls["fake-fast"] == 1


async def test_cache_is_scoped_per_tenant(state, client, anon_client):
    await client.post("/v1/chat/completions", json=CHAT)
    raw, _ = await state.keys.create("other", "globex")
    r = await anon_client.post(
        "/v1/chat/completions", json=CHAT, headers={"authorization": f"Bearer {raw}"}
    )
    assert r.headers["x-gateway-cache"] == "miss"


async def test_ttl_is_applied(state, client):
    await client.post("/v1/chat/completions", json=CHAT)
    keys = await state.redis.keys("cache:*")
    assert len(keys) == 1
    ttl = await state.redis.ttl(keys[0])
    assert 0 < ttl <= state.config.cache.ttl_s
