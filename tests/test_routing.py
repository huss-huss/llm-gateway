from __future__ import annotations

from typing import Any

import pytest

from llm_gateway.config import GatewayConfig, Target, expand_env
from llm_gateway.providers.base import InvalidRequestError
from llm_gateway.router import AllTargetsFailedError, NoRouteError, Router
from llm_gateway.schemas import ChatCompletionRequest
from llm_gateway.state import build_providers


def make_router(aliases: dict[str, list[str]], **routing: Any) -> tuple[Router, list[float]]:
    cfg = GatewayConfig.model_validate(
        {
            "defaults": {"timeout_s": 0.2},
            "providers": {
                "primary": {"type": "fake", "slow_s": 1, "flaky_failures": 1},
                "backup": {"type": "fake"},
            },
            "aliases": aliases,
            "routing": {"retries": 2, "backoff_base_s": 0.1, "backoff_max_s": 1, **routing},
        }
    )
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    return Router(cfg, build_providers(cfg), sleep=fake_sleep), sleeps


def req(model: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[{"role": "user", "content": "hi"}])


async def test_first_target_success():
    router, sleeps = make_router({"m": ["primary/ok", "backup/ok"]})
    result = await router.complete(req("m"))
    assert result.target.id == "primary/ok"
    assert [a.outcome for a in result.attempts] == ["ok"]
    assert sleeps == []


async def test_retry_then_succeed_on_same_target():
    router, sleeps = make_router({"m": ["primary/flaky", "backup/ok"]})
    result = await router.complete(req("m"))
    assert result.target.id == "primary/flaky"
    assert [a.outcome for a in result.attempts] == ["unavailable", "ok"]
    assert len(sleeps) == 1 and 0.05 <= sleeps[0] <= 0.1


async def test_fallback_after_retries_exhausted_on_429():
    router, sleeps = make_router({"m": ["primary/fail-429", "backup/ok"]})
    result = await router.complete(req("m"))
    assert result.target.id == "backup/ok"
    assert [a.outcome for a in result.attempts] == ["rate_limited"] * 3 + ["ok"]
    assert len(sleeps) == 2  # exponential backoff between the 3 attempts
    assert sleeps[1] > sleeps[0] * 0.9 or sleeps[1] >= 0.1


async def test_timeout_triggers_fallback():
    router, _ = make_router({"m": ["primary/slow", "backup/ok"]}, retries=0)
    result = await router.complete(req("m"))
    assert result.target.id == "backup/ok"
    assert result.attempts[0].outcome == "timeout"


async def test_auth_error_falls_back_without_retry():
    router, sleeps = make_router({"m": ["primary/fail-auth", "backup/ok"]})
    result = await router.complete(req("m"))
    assert [a.outcome for a in result.attempts] == ["auth", "ok"]
    assert sleeps == []


async def test_invalid_request_fails_fast_without_fallback():
    router, _ = make_router({"m": ["primary/fail-400", "backup/ok"]})
    with pytest.raises(InvalidRequestError):
        await router.complete(req("m"))
    assert router.breakers["primary"].state == "closed"


async def test_all_rate_limited_maps_to_429():
    router, _ = make_router({"m": ["primary/fail-429", "backup/fail-429"]}, retries=0)
    with pytest.raises(AllTargetsFailedError) as ei:
        await router.complete(req("m"))
    assert ei.value.status_code == 429


async def test_mixed_failures_map_to_502():
    router, _ = make_router({"m": ["primary/fail-500", "backup/fail-429"]}, retries=0)
    with pytest.raises(AllTargetsFailedError) as ei:
        await router.complete(req("m"))
    assert ei.value.status_code == 502
    assert "primary/fail-500:unavailable" in str(ei.value)


async def test_breaker_opens_and_skips_provider():
    router, _ = make_router(
        {"m": ["primary/fail-500", "backup/ok"]},
        retries=0,
        breaker={"failure_threshold": 2, "cooldown_s": 60},
    )
    for _ in range(2):
        await router.complete(req("m"))
    result = await router.complete(req("m"))
    assert [a.outcome for a in result.attempts] == ["circuit_open", "ok"]
    calls = router.providers["primary"].calls["fail-500"]
    assert calls == 2  # third request never reached the open provider


async def test_all_circuits_open_is_503():
    router, _ = make_router(
        {"m": ["primary/fail-500"]}, retries=0, breaker={"failure_threshold": 1}
    )
    with pytest.raises(AllTargetsFailedError):
        await router.complete(req("m"))
    with pytest.raises(AllTargetsFailedError) as ei:
        await router.complete(req("m"))
    assert ei.value.status_code == 503


async def test_stream_fallback_before_first_chunk():
    router, _ = make_router({"m": ["primary/fail-500", "backup/ok"]}, retries=0)
    route = await router.open_stream(req("m"))
    assert route.target.id == "backup/ok"
    text = "".join([d.content or "" async for d in route.deltas])
    assert "You said: hi" in text


async def test_unknown_model_and_disabled_provider(monkeypatch):
    router, _ = make_router({"m": ["primary/ok"]})
    with pytest.raises(NoRouteError):
        router.targets("unknown")
    with pytest.raises(NoRouteError):
        router.targets("ghost/model")


def test_provider_disabled_when_key_env_missing(monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)
    cfg = GatewayConfig.model_validate(
        {"providers": {"p": {"type": "fake", "api_key_env": "SOME_KEY"}, "q": {"type": "fake"}}}
    )
    assert not cfg.provider_enabled("p")
    assert cfg.provider_enabled("q")
    monkeypatch.setenv("SOME_KEY", "x")
    assert cfg.provider_enabled("p")


def test_env_expansion(monkeypatch):
    monkeypatch.setenv("A", "1")
    monkeypatch.delenv("B", raising=False)
    assert expand_env({"x": ["${A}", "${B:-two}", "plain"]}) == {"x": ["1", "two", "plain"]}


def test_target_parse():
    assert Target.parse("openai/gpt-4.1-mini").model == "gpt-4.1-mini"
    with pytest.raises(ValueError):
        Target.parse("nomodel")
