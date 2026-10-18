from __future__ import annotations

import json
import logging

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from llm_gateway.observability.logging import JSONFormatter
from llm_gateway.request_context import request_id_var

CHAT = {"model": "resilient", "messages": [{"role": "user", "content": "hi"}]}

_EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def tracer_provider():
    # The global provider can only be set once per process; the gateway's module-level
    # tracer is a proxy, so it picks this up.
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
    trace.set_tracer_provider(provider)
    yield


@pytest.fixture
def spans():
    _EXPORTER.clear()
    return _EXPORTER


async def test_request_and_provider_spans(client, spans):
    r = await client.post("/v1/chat/completions", json=CHAT, headers={"x-request-id": "trace-1"})
    assert r.status_code == 200
    finished = spans.get_finished_spans()
    root = next(s for s in finished if s.name == "chat.completions")
    calls = [s for s in finished if s.name == "provider.call"]
    assert root.attributes["llm_gateway.request_id"] == "trace-1"
    assert root.attributes["llm_gateway.provider"] == "fake"
    assert root.attributes["gen_ai.usage.output_tokens"] > 0
    # 2 failed attempts on fake/fail-500, then the fallback target.
    assert [c.attributes["llm_gateway.outcome"] for c in calls] == [
        "unavailable",
        "unavailable",
        "ok",
    ]
    assert all(c.parent.span_id == root.context.span_id for c in calls)
    assert calls[-1].attributes["gen_ai.request.model"] == "fake-ok"


async def test_stream_span_ends_after_stream(client, spans):
    r = await client.post("/v1/chat/completions", json={**CHAT, "stream": True})
    assert r.status_code == 200
    root = next(s for s in spans.get_finished_spans() if s.name == "chat.completions")
    assert root.attributes["llm_gateway.stream"] is True
    assert root.attributes["gen_ai.usage.output_tokens"] > 0
    calls = [s for s in spans.get_finished_spans() if s.name == "provider.call"]
    assert all(c.parent.span_id == root.context.span_id for c in calls)


async def test_prometheus_metrics(client):
    await client.post("/v1/chat/completions", json=CHAT)
    text = (await client.get("/metrics")).text
    assert 'llm_gateway_requests_total{model="resilient",status="ok",stream="false"}' in text
    assert (
        'llm_gateway_provider_attempts_total{model="fail-500",outcome="unavailable"'
        ',provider="fake"}' in text
    )
    assert "llm_gateway_tokens_total" in text
    assert "llm_gateway_cost_usd_total" in text
    assert "llm_gateway_request_duration_seconds_bucket" in text


async def test_readyz(client):
    r = await client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["checks"] == {"redis": "ok", "database": "ok"}


def test_json_log_format_includes_request_id_and_extras():
    token = request_id_var.set("req_abc")
    try:
        record = logging.makeLogRecord(
            {
                "name": "x",
                "levelname": "INFO",
                "msg": "hello",
                "provider": "fake",
                "color_message": "\x1b[36mhello\x1b[0m",
            }
        )
        out = json.loads(JSONFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert out["msg"] == "hello" and out["level"] == "info"
    assert out["request_id"] == "req_abc" and out["provider"] == "fake"
    assert "color_message" not in out
