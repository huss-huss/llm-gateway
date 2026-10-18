"""Anthropic <-> OpenAI translation, exercised through the real SDK with a mock transport."""

from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx2
import pytest

from llm_gateway.catalog import Catalog
from llm_gateway.providers.anthropic import (
    SERVER_FALLBACK_BETA,
    AnthropicProvider,
    build_params,
    convert_tool_choice,
)
from llm_gateway.providers.base import InvalidRequestError, RateLimitedError
from llm_gateway.schemas import ChatCompletionRequest

CATALOG = Catalog.model_validate(
    {
        "models": {
            "anthropic/claude-opus-5-5": {
                "input": 4,
                "output": 20,
                "capabilities": {
                    "sampling": False,
                    "forced_tool_choice": False,
                    "server_fallback": True,
                },
            },
            "anthropic/claude-haiku-5-5": {
                "input": 0.1,
                "output": 0.5,
                "capabilities": {"sampling": False, "server_fallback": False},
            },
        }
    }
)


def make_req(**kw: Any) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate({"model": "smart", **kw})


def test_system_tools_and_tool_results_translation():
    req = make_req(
        messages=[
            {"role": "system", "content": "Be terse."},
            {"role": "developer", "content": "Use metric units."},
            {"role": "user", "content": "Weather in Paris and Rome?"},
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
            {"role": "tool", "tool_call_id": "call_2", "content": "24C"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=False,
        stop="END",
        max_tokens=300,
        temperature=0.2,
    )
    p = build_params(req, "claude-opus-5-5", default_max_tokens=1024)
    assert p["system"] == "Be terse.\n\nUse metric units."
    assert p["max_tokens"] == 300
    assert p["stop_sequences"] == ["END"]
    assert "temperature" not in p and "extra_body" not in p  # sampling disabled
    assert p["tools"] == [
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    assert p["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    msgs = p["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Paris"}},
        {"type": "tool_use", "id": "call_2", "name": "get_weather", "input": {"city": "Rome"}},
    ]
    # Both tool results land in ONE user message.
    assert msgs[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "18C"},
        {"type": "tool_result", "tool_use_id": "call_2", "content": "24C"},
    ]


def test_images_and_default_max_tokens():
    req = make_req(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
                ],
            }
        ]
    )
    p = build_params(req, "m", default_max_tokens=777)
    assert p["max_tokens"] == 777
    blocks = p["messages"][0]["content"]
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }
    assert blocks[2] == {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/cat.jpg"},
    }


def test_json_schema_maps_to_output_config():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    req = make_req(
        messages=[{"role": "user", "content": "x"}],
        response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": schema}},
    )
    p = build_params(req, "m", default_max_tokens=10)
    assert p["output_config"] == {"format": {"type": "json_schema", "schema": schema}}


def test_sampling_passthrough_when_supported():
    req = make_req(messages=[{"role": "user", "content": "x"}], temperature=0)
    p = build_params(req, "m", default_max_tokens=10, sampling=True)
    assert p["extra_body"] == {"temperature": 0}


def test_tool_choice_mapping():
    assert convert_tool_choice("required", None, True) == {"type": "any"}
    assert convert_tool_choice("required", None, False) == {"type": "auto"}
    named = {"type": "function", "function": {"name": "f"}}
    assert convert_tool_choice(named, None, True) == {"type": "tool", "name": "f"}
    assert convert_tool_choice("none", False, True) == {"type": "none"}
    assert convert_tool_choice(None, None, True) is None
    with pytest.raises(InvalidRequestError):
        convert_tool_choice("bogus", None, True)


def test_invalid_tool_arguments_rejected():
    req = make_req(
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{not json"},
                    }
                ],
            }
        ]
    )
    with pytest.raises(InvalidRequestError):
        build_params(req, "m", default_max_tokens=10)


# ---------------------------------------------------------------------- SDK round-trips


class Recorder:
    def __init__(self, responder):
        self.requests: list[httpx2.Request] = []
        self.responder = responder

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.responder(request)


def make_provider(responder) -> tuple[AnthropicProvider, Recorder]:
    rec = Recorder(responder)
    client = anthropic.AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(rec)),
    )
    p = AnthropicProvider("anthropic", {"default_max_tokens": 512}, client=client)
    p.catalog = CATALOG
    return p, rec


MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 20, "output_tokens": 9, "cache_read_input_tokens": 5},
}


async def test_non_streaming_roundtrip():
    provider, rec = make_provider(lambda r: httpx2.Response(200, json=MESSAGE))
    req = make_req(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "weather?"}]
    )
    resp = await provider.chat(req, "claude-haiku-5-5")
    assert resp.content == "Let me check."
    assert resp.finish_reason == "tool_calls"
    assert resp.tool_calls == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
    ]
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens) == (25, 9)
    assert resp.usage.cached_prompt_tokens == 5
    sent = json.loads(rec.requests[0].content)
    assert rec.requests[0].url.path == "/v1/messages"
    assert sent["system"] == "sys" and sent["max_tokens"] == 512
    assert "fallbacks" not in sent  # haiku: no server-side fallback


async def test_server_fallback_uses_beta_endpoint():
    provider, rec = make_provider(
        lambda r: httpx2.Response(200, json={**MESSAGE, "stop_reason": "refusal"})
    )
    resp = await provider.chat(
        make_req(messages=[{"role": "user", "content": "x"}]), "claude-opus-5-5"
    )
    assert resp.finish_reason == "content_filter"
    req = rec.requests[0]
    assert json.loads(req.content)["fallbacks"] == "default"
    assert SERVER_FALLBACK_BETA in req.headers["anthropic-beta"]


async def test_rate_limit_maps_to_retryable_error():
    def limited(_):
        return httpx2.Response(
            429,
            headers={"retry-after": "3"},
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
        )

    provider, _ = make_provider(limited)
    with pytest.raises(RateLimitedError) as ei:
        await provider.chat(
            make_req(messages=[{"role": "user", "content": "x"}]), "claude-haiku-5-5"
        )
    assert ei.value.retry_after == 3.0


def sse_body(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


STREAM_EVENTS = [
    {
        "type": "message_start",
        "message": {
            **MESSAGE,
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": 12, "output_tokens": 1},
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "tool_use", "id": "toolu_9", "name": "lookup", "input": {}},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": '{"q": '},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": '"x"}'},
    },
    {"type": "content_block_stop", "index": 1},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": 15},
    },
    {"type": "message_stop"},
]


async def test_streaming_translation():
    provider, _ = make_provider(
        lambda r: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse_body(STREAM_EVENTS)
        )
    )
    deltas = [
        d
        async for d in provider.stream(
            make_req(messages=[{"role": "user", "content": "x"}]), "claude-haiku-5-5"
        )
    ]
    assert "".join(d.content or "" for d in deltas) == "Hello"
    tool_parts = [tc for d in deltas for tc in (d.tool_calls or [])]
    assert tool_parts[0] == {
        "index": 0,
        "id": "toolu_9",
        "type": "function",
        "function": {"name": "lookup", "arguments": ""},
    }
    assert "".join(tc["function"]["arguments"] for tc in tool_parts) == '{"q": "x"}'
    last = deltas[-1]
    assert last.finish_reason == "tool_calls"
    assert (last.usage.prompt_tokens, last.usage.completion_tokens) == (12, 15)
