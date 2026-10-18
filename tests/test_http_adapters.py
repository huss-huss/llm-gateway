"""OpenAI and Ollama adapters against mocked HTTP transports."""

from __future__ import annotations

import json

import httpx
import pytest

from llm_gateway.providers.base import (
    InvalidRequestError,
    ProviderAuthError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitedError,
)
from llm_gateway.providers.ollama import OllamaProvider
from llm_gateway.providers.openai import OpenAIProvider
from llm_gateway.schemas import ChatCompletionRequest


def req(**kw) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {"model": "fast", "messages": [{"role": "user", "content": "hi"}], **kw}
    )


def openai_with(handler) -> tuple[OpenAIProvider, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def wrapped(r: httpx.Request) -> httpx.Response:
        seen.append(r)
        return handler(r)

    return OpenAIProvider(
        "openai",
        {"api_key": "sk-test", "api_key_env": None},
        transport=httpx.MockTransport(wrapped),
    ), seen


async def test_openai_chat():
    body = {
        "choices": [
            {"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 3},
        },
    }
    p, seen = openai_with(lambda r: httpx.Response(200, json=body))
    resp = await p.chat(req(temperature=0, max_tokens=5), "gpt-4.1-mini")
    assert resp.content == "hello"
    assert (resp.usage.prompt_tokens, resp.usage.cached_prompt_tokens) == (7, 3)
    sent = json.loads(seen[0].content)
    assert sent["model"] == "gpt-4.1-mini" and sent["temperature"] == 0
    assert "stream" not in sent
    assert seen[0].headers["authorization"] == "Bearer sk-test"


async def test_openai_stream():
    lines = [
        {"choices": [{"delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " there"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2}},
    ]
    sse = "".join(f"data: {json.dumps(x)}\n\n" for x in lines) + "data: [DONE]\n\n"
    p, seen = openai_with(lambda r: httpx.Response(200, content=sse.encode()))
    deltas = [d async for d in p.stream(req(), "gpt-4.1-mini")]
    assert "".join(d.content or "" for d in deltas) == "Hi there"
    assert any(d.finish_reason == "stop" for d in deltas)
    assert deltas[-1].usage.completion_tokens == 2
    sent = json.loads(seen[0].content)
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize(
    ("status", "exc"),
    [
        (429, RateLimitedError),
        (500, ProviderUnavailableError),
        (401, ProviderAuthError),
        (400, InvalidRequestError),
    ],
)
async def test_openai_error_mapping(status, exc):
    p, _ = openai_with(lambda r: httpx.Response(status, json={"error": {"message": "boom"}}))
    with pytest.raises(exc, match="boom"):
        await p.chat(req(), "gpt-4.1-mini")


async def test_openai_timeout_mapping():
    def timeout(_):
        raise httpx.ReadTimeout("slow")

    p, _ = openai_with(timeout)
    with pytest.raises(ProviderTimeoutError):
        await p.chat(req(), "gpt-4.1-mini")
    with pytest.raises(ProviderTimeoutError):
        _ = [d async for d in p.stream(req(), "gpt-4.1-mini")]


def ollama_with(handler) -> tuple[OllamaProvider, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def wrapped(r):
        seen.append(r)
        return handler(r)

    return OllamaProvider("ollama", {}, transport=httpx.MockTransport(wrapped)), seen


async def test_ollama_chat_and_options():
    body = {
        "message": {"role": "assistant", "content": "local hi"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 11,
        "eval_count": 3,
    }
    p, seen = ollama_with(lambda r: httpx.Response(200, json=body))
    resp = await p.chat(
        req(temperature=0, max_tokens=50, stop=["\n"], response_format={"type": "json_object"}),
        "llama3.2",
    )
    assert resp.content == "local hi"
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens) == (11, 3)
    sent = json.loads(seen[0].content)
    assert seen[0].url.path == "/api/chat"
    assert sent["options"] == {"temperature": 0, "num_predict": 50, "stop": ["\n"]}
    assert sent["format"] == "json" and sent["stream"] is False


async def test_ollama_tool_calls():
    body = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Rome"}}}],
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 5,
        "eval_count": 5,
    }
    p, _ = ollama_with(lambda r: httpx.Response(200, json=body))
    resp = await p.chat(req(), "llama3.2")
    assert resp.finish_reason == "tool_calls"
    assert json.loads(resp.tool_calls[0]["function"]["arguments"]) == {"city": "Rome"}


async def test_ollama_stream():
    lines = [
        {"message": {"role": "assistant", "content": "a"}, "done": False},
        {"message": {"role": "assistant", "content": "b"}, "done": False},
        {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 2,
            "eval_count": 2,
        },
    ]
    nd = "\n".join(json.dumps(x) for x in lines) + "\n"
    p, _ = ollama_with(lambda r: httpx.Response(200, content=nd.encode()))
    deltas = [d async for d in p.stream(req(), "llama3.2")]
    assert "".join(d.content or "" for d in deltas) == "ab"
    assert deltas[-1].finish_reason == "length"


async def test_ollama_connection_error():
    def down(_):
        raise httpx.ConnectError("refused")

    p, _ = ollama_with(down)
    with pytest.raises(ProviderUnavailableError):
        await p.chat(req(), "llama3.2")
