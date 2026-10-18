from __future__ import annotations

import json


def parse_sse(text: str) -> list:
    events = []
    for block in text.strip().split("\n\n"):
        assert block.startswith("data: "), block
        payload = block[len("data: ") :]
        events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


async def test_stream_format(client):
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": "fast",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "capital of France?"}],
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert len({c["id"] for c in chunks}) == 1
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "The capital of France is Paris."
    finish = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert finish[-1]["choices"][0]["finish_reason"] == "stop"
    # include_usage: a final chunk with empty choices and a usage object
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["completion_tokens"] > 0


async def test_stream_without_usage_chunk(client):
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "fast", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    chunks = parse_sse(r.text)[:-1]
    assert all("usage" not in c for c in chunks)


async def test_stream_tool_calls(client):
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": "fast",
            "stream": True,
            "tools": tools,
            "messages": [{"role": "user", "content": "weather?"}],
        },
    )
    chunks = parse_sse(r.text)[:-1]
    calls = [
        c["choices"][0]["delta"]["tool_calls"]
        for c in chunks
        if c["choices"] and "tool_calls" in c["choices"][0]["delta"]
    ]
    assert calls[0][0]["index"] == 0
    assert calls[0][0]["function"]["name"] == "get_weather"
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


async def test_stream_error_before_first_chunk_is_http_error(client):
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": "fake/fail-400",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"
