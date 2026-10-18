from __future__ import annotations


async def test_healthz(client):
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_chat_completion_openai_shape(client):
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "fast", "messages": [{"role": "user", "content": "capital of France?"}]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["model"] == "fast"
    choice = body["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "The capital of France is Paris."}
    assert choice["finish_reason"] == "stop"
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert r.headers["x-gateway-provider"] == "fake"
    assert r.headers["x-gateway-model"] == "fake-fast"


async def test_direct_provider_model(client):
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "fake/anything", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert "You said: hi" in r.json()["choices"][0]["message"]["content"]


async def test_tool_call_response(client):
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": "fast",
            "tools": tools,
            "messages": [{"role": "user", "content": "weather in Paris?"}],
        },
    )
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


async def test_unknown_model_is_404(client):
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


async def test_validation_error_is_openai_style(client):
    r = await client.post("/v1/chat/completions", json={"model": "fast", "messages": []})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


async def test_list_models(client):
    r = await client.get("/v1/models")
    ids = [m["id"] for m in r.json()["data"]]
    assert {"fast", "smart", "fake/fake-fast"} <= set(ids)


async def test_fallback_is_visible_in_headers(client):
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "resilient", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.headers["x-gateway-model"] == "fake-ok"
    assert r.headers["x-gateway-attempts"] == "3"  # 2 tries on fail-500, then fallback
