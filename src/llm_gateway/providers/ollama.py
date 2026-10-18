"""Ollama adapter (local models) using the native ``/api/chat`` endpoint."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from llm_gateway.providers._http import http_status_error, transport_error
from llm_gateway.providers.base import Provider, ProviderUnavailableError
from llm_gateway.schemas import ChatCompletionRequest, ProviderResponse, StreamDelta, Usage


def to_ollama_messages(req: ChatCompletionRequest) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in req.messages:
        role = "system" if m.role == "developer" else m.role
        msg: dict[str, Any] = {"role": role, "content": m.text()}
        images = [
            p["image_url"]["url"].split(",", 1)[1]
            for p in (m.content if isinstance(m.content, list) else [])
            if p.get("type") == "image_url"
            and str(p.get("image_url", {}).get("url", "")).startswith("data:")
        ]
        if images:
            msg["images"] = images
        if m.tool_calls:
            msg["tool_calls"] = [
                {
                    "function": {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments or "{}"),
                    }
                }
                for tc in m.tool_calls
            ]
        out.append(msg)
    return out


def _tool_calls(msg: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": tc["function"]["name"],
                "arguments": json.dumps(tc["function"].get("arguments") or {}),
            },
        }
        for i, tc in enumerate(msg.get("tool_calls") or [])
    ]


def _finish(data: dict[str, Any], has_tools: bool) -> str:
    if has_tools:
        return "tool_calls"
    return "length" if data.get("done_reason") == "length" else "stop"


class OllamaProvider(Provider):
    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(name, config)
        self.client = httpx.AsyncClient(
            base_url=config.get("base_url", "http://localhost:11434"),
            timeout=httpx.Timeout(float(config.get("timeout_s", 120)), connect=5.0),
            transport=transport,
        )

    def build_body(self, req: ChatCompletionRequest, model: str, stream: bool) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if req.temperature is not None:
            options["temperature"] = req.temperature
        if req.top_p is not None:
            options["top_p"] = req.top_p
        if req.max_completion_tokens or req.max_tokens:
            options["num_predict"] = req.max_completion_tokens or req.max_tokens
        if req.stop_list():
            options["stop"] = req.stop_list()
        if req.seed is not None:
            options["seed"] = req.seed
        body: dict[str, Any] = {
            "model": model,
            "messages": to_ollama_messages(req),
            "stream": stream,
        }
        if options:
            body["options"] = options
        if req.tools:
            body["tools"] = req.tools
        rf = req.response_format or {}
        if rf.get("type") == "json_object":
            body["format"] = "json"
        elif rf.get("type") == "json_schema":
            body["format"] = (rf.get("json_schema") or {}).get("schema", "json")
        return body

    async def chat(self, req: ChatCompletionRequest, model: str) -> ProviderResponse:
        try:
            resp = await self.client.post("/api/chat", json=self.build_body(req, model, False))
        except httpx.HTTPError as exc:
            raise transport_error(exc, self.name) from exc
        if resp.status_code >= 400:
            raise http_status_error(resp, self.name)
        data = resp.json()
        msg = data.get("message") or {}
        calls = _tool_calls(msg)
        return ProviderResponse(
            content=msg.get("content") or (None if calls else ""),
            finish_reason=_finish(data, bool(calls)),
            tool_calls=calls,
            usage=Usage(data.get("prompt_eval_count", 0), data.get("eval_count", 0)),
        )

    async def stream(self, req: ChatCompletionRequest, model: str) -> AsyncIterator[StreamDelta]:
        body = self.build_body(req, model, True)
        n_calls = 0
        try:
            async with self.client.stream("POST", "/api/chat", json=body) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise http_status_error(resp, self.name)
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise ProviderUnavailableError(f"{self.name}: {data['error']}")
                    msg = data.get("message") or {}
                    calls = _tool_calls(msg)
                    for c in calls:
                        c["index"] = n_calls
                        c["id"] = f"call_{n_calls}"
                        n_calls += 1
                    if msg.get("content") or calls:
                        yield StreamDelta(
                            content=msg.get("content") or None, tool_calls=calls or None
                        )
                    if data.get("done"):
                        yield StreamDelta(
                            finish_reason=_finish(data, n_calls > 0),
                            usage=Usage(
                                data.get("prompt_eval_count", 0), data.get("eval_count", 0)
                            ),
                        )
        except httpx.HTTPError as exc:
            raise transport_error(exc, self.name) from exc

    async def aclose(self) -> None:
        await self.client.aclose()
