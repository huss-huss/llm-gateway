"""OpenAI Chat Completions adapter (also works with any OpenAI-compatible server via base_url)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from llm_gateway.providers._http import api_key_from, http_status_error, transport_error
from llm_gateway.providers.base import Provider
from llm_gateway.schemas import ChatCompletionRequest, ProviderResponse, StreamDelta, Usage

# Fields the gateway owns or that must not be forwarded verbatim.
_GATEWAY_FIELDS = {"model", "stream", "stream_options"}


def _usage(raw: dict[str, Any] | None) -> Usage:
    raw = raw or {}
    cached = (raw.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return Usage(raw.get("prompt_tokens", 0), raw.get("completion_tokens", 0), cached)


class OpenAIProvider(Provider):
    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(name, config)
        headers = {}
        key = api_key_from(config, "OPENAI_API_KEY")
        if key:
            headers["authorization"] = f"Bearer {key}"
        if config.get("organization"):
            headers["openai-organization"] = config["organization"]
        self.client = httpx.AsyncClient(
            base_url=config.get("base_url", "https://api.openai.com/v1"),
            headers=headers,
            timeout=httpx.Timeout(float(config.get("timeout_s", 60)), connect=5.0),
            transport=transport,
        )

    def build_body(self, req: ChatCompletionRequest, model: str, stream: bool) -> dict[str, Any]:
        body = req.model_dump(exclude_none=True, exclude=_GATEWAY_FIELDS)
        body["model"] = model
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body

    async def chat(self, req: ChatCompletionRequest, model: str) -> ProviderResponse:
        try:
            resp = await self.client.post(
                "/chat/completions", json=self.build_body(req, model, False)
            )
        except httpx.HTTPError as exc:
            raise transport_error(exc, self.name) from exc
        if resp.status_code >= 400:
            raise http_status_error(resp, self.name)
        data = resp.json()
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        return ProviderResponse(
            content=msg.get("content"),
            finish_reason=choice.get("finish_reason") or "stop",
            tool_calls=msg.get("tool_calls") or [],
            usage=_usage(data.get("usage")),
        )

    async def stream(self, req: ChatCompletionRequest, model: str) -> AsyncIterator[StreamDelta]:
        body = self.build_body(req, model, True)
        try:
            async with self.client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise http_status_error(resp, self.name)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    for choice in obj.get("choices") or []:
                        delta = choice.get("delta") or {}
                        yield StreamDelta(
                            content=delta.get("content") or None,
                            tool_calls=delta.get("tool_calls"),
                            finish_reason=choice.get("finish_reason"),
                        )
                    if obj.get("usage"):
                        yield StreamDelta(usage=_usage(obj["usage"]))
        except httpx.HTTPError as exc:
            raise transport_error(exc, self.name) from exc

    async def aclose(self) -> None:
        await self.client.aclose()
