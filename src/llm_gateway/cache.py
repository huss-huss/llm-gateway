"""Exact-match response cache in Redis.

Only deterministic requests are cached: ``temperature`` must be explicitly 0. The key is a
SHA-256 over the tenant and a normalized request (sorted JSON, transport-only fields removed,
text-only content parts collapsed to strings, ``max_tokens``/``max_completion_tokens`` unified).
Entries are scoped per tenant so one customer can never read another's cached completions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis

from llm_gateway.schemas import ChatCompletionRequest, ProviderResponse, Usage

KEY_VERSION = "v1"
_TRANSPORT_FIELDS = {"stream", "stream_options", "user", "max_completion_tokens"}
CACHEABLE_FINISH = {"stop", "tool_calls", "length"}


def is_cacheable(req: ChatCompletionRequest) -> bool:
    return req.temperature == 0


def _normalize_content(content: Any) -> Any:
    if isinstance(content, list) and all(
        isinstance(p, dict) and p.get("type") == "text" for p in content
    ):
        return "".join(p.get("text", "") for p in content)
    return content


def normalized_request(req: ChatCompletionRequest) -> dict[str, Any]:
    body = req.model_dump(exclude_none=True, exclude=_TRANSPORT_FIELDS)
    max_tokens = req.max_completion_tokens or req.max_tokens
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    for m in body["messages"]:
        if "content" in m:
            m["content"] = _normalize_content(m["content"])
    if isinstance(body.get("stop"), str):
        body["stop"] = [body["stop"]]
    return body


def cache_key(tenant: str, req: ChatCompletionRequest) -> str:
    canonical = json.dumps(
        normalized_request(req), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    digest = hashlib.sha256(f"{tenant}\x00{canonical}".encode()).hexdigest()
    return f"cache:{KEY_VERSION}:{digest}"


@dataclass
class CachedResponse:
    response: ProviderResponse
    provider: str
    model: str


class ResponseCache:
    def __init__(self, redis: Redis, ttl_s: int, enabled: bool = True) -> None:
        self.redis = redis
        self.ttl_s = ttl_s
        self.enabled = enabled

    async def get(self, key: str) -> CachedResponse | None:
        raw = await self.redis.get(key)
        if raw is None:
            return None
        d = json.loads(raw)
        resp = ProviderResponse(
            content=d["content"],
            finish_reason=d["finish_reason"],
            tool_calls=d["tool_calls"],
            usage=Usage(**d["usage"]),
        )
        return CachedResponse(resp, d["provider"], d["model"])

    async def set(self, key: str, resp: ProviderResponse, provider: str, model: str) -> bool:
        if resp.finish_reason not in CACHEABLE_FINISH:
            return False
        payload = {
            "content": resp.content,
            "finish_reason": resp.finish_reason,
            "tool_calls": resp.tool_calls,
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "cached_prompt_tokens": resp.usage.cached_prompt_tokens,
            },
            "provider": provider,
            "model": model,
        }
        await self.redis.set(key, json.dumps(payload), ex=self.ttl_s)
        return True
