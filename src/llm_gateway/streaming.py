"""Server-Sent Events encoding for OpenAI-compatible ``chat.completion.chunk`` streams."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from llm_gateway.schemas import ProviderResponse, StreamDelta, Usage

DONE = "data: [DONE]\n\n"


def sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"


def chunk(
    completion_id: str,
    model: str,
    created: int,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


@dataclass
class StreamAccumulator:
    """Rebuilds the full response from deltas (for the ledger, cache, and metrics)."""

    content: list[str] = field(default_factory=list)
    tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)

    def add(self, d: StreamDelta) -> None:
        if d.content:
            self.content.append(d.content)
        for tc in d.tool_calls or []:
            idx = tc.get("index", 0)
            cur = self.tool_calls.setdefault(
                idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            )
            if tc.get("id"):
                cur["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                cur["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                cur["function"]["arguments"] += fn["arguments"]
        if d.finish_reason:
            self.finish_reason = d.finish_reason
        if d.usage:
            self.usage = d.usage

    def response(self) -> ProviderResponse:
        text = "".join(self.content)
        return ProviderResponse(
            content=text if text or not self.tool_calls else None,
            finish_reason=self.finish_reason,
            tool_calls=[self.tool_calls[i] for i in sorted(self.tool_calls)],
            usage=self.usage,
        )


async def encode_stream(
    deltas: AsyncIterator[StreamDelta],
    *,
    completion_id: str,
    model: str,
    include_usage: bool,
    acc: StreamAccumulator,
) -> AsyncIterator[str]:
    """Turn provider deltas into SSE lines; ``acc`` collects the final response as a side effect."""
    created = int(time.time())
    yield sse(chunk(completion_id, model, created, {"role": "assistant", "content": ""}))
    async for d in deltas:
        acc.add(d)
        delta: dict[str, Any] = {}
        if d.content:
            delta["content"] = d.content
        if d.tool_calls:
            delta["tool_calls"] = d.tool_calls
        if delta:
            yield sse(chunk(completion_id, model, created, delta))
    yield sse(chunk(completion_id, model, created, {}, acc.finish_reason))
    if include_usage:
        final = chunk(completion_id, model, created, {})
        final["choices"] = []
        final["usage"] = acc.usage.to_openai()
        yield sse(final)


async def replay(resp: ProviderResponse) -> AsyncIterator[StreamDelta]:
    """Replay a complete response as a stream (used for cache hits)."""
    if resp.content:
        yield StreamDelta(content=resp.content)
    if resp.tool_calls:
        yield StreamDelta(tool_calls=[dict(tc, index=i) for i, tc in enumerate(resp.tool_calls)])
    yield StreamDelta(finish_reason=resp.finish_reason, usage=resp.usage)
