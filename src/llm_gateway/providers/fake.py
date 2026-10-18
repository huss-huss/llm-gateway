"""A deterministic provider for tests, demos, and evals. No network, no keys.

Behaviour is selected by the *model name*:

- any normal name (e.g. ``fake-fast``): reply from the first matching rule, else an echo
- ``fail-429`` / ``fail-500`` / ``fail-auth`` / ``fail-400``: raise the matching error
- ``slow``: sleep ``slow_s`` seconds before replying (exercises timeouts)
- ``flaky``: fail with a 503 for the first ``flaky_failures`` calls, then succeed
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from typing import Any

from llm_gateway.providers.base import (
    InvalidRequestError,
    Provider,
    ProviderAuthError,
    ProviderUnavailableError,
    RateLimitedError,
    estimate_tokens,
)
from llm_gateway.schemas import ChatCompletionRequest, ProviderResponse, StreamDelta, Usage

DEFAULT_REPLY = "This is a deterministic reply from the fake provider. You said: {input}"


class FakeProvider(Provider):
    def __init__(self, name: str, config: dict[str, Any]) -> None:
        super().__init__(name, config)
        self.rules: list[dict[str, Any]] = list(config.get("rules", []))
        self.default_reply: str = config.get("default_reply", DEFAULT_REPLY)
        self.latency_s: float = float(config.get("latency_ms", 0)) / 1000
        self.slow_s: float = float(config.get("slow_s", 5))
        self.flaky_failures: int = int(config.get("flaky_failures", 1))
        self.calls: dict[str, int] = {}

    async def _behave(self, model: str) -> None:
        self.calls[model] = self.calls.get(model, 0) + 1
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        if model == "fail-429":
            raise RateLimitedError("fake: rate limited", retry_after=0)
        if model == "fail-500":
            raise ProviderUnavailableError("fake: internal error")
        if model == "fail-auth":
            raise ProviderAuthError("fake: invalid api key")
        if model == "fail-400":
            raise InvalidRequestError("fake: invalid request")
        if model == "slow":
            await asyncio.sleep(self.slow_s)
        if model == "flaky" and self.calls[model] <= self.flaky_failures:
            raise ProviderUnavailableError("fake: transient failure")

    def _respond(self, req: ChatCompletionRequest) -> ProviderResponse:
        user_text = req.last_user_text()
        prompt_text = "\n".join(m.text() for m in req.messages)
        last_is_tool = req.messages[-1].role == "tool"
        for rule in self.rules:
            if not re.search(rule["match"], user_text, flags=re.IGNORECASE):
                continue
            if "tool_call" in rule and req.tools and not last_is_tool:
                call = rule["tool_call"]
                tool_calls = [
                    {
                        "id": "call_fake_0",
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call.get("arguments", {})),
                        },
                    }
                ]
                usage = Usage(estimate_tokens(prompt_text), estimate_tokens(json.dumps(call)))
                return ProviderResponse(None, "tool_calls", tool_calls, usage)
            if "reply" in rule:
                return self._text(rule["reply"], prompt_text, req)
        return self._text(self.default_reply.format(input=user_text), prompt_text, req)

    @staticmethod
    def _text(reply: str, prompt_text: str, req: ChatCompletionRequest) -> ProviderResponse:
        finish = "stop"
        limit = req.max_completion_tokens or req.max_tokens
        if limit is not None and estimate_tokens(reply) > limit:
            reply = reply[: limit * 4]
            finish = "length"
        return ProviderResponse(
            content=reply,
            finish_reason=finish,
            usage=Usage(estimate_tokens(prompt_text), estimate_tokens(reply)),
        )

    async def chat(self, req: ChatCompletionRequest, model: str) -> ProviderResponse:
        await self._behave(model)
        return self._respond(req)

    async def stream(self, req: ChatCompletionRequest, model: str) -> AsyncIterator[StreamDelta]:
        await self._behave(model)
        resp = self._respond(req)
        if resp.tool_calls:
            deltas = [dict(tc, index=i) for i, tc in enumerate(resp.tool_calls)]
            yield StreamDelta(tool_calls=deltas)
        else:
            # Split on spaces but keep them, so concatenating deltas reproduces the text.
            for piece in re.findall(r"\S+\s*|\s+", resp.content or ""):
                yield StreamDelta(content=piece)
        yield StreamDelta(finish_reason=resp.finish_reason, usage=resp.usage)
