"""Anthropic Messages API adapter, using the official ``anthropic`` SDK.

Translation (OpenAI chat -> Anthropic Messages):

- ``system``/``developer`` messages -> top-level ``system``
- ``assistant.tool_calls`` -> ``tool_use`` blocks; ``tool`` messages -> ``tool_result`` blocks in
  a single ``user`` turn (parallel results must share one message)
- ``tools[].function`` -> ``{name, description, input_schema}``
- ``tool_choice`` auto/none/required/named -> auto/none/any/tool (forced choices are downgraded
  to auto on models whose catalog entry says ``forced_tool_choice: false``)
- ``response_format: json_schema`` -> ``output_config.format``
- ``max_tokens`` defaults from config because Anthropic requires it

The gateway owns retries and fallback, so the SDK client is built with ``max_retries=0``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from llm_gateway.providers._http import api_key_from
from llm_gateway.providers.base import (
    InvalidRequestError,
    Provider,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    error_from_status,
    parse_retry_after,
)
from llm_gateway.schemas import ChatCompletionRequest, ProviderResponse, StreamDelta, Usage

SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"

STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


# ------------------------------------------------------------------------------------------
# Request translation
# ------------------------------------------------------------------------------------------


def _image_block(url: str) -> dict[str, Any]:
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = header[5:].split(";", 1)[0] or "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def content_blocks(content: str | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for part in content:
        ptype = part.get("type")
        if ptype == "text" and part.get("text"):
            blocks.append({"type": "text", "text": part["text"]})
        elif ptype == "image_url":
            image = part.get("image_url") or {}
            blocks.append(_image_block(image.get("url", "") if isinstance(image, dict) else image))
    return blocks


def convert_messages(req: ChatCompletionRequest) -> tuple[str | None, list[dict[str, Any]]]:
    """Return ``(system, messages)`` in Anthropic shape, merging consecutive same-role turns."""
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for m in req.messages:
        if m.role in ("system", "developer"):
            if m.text():
                system_parts.append(m.text())
            continue
        if m.role == "tool":
            role = "user"
            blocks = [
                {"type": "tool_result", "tool_use_id": m.tool_call_id or "", "content": m.text()}
            ]
        elif m.role == "assistant":
            role = "assistant"
            blocks = content_blocks(m.content)
            for tc in m.tool_calls or []:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    raise InvalidRequestError(
                        f"tool call {tc.id} has invalid JSON arguments"
                    ) from exc
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": args}
                )
        else:
            role = "user"
            blocks = content_blocks(m.content)
        if not blocks:
            continue
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})
    return ("\n\n".join(system_parts) or None), out


def convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    out = []
    for t in tools:
        fn = t.get("function", t)
        tool: dict[str, Any] = {
            "name": fn["name"],
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        }
        if fn.get("description"):
            tool["description"] = fn["description"]
        out.append(tool)
    return out


def convert_tool_choice(
    choice: str | dict[str, Any] | None, parallel: bool | None, forced_allowed: bool
) -> dict[str, Any] | None:
    if choice is None and parallel is None:
        return None
    if choice is None or choice == "auto":
        tc: dict[str, Any] = {"type": "auto"}
    elif choice == "none":
        return {"type": "none"}
    elif choice == "required":
        tc = {"type": "any"} if forced_allowed else {"type": "auto"}
    elif isinstance(choice, dict) and choice.get("type") == "function":
        name = (choice.get("function") or {}).get("name")
        tc = {"type": "tool", "name": name} if forced_allowed else {"type": "auto"}
    else:
        raise InvalidRequestError(f"unsupported tool_choice: {choice!r}")
    if parallel is False:
        tc["disable_parallel_tool_use"] = True
    return tc


def build_params(
    req: ChatCompletionRequest,
    model: str,
    *,
    default_max_tokens: int,
    sampling: bool = False,
    forced_tool_choice: bool = True,
) -> dict[str, Any]:
    system, messages = convert_messages(req)
    if not messages:
        raise InvalidRequestError("at least one user or assistant message is required")
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": req.effective_max_tokens(default_max_tokens),
        "messages": messages,
    }
    rf = req.response_format or {}
    if rf.get("type") == "json_object":
        hint = "Respond with a single valid JSON object and nothing else."
        system = f"{system}\n\n{hint}" if system else hint
    elif rf.get("type") == "json_schema":
        schema = (rf.get("json_schema") or {}).get("schema")
        if schema:
            params["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    if system:
        params["system"] = system
    if req.stop_list():
        params["stop_sequences"] = req.stop_list()
    tools = convert_tools(req.tools)
    if tools:
        params["tools"] = tools
    tool_choice = convert_tool_choice(req.tool_choice, req.parallel_tool_calls, forced_tool_choice)
    if tool_choice and tools:
        params["tool_choice"] = tool_choice
    if req.user:
        params["metadata"] = {"user_id": req.user}
    if sampling:
        # Older models accept sampling params; the current SDK no longer models them.
        extra = {
            k: v
            for k, v in (("temperature", req.temperature), ("top_p", req.top_p))
            if v is not None
        }
        if extra:
            params["extra_body"] = extra
    return params


# ------------------------------------------------------------------------------------------
# Response translation
# ------------------------------------------------------------------------------------------


def _usage(u: Any, output_tokens: int | None = None) -> Usage:
    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
    prompt = (getattr(u, "input_tokens", 0) or 0) + cache_read + cache_write
    out = output_tokens if output_tokens is not None else (getattr(u, "output_tokens", 0) or 0)
    return Usage(prompt, out, cache_read)


def from_message(msg: Any) -> ProviderResponse:
    texts: list[str] = []
    calls: list[dict[str, Any]] = []
    for block in msg.content:
        if block.type == "text":
            texts.append(block.text)
        elif block.type == "tool_use":
            calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": json.dumps(block.input)},
                }
            )
        # thinking / fallback / server-tool blocks have no OpenAI equivalent and are dropped.
    return ProviderResponse(
        content="".join(texts) if texts or not calls else None,
        finish_reason=STOP_REASONS.get(msg.stop_reason or "end_turn", "stop"),
        tool_calls=calls,
        usage=_usage(msg.usage),
    )


class StreamTranslator:
    """Maps raw Messages API stream events to OpenAI-style deltas."""

    def __init__(self) -> None:
        self.tool_index: dict[int, int] = {}  # anthropic block index -> openai tool_call index
        self.prompt_usage = Usage()
        self.finish = "stop"

    def feed(self, event: Any) -> list[StreamDelta]:
        etype = event.type
        if etype == "message_start":
            self.prompt_usage = _usage(event.message.usage, output_tokens=0)
        elif etype == "content_block_start":
            block = event.content_block
            if block.type == "tool_use":
                idx = len(self.tool_index)
                self.tool_index[event.index] = idx
                return [
                    StreamDelta(
                        tool_calls=[
                            {
                                "index": idx,
                                "id": block.id,
                                "type": "function",
                                "function": {"name": block.name, "arguments": ""},
                            }
                        ]
                    )
                ]
        elif etype == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta" and delta.text:
                return [StreamDelta(content=delta.text)]
            if delta.type == "input_json_delta" and event.index in self.tool_index:
                return [
                    StreamDelta(
                        tool_calls=[
                            {
                                "index": self.tool_index[event.index],
                                "function": {"arguments": delta.partial_json},
                            }
                        ]
                    )
                ]
        elif etype == "message_delta":
            if event.delta.stop_reason:
                self.finish = STOP_REASONS.get(event.delta.stop_reason, "stop")
            out_tokens = getattr(event.usage, "output_tokens", 0) or 0
            self.prompt_usage.completion_tokens = out_tokens
        elif etype == "message_stop":
            return [StreamDelta(finish_reason=self.finish, usage=self.prompt_usage)]
        return []


# ------------------------------------------------------------------------------------------
# Provider
# ------------------------------------------------------------------------------------------


def map_sdk_error(exc: Exception, provider: str) -> ProviderError:
    if isinstance(exc, anthropic.APITimeoutError):
        return ProviderTimeoutError(f"{provider}: upstream timeout")
    if isinstance(exc, anthropic.APIConnectionError):
        return ProviderUnavailableError(f"{provider}: connection error")
    if isinstance(exc, anthropic.APIStatusError):
        retry_after = parse_retry_after(exc.response.headers.get("retry-after"))
        return error_from_status(
            exc.status_code, f"{provider}: HTTP {exc.status_code}: {exc.message}", retry_after
        )
    return ProviderUnavailableError(f"{provider}: {type(exc).__name__}: {exc}")


class AnthropicProvider(Provider):
    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        super().__init__(name, config)
        self.default_max_tokens = int(config.get("default_max_tokens", 1024))
        self.client = client or anthropic.AsyncAnthropic(
            api_key=api_key_from(config, "ANTHROPIC_API_KEY"),
            base_url=config.get("base_url"),
            timeout=float(config.get("timeout_s", 60)),
            max_retries=0,
        )

    def params(self, req: ChatCompletionRequest, model: str) -> dict[str, Any]:
        return build_params(
            req,
            model,
            default_max_tokens=self.default_max_tokens,
            sampling=bool(self.capability(model, "sampling", False)),
            forced_tool_choice=bool(self.capability(model, "forced_tool_choice", True)),
        )

    def _create(self, model: str):
        """Pick the endpoint: the beta one when server-side refusal fallback is enabled."""
        if self.capability(model, "server_fallback", False) and self.config.get(
            "server_fallback", True
        ):
            create = self.client.beta.messages.create

            async def call(**params: Any) -> Any:
                return await create(**params, betas=[SERVER_FALLBACK_BETA], fallbacks="default")

            return call
        return self.client.messages.create

    async def chat(self, req: ChatCompletionRequest, model: str) -> ProviderResponse:
        params = self.params(req, model)
        try:
            msg = await self._create(model)(**params)
        except anthropic.AnthropicError as exc:
            raise map_sdk_error(exc, self.name) from exc
        return from_message(msg)

    async def stream(self, req: ChatCompletionRequest, model: str) -> AsyncIterator[StreamDelta]:
        params = self.params(req, model)
        translator = StreamTranslator()
        try:
            events = await self._create(model)(**params, stream=True)
            async for event in events:
                for delta in translator.feed(event):
                    yield delta
        except anthropic.AnthropicError as exc:
            raise map_sdk_error(exc, self.name) from exc

    async def aclose(self) -> None:
        await self.client.close()
