"""OpenAI-compatible request/response schemas plus the gateway's internal result types."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FunctionCall(BaseModel):
    name: str
    arguments: str = "{}"


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def text(self) -> str:
        """Concatenate the text parts of this message (ignores images)."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "".join(
            str(p.get("text", "")) for p in self.content if p.get("type") in ("text", None)
        )


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    user: str | None = None

    def effective_max_tokens(self, default: int) -> int:
        return self.max_completion_tokens or self.max_tokens or default

    def stop_list(self) -> list[str] | None:
        if self.stop is None:
            return None
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)

    def last_user_text(self) -> str:
        for m in reversed(self.messages):
            if m.role == "user":
                return m.text()
        return ""


# --------------------------------------------------------------------------------------
# Internal, provider-neutral result types
# --------------------------------------------------------------------------------------


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_openai(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.cached_prompt_tokens:
            out["prompt_tokens_details"] = {"cached_tokens": self.cached_prompt_tokens}
        return out


@dataclass
class ProviderResponse:
    """A complete (non-streaming) completion from one provider."""

    content: str | None
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


@dataclass
class StreamDelta:
    """One incremental piece of a streamed completion, already in OpenAI delta terms."""

    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None  # OpenAI delta format, with "index"
    finish_reason: str | None = None
    usage: Usage | None = None


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def completion_body(
    *, completion_id: str, model: str, resp: ProviderResponse, created: int | None = None
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": resp.content}
    if resp.tool_calls:
        message["tool_calls"] = resp.tool_calls
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created or int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": resp.finish_reason}],
        "usage": resp.usage.to_openai(),
    }
