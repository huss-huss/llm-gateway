"""The provider interface every adapter implements, and the error taxonomy the router uses."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from llm_gateway.schemas import ChatCompletionRequest, ProviderResponse, StreamDelta


class ProviderError(Exception):
    """Base class. ``retryable``: retry the same target; ``fallback``: try the next target."""

    retryable: bool = False
    fallback: bool = True
    status_code: int = 502
    kind: str = "provider_error"

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RateLimitedError(ProviderError):
    retryable = True
    status_code = 429
    kind = "rate_limited"


class ProviderTimeoutError(ProviderError):
    retryable = True
    status_code = 504
    kind = "timeout"


class ProviderUnavailableError(ProviderError):
    """5xx responses and connection failures."""

    retryable = True
    status_code = 502
    kind = "unavailable"


class ProviderAuthError(ProviderError):
    """Bad/missing provider credentials: retrying is pointless, another provider may work."""

    retryable = False
    status_code = 502
    kind = "auth"


class InvalidRequestError(ProviderError):
    """The request itself is invalid; neither retry nor fallback will help."""

    retryable = False
    fallback = False
    status_code = 400
    kind = "invalid_request"


def error_from_status(status: int, message: str, retry_after: float | None = None) -> ProviderError:
    """Map an upstream HTTP status to the gateway's error taxonomy."""
    if status == 429:
        return RateLimitedError(message, retry_after=retry_after)
    if status in (401, 403):
        return ProviderAuthError(message)
    if status in (408, 409) or status >= 500:
        return ProviderUnavailableError(message, retry_after=retry_after)
    if status == 404:
        # Unknown model on this provider: another provider in the chain may have it.
        err = ProviderUnavailableError(message)
        err.retryable = False
        return err
    return InvalidRequestError(message)


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for providers that do not report usage."""
    return max(1, math.ceil(len(text) / 4)) if text else 0


class Provider(ABC):
    """One upstream LLM API. Adapters translate to/from the OpenAI chat schema."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = name
        self.config = config

    @abstractmethod
    async def chat(self, req: ChatCompletionRequest, model: str) -> ProviderResponse: ...

    @abstractmethod
    def stream(self, req: ChatCompletionRequest, model: str) -> AsyncIterator[StreamDelta]: ...

    async def aclose(self) -> None:  # pragma: no cover - trivial default
        return None
