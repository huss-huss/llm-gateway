"""Shared helpers for adapters that talk to upstream APIs with httpx."""

from __future__ import annotations

import os
from typing import Any

import httpx

from llm_gateway.providers.base import (
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    error_from_status,
    parse_retry_after,
)


def api_key_from(config: dict[str, Any], default_env: str | None) -> str | None:
    env = config.get("api_key_env", default_env)
    return os.environ.get(env) if env else config.get("api_key")


def http_status_error(resp: httpx.Response, provider: str) -> ProviderError:
    try:
        body = resp.json()
        err = body.get("error", body)
        detail = err.get("message") if isinstance(err, dict) else str(err)
    except ValueError:
        detail = resp.text[:500]
    return error_from_status(
        resp.status_code,
        f"{provider}: HTTP {resp.status_code}: {detail}",
        parse_retry_after(resp.headers.get("retry-after")),
    )


def transport_error(exc: httpx.HTTPError, provider: str) -> ProviderError:
    if isinstance(exc, httpx.TimeoutException):
        return ProviderTimeoutError(f"{provider}: upstream timeout ({type(exc).__name__})")
    return ProviderUnavailableError(f"{provider}: connection error ({type(exc).__name__})")
