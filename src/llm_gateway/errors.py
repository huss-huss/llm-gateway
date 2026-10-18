"""OpenAI-style error envelopes."""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse


def error_body(message: str, type_: str, code: str | None = None) -> dict[str, Any]:
    return {"error": {"message": message, "type": type_, "code": code}}


def error_response(
    status: int, message: str, type_: str, code: str | None = None, headers: dict | None = None
) -> JSONResponse:
    return JSONResponse(error_body(message, type_, code), status_code=status, headers=headers)


class GatewayHTTPError(Exception):
    """Raised inside handlers; rendered as an OpenAI-style error by the app."""

    def __init__(
        self,
        status: int,
        message: str,
        type_: str,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type_
        self.code = code
        self.headers = headers or {}
