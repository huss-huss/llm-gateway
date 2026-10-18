"""FastAPI dependencies: state access and API key authentication."""

from __future__ import annotations

from fastapi import Header, Request

from llm_gateway.auth import AuthContext, bearer_token
from llm_gateway.errors import GatewayHTTPError
from llm_gateway.state import GatewayState


def get_state(request: Request) -> GatewayState:
    return request.app.state.gateway


def get_request_id(request: Request) -> str:
    return request.scope.get("state", {}).get("request_id") or "req_unknown"


async def require_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> AuthContext:
    state = get_state(request)
    if not state.settings.require_auth:
        return AuthContext.anonymous()
    token = bearer_token(authorization, x_api_key)
    if not token:
        raise GatewayHTTPError(401, "missing API key", "authentication_error", "missing_api_key")
    ctx = await state.keys.authenticate(token)
    if ctx is None:
        raise GatewayHTTPError(
            401, "invalid or revoked API key", "authentication_error", "invalid_api_key"
        )
    return ctx
