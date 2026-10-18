"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

from llm_gateway import __version__
from llm_gateway.api import chat
from llm_gateway.errors import GatewayHTTPError, error_response
from llm_gateway.request_context import RequestIDMiddleware
from llm_gateway.settings import Settings, get_settings
from llm_gateway.state import GatewayState, build_state


def create_app(settings: Settings | None = None, state: GatewayState | None = None) -> FastAPI:
    """Build the app. Tests pass a prebuilt ``state``; production builds it at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = False
        if getattr(app.state, "gateway", None) is None:
            app.state.gateway = await build_state(settings or get_settings())
            owned = True
        yield
        if owned:
            await app.state.gateway.aclose()

    app = FastAPI(title="llm-gateway", version=__version__, lifespan=lifespan)
    app.state.gateway = state

    @app.exception_handler(GatewayHTTPError)
    async def _gateway_error(_: Request, exc: GatewayHTTPError):
        return error_response(exc.status, exc.message, exc.type, exc.code, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        return error_response(
            400,
            f"{loc}: {first.get('msg', 'invalid request')}",
            "invalid_request_error",
            "invalid_request",
        )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "version": __version__}

    app.include_router(chat.router)
    app.add_middleware(RequestIDMiddleware)
    return app
