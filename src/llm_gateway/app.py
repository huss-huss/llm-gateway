"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from llm_gateway import __version__
from llm_gateway.api import admin, chat
from llm_gateway.errors import GatewayHTTPError, error_response
from llm_gateway.observability.logging import configure_logging
from llm_gateway.observability.tracing import setup_tracing
from llm_gateway.request_context import RequestIDMiddleware
from llm_gateway.settings import Settings, get_settings
from llm_gateway.state import GatewayState, build_state


def create_app(settings: Settings | None = None, state: GatewayState | None = None) -> FastAPI:
    """Build the app. Tests pass a prebuilt ``state``; production builds it at startup."""
    if state is None:
        cfg = settings or get_settings()
        configure_logging(cfg.log_level, cfg.log_json)
        setup_tracing(cfg.otel_exporter, cfg.service_name, cfg.otel_endpoint)

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

    @app.get("/readyz")
    async def readyz(request: Request):
        st = request.app.state.gateway
        checks = {}
        try:
            await st.redis.ping()
            checks["redis"] = "ok"
        except Exception as exc:
            checks["redis"] = f"error: {type(exc).__name__}"
        try:
            async with st.engine.connect() as conn:
                await conn.exec_driver_sql("SELECT 1")
            checks["database"] = "ok"
        except Exception as exc:
            checks["database"] = f"error: {type(exc).__name__}"
        ok = all(v == "ok" for v in checks.values())
        return JSONResponse(
            {
                "status": "ok" if ok else "degraded",
                "checks": checks,
                "providers": sorted(st.providers),
            },
            status_code=200 if ok else 503,
        )

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.include_router(chat.router)
    app.include_router(admin.router)
    app.add_middleware(RequestIDMiddleware)
    return app
