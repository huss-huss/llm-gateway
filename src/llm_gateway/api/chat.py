"""``POST /v1/chat/completions`` and ``GET /v1/models``."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_gateway.config import Target
from llm_gateway.errors import GatewayHTTPError, error_body
from llm_gateway.providers.base import ProviderError
from llm_gateway.router import Attempt, NoRouteError
from llm_gateway.schemas import ChatCompletionRequest, completion_body, new_completion_id
from llm_gateway.state import GatewayState
from llm_gateway.streaming import DONE, StreamAccumulator, encode_stream, sse

router = APIRouter()

SSE_HEADERS = {"cache-control": "no-cache", "x-accel-buffering": "no"}


def get_state(request: Request) -> GatewayState:
    return request.app.state.gateway


def route_headers(target: Target, attempts: list[Attempt]) -> dict[str, str]:
    return {
        "x-gateway-provider": target.provider,
        "x-gateway-model": target.model,
        "x-gateway-attempts": str(len(attempts)),
    }


def _to_http(exc: Exception) -> GatewayHTTPError:
    if isinstance(exc, NoRouteError):
        return GatewayHTTPError(404, str(exc), "invalid_request_error", "model_not_found")
    if isinstance(exc, ProviderError):
        type_ = "invalid_request_error" if exc.status_code == 400 else "upstream_error"
        headers = {"retry-after": "1"} if exc.status_code in (429, 503) else None
        return GatewayHTTPError(exc.status_code, str(exc), type_, exc.kind, headers)
    raise exc


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    state = get_state(request)
    completion_id = new_completion_id()

    if body.stream:
        try:
            route = await state.router.open_stream(body)
        except (NoRouteError, ProviderError) as exc:
            raise _to_http(exc) from exc
        include_usage = bool((body.stream_options or {}).get("include_usage"))
        acc = StreamAccumulator()

        async def events() -> AsyncIterator[str]:
            try:
                async for line in encode_stream(
                    route.deltas,
                    completion_id=completion_id,
                    model=body.model,
                    include_usage=include_usage,
                    acc=acc,
                ):
                    yield line
            except ProviderError as exc:
                # Headers are already sent; report the failure in-band, as OpenAI does.
                yield sse(error_body(str(exc), "upstream_error", exc.kind))
            yield DONE

        headers = {**SSE_HEADERS, **route_headers(route.target, route.attempts)}
        return StreamingResponse(events(), media_type="text/event-stream", headers=headers)

    try:
        result = await state.router.complete(body)
    except (NoRouteError, ProviderError) as exc:
        raise _to_http(exc) from exc
    payload = completion_body(completion_id=completion_id, model=body.model, resp=result.response)
    headers = route_headers(result.target, result.attempts)
    cost = state.catalog.cost(result.target.id, result.response.usage)
    if cost is not None:
        headers["x-gateway-cost-usd"] = f"{cost:.8f}"
    return JSONResponse(payload, headers=headers)


@router.get("/v1/models")
async def list_models(request: Request):
    state = get_state(request)
    now = int(time.time())
    data = [
        {"id": alias, "object": "model", "created": now, "owned_by": "llm-gateway"}
        for alias in state.config.aliases
    ]
    seen = set()
    for targets in state.config.aliases.values():
        for t in targets:
            if t.provider in state.providers and t.id not in seen:
                seen.add(t.id)
                data.append({"id": t.id, "object": "model", "created": now, "owned_by": t.provider})
    return {"object": "list", "data": data}
