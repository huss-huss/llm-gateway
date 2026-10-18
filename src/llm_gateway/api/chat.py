"""``POST /v1/chat/completions`` and ``GET /v1/models``."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from llm_gateway.errors import GatewayHTTPError
from llm_gateway.providers.base import ProviderError
from llm_gateway.router import NoRouteError
from llm_gateway.schemas import ChatCompletionRequest, completion_body, new_completion_id
from llm_gateway.state import GatewayState

router = APIRouter()


def get_state(request: Request) -> GatewayState:
    return request.app.state.gateway


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    state = get_state(request)
    try:
        result = await state.router.complete(body)
    except NoRouteError as exc:
        raise GatewayHTTPError(404, str(exc), "invalid_request_error", "model_not_found") from exc
    except ProviderError as exc:
        raise GatewayHTTPError(exc.status_code, str(exc), "upstream_error", exc.kind) from exc
    payload = completion_body(
        completion_id=new_completion_id(), model=body.model, resp=result.response
    )
    return JSONResponse(
        payload,
        headers={
            "x-gateway-provider": result.target.provider,
            "x-gateway-model": result.target.model,
        },
    )


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
