"""``POST /v1/chat/completions`` and ``GET /v1/models``."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_gateway.api.deps import get_request_id, get_state, require_api_key
from llm_gateway.auth import AuthContext
from llm_gateway.schemas import ChatCompletionRequest
from llm_gateway.service import ChatService

router = APIRouter()

SSE_HEADERS = {"cache-control": "no-cache", "x-accel-buffering": "no"}


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    ctx: AuthContext = Depends(require_api_key),
):
    service = ChatService(get_state(request))
    rid = get_request_id(request)
    if body.stream:
        result = await service.stream(ctx, body, rid)
        return StreamingResponse(
            result.events, media_type="text/event-stream", headers={**SSE_HEADERS, **result.headers}
        )
    result = await service.complete(ctx, body, rid)
    return JSONResponse(result.payload, headers=result.headers)


@router.get("/v1/models")
async def list_models(request: Request, _: AuthContext = Depends(require_api_key)):
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
