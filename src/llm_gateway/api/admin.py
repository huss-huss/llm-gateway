"""Admin API (``/admin``): API key lifecycle and usage reporting. Requires the admin key."""

from __future__ import annotations

import hmac
from datetime import date
from decimal import Decimal

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, Field

from llm_gateway.api.deps import get_state
from llm_gateway.auth import AuthContext, bearer_token
from llm_gateway.db.models import ApiKey
from llm_gateway.errors import GatewayHTTPError
from llm_gateway.usage import usage_summary

router = APIRouter(prefix="/admin", tags=["admin"])


async def require_admin(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    configured = get_state(request).settings.admin_key
    if configured is None or not configured.get_secret_value():
        raise GatewayHTTPError(
            503, "admin API disabled: set GATEWAY_ADMIN_KEY", "admin_disabled", "admin_disabled"
        )
    token = bearer_token(authorization, x_api_key) or ""
    if not hmac.compare_digest(token.encode(), configured.get_secret_value().encode()):
        raise GatewayHTTPError(
            401, "invalid admin key", "authentication_error", "invalid_admin_key"
        )


class CreateKey(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    tenant: str = Field(min_length=1, max_length=200)
    rate_limit_rpm: int | None = Field(default=None, ge=1)
    monthly_budget_usd: Decimal | None = Field(default=None, ge=0)


def key_view(row: ApiKey) -> dict:
    return {
        "id": row.id,
        "name": row.name,
        "tenant": row.tenant,
        "prefix": row.prefix,
        "rate_limit_rpm": row.rate_limit_rpm,
        "monthly_budget_usd": (
            float(row.monthly_budget_usd) if row.monthly_budget_usd is not None else None
        ),
        "created_at": row.created_at.isoformat(),
        "revoked_at": row.revoked_at.isoformat() if row.revoked_at else None,
    }


@router.post("/keys", status_code=201, dependencies=[Depends(require_admin)])
async def create_key(body: CreateKey, request: Request):
    state = get_state(request)
    raw, row = await state.keys.create(
        body.name, body.tenant, body.rate_limit_rpm, body.monthly_budget_usd
    )
    # The raw key is returned exactly once.
    return {**key_view(row), "key": raw}


@router.get("/keys", dependencies=[Depends(require_admin)])
async def list_keys(request: Request):
    state = get_state(request)
    out = []
    for row in await state.keys.list():
        view = key_view(row)
        ctx = AuthContext(row.id, row.name, row.tenant)
        view["month_spend_usd"] = round(await state.budgets.spent_usd(ctx), 8)
        out.append(view)
    return {"data": out}


@router.delete("/keys/{key_id}", dependencies=[Depends(require_admin)])
async def revoke_key(key_id: int, request: Request):
    row = await get_state(request).keys.revoke(key_id)
    if row is None:
        raise GatewayHTTPError(404, f"key {key_id} not found", "not_found", "key_not_found")
    return key_view(row)


@router.get("/usage", dependencies=[Depends(require_admin)])
async def usage(
    request: Request,
    since: date | None = None,
    until: date | None = None,
    api_key_id: int | None = None,
):
    rows = await usage_summary(
        get_state(request).sessionmaker, since=since, until=until, api_key_id=api_key_id
    )
    totals = {
        "requests": sum(r["requests"] for r in rows),
        "cost_usd": round(sum(r["cost_usd"] for r in rows), 8),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
    }
    return {"data": rows, "totals": totals}
