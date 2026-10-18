"""Usage aggregation queries over the ledger."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from sqlalchemy import Integer, case, cast, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from llm_gateway.db.models import ApiKey, UsageEvent


async def usage_summary(
    sessionmaker: async_sessionmaker,
    *,
    since: date | None = None,
    until: date | None = None,
    api_key_id: int | None = None,
) -> list[dict[str, Any]]:
    """Requests, tokens, cost and latency grouped by day, API key, and model."""
    day = func.date(UsageEvent.created_at).label("day")
    stmt = (
        select(
            day,
            UsageEvent.api_key_id,
            ApiKey.name.label("key_name"),
            UsageEvent.tenant,
            UsageEvent.requested_model,
            UsageEvent.provider,
            UsageEvent.model,
            func.count().label("requests"),
            func.sum(cast(UsageEvent.cache_hit, Integer)).label("cache_hits"),
            func.sum(case((UsageEvent.status != "ok", 1), else_=0)).label("errors"),
            func.sum(UsageEvent.prompt_tokens).label("prompt_tokens"),
            func.sum(UsageEvent.completion_tokens).label("completion_tokens"),
            func.sum(UsageEvent.cost_usd).label("cost_usd"),
            func.avg(UsageEvent.latency_ms).label("avg_latency_ms"),
        )
        .outerjoin(ApiKey, ApiKey.id == UsageEvent.api_key_id)
        .group_by(
            day,
            UsageEvent.api_key_id,
            ApiKey.name,
            UsageEvent.tenant,
            UsageEvent.requested_model,
            UsageEvent.provider,
            UsageEvent.model,
        )
        .order_by(day, UsageEvent.api_key_id, UsageEvent.requested_model)
    )
    if since:
        stmt = stmt.where(UsageEvent.created_at >= datetime.combine(since, time.min, UTC))
    if until:
        end = datetime.combine(until + timedelta(days=1), time.min, UTC)
        stmt = stmt.where(UsageEvent.created_at < end)
    if api_key_id is not None:
        stmt = stmt.where(UsageEvent.api_key_id == api_key_id)

    async with sessionmaker() as session:
        rows = (await session.execute(stmt)).mappings().all()
    return [
        {
            "day": str(r["day"]),
            "api_key_id": r["api_key_id"],
            "key_name": r["key_name"],
            "tenant": r["tenant"],
            "requested_model": r["requested_model"],
            "provider": r["provider"],
            "model": r["model"],
            "requests": int(r["requests"]),
            "cache_hits": int(r["cache_hits"] or 0),
            "errors": int(r["errors"] or 0),
            "prompt_tokens": int(r["prompt_tokens"] or 0),
            "completion_tokens": int(r["completion_tokens"] or 0),
            "cost_usd": round(float(r["cost_usd"] or 0), 8),
            "avg_latency_ms": round(float(r["avg_latency_ms"] or 0), 1),
        }
        for r in rows
    ]
