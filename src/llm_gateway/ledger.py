"""Writes one ``usage_events`` row per request."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from decimal import Decimal

from sqlalchemy.ext.asyncio import async_sessionmaker

from llm_gateway.db.models import UsageEvent

log = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    request_id: str
    requested_model: str
    status: str
    http_status: int
    api_key_id: int | None = None
    tenant: str | None = None
    provider: str | None = None
    model: str | None = None
    stream: bool = False
    cache_hit: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    attempts: int = 0


class UsageLedger:
    def __init__(self, sessionmaker: async_sessionmaker) -> None:
        self.sessionmaker = sessionmaker

    async def record(self, rec: UsageRecord) -> None:
        """Best effort: a ledger write failure is logged, never surfaced to the client."""
        data = asdict(rec)
        data["cost_usd"] = Decimal(str(rec.cost_usd))
        try:
            async with self.sessionmaker() as session:
                session.add(UsageEvent(**data))
                await session.commit()
        except Exception:
            log.exception("failed to write usage event", extra={"request_id": rec.request_id})
