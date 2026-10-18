"""Monthly USD budgets per API key.

Spend is kept in a Redis counter (integer micro-dollars) per key and calendar month (UTC).
On a cold counter, it is rebuilt from the usage ledger, so Redis can be flushed safely.
Budgets are checked *before* calling a provider; a request that starts under budget may finish
slightly over it (the overshoot is bounded by one request's cost).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal

from redis.asyncio import Redis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from llm_gateway.auth import AuthContext
from llm_gateway.db.models import UsageEvent

MICRO = 1_000_000
COUNTER_TTL_S = 40 * 24 * 3600


def month_start(now: datetime) -> datetime:
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


class BudgetTracker:
    def __init__(
        self,
        redis: Redis,
        sessionmaker: async_sessionmaker,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.redis = redis
        self.sessionmaker = sessionmaker
        self.clock = clock

    def _key(self, key_id: int, now: datetime) -> str:
        return f"budget:{key_id}:{now:%Y%m}"

    async def _db_spend_micro(self, key_id: int, since: datetime) -> int:
        async with self.sessionmaker() as session:
            total = await session.scalar(
                select(func.coalesce(func.sum(UsageEvent.cost_usd), 0)).where(
                    UsageEvent.api_key_id == key_id, UsageEvent.created_at >= since
                )
            )
        return int(Decimal(str(total)) * MICRO)

    async def _ensure(self, key_id: int, now: datetime) -> str:
        key = self._key(key_id, now)
        if not await self.redis.exists(key):
            micro = await self._db_spend_micro(key_id, month_start(now))
            await self.redis.set(key, micro, ex=COUNTER_TTL_S, nx=True)
        return key

    async def spent_usd(self, ctx: AuthContext) -> float:
        if ctx.key_id is None:
            return 0.0
        key = await self._ensure(ctx.key_id, self.clock())
        return int(await self.redis.get(key) or 0) / MICRO

    async def remaining_usd(self, ctx: AuthContext) -> float | None:
        """Remaining budget, or ``None`` if the key has no budget."""
        if ctx.monthly_budget_usd is None or ctx.key_id is None:
            return None
        return float(ctx.monthly_budget_usd) - await self.spent_usd(ctx)

    async def add(self, ctx: AuthContext, cost_usd: float) -> None:
        """Record spend. Must run before the ledger row is written (see ``_ensure``)."""
        if ctx.key_id is None or cost_usd <= 0:
            return
        key = await self._ensure(ctx.key_id, self.clock())
        await self.redis.incrby(key, round(cost_usd * MICRO))
