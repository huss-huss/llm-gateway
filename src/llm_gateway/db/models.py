"""SQLAlchemy 2.x ORM models. Schema changes go through Alembic (``migrations/``)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# BIGINT autoincrement on Postgres, INTEGER PRIMARY KEY (rowid) on SQLite.
BigIntPK = BigInteger().with_variant(Integer, "sqlite")
Money = Numeric(14, 8)


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200))
    tenant: Mapped[str] = mapped_column(String(200), index=True)
    # First characters of the key, safe to display and log ("lgw_AbCd1234").
    prefix: Mapped[str] = mapped_column(String(32))
    # HMAC-SHA256(pepper, key) hex. The raw key is shown once and never stored.
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    rate_limit_rpm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    monthly_budget_usd: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    @property
    def active(self) -> bool:
        return self.revoked_at is None


class UsageEvent(Base):
    """One row per chat completion request handled by the gateway (the usage ledger)."""

    __tablename__ = "usage_events"
    __table_args__ = (Index("ix_usage_events_key_created", "api_key_id", "created_at"),)

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    api_key_id: Mapped[int | None] = mapped_column(
        ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True
    )
    tenant: Mapped[str | None] = mapped_column(String(200), nullable=True)
    requested_model: Mapped[str] = mapped_column(String(200))
    provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[str] = mapped_column(String(40))  # "ok" or an error kind
    http_status: Mapped[int] = mapped_column(Integer)
    stream: Mapped[bool] = mapped_column(Boolean, default=False)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[Decimal] = mapped_column(Money, default=Decimal(0))
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    suite: Mapped[str] = mapped_column(String(200))
    model: Mapped[str] = mapped_column(String(200))
    cases: Mapped[int] = mapped_column(Integer)
    passed: Mapped[int] = mapped_column(Integer)
    pass_rate: Mapped[float] = mapped_column(Float)
    total_cost_usd: Mapped[Decimal] = mapped_column(Money)
    avg_latency_ms: Mapped[float] = mapped_column(Float)
    p95_latency_ms: Mapped[float] = mapped_column(Float)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True)
    case_id: Mapped[str] = mapped_column(String(200))
    passed: Mapped[bool] = mapped_column(Boolean)
    provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model: Mapped[str | None] = mapped_column(String(200), nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[Decimal] = mapped_column(Money)
    prompt_tokens: Mapped[int] = mapped_column(Integer)
    completion_tokens: Mapped[int] = mapped_column(Integer)
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    failures: Mapped[str] = mapped_column(Text)  # JSON list of failure messages
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
