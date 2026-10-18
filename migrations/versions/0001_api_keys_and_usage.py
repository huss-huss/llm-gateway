"""api keys and usage ledger

Revision ID: 0001
Revises:
Create Date: 2026-10-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

BigIntPK = sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def upgrade() -> None:
    op.create_table(
        "api_keys",
        sa.Column("id", BigIntPK, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("tenant", sa.String(200), nullable=False),
        sa.Column("prefix", sa.String(32), nullable=False),
        sa.Column("key_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("rate_limit_rpm", sa.Integer(), nullable=True),
        sa.Column("monthly_budget_usd", sa.Numeric(14, 8), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_api_keys_tenant", "api_keys", ["tenant"])

    op.create_table(
        "usage_events",
        sa.Column("id", BigIntPK, primary_key=True, autoincrement=True),
        sa.Column("request_id", sa.String(64), nullable=False),
        sa.Column(
            "api_key_id",
            BigIntPK,
            sa.ForeignKey("api_keys.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("tenant", sa.String(200), nullable=True),
        sa.Column("requested_model", sa.String(200), nullable=False),
        sa.Column("provider", sa.String(100), nullable=True),
        sa.Column("model", sa.String(200), nullable=True),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("http_status", sa.Integer(), nullable=False),
        sa.Column("stream", sa.Boolean(), nullable=False),
        sa.Column("cache_hit", sa.Boolean(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(14, 8), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_usage_events_request_id", "usage_events", ["request_id"])
    op.create_index("ix_usage_events_created_at", "usage_events", ["created_at"])
    op.create_index("ix_usage_events_key_created", "usage_events", ["api_key_id", "created_at"])


def downgrade() -> None:
    op.drop_table("usage_events")
    op.drop_table("api_keys")
