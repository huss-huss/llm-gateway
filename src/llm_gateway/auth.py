"""API key issuance, hashing, and lookup."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from llm_gateway.db.models import ApiKey

KEY_PREFIX = "lgw_"
UPDATABLE_FIELDS = {"rate_limit_rpm", "rate_limit_tpm", "monthly_budget_usd"}


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(raw: str, pepper: str) -> str:
    """Keys are 256-bit random tokens, so a keyed fast hash is sufficient (no bcrypt needed):
    there is nothing to brute-force, and per-request verification must be cheap."""
    return hmac.new(pepper.encode(), raw.encode(), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class AuthContext:
    key_id: int | None
    name: str
    tenant: str
    rate_limit_rpm: int | None = None
    monthly_budget_usd: Decimal | None = None
    rate_limit_tpm: int | None = None

    @classmethod
    def anonymous(cls) -> AuthContext:
        return cls(key_id=None, name="anonymous", tenant="anonymous")


class KeyStore:
    """Looks up keys by hash, with a short in-process cache to keep the hot path off the DB.

    Revocation through this process is immediate; other replicas notice within ``cache_ttl_s``.
    """

    def __init__(self, sessionmaker: async_sessionmaker, pepper: str, cache_ttl_s: float = 5.0):
        self.sessionmaker = sessionmaker
        self.pepper = pepper
        self.cache_ttl_s = cache_ttl_s
        self._cache: dict[str, tuple[float, AuthContext | None]] = {}

    async def create(
        self,
        name: str,
        tenant: str,
        rate_limit_rpm: int | None = None,
        monthly_budget_usd: Decimal | float | None = None,
        rate_limit_tpm: int | None = None,
    ) -> tuple[str, ApiKey]:
        raw = generate_key()
        row = ApiKey(
            name=name,
            tenant=tenant,
            prefix=raw[:12],
            key_hash=hash_key(raw, self.pepper),
            rate_limit_rpm=rate_limit_rpm,
            rate_limit_tpm=rate_limit_tpm,
            monthly_budget_usd=(
                Decimal(str(monthly_budget_usd)) if monthly_budget_usd is not None else None
            ),
        )
        async with self.sessionmaker() as session:
            session.add(row)
            await session.commit()
        return raw, row

    async def revoke(self, key_id: int) -> ApiKey | None:
        async with self.sessionmaker() as session:
            row = await session.get(ApiKey, key_id)
            if row is None:
                return None
            if row.revoked_at is None:
                row.revoked_at = datetime.now(UTC)
                await session.commit()
        self._forget(key_id)
        return row

    async def update(self, key_id: int, **fields: Any) -> ApiKey | None:
        """Change a key's limits (``rate_limit_rpm``, ``rate_limit_tpm``, ``monthly_budget_usd``).

        Like revocation, the change is immediate here and reaches other replicas within
        ``cache_ttl_s``.
        """
        unknown = set(fields) - UPDATABLE_FIELDS
        if unknown:
            raise ValueError(f"cannot update {sorted(unknown)}")
        if fields.get("monthly_budget_usd") is not None:
            fields["monthly_budget_usd"] = Decimal(str(fields["monthly_budget_usd"]))
        async with self.sessionmaker() as session:
            row = await session.get(ApiKey, key_id)
            if row is None:
                return None
            for name, value in fields.items():
                setattr(row, name, value)
            await session.commit()
        self._forget(key_id)
        return row

    def _forget(self, key_id: int) -> None:
        self._cache = {h: v for h, v in self._cache.items() if not v[1] or v[1].key_id != key_id}

    async def list(self) -> list[ApiKey]:
        async with self.sessionmaker() as session:
            return list((await session.scalars(select(ApiKey).order_by(ApiKey.id))).all())

    async def authenticate(self, raw: str) -> AuthContext | None:
        if not raw.startswith(KEY_PREFIX):
            return None
        digest = hash_key(raw, self.pepper)
        now = time.monotonic()
        hit = self._cache.get(digest)
        if hit and now - hit[0] < self.cache_ttl_s:
            return hit[1]
        async with self.sessionmaker() as session:
            row = await session.scalar(select(ApiKey).where(ApiKey.key_hash == digest))
        ctx = None
        if row is not None and row.active:
            ctx = AuthContext(
                key_id=row.id,
                name=row.name,
                tenant=row.tenant,
                rate_limit_rpm=row.rate_limit_rpm,
                monthly_budget_usd=row.monthly_budget_usd,
                rate_limit_tpm=row.rate_limit_tpm,
            )
        self._cache[digest] = (now, ctx)
        return ctx


def bearer_token(authorization: str | None, x_api_key: str | None) -> str | None:
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token.strip()
    return x_api_key.strip() if x_api_key else None
