"""Async engine/session factory."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine


def make_engine(url: str) -> AsyncEngine:
    kwargs = {}
    if url.startswith("postgresql"):
        kwargs = {"pool_size": 10, "max_overflow": 10, "pool_pre_ping": True}
    return create_async_engine(url, **kwargs)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)
