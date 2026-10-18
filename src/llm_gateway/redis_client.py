"""Redis connection factory. ``memory://`` gives an in-process fakeredis (dev/demo/tests only)."""

from __future__ import annotations

from redis.asyncio import Redis


def make_redis(url: str) -> Redis:
    if url.startswith("memory://"):
        try:
            from fakeredis import FakeAsyncRedis, FakeServer
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise RuntimeError(
                "GATEWAY_REDIS_URL=memory:// needs fakeredis: pip install 'llm-gateway[dev]'"
            ) from exc
        # A private server per call: state is never shared between app instances/tests.
        return FakeAsyncRedis(server=FakeServer(), decode_responses=True)
    return Redis.from_url(url, decode_responses=True)
