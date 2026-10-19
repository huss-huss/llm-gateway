"""The application container: everything a request handler needs, built once at startup."""

from __future__ import annotations

from dataclasses import dataclass

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from llm_gateway.auth import KeyStore
from llm_gateway.breaker import build_breakers
from llm_gateway.budget import BudgetTracker
from llm_gateway.cache import ResponseCache
from llm_gateway.catalog import Catalog, load_catalog
from llm_gateway.config import GatewayConfig, load_gateway_config
from llm_gateway.db.session import make_engine, make_sessionmaker
from llm_gateway.ledger import UsageLedger
from llm_gateway.providers import build_provider
from llm_gateway.providers.base import Provider
from llm_gateway.ratelimit import RateLimiter
from llm_gateway.redis_client import make_redis
from llm_gateway.router import Router
from llm_gateway.settings import Settings


@dataclass
class GatewayState:
    settings: Settings
    config: GatewayConfig
    catalog: Catalog
    providers: dict[str, Provider]
    router: Router
    engine: AsyncEngine
    sessionmaker: async_sessionmaker
    keys: KeyStore
    ledger: UsageLedger
    redis: Redis
    limiter: RateLimiter
    budgets: BudgetTracker
    cache: ResponseCache

    async def aclose(self) -> None:
        for provider in self.providers.values():
            await provider.aclose()
        await self.engine.dispose()
        await self.redis.aclose()


def build_providers(config: GatewayConfig, catalog: Catalog | None = None) -> dict[str, Provider]:
    catalog = catalog or Catalog()
    providers: dict[str, Provider] = {}
    for name, cfg in config.providers.items():
        if not config.provider_enabled(name):
            continue
        merged = {
            "timeout_s": config.defaults.timeout_s,
            "default_max_tokens": config.defaults.max_tokens,
            **cfg,
        }
        providers[name] = build_provider(name, merged, catalog)
    return providers


async def build_state(settings: Settings, config: GatewayConfig | None = None) -> GatewayState:
    config = config or load_gateway_config(settings.config_path)
    catalog = load_catalog(settings.models_path)
    providers = build_providers(config, catalog)
    engine = make_engine(settings.database_url)
    sessionmaker = make_sessionmaker(engine)
    redis = make_redis(settings.redis_url)
    return GatewayState(
        settings=settings,
        config=config,
        catalog=catalog,
        providers=providers,
        router=Router(config, providers, breakers=build_breakers(config, providers, redis)),
        engine=engine,
        sessionmaker=sessionmaker,
        keys=KeyStore(sessionmaker, settings.key_pepper.get_secret_value()),
        ledger=UsageLedger(sessionmaker),
        redis=redis,
        limiter=RateLimiter(redis),
        budgets=BudgetTracker(redis, sessionmaker),
        cache=ResponseCache(redis, config.cache.ttl_s, config.cache.enabled),
    )
