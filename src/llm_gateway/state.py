"""The application container: everything a request handler needs, built once at startup."""

from __future__ import annotations

from dataclasses import dataclass

from llm_gateway.catalog import Catalog, load_catalog
from llm_gateway.config import GatewayConfig, load_gateway_config
from llm_gateway.providers import build_provider
from llm_gateway.providers.base import Provider
from llm_gateway.router import Router
from llm_gateway.settings import Settings


@dataclass
class GatewayState:
    settings: Settings
    config: GatewayConfig
    catalog: Catalog
    providers: dict[str, Provider]
    router: Router

    async def aclose(self) -> None:
        for provider in self.providers.values():
            await provider.aclose()


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
    return GatewayState(
        settings=settings,
        config=config,
        catalog=catalog,
        providers=providers,
        router=Router(config, providers),
    )
