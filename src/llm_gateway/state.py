"""The application container: everything a request handler needs, built once at startup."""

from __future__ import annotations

from dataclasses import dataclass

from llm_gateway.config import GatewayConfig, load_gateway_config
from llm_gateway.providers import build_provider
from llm_gateway.providers.base import Provider
from llm_gateway.router import Router
from llm_gateway.settings import Settings


@dataclass
class GatewayState:
    settings: Settings
    config: GatewayConfig
    providers: dict[str, Provider]
    router: Router

    async def aclose(self) -> None:
        for provider in self.providers.values():
            await provider.aclose()


def build_providers(config: GatewayConfig) -> dict[str, Provider]:
    return {
        name: build_provider(name, cfg)
        for name, cfg in config.providers.items()
        if config.provider_enabled(name)
    }


async def build_state(settings: Settings, config: GatewayConfig | None = None) -> GatewayState:
    config = config or load_gateway_config(settings.config_path)
    providers = build_providers(config)
    return GatewayState(
        settings=settings, config=config, providers=providers, router=Router(config, providers)
    )
