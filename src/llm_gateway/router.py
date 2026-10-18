"""Resolve a requested model to a provider target and call it."""

from __future__ import annotations

from dataclasses import dataclass

from llm_gateway.config import GatewayConfig, Target
from llm_gateway.providers.base import Provider, ProviderError
from llm_gateway.schemas import ChatCompletionRequest, ProviderResponse


class NoRouteError(Exception):
    pass


@dataclass
class RouteResult:
    target: Target
    response: ProviderResponse


class Router:
    def __init__(self, config: GatewayConfig, providers: dict[str, Provider]) -> None:
        self.config = config
        self.providers = providers

    def targets(self, model: str) -> list[Target]:
        targets = [t for t in self.config.resolve(model) if t.provider in self.providers]
        if not targets:
            raise NoRouteError(f"model {model!r} is not an alias or an enabled provider/model")
        return targets

    async def complete(self, req: ChatCompletionRequest) -> RouteResult:
        target = self.targets(req.model)[0]
        try:
            resp = await self.providers[target.provider].chat(req, target.model)
        except ProviderError:
            raise
        return RouteResult(target, resp)
