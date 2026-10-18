"""Resolve a requested model to a provider target and call it."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from llm_gateway.config import GatewayConfig, Target
from llm_gateway.providers.base import Provider, ProviderError
from llm_gateway.schemas import ChatCompletionRequest, ProviderResponse, StreamDelta


class NoRouteError(Exception):
    pass


@dataclass
class RouteResult:
    target: Target
    response: ProviderResponse


@dataclass
class StreamRoute:
    target: Target
    deltas: AsyncIterator[StreamDelta]


async def _prepend(first: StreamDelta | None, rest: AsyncIterator[StreamDelta]):
    if first is not None:
        yield first
    async for d in rest:
        yield d


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

    async def open_stream(self, req: ChatCompletionRequest) -> StreamRoute:
        """Start a stream. The first delta is pulled eagerly so connection/HTTP errors surface
        here (where we can still fall back) rather than after the client saw a 200."""
        target = self.targets(req.model)[0]
        it = aiter(self.providers[target.provider].stream(req, target.model))
        try:
            first = await anext(it)
        except StopAsyncIteration:
            first = None
        return StreamRoute(target, _prepend(first, it))
