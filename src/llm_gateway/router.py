"""Model routing: alias resolution, retries with backoff, fallback, and circuit breaking."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from llm_gateway.breaker import CircuitBreaker
from llm_gateway.config import GatewayConfig, Target
from llm_gateway.providers.base import Provider, ProviderError, ProviderTimeoutError
from llm_gateway.schemas import ChatCompletionRequest, ProviderResponse, StreamDelta

T = TypeVar("T")


class NoRouteError(Exception):
    """The requested model is neither a known alias nor an enabled ``provider/model``."""


@dataclass
class Attempt:
    target: str
    outcome: str  # "ok", an error kind, or "circuit_open"
    latency_s: float = 0.0


class AllTargetsFailedError(ProviderError):
    """Every target in the chain failed or was skipped."""

    kind = "all_providers_failed"

    def __init__(self, attempts: list[Attempt], last: ProviderError | None) -> None:
        tried = ", ".join(f"{a.target}:{a.outcome}" for a in attempts) or "none"
        super().__init__(f"all providers failed ({tried})")
        self.attempts = attempts
        self.last = last
        outcomes = {a.outcome for a in attempts}
        if outcomes == {"circuit_open"}:
            self.status_code, self.kind = 503, "circuit_open"
        elif outcomes <= {"rate_limited", "circuit_open"}:
            self.status_code, self.kind = 429, "rate_limited"
        else:
            self.status_code = 502


@dataclass
class RouteResult:
    target: Target
    response: ProviderResponse
    attempts: list[Attempt] = field(default_factory=list)


@dataclass
class StreamRoute:
    target: Target
    deltas: AsyncIterator[StreamDelta]
    attempts: list[Attempt] = field(default_factory=list)


async def _prepend(first: StreamDelta | None, rest: AsyncIterator[StreamDelta]):
    if first is not None:
        yield first
    async for d in rest:
        yield d


class Router:
    def __init__(
        self,
        config: GatewayConfig,
        providers: dict[str, Provider],
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.providers = providers
        self._sleep = sleep
        bc = config.routing.breaker
        self.breakers = {
            name: CircuitBreaker(bc.failure_threshold, bc.cooldown_s, clock=clock)
            for name in providers
        }
        # Optional hook: called once per provider attempt (used for metrics/tracing).
        self.on_attempt: Callable[[Target, str, float], None] | None = None

    def targets(self, model: str) -> list[Target]:
        targets = [t for t in self.config.resolve(model) if t.provider in self.providers]
        if not targets:
            raise NoRouteError(f"model {model!r} is not an alias or an enabled provider/model")
        return targets

    def timeout_for(self, provider: str) -> float:
        cfg = self.config.providers.get(provider, {})
        return float(cfg.get("timeout_s", self.config.defaults.timeout_s))

    def backoff(self, attempt: int, retry_after: float | None) -> float:
        r = self.config.routing
        delay = min(r.backoff_max_s, r.backoff_base_s * (2**attempt)) * random.uniform(0.5, 1.0)
        if retry_after is not None:
            delay = max(delay, min(retry_after, r.backoff_max_s))
        return delay

    async def _run(
        self, model: str, call: Callable[[Provider, Target], Awaitable[T]]
    ) -> tuple[Target, T, list[Attempt]]:
        attempts: list[Attempt] = []
        last: ProviderError | None = None
        retries = self.config.routing.retries
        for target in self.targets(model):
            breaker = self.breakers[target.provider]
            provider = self.providers[target.provider]
            for n in range(retries + 1):
                if not breaker.allow():
                    attempts.append(Attempt(target.id, "circuit_open"))
                    self._notify(target, "circuit_open", 0.0)
                    break
                started = time.perf_counter()
                try:
                    async with asyncio.timeout(self.timeout_for(target.provider)):
                        result = await call(provider, target)
                except TimeoutError:
                    err: ProviderError = ProviderTimeoutError(
                        f"{target.id} timed out after {self.timeout_for(target.provider)}s"
                    )
                except ProviderError as exc:
                    err = exc
                else:
                    elapsed = time.perf_counter() - started
                    breaker.record_success()
                    attempts.append(Attempt(target.id, "ok", elapsed))
                    self._notify(target, "ok", elapsed)
                    return target, result, attempts
                elapsed = time.perf_counter() - started
                attempts.append(Attempt(target.id, err.kind, elapsed))
                self._notify(target, err.kind, elapsed)
                if not err.fallback:
                    # The request is at fault, not the provider: fail fast, don't trip breakers.
                    raise err
                breaker.record_failure()
                last = err
                if not err.retryable or n == retries:
                    break
                await self._sleep(self.backoff(n, err.retry_after))
        raise AllTargetsFailedError(attempts, last)

    def _notify(self, target: Target, outcome: str, elapsed: float) -> None:
        if self.on_attempt is not None:
            self.on_attempt(target, outcome, elapsed)

    async def complete(self, req: ChatCompletionRequest) -> RouteResult:
        async def call(provider: Provider, target: Target) -> ProviderResponse:
            return await provider.chat(req, target.model)

        target, resp, attempts = await self._run(req.model, call)
        return RouteResult(target, resp, attempts)

    async def open_stream(self, req: ChatCompletionRequest) -> StreamRoute:
        """Start a stream with fallback. The first delta is pulled eagerly (inside the retry
        loop) so connection/HTTP/timeout errors surface while we can still try another target.
        Once the first delta is returned to the client, a failure can no longer fall back."""

        async def call(
            provider: Provider, target: Target
        ) -> tuple[StreamDelta | None, AsyncIterator[StreamDelta]]:
            it = aiter(provider.stream(req, target.model))
            try:
                return await anext(it), it
            except StopAsyncIteration:
                return None, it

        target, (first, it), attempts = await self._run(req.model, call)
        return StreamRoute(target, _prepend(first, it), attempts)
