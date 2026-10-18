"""The request pipeline behind ``/v1/chat/completions``.

authenticate -> resolve route -> call providers (retry/fallback) -> price -> record usage
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from llm_gateway.auth import AuthContext
from llm_gateway.cache import cache_key, is_cacheable
from llm_gateway.config import Target
from llm_gateway.errors import GatewayHTTPError, error_body
from llm_gateway.ledger import UsageRecord
from llm_gateway.providers.base import ProviderError
from llm_gateway.router import AllTargetsFailedError, Attempt, NoRouteError
from llm_gateway.schemas import (
    ChatCompletionRequest,
    ProviderResponse,
    completion_body,
    new_completion_id,
)
from llm_gateway.state import GatewayState
from llm_gateway.streaming import DONE, StreamAccumulator, encode_stream, replay, sse


@dataclass
class ChatResult:
    payload: dict
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class StreamResult:
    events: AsyncIterator[str]
    headers: dict[str, str] = field(default_factory=dict)


def to_http_error(exc: Exception) -> GatewayHTTPError:
    if isinstance(exc, NoRouteError):
        return GatewayHTTPError(404, str(exc), "invalid_request_error", "model_not_found")
    if isinstance(exc, ProviderError):
        type_ = "invalid_request_error" if exc.status_code == 400 else "upstream_error"
        headers = {"retry-after": "1"} if exc.status_code in (429, 503) else None
        return GatewayHTTPError(exc.status_code, str(exc), type_, exc.kind, headers)
    raise exc


class ChatService:
    def __init__(self, state: GatewayState) -> None:
        self.state = state

    # ---------------------------------------------------------------- helpers

    def _headers(self, target: Target, attempts: list[Attempt], cost: float | None) -> dict:
        h = {
            "x-gateway-provider": target.provider,
            "x-gateway-model": target.model,
            "x-gateway-attempts": str(len(attempts)),
        }
        if cost is not None:
            h["x-gateway-cost-usd"] = f"{cost:.8f}"
        return h

    def _record(
        self,
        *,
        ctx: AuthContext,
        req: ChatCompletionRequest,
        request_id: str,
        started: float,
        status: str,
        http_status: int,
        target: Target | None = None,
        resp: ProviderResponse | None = None,
        cost: float | None = None,
        attempts: int = 0,
        cache_hit: bool = False,
    ) -> UsageRecord:
        usage = resp.usage if resp else None
        return UsageRecord(
            request_id=request_id,
            requested_model=req.model,
            status=status,
            http_status=http_status,
            api_key_id=ctx.key_id,
            tenant=ctx.tenant,
            provider=target.provider if target else None,
            model=target.model if target else None,
            stream=req.stream,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            cost_usd=cost or 0.0,
            latency_ms=int((time.perf_counter() - started) * 1000),
            attempts=attempts,
            cache_hit=cache_hit,
        )

    async def _fail(
        self, exc: Exception, ctx: AuthContext, req: ChatCompletionRequest, rid: str, started: float
    ) -> GatewayHTTPError:
        err = to_http_error(exc)
        attempts = len(exc.attempts) if isinstance(exc, AllTargetsFailedError) else 0
        await self.state.ledger.record(
            self._record(
                ctx=ctx,
                req=req,
                request_id=rid,
                started=started,
                status=err.code or "error",
                http_status=err.status,
                attempts=attempts,
            )
        )
        return err

    async def _preflight(self, ctx: AuthContext, req: ChatCompletionRequest) -> dict[str, str]:
        """Enforce rate limit and budget before any provider is called."""
        headers: dict[str, str] = {}
        rpm = ctx.rate_limit_rpm or self.state.config.limits.default_rpm
        if rpm:
            bucket = str(ctx.key_id) if ctx.key_id is not None else f"anon:{ctx.tenant}"
            decision = await self.state.limiter.hit(bucket, rpm)
            headers.update(decision.headers())
            if not decision.allowed:
                raise GatewayHTTPError(
                    429,
                    f"rate limit of {rpm} requests/minute exceeded for this API key",
                    "rate_limit_error",
                    "rate_limit_exceeded",
                    decision.headers(),
                )
        remaining = await self.state.budgets.remaining_usd(ctx)
        if remaining is not None:
            if remaining <= 0:
                raise GatewayHTTPError(
                    402,
                    f"monthly budget of ${ctx.monthly_budget_usd} exhausted for this API key",
                    "budget_exceeded",
                    "monthly_budget_exceeded",
                    headers,
                )
            headers["x-gateway-budget-remaining-usd"] = f"{remaining:.6f}"
        return headers

    async def _begin(
        self, ctx: AuthContext, req: ChatCompletionRequest, rid: str, started: float
    ) -> dict[str, str]:
        try:
            self.state.router.targets(req.model)
        except NoRouteError as exc:
            raise await self._fail(exc, ctx, req, rid, started) from exc
        return await self._preflight(ctx, req)

    def _cache_lookup_key(self, ctx: AuthContext, req: ChatCompletionRequest, bypass: bool):
        cache = self.state.cache
        if not cache.enabled or not is_cacheable(req):
            return None, "bypass"
        return cache_key(ctx.tenant, req), ("refresh" if bypass else "lookup")

    # ---------------------------------------------------------------- entry points

    async def complete(
        self,
        ctx: AuthContext,
        req: ChatCompletionRequest,
        request_id: str,
        *,
        cache_bypass: bool = False,
    ) -> ChatResult:
        started = time.perf_counter()
        headers = await self._begin(ctx, req, request_id, started)
        key, mode = self._cache_lookup_key(ctx, req, cache_bypass)

        if key and mode == "lookup" and (hit := await self.state.cache.get(key)):
            await self.state.ledger.record(
                self._record(
                    ctx=ctx,
                    req=req,
                    request_id=request_id,
                    started=started,
                    status="ok",
                    http_status=200,
                    target=Target(provider=hit.provider, model=hit.model),
                    cache_hit=True,
                )
            )
            payload = completion_body(
                completion_id=new_completion_id(), model=req.model, resp=hit.response
            )
            headers.update(
                {
                    "x-gateway-cache": "hit",
                    "x-gateway-provider": hit.provider,
                    "x-gateway-model": hit.model,
                    "x-gateway-attempts": "0",
                    "x-gateway-cost-usd": f"{0:.8f}",
                }
            )
            return ChatResult(payload, headers)

        try:
            result = await self.state.router.complete(req)
        except (NoRouteError, ProviderError) as exc:
            raise await self._fail(exc, ctx, req, request_id, started) from exc
        cost = self.state.catalog.cost(result.target.id, result.response.usage)
        await self.state.budgets.add(ctx, cost or 0.0)
        await self.state.ledger.record(
            self._record(
                ctx=ctx,
                req=req,
                request_id=request_id,
                started=started,
                status="ok",
                http_status=200,
                target=result.target,
                resp=result.response,
                cost=cost,
                attempts=len(result.attempts),
            )
        )
        if key:
            await self.state.cache.set(
                key, result.response, result.target.provider, result.target.model
            )
        payload = completion_body(
            completion_id=new_completion_id(), model=req.model, resp=result.response
        )
        headers.update(self._headers(result.target, result.attempts, cost))
        headers["x-gateway-cache"] = "miss" if key else "bypass"
        return ChatResult(payload, headers)

    async def stream(
        self,
        ctx: AuthContext,
        req: ChatCompletionRequest,
        request_id: str,
        *,
        cache_bypass: bool = False,
    ) -> StreamResult:
        started = time.perf_counter()
        headers = await self._begin(ctx, req, request_id, started)
        key, mode = self._cache_lookup_key(ctx, req, cache_bypass)
        include_usage = bool((req.stream_options or {}).get("include_usage"))
        catalog, ledger, budgets, cache = (
            self.state.catalog,
            self.state.ledger,
            self.state.budgets,
            self.state.cache,
        )

        hit = await cache.get(key) if key and mode == "lookup" else None
        if hit is not None:
            target = Target(provider=hit.provider, model=hit.model)
            attempts: list[Attempt] = []
            deltas = replay(hit.response)
            headers["x-gateway-cache"] = "hit"
        else:
            try:
                route = await self.state.router.open_stream(req)
            except (NoRouteError, ProviderError) as exc:
                raise await self._fail(exc, ctx, req, request_id, started) from exc
            target, attempts, deltas = route.target, route.attempts, route.deltas
            headers["x-gateway-cache"] = "miss" if key else "bypass"

        acc = StreamAccumulator()

        async def events() -> AsyncIterator[str]:
            status, http_status = "ok", 200
            try:
                async for line in encode_stream(
                    deltas,
                    completion_id=new_completion_id(),
                    model=req.model,
                    include_usage=include_usage,
                    acc=acc,
                ):
                    yield line
            except ProviderError as exc:
                # Headers are already sent; report the failure in-band, as OpenAI does.
                status, http_status = exc.kind, exc.status_code
                yield sse(error_body(str(exc), "upstream_error", exc.kind))
            except (GeneratorExit, asyncio.CancelledError):
                status, http_status = "client_closed", 499
                raise
            finally:
                resp = acc.response()
                if hit is not None:
                    await ledger.record(
                        self._record(
                            ctx=ctx,
                            req=req,
                            request_id=request_id,
                            started=started,
                            status=status,
                            http_status=http_status,
                            target=target,
                            cache_hit=True,
                        )
                    )
                else:
                    cost = catalog.cost(target.id, resp.usage)
                    await budgets.add(ctx, cost or 0.0)
                    await ledger.record(
                        self._record(
                            ctx=ctx,
                            req=req,
                            request_id=request_id,
                            started=started,
                            status=status,
                            http_status=http_status,
                            target=target,
                            resp=resp,
                            cost=cost,
                            attempts=len(attempts),
                        )
                    )
                    if key and status == "ok":
                        await cache.set(key, resp, target.provider, target.model)
            yield DONE

        headers.update(self._headers(target, attempts, None))
        return StreamResult(events(), headers)
