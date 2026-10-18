"""The request pipeline behind ``/v1/chat/completions``.

authenticate -> resolve route -> call providers (retry/fallback) -> price -> record usage
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from llm_gateway.auth import AuthContext
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
from llm_gateway.streaming import DONE, StreamAccumulator, encode_stream, sse


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

    # ---------------------------------------------------------------- entry points

    async def complete(
        self, ctx: AuthContext, req: ChatCompletionRequest, request_id: str
    ) -> ChatResult:
        started = time.perf_counter()
        pre_headers = await self._begin(ctx, req, request_id, started)
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
        payload = completion_body(
            completion_id=new_completion_id(), model=req.model, resp=result.response
        )
        return ChatResult(
            payload, {**pre_headers, **self._headers(result.target, result.attempts, cost)}
        )

    async def stream(
        self, ctx: AuthContext, req: ChatCompletionRequest, request_id: str
    ) -> StreamResult:
        started = time.perf_counter()
        pre_headers = await self._begin(ctx, req, request_id, started)
        try:
            route = await self.state.router.open_stream(req)
        except (NoRouteError, ProviderError) as exc:
            raise await self._fail(exc, ctx, req, request_id, started) from exc

        include_usage = bool((req.stream_options or {}).get("include_usage"))
        acc = StreamAccumulator()
        catalog, ledger, budgets = self.state.catalog, self.state.ledger, self.state.budgets

        async def events() -> AsyncIterator[str]:
            status, http_status = "ok", 200
            try:
                async for line in encode_stream(
                    route.deltas,
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
            finally:
                resp = acc.response()
                cost = catalog.cost(route.target.id, resp.usage)
                await budgets.add(ctx, cost or 0.0)
                await ledger.record(
                    self._record(
                        ctx=ctx,
                        req=req,
                        request_id=request_id,
                        started=started,
                        status=status,
                        http_status=http_status,
                        target=route.target,
                        resp=resp,
                        cost=cost,
                        attempts=len(route.attempts),
                    )
                )
            yield DONE

        return StreamResult(
            events(), {**pre_headers, **self._headers(route.target, route.attempts, None)}
        )
