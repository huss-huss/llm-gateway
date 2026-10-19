"""The request pipeline behind ``/v1/chat/completions``.

    resolve route -> request rate limit -> budget check -> cache lookup
      -> reserve TPM estimate -> call providers (retry/fallback/breaker)
      -> settle TPM -> price -> add spend -> ledger + metrics + log -> cache store

Every request that passes authentication produces exactly one ledger row, except requests
rejected by the rate limiters or budget check (those are counted in metrics only, so a client
hammering the gateway cannot flood the database). Cache hits never touch the TPM bucket.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from llm_gateway.auth import AuthContext
from llm_gateway.cache import cache_key, is_cacheable
from llm_gateway.config import Target
from llm_gateway.errors import GatewayHTTPError, error_body
from llm_gateway.ledger import UsageRecord
from llm_gateway.observability import metrics
from llm_gateway.observability.tracing import tracer
from llm_gateway.providers.base import ProviderError, estimate_tokens
from llm_gateway.ratelimit import TokenReservation, estimate_prompt_tokens
from llm_gateway.router import AllTargetsFailedError, Attempt, NoRouteError
from llm_gateway.schemas import (
    ChatCompletionRequest,
    ProviderResponse,
    completion_body,
    new_completion_id,
)
from llm_gateway.state import GatewayState
from llm_gateway.streaming import DONE, StreamAccumulator, encode_stream, replay, sse

log = logging.getLogger("llm_gateway.requests")


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


def _route_headers(target: Target, attempts: list[Attempt], cost: float | None) -> dict:
    h = {
        "x-gateway-provider": target.provider,
        "x-gateway-model": target.model,
        "x-gateway-attempts": str(len(attempts)),
    }
    if cost is not None:
        h["x-gateway-cost-usd"] = f"{cost:.8f}"
    return h


class ChatService:
    def __init__(self, state: GatewayState) -> None:
        self.state = state

    # ---------------------------------------------------------------- bookkeeping

    def _record(
        self,
        ctx: AuthContext,
        req: ChatCompletionRequest,
        request_id: str,
        started: float,
        *,
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
            cache_hit=cache_hit,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            cost_usd=cost or 0.0,
            latency_ms=int((time.perf_counter() - started) * 1000),
            attempts=attempts,
        )

    async def _emit(self, rec: UsageRecord, span: Span | None = None) -> None:
        """Ledger row + metrics + one structured log line + span attributes."""
        await self.state.ledger.record(rec)
        metrics.observe_completion(
            model=rec.requested_model,
            status=rec.status,
            stream=rec.stream,
            elapsed_s=rec.latency_ms / 1000,
            provider=rec.provider,
            target_model=rec.model,
            prompt_tokens=rec.prompt_tokens,
            completion_tokens=rec.completion_tokens,
            cost_usd=rec.cost_usd,
        )
        log.info(
            "chat completion",
            extra={
                "request_id": rec.request_id,
                "tenant": rec.tenant,
                "api_key_id": rec.api_key_id,
                "requested_model": rec.requested_model,
                "provider": rec.provider,
                "model": rec.model,
                "status": rec.status,
                "http_status": rec.http_status,
                "stream": rec.stream,
                "cache_hit": rec.cache_hit,
                "prompt_tokens": rec.prompt_tokens,
                "completion_tokens": rec.completion_tokens,
                "cost_usd": rec.cost_usd,
                "latency_ms": rec.latency_ms,
                "attempts": rec.attempts,
            },
        )
        if span is not None:
            span.set_attributes(
                {
                    "llm_gateway.status": rec.status,
                    "llm_gateway.cache_hit": rec.cache_hit,
                    "llm_gateway.cost_usd": rec.cost_usd,
                    "llm_gateway.attempts": rec.attempts,
                    "gen_ai.usage.input_tokens": rec.prompt_tokens,
                    "gen_ai.usage.output_tokens": rec.completion_tokens,
                }
            )
            if rec.provider:
                span.set_attribute("llm_gateway.provider", rec.provider)
                span.set_attribute("gen_ai.response.model", rec.model or "")
            if rec.http_status >= 400:
                span.set_status(Status(StatusCode.ERROR, rec.status))

    async def _fail(
        self,
        exc: Exception,
        ctx: AuthContext,
        req: ChatCompletionRequest,
        rid: str,
        started: float,
        span: Span | None,
    ) -> GatewayHTTPError:
        err = to_http_error(exc)
        attempts = len(exc.attempts) if isinstance(exc, AllTargetsFailedError) else 0
        await self._emit(
            self._record(
                ctx,
                req,
                rid,
                started,
                status=err.code or "error",
                http_status=err.status,
                attempts=attempts,
            ),
            span,
        )
        return err

    # ---------------------------------------------------------------- pre-flight

    @staticmethod
    def _bucket(ctx: AuthContext) -> str:
        return str(ctx.key_id) if ctx.key_id is not None else f"anon:{ctx.tenant}"

    async def _preflight(self, ctx: AuthContext, req: ChatCompletionRequest) -> dict[str, str]:
        """Enforce rate limit and budget before any provider is called."""
        headers: dict[str, str] = {}
        rpm = ctx.rate_limit_rpm or self.state.config.limits.default_rpm
        if rpm:
            decision = await self.state.limiter.hit(self._bucket(ctx), rpm)
            headers.update(decision.headers())
            if not decision.allowed:
                metrics.REJECTIONS.labels("rate_limited").inc()
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
                metrics.REJECTIONS.labels("budget_exceeded").inc()
                raise GatewayHTTPError(
                    402,
                    f"monthly budget of ${ctx.monthly_budget_usd} exhausted for this API key",
                    "budget_exceeded",
                    "monthly_budget_exceeded",
                    headers,
                )
            headers["x-gateway-budget-remaining-usd"] = f"{remaining:.6f}"
        return headers

    def _completion_allowance(self, req: ChatCompletionRequest) -> int:
        limits, defaults = self.state.config.limits, self.state.config.defaults
        return req.effective_max_tokens(limits.completion_token_allowance or defaults.max_tokens)

    async def _reserve_tokens(
        self, ctx: AuthContext, req: ChatCompletionRequest, headers: dict[str, str], span: Span
    ) -> TokenReservation | None:
        """Reserve the request's estimated tokens against the key's TPM bucket (atomic in
        Redis), or raise 429. Called only once a provider is actually going to be called."""
        tpm = ctx.rate_limit_tpm or self.state.config.limits.default_tpm
        if not tpm:
            return None
        prompt = estimate_prompt_tokens(req)
        try:
            if prompt > tpm:
                # Could never be admitted, however long the client waits.
                metrics.REJECTIONS.labels("token_rate_limited").inc()
                raise GatewayHTTPError(
                    429,
                    f"request too large: about {prompt} prompt tokens exceeds this API key's "
                    f"limit of {tpm} tokens/minute",
                    "rate_limit_error",
                    "request_too_large",
                    {**headers, "x-ratelimit-limit-tokens": str(tpm)},
                )
            # A request may always reserve up to the whole bucket, so a large max_tokens
            # can't lock a key out; settlement charges whatever it really used.
            cost = min(tpm, prompt + self._completion_allowance(req))
            bucket = self._bucket(ctx)
            decision = await self.state.limiter.reserve_tokens(bucket, tpm, cost)
            if not decision.allowed:
                metrics.REJECTIONS.labels("token_rate_limited").inc()
                raise GatewayHTTPError(
                    429,
                    f"rate limit of {tpm} tokens/minute exceeded for this API key "
                    f"(this request needs about {cost})",
                    "rate_limit_error",
                    "rate_limit_exceeded",
                    {**headers, **decision.headers()},
                )
        except GatewayHTTPError as err:
            span.set_attribute("llm_gateway.status", err.code or "rejected")
            span.set_status(Status(StatusCode.ERROR, err.code or "rejected"))
            raise
        headers.update(decision.headers())
        return TokenReservation(self.state.limiter, bucket, tpm, cost, prompt, decision)

    @staticmethod
    def _actual_tokens(res: TokenReservation, resp: ProviderResponse | None) -> int:
        """Tokens to charge: reported usage, else what is known (the prompt estimate plus the
        output received so far), e.g. when a stream ends early or reports no usage."""
        if resp is None:
            return 0
        if resp.usage.total_tokens:
            return resp.usage.total_tokens
        out = (resp.content or "") + "".join(
            tc["function"]["name"] + tc["function"]["arguments"] for tc in resp.tool_calls
        )
        return res.prompt_estimate + estimate_tokens(out)

    async def _settle(
        self,
        res: TokenReservation | None,
        resp: ProviderResponse | None,
        headers: dict[str, str] | None = None,
    ) -> None:
        if res is None:
            return
        decision = await res.settle(self._actual_tokens(res, resp))
        if headers is not None:
            headers.update(decision.headers())

    async def _begin(
        self,
        ctx: AuthContext,
        req: ChatCompletionRequest,
        rid: str,
        started: float,
        span: Span,
    ) -> dict[str, str]:
        try:
            self.state.router.targets(req.model)
        except NoRouteError as exc:
            raise await self._fail(exc, ctx, req, rid, started, span) from exc
        try:
            return await self._preflight(ctx, req)
        except GatewayHTTPError as err:
            span.set_attribute("llm_gateway.status", err.code or "rejected")
            span.set_status(Status(StatusCode.ERROR, err.code or "rejected"))
            raise

    def _cache_key(
        self, ctx: AuthContext, req: ChatCompletionRequest, bypass: bool
    ) -> tuple[str | None, bool]:
        """Returns ``(key, read)``; ``key`` is None when the request is not cacheable."""
        if not self.state.cache.enabled or not is_cacheable(req):
            return None, False
        return cache_key(ctx.tenant, req), not bypass

    @staticmethod
    def _span_attrs(ctx: AuthContext, req: ChatCompletionRequest, rid: str) -> dict:
        return {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": req.model,
            "llm_gateway.request_id": rid,
            "llm_gateway.tenant": ctx.tenant,
            "llm_gateway.stream": req.stream,
        }

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
        with tracer.start_as_current_span(
            "chat.completions", attributes=self._span_attrs(ctx, req, request_id)
        ) as span:
            headers = await self._begin(ctx, req, request_id, started, span)
            key, read = self._cache_key(ctx, req, cache_bypass)

            if key and read and (hit := await self.state.cache.get(key)):
                metrics.CACHE.labels("hit").inc()
                target = Target(provider=hit.provider, model=hit.model)
                await self._emit(
                    self._record(
                        ctx,
                        req,
                        request_id,
                        started,
                        status="ok",
                        http_status=200,
                        target=target,
                        cache_hit=True,
                    ),
                    span,
                )
                payload = completion_body(
                    completion_id=new_completion_id(), model=req.model, resp=hit.response
                )
                headers.update(_route_headers(target, [], 0.0))
                headers["x-gateway-cache"] = "hit"
                return ChatResult(payload, headers)

            metrics.CACHE.labels("miss" if key else "bypass").inc()
            res = await self._reserve_tokens(ctx, req, headers, span)
            resp: ProviderResponse | None = None
            try:
                result = await self.state.router.complete(req)
                resp = result.response
            except (NoRouteError, ProviderError) as exc:
                raise await self._fail(exc, ctx, req, request_id, started, span) from exc
            finally:
                # Failed calls report no usage, so their reservation is refunded in full.
                await self._settle(res, resp, headers)

            cost = self.state.catalog.cost(result.target.id, result.response.usage)
            # Spend must be added before the ledger row is written (see BudgetTracker._ensure).
            await self.state.budgets.add(ctx, cost or 0.0)
            await self._emit(
                self._record(
                    ctx,
                    req,
                    request_id,
                    started,
                    status="ok",
                    http_status=200,
                    target=result.target,
                    resp=result.response,
                    cost=cost,
                    attempts=len(result.attempts),
                ),
                span,
            )
            if key:
                await self.state.cache.set(
                    key, result.response, result.target.provider, result.target.model
                )
            payload = completion_body(
                completion_id=new_completion_id(), model=req.model, resp=result.response
            )
            headers.update(_route_headers(result.target, result.attempts, cost))
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
        # The root span must outlive this function (it ends when the stream ends), so it is
        # managed manually rather than with a context manager.
        span = tracer.start_span(
            "chat.completions", attributes=self._span_attrs(ctx, req, request_id)
        )
        token = otel_context.attach(trace.set_span_in_context(span))
        res: TokenReservation | None = None
        try:
            headers = await self._begin(ctx, req, request_id, started, span)
            key, read = self._cache_key(ctx, req, cache_bypass)
            hit = await self.state.cache.get(key) if key and read else None
            if hit is not None:
                metrics.CACHE.labels("hit").inc()
                target = Target(provider=hit.provider, model=hit.model)
                attempts: list[Attempt] = []
                deltas = replay(hit.response)
                headers["x-gateway-cache"] = "hit"
            else:
                metrics.CACHE.labels("miss" if key else "bypass").inc()
                res = await self._reserve_tokens(ctx, req, headers, span)
                try:
                    route = await self.state.router.open_stream(req)
                except BaseException as exc:
                    await self._settle(res, None)  # nothing was generated: full refund
                    if isinstance(exc, NoRouteError | ProviderError):
                        raise await self._fail(exc, ctx, req, request_id, started, span) from exc
                    raise
                target, attempts, deltas = route.target, route.attempts, route.deltas
                headers["x-gateway-cache"] = "miss" if key else "bypass"
        except BaseException:
            span.end()
            raise
        finally:
            otel_context.detach(token)

        acc = StreamAccumulator()
        include_usage = bool((req.stream_options or {}).get("include_usage"))

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
                try:
                    await self._finish_stream(
                        ctx,
                        req,
                        request_id,
                        started,
                        span,
                        acc,
                        target,
                        attempts,
                        key,
                        hit is not None,
                        status,
                        http_status,
                        res,
                    )
                finally:
                    span.end()
            yield DONE

        headers.update(_route_headers(target, attempts, None))
        return StreamResult(events(), headers)

    async def _finish_stream(
        self,
        ctx: AuthContext,
        req: ChatCompletionRequest,
        request_id: str,
        started: float,
        span: Span,
        acc: StreamAccumulator,
        target: Target,
        attempts: list[Attempt],
        key: str | None,
        cache_hit: bool,
        status: str,
        http_status: int,
        res: TokenReservation | None = None,
    ) -> None:
        if cache_hit:
            rec = self._record(
                ctx,
                req,
                request_id,
                started,
                status=status,
                http_status=http_status,
                target=target,
                cache_hit=True,
            )
        else:
            resp = acc.response()
            # Settles from the final usage chunk; after a disconnect or upstream error, from
            # what was received (headers are already sent, so the result isn't reported).
            await self._settle(res, resp)
            cost = self.state.catalog.cost(target.id, resp.usage)
            await self.state.budgets.add(ctx, cost or 0.0)
            rec = self._record(
                ctx,
                req,
                request_id,
                started,
                status=status,
                http_status=http_status,
                target=target,
                resp=resp,
                cost=cost,
                attempts=len(attempts),
            )
            if key and status == "ok":
                await self.state.cache.set(key, resp, target.provider, target.model)
        await self._emit(rec, span)
