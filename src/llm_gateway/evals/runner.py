"""Run a suite against a model alias in-process (same router, providers, and pricing as the
server, but no auth, rate limits, or cache) and optionally persist the results."""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from llm_gateway.catalog import Catalog
from llm_gateway.db.models import EvalResult, EvalRun
from llm_gateway.evals.assertions import check
from llm_gateway.evals.suite import Case, Suite
from llm_gateway.providers.base import ProviderError
from llm_gateway.router import NoRouteError, Router
from llm_gateway.schemas import ChatCompletionRequest


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    latency_ms: int
    cost_usd: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    provider: str | None = None
    model: str | None = None
    output: str | None = None
    failures: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class RunReport:
    suite: str
    model: str
    results: list[CaseResult]
    started_at: datetime
    finished_at: datetime
    run_id: int | None = None
    judge_cost_usd: float = 0.0

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed / len(self.results) if self.results else 0.0

    @property
    def total_cost_usd(self) -> float:
        return round(sum(r.cost_usd for r in self.results), 8)

    @property
    def avg_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.results) / len(self.results) if self.results else 0

    @property
    def p95_latency_ms(self) -> float:
        lat = sorted(r.latency_ms for r in self.results)
        return float(lat[max(0, math.ceil(0.95 * len(lat)) - 1)]) if lat else 0.0

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "suite": self.suite,
            "model": self.model,
            "cases": len(self.results),
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "total_cost_usd": self.total_cost_usd,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p95_latency_ms": self.p95_latency_ms,
            "judge_cost_usd": round(self.judge_cost_usd, 8),
        }


class EvalRunner:
    def __init__(
        self,
        router: Router,
        catalog: Catalog,
        *,
        judge_model: str = "smart",
        concurrency: int = 4,
    ) -> None:
        self.router = router
        self.catalog = catalog
        self.judge_model = judge_model
        self.sem = asyncio.Semaphore(max(1, concurrency))
        self.judge_cost = 0.0

    async def _judge(self, model: str, prompt: str) -> str:
        req = ChatCompletionRequest(
            model=model,
            temperature=0,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        result = await self.router.complete(req)
        self.judge_cost += self.catalog.cost(result.target.id, result.response.usage) or 0.0
        return result.response.content or ""

    async def run_case(self, suite: Suite, case: Case, model: str) -> CaseResult:
        params = {**suite.defaults, **case.params}
        req = ChatCompletionRequest.model_validate(
            {
                "model": model,
                "messages": case.build_messages(),
                **params,
                **({"response_format": case.response_format} if case.response_format else {}),
            }
        )
        async with self.sem:
            started = time.perf_counter()
            try:
                result = await self.router.complete(req)
            except (ProviderError, NoRouteError) as exc:
                return CaseResult(
                    case.id,
                    False,
                    int((time.perf_counter() - started) * 1000),
                    0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            latency_ms = int((time.perf_counter() - started) * 1000)
            resp = result.response
            output = resp.content or ""
            if not output and resp.tool_calls:
                output = json.dumps(resp.tool_calls)
            failures = []
            for a in case.assert_:
                msg = await check(a, output, self._judge, self.judge_model)
                if msg:
                    failures.append(msg)
        return CaseResult(
            case_id=case.id,
            passed=not failures,
            latency_ms=latency_ms,
            cost_usd=self.catalog.cost(result.target.id, resp.usage) or 0.0,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            provider=result.target.provider,
            model=result.target.model,
            output=output,
            failures=failures,
        )

    async def run(self, suite: Suite, model: str) -> RunReport:
        started = datetime.now(UTC)
        results = await asyncio.gather(*(self.run_case(suite, c, model) for c in suite.cases))
        return RunReport(
            suite.name,
            model,
            list(results),
            started,
            datetime.now(UTC),
            judge_cost_usd=self.judge_cost,
        )


async def store_report(sessionmaker, report: RunReport) -> int:
    async with sessionmaker() as session:
        run = EvalRun(
            suite=report.suite,
            model=report.model,
            cases=len(report.results),
            passed=report.passed,
            pass_rate=report.pass_rate,
            total_cost_usd=Decimal(str(report.total_cost_usd)),
            avg_latency_ms=report.avg_latency_ms,
            p95_latency_ms=report.p95_latency_ms,
            started_at=report.started_at,
            finished_at=report.finished_at,
        )
        session.add(run)
        await session.flush()
        for r in report.results:
            session.add(
                EvalResult(
                    run_id=run.id,
                    case_id=r.case_id,
                    passed=r.passed,
                    provider=r.provider,
                    model=r.model,
                    latency_ms=r.latency_ms,
                    cost_usd=Decimal(str(r.cost_usd)),
                    prompt_tokens=r.prompt_tokens,
                    completion_tokens=r.completion_tokens,
                    output=r.output,
                    failures=json.dumps(r.failures),
                    error=r.error,
                )
            )
        await session.commit()
        report.run_id = run.id
        return run.id


def format_report(report: RunReport) -> str:
    lines = [f"Suite: {report.suite}   model: {report.model}", ""]
    width = max(len(r.case_id) for r in report.results)
    for r in report.results:
        mark = "PASS" if r.passed else "FAIL"
        target = f"{r.provider}/{r.model}" if r.provider else "-"
        lines.append(
            f"  {mark}  {r.case_id:<{width}}  {r.latency_ms:>6} ms  ${r.cost_usd:.6f}  {target}"
        )
        for f in r.failures:
            lines.append(f"        - {f}")
        if r.error:
            lines.append(f"        - error: {r.error}")
    s = report.summary()
    lines += [
        "",
        f"Passed {s['passed']}/{s['cases']} ({s['pass_rate']:.1%})   "
        f"avg latency {s['avg_latency_ms']} ms   p95 {s['p95_latency_ms']:.0f} ms   "
        f"cost ${s['total_cost_usd']:.6f}"
        + (f" (+ judge ${s['judge_cost_usd']:.6f})" if s["judge_cost_usd"] else ""),
    ]
    if report.run_id is not None:
        lines.append(f"Stored as eval run #{report.run_id}")
    return "\n".join(lines)
