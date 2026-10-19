"""Compare two stored eval runs case by case (``llm-gateway eval diff``).

Cases are matched by ``case_id``. Each one gets a status:

- ``fixed``          failed in run A, passes in run B
- ``regressed``      passed in run A, fails in run B
- ``still_failing``  fails in both
- ``unchanged``      passes in both
- ``new``            only in run B
- ``removed``        only in run A

Runs from different suites or models can still be compared; the diff carries a warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import select

from llm_gateway.db.models import EvalResult, EvalRun

Status = Literal["fixed", "regressed", "still_failing", "unchanged", "new", "removed"]
FailOn = Literal["regression", "pass-rate", "never"]

# Display order: the things a reviewer has to look at first.
STATUS_ORDER: tuple[Status, ...] = (
    "regressed",
    "fixed",
    "still_failing",
    "new",
    "removed",
    "unchanged",
)
_EPS = 1e-9


class RunNotFoundError(LookupError):
    pass


@dataclass
class StoredCase:
    case_id: str
    passed: bool
    latency_ms: int
    cost_usd: float


@dataclass
class StoredRun:
    id: int
    suite: str
    model: str
    cases: int
    passed: int
    pass_rate: float
    total_cost_usd: float
    avg_latency_ms: float
    p95_latency_ms: float
    results: dict[str, StoredCase]

    def summary(self) -> dict:
        return {
            "run_id": self.id,
            "suite": self.suite,
            "model": self.model,
            "cases": self.cases,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "total_cost_usd": round(self.total_cost_usd, 8),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p95_latency_ms": self.p95_latency_ms,
        }


@dataclass
class CaseDiff:
    case_id: str
    status: Status
    a: StoredCase | None
    b: StoredCase | None

    @property
    def latency_delta_ms(self) -> int | None:
        return self.b.latency_ms - self.a.latency_ms if self.a and self.b else None

    @property
    def cost_delta_usd(self) -> float | None:
        return round(self.b.cost_usd - self.a.cost_usd, 8) if self.a and self.b else None

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "passed_a": self.a.passed if self.a else None,
            "passed_b": self.b.passed if self.b else None,
            "latency_ms_a": self.a.latency_ms if self.a else None,
            "latency_ms_b": self.b.latency_ms if self.b else None,
            "latency_delta_ms": self.latency_delta_ms,
            "cost_usd_a": self.a.cost_usd if self.a else None,
            "cost_usd_b": self.b.cost_usd if self.b else None,
            "cost_delta_usd": self.cost_delta_usd,
        }


@dataclass
class RunDiff:
    a: StoredRun
    b: StoredRun
    cases: list[CaseDiff]
    warnings: list[str] = field(default_factory=list)

    def by_status(self, status: Status) -> list[CaseDiff]:
        return [c for c in self.cases if c.status == status]

    def counts(self) -> dict[str, int]:
        return {s: len(self.by_status(s)) for s in STATUS_ORDER}

    @property
    def pass_rate_delta(self) -> float:
        return self.b.pass_rate - self.a.pass_rate

    @property
    def pass_rate_dropped(self) -> bool:
        return self.pass_rate_delta < -_EPS

    @property
    def regressed(self) -> bool:
        return bool(self.by_status("regressed"))

    def gate_failures(self, fail_on: FailOn) -> list[str]:
        """Reasons the CI gate fails under ``fail_on`` (empty list: the gate passes).

        ``regression`` fails on any regressed case or a lower pass rate, ``pass-rate`` only on a
        lower pass rate, ``never`` always passes.
        """
        reasons = []
        if fail_on == "regression" and self.regressed:
            n = len(self.by_status("regressed"))
            reasons.append(f"{n} case{'s' if n != 1 else ''} regressed")
        if fail_on in ("regression", "pass-rate") and self.pass_rate_dropped:
            reasons.append(f"pass rate dropped {self.a.pass_rate:.1%} -> {self.b.pass_rate:.1%}")
        return reasons

    def to_dict(self, fail_on: FailOn = "regression") -> dict:
        failures = self.gate_failures(fail_on)
        return {
            "run_a": self.a.summary(),
            "run_b": self.b.summary(),
            "warnings": self.warnings,
            "counts": self.counts(),
            "deltas": {
                "pass_rate": round(self.pass_rate_delta, 4),
                "passed": self.b.passed - self.a.passed,
                "avg_latency_ms": round(self.b.avg_latency_ms - self.a.avg_latency_ms, 1),
                "p95_latency_ms": round(self.b.p95_latency_ms - self.a.p95_latency_ms, 1),
                "total_cost_usd": round(self.b.total_cost_usd - self.a.total_cost_usd, 8),
            },
            "gate": {"fail_on": fail_on, "passed": not failures, "reasons": failures},
            "cases": [c.to_dict() for c in self.cases],
        }


async def load_run(sessionmaker, ref: str, suite: str | None = None) -> StoredRun:
    """Load a stored run by id, or ``latest`` / ``previous`` (optionally within ``suite``)."""
    async with sessionmaker() as session:
        if ref in ("latest", "previous"):
            q = select(EvalRun).order_by(EvalRun.id.desc())
            if suite is not None:
                q = q.where(EvalRun.suite == suite)
            q = q.offset(0 if ref == "latest" else 1).limit(1)
            run = (await session.scalars(q)).first()
            if run is None:
                scope = f" for suite {suite!r}" if suite else ""
                need = "no stored eval runs" if ref == "latest" else "fewer than two eval runs"
                raise RunNotFoundError(f"{ref}: {need}{scope}")
        else:
            try:
                run_id = int(ref.lstrip("#"))
            except ValueError:
                raise RunNotFoundError(
                    f"{ref!r} is not a run id, 'latest', or 'previous'"
                ) from None
            run = await session.get(EvalRun, run_id)
            if run is None:
                raise RunNotFoundError(f"eval run #{run_id} not found")
        rows = (
            await session.scalars(
                select(EvalResult).where(EvalResult.run_id == run.id).order_by(EvalResult.id)
            )
        ).all()
    return StoredRun(
        id=run.id,
        suite=run.suite,
        model=run.model,
        cases=run.cases,
        passed=run.passed,
        pass_rate=run.pass_rate,
        total_cost_usd=float(run.total_cost_usd),
        avg_latency_ms=run.avg_latency_ms,
        p95_latency_ms=run.p95_latency_ms,
        results={
            r.case_id: StoredCase(r.case_id, r.passed, r.latency_ms, float(r.cost_usd))
            for r in rows
        },
    )


def _status(a: StoredCase | None, b: StoredCase | None) -> Status:
    if a is None:
        return "new"
    if b is None:
        return "removed"
    if a.passed and b.passed:
        return "unchanged"
    if a.passed:
        return "regressed"
    return "fixed" if b.passed else "still_failing"


def diff_runs(a: StoredRun, b: StoredRun) -> RunDiff:
    warnings = []
    if a.id == b.id:
        warnings.append(f"comparing eval run #{a.id} with itself")
    if a.suite != b.suite:
        warnings.append(f"runs are from different suites ({a.suite!r} vs {b.suite!r})")
    if a.model != b.model:
        warnings.append(f"runs used different models ({a.model!r} vs {b.model!r})")
    if a.suite != b.suite or a.model != b.model:
        warnings.append("cases are matched by case id")

    # Keep run A's order, then append cases that only exist in run B.
    ids = list(a.results) + [c for c in b.results if c not in a.results]
    cases = []
    for cid in ids:
        ca, cb = a.results.get(cid), b.results.get(cid)
        cases.append(CaseDiff(cid, _status(ca, cb), ca, cb))
    cases.sort(key=lambda c: STATUS_ORDER.index(c.status))
    return RunDiff(a, b, cases, warnings)


def _signed(value: float, fmt: str) -> str:
    return ("+" if value > 0 else "") + format(value, fmt)


def _money(value: float) -> str:
    sign = "+" if value > 0 else "-" if value < 0 else ""
    return f"{sign}${abs(value):.6f}"


def format_diff(d: RunDiff, fail_on: FailOn = "regression") -> str:
    a, b = d.a, d.b
    lines = [
        f"Eval run #{a.id} ({a.suite}, {a.model})  ->  #{b.id} ({b.suite}, {b.model})",
    ]
    lines += [f"warning: {w}" for w in d.warnings]
    lines.append("")

    def mark(c: StoredCase | None) -> str:
        return "-" if c is None else ("PASS" if c.passed else "FAIL")

    if d.cases:
        width = max(len("case"), *(len(c.case_id) for c in d.cases))
        lines.append(
            f"  {'status':<13}  {'case':<{width}}  {'A':<4}  {'B':<4}  "
            f"{'latency ms (A -> B)':>22}  {'cost delta':>11}"
        )
        for c in d.cases:
            if c.a and c.b:
                lat = f"{c.a.latency_ms} -> {c.b.latency_ms} ({_signed(c.latency_delta_ms, 'd')})"
                cost = _money(c.cost_delta_usd)
            else:
                only = c.a or c.b
                lat, cost = f"{only.latency_ms}", "-"
            lines.append(
                f"  {c.status:<13}  {c.case_id:<{width}}  {mark(c.a):<4}  {mark(c.b):<4}  "
                f"{lat:>22}  {cost:>11}"
            )
        lines.append("")

    rows = [
        (
            "pass rate",
            f"{a.pass_rate:.1%}",
            f"{b.pass_rate:.1%}",
            f"{_signed(d.pass_rate_delta * 100, '.1f')} pp",
        ),
        (
            "passed",
            f"{a.passed}/{a.cases}",
            f"{b.passed}/{b.cases}",
            _signed(b.passed - a.passed, "d"),
        ),
        (
            "avg latency",
            f"{a.avg_latency_ms:.1f} ms",
            f"{b.avg_latency_ms:.1f} ms",
            f"{_signed(b.avg_latency_ms - a.avg_latency_ms, '.1f')} ms",
        ),
        (
            "p95 latency",
            f"{a.p95_latency_ms:.0f} ms",
            f"{b.p95_latency_ms:.0f} ms",
            f"{_signed(b.p95_latency_ms - a.p95_latency_ms, '.0f')} ms",
        ),
        (
            "cost",
            f"${a.total_cost_usd:.6f}",
            f"${b.total_cost_usd:.6f}",
            _money(b.total_cost_usd - a.total_cost_usd),
        ),
    ]
    head_a, head_b = f"run #{a.id}", f"run #{b.id}"
    lines.append(f"  {'':<12}  {head_a:>12}  {head_b:>12}  {'delta':>12}")
    for label, va, vb, delta in rows:
        lines.append(f"  {label:<12}  {va:>12}  {vb:>12}  {delta:>12}")

    counts = d.counts()
    lines += [
        "",
        ", ".join(f"{n} {s.replace('_', ' ')}" for s, n in counts.items()),
    ]
    reasons = d.gate_failures(fail_on)
    if reasons:
        lines.append(f"FAIL (--fail-on {fail_on}): " + "; ".join(reasons))
    else:
        lines.append(f"OK (--fail-on {fail_on})")
    return "\n".join(lines)
