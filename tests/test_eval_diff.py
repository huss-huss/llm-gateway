from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pytest

from llm_gateway.cli import main
from llm_gateway.db.models import Base
from llm_gateway.db.session import make_engine, make_sessionmaker
from llm_gateway.evals.diff import RunNotFoundError, diff_runs, format_diff, load_run
from llm_gateway.evals.runner import CaseResult, RunReport, store_report

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def report(cases: dict[str, tuple[bool, int, float]], suite="smoke", model="fast") -> RunReport:
    """A fabricated run: ``{case_id: (passed, latency_ms, cost_usd)}``."""
    results = [
        CaseResult(cid, passed, latency, cost, provider="fake", model="fake-fast")
        for cid, (passed, latency, cost) in cases.items()
    ]
    return RunReport(suite, model, results, T0, T0)


async def stored_diff(sessionmaker, a: RunReport, b: RunReport):
    ra, rb = await store_report(sessionmaker, a), await store_report(sessionmaker, b)
    return diff_runs(await load_run(sessionmaker, str(ra)), await load_run(sessionmaker, str(rb)))


BASELINE = {
    "gets-fixed": (False, 100, 0.001),
    "regresses": (True, 100, 0.001),
    "keeps-failing": (False, 100, 0.001),
    "keeps-passing": (True, 100, 0.001),
    "gets-removed": (True, 100, 0.001),
}
CANDIDATE = {
    "gets-fixed": (True, 120, 0.002),
    "regresses": (False, 80, 0.0005),
    "keeps-failing": (False, 100, 0.001),
    "keeps-passing": (True, 300, 0.001),
    "is-new": (False, 50, 0.003),
}


async def test_every_case_status(state):
    d = await stored_diff(state.sessionmaker, report(BASELINE), report(CANDIDATE))
    status = {c.case_id: c.status for c in d.cases}
    assert status == {
        "gets-fixed": "fixed",
        "regresses": "regressed",
        "keeps-failing": "still_failing",
        "keeps-passing": "unchanged",
        "gets-removed": "removed",
        "is-new": "new",
    }
    # Regressions are listed first, unchanged cases last.
    assert [c.status for c in d.cases] == [
        "regressed",
        "fixed",
        "still_failing",
        "new",
        "removed",
        "unchanged",
    ]
    assert d.counts() == {
        "regressed": 1,
        "fixed": 1,
        "still_failing": 1,
        "new": 1,
        "removed": 1,
        "unchanged": 1,
    }
    by_id = {c.case_id: c for c in d.cases}
    assert by_id["gets-fixed"].latency_delta_ms == 20
    assert by_id["gets-fixed"].cost_delta_usd == pytest.approx(0.001)
    assert by_id["regresses"].latency_delta_ms == -20
    assert by_id["is-new"].latency_delta_ms is None and by_id["is-new"].a is None
    assert by_id["gets-removed"].cost_delta_usd is None and by_id["gets-removed"].b is None
    assert d.warnings == []


async def test_run_level_deltas(state):
    d = await stored_diff(state.sessionmaker, report(BASELINE), report(CANDIDATE))
    out = d.to_dict()
    # A: 3/5 passed, B: 2/5 passed.
    assert out["run_a"]["pass_rate"] == 0.6 and out["run_b"]["pass_rate"] == 0.4
    assert out["deltas"]["pass_rate"] == pytest.approx(-0.2)
    assert out["deltas"]["passed"] == -1
    assert out["deltas"]["avg_latency_ms"] == pytest.approx(130 - 100)
    assert out["deltas"]["p95_latency_ms"] == pytest.approx(300 - 100)
    assert out["deltas"]["total_cost_usd"] == pytest.approx(0.0075 - 0.005)
    assert out["gate"] == {
        "fail_on": "regression",
        "passed": False,
        "reasons": ["1 case regressed", "pass rate dropped 60.0% -> 40.0%"],
    }
    assert len(out["cases"]) == 6
    json.dumps(out)  # serializable as-is


# (baseline, candidate, gate fails under: regression, pass-rate, never)
GATE_SCENARIOS = {
    # One case regressed, another was fixed: pass rate unchanged.
    "swap": (
        {"a": (True, 10, 0.0), "b": (False, 10, 0.0)},
        {"a": (False, 10, 0.0), "b": (True, 10, 0.0)},
        (True, False, False),
    ),
    # No case regressed, but a new failing case lowers the pass rate.
    "new-failure": (
        {"a": (True, 10, 0.0)},
        {"a": (True, 10, 0.0), "b": (False, 10, 0.0)},
        (True, True, False),
    ),
    # Regressed and the pass rate dropped.
    "regression": (
        {"a": (True, 10, 0.0), "b": (True, 10, 0.0)},
        {"a": (False, 10, 0.0), "b": (True, 10, 0.0)},
        (True, True, False),
    ),
    # Strictly better.
    "improvement": (
        {"a": (False, 10, 0.0), "b": (True, 10, 0.0)},
        {"a": (True, 10, 0.0), "b": (True, 10, 0.0)},
        (False, False, False),
    ),
}


@pytest.mark.parametrize("scenario", list(GATE_SCENARIOS))
async def test_fail_on_policies(state, scenario):
    a, b, expected = GATE_SCENARIOS[scenario]
    d = await stored_diff(state.sessionmaker, report(a), report(b))
    got = tuple(bool(d.gate_failures(p)) for p in ("regression", "pass-rate", "never"))
    assert got == expected


async def test_latest_previous_and_suite_scope(state):
    sm = state.sessionmaker
    first = await store_report(sm, report({"a": (True, 1, 0.0)}, suite="smoke"))
    other = await store_report(sm, report({"a": (True, 1, 0.0)}, suite="other"))
    last = await store_report(sm, report({"a": (False, 1, 0.0)}, suite="smoke"))

    assert (await load_run(sm, "latest")).id == last
    assert (await load_run(sm, "previous")).id == other
    assert (await load_run(sm, "previous", suite="smoke")).id == first
    assert (await load_run(sm, "latest", suite="other")).id == other
    assert (await load_run(sm, f"#{first}")).results["a"].passed is True

    with pytest.raises(RunNotFoundError, match="fewer than two"):
        await load_run(sm, "previous", suite="other")
    with pytest.raises(RunNotFoundError, match="no stored eval runs for suite 'nope'"):
        await load_run(sm, "latest", suite="nope")
    with pytest.raises(RunNotFoundError, match="not found"):
        await load_run(sm, "999999")
    with pytest.raises(RunNotFoundError, match="not a run id"):
        await load_run(sm, "yesterday")


async def test_different_suites_and_models_warn_and_match_by_case_id(state):
    a = report({"shared": (True, 10, 0.0), "only-a": (True, 10, 0.0)}, "smoke", "fast")
    b = report({"shared": (False, 10, 0.0), "only-b": (True, 10, 0.0)}, "nightly", "smart")
    d = await stored_diff(state.sessionmaker, a, b)
    assert d.warnings == [
        "runs are from different suites ('smoke' vs 'nightly')",
        "runs used different models ('fast' vs 'smart')",
        "cases are matched by case id",
    ]
    assert {c.case_id: c.status for c in d.cases} == {
        "shared": "regressed",
        "only-a": "removed",
        "only-b": "new",
    }
    run = await load_run(state.sessionmaker, "latest")
    assert diff_runs(run, run).warnings == [f"comparing eval run #{run.id} with itself"]


async def test_format_diff_text(state):
    d = await stored_diff(state.sessionmaker, report(BASELINE), report(CANDIDATE))
    text = format_diff(d)
    lines = text.splitlines()
    assert lines[0] == f"Eval run #{d.a.id} (smoke, fast)  ->  #{d.b.id} (smoke, fast)"
    row = next(line for line in lines if "regresses" in line)
    assert row.split()[:4] == ["regressed", "regresses", "PASS", "FAIL"]
    assert "100 -> 80 (-20)" in row and "-$0.000500" in row
    new = next(line for line in lines if "is-new" in line)
    assert new.split()[:4] == ["new", "is-new", "-", "FAIL"]
    assert any(line.split()[:4] == ["pass", "rate", "60.0%", "40.0%"] for line in lines)
    assert "-20.0 pp" in text and "+$0.002500" in text
    assert "1 regressed, 1 fixed, 1 still failing, 1 new, 1 removed, 1 unchanged" in text
    assert lines[-1] == (
        "FAIL (--fail-on regression): 1 case regressed; pass rate dropped 60.0% -> 40.0%"
    )
    assert format_diff(d, "never").splitlines()[-1] == "OK (--fail-on never)"


def test_cli_eval_diff(settings, capsys):
    url = settings.database_url

    async def seed() -> list[int]:
        engine = make_engine(url)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)
            sm = make_sessionmaker(engine)
            return [
                await store_report(sm, report(BASELINE)),
                await store_report(sm, report(CANDIDATE)),
                await store_report(sm, report(CANDIDATE)),
            ]
        finally:
            await engine.dispose()

    first, second, third = asyncio.run(seed())

    def run(*args: str) -> tuple[int, str, str]:
        code = main(["eval", "diff", *args, "--url", url])
        out, err = capsys.readouterr()
        return code, out, err

    code, out, _ = run(str(first), str(second))
    assert code == 1
    assert f"Eval run #{first}" in out and "FAIL (--fail-on regression)" in out

    code, out, _ = run(str(first), str(second), "--fail-on", "never")
    assert code == 0 and "OK (--fail-on never)" in out

    code, out, _ = run("previous", "latest", "--format", "json")
    assert code == 0
    payload = json.loads(out)
    assert (payload["run_a"]["run_id"], payload["run_b"]["run_id"]) == (second, third)
    assert payload["counts"]["unchanged"] == 2 and payload["gate"]["passed"] is True

    code, out, _ = run(str(first), "latest", "--format", "json", "--fail-on", "pass-rate")
    assert code == 1
    assert json.loads(out)["gate"]["reasons"] == ["pass rate dropped 60.0% -> 40.0%"]

    code, out, err = run(str(first), "999999")
    assert code == 2 and out == "" and "eval run #999999 not found" in err
