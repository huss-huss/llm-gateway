from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from llm_gateway.config import GatewayConfig, expand_env
from llm_gateway.db.models import EvalResult, EvalRun
from llm_gateway.evals.assertions import check
from llm_gateway.evals.runner import EvalRunner, format_report, store_report
from llm_gateway.evals.suite import Assertion, Suite, load_suite
from llm_gateway.router import Router
from llm_gateway.state import build_providers

ROOT = Path(__file__).resolve().parents[1]


def A(**kw) -> Assertion:
    return Assertion.model_validate(kw)


@pytest.mark.parametrize(
    ("assertion", "output", "passes"),
    [
        (A(type="exact", value="4"), " 4\n", True),
        (A(type="exact", value="4"), "4.", False),
        (A(type="exact", value="Paris", case_sensitive=True), "paris", False),
        (A(type="contains", value="paris"), "It is Paris.", True),
        (A(type="contains", value="Rome"), "It is Paris.", False),
        (A(type="not_contains", value="rome"), "It is Paris.", True),
        (A(type="regex", pattern=r"\b18\d\d\b"), "born 1815", True),
        (A(type="regex", pattern=r"^\d+$"), "abc", False),
        (A(type="json_valid"), '{"a": 1}', True),
        (A(type="json_valid"), '```json\n{"a": 1}\n```', True),
        (A(type="json_valid"), "{nope", False),
        (A(type="json_schema", schema={"type": "object", "required": ["a"]}), '{"a": 1}', True),
        (A(type="json_schema", schema={"type": "object", "required": ["b"]}), '{"a": 1}', False),
    ],
)
async def test_assertions(assertion, output, passes):
    assert (await check(assertion, output, None, "smart") is None) is passes


async def test_llm_judge_assertion():
    async def judge(model, prompt):
        assert model == "smart" and "RUBRIC" in prompt
        return "PASS\nlooks right" if "Paris" in prompt else "FAIL: wrong city"

    a = A(type="llm_judge", rubric="names the capital")
    assert await check(a, "Paris", judge, "smart") is None
    assert "FAIL" in await check(a, "Rome", judge, "smart")
    assert await check(a, "x", None, "smart") == "llm_judge assertion needs a judge model"


def test_suite_validation():
    with pytest.raises(ValidationError):
        A(type="contains")  # missing value
    with pytest.raises(ValidationError):
        Suite.model_validate(
            {
                "name": "s",
                "cases": [
                    {"id": "a", "prompt": "x", "messages": [], "assert": [{"type": "json_valid"}]}
                ],
            }
        )
    with pytest.raises(ValidationError):
        Suite.model_validate(
            {
                "name": "s",
                "cases": [
                    {"id": "a", "prompt": "x", "assert": [{"type": "json_valid"}]},
                    {"id": "a", "prompt": "y", "assert": [{"type": "json_valid"}]},
                ],
            }
        )


@pytest.fixture
def eval_router(monkeypatch) -> Router:
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OLLAMA_ENABLED"):
        monkeypatch.delenv(var, raising=False)
    import yaml

    raw = yaml.safe_load((ROOT / "config" / "gateway.yaml").read_text())
    raw["providers"]["fake"]["latency_ms"] = 0
    cfg = GatewayConfig.model_validate(expand_env(raw))
    return Router(cfg, build_providers(cfg))


async def test_example_suite_runs_on_fake_provider(state, eval_router):
    suite = load_suite(ROOT / "evals" / "suite.yaml")
    runner = EvalRunner(eval_router, state.catalog)
    report = await runner.run(suite, "fast")
    by_id = {r.case_id: r for r in report.results}
    assert report.passed == 5 and len(report.results) == 6
    assert not by_id["translate-hello"].passed
    assert by_id["translate-hello"].failures == ["expected output to contain 'bonjour'"]
    assert all(r.provider == "fake" for r in report.results)
    assert report.total_cost_usd > 0 and report.judge_cost_usd > 0

    run_id = await store_report(state.sessionmaker, report)
    async with state.sessionmaker() as s:
        run = await s.get(EvalRun, run_id)
        rows = (await s.scalars(select(EvalResult).where(EvalResult.run_id == run_id))).all()
    assert (run.cases, run.passed) == (6, 5)
    assert run.pass_rate == pytest.approx(5 / 6)
    assert len(rows) == 6
    text = format_report(report)
    assert "Passed 5/6 (83.3%)" in text and f"eval run #{run_id}" in text


async def test_unknown_model_is_reported_as_error(state, eval_router):
    suite = Suite.model_validate(
        {"name": "s", "cases": [{"id": "a", "prompt": "x", "assert": [{"type": "json_valid"}]}]}
    )
    report = await EvalRunner(eval_router, state.catalog).run(suite, "does-not-exist")
    assert not report.results[0].passed
    assert "NoRouteError" in report.results[0].error
