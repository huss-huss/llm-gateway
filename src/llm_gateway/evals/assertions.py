"""Assertion checks. Each returns ``None`` on pass or a human-readable failure message."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable

import jsonschema

from llm_gateway.evals.suite import Assertion

JUDGE_PROMPT = """You are an impartial grader. Decide whether the RESPONSE satisfies the RUBRIC.
Reply with exactly one word on the first line: PASS or FAIL. Optionally add a one-line reason.

RUBRIC:
{rubric}

RESPONSE:
{response}"""

# A judge receives a prompt and returns the judge model's reply text.
Judge = Callable[[str, str], Awaitable[str]]


def _parse_json(text: str):
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    return json.loads(fence.group(1) if fence else stripped)


async def check(a: Assertion, output: str, judge: Judge | None, default_judge: str) -> str | None:
    flags = 0 if a.case_sensitive else re.IGNORECASE
    norm = (lambda s: s) if a.case_sensitive else str.lower
    if a.type == "exact":
        if norm(output.strip()) != norm(a.value.strip()):
            return f"expected exactly {a.value!r}, got {output.strip()[:120]!r}"
    elif a.type == "contains":
        if norm(a.value) not in norm(output):
            return f"expected output to contain {a.value!r}"
    elif a.type == "not_contains":
        if norm(a.value) in norm(output):
            return f"expected output not to contain {a.value!r}"
    elif a.type == "regex":
        if not re.search(a.pattern, output, flags=flags):
            return f"expected output to match /{a.pattern}/"
    elif a.type in ("json_valid", "json_schema"):
        try:
            data = _parse_json(output)
        except (json.JSONDecodeError, ValueError) as exc:
            return f"output is not valid JSON ({getattr(exc, 'msg', exc)})"
        if a.type == "json_schema":
            try:
                jsonschema.validate(data, a.schema_)
            except jsonschema.ValidationError as exc:
                return f"JSON does not match schema: {exc.message}"
    elif a.type == "llm_judge":
        if judge is None:
            return "llm_judge assertion needs a judge model"
        model = a.model or default_judge
        verdict = await judge(model, JUDGE_PROMPT.format(rubric=a.rubric, response=output))
        first = (verdict.strip().splitlines() or [""])[0].strip().upper()
        if not first.startswith("PASS"):
            return f"judge ({model}) said: {verdict.strip()[:200]!r}"
    return None
