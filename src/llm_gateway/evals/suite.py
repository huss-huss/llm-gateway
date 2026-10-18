"""Eval suite file format (YAML).

name: smoke
defaults: { temperature: 0, max_tokens: 200 }
cases:
  - id: capital
    prompt: "What is the capital of France?"      # or `messages: [...]`, optional `system`
    assert:
      - { type: contains, value: Paris }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Assertion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "exact", "contains", "not_contains", "regex", "json_valid", "json_schema", "llm_judge"
    ]
    value: str | None = None
    pattern: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    case_sensitive: bool = False
    rubric: str | None = None
    model: str | None = None  # judge model alias for llm_judge

    @model_validator(mode="after")
    def _required_fields(self) -> Assertion:
        need = {
            "exact": "value",
            "contains": "value",
            "not_contains": "value",
            "regex": "pattern",
            "json_schema": "schema_",
            "llm_judge": "rubric",
        }
        field = need.get(self.type)
        if field and getattr(self, field) is None:
            raise ValueError(f"assertion {self.type!r} requires {field.rstrip('_')!r}")
        return self


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    prompt: str | None = None
    system: str | None = None
    messages: list[dict[str, Any]] | None = None
    params: dict[str, Any] = Field(default_factory=dict)  # temperature, max_tokens, ...
    response_format: dict[str, Any] | None = None
    assert_: list[Assertion] = Field(alias="assert", min_length=1)

    @model_validator(mode="after")
    def _prompt_or_messages(self) -> Case:
        if (self.prompt is None) == (self.messages is None):
            raise ValueError(f"case {self.id!r}: give exactly one of `prompt` or `messages`")
        return self

    def build_messages(self) -> list[dict[str, Any]]:
        msgs = list(self.messages or [{"role": "user", "content": self.prompt}])
        if self.system:
            msgs.insert(0, {"role": "system", "content": self.system})
        return msgs


class Suite(BaseModel):
    name: str
    description: str | None = None
    defaults: dict[str, Any] = Field(default_factory=dict)
    cases: list[Case] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> Suite:
        ids = [c.id for c in self.cases]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate case ids: {sorted(dupes)}")
        return self


def load_suite(path: str | Path) -> Suite:
    return Suite.model_validate(yaml.safe_load(Path(path).read_text()))
