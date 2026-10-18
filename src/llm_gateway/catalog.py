"""Model catalog: per-model prices and capability flags, loaded from ``config/models.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from llm_gateway.schemas import Usage

PER_MILLION = 1_000_000


class LongContext(BaseModel):
    threshold_tokens: int
    input: float
    output: float


class ModelInfo(BaseModel):
    input: float = 0.0  # USD per 1M prompt tokens
    output: float = 0.0  # USD per 1M completion tokens
    cache_read: float | None = None  # USD per 1M cached prompt tokens (defaults to input)
    long_context: LongContext | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)


class Catalog(BaseModel):
    as_of: str | None = None
    models: dict[str, ModelInfo] = Field(default_factory=dict)

    def get(self, target_id: str) -> ModelInfo | None:
        """Exact ``provider/model`` match first, then the ``provider/*`` wildcard."""
        if target_id in self.models:
            return self.models[target_id]
        provider = target_id.split("/", 1)[0]
        return self.models.get(f"{provider}/*")

    def capability(self, target_id: str, name: str, default: Any = None) -> Any:
        info = self.get(target_id)
        return info.capabilities.get(name, default) if info else default

    def cost(self, target_id: str, usage: Usage) -> float | None:
        """USD cost of one call, or ``None`` if the model is not in the catalog."""
        info = self.get(target_id)
        if info is None:
            return None
        in_rate, out_rate = info.input, info.output
        if info.long_context and usage.prompt_tokens > info.long_context.threshold_tokens:
            in_rate, out_rate = info.long_context.input, info.long_context.output
        cached = min(usage.cached_prompt_tokens, usage.prompt_tokens)
        cache_rate = info.cache_read if info.cache_read is not None else in_rate
        total = (
            (usage.prompt_tokens - cached) * in_rate
            + cached * cache_rate
            + usage.completion_tokens * out_rate
        ) / PER_MILLION
        return round(total, 8)


def load_catalog(path: str | Path) -> Catalog:
    p = Path(path)
    if not p.exists():
        return Catalog()
    return Catalog.model_validate(yaml.safe_load(p.read_text()) or {})
