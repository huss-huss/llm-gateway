from __future__ import annotations

from pathlib import Path

import pytest

from llm_gateway.catalog import Catalog, load_catalog
from llm_gateway.schemas import Usage

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return load_catalog(ROOT / "config" / "models.yaml")


def test_shipped_catalog_has_date_and_core_models(catalog):
    assert catalog.as_of
    for model in (
        "anthropic/claude-opus-5-5",
        "anthropic/claude-sonnet-5-5",
        "anthropic/claude-haiku-5-5",
        "openai/gpt-4.1-mini",
    ):
        assert catalog.get(model) is not None, model


def test_basic_cost(catalog):
    # 1M in @ $4 + 1M out @ $20
    assert catalog.cost("anthropic/claude-opus-5-5", Usage(1_000_000, 1_000_000)) == 24.0
    # 1,000 in @ $2/M + 500 out @ $10/M = 0.002 + 0.005
    assert catalog.cost("anthropic/claude-sonnet-5-5", Usage(1000, 500)) == pytest.approx(0.007)


def test_cached_tokens_use_cache_rate(catalog):
    # 1M prompt of which 800k cached: 200k * 0.40 + 800k * 0.10 = 0.08 + 0.08
    cost = catalog.cost("openai/gpt-4.1-mini", Usage(1_000_000, 0, cached_prompt_tokens=800_000))
    assert cost == pytest.approx(0.16)


def test_long_context_tier(catalog):
    short = catalog.cost("anthropic/claude-haiku-5-5", Usage(100_000, 1000))
    long = catalog.cost("anthropic/claude-haiku-5-5", Usage(100_001, 1000))
    assert short == pytest.approx(100_000 * 0.10 / 1e6 + 1000 * 0.50 / 1e6)
    assert long == pytest.approx(100_001 * 0.50 / 1e6 + 1000 * 2.50 / 1e6)


def test_wildcard_and_unknown(catalog):
    assert catalog.cost("ollama/llama3.2", Usage(5000, 5000)) == 0.0
    assert catalog.cost("fake/anything", Usage(1_000_000, 0)) == 0.5
    assert catalog.cost("mystery/model", Usage(10, 10)) is None


def test_capabilities(catalog):
    assert catalog.capability("anthropic/claude-opus-5-5", "sampling", True) is False
    assert catalog.capability("anthropic/claude-haiku-5-5", "server_fallback") is False
    assert catalog.capability("openai/gpt-4.1", "sampling", True) is True
