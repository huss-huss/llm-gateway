from __future__ import annotations

from pathlib import Path

from llm_gateway.config import load_gateway_config
from llm_gateway.state import build_providers

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_config_loads_and_falls_back_to_fake_without_keys(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OLLAMA_ENABLED"):
        monkeypatch.delenv(var, raising=False)
    cfg = load_gateway_config(ROOT / "config" / "gateway.yaml")
    providers = build_providers(cfg)
    assert set(providers) == {"fake"}
    assert {"fast", "smart", "balanced", "local"} <= set(cfg.aliases)
    assert cfg.aliases["smart"][0].id == "anthropic/claude-opus-5-5"


def test_shipped_config_enables_providers_with_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "y")
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    cfg = load_gateway_config(ROOT / "config" / "gateway.yaml")
    assert set(build_providers(cfg)) == {"anthropic", "openai", "ollama", "fake"}
