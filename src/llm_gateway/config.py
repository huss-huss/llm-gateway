"""Load the gateway YAML (providers, aliases, routing policy) with ``${ENV:-default}`` expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in strings."""
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


class Target(BaseModel):
    provider: str
    model: str

    @property
    def id(self) -> str:
        return f"{self.provider}/{self.model}"

    @classmethod
    def parse(cls, spec: str) -> Target:
        provider, sep, model = spec.partition("/")
        if not sep or not provider or not model:
            raise ValueError(f"target must look like 'provider/model', got {spec!r}")
        return cls(provider=provider, model=model)


class BreakerConfig(BaseModel):
    failure_threshold: int = 5
    cooldown_s: float = 30.0


class RoutingConfig(BaseModel):
    retries: int = 2  # extra attempts per target on retryable errors
    backoff_base_s: float = 0.25
    backoff_max_s: float = 4.0
    breaker: BreakerConfig = Field(default_factory=BreakerConfig)


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl_s: int = 3600


class LimitsConfig(BaseModel):
    default_rpm: int | None = 600  # applies to keys without their own rate_limit_rpm


class Defaults(BaseModel):
    max_tokens: int = 1024
    timeout_s: float = 60.0


class GatewayConfig(BaseModel):
    defaults: Defaults = Field(default_factory=Defaults)
    providers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    aliases: dict[str, list[Target]] = Field(default_factory=dict)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)

    @field_validator("aliases", mode="before")
    @classmethod
    def _parse_targets(cls, v: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return {
            alias: [Target.parse(t) if isinstance(t, str) else t for t in targets]
            for alias, targets in (v or {}).items()
        }

    def provider_enabled(self, name: str) -> bool:
        """A provider is enabled unless ``enabled`` is false or its API key env var is empty."""
        cfg = self.providers.get(name)
        if cfg is None:
            return False
        enabled = cfg.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in ("1", "true", "yes", "on")
        if not enabled:
            return False
        key_env = cfg.get("api_key_env")
        return not (key_env and not os.environ.get(key_env))

    def resolve(self, model: str) -> list[Target]:
        """Resolve a requested model (alias or ``provider/model``) to an ordered target list."""
        if model in self.aliases:
            return list(self.aliases[model])
        if "/" in model:
            target = Target.parse(model)
            if target.provider in self.providers:
                return [target]
        return []


def load_gateway_config(path: str | Path) -> GatewayConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return GatewayConfig.model_validate(expand_env(raw))
