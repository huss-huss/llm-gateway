"""Process-level settings, read from environment variables (prefix ``GATEWAY_``)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GATEWAY_", env_file=".env", extra="ignore")

    # Routing/provider configuration (YAML) and the model catalog (prices + capabilities).
    config_path: str = "config/gateway.yaml"
    models_path: str = "config/models.yaml"

    # Storage. ``memory://`` uses an in-process fakeredis (dev/demo only).
    database_url: str = "sqlite+aiosqlite:///./llm_gateway.db"
    redis_url: str = "memory://"

    # Security.
    admin_key: SecretStr | None = None
    key_pepper: SecretStr = SecretStr("dev-pepper-change-me")
    require_auth: bool = True

    # Observability.
    log_level: str = "INFO"
    log_json: bool = True
    otel_exporter: Literal["none", "console", "otlp"] = "none"
    otel_endpoint: str | None = None
    service_name: str = "llm-gateway"


def load_dotenv_file(path: str = ".env") -> None:
    """Load ``.env`` into the process environment (without overriding real env vars).

    Settings already read ``GATEWAY_*`` from ``.env``; this also exposes provider keys such as
    ``ANTHROPIC_API_KEY`` that the routing config looks up via ``os.environ``.
    """
    from dotenv import load_dotenv

    load_dotenv(path, override=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()
