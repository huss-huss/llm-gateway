from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from llm_gateway.app import create_app
from llm_gateway.config import GatewayConfig
from llm_gateway.db.models import Base
from llm_gateway.settings import Settings
from llm_gateway.state import GatewayState, build_state

ROOT = Path(__file__).resolve().parents[1]

BASE_CONFIG: dict[str, Any] = {
    "defaults": {"max_tokens": 256, "timeout_s": 2},
    "providers": {
        "fake": {
            "type": "fake",
            "rules": [
                {"match": "capital of france", "reply": "The capital of France is Paris."},
                {
                    "match": "weather",
                    "tool_call": {"name": "get_weather", "arguments": {"city": "Paris"}},
                },
                {"match": "weather", "reply": "Sunny."},
            ],
            "slow_s": 5,
        },
    },
    "aliases": {
        "fast": ["fake/fake-fast"],
        "smart": ["fake/fake-smart"],
        "resilient": ["fake/fail-500", "fake/fake-ok"],
    },
    "routing": {"retries": 1, "backoff_base_s": 0, "backoff_max_s": 0},
}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    import os

    db_url = os.environ.get("TEST_DATABASE_URL") or f"sqlite+aiosqlite:///{tmp_path}/test.db"
    return Settings(
        _env_file=None,
        models_path=str(ROOT / "config" / "models.yaml"),
        database_url=db_url,
        admin_key="test-admin-key",
        key_pepper="test-pepper",
    )


@pytest.fixture
def gateway_config() -> GatewayConfig:
    return GatewayConfig.model_validate(BASE_CONFIG)


@pytest.fixture
async def state(settings: Settings, gateway_config: GatewayConfig) -> AsyncIterator[GatewayState]:
    st = await build_state(settings, gateway_config)
    async with st.engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield st
    await st.aclose()


@pytest.fixture
async def api_key(state: GatewayState) -> str:
    raw, _ = await state.keys.create("test", "acme")
    return raw


@pytest.fixture
async def anon_client(settings: Settings, state: GatewayState) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings, state)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def client(anon_client: httpx.AsyncClient, api_key: str) -> httpx.AsyncClient:
    anon_client.headers["authorization"] = f"Bearer {api_key}"
    return anon_client
