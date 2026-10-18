from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect, select
from sqlalchemy.ext.asyncio import create_async_engine

from llm_gateway.auth import KEY_PREFIX, hash_key
from llm_gateway.cli import db_upgrade
from llm_gateway.db.models import ApiKey, Base, UsageEvent

CHAT = {"model": "fast", "messages": [{"role": "user", "content": "capital of France?"}]}


async def test_missing_key_is_401(anon_client):
    r = await anon_client.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "missing_api_key"


async def test_invalid_key_is_401(anon_client):
    r = await anon_client.post(
        "/v1/chat/completions", json=CHAT, headers={"authorization": "Bearer lgw_nope"}
    )
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"


async def test_x_api_key_header_accepted(anon_client, api_key):
    r = await anon_client.post("/v1/chat/completions", json=CHAT, headers={"x-api-key": api_key})
    assert r.status_code == 200


async def test_keys_are_stored_hashed(state, api_key):
    async with state.sessionmaker() as s:
        row = (await s.scalars(select(ApiKey))).one()
    assert api_key.startswith(KEY_PREFIX)
    assert row.key_hash == hash_key(api_key, "test-pepper")
    assert api_key not in (row.key_hash, row.prefix)
    assert api_key.startswith(row.prefix)


async def test_revoked_key_rejected(state, client, api_key):
    assert (await client.post("/v1/chat/completions", json=CHAT)).status_code == 200
    key_id = (await state.keys.list())[0].id
    await state.keys.revoke(key_id)
    r = await client.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 401


async def test_request_id_header(client):
    r = await client.post("/v1/chat/completions", json=CHAT, headers={"x-request-id": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    r = await client.post("/v1/chat/completions", json=CHAT)
    assert r.headers["x-request-id"].startswith("req_")


async def test_ledger_records_success_stream_and_failure(state, client):
    await client.post("/v1/chat/completions", json=CHAT, headers={"x-request-id": "r1"})
    await client.post(
        "/v1/chat/completions", json={**CHAT, "stream": True}, headers={"x-request-id": "r2"}
    )
    await client.post(
        "/v1/chat/completions", json={**CHAT, "model": "nope"}, headers={"x-request-id": "r3"}
    )
    async with state.sessionmaker() as s:
        rows = {e.request_id: e for e in (await s.scalars(select(UsageEvent))).all()}
    ok = rows["r1"]
    assert (ok.status, ok.http_status, ok.provider, ok.model) == ("ok", 200, "fake", "fake-fast")
    assert ok.tenant == "acme" and ok.api_key_id is not None
    assert ok.prompt_tokens > 0 and ok.completion_tokens > 0 and ok.cost_usd > 0
    assert ok.stream is False and ok.attempts == 1
    streamed = rows["r2"]
    assert streamed.stream is True and streamed.completion_tokens == ok.completion_tokens
    assert streamed.cost_usd == ok.cost_usd
    failed = rows["r3"]
    assert (failed.status, failed.http_status, failed.cost_usd) == ("model_not_found", 404, 0)


async def test_alembic_migrations_match_models(tmp_path: Path):
    url = f"sqlite+aiosqlite:///{tmp_path}/migrated.db"
    import asyncio

    await asyncio.to_thread(db_upgrade, url)
    engine = create_async_engine(url)
    async with engine.connect() as conn:

        def columns(sync_conn):
            insp = inspect(sync_conn)
            return {t: {c["name"] for c in insp.get_columns(t)} for t in insp.get_table_names()}

        migrated = await conn.run_sync(columns)
    await engine.dispose()
    for table in Base.metadata.sorted_tables:
        assert table.name in migrated, table.name
        assert {c.name for c in table.columns} == migrated[table.name], table.name
