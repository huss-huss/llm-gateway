from __future__ import annotations

ADMIN = {"authorization": "Bearer test-admin-key"}
CHAT = {"model": "fast", "messages": [{"role": "user", "content": "capital of France?"}]}


async def test_admin_requires_admin_key(anon_client, api_key):
    assert (await anon_client.get("/admin/keys")).status_code == 401
    r = await anon_client.get("/admin/keys", headers={"authorization": f"Bearer {api_key}"})
    assert r.status_code == 401  # a tenant key is not an admin key


async def test_admin_disabled_without_configured_key(anon_client, state):
    state.settings.admin_key = None
    r = await anon_client.get("/admin/keys", headers=ADMIN)
    assert r.status_code == 503


async def test_create_use_list_revoke(anon_client):
    r = await anon_client.post(
        "/admin/keys",
        headers=ADMIN,
        json={
            "name": "billing-svc",
            "tenant": "acme",
            "rate_limit_rpm": 100,
            "monthly_budget_usd": 25,
        },
    )
    assert r.status_code == 201
    created = r.json()
    raw = created["key"]
    assert raw.startswith("lgw_") and created["prefix"] == raw[:12]
    assert created["monthly_budget_usd"] == 25.0

    user = {"authorization": f"Bearer {raw}"}
    assert (
        await anon_client.post("/v1/chat/completions", json=CHAT, headers=user)
    ).status_code == 200

    listed = (await anon_client.get("/admin/keys", headers=ADMIN)).json()["data"]
    mine = next(k for k in listed if k["id"] == created["id"])
    assert "key" not in mine and "key_hash" not in mine
    assert mine["month_spend_usd"] > 0

    r = await anon_client.delete(f"/admin/keys/{created['id']}", headers=ADMIN)
    assert r.status_code == 200 and r.json()["revoked_at"]
    assert (
        await anon_client.post("/v1/chat/completions", json=CHAT, headers=user)
    ).status_code == 401
    assert (await anon_client.delete("/admin/keys/99999", headers=ADMIN)).status_code == 404


async def test_create_key_validation(anon_client):
    r = await anon_client.post("/admin/keys", headers=ADMIN, json={"name": "", "tenant": "t"})
    assert r.status_code == 400


async def test_usage_summary_groups_by_day_key_model(client, anon_client):
    for _ in range(2):
        await client.post("/v1/chat/completions", json=CHAT)
    await client.post("/v1/chat/completions", json={**CHAT, "model": "smart"})
    await client.post("/v1/chat/completions", json={**CHAT, "model": "nope"})
    r = await anon_client.get("/admin/usage", headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    rows = {(row["requested_model"], row["model"]): row for row in body["data"]}
    fast = rows[("fast", "fake-fast")]
    assert fast["requests"] == 2 and fast["errors"] == 0 and fast["key_name"] == "test"
    assert fast["cost_usd"] > 0 and len(fast["day"]) == 10
    assert rows[("smart", "fake-smart")]["requests"] == 1
    assert rows[("nope", None)]["errors"] == 1
    assert body["totals"]["requests"] == 4

    r = await anon_client.get(
        "/admin/usage", headers=ADMIN, params={"since": "2000-01-01", "until": "2000-01-02"}
    )
    assert r.json()["data"] == []
