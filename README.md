# llm-gateway

A self-hosted gateway for production LLM traffic: one OpenAI-compatible API across providers,
with routing and fallback, response caching, per-key budgets and rate limits, tracing, and evals.

```
POST /v1/chat/completions   (non-streaming and SSE streaming)
GET  /v1/models
```

Providers: **OpenAI** (or any OpenAI-compatible server), **Anthropic** (Messages API via the
official SDK), **Ollama** (local), and a deterministic **fake** provider that makes the whole
stack runnable, testable, and demoable with no API keys.

---

## Why

Once more than one service calls LLMs, the same problems show up in each of them:

- **Provider outages and 429s.** A single upstream is a single point of failure. Retrying the
  same provider forever makes things worse.
- **Spend nobody can see.** Tokens are billed per provider, per model, per key. When someone
  asks which team spent the money last Tuesday, nobody can answer.
- **Runaway clients.** One buggy loop can burn a month's budget in an hour.
- **Vendor-shaped code.** Each provider has its own message format, tool-call format, and
  streaming protocol, so switching models turns into a refactor.
- **No regression signal.** Without a repeatable eval, every model or prompt change is a guess.

`llm-gateway` handles all of these in one place. Applications speak the OpenAI chat format to
the gateway and ask for a model *alias* (`fast`, `smart`). The gateway picks which provider
serves it, enforces limits, records cost, and emits traces and metrics.

## Architecture

```mermaid
flowchart LR
    client["Client (OpenAI SDK, curl, services)"] --> api["FastAPI /v1/chat/completions"]
    api --> auth["Auth: KeyStore (HMAC-SHA256 key hash)"]
    auth --> limits["RateLimiter: RPM + TPM token buckets"]
    limits --> budget["BudgetTracker: month-to-date spend"]
    budget --> cache["ResponseCache (temperature 0, per tenant)"]
    cache -->|miss| router["Router: alias, retries, fallback"]
    router --> breaker["Circuit breakers (LocalBreakers / RedisBreakers)"]
    breaker --> adapters["Provider adapters: OpenAI, Anthropic, Ollama, Fake"]
    api -->|"after routing (ChatService)"| ledger["UsageLedger (Postgres)"]
    limits -.-> redis[("Redis")]
    budget -.-> redis
    cache -.-> redis
    breaker -.-> redis
    auth -.-> pg[("Postgres: keys, usage, eval results")]
    ledger -.-> pg
    api -.-> obs["OpenTelemetry traces + Prometheus metrics"]
    evals["EvalRunner (llm-gateway eval)"] --> router
```

A request is authenticated by `KeyStore`, then checked against Redis-backed RPM/TPM buckets
(`RateLimiter`) and the key's monthly budget (`BudgetTracker`). Deterministic requests can be
served from `ResponseCache`; otherwise `Router` resolves the model alias to an ordered list of
provider targets, retrying and falling back while per-provider circuit breakers (in-process or
shared via Redis) skip unhealthy upstreams. The priced result is written to the Postgres
`UsageLedger` and emitted as traces and metrics; `EvalRunner` drives the same `Router` for evals.

```
                         +--------------------------------------------------------------+
  clients                |  llm-gateway (FastAPI)                                       |
  (OpenAI SDKs,  ------> |                                                              |
   curl, services)       |  request-ID + access-log middleware                          |
                         |        |                                                     |
                         |  auth: API key -> HMAC-SHA256 hash lookup                    |
                         |        |                                                     |
                         |  rate limits (RPM + TPM Redis token buckets, atomic Lua)     |
                         |  budget      (Redis month-to-date spend, rebuilt from ledger)|
                         |  cache       (Redis, temperature == 0, per tenant)           |
                         |        |                                                     |
                         |  router: alias -> [provider/model, ...]                      |
                         |    retries + jittered backoff, timeouts, fallback,           |
                         |    per-provider circuit breaker (in-process or Redis-shared) |
                         |        |                                                     |
                         |  adapters:  openai | anthropic | ollama | fake               |
                         |        |                                                     |
                         |  price (config/models.yaml) -> ledger row, metrics, spans    |
                         +--------|-------------------------------|----------|----------+
                                  v                               v          v
               OpenAI / Anthropic / Ollama APIs            Redis       Postgres
                                                     (limits, cache,  (keys, usage ledger,
               OTLP collector (Jaeger), Prometheus     spend,          eval results)
                                                      breakers)
```

Code map (`src/llm_gateway/`):

| Module | Responsibility |
|---|---|
| `api/chat.py`, `api/admin.py` | HTTP endpoints |
| `service.py` | the request pipeline (limits, cache, routing, pricing, ledger) |
| `router.py`, `breaker.py` | alias resolution, retries/backoff, fallback, circuit breakers (in-process or Redis) |
| `providers/` | `openai.py`, `anthropic.py`, `ollama.py`, `fake.py` behind one `Provider` interface |
| `streaming.py` | OpenAI `chat.completion.chunk` SSE encoding and stream accumulation |
| `catalog.py` + `config/models.yaml` | prices and capability flags per model |
| `cache.py`, `ratelimit.py`, `budget.py` | Redis-backed cache, request/token buckets, spend counters |
| `auth.py`, `ledger.py`, `usage.py`, `db/` | keys, usage ledger, reporting, SQLAlchemy models |
| `observability/` | JSON logging, Prometheus metrics, OpenTelemetry setup |
| `evals/` | suite format, assertions, runner |
| `migrations/` | Alembic (async) |

## Request lifecycle

For `POST /v1/chat/completions`:

1. **Request ID.** A well-formed incoming `X-Request-ID` is kept. Otherwise the gateway
   generates one. The ID is echoed in the response, attached to every log line, and stored in
   the ledger.
2. **Authenticate.** `Authorization: Bearer lgw_...` (or `x-api-key`). The key is
   HMAC-SHA256'd with a server pepper and looked up by hash, with a 5 s in-process cache.
3. **Resolve the route.** For example, `fast` -> `[anthropic/claude-haiku-5-5,
   openai/gpt-4.1-mini, ollama/llama3.2, fake/fake-fast]`. Providers without credentials are
   skipped. An unknown model gets `404 model_not_found`.
4. **Request rate limit.** Each request takes one token from the key's request bucket
   (capacity = RPM, refilled continuously). Over the limit, the client gets `429` with
   `Retry-After`. These rejections are not written to the ledger.
5. **Budget.** The key's month-to-date spend must be below its monthly budget. Spend lives in a
   Redis micro-dollar counter that is rebuilt from the ledger when cold. Over budget, the
   request gets `402` before any provider is called.
6. **Cache.** If `temperature == 0`, look up `sha256(tenant + normalized request)`. A hit
   returns the stored completion (or replays it as SSE), costs $0, and is flagged in the ledger.
   It skips the token limit entirely.
7. **Token rate limit.** If the key has a TPM limit, reserve the request's estimated tokens
   (prompt estimate + `max_tokens`) in its token bucket, or return `429` with `Retry-After`.
   See [Rate limits](#rate-limits).
8. **Route.** Try each target in order. Each attempt is a `provider.call` span with a timeout.
   - 429, 5xx, timeout, connection error: retry the same target with jittered exponential
     backoff (respecting `Retry-After`, capped), then fall back to the next target.
   - 401/403 (bad provider credentials) or 404 (unknown model on that provider): fall back
     without retrying.
   - 400 (invalid request): fail fast. No fallback, and the breaker is not tripped.
   - A provider whose breaker is open is skipped. See [Circuit breakers](#circuit-breakers).
   - Streams: the first chunk is pulled inside the retry loop, so fallback still works until
     the first byte reaches the client. After that, upstream errors are reported in-band as an
     SSE `error` event.
9. **Settle and price.** Replace the token reservation with the actual usage (refunding or
   charging the difference). Then `usage x price` from `config/models.yaml`, with cached-token and long-context
   rates where defined.
10. **Record.** Add the spend to the key's counter and write one `usage_events` row (tokens
   in/out, cost, latency, provider, model, cache hit, status, attempts). Then update Prometheus
   metrics, log one JSON line, and close the span (for streams, when the stream ends).
11. **Respond** with gateway headers:

| Header | Meaning |
|---|---|
| `x-gateway-provider`, `x-gateway-model` | the target that actually served the request |
| `x-gateway-attempts` | provider attempts made (0 on a cache hit) |
| `x-gateway-cost-usd` | cost of this request (non-streaming responses) |
| `x-gateway-cache` | `hit`, `miss`, or `bypass` |
| `x-gateway-budget-remaining-usd` | remaining monthly budget (keys with a budget) |
| `x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests` | request rate limit state |
| `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-tokens`, `x-ratelimit-reset-tokens` | token rate limit state (keys with a TPM limit; not on cache hits) |
| `x-request-id` | request ID |

## Rate limits

Every key can have two limits. Both are Redis token buckets updated by one atomic Lua script,
with capacity equal to the per-minute limit (so a key can burst a full minute's allowance) and
continuous refill at `limit / 60` per second:

| Limit | Key field | Default (`config/gateway.yaml`) | Counts |
|---|---|---|---|
| requests per minute | `rate_limit_rpm` | `limits.default_rpm` (600) | 1 per request, including cache hits |
| tokens per minute | `rate_limit_tpm` | `limits.default_tpm` (unset) | prompt + completion tokens |

A key's own value takes precedence. When it is `null`, the configured default applies, and
when that is unset too, there is no limit. Set the fields at creation (`POST /admin/keys`,
`llm-gateway keys create --rpm N --tpm N`) or change them later with
`PATCH /admin/keys/{id}`, which updates only the fields in the body (`null` clears one).
Changes apply immediately on the replica that handled them and within 5 s elsewhere.

**How a request is counted against TPM:**

1. **Reserve.** Once the request is past the cache and a provider will be called, the gateway
   reserves `prompt estimate + completion allowance` tokens. The prompt estimate is ~4
   characters per token over message text, tool calls, and tool definitions, plus 4 tokens per
   message. The completion allowance is the request's `max_completion_tokens` / `max_tokens`,
   or `limits.completion_token_allowance` (default: `defaults.max_tokens`). The reservation is
   capped at the limit itself, so a large `max_tokens` can't lock a key out. The Lua script
   checks and takes the tokens in one step, so concurrent requests can't jointly reserve more
   than the bucket holds. If the bucket is short, the request gets `429` with `Retry-After`
   (seconds until enough tokens have refilled) and no provider is called.
2. **Settle.** After the response, the reservation is replaced with the actual
   `total_tokens` the provider reported. Unused tokens are refunded. If the request used more
   than it reserved, the difference is charged, and the bucket can go negative. Later requests
   then wait until it refills, so an under-estimate is paid back rather than forgiven. If the
   provider reports no usage, the gateway charges its prompt estimate plus the output length.
   A failed request (every target errored) is refunded in full.
3. **Streams** reserve before the first byte and settle when the stream ends, from the final
   usage chunk. If the client disconnects (or the upstream fails mid-stream) before usage
   arrives, the gateway settles with what it knows: the prompt estimate plus the output
   received so far.
4. **Cache hits** don't call a provider, so they never reserve or consume TPM, and they are
   served even when the token bucket is empty. They still count against RPM.

A prompt whose estimate alone exceeds the key's TPM limit could never be admitted, so it gets
`429` with code `request_too_large` and no `Retry-After`.

**Headers.** Responses that went through the token bucket carry OpenAI-style headers:
`x-ratelimit-limit-tokens`, `x-ratelimit-remaining-tokens` (never below 0), and
`x-ratelimit-reset-tokens` (time until the bucket is full again, e.g. `453ms`, `10s`,
`1m30s`). Non-streaming responses report the bucket after settlement. Streaming responses send
headers before the body, so they report it after the reservation. Cache hits, and keys without
a TPM limit, get no token headers.

## Circuit breakers

Each provider has a breaker (one per provider, so failures on one of its models count against
its other models too). Only failures that fall back count: 429, 5xx, timeouts, connection
errors, and 401/403/404 from the provider. A 400 is the request's fault and never counts.

- **Closed:** calls go through. `failure_threshold` failures within `window_s` (counted from
  the first failure in the window) open the circuit. A success resets the count.
- **Open:** the provider is skipped (attempt outcome `circuit_open`) for `cooldown_s`. When
  every target is open, the client gets `503 circuit_open`.
- **Half-open:** after the cooldown, exactly one call (the probe) goes through. If it succeeds
  the circuit closes; if it fails the circuit opens for another cooldown. The probe holds a
  lease of `probe_lease_s` (default: the provider's `timeout_s`), so a probe that never
  reports back (a crashed replica) is replaced once the lease expires. A probe that ends in a
  400 hands the lease back at once.

**Backends** (`routing.breaker.backend`):

| Backend | State | Use it for |
|---|---|---|
| `memory` (default) | per process | a single replica, or when replicas should act independently |
| `redis` | shared through `GATEWAY_REDIS_URL` | several replicas: a failing provider is tripped once for all of them |

With `redis`, each provider's breaker is one hash, `breaker:{provider}`, holding the state,
the failure count and window start, the time the circuit opened, and the probe lease. Every
operation (allow, success, failure, release) is one Lua script, so concurrent replicas can't
both take the probe or lose a failure. Keys clean themselves up: a healthy provider has no
key, a closed circuit's failure count expires with its window, and an open or half-open
circuit with no activity expires `state_ttl_s` after its cooldown (a missing key reads as
closed). The Docker Compose file selects `redis`; elsewhere set
`GATEWAY_BREAKER_BACKEND=redis` or edit `config/gateway.yaml`.

**If Redis is unavailable,** the breaker store never fails a request. A Redis error, or a call
slower than `store_timeout_s` (250 ms), switches that replica to fresh in-process breakers,
logs a warning (`circuit breaker store unavailable, falling back to in-process breakers`), and
sets `llm_gateway_circuit_store_degraded` to 1. The replica tries Redis again after
`store_retry_s` (5 s); when Redis answers, it logs the recovery, resets the gauge, and goes
back to the shared state. Errors are counted in `llm_gateway_circuit_store_errors_total`.

**Metrics.** `llm_gateway_circuit_transitions_total{provider, from_state, to_state, backend}`
counts every state change. With `redis`, each change is counted once, by the replica that made
it (the replica that sees an open circuit's cooldown end records `open -> half_open`), so a
sum across replicas is the true count. `llm_gateway_circuit_state` is the last state this
replica saw. Transitions are also logged (`circuit breaker closed -> open`, at warning level
when a circuit opens).

## Quick start (no API keys, no Docker)

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). The defaults use SQLite and an
in-process Redis substitute (`GATEWAY_REDIS_URL=memory://`, backed by fakeredis), so nothing
else has to be running. Every response below comes from the deterministic `fake` provider.

```bash
git clone <this repo> llm-gateway && cd llm-gateway
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"
cp .env.example .env              # demo admin key + pepper; SQLite + memory:// Redis
.venv/bin/llm-gateway db upgrade  # Alembic migrations
.venv/bin/llm-gateway serve --port 8080
```

These outputs were captured from a fresh clone that followed exactly these steps. IDs,
timestamps, and the key are different on every run.

**Create an API key** (admin API; the raw key is shown once):

```bash
curl -s -X POST localhost:8080/admin/keys \
  -H "Authorization: Bearer change-me-admin-key" -H "Content-Type: application/json" \
  -d '{"name": "quickstart", "tenant": "acme", "rate_limit_rpm": 60, "rate_limit_tpm": 2000, "monthly_budget_usd": 5}'
```
```json
{"id":1,"name":"quickstart","tenant":"acme","prefix":"lgw_CVb0SLRb","rate_limit_rpm":60,"rate_limit_tpm":2000,"monthly_budget_usd":5.0,"created_at":"2026-10-19T11:08:49.646729+00:00","revoked_at":null,"key":"lgw_CVb0SLRb..."}
```

```bash
export KEY=lgw_...   # the "key" value from above
```

**Chat completion:**

```bash
curl -s -i localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "fast", "temperature": 0, "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```
```
HTTP/1.1 200 OK
x-ratelimit-limit-requests: 60
x-ratelimit-remaining-requests: 59
x-gateway-budget-remaining-usd: 5.000000
x-ratelimit-limit-tokens: 2000
x-ratelimit-remaining-tokens: 1984
x-ratelimit-reset-tokens: 453ms
x-gateway-provider: fake
x-gateway-model: fake-fast
x-gateway-attempts: 1
x-gateway-cost-usd: 0.00001600
x-gateway-cache: miss
content-type: application/json
x-request-id: req_07000001060b4ecea6450c3297be1a03

{"id":"chatcmpl-f07e6355cc8c475183c89c90","object":"chat.completion","created":1792340318,"model":"fast","choices":[{"index":0,"message":{"role":"assistant","content":"The capital of France is Paris."},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":8,"total_tokens":16}}
```

**The same request again** comes from the cache (`temperature: 0`) and costs nothing:

```
x-gateway-budget-remaining-usd: 4.999984
x-gateway-provider: fake
x-gateway-model: fake-fast
x-gateway-attempts: 0
x-gateway-cost-usd: 0.00000000
x-gateway-cache: hit
```

**Streaming** (`stream_options.include_usage` adds a final usage chunk):

```bash
curl -s -N localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "fast", "stream": true, "stream_options": {"include_usage": true}, "messages": [{"role": "user", "content": "What is 2 + 2?"}]}'
```
```
data: {"id":"chatcmpl-5b7c019f528d4a058e38bd4d","object":"chat.completion.chunk","created":1792340318,"model":"fast","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-5b7c019f528d4a058e38bd4d","object":"chat.completion.chunk","created":1792340318,"model":"fast","choices":[{"index":0,"delta":{"content":"4"},"finish_reason":null}]}

data: {"id":"chatcmpl-5b7c019f528d4a058e38bd4d","object":"chat.completion.chunk","created":1792340318,"model":"fast","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-5b7c019f528d4a058e38bd4d","object":"chat.completion.chunk","created":1792340318,"model":"fast","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}

data: [DONE]
```

**Fallback.** The `demo-fallback` alias points at `fake/fail-500`, which always returns a 5xx,
and then at `fake/fake-fast`. The gateway retries the failing target twice with backoff, then
falls back:

```bash
curl -s -i localhost:8080/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "demo-fallback", "messages": [{"role": "user", "content": "hello"}]}' | grep -i "^x-gateway-"
```
```
x-gateway-provider: fake
x-gateway-model: fake-fast
x-gateway-attempts: 4
```

**Tool calls** use the OpenAI format for every provider:

```bash
curl -s localhost:8080/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "fast", "messages": [{"role": "user", "content": "What is the weather in Addis Ababa?"}], "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]}'
```
```json
{"id":"chatcmpl-c1ae420c9793486487b15ce9","object":"chat.completion","created":1792340318,"model":"fast","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_fake_0","type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Addis Ababa\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":9,"completion_tokens":16,"total_tokens":25}}
```

**Limits.** A key with `rate_limit_rpm: 2` gets a 429 on its third request within a minute.
A key that has used up its monthly budget gets a 402. These are the observed responses:

```json
{"error":{"message":"rate limit of 2 requests/minute exceeded for this API key","type":"rate_limit_error","code":"rate_limit_exceeded"}}
{"error":{"message":"monthly budget of $0.00002000 exhausted for this API key","type":"budget_exceeded","code":"monthly_budget_exceeded"}}
```

**Token limits.** Lowering the key's TPM limit to 100 with `PATCH /admin/keys/1`
(`{"rate_limit_tpm": 100}`) and sending two requests without `max_tokens`: each one reserves
the whole bucket (the default allowance of 1024 is capped at the limit), and the first settles
at the 16 tokens it used. The second is short 16 tokens, which take 10 s to refill:

```
HTTP/1.1 200 OK
x-ratelimit-limit-tokens: 100
x-ratelimit-remaining-tokens: 84
x-ratelimit-reset-tokens: 10s

HTTP/1.1 429 Too Many Requests
x-ratelimit-limit-tokens: 100
x-ratelimit-remaining-tokens: 84
x-ratelimit-reset-tokens: 10s
retry-after: 10
```
```json
{"error":{"message":"rate limit of 100 tokens/minute exceeded for this API key (this request needs about 100)","type":"rate_limit_error","code":"rate_limit_exceeded"}}
```

Sending `max_tokens` keeps the reservation small, so more requests fit in the bucket.

**OpenAI SDK.** Existing clients only need a new `base_url`. This was observed with the
official `openai` Python package against the fake provider:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key=KEY)
r = client.chat.completions.create(
    model="smart", messages=[{"role": "user", "content": "What is the capital of France?"}]
)
print(r.choices[0].message.content, r.usage.total_tokens)
# The capital of France is Paris. 16
```

**Usage report** (per day, key, and model):

```bash
curl -s "localhost:8080/admin/usage?since=2026-10-01" -H "Authorization: Bearer change-me-admin-key"
```
One row of `data` (the response also carries `totals`):
```json
{
  "day": "2026-10-18",
  "api_key_id": 1,
  "key_name": "quickstart",
  "tenant": "acme",
  "requested_model": "demo-fallback",
  "provider": "fake",
  "model": "fake-fast",
  "requests": 1,
  "cache_hits": 0,
  "errors": 0,
  "prompt_tokens": 2,
  "completion_tokens": 18,
  "cost_usd": 2.8e-05,
  "avg_latency_ms": 605.0
}
```

Other admin endpoints: `GET /admin/keys` lists keys with month-to-date spend,
`PATCH /admin/keys/{id}` changes a key's `rate_limit_rpm`, `rate_limit_tpm`, or
`monthly_budget_usd`, and `DELETE /admin/keys/{id}` revokes one. The CLI does the same offline:
`llm-gateway keys create|list|revoke`.

### With real providers

Put keys in `.env` or the environment. Providers whose key is unset stay disabled:

```bash
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
OLLAMA_ENABLED=true          # uses http://localhost:11434
```

With those set, `fast` is served by `anthropic/claude-haiku-5-5` first and falls back down its
chain. You can also address a target directly as `provider/model`, e.g.
`"model": "ollama/llama3.1"`. The Ollama adapter was checked by hand against a local Ollama
server (`ollama/llama3.1`, streaming and non-streaming). The OpenAI and Anthropic adapters are
covered by tests against mocked HTTP transports. Building and verifying this release made no
calls to paid APIs.

### Docker Compose (gateway + Postgres + Redis, optional Jaeger)

```bash
cp .env.example .env     # set GATEWAY_ADMIN_KEY and GATEWAY_KEY_PEPPER to long random values
docker compose up --build
# with tracing: GATEWAY_OTEL_EXPORTER=otlp docker compose --profile tracing up --build
#   Jaeger UI at http://localhost:16686
```

The container applies migrations on start and listens on `:8080`.

## Configuration

**Environment** (`GATEWAY_` prefix, read from the environment or `.env`):

| Variable | Default | Notes |
|---|---|---|
| `GATEWAY_DATABASE_URL` | `sqlite+aiosqlite:///./llm_gateway.db` | e.g. `postgresql+asyncpg://user:pw@host/db` |
| `GATEWAY_REDIS_URL` | `memory://` | `redis://host:6379/0`; `memory://` is for dev only |
| `GATEWAY_BREAKER_BACKEND` | `memory` | `redis` shares circuit breakers across replicas (read by `config/gateway.yaml`) |
| `GATEWAY_ADMIN_KEY` | unset | the admin API returns 503 until this is set |
| `GATEWAY_KEY_PEPPER` | `dev-pepper-change-me` | HMAC secret for key hashes; **set it in production** |
| `GATEWAY_REQUIRE_AUTH` | `true` | `false` allows anonymous use (local experiments only) |
| `GATEWAY_CONFIG_PATH` | `config/gateway.yaml` | providers, aliases, routing policy |
| `GATEWAY_MODELS_PATH` | `config/models.yaml` | prices and capability flags |
| `GATEWAY_LOG_LEVEL` / `GATEWAY_LOG_JSON` | `INFO` / `true` | |
| `GATEWAY_OTEL_EXPORTER` | `none` | `console` or `otlp` (OTLP/HTTP) |
| `GATEWAY_OTEL_ENDPOINT` | unset | e.g. `http://localhost:4318` |
| `GATEWAY_HOST` / `GATEWAY_PORT` | `127.0.0.1` / `8080` | for `llm-gateway serve` |

**`config/gateway.yaml`** (supports `${VAR}` / `${VAR:-default}`), abridged:

```yaml
providers:
  anthropic: { type: anthropic, api_key_env: ANTHROPIC_API_KEY, timeout_s: 60, server_fallback: true }
  openai:    { type: openai, api_key_env: OPENAI_API_KEY, base_url: "${OPENAI_BASE_URL:-https://api.openai.com/v1}" }
  ollama:    { type: ollama, enabled: "${OLLAMA_ENABLED:-false}", base_url: "${OLLAMA_BASE_URL:-http://localhost:11434}" }
  fake:      { type: fake, latency_ms: 25, rules: [...] }

routing:
  retries: 2            # extra attempts per target on 429 / 5xx / timeout
  backoff_base_s: 0.25
  backoff_max_s: 4
  breaker:
    backend: ${GATEWAY_BREAKER_BACKEND:-memory}  # memory | redis (shared across replicas)
    failure_threshold: 5  # failures within window_s that open the circuit
    window_s: 60
    cooldown_s: 30        # open -> half-open (one probe)
    probe_lease_s: null   # null = the provider's timeout_s
    state_ttl_s: 3600     # redis only: idle open circuits expire this long after the cooldown
    store_timeout_s: 0.25 # redis only: per call; on error/timeout use in-process breakers...
    store_retry_s: 5      # ...for this long, then try Redis again

limits:                            # for keys without their own limits
  default_rpm: 600
  default_tpm: null                # tokens per minute; null = unlimited
  completion_token_allowance: null # reserved when a request has no max_tokens (null = defaults.max_tokens)
cache:  { enabled: true, ttl_s: 3600 }

aliases:
  fast:  [anthropic/claude-haiku-5-5, openai/gpt-4.1-mini, ollama/llama3.2, fake/fake-fast]
  smart: [anthropic/claude-opus-5-5, openai/gpt-4.1, fake/fake-smart]
```

The `fake/*` entries in the alias chains only exist so the quick start works without keys.
Remove them in production.

**`config/models.yaml`** holds USD prices per 1M tokens, with an `as_of` date and a source note
for each price group. Anthropic prices come from Anthropic's current model and pricing table:
Claude Opus 5.5 $4/$20, Sonnet 5.5 $2/$10, and Haiku 5.5 $0.10/$0.50 (with a long-context tier
above 100K prompt tokens). The OpenAI prices are the last values the author verified, and the
file marks them for re-checking. The file also has capability flags that the Anthropic adapter
honours:

- `sampling: false`: `temperature` and `top_p` are not sent, because current Claude models
  reject them.
- `forced_tool_choice: false`: `tool_choice: "required"` or a named tool is sent as `auto`,
  because those models return 400 for forced tool use.
- `server_fallback: true`: the request goes to the Messages API beta endpoint with
  `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`). If a safety classifier
  declines the request, Anthropic re-runs it on its recommended fallback model within the same
  call. A refusal that still comes back maps to `finish_reason: "content_filter"`. Set
  `server_fallback: false` on the provider to opt out.

## Anthropic translation

| OpenAI chat | Anthropic Messages |
|---|---|
| `system` / `developer` messages | top-level `system` (joined) |
| `assistant.tool_calls[]` | `tool_use` blocks (`arguments` JSON-parsed; invalid JSON -> 400) |
| consecutive `tool` messages | `tool_result` blocks in **one** `user` message |
| `image_url` (data: URL / https URL) | `image` block with `base64` / `url` source |
| `tools[].function.parameters` | `input_schema` |
| `tool_choice` `auto` / `none` / `required` / named | `auto` / `none` / `any` / `tool` (forced -> `auto` where unsupported) |
| `parallel_tool_calls: false` | `disable_parallel_tool_use: true` |
| `response_format: json_schema` | `output_config.format` |
| `response_format: json_object` | system instruction (no native equivalent) |
| `stop` | `stop_sequences` |
| missing `max_tokens` | `defaults.max_tokens` (Anthropic requires it) |
| stop reasons `end_turn`, `max_tokens`, `tool_use`, `refusal` | `stop`, `length`, `tool_calls`, `content_filter` |
| `input_tokens + cache_read + cache_creation` | `prompt_tokens` (`cached_tokens` = cache reads) |

Streaming maps `message_start` / `content_block_*` / `message_delta` / `message_stop` events to
OpenAI chunks, including incremental tool-call arguments (`input_json_delta`). The SDK client
is created with `max_retries=0` because the gateway does its own retries and fallback.

## Metrics

`GET /metrics` (Prometheus text format):

| Metric | Labels |
|---|---|
| `llm_gateway_requests_total` | `model`, `status`, `stream` |
| `llm_gateway_request_duration_seconds` (histogram) | `model`, `stream` |
| `llm_gateway_provider_attempts_total` | `provider`, `model`, `outcome` (`ok`, `rate_limited`, `timeout`, `unavailable`, `auth`, `invalid_request`, `circuit_open`) |
| `llm_gateway_provider_duration_seconds` (histogram) | `provider`, `model` |
| `llm_gateway_tokens_total` | `provider`, `model`, `direction` |
| `llm_gateway_cost_usd_total` | `provider`, `model` |
| `llm_gateway_cache_requests_total` | `result` |
| `llm_gateway_rejections_total` | `reason` (`rate_limited`, `token_rate_limited`, `budget_exceeded`) |
| `llm_gateway_circuit_state` | `provider` (0 closed, 1 half-open, 2 open) |
| `llm_gateway_circuit_transitions_total` | `provider`, `from_state`, `to_state`, `backend` (`memory`, `redis`) |
| `llm_gateway_circuit_store_degraded` | none (1 while the Redis breaker store is unavailable and in-process breakers are used) |
| `llm_gateway_circuit_store_errors_total` | `op` (`allow`, `success`, `failure`, `release`, `state`) |

Some of the values observed after the quick start:

```
llm_gateway_requests_total{model="fast",status="ok",stream="false"} 3.0
llm_gateway_requests_total{model="demo-fallback",status="ok",stream="false"} 1.0
llm_gateway_provider_attempts_total{model="fake-fast",outcome="ok",provider="fake"} 5.0
llm_gateway_provider_attempts_total{model="fail-500",outcome="unavailable",provider="fake"} 3.0
llm_gateway_cache_requests_total{result="hit"} 1.0
```

**Traces.** Each request gets a `chat.completions` span with one `provider.call` child per
attempt. Spans use OpenTelemetry GenAI attribute names where they exist
(`gen_ai.request.model`, `gen_ai.usage.input_tokens`, ...), plus `llm_gateway.*` attributes
(tenant, request ID, outcome, cost, cache hit). For streams, the root span ends when the stream
ends, and each `provider.call` span covers only the time to the first chunk. Set
`GATEWAY_OTEL_EXPORTER=console` to print spans locally.

**Logs.** Every line is JSON. Each request produces one access-log line and one
`chat completion` line with tenant, model, provider, tokens, cost, latency, and status. Prompt
and completion text are never logged.

`GET /healthz` is the liveness check. `GET /readyz` pings Redis and the database.

## Evals

```bash
.venv/bin/llm-gateway eval run evals/suite.yaml --model fast
```
```
Suite: smoke   model: fast

  PASS  capital-of-france       26 ms  $0.000019  fake/fake-fast
  PASS  arithmetic-exact        26 ms  $0.000007  fake/fake-fast
  PASS  person-json-schema      26 ms  $0.000029  fake/fake-fast
  PASS  year-regex              27 ms  $0.000022  fake/fake-fast
  PASS  judged-capital          25 ms  $0.000016  fake/fake-fast
  FAIL  translate-hello         25 ms  $0.000050  fake/fake-fast
        - expected output to contain 'bonjour'

Passed 5/6 (83.3%)   avg latency 25.8 ms   p95 27 ms   cost $0.000141 (+ judge $0.000038)
Stored as eval run #1
```

`translate-hello` is designed to fail on the fake provider, so you can see how failures are
reported. A suite is a YAML file of cases. Each case has a `prompt` or `messages`, plus
optional `system`, `params`, and `response_format`, and a list of assertions:

| Assertion | Passes when |
|---|---|
| `exact` | output equals `value` (trimmed; case-insensitive unless `case_sensitive: true`) |
| `contains` / `not_contains` | `value` is / is not a substring |
| `regex` | `pattern` matches |
| `json_valid` | output parses as JSON (fenced ```` ```json ```` blocks are accepted) |
| `json_schema` | output parses and validates against `schema` |
| `llm_judge` | the judge model (`--judge-model`, default `smart`) answers PASS for `rubric` |

Options:

- `--concurrency N`
- `--json` for machine-readable output
- `--no-store`
- `--min-pass-rate 0.9` exits non-zero below the threshold (CI runs the example suite with
  `0.8`)

Runs and per-case results, including model outputs, are stored in `eval_runs` and
`eval_results`. Evals call the router in-process. They use the same providers, fallback, and
pricing as live traffic, but skip auth, rate limits, and the cache. In the fake config the
judge always answers PASS, so it only tests the plumbing. Point `--judge-model` at a real model
for real grading.

### Comparing runs

`eval diff` compares two stored runs case by case. A run is a run id, `latest`, or `previous`:

```bash
.venv/bin/llm-gateway eval diff previous latest
```
```
Eval run #1 (smoke, fast)  ->  #2 (smoke, fast)

  status         case                A     B        latency ms (A -> B)   cost delta
  regressed      capital-of-france   PASS  FAIL            26 -> 26 (0)    $0.000000
  still_failing  translate-hello     FAIL  FAIL            26 -> 26 (0)    $0.000000
  unchanged      arithmetic-exact    PASS  PASS            26 -> 26 (0)    $0.000000
  unchanged      person-json-schema  PASS  PASS            26 -> 26 (0)    $0.000000
  unchanged      year-regex          PASS  PASS            26 -> 26 (0)    $0.000000
  unchanged      judged-capital      PASS  PASS            26 -> 26 (0)    $0.000000

                      run #1        run #2         delta
  pass rate            83.3%         66.7%      -16.7 pp
  passed                 5/6           4/6            -1
  avg latency        26.0 ms       26.0 ms        0.0 ms
  p95 latency          26 ms         26 ms          0 ms
  cost             $0.000141     $0.000141     $0.000000

1 regressed, 0 fixed, 1 still failing, 0 new, 0 removed, 4 unchanged
FAIL (--fail-on regression): 1 case regressed; pass rate dropped 83.3% -> 66.7%
```

In this example the second run used a stricter assertion for `capital-of-france`. Cases are
matched by case id and get one of these statuses: `fixed`, `regressed`,
`still_failing`, `unchanged`, `new` (only in the second run), or `removed` (only in the first).
Regressions are listed first. If the runs come from different suites or models, `eval diff`
prints a warning and still compares by case id.

Options:

- `--format json` for machine-readable output (runs, deltas, per-case statuses, gate result)
- `--fail-on regression` (default) exits 1 if any case regressed or the pass rate dropped;
  `--fail-on pass-rate` exits 1 only if the pass rate dropped; `--fail-on never` always exits 0
- `--suite NAME` resolves `latest` / `previous` among runs of that suite only
- `--url` reads from another database (default: `GATEWAY_DATABASE_URL`)

Exit codes are 0 (gate passed), 1 (gate failed), and 2 (a run was not found). That makes it
usable as a CI gate against a stored baseline:

```bash
llm-gateway eval run evals/suite.yaml --model fast
llm-gateway eval diff "$BASELINE_RUN_ID" latest --suite smoke
```

CI runs the example suite twice on the fake provider and diffs the two runs
(`eval diff previous latest --suite smoke`). The fake provider is deterministic, so this checks
the command end to end. With a persistent eval database, the same step compares each change
against the last run on `main`.

## Design trade-offs

- **OpenAI wire format as the common language.** Most SDKs and tools already speak it. The
  downside is that Anthropic-only features (thinking blocks, prompt-cache breakpoints,
  citations) have nowhere to go in that format, so they are dropped. A passthrough
  `/v1/messages` endpoint is on the roadmap.
- **Fallback only before the first byte.** Once a client has part of a stream, switching models
  mid-answer would splice two different completions together. So the gateway pulls the first
  chunk inside the retry loop and stops failing over after that.
- **Circuit breakers per provider, in-process by default, optionally shared in Redis.** The
  in-process backend needs no coordination, but with N replicas a failing provider takes N
  times the failures to trip everywhere and gets N probes. The Redis backend trips it once
  for all replicas and allows one probe at a time, at the cost of one Redis round trip per
  provider attempt and per outcome. Redis is kept off the critical path for availability: if
  it errors or is slow, a replica falls back to in-process breakers rather than failing
  requests. Because a breaker covers a whole provider, repeated failures on one of its models
  also count against its other models.
- **Budget checked before the call, settled after.** Exact pre-flight pricing would need the
  output token count, which isn't known in advance. A request that starts under budget can
  overshoot by its own cost. Under concurrency, every request already in flight when the budget
  runs out can overshoot.
- **Token limits reserve an estimate, then settle.** The true token count is only known after
  the response, so the gateway reserves a pessimistic estimate up front and corrects it
  afterwards. Reserving `max_tokens` (like OpenAI does) keeps concurrent requests from
  overshooting, at the cost of rejecting some requests that would have fit. The prompt estimate
  is a character count, not a tokenizer, so it is provider-agnostic and cheap but approximate.
  Charging under-estimates as debt keeps the long-run rate correct.
- **Spend in Redis, source of truth in Postgres.** The counter caches the month's
  `SUM(cost_usd)` for each key. If Redis is flushed, the counter is rebuilt from the ledger on
  the next request. Initialization uses `SET NX`, so concurrent rebuilds don't double count.
- **Exact-match cache, `temperature == 0` only, scoped per tenant.** A semantic cache would hit
  more often, but it can return answers to questions nobody asked. Scoping by tenant prevents
  leaks and probing across customers.
- **HMAC-SHA256 for key hashes, not bcrypt/argon2.** Keys are 256-bit random tokens, not
  passwords, so there is nothing to brute-force, and verification runs on every request. The
  pepper makes stolen hashes useless without the server secret.
- **Ledger writes on the request path.** This is simple and loses nothing on a crash. At higher
  volume, a queue or batch writer would take it off the latency path (on the roadmap).

## Security notes

- Raw API keys are shown once, at creation. The database stores only
  `HMAC-SHA256(pepper, key)` and a 12-character display prefix. Revocation takes effect
  immediately on the replica that handled it, and within 5 s on others (the key cache TTL).
- The admin API stays disabled until `GATEWAY_ADMIN_KEY` is set. It uses a separate credential,
  compared in constant time. Keep `/admin` and `/metrics` inside your network boundary. They
  are not meant for the public internet.
- Set `GATEWAY_KEY_PEPPER` and `GATEWAY_ADMIN_KEY` to long random values. Changing the pepper
  invalidates every existing key.
- Provider API keys are read only from the environment and are never logged. `.env` is
  gitignored.
- Prompts and completions are not logged and are not stored in the usage ledger. Two places do
  hold them: cached completions stay in Redis for `ttl_s`, and eval results store model output.
  Treat Redis and the eval tables as sensitive.
- Run the gateway behind TLS termination. It speaks plain HTTP itself.

## Testing

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/pytest -q                       # SQLite + fakeredis, no services, no network
# against real services:
TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/llm_gateway_test \
TEST_REDIS_URL=redis://localhost:6379/15 .venv/bin/pytest -q
```

What the suite covers:

- routing and fallback, retries and backoff, the circuit breaker, timeouts
- shared breakers: two gateway instances on one Redis seeing the same open state and failure
  count, a single half-open probe across concurrent callers (and lease expiry), recovery,
  failed probes, TTL cleanup, falling back when Redis is down or slow, and returning to Redis
- the cache: normalization, tenant scoping, TTL, streaming replay
- rate limits (the Lua token bucket) and budgets (enforcement, rebuild from the ledger)
- token limits: estimate, reserve and settle (refund and charge), concurrent requests not
  overshooting, streams (final usage chunk and client disconnect), cache hits, headers, and the
  admin API
- Anthropic <-> OpenAI translation through the real `anthropic` SDK with a mock transport
- the OpenAI and Ollama adapters with mock transports
- SSE format, cost calculation, auth and revocation, admin endpoints
- tracing spans, metrics, and Alembic migrations matching the models
- evals, and `eval diff` (every case status and `--fail-on` policy)

No test makes a network call. CI (`.github/workflows/ci.yml`) runs:

- lint
- the suite on SQLite
- the example eval with a pass-rate gate, and `eval diff` against a second run
- the suite against Postgres 17 + Redis 7.4, after `alembic upgrade` and `alembic check`
  (including the shared-breaker tests, against the real Redis)

## Limitations (v0.1)

- Only chat completions. There's no support for embeddings, legacy completions, images, audio,
  the Responses API, or the Batch API.
- Only the OpenAI adapter honours `n > 1` and `logprobs` (it passes them through). The other
  adapters return a single choice.
- Anthropic thinking blocks, prompt-cache breakpoints, and citations aren't exposed. Cache-write
  tokens are priced at the normal input rate, so the real cache-write premium isn't modelled.
  Mid-conversation `system` messages are moved into the top-level system prompt.
- Ollama is priced at $0. The fake provider's token counts are estimates (~4 characters per
  token).
- No mid-stream fallback (by design). There's no per-chunk idle timeout beyond the HTTP
  client's read timeout.
- Token limits are enforced on estimates. The prompt estimate counts ~4 characters per token
  and ignores images, so actual usage can exceed what was reserved. That overshoot is charged
  to the bucket afterwards, so concurrent in-flight requests can briefly go over the limit by
  their estimation error. Requests without `max_tokens` reserve the default allowance, which can
  reject requests that would have fit.
- If a client disconnects before a stream's body is first read, the reservation is not settled
  and returns only through refill (within a minute). Failed provider attempts are refunded in
  full even if the upstream billed for them.
- Provider errors (4xx/5xx from upstream) don't carry `x-ratelimit-*` headers.
- The budget is a soft cap (see the trade-offs above), and budgets are per key, not per tenant.
- Circuit breakers are in-process unless `routing.breaker.backend: redis` is set. Shared
  breakers use each replica's wall clock for the window, cooldown, and lease, so clock skew
  between replicas shifts those by the skew. During a Redis outage each replica falls back to
  its own breakers, starting closed, and the state it builds is discarded when Redis returns.
  `llm_gateway_circuit_state` is per replica, so with `redis` it can lag the shared state on
  replicas that haven't touched a provider since it changed.
- The Dockerfile and compose file pass `docker compose config` validation, but this release
  has not been run under Docker. The test suite has been run locally against real Postgres 17
  and Redis 6.2.
- The OpenAI price list is the author's last verified snapshot. Check it before using the
  costs for billing.

## Roadmap

- Per-tenant budgets with alert thresholds
- Native `/v1/messages` passthrough for Anthropic-specific features
- Async/batched ledger writes and usage export (CSV / Parquet)
- Weighted and latency-aware routing, plus per-request routing hints
- Opt-in semantic cache, per alias
- Importing eval datasets from the ledger
- Embeddings endpoint
- Helm chart

## License

MIT. See [LICENSE](LICENSE).
