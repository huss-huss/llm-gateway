"""Prometheus metrics (exposed at ``GET /metrics``)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120)

REQUESTS = Counter(
    "llm_gateway_requests_total",
    "Chat completion requests by requested model and outcome.",
    ["model", "status", "stream"],
)
REQUEST_LATENCY = Histogram(
    "llm_gateway_request_duration_seconds",
    "End-to-end request latency (to last byte for streams).",
    ["model", "stream"],
    buckets=LATENCY_BUCKETS,
)
PROVIDER_ATTEMPTS = Counter(
    "llm_gateway_provider_attempts_total",
    "Provider calls by outcome (ok, rate_limited, timeout, unavailable, auth, circuit_open...).",
    ["provider", "model", "outcome"],
)
PROVIDER_LATENCY = Histogram(
    "llm_gateway_provider_duration_seconds",
    "Provider call latency (time to first chunk for streams).",
    ["provider", "model"],
    buckets=LATENCY_BUCKETS,
)
TOKENS = Counter(
    "llm_gateway_tokens_total", "Tokens processed.", ["provider", "model", "direction"]
)
COST = Counter("llm_gateway_cost_usd_total", "Estimated spend in USD.", ["provider", "model"])
CACHE = Counter("llm_gateway_cache_requests_total", "Response cache lookups.", ["result"])
REJECTIONS = Counter(
    "llm_gateway_rejections_total", "Requests rejected before routing.", ["reason"]
)
BREAKER_STATE = Gauge(
    "llm_gateway_circuit_state",
    "Circuit state per provider (0=closed, 1=half_open, 2=open).",
    ["provider"],
)

_BREAKER_VALUES = {"closed": 0, "half_open": 1, "open": 2}


def observe_attempt(provider: str, model: str, outcome: str, elapsed_s: float) -> None:
    PROVIDER_ATTEMPTS.labels(provider, model, outcome).inc()
    if outcome != "circuit_open":
        PROVIDER_LATENCY.labels(provider, model).observe(elapsed_s)


def observe_breaker(provider: str, state: str) -> None:
    BREAKER_STATE.labels(provider).set(_BREAKER_VALUES.get(state, 0))


def observe_completion(
    *,
    model: str,
    status: str,
    stream: bool,
    elapsed_s: float,
    provider: str | None = None,
    target_model: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    s = "true" if stream else "false"
    REQUESTS.labels(model, status, s).inc()
    REQUEST_LATENCY.labels(model, s).observe(elapsed_s)
    if provider and target_model:
        if prompt_tokens:
            TOKENS.labels(provider, target_model, "input").inc(prompt_tokens)
        if completion_tokens:
            TOKENS.labels(provider, target_model, "output").inc(completion_tokens)
        if cost_usd:
            COST.labels(provider, target_model).inc(cost_usd)
