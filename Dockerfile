FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first for layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY config ./config
COPY migrations ./migrations
COPY evals ./evals

RUN useradd --system --uid 10001 gateway
USER gateway

ENV GATEWAY_HOST=0.0.0.0 \
    GATEWAY_PORT=8080 \
    GATEWAY_MIGRATIONS_DIR=/app/migrations

EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz')"

# Migrations run on start; with several replicas, run `llm-gateway db upgrade` as a job instead.
CMD ["sh", "-c", "llm-gateway db upgrade && exec llm-gateway serve"]
