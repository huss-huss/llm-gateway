"""OpenTelemetry tracing setup. Exporter: ``none`` (default), ``console``, or ``otlp`` (HTTP)."""

from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from llm_gateway import __version__

log = logging.getLogger(__name__)
tracer = trace.get_tracer("llm_gateway", __version__)
_configured = False


def setup_tracing(exporter: str, service_name: str, endpoint: str | None = None) -> None:
    """Install a global tracer provider once per process. With ``none`` spans are no-ops
    (unless something else, e.g. a test, installed a provider)."""
    global _configured
    if _configured or exporter == "none":
        return
    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name, "service.version": __version__})
    )
    if exporter == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        # Endpoint defaults to OTEL_EXPORTER_OTLP_ENDPOINT / http://localhost:4318.
        url = f"{endpoint.rstrip('/')}/v1/traces" if endpoint else None
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=url)))
    else:
        raise ValueError(f"unknown OTEL exporter {exporter!r}")
    trace.set_tracer_provider(provider)
    _configured = True
    log.info("tracing enabled", extra={"exporter": exporter})
