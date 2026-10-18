"""Provider adapters and the factory that builds them from configuration."""

from __future__ import annotations

from typing import Any

from llm_gateway.catalog import Catalog
from llm_gateway.providers.base import Provider
from llm_gateway.providers.fake import FakeProvider

PROVIDER_TYPES: dict[str, type[Provider]] = {"fake": FakeProvider}


def build_provider(name: str, config: dict[str, Any], catalog: Catalog | None = None) -> Provider:
    ptype = config.get("type", name)
    try:
        cls = PROVIDER_TYPES[ptype]
    except KeyError as exc:
        raise ValueError(f"unknown provider type {ptype!r} for provider {name!r}") from exc
    provider = cls(name, config)
    provider.catalog = catalog or Catalog()
    return provider
