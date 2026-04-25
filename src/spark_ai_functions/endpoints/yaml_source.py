"""YAML file endpoint source — the standalone-mode fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import yaml

from .registry import EndpointConfig, EndpointSource


class YamlEndpointSource(EndpointSource):
    """Reads a YAML file with shape:

        endpoints:
          - name: gpt-4o-mini
            endpoint_type: openai_chat
            base_url: https://api.openai.com/v1
            model_id: gpt-4o-mini
            credential_name: openai
            default_params: {temperature: 0.0, max_tokens: 1024}
            data_residency: external
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._cache: Optional[dict[str, EndpointConfig]] = None

    def _load(self) -> dict[str, EndpointConfig]:
        if self._cache is not None:
            return self._cache
        if not self._path.exists():
            raise FileNotFoundError(f"Endpoint YAML not found: {self._path}")
        raw = yaml.safe_load(self._path.read_text()) or {}
        entries = raw.get("endpoints", [])
        out: dict[str, EndpointConfig] = {}
        for e in entries:
            cfg = EndpointConfig(
                name=e["name"],
                endpoint_type=e["endpoint_type"],
                base_url=e["base_url"],
                model_id=e["model_id"],
                credential_name=e.get("credential_name", "openai"),
                default_params=e.get("default_params", {}) or {},
                data_residency=e.get("data_residency", "external"),
                extras={k: v for k, v in e.items() if k not in {
                    "name", "endpoint_type", "base_url", "model_id",
                    "credential_name", "default_params", "data_residency",
                }},
            )
            out[cfg.name] = cfg
        self._cache = out
        return out

    def get(self, name: str) -> Optional[EndpointConfig]:
        return self._load().get(name)

    def list(self) -> Iterable[EndpointConfig]:
        return list(self._load().values())


class InMemoryEndpointSource(EndpointSource):
    """Tests and register(yaml_endpoints=[...])."""

    def __init__(self, configs: Iterable[EndpointConfig]):
        self._m = {c.name: c for c in configs}

    def get(self, name: str) -> Optional[EndpointConfig]:
        return self._m.get(name)

    def list(self) -> Iterable[EndpointConfig]:
        return list(self._m.values())
