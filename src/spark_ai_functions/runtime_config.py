"""Runtime configuration helpers for endpoint source resolution."""

from __future__ import annotations

import json
import os
from dataclasses import fields as dataclass_fields
from typing import Any, Iterable, Optional

from .endpoints.registry import EndpointConfig, EndpointSource
from .endpoints.yaml_source import InMemoryEndpointSource, YamlEndpointSource

_ENDPOINT_STD_FIELDS = {f.name for f in dataclass_fields(EndpointConfig)} - {"extras"}


def default_yaml_path() -> Optional[str]:
    cwd = os.getcwd()
    for candidate in ("endpoints.yaml", "endpoints.yml"):
        path = os.path.join(cwd, candidate)
        if os.path.exists(path):
            return path
    env = os.environ.get("SPARK_AI_ENDPOINTS_YAML")
    return env if env and os.path.exists(env) else None


def load_endpoints_json_file(path: str) -> list[EndpointConfig]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    return load_endpoints_json_text(raw)


def load_endpoints_json_text(raw: str) -> list[EndpointConfig]:
    payload = json.loads(raw or "{}")
    if isinstance(payload, dict):
        entries = payload.get("endpoints", [])
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError("Endpoint JSON must be an object with 'endpoints' or a list")
    out: list[EndpointConfig] = []
    for e in entries:
        if not isinstance(e, dict):
            raise ValueError("Endpoint entries must be JSON objects")
        out.append(
            EndpointConfig(
                name=e["name"],
                endpoint_type=e["endpoint_type"],
                base_url=e["base_url"],
                model_id=e["model_id"],
                credential_name=e.get("credential_name", "openai"),
                default_params=e.get("default_params", {}) or {},
                data_residency=e.get("data_residency", "external"),
                extras={k: v for k, v in e.items() if k not in _ENDPOINT_STD_FIELDS},
            )
        )
    return out


def resolve_endpoint_sources(
    *,
    yaml_endpoints: Optional[list[EndpointConfig]],
    endpoint_config_file: Optional[str],
    endpoint_config_json: Optional[str],
    yaml_path: Optional[str],
    additional_sources: Optional[Iterable[EndpointSource]],
) -> list[EndpointSource]:
    sources: list[EndpointSource] = []
    if additional_sources:
        sources.extend(additional_sources)
    if yaml_endpoints:
        sources.append(InMemoryEndpointSource(yaml_endpoints))
    if endpoint_config_file:
        sources.append(InMemoryEndpointSource(load_endpoints_json_file(endpoint_config_file)))
    if endpoint_config_json:
        sources.append(InMemoryEndpointSource(load_endpoints_json_text(endpoint_config_json)))
    if yaml_path:
        sources.append(YamlEndpointSource(yaml_path))
    if sources:
        return sources

    json_file = os.environ.get("SPARK_AI_ENDPOINTS_JSON_PATH")
    if json_file:
        if not os.path.exists(json_file):
            raise FileNotFoundError(
                f"SPARK_AI_ENDPOINTS_JSON_PATH is set but file does not exist: {json_file}"
            )
        sources.append(InMemoryEndpointSource(load_endpoints_json_file(json_file)))
        return sources

    json_text = os.environ.get("SPARK_AI_ENDPOINTS_JSON")
    if json_text:
        sources.append(InMemoryEndpointSource(load_endpoints_json_text(json_text)))
        return sources

    default = default_yaml_path()
    if default is not None:
        sources.append(YamlEndpointSource(default))
    else:
        sources.append(InMemoryEndpointSource([]))
    return sources


def register_defaults_from_env() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "yaml_path": os.environ.get("SPARK_AI_ENDPOINTS_YAML"),
        "endpoint_config_json": os.environ.get("SPARK_AI_ENDPOINTS_JSON"),
        "endpoint_config_file": os.environ.get("SPARK_AI_ENDPOINTS_JSON_PATH"),
        "gravitino_uri": os.environ.get("SPARK_AI_GRAVITINO_URI"),
        "metalake": os.environ.get("SPARK_AI_METALAKE"),
        "catalog": os.environ.get("SPARK_AI_CATALOG"),
        "endpoints_schema": os.environ.get("SPARK_AI_ENDPOINTS_SCHEMA") or "endpoints",
        "functions_schema": os.environ.get("SPARK_AI_FUNCTIONS_SCHEMA") or "functions",
        "presets_path": os.environ.get("SPARK_AI_PRESETS_PATH"),
    }
    fn_csv = os.environ.get("SPARK_AI_FUNCTION_NAMES")
    if fn_csv:
        defaults["function_names"] = [p.strip() for p in fn_csv.split(",") if p.strip()]
    skip = os.environ.get("SPARK_AI_SKIP_PLUGIN_CHECK")
    if skip is not None:
        defaults["skip_plugin_check"] = skip.strip().lower() in {"1", "true", "yes", "y"}
    return defaults

