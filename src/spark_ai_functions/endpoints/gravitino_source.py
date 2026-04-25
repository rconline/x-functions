"""Gravitino Model-Catalog endpoint source — Layer 1 primary resolver.

Reads Model entries from a Gravitino Model Catalog and converts them to
`EndpointConfig`s. Per §17.2, package + import names must be exact:

    pip install apache-gravitino
    from gravitino import GravitinoClient

Per §18.6 note, the exact 1.2.x method names (`as_model_catalog`,
`register_model`, ...) may differ slightly — we isolate the calls here so a
Gravitino client bump only touches this file.
"""

from __future__ import annotations

import json
from typing import Iterable, Optional

from .registry import EndpointConfig, EndpointSource


class GravitinoEndpointSource(EndpointSource):
    def __init__(
        self,
        *,
        gravitino_uri: str,
        metalake: str,
        catalog: str,
        schema: str = "endpoints",
        client: Optional[object] = None,
    ):
        self._uri = gravitino_uri
        self._metalake = metalake
        self._catalog = catalog
        self._schema = schema
        self._client = client  # test seam

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from gravitino import GravitinoClient  # apache-gravitino
        except ImportError as e:
            raise RuntimeError(
                "Governed mode requires `apache-gravitino`. "
                "Install with `pip install spark-ai-functions[governance]`."
            ) from e
        self._client = GravitinoClient(uri=self._uri, metalake_name=self._metalake)
        return self._client

    def _schema_handle(self):
        client = self._ensure_client()
        catalog = client.load_catalog(self._catalog)
        # §18.6 — the exact method name can differ across 1.2.x. Try both.
        try:
            model_cat = catalog.as_model_catalog()  # type: ignore[attr-defined]
        except AttributeError:
            model_cat = catalog  # some builds expose the model surface directly
        try:
            return model_cat.load_schema(self._schema)
        except AttributeError:
            return model_cat.schema(self._schema)  # type: ignore[attr-defined]

    def get(self, name: str) -> Optional[EndpointConfig]:
        schema = self._schema_handle()
        try:
            model = schema.load_model(name)  # type: ignore[attr-defined]
        except Exception:
            return None
        return _model_to_config(name, _model_properties(model))

    def list(self) -> Iterable[EndpointConfig]:
        schema = self._schema_handle()
        try:
            names = list(schema.list_models())  # type: ignore[attr-defined]
        except AttributeError:
            return []
        out: list[EndpointConfig] = []
        for n in names:
            try:
                model = schema.load_model(n)  # type: ignore[attr-defined]
                out.append(_model_to_config(n, _model_properties(model)))
            except Exception:
                continue
        return out


def _model_properties(model) -> dict:
    # Python client property access has varied — normalise.
    if hasattr(model, "properties") and callable(model.properties):
        return dict(model.properties() or {})
    if hasattr(model, "properties"):
        return dict(getattr(model, "properties") or {})
    if isinstance(model, dict):
        return dict(model.get("properties") or {})
    return {}


def _model_to_config(name: str, props: dict) -> EndpointConfig:
    default_params_raw = props.get("default_params") or "{}"
    try:
        default_params = json.loads(default_params_raw) if isinstance(default_params_raw, str) else dict(default_params_raw)
    except Exception:
        default_params = {}
    return EndpointConfig(
        name=name,
        endpoint_type=props.get("endpoint_type", "openai_chat"),
        base_url=props.get("base_url", ""),
        model_id=props.get("model_id", name),
        credential_name=props.get("credential_name", ""),
        default_params=default_params,
        data_residency=props.get("data_residency", "external"),
        extras={k: v for k, v in props.items() if k not in {
            "endpoint_type", "base_url", "model_id",
            "credential_name", "default_params", "data_residency",
        }},
    )
