"""Exercise gravitino_source with a fake client — no real gravitino package needed."""

from spark_ai_functions.endpoints.gravitino_source import (
    GravitinoEndpointSource,
    _model_to_config,
)


def test_model_to_config_parses_default_params_json():
    cfg = _model_to_config(
        "gpt-4o-mini",
        {
            "endpoint_type": "openai_chat",
            "base_url": "https://api.openai.com/v1",
            "model_id": "gpt-4o-mini",
            "credential_name": "openai-prod",
            "default_params": '{"temperature": 0.1}',
            "data_residency": "external",
        },
    )
    assert cfg.name == "gpt-4o-mini"
    assert cfg.endpoint_type == "openai_chat"
    assert cfg.default_params == {"temperature": 0.1}
    assert cfg.data_residency == "external"


def test_model_to_config_empty_params_still_works():
    cfg = _model_to_config("n", {})
    assert cfg.endpoint_type == "openai_chat"
    assert cfg.default_params == {}


def test_model_to_config_invalid_json_defaults_to_empty():
    cfg = _model_to_config("n", {"default_params": "not{json"})
    assert cfg.default_params == {}


class _Model:
    def __init__(self, props):
        self._p = dict(props)

    def properties(self):
        return self._p


class _Schema:
    def __init__(self, models):
        self._m = models

    def load_model(self, name):
        if name not in self._m:
            raise RuntimeError(f"no model {name!r}")
        return _Model(self._m[name])

    def list_models(self):
        return list(self._m.keys())


class _Catalog:
    def __init__(self, schema):
        self._s = schema

    def as_model_catalog(self):
        return self

    def load_schema(self, name):  # noqa: ARG002
        return self._s


class _Client:
    def __init__(self, schema):
        self._s = schema

    def load_catalog(self, name):  # noqa: ARG002
        return _Catalog(self._s)


def test_gravitino_endpoint_source_get_and_list():
    models = {
        "gpt-4o-mini": {
            "endpoint_type": "openai_chat",
            "base_url": "https://api.openai.com/v1",
            "model_id": "gpt-4o-mini",
            "credential_name": "k",
            "default_params": '{"temperature": 0.0}',
            "data_residency": "external",
        },
        "embed-small": {
            "endpoint_type": "openai_embedding",
            "base_url": "https://api.openai.com/v1",
            "model_id": "text-embedding-3-small",
            "credential_name": "k",
        },
    }
    src = GravitinoEndpointSource(
        gravitino_uri="http://g:8090", metalake="m", catalog="c",
        client=_Client(_Schema(models)),
    )
    got = src.get("gpt-4o-mini")
    assert got is not None and got.endpoint_type == "openai_chat"
    assert src.get("missing") is None
    names = {c.name for c in src.list()}
    assert names == {"gpt-4o-mini", "embed-small"}
