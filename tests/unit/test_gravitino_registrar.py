"""Exercise catalog/gravitino_registrar with an injected client."""

import pytest

from spark_ai_functions.catalog.gravitino_registrar import (
    CANONICAL_FUNCTIONS,
    GravitinoUDFRegistrar,
    register_endpoint_model,
)


class _FnSchema:
    def __init__(self):
        self.registered: dict[str, dict] = {}

    def load_function(self, name):
        if name not in self.registered:
            raise KeyError(name)
        return self.registered[name]

    def register_function(self, name, properties):
        self.registered[name] = dict(properties)

    def alter_function(self, name, properties):
        self.registered[name] = dict(properties)


class _FnCatalog:
    def __init__(self, schema):
        self._s = schema

    def as_function_catalog(self):
        return self

    def load_schema(self, name):  # noqa: ARG002
        return self._s


class _FnClient:
    def __init__(self, schema):
        self._s = schema

    def load_catalog(self, name):  # noqa: ARG002
        return _FnCatalog(self._s)


def test_ensure_registered_writes_every_canonical_function():
    schema = _FnSchema()
    registrar = GravitinoUDFRegistrar(
        gravitino_uri="http://g:8090", metalake="m", catalog="c",
        client=_FnClient(schema),
    )
    touched = registrar.ensure_registered()
    assert set(touched) == {f.name for f in CANONICAL_FUNCTIONS}
    assert "ai_query" in schema.registered
    props = schema.registered["ai_query"]
    assert props["class"] == "spark_ai_functions.core.ai_query.ai_query_impl"
    assert props["engine"] == "pyspark"


def test_ensure_registered_alters_existing_function():
    schema = _FnSchema()
    schema.register_function("ai_query", properties={"class": "stale", "engine": "pyspark"})
    registrar = GravitinoUDFRegistrar(
        gravitino_uri="http://g:8090", metalake="m", catalog="c",
        client=_FnClient(schema),
    )
    registrar.ensure_registered([f for f in CANONICAL_FUNCTIONS if f.name == "ai_query"])
    assert schema.registered["ai_query"]["class"] == "spark_ai_functions.core.ai_query.ai_query_impl"


class _FailingFnSchema(_FnSchema):
    def alter_function(self, name, properties):  # noqa: ARG002
        raise RuntimeError("alter failed")

    def drop_function(self, name):  # noqa: ARG002
        raise RuntimeError("drop failed")


def test_ensure_registered_raises_when_alter_and_replace_fail():
    schema = _FailingFnSchema()
    schema.register_function("ai_query", properties={"class": "stale", "engine": "pyspark"})
    registrar = GravitinoUDFRegistrar(
        gravitino_uri="http://g:8090", metalake="m", catalog="c",
        client=_FnClient(schema),
    )
    with pytest.raises(RuntimeError, match="Failed to update Gravitino function"):
        registrar.ensure_registered([f for f in CANONICAL_FUNCTIONS if f.name == "ai_query"])


class _ExplodingFnSchema(_FnSchema):
    def load_function(self, name):  # noqa: ARG002
        raise RuntimeError("backend unavailable")


def test_ensure_registered_propagates_unexpected_load_errors():
    schema = _ExplodingFnSchema()
    registrar = GravitinoUDFRegistrar(
        gravitino_uri="http://g:8090", metalake="m", catalog="c",
        client=_FnClient(schema),
    )
    with pytest.raises(RuntimeError, match="backend unavailable"):
        registrar.ensure_registered([f for f in CANONICAL_FUNCTIONS if f.name == "ai_query"])


class _ModelSchema:
    def __init__(self):
        self.models: dict[str, dict] = {}

    def register_model(self, name, comment, properties):  # noqa: ARG002
        self.models[name] = dict(properties)


class _ModelCatalog:
    def __init__(self, schema):
        self._s = schema

    def as_model_catalog(self):
        return self

    def load_schema(self, name):  # noqa: ARG002
        return self._s


class _ModelClient:
    def __init__(self, schema):
        self._s = schema

    def load_catalog(self, name):  # noqa: ARG002
        return _ModelCatalog(self._s)


def test_register_endpoint_model_happy_path():
    schema = _ModelSchema()
    register_endpoint_model(
        gravitino_uri="http://g:8090", metalake="m", catalog="c", schema="endpoints",
        name="gpt-4o-mini", endpoint_type="openai_chat",
        base_url="https://api.openai.com/v1", model_id="gpt-4o-mini",
        credential_name="openai-prod",
        default_params={"temperature": 0.0},
        data_residency="external",
        client=_ModelClient(schema),
    )
    props = schema.models["gpt-4o-mini"]
    assert props["endpoint_type"] == "openai_chat"
    assert props["default_params"] == '{"temperature": 0.0}'
    assert props["data_residency"] == "external"


class _BrokenModelCatalog(_ModelCatalog):
    def load_schema(self, name):  # noqa: ARG002
        raise RuntimeError("catalog offline")


class _BrokenModelClient:
    def __init__(self, schema):
        self._s = schema

    def load_catalog(self, name):  # noqa: ARG002
        return _BrokenModelCatalog(self._s)


def test_register_endpoint_model_propagates_unexpected_schema_errors():
    schema = _ModelSchema()
    with pytest.raises(RuntimeError, match="catalog offline"):
        register_endpoint_model(
            gravitino_uri="http://g:8090", metalake="m", catalog="c", schema="endpoints",
            name="gpt-4o-mini", endpoint_type="openai_chat",
            base_url="https://api.openai.com/v1", model_id="gpt-4o-mini",
            credential_name="openai-prod",
            client=_BrokenModelClient(schema),
        )
