"""Behavioral tests for register() wiring and governed plugin checks."""

from __future__ import annotations

import importlib

import pytest

from spark_ai_functions.endpoints.registry import EndpointConfig
from spark_ai_functions.endpoints.yaml_source import InMemoryEndpointSource
from spark_ai_functions.governance.audit import InMemoryAuditSink
from spark_ai_functions.governance.credential_vending import StaticCredentialVendor
from spark_ai_functions.governance.ranger_authorizer import AllowAllAuthorizer
from spark_ai_functions.governance.tag_policy import PassThroughTagPolicyEnforcer
from spark_ai_functions.governance.user_resolver import ExplicitUserResolver

register_module = importlib.import_module("spark_ai_functions.register")


class _SparkConf:
    def __init__(self, plugins: str = ""):
        self._plugins = plugins

    def get(self, key: str, default: str = "") -> str:
        if key == "spark.plugins":
            return self._plugins
        return default


class _FakeSpark:
    def __init__(self, plugins: str = ""):
        self.conf = _SparkConf(plugins=plugins)


def _common_overrides():
    return dict(
        audit_sink=InMemoryAuditSink(),
        authorizer=AllowAllAuthorizer(),
        tag_policy_enforcer=PassThroughTagPolicyEnforcer(),
        credential_vendor=StaticCredentialVendor({"e": "k"}),
    )


def test_register_standalone_happy_path(monkeypatch):
    cfg = EndpointConfig("e", "openai_chat", "u", "m", "c")
    monkeypatch.setattr(
        register_module,
        "resolve_endpoint_sources",
        lambda **kwargs: [InMemoryEndpointSource([cfg])],  # noqa: ARG005
    )
    monkeypatch.setattr(register_module, "load_presets", lambda *a, **k: {})  # noqa: ARG005
    monkeypatch.setattr(
        register_module, "default_chain", lambda user: ExplicitUserResolver(user or "u")
    )
    monkeypatch.setattr(
        register_module, "register_udfs", lambda spark, registry, presets, names: ["ai_query"]  # noqa: ARG005
    )

    out = register_module.register(_FakeSpark(), **_common_overrides())

    assert out.mode == "standalone"
    assert out.registered_function_names == ["ai_query"]


def test_register_governed_invokes_plugin_check_and_registrar(monkeypatch):
    calls = {"plugin_check": 0, "registrar": 0}

    monkeypatch.setattr(register_module, "load_presets", lambda *a, **k: {})  # noqa: ARG005
    monkeypatch.setattr(register_module, "resolve_endpoint_sources", lambda **kwargs: [])  # noqa: ARG005
    monkeypatch.setattr(
        register_module, "default_chain", lambda user: ExplicitUserResolver(user or "u")
    )
    monkeypatch.setattr(
        register_module, "register_udfs", lambda spark, registry, presets, names: []  # noqa: ARG005
    )

    def _fake_assert_plugin_loaded(spark, *, skip):
        calls["plugin_check"] += 1
        assert skip is True

    monkeypatch.setattr(register_module, "_assert_plugin_loaded", _fake_assert_plugin_loaded)

    import spark_ai_functions.catalog.gravitino_registrar as registrar_mod

    class _FakeRegistrar:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

        def ensure_registered(self):
            calls["registrar"] += 1
            return ["ai_query"]

    monkeypatch.setattr(registrar_mod, "GravitinoUDFRegistrar", _FakeRegistrar)

    out = register_module.register(
        _FakeSpark(),
        gravitino_uri="http://g:8090",
        metalake="m",
        catalog="c",
        skip_plugin_check=True,
        **_common_overrides(),
    )

    assert out.mode == "governed"
    assert calls["plugin_check"] == 1
    assert calls["registrar"] == 1


def test_assert_plugin_loaded_raises_when_plugin_missing():
    spark = _FakeSpark(plugins="")
    with pytest.raises(RuntimeError, match="Governed mode requires the Gravitino Spark plugin"):
        register_module._assert_plugin_loaded(spark, skip=False)  # type: ignore[attr-defined]
