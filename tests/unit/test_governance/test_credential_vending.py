import os

import pytest

from spark_ai_functions.governance.credential_vending import (
    EnvCredentialVendor,
    GravitinoCredentialVendor,
    StaticCredentialVendor,
    EndpointMetadataIndex,
    _default_env_name,
)
from spark_ai_functions.governance.errors import CredentialUnavailableError


def test_default_env_name_sanitises():
    assert _default_env_name("gpt-4o-mini") == "SPARK_AI_ENDPOINT_GPT_4O_MINI__API_KEY"


def test_env_vendor_prefers_namespaced(monkeypatch):
    monkeypatch.setenv("SPARK_AI_ENDPOINT_FOO__API_KEY", "sk-foo")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert EnvCredentialVendor().get("foo") == "sk-foo"


def test_env_vendor_falls_back_to_openai(monkeypatch):
    monkeypatch.delenv("SPARK_AI_ENDPOINT_BAR__API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-generic")
    assert EnvCredentialVendor().get("bar") == "sk-generic"


def test_env_vendor_raises_when_missing(monkeypatch):
    monkeypatch.delenv("SPARK_AI_ENDPOINT_BAZ__API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(CredentialUnavailableError):
        EnvCredentialVendor().get("baz")


def test_static_vendor():
    v = StaticCredentialVendor({"a": "1"})
    assert v.get("a") == "1"
    with pytest.raises(CredentialUnavailableError):
        v.get("b")


class _Meta:
    def __init__(self, credential_name, model_id="m", data_residency="external"):
        self.credential_name = credential_name
        self.model_id = model_id
        self.data_residency = data_residency


class _Client:
    def __init__(self, mapping):
        self._m = mapping
        self.calls = 0

    def vend_credential(self, name):
        self.calls += 1
        return self._m[name]


def test_gravitino_credential_vendor_caches():
    idx = EndpointMetadataIndex(loader=lambda n: _Meta(credential_name=f"cred-{n}"))
    client = _Client({"cred-a": "secret"})
    v = GravitinoCredentialVendor(
        gravitino_client=client, catalog="c", endpoint_index=idx, ttl_seconds=10,
    )
    assert v.get("a") == "secret"
    assert v.get("a") == "secret"
    assert client.calls == 1
    assert v.model_id_for("a") == "m"
    assert v.residency_for("a") == "external"


def test_gravitino_credential_vendor_handles_unknown_shape():
    class _WeirdClient:
        def vend_credential(self, name):
            return 42

    idx = EndpointMetadataIndex(loader=lambda n: _Meta(credential_name="x"))
    v = GravitinoCredentialVendor(gravitino_client=_WeirdClient(), catalog="c", endpoint_index=idx)
    with pytest.raises(CredentialUnavailableError):
        v.get("a")
