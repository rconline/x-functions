"""Service-config helpers for register(): env and JSON endpoint payloads."""

from __future__ import annotations

import importlib

import pytest

register_module = importlib.import_module("spark_ai_functions.register")


def test_load_endpoints_json_text_object_shape():
    cfgs = register_module._load_endpoints_json_text(  # type: ignore[attr-defined]
        """
        {
          "endpoints": [
            {
              "name": "internal-llm",
              "endpoint_type": "openai_chat",
              "base_url": "https://llm.internal/v1",
              "model_id": "llama-3.3-70b",
              "credential_name": "internal-llm",
              "default_params": {"temperature": 0.0},
              "data_residency": "internal"
            }
          ]
        }
        """
    )
    assert len(cfgs) == 1
    assert cfgs[0].name == "internal-llm"
    assert cfgs[0].endpoint_type == "openai_chat"
    assert cfgs[0].data_residency == "internal"
    assert cfgs[0].default_params["temperature"] == 0.0


def test_load_endpoints_json_text_list_shape():
    cfgs = register_module._load_endpoints_json_text(  # type: ignore[attr-defined]
        """
        [
          {
            "name": "bedrock-sonnet",
            "endpoint_type": "bedrock_chat",
            "base_url": "bedrock://us-west-2",
            "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "credential_name": "bedrock-prod"
          }
        ]
        """
    )
    assert len(cfgs) == 1
    assert cfgs[0].name == "bedrock-sonnet"
    assert cfgs[0].endpoint_type == "bedrock_chat"


def test_load_endpoints_json_text_malformed_raises():
    with pytest.raises(Exception):
        register_module._load_endpoints_json_text("not-json")  # type: ignore[attr-defined]


def test_load_endpoints_json_text_populates_extras():
    cfgs = register_module._load_endpoints_json_text(  # type: ignore[attr-defined]
        """
        {"endpoints":[
          {
            "name":"e",
            "endpoint_type":"openai_chat",
            "base_url":"https://api.openai.com/v1",
            "model_id":"gpt-4o-mini",
            "credential_name":"openai",
            "tenant":"finance"
          }
        ]}
        """
    )
    assert cfgs[0].extras["tenant"] == "finance"


def test_register_from_env_standalone_mode_clears_governed_settings(monkeypatch):
    captured = {}

    def _fake_register(spark, **kwargs):
        captured["spark"] = spark
        captured["kwargs"] = kwargs
        return "ok"

    monkeypatch.setenv("SPARK_AI_MODE", "standalone")
    monkeypatch.setenv("SPARK_AI_GRAVITINO_URI", "http://gravitino:8090")
    monkeypatch.setenv("SPARK_AI_METALAKE", "prod")
    monkeypatch.setenv("SPARK_AI_CATALOG", "ai_functions")
    monkeypatch.setenv("SPARK_AI_ENDPOINTS_JSON", '{"endpoints":[]}')
    monkeypatch.setattr(register_module, "register", _fake_register)

    out = register_module.register_from_env("spark-session")

    assert out == "ok"
    assert captured["spark"] == "spark-session"
    assert captured["kwargs"]["gravitino_uri"] is None
    assert captured["kwargs"]["metalake"] is None
    assert captured["kwargs"]["catalog"] is None
    assert captured["kwargs"]["endpoint_config_json"] == '{"endpoints":[]}'


def test_register_from_env_governed_requires_all_keys(monkeypatch):
    monkeypatch.setenv("SPARK_AI_MODE", "governed")
    monkeypatch.delenv("SPARK_AI_GRAVITINO_URI", raising=False)
    monkeypatch.setenv("SPARK_AI_METALAKE", "prod")
    monkeypatch.delenv("SPARK_AI_CATALOG", raising=False)
    with pytest.raises(ValueError, match="SPARK_AI_GRAVITINO_URI"):
        register_module.register_from_env("spark-session")


def test_register_raises_for_missing_env_json_path(monkeypatch):
    class _DummySpark:
        pass

    monkeypatch.setenv("SPARK_AI_ENDPOINTS_JSON_PATH", "/tmp/does-not-exist-xyz.json")
    monkeypatch.delenv("SPARK_AI_ENDPOINTS_JSON", raising=False)
    monkeypatch.delenv("SPARK_AI_ENDPOINTS_YAML", raising=False)
    with pytest.raises(FileNotFoundError, match="SPARK_AI_ENDPOINTS_JSON_PATH"):
        register_module.register(_DummySpark())
