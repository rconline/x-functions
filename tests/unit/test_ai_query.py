"""§18.5 FakeBackend pattern — no Spark, no HTTP."""

import pandas as pd
import pytest

from spark_ai_functions.core.ai_query import _core_ai_query
from spark_ai_functions.endpoints.registry import EndpointConfig, EndpointRegistry
from spark_ai_functions.endpoints.yaml_source import InMemoryEndpointSource


class FakeChatBackend:
    def __init__(self, canned):
        self._canned = list(canned)

    def batch_chat_complete(self, series, params):
        return pd.Series(self._canned[: len(series)], index=series.index)


def _registry(monkeypatch, canned):
    cfg = EndpointConfig(
        name="t",
        endpoint_type="openai_chat",
        base_url="x",
        model_id="x",
        credential_name="x",
        default_params={},
        data_residency="external",
    )
    monkeypatch.setattr(cfg, "make_backend", lambda credential: FakeChatBackend(canned))
    return EndpointRegistry([InMemoryEndpointSource([cfg])])


def test_ai_query_happy_path(monkeypatch):
    registry = _registry(monkeypatch, ["A", "B", "C"])
    result = _core_ai_query(
        registry=registry,
        endpoint_name="t",
        request_series=pd.Series(["x", "y", "z"]),
        credential="fake",
    )
    assert result.tolist() == ["A", "B", "C"]


def test_ai_query_return_type_double(monkeypatch):
    registry = _registry(monkeypatch, ["1.5", "2.5", "3.5"])
    out = _core_ai_query(
        registry=registry,
        endpoint_name="t",
        request_series=pd.Series(["a", "b", "c"]),
        credential="fake",
        return_type="DOUBLE",
    )
    assert out.tolist() == [1.5, 2.5, 3.5]


def test_ai_query_return_type_json(monkeypatch):
    registry = _registry(monkeypatch, ['{"a":1}', "oops", '{"b":2}'])
    out = _core_ai_query(
        registry=registry,
        endpoint_name="t",
        request_series=pd.Series(["x", "y", "z"]),
        credential="fake",
        return_type="JSON",
    )
    assert out.tolist() == [{"a": 1}, {"raw": "oops"}, {"b": 2}]


def test_ai_query_applies_response_format(monkeypatch):
    captured = {}

    class SpyBackend(FakeChatBackend):
        def batch_chat_complete(self, series, params):
            captured.update(params)
            return pd.Series(["ok"] * len(series), index=series.index)

    cfg = EndpointConfig("t", "openai_chat", "u", "m", "c")
    monkeypatch.setattr(cfg, "make_backend", lambda credential: SpyBackend([]))
    registry = EndpointRegistry([InMemoryEndpointSource([cfg])])
    _core_ai_query(
        registry=registry,
        endpoint_name="t",
        request_series=pd.Series(["hi"]),
        credential="k",
        response_format={"type": "json_object"},
    )
    assert captured["response_format"] == {"type": "json_object"}
