"""Exercise remaining return-type branches in _core_ai_query."""

import pandas as pd

from spark_ai_functions.core.ai_query import _core_ai_query
from spark_ai_functions.endpoints.registry import EndpointConfig, EndpointRegistry
from spark_ai_functions.endpoints.yaml_source import InMemoryEndpointSource


class _FakeChat:
    def __init__(self, canned):
        self._c = list(canned)

    def batch_chat_complete(self, series, params):
        return pd.Series(self._c[: len(series)], index=series.index)


def _reg(monkeypatch, canned):
    cfg = EndpointConfig("t", "openai_chat", "u", "m", "c")
    monkeypatch.setattr(cfg, "make_backend", lambda credential: _FakeChat(canned))
    return EndpointRegistry([InMemoryEndpointSource([cfg])])


def test_return_type_int(monkeypatch):
    reg = _reg(monkeypatch, ["1", "2.9", "3"])
    out = _core_ai_query(
        registry=reg, endpoint_name="t",
        request_series=pd.Series(["a", "b", "c"]), credential="k",
        return_type="INT",
    )
    assert out.tolist() == [1, 2, 3]


def test_return_type_boolean(monkeypatch):
    reg = _reg(monkeypatch, ["true", "no", "YES"])
    out = _core_ai_query(
        registry=reg, endpoint_name="t",
        request_series=pd.Series(["a", "b", "c"]), credential="k",
        return_type="BOOLEAN",
    )
    assert out.tolist() == [True, False, True]


def test_return_type_unknown_passes_through(monkeypatch):
    reg = _reg(monkeypatch, ["x", "y"])
    out = _core_ai_query(
        registry=reg, endpoint_name="t",
        request_series=pd.Series(["a", "b"]), credential="k",
        return_type="DATE",  # unsupported; impl should not crash
    )
    assert out.tolist() == ["x", "y"]


def test_response_format_string_wraps_as_dict(monkeypatch):
    captured = {}

    class _Spy:
        def batch_chat_complete(self, series, params):
            captured.update(params)
            return pd.Series(["ok"] * len(series), index=series.index)

    cfg = EndpointConfig("t", "openai_chat", "u", "m", "c")
    monkeypatch.setattr(cfg, "make_backend", lambda credential: _Spy())
    reg = EndpointRegistry([InMemoryEndpointSource([cfg])])
    _core_ai_query(
        registry=reg, endpoint_name="t",
        request_series=pd.Series(["a"]), credential="k",
        response_format="json_object",
    )
    assert captured["response_format"] == {"type": "json_object"}


def test_fail_on_error_false_preserves_none(monkeypatch):
    class _NoneBackend:
        def batch_chat_complete(self, series, params):
            return pd.Series([None] * len(series), index=series.index)

    cfg = EndpointConfig("t", "openai_chat", "u", "m", "c")
    monkeypatch.setattr(cfg, "make_backend", lambda credential: _NoneBackend())
    reg = EndpointRegistry([InMemoryEndpointSource([cfg])])
    out = _core_ai_query(
        registry=reg, endpoint_name="t",
        request_series=pd.Series(["a"]), credential="k",
        fail_on_error=False,
    )
    assert out.iloc[0] is None
