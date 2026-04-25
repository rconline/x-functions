"""Unit tests for backend helper functions that don't need real SDKs."""

import pandas as pd

from spark_ai_functions.endpoints.mlflow_backend import (
    _coerce_messages as _ml_coerce,
    _extract_chat,
    _extract_embedding,
)
from spark_ai_functions.endpoints.openai_backend import (
    OpenAIChatBackend,
    _coerce_messages,
    _openai_params,
)


def test_coerce_messages_from_string():
    assert _coerce_messages("hi") == [{"role": "user", "content": "hi"}]


def test_coerce_messages_passthrough_list():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert _coerce_messages(msgs) == msgs


def test_coerce_messages_stringifies_other():
    assert _coerce_messages(42) == [{"role": "user", "content": "42"}]


def test_openai_params_drops_adapter_fields():
    params = {
        "temperature": 0.0,
        "max_concurrency": 4,
        "concurrency": 4,
        "timeout": 30,
        "retry_policy": {},
    }
    out = _openai_params(params)
    assert "temperature" in out
    assert "max_concurrency" not in out
    assert "concurrency" not in out
    assert "timeout" not in out


def test_openai_params_embedding_mode_drops_chat_fields():
    out = _openai_params(
        {"temperature": 0.0, "max_tokens": 50, "response_format": {"type": "json_object"}},
        embedding=True,
    )
    assert out == {}


def test_openai_chat_backend_dispatches_each_row(monkeypatch):
    """Smoke test: the chat backend calls the SDK once per row and returns
    a Series indexed identically to the input."""
    import sys
    import types

    seen: list[str] = []

    class _FakeChatCompletions:
        def create(self, *, model, messages, **kw):
            seen.append(messages[-1]["content"])
            choice = types.SimpleNamespace(
                message=types.SimpleNamespace(content=messages[-1]["content"].upper())
            )
            return types.SimpleNamespace(choices=[choice])

    class _FakeClient:
        def __init__(self, *_, **__):
            self.chat = types.SimpleNamespace(completions=_FakeChatCompletions())

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    be = OpenAIChatBackend(
        endpoint_name="t",
        base_url="https://example.invalid/v1",
        model_id="m",
        credential="k",
        default_params={"max_concurrency": 4},
    )
    s = pd.Series(["a", "b", "c", "d"], index=[10, 11, 12, 13])
    out = be.batch_chat_complete(s, {})
    assert out.tolist() == ["A", "B", "C", "D"]
    assert list(out.index) == [10, 11, 12, 13]
    assert sorted(seen) == ["a", "b", "c", "d"]


def test_mlflow_extract_chat_openai_shape():
    resp = {"choices": [{"message": {"content": "hi"}}]}
    assert _extract_chat(resp) == "hi"


def test_mlflow_extract_chat_raw_string():
    assert _extract_chat("plain") == "plain"


def test_mlflow_extract_embedding_list():
    assert _extract_embedding([0.1, 0.2]) == [0.1, 0.2]


def test_mlflow_extract_embedding_openai_shape():
    assert _extract_embedding({"data": [{"embedding": [1, 2, 3]}]}) == [1, 2, 3]


def test_mlflow_extract_embedding_direct_key():
    assert _extract_embedding({"embedding": [9, 9]}) == [9, 9]


def test_mlflow_coerce_messages_string():
    assert _ml_coerce("x") == [{"role": "user", "content": "x"}]
