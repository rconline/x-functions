"""EndpointConfig.make_backend dispatch — no heavy SDKs actually instantiated
(we only check that the type dispatch picks the right adapter class)."""

import pytest

from spark_ai_functions.endpoints.registry import EndpointConfig


def test_dispatch_openai_chat(monkeypatch):
    # We patch the backend classes to capture which was chosen.
    chosen = {}

    class _FakeChat:
        def __init__(self, *, endpoint_name, **kw):
            chosen["kind"] = "openai_chat"
            chosen["endpoint_name"] = endpoint_name

    class _FakeEmbed:
        def __init__(self, *, endpoint_name, **kw):
            chosen["kind"] = "openai_embedding"

    class _FakeML:
        def __init__(self, *, target_uri, endpoint, default_params):
            chosen["kind"] = "mlflow"

    import spark_ai_functions.endpoints.openai_backend as O
    import spark_ai_functions.endpoints.mlflow_backend as M
    monkeypatch.setattr(O, "OpenAIChatBackend", _FakeChat)
    monkeypatch.setattr(O, "OpenAIEmbeddingBackend", _FakeEmbed)
    monkeypatch.setattr(M, "MLflowDeploymentsBackend", _FakeML)

    cfg = EndpointConfig("e", "openai_chat", "u", "m", "c")
    cfg.make_backend("k")
    assert chosen["kind"] == "openai_chat"

    chosen.clear()
    cfg2 = EndpointConfig("e2", "openai_embedding", "u", "m", "c")
    cfg2.make_backend("k")
    assert chosen["kind"] == "openai_embedding"

    chosen.clear()
    cfg3 = EndpointConfig("e3", "mlflow_chat", "u", "m", "c")
    cfg3.make_backend("k")
    assert chosen["kind"] == "mlflow"


def test_dispatch_unknown_type_raises():
    cfg = EndpointConfig("e", "quantum_chat", "u", "m", "c")
    with pytest.raises(ValueError, match="Unsupported"):
        cfg.make_backend("k")
