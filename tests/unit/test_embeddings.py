import math

import pandas as pd
import pytest

from spark_ai_functions.core.embeddings import _core_prep_search, _core_similarity, _cosine
from spark_ai_functions.endpoints.registry import EndpointConfig, EndpointRegistry
from spark_ai_functions.endpoints.yaml_source import InMemoryEndpointSource


class _EmbBackend:
    def __init__(self, lookup):
        self._lookup = lookup

    def batch_embed(self, texts, params):
        return pd.Series([self._lookup[str(t)] for t in texts], index=texts.index)


def _registry(monkeypatch, lookup):
    cfg = EndpointConfig("e", "openai_embedding", "u", "m", "c")
    monkeypatch.setattr(cfg, "make_backend", lambda credential: _EmbBackend(lookup))
    return EndpointRegistry([InMemoryEndpointSource([cfg])])


def test_cosine_identical():
    assert _cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal():
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector():
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_similarity_flow(monkeypatch):
    reg = _registry(monkeypatch, {
        "cats": [1.0, 0.0, 0.0],
        "kittens": [1.0, 0.0, 0.0],
        "dogs": [0.0, 1.0, 0.0],
    })
    out = _core_similarity(
        registry=reg, endpoint_name="e", credential="k",
        a=pd.Series(["cats", "cats"]), b=pd.Series(["kittens", "dogs"]),
    )
    assert out.iloc[0] == pytest.approx(1.0)
    assert out.iloc[1] == pytest.approx(0.0)


def test_prep_search_chunks_and_embeds(monkeypatch):
    # Each chunk gets a unique embedding based on its text.
    def lookup(text):
        return [float(len(text))]

    class _BackendFn:
        def batch_embed(self, texts, params):
            return pd.Series([lookup(t) for t in texts], index=texts.index)

    cfg = EndpointConfig("e", "openai_embedding", "u", "m", "c")
    monkeypatch.setattr(cfg, "make_backend", lambda credential: _BackendFn())
    reg = EndpointRegistry([InMemoryEndpointSource([cfg])])
    out = _core_prep_search(
        registry=reg, endpoint_name="e", credential="k",
        texts=pd.Series(["abcdef", "xyz"]),
        chunk_size=3, chunk_overlap=1,
    )
    assert len(out.iloc[0]) >= 2
    assert isinstance(out.iloc[0][0]["chunk"], str)
    assert isinstance(out.iloc[0][0]["embedding"], list)


def test_prep_search_validates_overlap():
    cfg = EndpointConfig("e", "openai_embedding", "u", "m", "c")
    reg = EndpointRegistry([InMemoryEndpointSource([cfg])])
    with pytest.raises(ValueError):
        _core_prep_search(
            registry=reg, endpoint_name="e", credential="k",
            texts=pd.Series(["abc"]), chunk_size=3, chunk_overlap=5,
        )
