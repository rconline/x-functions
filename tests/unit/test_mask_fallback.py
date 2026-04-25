"""Tests for the LLM fallback path in _core_ai_mask."""

import pandas as pd

from spark_ai_functions.core.mask import _core_ai_mask
from spark_ai_functions.endpoints.registry import EndpointConfig, EndpointRegistry
from spark_ai_functions.endpoints.yaml_source import InMemoryEndpointSource
from spark_ai_functions.presets.loader import Preset


class _NoopMasker:
    """Returns input unchanged (simulates Presidio finding nothing)."""

    def mask(self, text, entities=None, language="en"):
        return text

    def mask_series(self, s):
        return s


class _FakeChat:
    def batch_chat_complete(self, series, params):
        # Rewrite every input as "LLM_MASKED".
        return pd.Series(["LLM_MASKED"] * len(series), index=series.index)


def test_llm_fallback_triggers_when_presidio_no_op(monkeypatch):
    cfg = EndpointConfig("e", "openai_chat", "u", "m", "c")
    monkeypatch.setattr(cfg, "make_backend", lambda credential: _FakeChat())
    reg = EndpointRegistry([InMemoryEndpointSource([cfg])])

    preset = Preset(
        name="ai_mask",
        system="mask this",
        user_template="Entity types to mask: {labels}\n\nText: {text}",
    )
    out = _core_ai_mask(
        registry=reg, endpoint_name="e",
        text=pd.Series(["foo", "bar"]), labels=["PERSON"],
        credential="k", masker=_NoopMasker(), preset=preset,
    )
    assert out.tolist() == ["LLM_MASKED", "LLM_MASKED"]


def test_llm_fallback_skipped_when_preset_or_endpoint_missing():
    out = _core_ai_mask(
        registry=None, endpoint_name=None,
        text=pd.Series(["foo"]), labels=None, credential=None,
        masker=_NoopMasker(), preset=None,
    )
    assert out.tolist() == ["foo"]
