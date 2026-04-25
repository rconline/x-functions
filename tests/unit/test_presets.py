import pandas as pd
import pytest

from spark_ai_functions.core.ai_query import _core_ai_query
from spark_ai_functions.endpoints.registry import EndpointConfig, EndpointRegistry
from spark_ai_functions.endpoints.yaml_source import InMemoryEndpointSource
from spark_ai_functions.presets.loader import Preset, load_presets, render_preset, response_format_for


def test_load_presets_ships_all_expected():
    presets = load_presets()
    for expected in ("ai_analyze_sentiment", "ai_classify", "ai_fix_grammar",
                      "ai_summarize", "ai_translate", "ai_gen"):
        assert expected in presets, f"missing preset {expected}"


def test_render_sentiment_preset():
    presets = load_presets()
    msgs = render_preset(presets["ai_analyze_sentiment"], text="I love it")
    assert msgs[0]["role"] == "system"
    assert "sentiment" in msgs[0]["content"].lower()
    assert msgs[1]["content"] == "I love it"


def test_classify_response_format_from_labels():
    presets = load_presets()
    rf = response_format_for(presets["ai_classify"], labels=["spam", "ham"])
    assert rf is not None
    schema = rf["json_schema"]["schema"]
    assert schema["enum"] == ["spam", "ham"]


def test_summarize_honours_max_words_variable():
    preset = load_presets()["ai_summarize"]
    msgs = render_preset(preset, text="long text", max_words=5)
    assert "5" in msgs[0]["content"]


class _Backend:
    def __init__(self): self.calls = []
    def batch_chat_complete(self, series, params):
        self.calls.append((series.tolist(), params))
        return pd.Series(["positive"] * len(series), index=series.index)


def test_preset_flow_through_core_ai_query(monkeypatch):
    presets = load_presets()
    cfg = EndpointConfig("e", "openai_chat", "u", "m", "c")
    backend = _Backend()
    monkeypatch.setattr(cfg, "make_backend", lambda credential: backend)
    registry = EndpointRegistry([InMemoryEndpointSource([cfg])])
    out = _core_ai_query(
        registry=registry,
        endpoint_name="e",
        request_series=pd.Series(["I love it", "meh"]),
        credential="k",
        preset=presets["ai_analyze_sentiment"],
    )
    assert out.tolist() == ["positive", "positive"]
    rendered = backend.calls[0][0]
    assert rendered[0][0]["role"] == "system"
    assert rendered[0][1]["content"] == "I love it"
