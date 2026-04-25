from pathlib import Path

import pytest
import yaml

from spark_ai_functions.endpoints.registry import EndpointConfig, EndpointRegistry
from spark_ai_functions.endpoints.yaml_source import InMemoryEndpointSource, YamlEndpointSource
from spark_ai_functions.governance.errors import EndpointNotFoundError


def test_registry_returns_first_hit():
    a = EndpointConfig("x", "openai_chat", "u", "m", "c")
    b = EndpointConfig("y", "openai_chat", "u", "m", "c")
    reg = EndpointRegistry([InMemoryEndpointSource([a]), InMemoryEndpointSource([b])])
    assert reg.get("x") is a
    assert reg.get("y") is b


def test_registry_raises_when_missing():
    reg = EndpointRegistry([InMemoryEndpointSource([])])
    with pytest.raises(EndpointNotFoundError):
        reg.get("nope")


def test_yaml_source_roundtrip(tmp_path: Path):
    path = tmp_path / "endpoints.yaml"
    path.write_text(yaml.safe_dump({
        "endpoints": [
            {
                "name": "e1",
                "endpoint_type": "openai_chat",
                "base_url": "https://api",
                "model_id": "gpt",
                "credential_name": "k",
                "default_params": {"temperature": 0},
                "data_residency": "external",
            }
        ]
    }))
    src = YamlEndpointSource(str(path))
    cfg = src.get("e1")
    assert cfg is not None
    assert cfg.endpoint_type == "openai_chat"
    assert cfg.default_params == {"temperature": 0}
    assert src.get("absent") is None
    assert len(list(src.list())) == 1


def test_list_dedupes_across_sources():
    a1 = EndpointConfig("shared", "openai_chat", "u", "m", "c")
    a2 = EndpointConfig("shared", "openai_chat", "u2", "m2", "c2")
    reg = EndpointRegistry([InMemoryEndpointSource([a1]), InMemoryEndpointSource([a2])])
    names = [c.name for c in reg.list_all()]
    assert names == ["shared"]
