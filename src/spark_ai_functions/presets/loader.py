"""Load & render the preset prompts shipped with the package."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class Preset:
    name: str
    system: str
    user_template: str
    response_format: Optional[dict[str, Any]] = None
    response_format_from_labels: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


def load_presets(path: Optional[str | Path] = None) -> dict[str, Preset]:
    if path is None:
        text = files("spark_ai_functions.presets").joinpath("prompts.yaml").read_text()
    else:
        text = Path(path).read_text()
    data = yaml.safe_load(text) or {}
    out: dict[str, Preset] = {}
    for name, entry in data.items():
        out[name] = Preset(
            name=name,
            system=entry.get("system", ""),
            user_template=entry.get("user_template", "{text}"),
            response_format=entry.get("response_format"),
            response_format_from_labels=bool(entry.get("response_format_from_labels", False)),
            raw=dict(entry),
        )
    return out


def render_preset(preset: Preset, **vars: Any) -> list[dict[str, str]]:
    """Produce the OpenAI-style messages array for a row's variables."""
    system = preset.system.format(**{k: v for k, v in vars.items() if v is not None})
    user = preset.user_template.format(**vars)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def response_format_for(preset: Preset, labels: Optional[list[str]] = None) -> Optional[dict[str, Any]]:
    """Build the `response_format` payload, including the dynamic label enum."""
    if preset.response_format is not None:
        return preset.response_format
    if preset.response_format_from_labels and labels:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": preset.name,
                "schema": {"type": "string", "enum": list(labels)},
            },
        }
    return None
