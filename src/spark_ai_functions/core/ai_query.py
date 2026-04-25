"""`ai_query` — generic LLM call + preset-driven variants.

Two layers:
  - `_core_ai_query(...)` — the pure function used by unit tests (§18.5).
  - `ai_query_impl(...)` / `ai_<preset>_impl(...)` — the `@governed`-decorated
    wrappers the register step wires to `spark.udf.register` (and that
    Gravitino references by fully-qualified class path).
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd

from ..endpoints.registry import EndpointRegistry
from ..governance.decorator import governed
from ..presets.loader import Preset, load_presets, render_preset, response_format_for


def _core_ai_query(
    *,
    registry: EndpointRegistry,
    endpoint_name: str,
    request_series: pd.Series,
    credential: str,
    return_type: Optional[str] = None,
    fail_on_error: bool = True,
    model_parameters: Optional[dict[str, Any]] = None,
    response_format: Optional[dict[str, Any] | str] = None,
    preset: Optional[Preset] = None,
    preset_vars: Optional[list[dict[str, Any]]] = None,
) -> pd.Series:
    """Pure function — takes a Series in, returns a Series out.

    `request_series` is either a series of raw strings or (when `preset` is
    provided) arbitrary per-row var dicts are constructed by the caller and
    rendered into OpenAI-style messages here.
    """
    params: dict[str, Any] = dict(model_parameters or {})
    if isinstance(response_format, str):
        params["response_format"] = {"type": response_format}
    elif isinstance(response_format, dict):
        params["response_format"] = response_format

    backend = registry.make_backend(endpoint_name, credential)

    if preset is not None:
        if preset_vars is None:
            preset_vars = [{"text": v} for v in request_series.tolist()]
        rendered = pd.Series(
            [render_preset(preset, **v) for v in preset_vars],
            index=request_series.index,
        )
        rf = response_format_for(preset, labels=(preset_vars[0].get("labels") if preset_vars else None))
        if rf is not None and "response_format" not in params:
            params["response_format"] = rf
        raw = backend.batch_chat_complete(rendered, params)  # type: ignore[attr-defined]
    else:
        raw = backend.batch_chat_complete(request_series, params)  # type: ignore[attr-defined]

    if not fail_on_error:
        raw = raw.where(raw.notna(), None)

    if return_type in (None, "STRING", "string"):
        return raw.astype(object)
    if return_type in ("JSON", "json"):
        return raw.map(_safe_json)
    # Best-effort type coercion for primitive returns.
    if return_type in ("DOUBLE", "double"):
        return raw.map(lambda v: None if v is None else float(v))
    if return_type in ("INT", "int", "INTEGER", "integer"):
        return raw.map(lambda v: None if v is None else int(float(v)))
    if return_type in ("BOOLEAN", "bool", "boolean"):
        return raw.map(_to_bool)
    return raw.astype(object)


def _safe_json(v):
    if v is None:
        return None
    try:
        return json.loads(v)
    except Exception:
        return {"raw": v}


def _to_bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"true", "1", "yes", "y"}


@governed("ai_query")
def ai_query_impl(
    endpoint_name: str,
    request: pd.Series,
    *,
    credential: str,
    registry: EndpointRegistry,
    return_type: Optional[str] = None,
    fail_on_error: bool = True,
    model_parameters: Optional[dict[str, Any]] = None,
    response_format: Optional[dict[str, Any] | str] = None,
) -> pd.Series:
    return _core_ai_query(
        registry=registry,
        endpoint_name=endpoint_name,
        request_series=request,
        credential=credential,
        return_type=return_type,
        fail_on_error=fail_on_error,
        model_parameters=model_parameters,
        response_format=response_format,
    )


# ---- Preset-driven impls ----------------------------------------------------
#
# Each wraps `_core_ai_query` with a packaged preset, then @governed so the
# decorator applies auth / tag-policy / credential-vending / audit with the
# correct `function_name`. These are the targets Gravitino references as class
# paths in CANONICAL_FUNCTIONS.


def _packaged_preset(name: str) -> Preset:
    presets = load_presets()
    try:
        return presets[name]
    except KeyError as e:
        raise KeyError(f"Preset {name!r} not found in packaged prompts.yaml") from e


def _labels_from_series(labels: pd.Series) -> list[str]:
    raw = labels.iloc[0] if len(labels) else None
    if raw is None:
        return []
    return list(raw)


@governed("ai_analyze_sentiment")
def ai_analyze_sentiment_impl(
    endpoint_name: str,
    text: pd.Series,
    *,
    credential: str,
    registry: EndpointRegistry,
) -> pd.Series:
    preset = _packaged_preset("ai_analyze_sentiment")
    return _core_ai_query(
        registry=registry,
        endpoint_name=endpoint_name,
        request_series=text,
        credential=credential,
        preset=preset,
    )


@governed("ai_fix_grammar")
def ai_fix_grammar_impl(
    endpoint_name: str,
    text: pd.Series,
    *,
    credential: str,
    registry: EndpointRegistry,
) -> pd.Series:
    preset = _packaged_preset("ai_fix_grammar")
    return _core_ai_query(
        registry=registry,
        endpoint_name=endpoint_name,
        request_series=text,
        credential=credential,
        preset=preset,
    )


@governed("ai_gen")
def ai_gen_impl(
    endpoint_name: str,
    prompt: pd.Series,
    *,
    credential: str,
    registry: EndpointRegistry,
) -> pd.Series:
    preset = _packaged_preset("ai_gen")
    preset_vars = [{"prompt": p} for p in prompt.tolist()]
    return _core_ai_query(
        registry=registry,
        endpoint_name=endpoint_name,
        request_series=prompt,
        credential=credential,
        preset=preset,
        preset_vars=preset_vars,
    )


@governed("ai_generate_text")
def ai_generate_text_impl(
    endpoint_name: str,
    prompt: pd.Series,
    *,
    credential: str,
    registry: EndpointRegistry,
) -> pd.Series:
    preset = _packaged_preset("ai_generate_text")
    preset_vars = [{"prompt": p} for p in prompt.tolist()]
    return _core_ai_query(
        registry=registry,
        endpoint_name=endpoint_name,
        request_series=prompt,
        credential=credential,
        preset=preset,
        preset_vars=preset_vars,
    )


@governed("ai_classify")
def ai_classify_impl(
    endpoint_name: str,
    text: pd.Series,
    labels: pd.Series,
    *,
    credential: str,
    registry: EndpointRegistry,
) -> pd.Series:
    preset = _packaged_preset("ai_classify")
    label_list = _labels_from_series(labels)
    labels_str = ",".join(label_list)
    rendered = pd.Series(
        [render_preset(preset, text=t, labels=labels_str) for t in text.tolist()],
        index=text.index,
    )
    rf = response_format_for(preset, labels=label_list or None)
    return _core_ai_query(
        registry=registry,
        endpoint_name=endpoint_name,
        request_series=rendered,
        credential=credential,
        response_format=rf,
    )


@governed("ai_extract")
def ai_extract_impl(
    endpoint_name: str,
    text: pd.Series,
    labels: pd.Series,
    *,
    credential: str,
    registry: EndpointRegistry,
) -> pd.Series:
    preset = _packaged_preset("ai_extract")
    label_list = _labels_from_series(labels)
    labels_str = ",".join(label_list)
    rendered = pd.Series(
        [render_preset(preset, text=t, labels=labels_str) for t in text.tolist()],
        index=text.index,
    )
    rf = response_format_for(preset) or {"type": "json_object"}
    return _core_ai_query(
        registry=registry,
        endpoint_name=endpoint_name,
        request_series=rendered,
        credential=credential,
        response_format=rf,
    )


@governed("ai_summarize")
def ai_summarize_impl(
    endpoint_name: str,
    text: pd.Series,
    max_words: pd.Series,
    *,
    credential: str,
    registry: EndpointRegistry,
) -> pd.Series:
    preset = _packaged_preset("ai_summarize")
    mw = int(max_words.iloc[0]) if len(max_words) else 100
    rendered = pd.Series(
        [render_preset(preset, text=t, max_words=mw) for t in text.tolist()],
        index=text.index,
    )
    return _core_ai_query(
        registry=registry,
        endpoint_name=endpoint_name,
        request_series=rendered,
        credential=credential,
    )


@governed("ai_translate")
def ai_translate_impl(
    endpoint_name: str,
    text: pd.Series,
    target_language: pd.Series,
    *,
    credential: str,
    registry: EndpointRegistry,
) -> pd.Series:
    preset = _packaged_preset("ai_translate")
    tl = target_language.iloc[0] if len(target_language) else "English"
    rendered = pd.Series(
        [render_preset(preset, text=t, target_language=tl) for t in text.tolist()],
        index=text.index,
    )
    return _core_ai_query(
        registry=registry,
        endpoint_name=endpoint_name,
        request_series=rendered,
        credential=credential,
    )
