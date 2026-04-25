"""`ai_mask` — Presidio-first, LLM fallback.

Design:
  - Primary path: Presidio `AnalyzerEngine` + `AnonymizerEngine`. Deterministic,
    offline, auditable. Happy path for PII.
  - Fallback: if Presidio detects zero entities AND the caller explicitly
    requested a non-Presidio-supported entity type, route through the
    configured LLM endpoint with the `ai_mask` preset (§10).

The Presidio adapter is wrapped behind a thin class so it's swappable in tests.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from ..endpoints.registry import EndpointRegistry
from ..governance.decorator import governed
from ..presets.loader import Preset, render_preset


class PresidioMasker:
    """Thin wrapper — lazily imports Presidio so library import stays cheap."""

    def __init__(self, nlp_engine_name: str = "spacy", model_name: str = "en_core_web_sm"):
        self._nlp_engine_name = nlp_engine_name
        self._model_name = model_name
        self._analyzer = None
        self._anonymizer = None

    def _ensure_loaded(self):
        if self._analyzer is not None:
            return
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

    def mask(self, text: str, entities: Optional[list[str]] = None, language: str = "en") -> str:
        if not text:
            return text
        self._ensure_loaded()
        results = self._analyzer.analyze(text=text, entities=entities, language=language)  # type: ignore[union-attr]
        if not results:
            return text
        anonymised = self._anonymizer.anonymize(text=text, analyzer_results=results)  # type: ignore[union-attr]
        return anonymised.text

    def mask_series(self, s: pd.Series) -> pd.Series:
        return s.astype(object).map(lambda t: self.mask(t or ""))


def _core_ai_mask(
    *,
    registry: Optional[EndpointRegistry],
    endpoint_name: Optional[str],
    text: pd.Series,
    labels: Optional[list[str]],
    credential: Optional[str] = None,
    masker: Optional[PresidioMasker] = None,
    preset: Optional[Preset] = None,
) -> pd.Series:
    masker = masker or PresidioMasker()

    def _safe(t):
        if t is None:
            return ""
        try:
            if pd.isna(t):
                return ""
        except (TypeError, ValueError):
            pass
        return t

    masked = text.astype(object).map(lambda t: masker.mask(_safe(t), entities=labels))

    # LLM fallback: if Presidio didn't change anything on a row and a preset +
    # endpoint are configured, rewrite that subset through the LLM.
    if preset is not None and registry is not None and endpoint_name is not None and credential is not None:
        unchanged_mask = masked == text
        if unchanged_mask.any():
            subset = text[unchanged_mask]
            rendered = pd.Series(
                [render_preset(preset, text=t, labels=",".join(labels or [])) for t in subset.tolist()],
                index=subset.index,
            )
            backend = registry.make_backend(endpoint_name, credential)
            rewritten = backend.batch_chat_complete(rendered, {})  # type: ignore[attr-defined]
            masked.loc[unchanged_mask] = rewritten
    return masked


@governed("ai_mask")
def mask_impl(
    endpoint_name: str,
    text: pd.Series,
    labels: pd.Series,
    *,
    credential: str,
    registry: Optional[EndpointRegistry] = None,
    masker: Optional[PresidioMasker] = None,
    preset: Optional[Preset] = None,
) -> pd.Series:
    # Every row in a batch shares the same labels array (Pandas UDFs can't
    # easily carry per-row arrays here); Spark passes a Series of the same
    # array for each row, so pull the first.
    raw_labels = labels.iloc[0] if len(labels) else None
    label_list = list(raw_labels) if raw_labels else None
    return _core_ai_mask(
        registry=registry,
        endpoint_name=endpoint_name,
        text=text,
        labels=label_list,
        credential=credential,
        masker=masker,
        preset=preset,
    )
