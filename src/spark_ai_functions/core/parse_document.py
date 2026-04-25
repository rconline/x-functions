"""`ai_parse_document` — Docling-backed BINARY → struct parser."""

from __future__ import annotations

import io
import mimetypes
from typing import Any, Optional

import pandas as pd

from ..governance.decorator import governed


class DoclingParser:
    def __init__(self):
        self._converter = None

    def _ensure_loaded(self):
        if self._converter is not None:
            return
        from docling.document_converter import DocumentConverter  # type: ignore
        self._converter = DocumentConverter()

    def parse(self, content: bytes, mime: Optional[str] = None) -> dict[str, Any]:
        self._ensure_loaded()
        from docling.datamodel.document import DoclingDocument  # type: ignore
        from docling.datamodel.base_models import DocumentStream  # type: ignore
        stream = DocumentStream(name="input", stream=io.BytesIO(content))
        result = self._converter.convert(stream)  # type: ignore[union-attr]
        doc = result.document
        md = doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else ""
        text = doc.export_to_text() if hasattr(doc, "export_to_text") else md
        pages: list[dict[str, Any]] = []
        for i, page in enumerate(getattr(doc, "pages", []) or [], start=1):
            pages.append({"page_num": i, "text": getattr(page, "text", "") or ""})
        meta = dict(getattr(doc, "metadata", {}) or {})
        # Serialize any non-string metadata values to strings (MAP<STRING,STRING>).
        meta = {str(k): str(v) for k, v in meta.items()}
        return {"markdown": md or "", "text": text or md or "", "pages": pages, "metadata": meta}


def _core_parse_document(content_series: pd.Series, parser: Optional[DoclingParser] = None) -> pd.Series:
    parser = parser or DoclingParser()
    return pd.Series(
        [parser.parse(bytes(c)) if c is not None else None for c in content_series],
        index=content_series.index,
    )


@governed("ai_parse_document", requires_endpoint=False)
def parse_document_impl(
    endpoint_name: str,  # unused — accepted so the @governed signature is uniform
    content: pd.Series,
    *,
    parser: Optional[DoclingParser] = None,
) -> pd.Series:
    return _core_parse_document(content, parser=parser)
