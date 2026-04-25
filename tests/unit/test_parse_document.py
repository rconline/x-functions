"""Docling parse tests — structural (Docling itself isn't invoked without the
`DOCLING_TESTS=1` env flag, which needs the heavy transitive deps)."""

import os

import pandas as pd
import pytest

from spark_ai_functions.core.parse_document import _core_parse_document


class _FakeParser:
    def parse(self, content, mime=None):
        return {
            "markdown": "# Title",
            "text": "Title",
            "pages": [{"page_num": 1, "text": "Title"}],
            "metadata": {"bytes": str(len(content))},
        }


def test_core_parse_document_uses_injected_parser():
    s = pd.Series([b"%PDF-1.4 ...", b"ignored"])
    out = _core_parse_document(s, parser=_FakeParser())
    assert out.iloc[0]["markdown"] == "# Title"
    assert out.iloc[0]["pages"][0]["page_num"] == 1
    assert out.iloc[0]["metadata"]["bytes"] == str(len(b"%PDF-1.4 ..."))


def test_core_parse_document_handles_none():
    s = pd.Series([None, b"x"])
    out = _core_parse_document(s, parser=_FakeParser())
    assert out.iloc[0] is None
    assert out.iloc[1]["text"] == "Title"


@pytest.mark.skipif(not os.environ.get("DOCLING_TESTS"), reason="set DOCLING_TESTS=1")
def test_docling_real_parser_minimal_pdf():
    from spark_ai_functions.core.parse_document import DoclingParser

    # A minimal single-page PDF (valid enough for Docling to parse).
    pdf = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 0/Kids[]>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF"
    parser = DoclingParser()
    got = parser.parse(pdf)
    assert "markdown" in got
