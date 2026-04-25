"""ai_mask tests — Presidio is not imported; use an injected masker.

Real Presidio tests require `spacy` + `en_core_web_sm` and are gated by
`PRESIDIO_TESTS=1`.
"""

import os

import pandas as pd
import pytest

from spark_ai_functions.core.mask import _core_ai_mask


class _StubMasker:
    def __init__(self, replacements):
        self._m = replacements

    def mask(self, text, entities=None, language="en"):
        for k, v in self._m.items():
            text = text.replace(k, v)
        return text

    def mask_series(self, s):
        return s.map(lambda t: self.mask(t or ""))


def test_presidio_fast_path():
    text = pd.Series(["Call Alice at 555-1234", "No PII here"])
    out = _core_ai_mask(
        registry=None, endpoint_name=None,
        text=text, labels=["PERSON", "PHONE_NUMBER"],
        credential=None,
        masker=_StubMasker({"Alice": "<PERSON>", "555-1234": "<PHONE>"}),
    )
    assert out.iloc[0] == "Call <PERSON> at <PHONE>"
    assert out.iloc[1] == "No PII here"


def test_empty_text_stays_empty():
    out = _core_ai_mask(
        registry=None, endpoint_name=None,
        text=pd.Series(["", None]), labels=None, credential=None,
        masker=_StubMasker({}),
    )
    assert out.iloc[0] == ""
    assert out.iloc[1] == ""


@pytest.mark.skipif(not os.environ.get("PRESIDIO_TESTS"), reason="set PRESIDIO_TESTS=1")
def test_real_presidio_masks_email():
    from spark_ai_functions.core.mask import PresidioMasker

    m = PresidioMasker()
    out = m.mask("Contact: bob@example.com", entities=["EMAIL_ADDRESS"])
    assert "bob@example.com" not in out
