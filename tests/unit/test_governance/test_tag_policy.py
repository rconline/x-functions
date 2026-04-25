import pandas as pd
import pytest

from spark_ai_functions.governance.errors import TagPolicyViolationError
from spark_ai_functions.governance.tag_policy import (
    ColumnTagPolicy,
    DefaultTagPolicyEnforcer,
    DictEndpointResidency,
    PassThroughTagPolicyEnforcer,
    PIIMasker,
    StaticRoleLookup,
)


class _FakeMasker(PIIMasker):
    def __init__(self, replacement="[REDACTED]"):
        self._rep = replacement

    def mask_series(self, s):
        return s.astype(object).map(lambda t: self._rep if t else t)


def test_pass_through_returns_unchanged():
    s = [pd.Series(["hi"])]
    out, actions = PassThroughTagPolicyEnforcer().apply(
        user="u", series=s, tags=["pii"], endpoint_name="e"
    )
    assert out is s
    assert actions == []


def test_restricted_always_denied():
    enf = DefaultTagPolicyEnforcer(
        ColumnTagPolicy(),
        endpoint_residency=DictEndpointResidency({"e": "external"}),
        role_lookup=StaticRoleLookup({"u": {"PHI_OVERRIDE"}}),
        pii_masker=_FakeMasker(),
    )
    with pytest.raises(TagPolicyViolationError):
        enf.apply(user="u", series=[pd.Series(["x"])], tags=["restricted"], endpoint_name="e")


def test_phi_requires_override():
    enf = DefaultTagPolicyEnforcer(
        ColumnTagPolicy(),
        endpoint_residency=DictEndpointResidency({"e": "external"}),
        role_lookup=StaticRoleLookup({"admin": {"PHI_OVERRIDE"}}),
        pii_masker=_FakeMasker(),
    )
    with pytest.raises(TagPolicyViolationError):
        enf.apply(user="bob", series=[pd.Series(["x"])], tags=["phi"], endpoint_name="e")
    _, actions = enf.apply(
        user="admin", series=[pd.Series(["x"])], tags=["phi"], endpoint_name="e"
    )
    assert "phi:override" in actions


def test_confidential_allowed_only_if_internal():
    enf = DefaultTagPolicyEnforcer(
        ColumnTagPolicy(),
        endpoint_residency=DictEndpointResidency({"internal_ep": "internal", "external_ep": "external"}),
        role_lookup=StaticRoleLookup(),
        pii_masker=_FakeMasker(),
    )
    with pytest.raises(TagPolicyViolationError):
        enf.apply(user="u", series=[pd.Series(["x"])], tags=["confidential"], endpoint_name="external_ep")
    _, actions = enf.apply(
        user="u", series=[pd.Series(["x"])], tags=["confidential"], endpoint_name="internal_ep"
    )
    assert "confidential:internal_ok" in actions


def test_pii_is_masked_and_action_audited():
    enf = DefaultTagPolicyEnforcer(
        ColumnTagPolicy(),
        endpoint_residency=DictEndpointResidency({"e": "external"}),
        role_lookup=StaticRoleLookup(),
        pii_masker=_FakeMasker(replacement="XXX"),
    )
    series, actions = enf.apply(
        user="u", series=[pd.Series(["secret"])], tags=["pii"], endpoint_name="e"
    )
    assert series[0].iloc[0] == "XXX"
    assert "pii:masked_before_send" in actions


def test_pii_masker_empty_output_fails():
    class _EmptyMasker(PIIMasker):
        def mask_series(self, s):
            return pd.Series([""] * len(s), index=s.index)

    enf = DefaultTagPolicyEnforcer(
        ColumnTagPolicy(),
        endpoint_residency=DictEndpointResidency({"e": "external"}),
        role_lookup=StaticRoleLookup(),
        pii_masker=_EmptyMasker(),
    )
    with pytest.raises(TagPolicyViolationError):
        enf.apply(user="u", series=[pd.Series(["secret"])], tags=["pii"], endpoint_name="e")
