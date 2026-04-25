"""@governed decorator behaviour — the load-bearing §18.1 contract."""

import pandas as pd
import pytest

from spark_ai_functions.governance.decorator import GovernanceContext, governed, init_governance
from spark_ai_functions.governance.audit import InMemoryAuditSink
from spark_ai_functions.governance.credential_vending import StaticCredentialVendor
from spark_ai_functions.governance.errors import AuthorizationDeniedError
from spark_ai_functions.governance.ranger_authorizer import AllowAllAuthorizer, DenyAllAuthorizer
from spark_ai_functions.governance.tag_policy import PassThroughTagPolicyEnforcer
from spark_ai_functions.governance.user_resolver import ExplicitUserResolver


def _mk_ctx(authz=None, audit=None):
    return GovernanceContext(
        user_resolver=ExplicitUserResolver("alice@t"),
        authorizer=authz or AllowAllAuthorizer(),
        tag_policy=PassThroughTagPolicyEnforcer(),
        credential_vendor=StaticCredentialVendor({"gpt": "sk-1"}),
        audit_sink=audit or InMemoryAuditSink(),
        catalog_name="test.ai_functions",
    )


def test_governed_happy_path_emits_success_audit():
    sink = InMemoryAuditSink()
    init_governance(_mk_ctx(audit=sink))

    @governed("ai_test")
    def impl(endpoint, s, *, credential):
        assert credential == "sk-1"
        return s.str.upper()

    out = impl("gpt", pd.Series(["a", "b"]))
    assert out.tolist() == ["A", "B"]

    event = sink.only(function_name="ai_test", status="success")
    assert event["user"] == "alice@t"
    assert event["row_count"] == 2
    assert event["endpoint_name"] == "gpt"
    assert event["function_catalog"] == "test.ai_functions"


def test_governed_authz_denial_surfaces_and_audits():
    sink = InMemoryAuditSink()
    init_governance(_mk_ctx(authz=DenyAllAuthorizer(), audit=sink))

    @governed("ai_test")
    def impl(endpoint, s, *, credential):
        return s

    with pytest.raises(AuthorizationDeniedError):
        impl("gpt", pd.Series(["x"]))
    evt = sink.only(status="denied")
    assert evt["function_name"] == "ai_test"
    assert evt["error_class"] == "AuthorizationDeniedError"


def test_governed_runtime_error_audits_failure():
    sink = InMemoryAuditSink()
    init_governance(_mk_ctx(audit=sink))

    @governed("ai_test")
    def impl(endpoint, s, *, credential):
        raise ValueError("backend exploded")

    with pytest.raises(ValueError):
        impl("gpt", pd.Series(["x"]))
    evt = sink.only(status="failure")
    assert evt["error_class"] == "ValueError"
    assert "exploded" in (evt["error_message"] or "")


def test_governed_rejects_if_not_initialised(monkeypatch):
    from spark_ai_functions.governance import decorator as D

    monkeypatch.setattr(D, "_CTX", None)

    @governed("ai_test")
    def impl(endpoint, s, *, credential):
        return s

    with pytest.raises(RuntimeError, match="not initialized"):
        impl("gpt", pd.Series(["x"]))


def test_governed_requires_endpoint_false_skips_credential_vending():
    sink = InMemoryAuditSink()
    init_governance(_mk_ctx(audit=sink))

    @governed("ai_parse_document", requires_endpoint=False)
    def impl(endpoint, s):
        assert endpoint == "__none__"
        return s.map(lambda x: x.upper())

    out = impl("__none__", pd.Series(["a"]))
    assert out.tolist() == ["A"]
    evt = sink.only(function_name="ai_parse_document", status="success")
    assert evt["endpoint_name"] == "__none__"
