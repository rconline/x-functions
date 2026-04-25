import time

import pytest

from spark_ai_functions.governance.errors import AuthorizationDeniedError
from spark_ai_functions.governance.ranger_authorizer import (
    AllowAllAuthorizer,
    DenyAllAuthorizer,
    GravitinoRangerAuthorizer,
    _DecisionCache,
)


def test_allow_all_passes():
    AllowAllAuthorizer().check("alice", "ai_query", "gpt-4o-mini")


def test_deny_all_raises():
    with pytest.raises(AuthorizationDeniedError):
        DenyAllAuthorizer().check("alice", "ai_query", "gpt-4o-mini")


def test_decision_cache_ttl():
    c = _DecisionCache(ttl_seconds=0.05)
    c.put(("u", "f", "e"), True)
    assert c.get(("u", "f", "e")) is True
    time.sleep(0.1)
    assert c.get(("u", "f", "e")) is None


def test_decision_cache_lru_eviction():
    c = _DecisionCache(maxsize=2, ttl_seconds=10)
    c.put(("u", "f", "a"), True)
    c.put(("u", "f", "b"), True)
    c.put(("u", "f", "c"), True)
    assert c.get(("u", "f", "a")) is None
    assert c.get(("u", "f", "b")) is True
    assert c.get(("u", "f", "c")) is True


class _FakeResp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


class _FakeSession:
    def __init__(self, resp):
        self._r = resp
        self.calls = []

    def post(self, url, json, timeout):
        self.calls.append((url, json, timeout))
        return self._r


def test_gravitino_allow():
    authz = GravitinoRangerAuthorizer(
        gravitino_uri="http://gravitino:8090", metalake="m", catalog="c",
        session=_FakeSession(_FakeResp(200, {"allowed": True})),
    )
    authz.check("alice", "ai_query", "gpt")


def test_gravitino_deny_raises_with_reason():
    authz = GravitinoRangerAuthorizer(
        gravitino_uri="http://gravitino:8090", metalake="m", catalog="c",
        session=_FakeSession(_FakeResp(200, {"allowed": False, "reason": "no policy"})),
    )
    with pytest.raises(AuthorizationDeniedError) as ei:
        authz.check("alice", "ai_query", "gpt")
    assert "no policy" in str(ei.value)


def test_gravitino_fails_closed_on_http_error():
    authz = GravitinoRangerAuthorizer(
        gravitino_uri="http://gravitino:8090", metalake="m", catalog="c",
        session=_FakeSession(_FakeResp(500, {"error": "boom"})),
    )
    with pytest.raises(AuthorizationDeniedError):
        authz.check("alice", "ai_query", "gpt")


def test_allow_caches_no_extra_call():
    session = _FakeSession(_FakeResp(200, {"allowed": True}))
    authz = GravitinoRangerAuthorizer(
        gravitino_uri="http://gravitino:8090", metalake="m", catalog="c", session=session,
    )
    authz.check("alice", "ai_query", "gpt")
    authz.check("alice", "ai_query", "gpt")
    assert len(session.calls) == 1
