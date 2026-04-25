"""Ranger authorization via Gravitino push-down — §3, §17.7.

Per §14 guardrail: this library NEVER calls Ranger APIs directly. Authorization
goes through Gravitino's authz REST; Gravitino's own plugin pushes the decision
down to Ranger.

Per §17.7: decisions are cached per (user, function, endpoint) on the executor
for 60 s via a small LRU. Tag-policy checks are NOT cached.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from threading import Lock
from typing import Any, Optional

from .errors import AuthorizationDeniedError


class RangerAuthorizer(ABC):
    @abstractmethod
    def check(self, user: str, function_name: str, endpoint_name: str) -> None:
        """Raise `AuthorizationDeniedError` if the user lacks privilege, else return."""


class AllowAllAuthorizer(RangerAuthorizer):
    """Pass-through authorizer — standalone mode and Phase 1.5 scaffolding."""

    def check(self, user: str, function_name: str, endpoint_name: str) -> None:  # noqa: ARG002
        return None


class DenyAllAuthorizer(RangerAuthorizer):
    """Testing helper."""

    def check(self, user: str, function_name: str, endpoint_name: str) -> None:
        raise AuthorizationDeniedError(
            user=user,
            privilege="EXECUTE",
            resource=f"function:{function_name},endpoint:{endpoint_name}",
            reason="DenyAllAuthorizer configured",
        )


class _DecisionCache:
    """60-second LRU keyed by (user, function, endpoint). Thread-safe."""

    def __init__(self, maxsize: int = 1024, ttl_seconds: float = 60.0):
        self._m: OrderedDict[tuple[str, str, str], tuple[bool, float]] = OrderedDict()
        self._lock = Lock()
        self._max = maxsize
        self._ttl = ttl_seconds

    def get(self, key: tuple[str, str, str]) -> Optional[bool]:
        now = time.time()
        with self._lock:
            hit = self._m.get(key)
            if hit is None:
                return None
            allowed, expires = hit
            if expires < now:
                self._m.pop(key, None)
                return None
            self._m.move_to_end(key)
            return allowed

    def put(self, key: tuple[str, str, str], allowed: bool) -> None:
        with self._lock:
            self._m[key] = (allowed, time.time() + self._ttl)
            self._m.move_to_end(key)
            while len(self._m) > self._max:
                self._m.popitem(last=False)


class GravitinoRangerAuthorizer(RangerAuthorizer):
    """Check via Gravitino's authz REST API.

    Per §17.7: 60s LRU of allow decisions on the executor. Denies are NOT cached
    so that a policy fix takes effect immediately.
    """

    def __init__(
        self,
        gravitino_uri: str,
        metalake: str,
        catalog: str,
        *,
        session: Optional[object] = None,  # allow injection (requests.Session or test double)
        cache_ttl_seconds: float = 60.0,
        timeout_seconds: float = 5.0,
    ):
        self._uri = gravitino_uri.rstrip("/")
        self._metalake = metalake
        self._catalog = catalog
        self._cache = _DecisionCache(ttl_seconds=cache_ttl_seconds)
        self._timeout = timeout_seconds
        self._session = session

    def _http(self):
        if self._session is not None:
            return self._session
        import requests  # local import — governance extra

        self._session = requests.Session()
        return self._session

    def check(self, user: str, function_name: str, endpoint_name: str) -> None:
        key = (user, function_name, endpoint_name)
        cached = self._cache.get(key)
        if cached is True:
            return
        if cached is False:
            # Shouldn't happen — denies aren't cached — but defensive.
            self._deny(user, function_name, endpoint_name, "cached deny")
        allowed, reason = self._ask_gravitino(user, function_name, endpoint_name)
        if not allowed:
            self._deny(user, function_name, endpoint_name, reason)
        self._cache.put(key, True)

    def _ask_gravitino(self, user: str, function_name: str, endpoint_name: str) -> tuple[bool, str]:
        # Gravitino's authz push-down endpoint. We POST a decision request.
        # Exact path/shape differs slightly across 1.2.x; keep this single call
        # site so that if the API shifts we fix it here only.
        url = (
            f"{self._uri}/api/metalakes/{self._metalake}"
            f"/authorization/decision"
        )
        privileges: list[dict[str, Any]] = [
            {"type": "EXECUTE", "resource": {
                "kind": "function",
                "catalog": self._catalog,
                "name": function_name,
            }},
        ]
        # Endpoint-free functions (e.g. ai_parse_document) pass "" here — skip
        # the model USE privilege so authz doesn't trip on a non-existent model.
        if endpoint_name:
            privileges.append(
                {"type": "USE", "resource": {
                    "kind": "model",
                    "catalog": self._catalog,
                    "name": endpoint_name,
                }},
            )
        body = {"user": user, "privileges": privileges}
        try:
            resp = self._http().post(url, json=body, timeout=self._timeout)
        except Exception as e:  # network error → fail-closed
            return False, f"gravitino unreachable: {e}"
        if resp.status_code != 200:
            return False, f"gravitino status {resp.status_code}: {resp.text[:200]}"
        payload = resp.json() if hasattr(resp, "json") else {}
        if isinstance(payload, dict) and payload.get("allowed") is True:
            return True, ""
        reason = payload.get("reason", "denied") if isinstance(payload, dict) else "denied"
        return False, reason

    @staticmethod
    def _deny(user: str, function_name: str, endpoint_name: str, reason: str) -> None:
        raise AuthorizationDeniedError(
            user=user,
            privilege="EXECUTE",
            resource=f"function:{function_name},endpoint:{endpoint_name}",
            reason=reason,
        )
