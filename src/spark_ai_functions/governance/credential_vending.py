"""Credential vending — §3, §6, §14.

Standalone mode: read API keys from environment variables, namespaced by endpoint.
Governed mode: Gravitino vends short-lived credentials; cache 5 min on the executor.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, Optional

from .errors import CredentialUnavailableError


class CredentialVendor(ABC):
    @abstractmethod
    def get(self, endpoint_name: str) -> str:
        """Return a credential string for the named endpoint."""

    def model_id_for(self, endpoint_name: str) -> Optional[str]:  # noqa: ARG002
        return None

    def residency_for(self, endpoint_name: str) -> Optional[str]:  # noqa: ARG002
        return None


class EnvCredentialVendor(CredentialVendor):
    """Resolve each endpoint's secret from an env var.

    Naming pattern: `SPARK_AI_ENDPOINT_<NAME>__API_KEY` (double underscore before field).
    Falls back to `OPENAI_API_KEY` for endpoints with credential_name == 'openai'.
    """

    FALLBACK_ENV = "OPENAI_API_KEY"

    def __init__(self, endpoint_to_env: Optional[dict[str, str]] = None):
        self._map = endpoint_to_env or {}

    def get(self, endpoint_name: str) -> str:
        env_name = self._map.get(endpoint_name) or _default_env_name(endpoint_name)
        val = os.environ.get(env_name)
        if val:
            return val
        # Fallback: OPENAI_API_KEY if present.
        fallback = os.environ.get(self.FALLBACK_ENV)
        if fallback:
            return fallback
        raise CredentialUnavailableError(
            f"No credential for endpoint {endpoint_name!r}: looked in env "
            f"{env_name!r} and fallback {self.FALLBACK_ENV!r}"
        )


def _default_env_name(endpoint_name: str) -> str:
    safe = endpoint_name.upper().replace("-", "_").replace(".", "_")
    return f"SPARK_AI_ENDPOINT_{safe}__API_KEY"


class StaticCredentialVendor(CredentialVendor):
    """Tests / dev-only pre-baked credentials."""

    def __init__(self, creds: dict[str, str]):
        self._creds = creds

    def get(self, endpoint_name: str) -> str:
        try:
            return self._creds[endpoint_name]
        except KeyError as e:
            raise CredentialUnavailableError(str(e)) from e


class _TTLCache:
    def __init__(self, ttl_seconds: float):
        self._ttl = ttl_seconds
        self._m: dict[str, tuple[str, float]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Optional[str]:
        now = time.time()
        with self._lock:
            hit = self._m.get(key)
            if hit is None:
                return None
            val, expires = hit
            if expires < now:
                self._m.pop(key, None)
                return None
            return val

    def put(self, key: str, val: str) -> None:
        with self._lock:
            self._m[key] = (val, time.time() + self._ttl)


class GravitinoCredentialVendor(CredentialVendor):
    """Fetch via Gravitino's credential-vending subsystem.

    Per §3: cache for 5 minutes on the executor. Per §14: never store keys in
    library config — always vend.
    """

    TTL_SECONDS = 300.0

    def __init__(
        self,
        gravitino_client: Any,
        *,
        catalog: str,
        endpoint_index: "EndpointMetadataIndex",
        ttl_seconds: float = TTL_SECONDS,
    ):
        self._client = gravitino_client
        self._catalog = catalog
        self._index = endpoint_index
        self._cache = _TTLCache(ttl_seconds)

    def get(self, endpoint_name: str) -> str:
        cached = self._cache.get(endpoint_name)
        if cached is not None:
            return cached
        meta = self._index.get(endpoint_name)
        if meta is None:
            raise CredentialUnavailableError(f"Unknown endpoint: {endpoint_name}")
        try:
            cred = self._vend(meta.credential_name)
        except Exception as e:
            raise CredentialUnavailableError(f"Gravitino vend failed for {endpoint_name}: {e}") from e
        self._cache.put(endpoint_name, cred)
        return cred

    def _vend(self, credential_name: str) -> str:
        # The exact Gravitino Python client API for credential vending differs
        # across releases (0.9 → 1.2). Keep the single call site here.
        try:
            vendor = self._client.credential_vendor()  # type: ignore[attr-defined]
            vended = vendor.vend(credential_name)
        except AttributeError:
            # Fallback path present in some 1.2.x builds.
            vended = self._client.vend_credential(credential_name)  # type: ignore[attr-defined]
        # Accept either a raw string or a {"secret": ...} object.
        if isinstance(vended, str):
            return vended
        if isinstance(vended, dict):
            for key in ("secret", "api_key", "token", "value"):
                if key in vended:
                    return str(vended[key])
        raise CredentialUnavailableError(
            f"Gravitino returned unrecognized credential shape for {credential_name!r}"
        )

    def model_id_for(self, endpoint_name: str) -> Optional[str]:
        meta = self._index.get(endpoint_name)
        return meta.model_id if meta is not None else None

    def residency_for(self, endpoint_name: str) -> Optional[str]:
        meta = self._index.get(endpoint_name)
        return meta.data_residency if meta is not None else None


class EndpointMetadataIndex:
    """Tiny read-through cache of endpoint metadata for audit/residency annotations."""

    def __init__(self, loader):
        self._loader = loader
        self._m: dict[str, Any] = {}
        self._lock = Lock()

    def get(self, name: str):
        with self._lock:
            hit = self._m.get(name)
        if hit is not None:
            return hit
        loaded = self._loader(name)
        if loaded is None:
            return None
        with self._lock:
            self._m[name] = loaded
        return loaded
