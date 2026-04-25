"""`@governed` decorator — §18.1.

Wraps a core UDF impl with auth, tag policy, credential vending, and audit.
One audit event per Pandas UDF batch (not per row).
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Optional

import pandas as pd

try:  # pyspark is a hard dependency but keep import cheap-failing for type checkers
    from pyspark import TaskContext
except Exception:  # pragma: no cover
    TaskContext = None  # type: ignore[assignment]

from .audit import AuditSink
from .credential_vending import CredentialVendor
from .errors import AuthorizationDeniedError, TagPolicyViolationError
from .ranger_authorizer import RangerAuthorizer
from .tag_policy import TagPolicyEnforcer
from .user_resolver import UserResolver


class GovernanceContext:
    """Bundle of plane components set once per JVM via `init_governance`."""

    def __init__(
        self,
        user_resolver: UserResolver,
        authorizer: RangerAuthorizer,
        tag_policy: TagPolicyEnforcer,
        credential_vendor: CredentialVendor,
        audit_sink: AuditSink,
        catalog_name: Optional[str] = None,
    ):
        self.user_resolver = user_resolver
        self.authorizer = authorizer
        self.tag_policy = tag_policy
        self.credential_vendor = credential_vendor
        self.audit_sink = audit_sink
        self.catalog_name = catalog_name


_CTX: Optional[GovernanceContext] = None


def init_governance(ctx: GovernanceContext) -> None:
    """Install a governance context as the process-wide default."""
    global _CTX
    _CTX = ctx


def current_context() -> Optional[GovernanceContext]:
    return _CTX


def governed(function_name: str, *, requires_endpoint: bool = True) -> Callable:
    """Wrap a core UDF impl with auth, tag policy, cred vending, audit.

    The wrapped impl signature must be:
        impl(endpoint_name: str, *pd_series, credential: str, **kwargs) -> pd.Series | pd.DataFrame

    Set `requires_endpoint=False` for functions that don't resolve a remote
    endpoint (e.g. local binary parsers). The wrapper then skips credential
    vending and passes an empty endpoint name to the authorizer so endpoint-
    scoped authz (e.g. a model USE check) can be short-circuited.
    """

    def decorator(impl: Callable) -> Callable:
        @functools.wraps(impl)
        def wrapper(endpoint_name: str, *args: pd.Series, **kwargs: Any):
            if _CTX is None:
                raise RuntimeError("Governance not initialized. Call register() first.")
            tc = TaskContext.get() if TaskContext is not None else None  # type: ignore[union-attr]
            user = _CTX.user_resolver.resolve(tc)
            start = time.time()
            row_count = len(args[0]) if args and hasattr(args[0], "__len__") else 0
            tag_actions: list[str] = []
            column_tags = kwargs.pop("_column_tags", []) or []
            source_columns = kwargs.pop("_source_columns", []) or []
            authz_endpoint = endpoint_name if requires_endpoint else ""
            try:
                _CTX.authorizer.check(user, function_name, authz_endpoint)
                if column_tags:
                    args, tag_actions = _CTX.tag_policy.apply(
                        user=user,
                        series=list(args),
                        tags=column_tags,
                        endpoint_name=endpoint_name,
                    )
                if requires_endpoint:
                    credential = _CTX.credential_vendor.get(endpoint_name)
                    result = impl(endpoint_name, *args, credential=credential, **kwargs)
                else:
                    result = impl(endpoint_name, *args, **kwargs)
                _audit(
                    user=user,
                    function_name=function_name,
                    endpoint_name=endpoint_name,
                    status="success",
                    row_count=row_count,
                    latency_ms=int((time.time() - start) * 1000),
                    tag_actions=tag_actions,
                    source_columns=source_columns,
                    column_tags=column_tags,
                )
                return result
            except AuthorizationDeniedError as e:
                _audit(
                    user=user,
                    function_name=function_name,
                    endpoint_name=endpoint_name,
                    status="denied",
                    row_count=row_count,
                    latency_ms=int((time.time() - start) * 1000),
                    tag_actions=tag_actions,
                    source_columns=source_columns,
                    column_tags=column_tags,
                    error_class=type(e).__name__,
                    error_message=str(e),
                )
                raise
            except TagPolicyViolationError as e:
                _audit(
                    user=user,
                    function_name=function_name,
                    endpoint_name=endpoint_name,
                    status="denied",
                    row_count=row_count,
                    latency_ms=int((time.time() - start) * 1000),
                    tag_actions=tag_actions,
                    source_columns=source_columns,
                    column_tags=column_tags,
                    error_class=type(e).__name__,
                    error_message=str(e),
                )
                raise
            except Exception as e:
                _audit(
                    user=user,
                    function_name=function_name,
                    endpoint_name=endpoint_name,
                    status="failure",
                    row_count=row_count,
                    latency_ms=int((time.time() - start) * 1000),
                    tag_actions=tag_actions,
                    source_columns=source_columns,
                    column_tags=column_tags,
                    error_class=type(e).__name__,
                    error_message=str(e),
                )
                raise

        wrapper.__governed_function_name__ = function_name  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _audit(
    *,
    user: str,
    function_name: str,
    endpoint_name: str,
    status: str,
    row_count: int,
    latency_ms: int,
    tag_actions: list[str],
    source_columns: list[str],
    column_tags: list[str],
    error_class: Optional[str] = None,
    error_message: Optional[str] = None,
    input_token_count: Optional[int] = None,
    output_token_count: Optional[int] = None,
) -> None:
    if _CTX is None or _CTX.audit_sink is None:
        return
    catalog = _CTX.catalog_name or ""
    _CTX.audit_sink.emit(
        {
            "event_type": "ai_function_invocation",
            "timestamp": _iso_now(),
            "user": user,
            "session_id": _session_id(),
            "query_id": _query_id(),
            "function_name": function_name,
            "function_catalog": catalog,
            "endpoint_name": endpoint_name,
            "endpoint_model_id": _CTX.credential_vendor.model_id_for(endpoint_name)
            if hasattr(_CTX.credential_vendor, "model_id_for")
            else None,
            "row_count": row_count,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "latency_ms": latency_ms,
            "status": status,
            "error_class": error_class,
            "error_message": error_message,
            "source_columns": list(source_columns),
            "column_tags_observed": list(column_tags),
            "tag_policy_actions": list(tag_actions),
            "data_residency": _residency_for(endpoint_name),
        }
    )


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _session_id() -> str:
    try:
        from pyspark.sql import SparkSession

        s = SparkSession.getActiveSession()
        if s is not None:
            return s.conf.get("spark.app.id", "")
    except Exception:
        pass
    return ""


def _query_id() -> str:
    tc = TaskContext.get() if TaskContext is not None else None  # type: ignore[union-attr]
    if tc is None:
        return ""
    try:
        return tc.getLocalProperty("spark.sql.execution.id") or ""
    except Exception:
        return ""


def _residency_for(endpoint_name: str) -> Optional[str]:
    if _CTX is None:
        return None
    vendor = _CTX.credential_vendor
    if hasattr(vendor, "residency_for"):
        try:
            return vendor.residency_for(endpoint_name)
        except Exception:
            return None
    return None
