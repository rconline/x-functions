"""Three-tier user resolution — §17.1, §18.2.

Ordering:
    1. `KerberosUserResolver` — if UserGroupInformation is wired up.
    2. `SparkPropertyUserResolver` — reads `spark.ai.user` local property set by
       `execute_as(...)` on the driver. This value DOES propagate to TaskContext
       on executors (verified Spark behaviour).
    3. `ExplicitUserResolver` — dev fallback (`register(..., user="...")`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

try:
    from pyspark import TaskContext
except Exception:  # pragma: no cover
    TaskContext = None  # type: ignore[assignment]

UNKNOWN_USER = "<unknown>"


class UserResolver(ABC):
    @abstractmethod
    def resolve(self, tc: Optional["TaskContext"]) -> str:
        """Return a user string, or UNKNOWN_USER if this resolver cannot."""


class ExplicitUserResolver(UserResolver):
    """Use a fixed user. Dev/tests only."""

    def __init__(self, user: str):
        self._u = user

    def resolve(self, tc: Optional["TaskContext"]) -> str:  # noqa: ARG002
        return self._u or UNKNOWN_USER


class SparkPropertyUserResolver(UserResolver):
    """Read the `spark.ai.user` local property set by the driver via `execute_as`."""

    KEY = "spark.ai.user"

    def resolve(self, tc: Optional["TaskContext"]) -> str:
        if tc is None:
            # Driver-side call path (no active task): fall back to SparkContext's
            # local property, which is the same value we'd see via TaskContext.
            try:
                from pyspark import SparkContext

                sc = SparkContext._active_spark_context  # type: ignore[attr-defined]
                if sc is None:
                    return UNKNOWN_USER
                return sc.getLocalProperty(self.KEY) or UNKNOWN_USER
            except Exception:
                return UNKNOWN_USER
        try:
            return tc.getLocalProperty(self.KEY) or UNKNOWN_USER
        except Exception:
            return UNKNOWN_USER


class KerberosUserResolver(UserResolver):
    """Read the current UGI principal short name."""

    def resolve(self, tc: Optional["TaskContext"]) -> str:  # noqa: ARG002
        try:
            from pyspark import SparkContext

            sc = SparkContext._active_spark_context  # type: ignore[attr-defined]
            if sc is None:
                return UNKNOWN_USER
            UGI = sc._jvm.org.apache.hadoop.security.UserGroupInformation  # type: ignore[attr-defined]
            return UGI.getCurrentUser().getShortUserName() or UNKNOWN_USER
        except Exception:
            return UNKNOWN_USER


class ChainedUserResolver(UserResolver):
    """Try each resolver in order; return first non-UNKNOWN result."""

    def __init__(self, *resolvers: UserResolver):
        if not resolvers:
            raise ValueError("ChainedUserResolver requires at least one resolver")
        self._resolvers = resolvers

    def resolve(self, tc: Optional["TaskContext"]) -> str:
        for r in self._resolvers:
            u = r.resolve(tc)
            if u and u != UNKNOWN_USER:
                return u
        return UNKNOWN_USER


def default_chain(explicit_user: Optional[str] = None) -> ChainedUserResolver:
    """Canonical chain per §17.1: Kerberos → SparkProperty → Explicit."""
    resolvers: list[UserResolver] = [KerberosUserResolver(), SparkPropertyUserResolver()]
    if explicit_user:
        resolvers.append(ExplicitUserResolver(explicit_user))
    return ChainedUserResolver(*resolvers)
