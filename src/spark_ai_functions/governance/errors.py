"""Governance-specific exceptions.

Per §13: `AuthorizationDeniedError` and `TagPolicyViolationError` are NEVER swallowed
regardless of `failOnError`. `CredentialUnavailableError` IS subject to `failOnError`.
"""

from __future__ import annotations


class GovernanceError(Exception):
    """Base class for governance-plane errors."""


class AuthorizationDeniedError(GovernanceError):
    """Raised by RangerAuthorizer when the caller lacks a required privilege.

    Per §13: always propagates regardless of `failOnError`.
    """

    def __init__(self, user: str, privilege: str, resource: str, reason: str = ""):
        self.user = user
        self.privilege = privilege
        self.resource = resource
        self.reason = reason
        msg = f"User {user!r} denied {privilege!r} on {resource!r}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class TagPolicyViolationError(GovernanceError):
    """Raised by TagPolicyEnforcer when column tags forbid the operation.

    Per §13: always propagates regardless of `failOnError`.
    """

    def __init__(self, tag: str, action: str, reason: str = ""):
        self.tag = tag
        self.action = action
        self.reason = reason
        super().__init__(f"Tag policy violation: {tag!r} → {action}" + (f" ({reason})" if reason else ""))


class CredentialUnavailableError(GovernanceError):
    """Raised when credential vending cannot supply an endpoint's secret.

    Per §13: subject to `failOnError` (can be transient).
    """


class EndpointNotFoundError(GovernanceError):
    """Raised when an endpoint name cannot be resolved by any source."""
