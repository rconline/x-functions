"""Governance plane: auth, tag policy, credential vending, audit."""

from .audit import AuditEvent, AuditSink, InMemoryAuditSink, StdoutAuditSink
from .credential_vending import CredentialUnavailableError, CredentialVendor, EnvCredentialVendor
from .decorator import GovernanceContext, governed, init_governance
from .errors import (
    AuthorizationDeniedError,
    CredentialUnavailableError as _CredErr,  # re-export alias
    GovernanceError,
    TagPolicyViolationError,
)
from .ranger_authorizer import (
    AllowAllAuthorizer,
    GravitinoRangerAuthorizer,
    RangerAuthorizer,
)
from .tag_policy import ColumnTagPolicy, DefaultTagPolicyEnforcer, PassThroughTagPolicyEnforcer, TagPolicyEnforcer
from .user_resolver import (
    ChainedUserResolver,
    ExplicitUserResolver,
    KerberosUserResolver,
    SparkPropertyUserResolver,
    UserResolver,
)

__all__ = [
    # decorator
    "governed",
    "init_governance",
    "GovernanceContext",
    # errors
    "GovernanceError",
    "AuthorizationDeniedError",
    "TagPolicyViolationError",
    "CredentialUnavailableError",
    # authz
    "RangerAuthorizer",
    "GravitinoRangerAuthorizer",
    "AllowAllAuthorizer",
    # tag
    "TagPolicyEnforcer",
    "DefaultTagPolicyEnforcer",
    "PassThroughTagPolicyEnforcer",
    "ColumnTagPolicy",
    # creds
    "CredentialVendor",
    "EnvCredentialVendor",
    # audit
    "AuditSink",
    "AuditEvent",
    "InMemoryAuditSink",
    "StdoutAuditSink",
    # user
    "UserResolver",
    "ChainedUserResolver",
    "ExplicitUserResolver",
    "SparkPropertyUserResolver",
    "KerberosUserResolver",
]
