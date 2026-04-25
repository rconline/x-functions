"""Column-tag policy — §9.

| Tag            | Action                                                             |
| -------------- | ------------------------------------------------------------------ |
| `pii`          | Mask with Presidio before send. Fail if masking empties value.     |
| `phi`          | Deny unless caller has Gravitino role with `PHI_OVERRIDE` property |
| `confidential` | Allow only if endpoint has `data_residency=internal`               |
| `restricted`   | Always deny                                                        |
| *(untagged)*   | Allow                                                              |

Per §17.7: tag-policy checks are NOT cached — tag membership can change and
mis-applied tag policy has unbounded downside.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Optional

import pandas as pd

from .errors import TagPolicyViolationError


@dataclass(frozen=True)
class ColumnTagPolicy:
    """User-overridable rules — see `register(tag_policy=...)`."""

    pii: str = "mask"
    phi: str = "deny_unless_override"
    confidential: str = "allow_if_internal"
    restricted: str = "deny"

    # Overrides; dict keys are tag name → action string.
    extra: dict[str, str] = field(default_factory=dict)

    def action_for(self, tag: str) -> str:
        t = tag.lower()
        if t in self.extra:
            return self.extra[t]
        return {
            "pii": self.pii,
            "phi": self.phi,
            "confidential": self.confidential,
            "restricted": self.restricted,
        }.get(t, "allow")


class TagPolicyEnforcer(ABC):
    """Applies ColumnTagPolicy to a batch. Mutates input series in-place per rule."""

    @abstractmethod
    def apply(
        self,
        *,
        user: str,
        series: list[pd.Series],
        tags: list[str],
        endpoint_name: str,
    ) -> tuple[list[pd.Series], list[str]]:
        """Return (possibly-mutated series, list of action strings for audit)."""


class PassThroughTagPolicyEnforcer(TagPolicyEnforcer):
    """Standalone mode — §9: "all values pass through"."""

    def apply(self, *, user, series, tags, endpoint_name):  # noqa: ARG002
        return series, []


class DefaultTagPolicyEnforcer(TagPolicyEnforcer):
    """Governed-mode default.

    Depends on:
      - endpoint lookup for data_residency
      - role-lookup for `PHI_OVERRIDE`
      - a masker for PII (Presidio by default — injected to keep the import optional)
    """

    def __init__(
        self,
        policy: ColumnTagPolicy,
        *,
        endpoint_residency: "EndpointResidencyLookup",
        role_lookup: "RoleLookup",
        pii_masker: "PIIMasker",
    ):
        self._policy = policy
        self._residency = endpoint_residency
        self._roles = role_lookup
        self._masker = pii_masker

    def apply(
        self,
        *,
        user: str,
        series: list[pd.Series],
        tags: list[str],
        endpoint_name: str,
    ) -> tuple[list[pd.Series], list[str]]:
        actions: list[str] = []
        mutated = list(series)
        # Per-tag evaluation. Order: restricted > phi > confidential > pii.
        for tag in sorted(set(t.lower() for t in tags), key=_severity_key):
            action = self._policy.action_for(tag)
            if action == "allow":
                continue
            if action == "deny":
                raise TagPolicyViolationError(tag=tag, action=action, reason="restricted tag on input")
            if action == "deny_unless_override":
                if not self._roles.has_property(user, "PHI_OVERRIDE"):
                    raise TagPolicyViolationError(tag=tag, action=action, reason="missing PHI_OVERRIDE")
                actions.append(f"{tag}:override")
                continue
            if action == "allow_if_internal":
                if self._residency.residency_for(endpoint_name) != "internal":
                    raise TagPolicyViolationError(tag=tag, action=action, reason="endpoint not internal")
                actions.append(f"{tag}:internal_ok")
                continue
            if action == "mask":
                mutated = [self._mask(s) for s in mutated]
                actions.append(f"{tag}:masked_before_send")
                continue
            # Unknown action: treat as deny to be safe.
            raise TagPolicyViolationError(tag=tag, action=action, reason="unknown tag action")
        return mutated, actions

    def _mask(self, s: pd.Series) -> pd.Series:
        masked = self._masker.mask_series(s)
        # §9: "Fail if masking empties value." — only applies when the input
        # had content. Null-in → null/empty-out is always allowed; otherwise
        # we'd deny legitimately nullable PII columns.
        orig_len = s.fillna("").astype(str).str.len()
        mask_len = masked.fillna("").astype(str).str.len()
        if ((mask_len == 0) & (orig_len > 0)).any():
            raise TagPolicyViolationError(
                tag="pii",
                action="mask",
                reason="PII masker returned empty string for non-empty input",
            )
        return masked


_SEVERITY = {"restricted": 0, "phi": 1, "confidential": 2, "pii": 3}


def _severity_key(t: str) -> int:
    return _SEVERITY.get(t, 99)


# ---- collaborator protocols (kept as Protocol-like classes to avoid typing import cost) ----


class EndpointResidencyLookup:
    def residency_for(self, endpoint_name: str) -> Optional[str]:  # pragma: no cover - interface
        raise NotImplementedError


class RoleLookup:
    def has_property(self, user: str, prop: str) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


class PIIMasker:
    def mask_series(self, s: pd.Series) -> pd.Series:  # pragma: no cover - interface
        raise NotImplementedError


# ---- simple defaults ----


class StaticRoleLookup(RoleLookup):
    """Test/dev helper — configure property holders up front."""

    def __init__(self, user_properties: Optional[dict[str, set[str]]] = None):
        self._up: dict[str, set[str]] = user_properties or {}

    def has_property(self, user: str, prop: str) -> bool:
        return prop in self._up.get(user, set())


class DictEndpointResidency(EndpointResidencyLookup):
    def __init__(self, m: dict[str, str]):
        self._m = m

    def residency_for(self, endpoint_name: str) -> Optional[str]:
        return self._m.get(endpoint_name)


def default_enforcer(
    policy: Optional[ColumnTagPolicy] = None,
    *,
    endpoint_residency: Optional[EndpointResidencyLookup] = None,
    role_lookup: Optional[RoleLookup] = None,
    pii_masker: Optional[PIIMasker] = None,
) -> TagPolicyEnforcer:
    if endpoint_residency is None or pii_masker is None:
        return PassThroughTagPolicyEnforcer()
    return DefaultTagPolicyEnforcer(
        policy or ColumnTagPolicy(),
        endpoint_residency=endpoint_residency,
        role_lookup=role_lookup or StaticRoleLookup(),
        pii_masker=pii_masker,
    )
