# Governance model

This library's governance plane sits between the Pandas UDF impl and its
backend. Every invocation flows through five steps (§18.1):

1. **User resolution** (§17.1, §18.2) — the `ChainedUserResolver` returns the
   caller's identity. In governed mode this is the Spark driver's
   `spark.ai.user` local property (set by `execute_as(...)`), which propagates
   to executors via `TaskContext.getLocalProperty`.
2. **Authorization** — `RangerAuthorizer` calls Gravitino's authz REST
   endpoint, which pushes the decision down to Ranger. Allow decisions are
   cached per `(user, function, endpoint)` on the executor for 60 s (§17.7);
   denies are never cached.
3. **Column-tag policy** (§9) — the `TagPolicyEnforcer` inspects the
   source-column tags for `pii`, `phi`, `confidential`, and `restricted`.
   Tag checks are NOT cached.
4. **Credential vending** — `GravitinoCredentialVendor` fetches the endpoint's
   short-lived secret and caches it on the executor for 5 minutes.
5. **Audit** — one event per Pandas UDF batch, whether the call succeeded,
   was denied, or raised. Schema per §8.

## Guardrails (§14)

- **Do not** call Ranger APIs directly — always go through Gravitino.
- **Do not** add a governance-off override. Standalone vs governed is decided
  at `register()` time; once governed, all calls go through the plane.
- **Do not** store API keys in library config. Vend via Gravitino or env vars.

## Errors

| Error | `failOnError` respected? | Notes |
| --- | --- | --- |
| `AuthorizationDeniedError` | **No** — always raises | Policy decisions must surface |
| `TagPolicyViolationError` | **No** — always raises | Column-tag breaches must surface |
| `CredentialUnavailableError` | Yes | Transient vending failures can be swallowed |
| Backend error (e.g., 429, 5xx) | Yes | OpenAI SDK retries internally; surfaced after exhaustion |

## Caching summary

| What | Where | TTL |
| --- | --- | --- |
| Authz allow decisions | Executor | 60 s LRU (max 1024) |
| Authz denies | — | never cached |
| Tag policy decisions | — | never cached |
| Vended credentials | Executor | 5 min |
| Endpoint metadata | Executor | process lifetime |
