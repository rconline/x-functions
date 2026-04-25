# Security policy

## Reporting a vulnerability

Open a private security advisory via GitHub Security → Report a vulnerability, or
email the maintainers. Please do not open public issues for suspected vulnerabilities.

## Dependency CVE status (as of 2026-04-24)

We audit declared dependencies with both `pip-audit` (PyPA advisory DB) and
[OSV](https://osv.dev). Every floor in `pyproject.toml` has been checked. Findings
and patches below.

### Patched in `pyproject.toml`

| Package | Prior floor | New floor | CVE | Scope |
| --- | --- | --- | --- | --- |
| `requests` | `>=2.31` | `>=2.33.0` | CVE-2026-25645 (also CVE-2024-35195, CVE-2024-47081 in ≤ 2.32.0) | governance extra + pinned as transitive override |
| `urllib3` | transitive | `>=2.6.3` | CVE-2026-21441 | transitive override |
| `certifi` | transitive | `>=2024.7.4` | trust-bundle hygiene | transitive override |
| `cryptography` | transitive | `>=46.0.7,<47` | CVE-2026-26007, CVE-2026-34073, CVE-2026-39892 (upper bound matches mlflow's `<47`) | transitive override |
| `jinja2` | transitive | `>=3.1.6` | CVE-2025-27516, CVE-2024-56201, CVE-2024-56326 | transitive override |
| `pyyaml` | `>=6.0` | `>=6.0.1` | legacy CVE-2020-14343 in pre-5.4; defensive floor | — |
| `pydantic` | `>=2.5` | `>=2.7` | precautionary; 2.5.x class | — |
| `black` | pulled by `apache-gravitino` | `>=26.3.1` | CVE-2026-32274 | governance extra override |
| `pytest`, `pytest-mock`, `pytest-cov`, `responses`, `ruff`, `testcontainers` | various | bumped to current stable | general hygiene | dev extra |

### Knowingly accepted risk — **MLflow**

`mlflow 2.14` (our declared floor) has **31 known CVEs** per OSV. Every MLflow
version through `3.5.0` still carries ≥ 10 open advisories — these are not
going away by bumping the pin.

**Virtually all** MLflow CVEs target the **tracking server and model-serving
HTTP surface** (artifact upload paths, experiment routes, model-registry
endpoints). This library uses exactly one MLflow API:

```python
mlflow.deployments.get_deploy_client(target_uri).predict(endpoint, inputs)
```

That call path is a *client* against a `target_uri` the operator controls. We
never stand up MLflow's tracking server, never import `mlflow.server`, and
never write MLflow artifacts. Users who deploy MLflow as the deployments
gateway should apply MLflow's own hardening guidance to that gateway — out of
scope for this library.

If you want to avoid MLflow entirely, install without the gateway path and
only use SynapseML- or direct-OpenAI-backed endpoints. MLflow remains a
declared dependency because `mlflow.deployments` is the only portable
interface to MLflow-registered model gateways.

### Cannot be fixed from this repo

| Package | Where | Status |
| --- | --- | --- |
| `pip` (CVE-2026-3219) | User's Python environment | No fix version published as of this audit. Users should track pip advisories separately and upgrade when a fix ships. |

### Not tested on Python 3.14

Our lower-bound floors are validated clean per OSV. The development scan
installs and exercises the runtime dep set on Python 3.14, but some heavy
binaries (prophet, docling, presidio, synapseml) don't have 3.14 wheels yet
and fall back to source builds that fail locally. Supported Python range is
3.10–3.13 per `requires-python = ">=3.10"`.

## Running the audit yourself

```bash
pip install pip-audit
pip-audit                              # scans your installed env
pip-audit -r <(pip freeze)             # scans resolved tree
```

To check a specific package version against OSV:

```bash
curl -s https://api.osv.dev/v1/query \
  -H "Content-Type: application/json" \
  -d '{"package":{"name":"mlflow","ecosystem":"PyPI"},"version":"3.5.0"}' \
  | python -m json.tool
```

## What this library exposes in terms of attack surface

- **No network listener.** This is a Spark client library; nothing listens.
- **LLM HTTP egress** from executors to the endpoints the operator registers
  in Gravitino (or the YAML in standalone mode). Traffic is TLS by default
  (`base_url=https://...`); explicitly allowing plain `http://` endpoints is
  an operator decision.
- **Gravitino REST** for endpoint metadata + authorization checks + credential
  vending, and Gravitino's own authz push-down to Ranger. The library
  **never** calls Ranger directly (§14 guardrail in the spec).
- **Credentials** are fetched from Gravitino or environment variables and
  cached in executor-process memory for ≤ 5 minutes. They are never written
  to disk, never logged at INFO level (§13 in the spec), and never included
  in audit events.
- **PII** flagged via column tags is masked with Presidio *before* the LLM
  call. A masker returning an empty string for non-empty input fails the
  batch (§9). PHI tags fail-closed without an explicit Gravitino role
  property override.

## Governance-plane guarantees

- `AuthorizationDeniedError` and `TagPolicyViolationError` are **never**
  swallowed regardless of `failOnError` (§13).
- Audit events are emitted for **every** Pandas UDF batch — success,
  denied, *and* failure — before the exception propagates.
- Authz allow decisions are cached 60 s per `(user, function, endpoint)`;
  denies are **never** cached so policy fixes take effect immediately (§17.7).
