# TODO — open items

Items here are spec-referenced deferrals, not general backlog. Each must have a clear next-step owner and concrete entry criteria per §11.

## Blocked / needs human

_(empty — nothing blocked as of 2026-04-24 initial build)_

## Deferred to Phase 4.5 live validation

- [ ] `§19.4.5` preflight against a running `docker/docker-compose.yml` stack.
      Requires: local Docker Desktop with ~6 GiB RAM free; ports 8090, 6080, 13306 available.
      Status: compose file checked in, images pre-pulled; stack has not yet been booted
      end-to-end on this host.
- [ ] Verify `spark-sql --conf spark.plugins=... --conf spark.sql.gravitino.uri=...`
      starts and can SELECT from a Gravitino-registered catalog.
- [ ] Confirm exact Gravitino Python client method names (`as_model_catalog`, `register_model`, …)
      match the installed `apache-gravitino==1.2.0`. Adjust `gravitino_registrar.py` if divergent.
      (The registrar already tries multiple method-name fallbacks.)

## Coverage gaps (unit tests bypass heavy deps)

Per §12 the 80% coverage target applies to `core/`, `endpoints/`, `governance/`.
Currently met comfortably on governance/ (88% avg) and on every endpoint/
and core/ file whose code path does NOT depend on a heavy external SDK.

Files below 80% and why:

| File | Coverage | Gated by |
| --- | --- | --- |
| `core/mask.py` | 53% | `PRESIDIO_TESTS=1` — needs `spacy` + `en_core_web_sm` |
| `core/parse_document.py` | 38% | `DOCLING_TESTS=1` — Docling wheel + poppler |
| `core/forecast.py` | 22% | `PROPHET_TESTS=1` — Prophet + CmdStanPy |
| `endpoints/synapseml_backend.py` | 45% | Requires live chat endpoint |
| `endpoints/mlflow_backend.py` | 52% | Requires MLflow Deployments target |
| `endpoints/gravitino_source.py` | 68% | Fallback branches for client-API variants |

These are covered by the respective integration tests (gated by env vars, see
`tests/integration/`). None of them affect the governance correctness surface.

## Deferred to v2 per §17.1

- [ ] Per-query user derivation from JWT tokens or Spark Connect session metadata
      (excluded from v1 due to propagation subtleties).

## Phase-5 polish

- [ ] Produce executable `.ipynb` versions of `examples/notebooks/*` via `jupytext --to ipynb`.
      Source `.py` scripts are checked in as the canonical form.
- [ ] TestPyPI release rehearsal and "fresh venv in <5 min" validation (§21).
