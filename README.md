# spark-ai-functions

Databricks-AI-Functions-compatible SQL surface for vanilla Apache Spark 3.4+,
governed through [Apache Gravitino](https://gravitino.apache.org/) and
[Apache Ranger](https://ranger.apache.org/).

```python
from spark_ai_functions import register
register(spark)
```

```sql
SELECT ai_analyze_sentiment(body) FROM customer_feedback;
SELECT ai_query('gpt-4o-mini', concat('Summarise in 20 words: ', body)) FROM tickets;
```

**Contents**
- [What you get](#what-you-get)
- [Code organization](#code-organization)
- [Prerequisites](#prerequisites)
- [Install](#install)
- [5-minute standalone quickstart](#5-minute-standalone-quickstart)
- [Governed-mode quickstart](#governed-mode-quickstart)
- [Function reference](#function-reference)
- [Configuration reference](#configuration-reference)
- [Packaging and cluster deployment](#packaging-and-cluster-deployment)
- [Troubleshooting](#troubleshooting)
- [Version compatibility](#version-compatibility)
- [For platform / product owners](#for-platform--product-owners)
- [Development](#development)
- [License](#license)

---

## What you get

- **14 SQL functions**, signatures identical to
  [Databricks AI Functions](https://docs.databricks.com/aws/en/large-language-models/ai-functions) —
  drop-in for existing notebooks.
- **Two deployment modes**: *standalone* (YAML endpoints, no authz) and
  *governed* (Gravitino + Ranger, column-tag-aware, audited).
- **Any OpenAI-compatible endpoint**: OpenAI, Azure OpenAI, Ollama, vLLM,
  llama.cpp, internal gateways. MLflow Deployments backend also supported.
  Pure Python — no Spark/Scala JAR required for the LLM call path.
- **Zero vendor lock-in**: vanilla Apache Spark, Apache 2.0 license.

| Mode | Authorization | Audit | Endpoint source |
| --- | --- | --- | --- |
| **Standalone** (dev/test) | None | stdout (JSON/line) | YAML file |
| **Governed** (production) | Ranger via Gravitino push-down | structured sink | Gravitino Model Catalog |

---

## Code organization

Core modules are intentionally split by responsibility:

- `src/spark_ai_functions/register.py`: top-level orchestration (`register`, `register_from_env`), governance wiring, governed catalog registration checks.
- `src/spark_ai_functions/runtime_config.py`: endpoint source resolution and JSON/env parsing (`SPARK_AI_ENDPOINTS_JSON*`, YAML fallback).
- `src/spark_ai_functions/udf_registration.py`: Pandas UDF registration and per-function dispatch behavior.
- `src/spark_ai_functions/core/*`: pure function implementations (query, embeddings, masking, parsing, forecast).
- `src/spark_ai_functions/governance/*`: authz, credential vending, tag policy, audit, and decorator.

For service-managed environments (ODP/Ambari/xDP), prefer:

```python
from spark_ai_functions import register_from_env
register_from_env(spark)
```

---

## Prerequisites

| Thing | Minimum | Notes |
| --- | --- | --- |
| Python | 3.10 (3.10 – 3.13 tested) | Gravitino 1.2 client dropped 3.9. 3.14 wheels for `prophet`/`docling`/`presidio` not yet published. |
| Apache Spark | 3.4.x *or* 3.5.x | Pick the matching Gravitino Spark connector JAR — see [Spark JARs](#spark-jars). |
| Java | 11 or 17 | Matches your Spark install. |
| LLM endpoint | one of OpenAI / Azure OpenAI / Anthropic / Bedrock / MLflow / local vLLM / Ollama | Only needed to actually call an LLM — install still works without. |
| Gravitino + Ranger | *governed mode only* | Images used: `apache/gravitino:1.2.0`, `apache/gravitino-playground:ranger-0.1.0`. See [`docker/README.md`](docker/README.md). |

System packages: `ai_parse_document` requires Docling's native deps (poppler),
`ai_forecast` needs a C++ toolchain for Prophet's `cmdstanpy` compile step on
first use. Skip by not calling those functions; install fails will be loud.

---

## Install

From PyPI (once published):

```bash
pip install spark-ai-functions                       # standalone-only
pip install "spark-ai-functions[governance]"         # + Gravitino client, requests
pip install "spark-ai-functions[anthropic]"          # + anthropic SDK
pip install "spark-ai-functions[bedrock]"            # + boto3
pip install "spark-ai-functions[dev]"                # + pytest, ruff, pip-audit
```

From source:

```bash
git clone https://github.com/apache/spark-ai-functions.git
cd spark-ai-functions
pip install -e ".[dev,governance]"
```

The package ships no JARs. The LLM call path is pure Python (OpenAI SDK), so
no extra Spark JAR is needed for standalone mode. Governed mode requires
Gravitino's Spark connector at Spark startup — see [Spark JARs](#spark-jars).

---

## 5-minute standalone quickstart

Zero infrastructure. Just needs an OpenAI (or compatible) API key.

**1.** Install:

```bash
pip install spark-ai-functions
export OPENAI_API_KEY=sk-...
```

**2.** Create `endpoints.yaml` next to your script:

```yaml
endpoints:
  - name: openai-gpt-4o-mini
    endpoint_type: openai_chat
    base_url: https://api.openai.com/v1
    model_id: gpt-4o-mini
    credential_name: openai
    default_params:
      temperature: 0.0
      max_tokens: 1024
    data_residency: external

  - name: openai-text-embedding-3-small
    endpoint_type: openai_embedding
    base_url: https://api.openai.com/v1
    model_id: text-embedding-3-small
    credential_name: openai
```

**3.** Run:

```python
from pyspark.sql import SparkSession
from spark_ai_functions import register

spark = (
    SparkSession.builder
    .appName("ai-functions-quickstart")
    .master("local[*]")
    .getOrCreate()
)

ai = register(spark, yaml_path="endpoints.yaml")
# Set the default chat endpoint referenced by preset functions:
import os; os.environ["SPARK_AI_DEFAULT_ENDPOINT"] = "openai-gpt-4o-mini"
os.environ["SPARK_AI_DEFAULT_EMBEDDING_ENDPOINT"] = "openai-text-embedding-3-small"

df = spark.createDataFrame(
    [(1, "I love this product!"), (2, "This is terrible.")],
    ["id", "body"],
)
df.selectExpr("id", "ai_analyze_sentiment(body) AS sentiment").show()
```

Expected output:

```
+---+---------+
| id|sentiment|
+---+---------+
|  1| positive|
|  2| negative|
+---+---------+
```

Full script: [`examples/quickstart_standalone.py`](examples/quickstart_standalone.py).

---

## Governed-mode quickstart

For the full Gravitino + Ranger + audit + tag-policy loop. Spec §15's acceptance scenario.

**1.** Boot the stack (adapted from [`apache/gravitino-playground`](https://github.com/apache/gravitino-playground)):

```bash
cd docker
./bootstrap.sh              # fetch playground's init/ and healthcheck/
docker compose up -d        # ~3 min: gravitino, ranger, mysql, spark
curl -sf http://localhost:8090/api/metalakes               # should 200
curl -sf http://localhost:6080/service/public/v2/api/servicedef  # Ranger 200
```

**2.** Register your first endpoint and the 14 UDFs in Gravitino:

```bash
pip install "spark-ai-functions[governance]"

spark-ai-functions register-endpoint \
  --gravitino-uri http://localhost:8090 \
  --metalake prod --catalog ai_functions --schema endpoints \
  --name gpt-4o-mini --type openai_chat \
  --base-url https://api.openai.com/v1 --model-id gpt-4o-mini \
  --credential-name openai-prod

spark-ai-functions register-functions \
  --gravitino-uri http://localhost:8090 \
  --metalake prod --catalog ai_functions --schema functions
```

**3.** Run — **JARs and plugin are mandatory** (see [Spark JARs](#spark-jars)):

```python
from pyspark.sql import SparkSession
from spark_ai_functions import execute_as, register

spark = (
    SparkSession.builder
    .master("local[*]")
    .config("spark.jars.packages",
            "org.apache.gravitino:gravitino-spark-connector-runtime-3.4_2.12:1.2.0")
    .config("spark.plugins",
            "org.apache.gravitino.spark.connector.plugin.GravitinoSparkPlugin")
    .config("spark.sql.gravitino.uri", "http://localhost:8090")
    .config("spark.sql.gravitino.metalake", "prod")
    .getOrCreate()
)

ai = register(spark, gravitino_uri="http://localhost:8090",
              metalake="prod", catalog="ai_functions")

with execute_as(spark, "alice@company.com"):
    spark.sql("SELECT ai_analyze_sentiment('I love it')").show()
```

Full script: [`examples/quickstart_governed.py`](examples/quickstart_governed.py).
Demo notebook: [`examples/notebooks/02_governance_demo.py`](examples/notebooks/02_governance_demo.py).

---

## Function reference

14 functions, signatures identical to Databricks AI Functions. Every one
flows through the `@governed` decorator — authz → tag policy → credential
vend → backend call → audit.

| Function | Signature | Returns |
| --- | --- | --- |
| `ai_query` | `ai_query(endpoint STRING, request STRING [, returnType, failOnError, modelParameters MAP, responseFormat])` | inferred |
| `ai_analyze_sentiment` | `ai_analyze_sentiment(text STRING)` | `STRING` — `positive`/`negative`/`neutral`/`mixed` |
| `ai_classify` | `ai_classify(text STRING, labels ARRAY<STRING>)` | `STRING` (one of `labels`) |
| `ai_extract` | `ai_extract(text STRING, labels ARRAY<STRING>)` | `STRUCT` / JSON |
| `ai_fix_grammar` | `ai_fix_grammar(text STRING)` | `STRING` |
| `ai_gen` | `ai_gen(prompt STRING)` | `STRING` |
| `ai_generate_text` | `ai_generate_text(prompt STRING, endpoint STRING)` | `STRING` |
| `ai_mask` | `ai_mask(text STRING, labels ARRAY<STRING>)` | `STRING` |
| `ai_summarize` | `ai_summarize(text STRING, max_words INT)` | `STRING` |
| `ai_translate` | `ai_translate(text STRING, target_language STRING)` | `STRING` |
| `ai_similarity` | `ai_similarity(text1 STRING, text2 STRING)` | `DOUBLE` (0–1 cosine) |
| `ai_prep_search` | `ai_prep_search(text STRING, chunk_size INT, chunk_overlap INT)` | `ARRAY<STRUCT<chunk:STRING, embedding:ARRAY<FLOAT>>>` |
| `ai_parse_document` | `ai_parse_document(content BINARY)` | `STRUCT<markdown, text, pages[…], metadata>` |
| `ai_forecast` | `ai_forecast(TABLE(observed), horizon TIMESTAMP, time_col, value_col [, group_col, frequency, parameters])` | `TABLE` |

Examples:

```sql
-- Classify into a fixed label set
SELECT id, ai_classify(body, array('bug', 'feature_request', 'feedback')) AS label
FROM   customer_feedback;

-- Extract structured fields into JSON
SELECT ai_extract(body, array('customer_name', 'order_id', 'issue_type'))
FROM   tickets;

-- Similarity search
SELECT ai_similarity('kittens', 'cats') AS score;   -- ~0.85

-- PII masking before LLM send
SELECT ai_mask(body, array('PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER'))
FROM   support_messages;
```

Full walkthrough: [`examples/notebooks/01_databricks_parity_demo.py`](examples/notebooks/01_databricks_parity_demo.py).

---

## Configuration reference

### `register()` parameters

| Parameter | Default | Purpose |
| --- | --- | --- |
| `spark` | *required* | Active SparkSession. |
| `yaml_path` | auto-detects `./endpoints.yaml` | Standalone endpoint YAML file. |
| `endpoint_config_json` | `None` | Endpoint inventory as JSON text (`{"endpoints":[...]}` or `[...]`). Useful for service-managed config stores. |
| `endpoint_config_file` | `None` | Path to endpoint inventory JSON file. |
| `yaml_endpoints` | `None` | List of `EndpointConfig` objects (tests). |
| `gravitino_uri` | `None` | If set → governed mode. |
| `metalake` / `catalog` | `None` | Gravitino path. Required in governed mode. |
| `endpoints_schema` / `functions_schema` | `"endpoints"`, `"functions"` | Gravitino schema names. |
| `user` | `$SPARK_AI_USER` or `<unknown>` | Dev fallback when no Kerberos / `execute_as`. |
| `tag_policy` | default `ColumnTagPolicy()` | Override per-tag actions. |
| `audit_sink` | `StdoutAuditSink()` | Swap for Kafka / HTTP / in-memory. |
| `authorizer` | governed: `GravitinoRangerAuthorizer`, standalone: `AllowAllAuthorizer` | |
| `credential_vendor` | governed: `GravitinoCredentialVendor`, standalone: `EnvCredentialVendor` | |
| `pii_masker` | `None` (pass-through); pass a Presidio-backed masker to enable PII tag masking | |
| `function_names` | all 14 | Subset if you only want some. |
| `skip_plugin_check` | `False` | Bypass the `spark.plugins` contains-check (tests only). |

Returns an `AIFunctions` handle exposing `.registry`, `.governance`,
`.presets`, `.mode`, `.registered_function_names`, and the
`.forecast(df, ...)` helper.

For service deployments (ODP/Ambari/xDP), you can also call:

```python
from spark_ai_functions import register_from_env
register_from_env(spark)
```

`register_from_env()` maps service properties from env vars into `register(...)`.

### Environment variables

| Var | Used for |
| --- | --- |
| `OPENAI_API_KEY` | Fallback credential when no namespaced var is set. |
| `SPARK_AI_ENDPOINT_<UPPERNAME>__API_KEY` | Per-endpoint credential, e.g. `SPARK_AI_ENDPOINT_GPT_4O_MINI__API_KEY`. |
| `SPARK_AI_DEFAULT_ENDPOINT` | Endpoint name used by preset functions. |
| `SPARK_AI_DEFAULT_EMBEDDING_ENDPOINT` | Endpoint name used by `ai_similarity` / `ai_prep_search`. |
| `SPARK_AI_MODE` | `auto` (default), `standalone`, or `governed` for `register_from_env()`. |
| `SPARK_AI_GRAVITINO_URI` / `SPARK_AI_METALAKE` / `SPARK_AI_CATALOG` | Governed-mode path for `register_from_env()`. |
| `SPARK_AI_ENDPOINTS_SCHEMA` / `SPARK_AI_FUNCTIONS_SCHEMA` | Optional schema overrides for governed mode. |
| `SPARK_AI_ENDPOINTS_JSON` | Endpoint inventory JSON payload (service-config friendly). |
| `SPARK_AI_ENDPOINTS_JSON_PATH` | Path to endpoint inventory JSON file. |
| `SPARK_AI_ENDPOINTS_YAML` | Absolute path to a YAML file if auto-detect doesn't suit. |
| `SPARK_AI_PRESETS_PATH` | Optional preset YAML override path. |
| `SPARK_AI_FUNCTION_NAMES` | Optional comma-separated subset of functions to register. |
| `SPARK_AI_SKIP_PLUGIN_CHECK` | Optional `true/false` for test harnesses. |
| `SPARK_AI_USER` | Fallback user identity. |
| `PYSPARK_TESTS=1` | Enables Spark-backed tests in this repo. |
| `SPARK_AI_GOVERNED_TESTS=1` | Enables docker-compose integration tests. |

### Endpoint YAML shape

```yaml
endpoints:
  - name: <logical-name-used-in-sql>       # e.g. "gpt-4o-mini"
    endpoint_type: openai_chat             # openai_chat|openai_embedding|mlflow_chat|mlflow_embedding|anthropic_chat|bedrock_chat|azure_openai_chat
    base_url: https://api.openai.com/v1
    model_id: gpt-4o-mini                  # what the provider calls it
    credential_name: openai                # logical name — how the vendor resolves the secret
    default_params:                        # merged into every call
      temperature: 0.0
      max_tokens: 1024
    data_residency: external               # internal|external — drives the `confidential` tag policy
```

Equivalent JSON shape (for Ambari/mpack or xDP service properties):

```json
{
  "endpoints": [
    {
      "name": "gpt-4o-mini",
      "endpoint_type": "openai_chat",
      "base_url": "https://api.openai.com/v1",
      "model_id": "gpt-4o-mini",
      "credential_name": "openai",
      "default_params": {"temperature": 0.0},
      "data_residency": "external"
    }
  ]
}
```

### Spark JARs

Standalone mode: **none required.** The LLM call path runs through the
`openai` Python SDK on executors — no Spark JAR is needed.

Governed mode:

```
spark.jars.packages=org.apache.gravitino:gravitino-spark-connector-runtime-3.4_2.12:1.2.0
spark.plugins=org.apache.gravitino.spark.connector.plugin.GravitinoSparkPlugin
spark.sql.gravitino.uri=http://gravitino:8090
spark.sql.gravitino.metalake=prod
```

Spark 3.5 users: substitute `gravitino-spark-connector-runtime-3.5_2.12:1.2.0`.

---

## Packaging and cluster deployment

The library is **pure Python**; production deployment is about getting the
wheel plus JARs onto every executor. Pick the pattern that matches your
cluster's existing Python distribution story.

### Building the wheel

```bash
pip install build
python -m build                          # produces dist/spark_ai_functions-0.3.0-py3-none-any.whl
```

Everything under `src/spark_ai_functions/` is included, plus `presets/*.yaml`
and `schemas/*.json` via `[tool.setuptools.package-data]`.

### Deployment patterns

| Pattern | When to use | How |
| --- | --- | --- |
| **`--py-files` on spark-submit** | Ad-hoc / small clusters | Put the wheel on HDFS or a shared path; `spark-submit --py-files hdfs:///libs/spark_ai_functions-0.3.0-py3-none-any.whl your_app.py`. Wheel is unzipped into the executor PYTHONPATH. |
| **`conda-pack` / `pex` env** | Long-lived clusters with a managed Python env | Install `spark-ai-functions` plus its deps into the env, `conda-pack` it, ship via `spark.archives` (Spark 3.4+) or `--archives`. |
| **Image bake** | Kubernetes Spark / OpenShift / EMR-on-EKS | Add `pip install spark-ai-functions[governance]` to the Spark executor Dockerfile. Users pick up automatically. |
| **Cluster-wide `pip install`** | YARN with per-node `pip` access (rare in prod) | Last resort; don't forget `PYSPARK_PYTHON` pointing at the right interpreter on every node. |

### Spark configs (set once in `spark-defaults.conf` or equivalent)

```
spark.jars.packages  org.apache.gravitino:gravitino-spark-connector-runtime-3.4_2.12:1.2.0
spark.plugins        org.apache.gravitino.spark.connector.plugin.GravitinoSparkPlugin
spark.sql.gravitino.uri        http://gravitino.prod.internal:8090
spark.sql.gravitino.metalake   prod
# Recommended — makes audit granularity predictable:
spark.sql.execution.arrow.maxRecordsPerBatch   10000
```

Once those are defaulted at the cluster level, user code reduces to:

```python
from spark_ai_functions import register
register(spark, gravitino_uri="http://gravitino.prod.internal:8090",
         metalake="prod", catalog="ai_functions")
```

### ODP/xDP service config contract

For Ambari mpacks or xDP managed services, represent endpoint/provider choice in
service configs, then invoke `register_from_env(spark)`.

Reference template: [`docs/service-site.xml`](docs/service-site.xml).

Example service properties:

```bash
SPARK_AI_MODE=standalone
SPARK_AI_ENDPOINTS_JSON='{
  "endpoints": [
    {
      "name": "bedrock-claude-sonnet",
      "endpoint_type": "bedrock_chat",
      "base_url": "bedrock://us-west-2",
      "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
      "credential_name": "bedrock-prod",
      "data_residency": "external",
      "default_params": {"temperature": 0.0}
    },
    {
      "name": "openai-direct",
      "endpoint_type": "openai_chat",
      "base_url": "https://api.openai.com/v1",
      "model_id": "gpt-4o-mini",
      "credential_name": "openai-prod",
      "data_residency": "external"
    },
    {
      "name": "internal-llm",
      "endpoint_type": "openai_chat",
      "base_url": "https://llm-gateway.internal.company/v1",
      "model_id": "llama-3.3-70b-instruct",
      "credential_name": "internal-llm",
      "data_residency": "internal"
    }
  ]
}'
SPARK_AI_DEFAULT_ENDPOINT=internal-llm
```

Credential mapping stays namespaced with:
`SPARK_AI_ENDPOINT_<UPPERNAME>__API_KEY`.

### Air-gapped / proxied Maven

If `spark.jars.packages` can't reach Maven Central, mirror the two JARs into
your internal Nexus/Artifactory and set `spark.jars.ivySettings` to a custom
resolver. Same coordinates — only the resolver changes.

### Publishing to your internal index

```bash
python -m build
python -m twine upload --repository-url https://your-pypi.internal/ dist/*
```

Downstream installs:

```bash
pip install --index-url https://your-pypi.internal/simple/ spark-ai-functions
```

### Reproducibility checklist

- [ ] Wheel built from a tagged commit (`git describe --tags --exact-match`).
- [ ] Wheel hash recorded in your artifact store.
- [ ] JAR coordinates frozen in `spark-defaults.conf` (no `LATEST`).
- [ ] Gravitino endpoint inventory exported: `spark-ai-functions list-endpoints`
      (or the REST `GET /api/metalakes/prod/catalogs/ai_functions/schemas/endpoints/models`)
      stored next to the release manifest.
- [ ] `SECURITY.md` reviewed against the current `pip-audit` run.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `RuntimeError: Governed mode requires the Gravitino Spark plugin` | `spark.plugins` doesn't contain `GravitinoSparkPlugin` | Add the config (§17.8); or pass `skip_plugin_check=True` for tests only. |
| `EndpointNotFoundError: No endpoint 'foo' in any configured source` | YAML missing entry, or Gravitino catalog/schema/name mismatch | `spark-ai-functions register-endpoint …` or add to `endpoints.yaml`. |
| `CredentialUnavailableError: No credential for endpoint 'foo'` | Env var not set in the **executor** process | Set via `spark-submit --conf spark.executorEnv.OPENAI_API_KEY=…` or image-bake. Driver-only env doesn't reach workers. |
| `AuthorizationDeniedError: User 'alice' denied EXECUTE on …` | Ranger policy missing, or no `spark.ai.user` set | Wrap call in `with execute_as(spark, "alice@…")`; verify policy in Ranger UI at `http://<ranger>:6080`. |
| `TagPolicyViolationError: pii → mask (PII masker returned empty string)` | Presidio masked the whole column | Check tag annotation; confirm at least some content survives masking; or revisit the tag. |
| `ModuleNotFoundError: gravitino` | Standalone install used in governed config | `pip install 'spark-ai-functions[governance]'`. |
| SQL returns `NULL` for every row | `failOnError=false` swallowed the backend error | Re-run with default `failOnError=true`, or inspect audit log for the `error_class`. |
| Prophet / Docling / Presidio install fails on Py3.14 | No wheels yet | Use Python 3.10–3.13. |
| `apache-gravitino 1.2.0 requires black==26.1.0…` conflict warning | Upstream bad pin; see `SECURITY.md` | Harmless — our override wins; installed black is 26.3.1+. |
| Throttling / 429s from the LLM endpoint | Executor-level concurrency exceeds endpoint RPM | Reduce `spark.executor.cores`, `spark.sql.shuffle.partitions`, or set `default_params.max_concurrency` (default 8) on the endpoint; the OpenAI SDK retries 429s automatically. |

Full governance-model reference: [`docs/governance_model.md`](docs/governance_model.md).
Identity propagation specifics: [`docs/user_identity.md`](docs/user_identity.md).

---

## Version compatibility

| spark-ai-functions | Spark | Gravitino | Gravitino Spark JAR | Python |
| --- | --- | --- | --- | --- |
| `0.3.x` | 3.4.x | 1.2.x | `gravitino-spark-connector-runtime-3.4_2.12:1.2.0` | 3.10 – 3.13 |
| `0.3.x` | 3.5.x | 1.2.x | `gravitino-spark-connector-runtime-3.5_2.12:1.2.0` | 3.10 – 3.13 |

Governed mode requires Scala 2.12 Spark builds — that's what the Gravitino
connector coordinate above expects. Standalone mode has no JVM-side dep.

---

## For platform / product owners

Drop this copy into your team's internal onboarding doc:

> **What this is:** a Python library that registers `ai_query` / `ai_classify` /
> `ai_forecast` / etc. as Spark SQL functions. Behaviour matches Databricks
> AI Functions; code works on vanilla Apache Spark clusters.
>
> **How to get it:** `pip install spark-ai-functions` (or the
> `[governance]` extra for production). Wheel is available at
> `<your-internal-pypi>/spark-ai-functions`.
>
> **How to use it:** one line — `register(spark)` (dev) or `register(spark,
> gravitino_uri=…, metalake=…, catalog=…)` (prod). Then call functions from
> Spark SQL directly.
>
> **What we operate for you:** Gravitino (endpoint + function + credential
> catalog), Ranger (authz), and the LLM endpoint allowlist.
>
> **What you own:** the business logic and the Ranger policy requests for
> any new endpoints you need.
>
> **To onboard a new endpoint:** file a ticket with {endpoint name, base URL,
> model ID, data residency class (`internal` / `external`), credential name}.
> We'll register it in Gravitino and provision the credential.
>
> **To scope a query to a specific user (for row-level audit):**
>
> ```python
> with execute_as(spark, "alice@company.com"):
>     spark.sql("SELECT ai_analyze_sentiment(body) FROM feedback").show()
> ```
>
> **Audit events** land in `<your logging pipeline>` under
> `event_type = ai_function_invocation`. One event per Pandas UDF batch.
>
> **Need help?** `<your support channel>`; escalation path for denied
> authorizations goes to `<your data-governance team>`.

Adapt the placeholders and ship.

Suggested rollout sequence for a new team:

1. **Demo day**: point them at `examples/quickstart_standalone.py` + an
   OpenAI key. They get a feel for the SQL surface in 10 min.
2. **Sandbox endpoint**: register a low-cost chat endpoint in your staging
   Gravitino. Hand them the `register-endpoint` invocation.
3. **Policy draft**: data-governance approves a starter Ranger policy
   allowing their team role to EXECUTE their chosen functions.
4. **Prod promotion**: team starts calling functions against prod-registered
   endpoints, audit events flowing to your standard pipeline.

---

## Development

```bash
make install-dev     # editable install + dev + governance extras
make test            # unit tests only (no Spark, no HTTP)
make test-governed   # integration against docker-compose stack
make lint            # ruff
make fmt             # ruff format + autofix
make preflight       # §19.0 environment checks
make docker-up       # bring up governance stack
make docker-down     # tear it down
```

Running just unit tests without installing Spark is also supported:

```bash
PYTHONPATH=src pytest tests/unit -q
```

Architecture reference — spec §3:

- Layer 1 — endpoint abstraction (`src/spark_ai_functions/endpoints/`)
- Layer 2 — core UDF implementations (`src/spark_ai_functions/core/`)
- Layer 3 — presets & registration (`src/spark_ai_functions/presets/`, `register.py`)
- Layer 4 — governance plane (`src/spark_ai_functions/governance/`)

Security policy and dependency CVE ledger: [`SECURITY.md`](SECURITY.md).
Open items: [`TODO.md`](TODO.md).
Docker stack provenance: [`docker/README.md`](docker/README.md).

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

The `docker/` stack adapts files from
[apache/gravitino-playground](https://github.com/apache/gravitino-playground)
(Apache 2.0).
