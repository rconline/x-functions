# Endpoint registration

Endpoints describe **where** the library sends LLM and embedding requests. They
can live in two places:

1. **Gravitino Model Catalog** (governed mode) — the canonical, policy-able source.
2. **YAML file** (standalone mode) — a dev-mode convenience.

## YAML shape

```yaml
endpoints:
  - name: openai-gpt-4o-mini
    endpoint_type: openai_chat      # openai_chat | openai_embedding | mlflow_chat | mlflow_embedding
    base_url: https://api.openai.com/v1
    model_id: gpt-4o-mini
    credential_name: openai         # resolved via EnvCredentialVendor → SPARK_AI_ENDPOINT_OPENAI_GPT_4O_MINI__API_KEY or OPENAI_API_KEY
    default_params:
      temperature: 0.0
      max_tokens: 1024
    data_residency: external        # external | internal — used by the confidential tag policy
```

## Gravitino registration

```bash
spark-ai-functions register-endpoint \
  --gravitino-uri http://gravitino:8090 \
  --metalake prod --catalog ai_functions --schema endpoints \
  --name gpt-4o-mini \
  --type openai_chat \
  --base-url https://api.openai.com/v1 \
  --model-id gpt-4o-mini \
  --credential-name openai-prod \
  --default-params '{"temperature":0.0,"max_tokens":1024}' \
  --data-residency external
```

What this does under the hood (§18.6):

```python
from gravitino import GravitinoClient

client = GravitinoClient(uri="http://gravitino:8090", metalake_name="prod")
catalog = client.load_catalog("ai_functions")
schema = catalog.as_model_catalog().load_schema("endpoints")

schema.register_model(
    name="gpt-4o-mini",
    comment="OpenAI GPT-4o-mini, production",
    properties={
        "endpoint_type": "openai_chat",
        "base_url": "https://api.openai.com/v1",
        "model_id": "gpt-4o-mini",
        "credential_name": "openai-prod-api-key",
        "default_params": '{"temperature": 0.0, "max_tokens": 1024}',
        "data_residency": "external",
    },
)
```

## Registering the 14 SQL functions

```bash
spark-ai-functions register-functions \
  --gravitino-uri http://gravitino:8090 \
  --metalake prod --catalog ai_functions --schema functions
```

Once registered, the Gravitino Spark plugin auto-discovers them — no SQL
`CREATE FUNCTION` needed. See §7.

## Ordering constraint (§17.8)

`register(...)` must be called AFTER the `SparkSession` is built and the
Gravitino plugin is loaded, but BEFORE any AI function is referenced in SQL.
`register()` verifies plugin presence; a clear error is raised otherwise.
