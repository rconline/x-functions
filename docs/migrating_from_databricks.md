# Migrating from Databricks AI Functions

All 14 function signatures are identical, so existing SQL should run
unchanged. Registration differs:

| Databricks | This library |
| --- | --- |
| Functions pre-registered on every cluster | `from spark_ai_functions import register; register(spark)` |
| Endpoints are Databricks Serving endpoints | Endpoints are Gravitino Model Catalog entries (or YAML in standalone mode) |
| Authorization via Unity Catalog | Authorization via Gravitino → Ranger |
| Audit goes to the Databricks system table | Audit goes to your configured `AuditSink` (stdout by default) |

## Drop-in example

```python
# Databricks notebook — works as-is after register(spark).
spark.sql("""
    SELECT id,
           ai_analyze_sentiment(body) AS sentiment,
           ai_classify(body, array('bug', 'feature')) AS label,
           ai_summarize(body, 15)                    AS summary,
           ai_translate(body, 'Japanese')            AS body_ja
    FROM   customer_feedback
""").show()
```

## Functions with minor differences

- `ai_query` — the `endpoint` first argument is a Gravitino Model name rather
  than a Databricks Serving endpoint URL. Signatures otherwise match.
- `ai_forecast` — exposed as both a SQL TVF (Spark 3.5+) and as a Python
  helper `ai.forecast(df, ...)` for Spark 3.4. See
  [`core/forecast.py`](../src/spark_ai_functions/core/forecast.py).

## Features intentionally excluded (v1)

- Unity Catalog integration (governance is Gravitino → Ranger)
- Delta Live Tables pipelines
- Lakeflow hooks
- Cost dashboard
- Multimodal inputs (stretch goal)
- Scala API

See §2 of the spec for the full scope boundary.
