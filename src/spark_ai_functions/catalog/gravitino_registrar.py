"""Register the 14 function signatures into a Gravitino UDF Catalog.

Per §7: "Once a function is registered in Gravitino, Spark can discover and
invoke it through standard Spark SQL syntax — no additional CREATE FUNCTION
statement is needed."

Per §18.6: method names on the Python client changed between 0.8 → 1.2; we
isolate the calls here so a client bump only touches this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    signature: str        # DDL-style argument signature
    return_type: str      # DDL-style return
    class_path: str       # fully-qualified Python path of the impl
    description: str = ""


# The canonical surface per §7. Matches Databricks AI Functions 1:1.
CANONICAL_FUNCTIONS: list[FunctionSpec] = [
    FunctionSpec(
        name="ai_query",
        signature="endpoint STRING, request STRING, returnType STRING, failOnError BOOLEAN, modelParameters MAP<STRING,STRING>, responseFormat STRING",
        return_type="STRING",
        class_path="spark_ai_functions.core.ai_query.ai_query_impl",
        description="Generic LLM call against a Gravitino-registered endpoint.",
    ),
    FunctionSpec("ai_analyze_sentiment", "text STRING", "STRING",
                 "spark_ai_functions.core.ai_query.ai_analyze_sentiment_impl"),
    FunctionSpec("ai_classify", "text STRING, labels ARRAY<STRING>", "STRING",
                 "spark_ai_functions.core.ai_query.ai_classify_impl"),
    FunctionSpec("ai_extract", "text STRING, labels ARRAY<STRING>", "STRING",
                 "spark_ai_functions.core.ai_query.ai_extract_impl"),
    FunctionSpec("ai_fix_grammar", "text STRING", "STRING",
                 "spark_ai_functions.core.ai_query.ai_fix_grammar_impl"),
    FunctionSpec("ai_gen", "prompt STRING", "STRING",
                 "spark_ai_functions.core.ai_query.ai_gen_impl"),
    FunctionSpec("ai_generate_text", "prompt STRING, endpoint STRING", "STRING",
                 "spark_ai_functions.core.ai_query.ai_generate_text_impl"),
    FunctionSpec("ai_summarize", "text STRING, max_words INT", "STRING",
                 "spark_ai_functions.core.ai_query.ai_summarize_impl"),
    FunctionSpec("ai_translate", "text STRING, target_language STRING", "STRING",
                 "spark_ai_functions.core.ai_query.ai_translate_impl"),
    FunctionSpec("ai_mask", "text STRING, labels ARRAY<STRING>", "STRING",
                 "spark_ai_functions.core.mask.mask_impl"),
    FunctionSpec("ai_similarity", "text1 STRING, text2 STRING", "DOUBLE",
                 "spark_ai_functions.core.embeddings.ai_similarity_impl"),
    FunctionSpec("ai_prep_search",
                 "text STRING, chunk_size INT, chunk_overlap INT",
                 "ARRAY<STRUCT<chunk:STRING,embedding:ARRAY<FLOAT>>>",
                 "spark_ai_functions.core.embeddings.ai_prep_search_impl"),
    FunctionSpec("ai_parse_document",
                 "content BINARY",
                 "STRUCT<markdown:STRING,text:STRING,pages:ARRAY<STRUCT<page_num:INT,text:STRING>>,metadata:MAP<STRING,STRING>>",
                 "spark_ai_functions.core.parse_document.parse_document_impl"),
    FunctionSpec("ai_forecast",
                 "observed TABLE, horizon TIMESTAMP, time_col STRING, value_col STRING, group_col ARRAY<STRING>, frequency STRING, parameters STRING",
                 "TABLE",
                 "spark_ai_functions.core.forecast.forecast_impl"),
]


class GravitinoUDFRegistrar:
    def __init__(
        self,
        *,
        gravitino_uri: str,
        metalake: str,
        catalog: str,
        schema: str = "functions",
        client: Optional[Any] = None,
    ):
        self._uri = gravitino_uri
        self._metalake = metalake
        self._catalog = catalog
        self._schema = schema
        self._client = client

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        from gravitino import GravitinoClient  # apache-gravitino
        self._client = GravitinoClient(uri=self._uri, metalake_name=self._metalake)
        return self._client

    def _schema_handle(self):
        client = self._ensure_client()
        catalog = client.load_catalog(self._catalog)
        try:
            udf_cat = catalog.as_function_catalog()  # type: ignore[attr-defined]
        except AttributeError:
            try:
                udf_cat = catalog.as_udf_catalog()  # type: ignore[attr-defined]
            except AttributeError:
                udf_cat = catalog
        try:
            return udf_cat.load_schema(self._schema)
        except Exception:
            try:
                return udf_cat.schema(self._schema)  # type: ignore[attr-defined]
            except Exception:
                return udf_cat.create_schema(self._schema)  # type: ignore[attr-defined]

    def ensure_registered(self, specs: Optional[list[FunctionSpec]] = None) -> list[str]:
        """Idempotent: create if missing, replace properties if diverged.

        Returns the list of function names touched.
        """
        schema = self._schema_handle()
        touched: list[str] = []
        for spec in specs or CANONICAL_FUNCTIONS:
            self._register_one(schema, spec)
            touched.append(spec.name)
        return touched

    def _register_one(self, schema, spec: FunctionSpec) -> None:
        props = {
            "class": spec.class_path,
            "signature": spec.signature,
            "return_type": spec.return_type,
            "description": spec.description,
            "language": "python",
            "engine": "pyspark",
        }
        # Try load-and-alter, fall back to register. `load_function` failing
        # means either the function doesn't exist yet or the client build
        # doesn't expose it — in both cases we drop to the register path.
        try:
            schema.load_function(spec.name)  # type: ignore[attr-defined]
            exists = True
        except Exception:
            exists = False

        if exists:
            try:
                schema.alter_function(spec.name, properties=props)  # type: ignore[attr-defined]
                return
            except Exception as alter_exc:
                # Some builds require drop+create. If that also fails, surface
                # the failure instead of reporting deceptive success.
                try:
                    schema.drop_function(spec.name)  # type: ignore[attr-defined]
                    schema.register_function(spec.name, properties=props)  # type: ignore[attr-defined]
                    return
                except Exception as replace_exc:
                    raise RuntimeError(
                        f"Failed to update Gravitino function {spec.name!r}: "
                        f"alter_function raised {type(alter_exc).__name__}: {alter_exc}; "
                        f"drop+register fallback raised {type(replace_exc).__name__}: {replace_exc}"
                    ) from replace_exc

        try:
            schema.register_function(spec.name, properties=props)  # type: ignore[attr-defined]
        except AttributeError:
            # Some older releases expose `create_function`.
            schema.create_function(spec.name, properties=props)  # type: ignore[attr-defined]


def register_endpoint_model(
    *,
    gravitino_uri: str,
    metalake: str,
    catalog: str,
    schema: str,
    name: str,
    endpoint_type: str,
    base_url: str,
    model_id: str,
    credential_name: str,
    default_params: Optional[dict[str, Any]] = None,
    data_residency: str = "external",
    client: Optional[Any] = None,
) -> None:
    """§18.6-style model-catalog registration."""
    import json as _json
    if client is None:
        from gravitino import GravitinoClient
        client = GravitinoClient(uri=gravitino_uri, metalake_name=metalake)
    cat = client.load_catalog(catalog)
    try:
        model_cat = cat.as_model_catalog()
    except AttributeError:
        model_cat = cat
    try:
        sch = model_cat.load_schema(schema)
    except Exception:
        sch = model_cat.create_schema(schema)  # type: ignore[attr-defined]
    props = {
        "endpoint_type": endpoint_type,
        "base_url": base_url,
        "model_id": model_id,
        "credential_name": credential_name,
        "default_params": _json.dumps(default_params or {}),
        "data_residency": data_residency,
    }
    try:
        sch.register_model(name=name, comment=f"Registered by spark-ai-functions", properties=props)  # type: ignore[attr-defined]
    except AttributeError:
        sch.create_model(name=name, comment=f"Registered by spark-ai-functions", properties=props)  # type: ignore[attr-defined]
