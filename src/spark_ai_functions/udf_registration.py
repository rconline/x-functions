"""Pandas UDF registration for spark-ai-functions."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from .core.ai_query import (
    ai_analyze_sentiment_impl,
    ai_classify_impl,
    ai_extract_impl,
    ai_fix_grammar_impl,
    ai_gen_impl,
    ai_generate_text_impl,
    ai_query_impl,
    ai_summarize_impl,
    ai_translate_impl,
)
from .core.embeddings import ai_prep_search_impl, ai_similarity_impl
from .core.mask import mask_impl
from .core.parse_document import parse_document_impl
from .endpoints.registry import EndpointRegistry
from .presets.loader import Preset

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def register_udfs(
    spark: "SparkSession",
    registry: EndpointRegistry,
    presets: dict[str, Preset],
    names_wanted: Optional[set[str]],
) -> list[str]:
    out: list[str] = []

    def want(name: str) -> bool:
        return names_wanted is None or name in names_wanted

    if want("ai_query"):
        @pandas_udf(StringType())
        def _udf_ai_query(*cols: pd.Series) -> pd.Series:
            if len(cols) < 2:
                raise ValueError("ai_query requires at least (endpoint, request)")
            endpoint = cols[0].astype(object)
            request = cols[1]

            return_type = (
                cols[2].astype(object)
                if len(cols) >= 3
                else pd.Series([None] * len(request), index=request.index, dtype=object)
            )
            fail_on_error = (
                cols[3].astype(object)
                if len(cols) >= 4
                else pd.Series([None] * len(request), index=request.index, dtype=object)
            )
            model_parameters = (
                cols[4].astype(object)
                if len(cols) >= 5
                else pd.Series([None] * len(request), index=request.index, dtype=object)
            )
            response_format = (
                cols[5].astype(object)
                if len(cols) >= 6
                else pd.Series([None] * len(request), index=request.index, dtype=object)
            )

            frame = pd.DataFrame(
                {
                    "endpoint": endpoint,
                    "request": request,
                    "return_type": return_type,
                    "fail_on_error": fail_on_error,
                    "model_parameters": model_parameters,
                    "response_format": response_format,
                },
                index=request.index,
            )
            result = pd.Series([None] * len(request), index=request.index, dtype=object)
            sentinel = "__NULL__"

            def _norm(v: Any) -> Any:
                if v is None:
                    return sentinel
                try:
                    if pd.isna(v):
                        return sentinel
                except (TypeError, ValueError):
                    pass
                return v

            key = frame.apply(
                lambda r: (
                    _norm(r["endpoint"]),
                    _norm(r["return_type"]),
                    _norm(r["fail_on_error"]),
                    json.dumps(r["model_parameters"], sort_keys=True, default=str)
                    if isinstance(r["model_parameters"], dict)
                    else _norm(r["model_parameters"]),
                    json.dumps(r["response_format"], sort_keys=True, default=str)
                    if isinstance(r["response_format"], dict)
                    else _norm(r["response_format"]),
                ),
                axis=1,
            )

            for _, idx in key.groupby(key).groups.items():
                sub = frame.loc[list(idx)]
                kwargs: dict[str, Any] = {"registry": registry}
                rt = sub["return_type"].iloc[0]
                if rt is not None:
                    kwargs["return_type"] = rt
                foe = sub["fail_on_error"].iloc[0]
                if foe is not None:
                    if isinstance(foe, str):
                        kwargs["fail_on_error"] = foe.strip().lower() in {"1", "true", "yes", "y"}
                    else:
                        kwargs["fail_on_error"] = bool(foe)
                mp = sub["model_parameters"].iloc[0]
                if mp is not None:
                    kwargs["model_parameters"] = dict(mp)
                rf = sub["response_format"].iloc[0]
                if rf is not None:
                    kwargs["response_format"] = rf
                name = sub["endpoint"].iloc[0]
                result.loc[sub.index] = ai_query_impl(name, sub["request"], **kwargs).astype(object)
            return result

        spark.udf.register("ai_query", _udf_ai_query)
        out.append("ai_query")

    if want("ai_analyze_sentiment"):
        @pandas_udf(StringType())
        def _udf_ai_analyze_sentiment(text: pd.Series) -> pd.Series:
            endpoint = _default_endpoint(registry)
            return ai_analyze_sentiment_impl(endpoint, text, registry=registry).astype(object)

        spark.udf.register("ai_analyze_sentiment", _udf_ai_analyze_sentiment)
        out.append("ai_analyze_sentiment")

    if want("ai_fix_grammar"):
        @pandas_udf(StringType())
        def _udf_ai_fix_grammar(text: pd.Series) -> pd.Series:
            endpoint = _default_endpoint(registry)
            return ai_fix_grammar_impl(endpoint, text, registry=registry).astype(object)

        spark.udf.register("ai_fix_grammar", _udf_ai_fix_grammar)
        out.append("ai_fix_grammar")

    if want("ai_gen"):
        @pandas_udf(StringType())
        def _udf_ai_gen(prompt: pd.Series) -> pd.Series:
            endpoint = _default_endpoint(registry)
            return ai_gen_impl(endpoint, prompt, registry=registry).astype(object)

        spark.udf.register("ai_gen", _udf_ai_gen)
        out.append("ai_gen")

    if want("ai_classify"):
        @pandas_udf(StringType())
        def _udf_ai_classify(text: pd.Series, labels: pd.Series) -> pd.Series:
            endpoint = _default_endpoint(registry)
            return ai_classify_impl(endpoint, text, labels, registry=registry).astype(object)

        spark.udf.register("ai_classify", _udf_ai_classify)
        out.append("ai_classify")

    if want("ai_extract"):
        @pandas_udf(StringType())
        def _udf_ai_extract(text: pd.Series, labels: pd.Series) -> pd.Series:
            endpoint = _default_endpoint(registry)
            return ai_extract_impl(endpoint, text, labels, registry=registry).astype(object)

        spark.udf.register("ai_extract", _udf_ai_extract)
        out.append("ai_extract")

    if want("ai_generate_text"):
        @pandas_udf(StringType())
        def _udf_ai_generate_text(prompt: pd.Series, endpoint: pd.Series) -> pd.Series:
            result = pd.Series([None] * len(prompt), index=prompt.index, dtype=object)
            null_mask = endpoint.isna()
            non_null = endpoint[~null_mask]
            for ep_val in non_null.drop_duplicates().tolist():
                rows = endpoint == ep_val
                sub = prompt[rows]
                if sub.empty:
                    continue
                result.loc[sub.index] = ai_generate_text_impl(str(ep_val), sub, registry=registry).astype(
                    object
                )
            if null_mask.any():
                default_ep = _default_endpoint(registry)
                sub = prompt[null_mask]
                result.loc[sub.index] = ai_generate_text_impl(default_ep, sub, registry=registry).astype(
                    object
                )
            return result

        spark.udf.register("ai_generate_text", _udf_ai_generate_text)
        out.append("ai_generate_text")

    if want("ai_summarize"):
        @pandas_udf(StringType())
        def _udf_ai_summarize(text: pd.Series, max_words: pd.Series) -> pd.Series:
            endpoint = _default_endpoint(registry)
            return ai_summarize_impl(endpoint, text, max_words, registry=registry).astype(object)

        spark.udf.register("ai_summarize", _udf_ai_summarize)
        out.append("ai_summarize")

    if want("ai_translate"):
        @pandas_udf(StringType())
        def _udf_ai_translate(text: pd.Series, target_language: pd.Series) -> pd.Series:
            endpoint = _default_endpoint(registry)
            return ai_translate_impl(endpoint, text, target_language, registry=registry).astype(object)

        spark.udf.register("ai_translate", _udf_ai_translate)
        out.append("ai_translate")

    if want("ai_mask"):
        @pandas_udf(StringType())
        def _udf_ai_mask(text: pd.Series, labels: pd.Series) -> pd.Series:
            endpoint = _default_endpoint(registry)
            return mask_impl(
                endpoint,
                text,
                labels,
                registry=registry,
                preset=presets.get("ai_mask"),
            )

        spark.udf.register("ai_mask", _udf_ai_mask)
        out.append("ai_mask")

    if want("ai_similarity"):
        @pandas_udf(DoubleType())
        def _udf_ai_similarity(a: pd.Series, b: pd.Series) -> pd.Series:
            endpoint = _default_embedding_endpoint(registry)
            return ai_similarity_impl(endpoint, a, b, registry=registry)

        spark.udf.register("ai_similarity", _udf_ai_similarity)
        out.append("ai_similarity")

    if want("ai_prep_search"):
        schema = ArrayType(
            StructType(
                [
                    StructField("chunk", StringType(), True),
                    StructField("embedding", ArrayType(FloatType()), True),
                ]
            )
        )

        @pandas_udf(schema)
        def _udf_ai_prep_search(text: pd.Series, size: pd.Series, overlap: pd.Series) -> pd.Series:
            endpoint = _default_embedding_endpoint(registry)
            return ai_prep_search_impl(endpoint, text, size, overlap, registry=registry)

        spark.udf.register("ai_prep_search", _udf_ai_prep_search)
        out.append("ai_prep_search")

    if want("ai_parse_document"):
        parse_schema = StructType(
            [
                StructField("markdown", StringType(), True),
                StructField("text", StringType(), True),
                StructField(
                    "pages",
                    ArrayType(
                        StructType(
                            [
                                StructField("page_num", IntegerType(), True),
                                StructField("text", StringType(), True),
                            ]
                        )
                    ),
                    True,
                ),
                StructField("metadata", MapType(StringType(), StringType()), True),
            ]
        )

        @pandas_udf(parse_schema)
        def _udf_ai_parse(content: pd.Series) -> pd.Series:
            return parse_document_impl("", content)

        spark.udf.register("ai_parse_document", _udf_ai_parse)
        out.append("ai_parse_document")

    return out


def _default_endpoint(registry: EndpointRegistry) -> str:
    env = os.environ.get("SPARK_AI_DEFAULT_ENDPOINT")
    if env:
        return env
    for c in registry.list_all():
        if c.endpoint_type.endswith("chat"):
            return c.name
    raise RuntimeError(
        "No default chat endpoint configured. Set SPARK_AI_DEFAULT_ENDPOINT or register a chat endpoint."
    )


def _default_embedding_endpoint(registry: EndpointRegistry) -> str:
    env = os.environ.get("SPARK_AI_DEFAULT_EMBEDDING_ENDPOINT")
    if env:
        return env
    for c in registry.list_all():
        if "embedding" in c.endpoint_type:
            return c.name
    raise RuntimeError(
        "No default embedding endpoint configured. Set SPARK_AI_DEFAULT_EMBEDDING_ENDPOINT or "
        "register an endpoint with endpoint_type=openai_embedding."
    )

