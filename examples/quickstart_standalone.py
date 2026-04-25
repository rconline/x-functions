"""Standalone quickstart — no Gravitino, endpoints from YAML, audit to stdout.

Run:
    export OPENAI_API_KEY=sk-...
    python examples/quickstart_standalone.py
"""

from __future__ import annotations

from pyspark.sql import SparkSession

from spark_ai_functions import register


def main():
    spark = (
        SparkSession.builder
        .appName("spark-ai-functions-quickstart")
        .master("local[*]")
        .getOrCreate()
    )

    ai = register(
        spark,
        yaml_path="examples/endpoints.example.yaml",
    )
    print(f"Registered functions: {ai.registered_function_names}")

    df = spark.createDataFrame(
        [
            (1, "I love this product!"),
            (2, "This is the worst experience I've ever had."),
            (3, "It works, I guess."),
        ],
        ["id", "body"],
    )

    # Sentiment
    df.selectExpr(
        "id",
        "body",
        "ai_analyze_sentiment(body) AS sentiment",
    ).show(truncate=False)

    # Classify
    df.selectExpr(
        "id",
        "ai_classify(body, array('bug', 'feature_request', 'feedback')) AS label",
    ).show(truncate=False)

    # Summarize
    df.selectExpr(
        "id",
        "ai_summarize(body, 8) AS summary",
    ).show(truncate=False)

    # Generic ai_query
    df.selectExpr(
        "id",
        "ai_query('openai-gpt-4o-mini', concat('Translate to pirate English: ', body)) AS pirate",
    ).show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
