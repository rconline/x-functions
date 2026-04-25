"""Governed-mode quickstart — endpoints from Gravitino, Ranger authz, audit ships to your sink.

Prerequisites:
    cd docker && ./bootstrap.sh && docker compose up -d

Then register an endpoint via the CLI:
    spark-ai-functions register-endpoint \\
      --gravitino-uri http://localhost:8090 \\
      --metalake prod --catalog ai_functions --schema endpoints \\
      --name gpt-4o-mini \\
      --type openai_chat \\
      --base-url https://api.openai.com/v1 \\
      --model-id gpt-4o-mini \\
      --credential-name openai-prod
"""

from __future__ import annotations

from pyspark.sql import SparkSession

from spark_ai_functions import execute_as, register


def main():
    spark = (
        SparkSession.builder
        .appName("spark-ai-functions-governed-quickstart")
        .master("local[*]")
        .config(
            "spark.jars.packages",
            "org.apache.gravitino:gravitino-spark-connector-runtime-3.4_2.12:1.2.0",
        )
        .config(
            "spark.plugins",
            "org.apache.gravitino.spark.connector.plugin.GravitinoSparkPlugin",
        )
        .config("spark.sql.gravitino.uri", "http://localhost:8090")
        .config("spark.sql.gravitino.metalake", "prod")
        .getOrCreate()
    )

    ai = register(
        spark,
        gravitino_uri="http://localhost:8090",
        metalake="prod",
        catalog="ai_functions",
    )
    print(f"Mode: {ai.mode} | registered: {ai.registered_function_names}")

    df = spark.createDataFrame(
        [(1, "Please reset my password"), (2, "Ticket #123 is urgent")],
        ["id", "body"],
    )

    # Scope the query to a specific user (§18.3). The UDFs will see this via
    # TaskContext.getLocalProperty("spark.ai.user").
    with execute_as(spark, "alice@company.com"):
        df.selectExpr(
            "id",
            "ai_query('gpt-4o-mini', concat('Summarise in 10 words: ', body)) AS summary",
        ).show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
