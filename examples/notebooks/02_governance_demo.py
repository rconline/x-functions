# %% [markdown]
# # Governance demo — Gravitino + Ranger + Spark
#
# Prereq: `cd docker && ./bootstrap.sh && docker compose up -d`.

# %%
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("spark-ai-functions-governance-demo")
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

from spark_ai_functions import execute_as, register
from spark_ai_functions.governance.audit import InMemoryAuditSink

audit = InMemoryAuditSink()
ai = register(
    spark,
    gravitino_uri="http://localhost:8090",
    metalake="prod",
    catalog="ai_functions",
    audit_sink=audit,
)

# %% [markdown]
# ## 1. Allowed invocation as alice

# %%
df = spark.createDataFrame([(1, "order number 123 was late")], ["id", "body"])
with execute_as(spark, "alice@company.com"):
    df.selectExpr("id", "ai_analyze_sentiment(body) AS s").show()
[e for e in audit.events if e["status"] == "success"]

# %% [markdown]
# ## 2. Denied invocation as mallory

# %%
try:
    with execute_as(spark, "mallory@company.com"):
        df.selectExpr("ai_query('gpt-4o-mini', body)").show()
except Exception as e:
    print("denied:", e)
[e for e in audit.events if e["status"] == "denied"]

# %% [markdown]
# ## 3. PII-tagged column auto-masked

# %%
# (see docs/governance_model.md for how column tags are declared in Gravitino)

spark.stop()
