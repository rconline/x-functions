# %% [markdown]
# # Databricks AI Functions parity demo — standalone mode
#
# Requires `OPENAI_API_KEY` set.

# %%
import os
assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY before running"

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("spark-ai-functions-parity-demo")
    .master("local[*]")
    .getOrCreate()
)

from spark_ai_functions import register

ai = register(spark, yaml_path="../endpoints.example.yaml")
ai.registered_function_names

# %% [markdown]
# ## `ai_analyze_sentiment`

# %%
df = spark.createDataFrame(
    [
        (1, "I love it!"),
        (2, "Absolutely terrible."),
        (3, "Meh, it's fine."),
    ],
    ["id", "body"],
)
df.selectExpr("id", "body", "ai_analyze_sentiment(body) AS sentiment").show(truncate=False)

# %% [markdown]
# ## `ai_classify`

# %%
df.selectExpr("id", "ai_classify(body, array('positive','negative','neutral')) AS label").show(truncate=False)

# %% [markdown]
# ## `ai_fix_grammar`

# %%
spark.createDataFrame([(1, "she dont like it")], ["id", "body"]).selectExpr(
    "id", "ai_fix_grammar(body) AS fixed"
).show(truncate=False)

# %% [markdown]
# ## `ai_summarize`

# %%
df.selectExpr("id", "ai_summarize(body, 6) AS summary").show(truncate=False)

# %% [markdown]
# ## `ai_translate`

# %%
df.selectExpr("id", "ai_translate(body, 'French') AS fr").show(truncate=False)

# %% [markdown]
# ## `ai_query` (generic LLM call)

# %%
df.selectExpr("id", "ai_query('openai-gpt-4o-mini', concat('Rate 1-10: ', body)) AS rating").show(truncate=False)

# %% [markdown]
# ## `ai_similarity`

# %%
spark.createDataFrame([("cats", "kittens"), ("cats", "trucks")], ["a", "b"]).selectExpr(
    "a", "b", "ai_similarity(a, b) AS score"
).show()

spark.stop()
