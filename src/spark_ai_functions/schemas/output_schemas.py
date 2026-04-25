"""PySpark StructType shapes for array/struct return types per §7.

These are expressed as DDL strings to avoid importing pyspark.sql.types at
module load (keeps `from spark_ai_functions import register` cheap).
"""

from __future__ import annotations

# ai_prep_search
CHUNK_WITH_EMBEDDING_SCHEMA = (
    "array<struct<chunk:string,embedding:array<float>>>"
)

# ai_parse_document
PARSE_DOCUMENT_SCHEMA = (
    "struct<"
    "markdown:string,"
    "text:string,"
    "pages:array<struct<page_num:int,text:string>>,"
    "metadata:map<string,string>"
    ">"
)

# ai_forecast UDTF output
FORECAST_SCHEMA = (
    "struct<"
    "ts:timestamp,"
    "yhat:double,"
    "yhat_lower:double,"
    "yhat_upper:double,"
    "group_key:string"
    ">"
)
