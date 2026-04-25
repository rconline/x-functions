"""spark-ai-functions — Databricks AI Functions parity for vanilla Spark, governed via Gravitino + Ranger."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from .register import register, register_from_env

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

__all__ = ["register", "register_from_env", "execute_as", "__version__"]
__version__ = "0.3.0"


@contextmanager
def execute_as(spark: "SparkSession", user: str):
    """Scope a block of Spark queries to a specific user identity.

        with execute_as(spark, "alice@company.com"):
            spark.sql("SELECT ai_query(...)").show()

    Sets the `spark.ai.user` local property on the driver SparkContext;
    `SparkPropertyUserResolver` picks it up from `TaskContext` on executors.
    """
    sc = spark.sparkContext
    prior = sc.getLocalProperty("spark.ai.user")
    sc.setLocalProperty("spark.ai.user", user)
    try:
        yield
    finally:
        sc.setLocalProperty("spark.ai.user", prior if prior is not None else "")
