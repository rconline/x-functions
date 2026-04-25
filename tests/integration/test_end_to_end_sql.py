"""Integration test — requires a real endpoint.

Skipped unless `OPENAI_API_KEY` and `PYSPARK_TESTS=1` are set.
"""

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") and os.environ.get("PYSPARK_TESTS")),
    reason="set OPENAI_API_KEY and PYSPARK_TESTS=1",
)
def test_ai_query_round_trip(spark, tmp_path):
    import yaml
    from spark_ai_functions import register

    ep_yaml = tmp_path / "endpoints.yaml"
    ep_yaml.write_text(yaml.safe_dump({
        "endpoints": [{
            "name": "openai-gpt-4o-mini",
            "endpoint_type": "openai_chat",
            "base_url": "https://api.openai.com/v1",
            "model_id": "gpt-4o-mini",
            "credential_name": "openai",
            "default_params": {"temperature": 0.0, "max_tokens": 64},
            "data_residency": "external",
        }]
    }))
    os.environ["SPARK_AI_DEFAULT_ENDPOINT"] = "openai-gpt-4o-mini"
    register(spark, yaml_path=str(ep_yaml))
    df = spark.createDataFrame([(1, "Say 'hi'"),], ["id", "req"])
    out = df.selectExpr("id", "ai_query('openai-gpt-4o-mini', req) AS r").collect()
    assert len(out) == 1
    assert isinstance(out[0].r, str)
