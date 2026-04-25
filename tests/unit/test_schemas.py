from spark_ai_functions.schemas import (
    CHUNK_WITH_EMBEDDING_SCHEMA,
    FORECAST_SCHEMA,
    PARSE_DOCUMENT_SCHEMA,
)


def test_schemas_are_ddl_strings():
    assert "chunk" in CHUNK_WITH_EMBEDDING_SCHEMA
    assert "markdown" in PARSE_DOCUMENT_SCHEMA
    assert "yhat" in FORECAST_SCHEMA
