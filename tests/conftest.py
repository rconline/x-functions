"""Pytest fixtures — §18.4, §12.

Notes:
  - Spark fixture is session-scoped and gated on `PYSPARK_TESTS=1` so that
    unit tests (which should never touch Spark) run without a JVM.
  - `audit_sink` / `governance_context` / `registered` give unit tests a
    pre-wired governance plane with an in-memory audit buffer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import pytest

# Ensure tests work without requiring editable install or manual PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spark_ai_functions.endpoints.registry import EndpointConfig, EndpointRegistry
from spark_ai_functions.endpoints.yaml_source import InMemoryEndpointSource
from spark_ai_functions.governance.audit import InMemoryAuditSink
from spark_ai_functions.governance.credential_vending import StaticCredentialVendor
from spark_ai_functions.governance.decorator import GovernanceContext, init_governance
from spark_ai_functions.governance.ranger_authorizer import AllowAllAuthorizer
from spark_ai_functions.governance.tag_policy import PassThroughTagPolicyEnforcer
from spark_ai_functions.governance.user_resolver import ExplicitUserResolver


# ---- Fake backend (§18.5) ----

class FakeChatBackend:
    def __init__(self, canned: Iterable[str]):
        self._canned = list(canned)

    def batch_chat_complete(self, series, params):  # noqa: ARG002
        out = []
        for i, _ in enumerate(series):
            out.append(self._canned[i % len(self._canned)] if self._canned else None)
        return pd.Series(out, index=series.index)


class FakeEmbeddingBackend:
    def __init__(self, dim: int = 4):
        self._dim = dim

    def batch_embed(self, texts, params):  # noqa: ARG002
        # Deterministic per-input vector for test assertions.
        def vec(s: str) -> list[float]:
            base = [float((ord(c) if s else 0) + i) for i, c in enumerate((s or "_")[: self._dim])]
            if len(base) < self._dim:
                base.extend([0.0] * (self._dim - len(base)))
            return base

        return pd.Series([vec(str(t)) for t in texts], index=texts.index)


# ---- Fixtures ----

@pytest.fixture
def fake_endpoint_config(monkeypatch):
    cfg = EndpointConfig(
        name="t-chat",
        endpoint_type="openai_chat",
        base_url="https://example.invalid/v1",
        model_id="gpt-x",
        credential_name="openai",
        default_params={},
        data_residency="external",
    )
    monkeypatch.setattr(cfg, "make_backend", lambda credential: FakeChatBackend(["A", "B", "C"]))
    return cfg


@pytest.fixture
def fake_embedding_config(monkeypatch):
    cfg = EndpointConfig(
        name="t-embed",
        endpoint_type="openai_embedding",
        base_url="https://example.invalid/v1",
        model_id="text-embed",
        credential_name="openai",
        default_params={},
        data_residency="internal",
    )
    monkeypatch.setattr(cfg, "make_backend", lambda credential: FakeEmbeddingBackend(dim=4))
    return cfg


@pytest.fixture
def registry(fake_endpoint_config, fake_embedding_config) -> EndpointRegistry:
    src = InMemoryEndpointSource([fake_endpoint_config, fake_embedding_config])
    return EndpointRegistry([src])


@pytest.fixture
def audit_sink() -> InMemoryAuditSink:
    return InMemoryAuditSink()


@pytest.fixture
def governance_context(audit_sink) -> GovernanceContext:
    ctx = GovernanceContext(
        user_resolver=ExplicitUserResolver("alice@test"),
        authorizer=AllowAllAuthorizer(),
        tag_policy=PassThroughTagPolicyEnforcer(),
        credential_vendor=StaticCredentialVendor({"t-chat": "k", "t-embed": "k"}),
        audit_sink=audit_sink,
        catalog_name="test.ai_functions",
    )
    init_governance(ctx)
    return ctx


@pytest.fixture
def assert_audit_event(audit_sink):
    def _assert(**criteria):
        matches = audit_sink.find(**criteria)
        if not matches:
            raise AssertionError(
                f"no audit event matching {criteria}. Events: {audit_sink.events!r}"
            )
        return matches
    return _assert


# ---- Spark fixture (§18.4) ----

@pytest.fixture(scope="session")
def spark():
    if not os.environ.get("PYSPARK_TESTS"):
        pytest.skip("Set PYSPARK_TESTS=1 to run Spark-backed tests.")

    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder
        .appName("spark-ai-functions-tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
    )
    if os.environ.get("SPARK_AI_GOVERNED_TESTS"):
        builder = (
            builder
            .config("spark.jars.packages",
                    "org.apache.gravitino:gravitino-spark-connector-runtime-3.4_2.12:1.2.0")
            .config("spark.plugins",
                    "org.apache.gravitino.spark.connector.plugin.GravitinoSparkPlugin")
            .config("spark.sql.gravitino.uri", "http://localhost:8090")
            .config("spark.sql.gravitino.metalake", "test")
        )
    session = builder.getOrCreate()
    yield session
    session.stop()
