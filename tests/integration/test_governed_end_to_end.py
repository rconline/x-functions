"""§15 governance scenario — requires docker-compose stack.

Skipped unless `SPARK_AI_GOVERNED_TESTS=1` is set. See §19.4.5 for the
preflight that the stack is healthy before running this suite.
"""

import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.governed]


@pytest.mark.skipif(
    not os.environ.get("SPARK_AI_GOVERNED_TESTS"),
    reason="set SPARK_AI_GOVERNED_TESTS=1 and start docker/docker-compose.yml",
)
def test_gravitino_reachable():
    import urllib.request

    req = urllib.request.Request(
        os.environ.get("GRAVITINO_URI", "http://localhost:8090") + "/api/metalakes"
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200


@pytest.mark.skipif(
    not os.environ.get("SPARK_AI_GOVERNED_TESTS"),
    reason="set SPARK_AI_GOVERNED_TESTS=1 and start docker/docker-compose.yml",
)
def test_ranger_reachable():
    import urllib.request

    req = urllib.request.Request("http://localhost:6080/service/public/v2/api/servicedef")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
