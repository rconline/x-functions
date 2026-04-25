#!/usr/bin/env bash
# Fetches init/ and healthcheck/ from apache/gravitino-playground at the pinned
# image tag and strips services we don't run. Idempotent.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="${TMPDIR:-/tmp}/gravitino-playground-bootstrap"
TAG="${GRAVITINO_VERSION:-1.2.0}"

echo "→ Cloning apache/gravitino-playground (shallow)…"
rm -rf "$TMP"
git clone --depth 1 https://github.com/apache/gravitino-playground "$TMP"

echo "→ Copying init/ and healthcheck/…"
rm -rf "$HERE/init" "$HERE/healthcheck"
cp -r "$TMP/init" "$HERE/init"
cp -r "$TMP/healthcheck" "$HERE/healthcheck"

echo "→ Stripping unused service dirs…"
rm -rf "$HERE"/init/trino "$HERE"/init/jupyter "$HERE"/init/prometheus "$HERE"/init/grafana "$HERE"/init/hive "$HERE"/init/postgres "$HERE"/init/hdfs || true
rm -f "$HERE"/healthcheck/hive-healthcheck.sh "$HERE"/healthcheck/trino-healthcheck.sh || true

echo "✓ Bootstrap complete. Run: cd $HERE && docker compose up -d"
