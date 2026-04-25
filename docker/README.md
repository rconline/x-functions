# Docker stack (Gravitino + Ranger + Spark + MySQL)

This compose file is **adapted from [apache/gravitino-playground](https://github.com/apache/gravitino-playground)**
(Apache 2.0, same license as this project). Per §17.3 we do not build a fresh
Ranger image — the only reliably maintained one is
`apache/gravitino-playground:ranger-0.1.0`, shipped by the playground.

## What we kept

| Service | Why |
| --- | --- |
| `gravitino` | Metalake + Model/UDF catalogs + authz push-down |
| `ranger` | Policy store |
| `mysql` | Gravitino's persistent backend |
| `spark` | Spark 3.4.1 with the Gravitino plugin preloaded |

## What we dropped (from the playground)

`trino`, `jupyter`, `prometheus`, `grafana`, `hive`, `hdfs`, `postgres` — none
are needed by this project and each adds 1–3 GiB of startup work.

Removing `hive` also requires editing the `gravitino` service's `depends_on`
(we drop the `hive: service_healthy` condition present in the upstream file).

## Provenance files

`init/` and `healthcheck/` directories should be copied verbatim from the
upstream playground at the same image tag (1.2.0 / ranger-0.1.0).

```bash
# From the repo root:
git clone --depth 1 https://github.com/apache/gravitino-playground /tmp/gp
cp -r /tmp/gp/init docker/init
cp -r /tmp/gp/healthcheck docker/healthcheck
# Remove directories for services we don't run:
rm -rf docker/init/{trino,jupyter,prometheus,grafana,hive,postgres}
```

## Host ports

| Port | Service |
| --- | --- |
| 8090 | Gravitino REST |
| 9001 | Gravitino admin |
| 6080 | Ranger admin UI |
| 13306 | MySQL (mapped from container 3306) |
| 14040 | Spark UI |

The 1-prefix pattern avoids collisions with Docker Desktop defaults.

## Starting / stopping

```bash
cd docker && docker compose up -d
# ... do work ...
docker compose down -v   # remove volumes (Gravitino's MySQL data)
```

Expect 90–180 s for the first boot. Ranger in particular sits in its healthcheck
loop while its DB initialises.

## Preflight for Phase 4.5 (§19.4.5)

```bash
# from docker/
docker compose up -d
curl -sf http://localhost:8090/api/metalakes
curl -sf http://localhost:6080/service/public/v2/api/servicedef
```
