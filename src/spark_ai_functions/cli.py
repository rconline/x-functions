"""Command-line helper — §6.

Supports `register-endpoint` (Model Catalog) and `register-functions` (UDF Catalog).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="spark-ai-functions")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ep = sub.add_parser("register-endpoint", help="Register an LLM endpoint as a Gravitino Model")
    p_ep.add_argument("--gravitino-uri", required=True)
    p_ep.add_argument("--metalake", required=True)
    p_ep.add_argument("--catalog", required=True)
    p_ep.add_argument("--schema", default="endpoints")
    p_ep.add_argument("--name", required=True)
    p_ep.add_argument("--type", dest="endpoint_type", required=True,
                      help="e.g. openai_chat, openai_embedding, mlflow_chat")
    p_ep.add_argument("--base-url", required=True)
    p_ep.add_argument("--model-id", required=True)
    p_ep.add_argument("--credential-name", required=True)
    p_ep.add_argument("--default-params", default="{}",
                      help="JSON of default_params; e.g. '{\"temperature\":0.0}'")
    p_ep.add_argument("--data-residency", default="external",
                      choices=["internal", "external"])

    p_fn = sub.add_parser("register-functions", help="Register the 14 SQL functions in Gravitino")
    p_fn.add_argument("--gravitino-uri", required=True)
    p_fn.add_argument("--metalake", required=True)
    p_fn.add_argument("--catalog", required=True)
    p_fn.add_argument("--schema", default="functions")

    p_pre = sub.add_parser("preflight", help="Run §19.0 preflight checks and emit PREFLIGHT-REPORT.md")
    p_pre.add_argument("--output", default="PREFLIGHT-REPORT.md")

    args = parser.parse_args(argv)

    if args.cmd == "register-endpoint":
        from .catalog.gravitino_registrar import register_endpoint_model
        register_endpoint_model(
            gravitino_uri=args.gravitino_uri,
            metalake=args.metalake,
            catalog=args.catalog,
            schema=args.schema,
            name=args.name,
            endpoint_type=args.endpoint_type,
            base_url=args.base_url,
            model_id=args.model_id,
            credential_name=args.credential_name,
            default_params=json.loads(args.default_params),
            data_residency=args.data_residency,
        )
        print(f"Registered endpoint {args.name!r} in {args.metalake}.{args.catalog}.{args.schema}")
        return 0

    if args.cmd == "register-functions":
        from .catalog.gravitino_registrar import GravitinoUDFRegistrar
        registrar = GravitinoUDFRegistrar(
            gravitino_uri=args.gravitino_uri,
            metalake=args.metalake,
            catalog=args.catalog,
            schema=args.schema,
        )
        touched = registrar.ensure_registered()
        print(f"Registered {len(touched)} functions: {', '.join(touched)}")
        return 0

    if args.cmd == "preflight":
        from .preflight import run_preflight
        ok = run_preflight(args.output)
        return 0 if ok else 1

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
