"""The `register(spark, ...)` entrypoint — ties every other module together.

Standalone mode:
    register(spark)
    register(spark, yaml_path="endpoints.yaml")

Governed mode:
    register(
        spark,
        gravitino_uri="http://gravitino:8090",
        metalake="prod",
        catalog="ai_functions",
    )

Per §17.8: we verify the Gravitino Spark plugin is loaded before touching the
UDF catalog, and raise a clear error if it's not.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Optional

from .endpoints.registry import EndpointConfig, EndpointRegistry, EndpointSource
from .governance.audit import AuditSink, StdoutAuditSink
from .governance.credential_vending import (
    CredentialVendor,
    EnvCredentialVendor,
    GravitinoCredentialVendor,
    EndpointMetadataIndex,
)
from .governance.decorator import GovernanceContext, init_governance
from .governance.ranger_authorizer import (
    AllowAllAuthorizer,
    GravitinoRangerAuthorizer,
    RangerAuthorizer,
)
from .governance.tag_policy import (
    ColumnTagPolicy,
    DefaultTagPolicyEnforcer,
    DictEndpointResidency,
    PassThroughTagPolicyEnforcer,
    PIIMasker,
    StaticRoleLookup,
    TagPolicyEnforcer,
)
from .governance.user_resolver import default_chain
from .presets.loader import Preset, load_presets
from .runtime_config import (
    default_yaml_path,
    load_endpoints_json_file,
    load_endpoints_json_text,
    register_defaults_from_env,
    resolve_endpoint_sources,
)
from .udf_registration import register_udfs

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


GOVERNED_PLUGIN_CLASS = "GravitinoSparkPlugin"


@dataclass
class AIFunctions:
    """Handle returned from `register()`. Keeps references to components so
    tests can inspect the wired-up graph.
    """

    spark: Any
    registry: EndpointRegistry
    governance: GovernanceContext
    presets: dict[str, Preset] = field(default_factory=dict)
    mode: str = "standalone"                # "standalone" | "governed"
    registered_function_names: list[str] = field(default_factory=list)

    def forecast(
        self,
        df,
        *,
        horizon,
        time_col: str,
        value_col: str,
        group_cols: Optional[Iterable[str]] = None,
        frequency: str = "D",
        parameters: Optional[str | dict[str, Any]] = None,
    ):
        """Helper for SQL callers on Spark 3.4 where TVFs aren't first-class.

            out = ai.forecast(sales_df, horizon="2026-07-01", time_col="ds",
                              value_col="y", group_cols=["region"])
        """
        from .core.forecast import forecast_impl

        # Accept either a Spark DataFrame or a pandas DataFrame.
        pdf = df.toPandas() if hasattr(df, "toPandas") else df
        return forecast_impl(
            pdf,
            horizon=horizon,
            time_col=time_col,
            value_col=value_col,
            group_cols=group_cols,
            frequency=frequency,
            parameters=parameters,
        )


def register(
    spark: "SparkSession",
    *,
    # endpoint sources
    yaml_path: Optional[str] = None,
    endpoint_config_json: Optional[str] = None,
    endpoint_config_file: Optional[str] = None,
    yaml_endpoints: Optional[list[EndpointConfig]] = None,
    # governed-mode config
    gravitino_uri: Optional[str] = None,
    metalake: Optional[str] = None,
    catalog: Optional[str] = None,
    endpoints_schema: str = "endpoints",
    functions_schema: str = "functions",
    # governance component overrides
    user: Optional[str] = None,
    tag_policy: Optional[ColumnTagPolicy] = None,
    audit_sink: Optional[AuditSink] = None,
    authorizer: Optional[RangerAuthorizer] = None,
    tag_policy_enforcer: Optional[TagPolicyEnforcer] = None,
    credential_vendor: Optional[CredentialVendor] = None,
    pii_masker: Optional[PIIMasker] = None,
    # testing / extension hooks
    additional_sources: Optional[Iterable[EndpointSource]] = None,
    presets_path: Optional[str] = None,
    function_names: Optional[Iterable[str]] = None,
    skip_plugin_check: bool = False,
) -> AIFunctions:
    """Wire up endpoint resolution, governance, and Spark UDF registration."""
    mode = "governed" if gravitino_uri else "standalone"
    if mode == "governed" and (not metalake or not catalog):
        raise ValueError("Governed mode requires both metalake and catalog")

    sources: list[EndpointSource] = []
    if mode == "governed":
        from .endpoints.gravitino_source import GravitinoEndpointSource

        sources.append(
            GravitinoEndpointSource(
                gravitino_uri=gravitino_uri,
                metalake=metalake,
                catalog=catalog,
                schema=endpoints_schema,
            )
        )
    sources.extend(
        resolve_endpoint_sources(
            yaml_endpoints=yaml_endpoints,
            endpoint_config_file=endpoint_config_file,
            endpoint_config_json=endpoint_config_json,
            yaml_path=yaml_path,
            additional_sources=additional_sources,
        )
    )
    registry = EndpointRegistry(sources)

    # ---- Governance context ----
    presets = load_presets(presets_path) if presets_path else load_presets()

    resolver = default_chain(user or os.environ.get("SPARK_AI_USER"))

    if audit_sink is None:
        audit_sink = StdoutAuditSink()

    if authorizer is None:
        if mode == "governed":
            authorizer = GravitinoRangerAuthorizer(
                gravitino_uri=gravitino_uri, metalake=metalake, catalog=catalog
            )
        else:
            authorizer = AllowAllAuthorizer()

    if credential_vendor is None:
        if mode == "governed":
            meta_index = EndpointMetadataIndex(loader=registry.get)
            try:
                from gravitino import GravitinoClient
                client = GravitinoClient(uri=gravitino_uri, metalake_name=metalake)
            except Exception:
                client = None
            credential_vendor = GravitinoCredentialVendor(
                gravitino_client=client, catalog=catalog, endpoint_index=meta_index
            ) if client is not None else EnvCredentialVendor()
        else:
            credential_vendor = EnvCredentialVendor()

    if tag_policy_enforcer is None:
        if mode == "governed":
            # Governance defaults on: restricted/phi/confidential rules apply
            # out of the box. If no PII masker is injected, fall back to the
            # Presidio-based masker shipped with the package (core dependency
            # per pyproject.toml).
            masker = pii_masker
            if masker is None:
                from .core.mask import PresidioMasker

                masker = PresidioMasker()
            residency = DictEndpointResidency(
                {c.name: c.data_residency for c in registry.list_all()}
            )
            tag_policy_enforcer = DefaultTagPolicyEnforcer(
                tag_policy or ColumnTagPolicy(),
                endpoint_residency=residency,
                role_lookup=StaticRoleLookup(),
                pii_masker=masker,
            )
        else:
            tag_policy_enforcer = PassThroughTagPolicyEnforcer()

    ctx = GovernanceContext(
        user_resolver=resolver,
        authorizer=authorizer,
        tag_policy=tag_policy_enforcer,
        credential_vendor=credential_vendor,
        audit_sink=audit_sink,
        catalog_name=f"{metalake}.{catalog}" if catalog else None,
    )
    init_governance(ctx)

    # ---- Governed mode: push UDF specs into Gravitino (plugin auto-discovers) ----
    if mode == "governed":
        _assert_plugin_loaded(spark, skip=skip_plugin_check)
        from .catalog.gravitino_registrar import GravitinoUDFRegistrar

        GravitinoUDFRegistrar(
            gravitino_uri=gravitino_uri,
            metalake=metalake,
            catalog=catalog,
            schema=functions_schema,
        ).ensure_registered()

    # ---- Register Pandas UDFs for driver-side SQL in both modes. ----
    # In governed mode the plugin handles SQL discovery, but spark.udf.register
    # still makes the functions callable from the same session immediately.
    names_wanted = set(function_names) if function_names else None
    registered = register_udfs(spark, registry, presets, names_wanted)

    return AIFunctions(
        spark=spark,
        registry=registry,
        governance=ctx,
        presets=presets,
        mode=mode,
        registered_function_names=registered,
    )


def register_from_env(
    spark: "SparkSession",
    **overrides: Any,
) -> AIFunctions:
    """Service-friendly entrypoint that resolves register() args from env vars.

    Useful for ODP/Ambari/xDP deployments where runtime config is injected via
    service properties instead of hardcoded Python.
    """
    mode = (os.environ.get("SPARK_AI_MODE") or "auto").strip().lower()
    defaults = register_defaults_from_env()
    if mode == "standalone":
        defaults["gravitino_uri"] = None
        defaults["metalake"] = None
        defaults["catalog"] = None
    elif mode == "governed":
        missing: list[str] = []
        if not defaults["gravitino_uri"]:
            missing.append("SPARK_AI_GRAVITINO_URI")
        if not defaults["metalake"]:
            missing.append("SPARK_AI_METALAKE")
        if not defaults["catalog"]:
            missing.append("SPARK_AI_CATALOG")
        if missing:
            raise ValueError(
                "SPARK_AI_MODE=governed requires: " + ", ".join(missing)
            )

    defaults.update(overrides)
    return register(spark, **defaults)


# Backward-compatible wrappers for tests and imports.
def _default_yaml_path() -> Optional[str]:
    return default_yaml_path()


def _load_endpoints_json_file(path: str) -> list[EndpointConfig]:
    return load_endpoints_json_file(path)


def _load_endpoints_json_text(raw: str) -> list[EndpointConfig]:
    return load_endpoints_json_text(raw)


def _assert_plugin_loaded(spark, *, skip: bool) -> None:
    if skip:
        return
    plugins = ""
    try:
        plugins = spark.conf.get("spark.plugins", "")
    except Exception:
        plugins = ""
    if GOVERNED_PLUGIN_CLASS not in plugins:
        raise RuntimeError(
            "Governed mode requires the Gravitino Spark plugin. "
            "Set spark.plugins=org.apache.gravitino.spark.connector.plugin.GravitinoSparkPlugin "
            "and include org.apache.gravitino:gravitino-spark-connector-runtime-3.4_2.12:1.2.0 on "
            "spark.jars.packages. Pass skip_plugin_check=True to bypass for tests."
        )
