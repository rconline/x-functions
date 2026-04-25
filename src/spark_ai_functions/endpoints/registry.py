"""Endpoint registry — resolves endpoint names from Gravitino first, YAML fallback.

An `EndpointConfig` carries enough info to construct a `Backend` on demand via
`make_backend(credential)`. The test suite (§18.5) injects a fake backend by
monkeypatching that method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Protocol

from ..governance.errors import EndpointNotFoundError


class Backend(Protocol):
    """Minimal contract every backend must satisfy. Core functions pick between
    `batch_chat_complete` or `batch_embed` depending on their needs."""


class ChatBackend(Backend, Protocol):
    def batch_chat_complete(self, messages, params: dict) -> Any: ...


class EmbeddingBackend(Backend, Protocol):
    def batch_embed(self, texts, params: dict) -> Any: ...


@dataclass
class EndpointConfig:
    name: str
    endpoint_type: str                # "openai_chat" | "openai_embedding" | "mlflow_chat" | ...
    base_url: str
    model_id: str
    credential_name: str
    default_params: dict[str, Any] = field(default_factory=dict)
    data_residency: str = "external"   # "internal" | "external"
    extras: dict[str, Any] = field(default_factory=dict)

    # Hook overridden in tests — see §18.5.
    def make_backend(self, credential: str) -> Backend:
        from .openai_backend import OpenAIChatBackend, OpenAIEmbeddingBackend
        from .mlflow_backend import MLflowDeploymentsBackend

        # OpenAI-compatible chat endpoints (OpenAI, Azure OpenAI, vLLM, llama.cpp,
        # internal gateways speaking the OpenAI protocol). bedrock_chat and
        # anthropic_chat route here too — they work only against gateways that
        # translate to the OpenAI shape; native Bedrock/Anthropic backends are a
        # follow-up.
        if self.endpoint_type in ("openai_chat", "azure_openai_chat", "bedrock_chat", "anthropic_chat"):
            return OpenAIChatBackend(
                endpoint_name=self.name,
                base_url=self.base_url,
                model_id=self.model_id,
                credential=credential,
                default_params=self.default_params,
            )
        if self.endpoint_type in ("openai_embedding", "azure_openai_embedding"):
            return OpenAIEmbeddingBackend(
                endpoint_name=self.name,
                base_url=self.base_url,
                model_id=self.model_id,
                credential=credential,
                default_params=self.default_params,
            )
        if self.endpoint_type in ("mlflow_chat", "mlflow_embedding"):
            return MLflowDeploymentsBackend(
                target_uri=self.base_url,
                endpoint=self.model_id,
                default_params=self.default_params,
            )
        raise ValueError(f"Unsupported endpoint_type: {self.endpoint_type!r}")


class EndpointSource(Protocol):
    """Read-only lookup of endpoint configs by name."""

    def get(self, name: str) -> Optional[EndpointConfig]: ...

    def list(self) -> Iterable[EndpointConfig]: ...


class EndpointRegistry:
    """Chain of sources — first non-None wins. Gravitino is typically first."""

    def __init__(self, sources: Iterable[EndpointSource]):
        self._sources = list(sources)

    def get(self, name: str) -> EndpointConfig:
        for s in self._sources:
            cfg = s.get(name)
            if cfg is not None:
                return cfg
        raise EndpointNotFoundError(f"No endpoint {name!r} in any configured source")

    def make_backend(self, name: str, credential: str) -> Backend:
        cfg = self.get(name)
        return cfg.make_backend(credential)

    def list_all(self) -> list[EndpointConfig]:
        seen: dict[str, EndpointConfig] = {}
        for s in self._sources:
            for cfg in s.list():
                seen.setdefault(cfg.name, cfg)
        return list(seen.values())
