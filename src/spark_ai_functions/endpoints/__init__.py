"""Endpoint abstraction layer.

Layer 1 per §3: resolves endpoint names, then returns a `Backend` the core
functions can call without knowing whether the OpenAI SDK, MLflow Deployments,
or a future adapter is underneath.
"""

from .registry import EndpointConfig, EndpointRegistry, EndpointSource
from .yaml_source import YamlEndpointSource
from .openai_backend import OpenAIChatBackend, OpenAIEmbeddingBackend
from .mlflow_backend import MLflowDeploymentsBackend

__all__ = [
    "EndpointConfig",
    "EndpointRegistry",
    "EndpointSource",
    "YamlEndpointSource",
    "OpenAIChatBackend",
    "OpenAIEmbeddingBackend",
    "MLflowDeploymentsBackend",
]
