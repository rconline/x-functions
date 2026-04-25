"""MLflow Deployments backend — §17.5.

API shape is deliberately minimal: `get_deploy_client(target_uri).predict(endpoint, inputs)`.
If upstream renames the module or flips argument names, this is the only file
that needs patching.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class MLflowDeploymentsBackend:
    def __init__(
        self,
        *,
        target_uri: str,
        endpoint: str,
        default_params: dict[str, Any],
    ):
        self.target_uri = target_uri
        self.endpoint = endpoint
        self._default_params = default_params

    def batch_chat_complete(self, messages: pd.Series, params: dict) -> pd.Series:
        client = self._client()
        merged = {**self._default_params, **params}
        out: list[Any] = []
        for msg in messages:
            payload = {"messages": _coerce_messages(msg), **merged}
            resp = client.predict(endpoint=self.endpoint, inputs=payload)
            out.append(_extract_chat(resp))
        return pd.Series(out, index=messages.index)

    def batch_embed(self, texts: pd.Series, params: dict) -> pd.Series:
        client = self._client()
        merged = {**self._default_params, **params}
        out: list[list[float]] = []
        for t in texts:
            resp = client.predict(endpoint=self.endpoint, inputs={"input": t, **merged})
            out.append(_extract_embedding(resp))
        return pd.Series(out, index=texts.index)

    def _client(self):
        from mlflow.deployments import get_deploy_client  # mlflow>=2.14

        return get_deploy_client(self.target_uri)


def _coerce_messages(x):
    if isinstance(x, str):
        return [{"role": "user", "content": x}]
    if isinstance(x, list):
        return x
    return [{"role": "user", "content": str(x)}]


def _extract_chat(resp):
    # Accept common shapes: OpenAI-style, raw string, or dict with "choices".
    if isinstance(resp, str):
        return resp
    if hasattr(resp, "choices") and resp.choices:
        return resp.choices[0].message["content"] if isinstance(resp.choices[0].message, dict) else resp.choices[0].message.content
    if isinstance(resp, dict):
        choices = resp.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            return msg.get("content")
    return None


def _extract_embedding(resp):
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        data = resp.get("data") or []
        if data:
            first = data[0]
            if isinstance(first, dict):
                return first.get("embedding") or first.get("values")
            return first
        if "embedding" in resp:
            return resp["embedding"]
    return None
