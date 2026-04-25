"""OpenAI-compatible chat & embedding backends.

Pure-Python adapter built on the official `openai` SDK. Works against any
OpenAI-compatible HTTP endpoint (OpenAI, Azure OpenAI, vLLM, llama.cpp, an
internal gateway, …) by setting `base_url` accordingly. No Spark JVM jar is
required, so notebook setup is just `pip install`.

Per-batch concurrency is bounded by `default_params["max_concurrency"]`
(default 8). The openai SDK handles retries internally; per-call timeouts
flow through `default_params["timeout"]`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd


_DEFAULT_CONCURRENCY = 8
_EMBED_CHUNK = 1024


class OpenAIChatBackend:
    """Concurrent chat-completion adapter.

    `messages` is a Pandas Series of either raw user strings or full OpenAI
    message arrays (when the caller pre-renders system+user themselves).
    """

    def __init__(
        self,
        *,
        endpoint_name: str,
        base_url: str,
        model_id: str,
        credential: str,
        default_params: dict[str, Any],
    ):
        self.endpoint_name = endpoint_name
        self.base_url = base_url
        self.model_id = model_id
        self._credential = credential
        self._default_params = default_params

    def batch_chat_complete(self, messages: pd.Series, params: dict) -> pd.Series:
        from openai import OpenAI

        merged = {**self._default_params, **params}
        max_workers = max(1, int(merged.get("max_concurrency") or _DEFAULT_CONCURRENCY))
        client = OpenAI(**_client_kwargs(self._credential, self.base_url, merged))
        call_params = _openai_params(merged)

        def _one(msg: Any):
            completion = client.chat.completions.create(
                model=self.model_id,
                messages=_coerce_messages(msg),
                **call_params,
            )
            return completion.choices[0].message.content if completion.choices else None

        index = messages.index
        values = list(messages)
        if max_workers == 1 or len(values) <= 1:
            results = [_one(v) for v in values]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                results = list(pool.map(_one, values))
        return pd.Series(results, index=index)


class OpenAIEmbeddingBackend:
    """Embedding adapter — chunk-batched over the OpenAI `embeddings.create` call."""

    def __init__(
        self,
        *,
        endpoint_name: str,
        base_url: str,
        model_id: str,
        credential: str,
        default_params: dict[str, Any],
    ):
        self.endpoint_name = endpoint_name
        self.base_url = base_url
        self.model_id = model_id
        self._credential = credential
        self._default_params = default_params

    def batch_embed(self, texts: pd.Series, params: dict) -> pd.Series:
        from openai import OpenAI

        merged = {**self._default_params, **params}
        client = OpenAI(**_client_kwargs(self._credential, self.base_url, merged))
        call_params = _openai_params(merged, embedding=True)

        idx = list(texts.index)
        values = texts.tolist()
        out: list[list[float]] = []
        for start in range(0, len(values), _EMBED_CHUNK):
            chunk = values[start : start + _EMBED_CHUNK]
            resp = client.embeddings.create(
                model=self.model_id, input=chunk, **call_params
            )
            out.extend([d.embedding for d in resp.data])
        return pd.Series(out, index=idx)


def _client_kwargs(credential: str, base_url: str, merged: dict) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"api_key": credential, "base_url": base_url}
    timeout = merged.get("timeout")
    if timeout is not None:
        kwargs["timeout"] = timeout
    return kwargs


def _coerce_messages(x: Any) -> list[dict[str, str]]:
    if isinstance(x, str):
        return [{"role": "user", "content": x}]
    if isinstance(x, list):
        return x
    return [{"role": "user", "content": str(x)}]


def _openai_params(params: dict, *, embedding: bool = False) -> dict:
    """Drop adapter-only fields the OpenAI SDK doesn't accept on the call itself."""
    banned = {"max_concurrency", "concurrency", "timeout", "retry_policy"}
    out = {k: v for k, v in params.items() if k not in banned}
    if embedding:
        out.pop("temperature", None)
        out.pop("max_tokens", None)
        out.pop("response_format", None)
    return out
