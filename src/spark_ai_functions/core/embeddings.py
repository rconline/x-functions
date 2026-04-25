"""Embedding-backed functions: `ai_similarity` + `ai_prep_search`.

All of these reduce to `backend.batch_embed(texts, params)` under the hood.
The chunker in `_core_prep_search` is deliberately simple (character-based
with overlap) — callers who want true token-aware chunking can swap it out.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd

from ..endpoints.registry import EndpointRegistry
from ..governance.decorator import governed


def _core_embed(
    *,
    registry: EndpointRegistry,
    endpoint_name: str,
    texts: pd.Series,
    credential: str,
    params: Optional[dict[str, Any]] = None,
) -> pd.Series:
    backend = registry.make_backend(endpoint_name, credential)
    return backend.batch_embed(texts, params or {})  # type: ignore[attr-defined]


def _core_similarity(
    *,
    registry: EndpointRegistry,
    endpoint_name: str,
    a: pd.Series,
    b: pd.Series,
    credential: str,
) -> pd.Series:
    backend = registry.make_backend(endpoint_name, credential)
    emb_a = backend.batch_embed(a, {})  # type: ignore[attr-defined]
    emb_b = backend.batch_embed(b, {})  # type: ignore[attr-defined]
    return pd.Series(
        [_cosine(x, y) for x, y in zip(emb_a.tolist(), emb_b.tolist())],
        index=a.index,
        dtype="float64",
    )


def _cosine(a, b):
    if a is None or b is None:
        return None
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _core_prep_search(
    *,
    registry: EndpointRegistry,
    endpoint_name: str,
    texts: pd.Series,
    chunk_size: int,
    chunk_overlap: int,
    credential: str,
) -> pd.Series:
    """For each input text, produce a list of {chunk, embedding} structs."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("0 <= chunk_overlap < chunk_size")

    # Step 1: chunk every text into its own list.
    per_row_chunks = [_chunk_text(str(t), chunk_size, chunk_overlap) for t in texts]

    # Step 2: flatten for a single batched embed call.
    flat_chunks: list[str] = []
    lengths: list[int] = []
    for chunks in per_row_chunks:
        flat_chunks.extend(chunks)
        lengths.append(len(chunks))

    if not flat_chunks:
        return pd.Series([[] for _ in texts], index=texts.index)

    backend = registry.make_backend(endpoint_name, credential)
    embeddings = backend.batch_embed(pd.Series(flat_chunks), {}).tolist()  # type: ignore[attr-defined]

    # Step 3: re-assemble per-row structs.
    out: list[list[dict]] = []
    cursor = 0
    for chunks, n in zip(per_row_chunks, lengths):
        row = [
            {"chunk": chunks[i], "embedding": embeddings[cursor + i]}
            for i in range(n)
        ]
        out.append(row)
        cursor += n
    return pd.Series(out, index=texts.index)


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if not text:
        return []
    step = size - overlap
    return [text[i : i + size] for i in range(0, max(len(text) - overlap, 1), step)]


@governed("ai_similarity")
def ai_similarity_impl(
    endpoint_name: str,
    a: pd.Series,
    b: pd.Series,
    *,
    credential: str,
    registry: EndpointRegistry,
) -> pd.Series:
    return _core_similarity(
        registry=registry,
        endpoint_name=endpoint_name,
        a=a,
        b=b,
        credential=credential,
    )


@governed("ai_prep_search")
def ai_prep_search_impl(
    endpoint_name: str,
    texts: pd.Series,
    chunk_size: pd.Series,
    chunk_overlap: pd.Series,
    *,
    credential: str,
    registry: EndpointRegistry,
) -> pd.Series:
    cs = int(chunk_size.iloc[0]) if len(chunk_size) else 1024
    co = int(chunk_overlap.iloc[0]) if len(chunk_overlap) else 64
    return _core_prep_search(
        registry=registry,
        endpoint_name=endpoint_name,
        texts=texts,
        chunk_size=cs,
        chunk_overlap=co,
        credential=credential,
    )
