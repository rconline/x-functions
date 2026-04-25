"""Core function implementations — Layer 2.

Each pure function is decorator-ready: its first positional arg is
`endpoint_name` (a string), subsequent positional args are Pandas `Series`,
and `credential=` is a keyword-only final arg (so the `@governed` decorator
can inject it). See §18.1.
"""

from .ai_query import _core_ai_query, ai_query_impl
from .embeddings import _core_embed, _core_prep_search, _core_similarity
from .forecast import forecast_impl
from .mask import _core_ai_mask, mask_impl
from .parse_document import parse_document_impl

__all__ = [
    "ai_query_impl",
    "_core_ai_query",
    "_core_similarity",
    "_core_prep_search",
    "_core_embed",
    "forecast_impl",
    "mask_impl",
    "_core_ai_mask",
    "parse_document_impl",
]
