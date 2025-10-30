"""Utility helpers for working with embedding dimensions.

This module centralises logic for discovering the source embedding
dimension, choosing the target width used by persistent stores, and
normalising vectors so they always match the configured size.  Keeping
that logic in one place helps avoid subtle mismatches that surface as
pgvector dimension errors at write time.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Default width used historically across the codebase.  We still fall back
# to this when neither configuration nor runtime detection yields a value.
DEFAULT_EMBEDDING_DIMENSION: int = 1536

# Probe text for inexpensive dimension checks.  Using a constant keeps the
# detection deterministic and makes caching easier.
_PROBE_TEXT: str = "embedding-dimension-probe"

# Cache for the resolved target dimension – most callers only need a single
# consistent number per process, so we memoise it here.
_CACHED_TARGET_DIMENSION: Optional[int] = None


def _coerce_int(value: Any) -> Optional[int]:
    """Attempt to coerce *value* to ``int``; return ``None`` on failure."""

    if value is None:
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    if coerced <= 0:
        return None
    return coerced


def _extract_dimension_from_config(config: Optional[Dict[str, Any]]) -> Optional[int]:
    """Look for an embedding dimension hint inside a config mapping."""

    if not isinstance(config, Dict):
        return None

    # Collect candidate dictionaries to inspect.  We look at the config
    # itself as well as common nested keys used across the project.
    sections: List[Dict[str, Any]] = [config]
    for key in ("vector_store", "embedding", "memory", "context", "vector"):
        section = config.get(key)
        if isinstance(section, Dict):
            sections.append(section)

    candidate_keys: Iterable[str] = (
        "dimension",
        "embedding_dim",
        "vector_dimension",
        "vector_dim",
        "dim",
    )

    for section in sections:
        for candidate in candidate_keys:
            value = section.get(candidate)
            coerced = _coerce_int(value)
            if coerced:
                return coerced

    return None


def _extract_dimension_from_env() -> Optional[int]:
    """Inspect environment overrides for the embedding dimension."""

    for key in (
        "MEMORY_EMBEDDING_DIMENSION",
        "EMBEDDING_DIMENSION",
        "DEFAULT_EMBEDDING_DIMENSION",
    ):
        value = _coerce_int(os.getenv(key))
        if value:
            return value
    return None


def _resolve_embedding_model(
    embedding_model: Optional[str],
    config: Optional[Dict[str, Any]],
) -> str:
    """Determine whether we're using local embeddings or OpenAI."""

    if embedding_model:
        return embedding_model.lower()

    if isinstance(config, Dict):
        embedding_section = config.get("embedding")
        if isinstance(embedding_section, Dict):
            embed_type = embedding_section.get("type")
            if isinstance(embed_type, str):
                return embed_type.lower()

        embed_type = config.get("embedding_type")
        if isinstance(embed_type, str):
            return embed_type.lower()

    env_override = os.getenv("MEMORY_EMBEDDING_TYPE")
    if env_override:
        return env_override.lower()

    return "local"


def measure_embedding_dimension(
    embedding_instance: Any,
    probe_text: str = _PROBE_TEXT,
) -> int:
    """Return the dimensionality of *embedding_instance* via ``embed_query``."""

    vector = embedding_instance.embed_query(probe_text)

    if hasattr(vector, "tolist"):
        vector = vector.tolist()

    if not isinstance(vector, Sequence):
        vector = list(vector)  # type: ignore[arg-type]

    dimension = len(vector)
    if dimension <= 0:
        raise ValueError("Embedding probe returned an empty vector")

    return dimension


def determine_embedding_dimension(
    embedding_model: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> int:
    """Best-effort detection of the *source* embedding dimension."""

    model_type = _resolve_embedding_model(embedding_model, config)

    try:
        if model_type == "openai":
            from langchain_community.embeddings import OpenAIEmbeddings

            embedding_cfg: Dict[str, Any] = {}
            if isinstance(config, Dict):
                embedding_cfg = config.get("embedding", {}) or {}

            model_name = embedding_cfg.get("openai_model") or config.get("openai_model") if isinstance(config, Dict) else None
            if not isinstance(model_name, str) or not model_name.strip():
                model_name = "text-embedding-3-small"

            api_key = embedding_cfg.get("api_key") if isinstance(embedding_cfg, Dict) else None
            if not isinstance(api_key, str) or not api_key.strip():
                api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required to probe OpenAI embeddings")

            embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=model_name)
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            embedding_cfg = {}
            if isinstance(config, Dict):
                embedding_cfg = config.get("embedding", {}) or {}

            model_name = embedding_cfg.get("model_name") or embedding_cfg.get("hf_embedding_model")
            if not isinstance(model_name, str) or not model_name.strip():
                # langchain defaults to sentence-transformers/all-mpnet-base-v2 when omitted,
                # but we stay explicit for reproducibility.
                model_name = "sentence-transformers/all-mpnet-base-v2"

            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        return measure_embedding_dimension(embeddings)

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "Falling back to default embedding dimension after probe failure: %s",
            exc,
        )
        configured = _extract_dimension_from_config(config)
        if configured:
            return configured
        env_override = _extract_dimension_from_env()
        if env_override:
            return env_override
        return DEFAULT_EMBEDDING_DIMENSION


def get_target_embedding_dimension(
    config: Optional[Dict[str, Any]] = None,
    embedding_model: Optional[str] = None,
) -> int:
    """Return the dimension that persisted vectors should conform to."""

    global _CACHED_TARGET_DIMENSION

    if _CACHED_TARGET_DIMENSION is not None:
        return _CACHED_TARGET_DIMENSION

    dimension = _extract_dimension_from_env()
    if dimension is None and config is not None:
        dimension = _extract_dimension_from_config(config)

    if dimension is None:
        dimension = determine_embedding_dimension(embedding_model, config)

    if dimension is None:
        dimension = DEFAULT_EMBEDDING_DIMENSION

    _CACHED_TARGET_DIMENSION = dimension
    return dimension


def adjust_embedding_vector(
    vector: Sequence[float],
    dimension: Optional[int] = None,
) -> List[float]:
    """Pad or truncate *vector* so it matches *dimension* exactly."""

    target = dimension or get_target_embedding_dimension()
    data = list(vector)

    if len(data) == target:
        return data

    if len(data) > target:
        return data[:target]

    return data + [0.0] * (target - len(data))


def build_zero_vector(dimension: Optional[int] = None) -> List[float]:
    """Return a zero-filled vector of length *dimension*."""

    target = dimension or get_target_embedding_dimension()
    return [0.0] * target


def apply_embedding_dimension(sql: str, dimension: Optional[int] = None) -> str:
    """Replace any ``vector(1536)`` occurrences with the configured width."""

    target = dimension or get_target_embedding_dimension()
    replacements = {
        "vector(1536)": f"vector({target})",
        "Vector(1536)": f"Vector({target})",
        "VECTOR(1536)": f"VECTOR({target})",
        "!= 1536": f"!= {target}",
        "expected 1536": f"expected {target}",
    }

    updated = sql
    for source, replacement in replacements.items():
        updated = updated.replace(source, replacement)

    return updated


__all__ = [
    "DEFAULT_EMBEDDING_DIMENSION",
    "adjust_embedding_vector",
    "apply_embedding_dimension",
    "build_zero_vector",
    "determine_embedding_dimension",
    "get_target_embedding_dimension",
    "measure_embedding_dimension",
]

