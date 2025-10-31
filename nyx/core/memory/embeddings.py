"""Shared embedding helpers for Nyx memory components.

This module centralises the logic for selecting an embedding backend.  When
the configuration requests an OpenAI model (``text-embedding-3-small`` by
default) we route embedding requests through :mod:`rag.ask`, which in turn
talks to the OpenAI Embeddings API.  If the API is unavailable – e.g. the
deployment does not have network access or the ``OPENAI_API_KEY`` is missing –
we fall back to a lightweight SentenceTransformer pipeline so the app can still
boot with deterministic vectors.  The fallback is only intended for offline or
test environments and the returned vectors are padded/truncated to match the
OpenAI dimensionality so FAISS/pgvector integrations keep working.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer, models

logger = logging.getLogger(__name__)

try:  # pragma: no cover - rag is optional in isolated tests
    from rag import ask as rag_ask
except Exception as exc:  # pragma: no cover
    rag_ask = None  # type: ignore[assignment]
    _RAG_IMPORT_ERROR = exc
else:
    _RAG_IMPORT_ERROR = None

try:  # pragma: no cover - configuration import may fail in minimal test harnesses
    from nyx.config import get_memory_config
except Exception:  # pragma: no cover
    get_memory_config = None  # type: ignore[assignment]


DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
_OPENAI_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def _resolve_configured_model() -> str:
    """Return the embedding model name requested by configuration/env."""

    env_overrides: Iterable[str] = (
        os.getenv("NYX_MEMORY_EMBEDDING_MODEL", ""),
        os.getenv("MEMORY_EMBEDDING_MODEL", ""),
    )
    for candidate in env_overrides:
        if candidate:
            return candidate

    if callable(get_memory_config):  # pragma: no branch - small helper
        try:
            memory_cfg = get_memory_config() or {}
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Failed to load Nyx memory config: %s", exc)
        else:
            if isinstance(memory_cfg, dict):
                value = memory_cfg.get("embedding_model")
                if isinstance(value, str) and value.strip():
                    return value

                nested = memory_cfg.get("embedding")
                if isinstance(nested, dict):
                    value = nested.get("model")
                    if isinstance(value, str) and value.strip():
                        return value

    return DEFAULT_OPENAI_MODEL


_CONFIGURED_MODEL = _resolve_configured_model()
_CONFIGURED_MODEL = _CONFIGURED_MODEL.strip() or DEFAULT_OPENAI_MODEL

_USING_OPENAI = _CONFIGURED_MODEL in _OPENAI_DIMENSIONS
_TARGET_DIMENSION = _OPENAI_DIMENSIONS.get(_CONFIGURED_MODEL, 1536)

_OFFLINE_MODEL: Optional[SentenceTransformer] = None
_OFFLINE_DIMENSION: Optional[int] = None
_OPENAI_DISABLED = False


def _load_embedding_model(model_name: str = DEFAULT_OPENAI_MODEL) -> SentenceTransformer:
    """Return a SentenceTransformer with a deterministic offline fallback."""

    local_root = Path(os.getenv("HF_MODEL_DIR", "/app/models/embeddings"))
    local_path = local_root / model_name

    if local_path.exists():
        logger.info("Loading sentence-transformer from %s", local_path)
        return SentenceTransformer(str(local_path))

    try:
        logger.info("Attempting download of %s from Hugging Face hub…", model_name)
        return SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - hub unreachable / 429 / etc.
        logger.warning("HF load failed (%s). Using offline stub.", exc)

    bert = models.Transformer("distilbert-base-uncased", max_seq_length=256)
    pool = models.Pooling(bert.get_word_embedding_dimension(), pooling_mode="mean")
    return SentenceTransformer(modules=[bert, pool])


def _get_offline_model() -> SentenceTransformer:
    """Initialise the offline embedding pipeline lazily."""

    global _OFFLINE_MODEL, _OFFLINE_DIMENSION  # pylint: disable=global-statement

    if _OFFLINE_MODEL is None:
        model_name = _CONFIGURED_MODEL if not _USING_OPENAI else DEFAULT_OPENAI_MODEL
        _OFFLINE_MODEL = _load_embedding_model(model_name)
        _OFFLINE_DIMENSION = _OFFLINE_MODEL.get_sentence_embedding_dimension()

        if _OFFLINE_DIMENSION != _TARGET_DIMENSION:
            logger.info(
                "Offline embedding dimension (%s) differs from target %s – "
                "vectors will be padded/truncated for consistency.",
                _OFFLINE_DIMENSION,
                _TARGET_DIMENSION,
            )

    return _OFFLINE_MODEL


def _coerce_dimension(vector: Sequence[float]) -> np.ndarray:
    """Pad or truncate *vector* so it matches the target dimension."""

    data = np.asarray(vector, dtype="float32").flatten()
    if data.size == _TARGET_DIMENSION:
        return data

    coerced = np.zeros(_TARGET_DIMENSION, dtype="float32")
    limit = min(data.size, _TARGET_DIMENSION)
    if limit:
        coerced[:limit] = data[:limit]
    return coerced


async def _embed_via_openai(texts: Sequence[str]) -> np.ndarray:
    """Return embeddings via OpenAI. Raises on failure so callers can fallback."""

    global _OPENAI_DISABLED  # pylint: disable=global-statement

    if _OPENAI_DISABLED:
        raise RuntimeError("OpenAI embeddings disabled after previous failure")

    if rag_ask is None:
        _OPENAI_DISABLED = True
        reason = _RAG_IMPORT_ERROR or RuntimeError("rag.ask unavailable")
        raise RuntimeError("rag.ask is unavailable; cannot reach OpenAI") from reason

    vectors: List[np.ndarray] = []

    for text in texts:
        try:
            response = await rag_ask(
                text,
                mode="embedding",
                model=_CONFIGURED_MODEL or DEFAULT_OPENAI_MODEL,
                metadata={
                    "component": "nyx.core.memory.embeddings",
                    "batch_size": len(texts),
                },
            )
        except Exception as exc:  # pragma: no cover - network failure
            _OPENAI_DISABLED = True
            raise RuntimeError(f"OpenAI embedding request failed: {exc}") from exc

        vector: Iterable[float]
        if isinstance(response, dict):
            vector = response.get("embedding") or []
        else:
            vector = []

        values = [float(v) for v in vector]
        if not values:
            _OPENAI_DISABLED = True
            raise RuntimeError("OpenAI embedding response missing vector payload")

        vectors.append(_coerce_dimension(values))

    if not vectors:
        raise RuntimeError("No embeddings generated via OpenAI")

    return np.vstack(vectors).astype("float32")


async def _embed_via_offline(texts: Sequence[str]) -> np.ndarray:
    """Return embeddings via the local SentenceTransformer fallback."""

    model = _get_offline_model()
    raw = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    arr = np.asarray(raw, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] != _TARGET_DIMENSION:
        arr = np.vstack(_coerce_dimension(row) for row in arr)

    return arr


async def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Generate embeddings for *texts*, falling back to offline mode if needed."""

    if not texts:
        return np.zeros((0, _TARGET_DIMENSION), dtype="float32")

    if _USING_OPENAI:
        try:
            return await _embed_via_openai(texts)
        except Exception as exc:
            logger.warning(
                "OpenAI embedding unavailable (model=%s): %s; falling back to offline pipeline.",
                _CONFIGURED_MODEL,
                exc,
            )

    return await _embed_via_offline(texts)


async def embed_text(text: str) -> np.ndarray:
    """Convenience wrapper that embeds a single string."""

    vectors = await embed_texts([text])
    return vectors[0]


def get_embedding_dimension() -> int:
    """Return the dimension vectors generated by :func:`embed_texts` will have."""

    return _TARGET_DIMENSION


def using_openai_backend() -> bool:
    """Return ``True`` if the OpenAI pathway was selected by configuration."""

    return _USING_OPENAI and not _OPENAI_DISABLED


__all__ = [
    "DEFAULT_OPENAI_MODEL",
    "embed_text",
    "embed_texts",
    "get_embedding_dimension",
    "using_openai_backend",
]

