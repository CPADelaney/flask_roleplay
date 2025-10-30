"""Minimal asynchronous helpers for embeddings and retrieval via Agents."""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

from openai import AsyncOpenAI

try:  # pragma: no cover - defensive import guard
    from agents import setup as agents_setup  # type: ignore
except Exception as exc:  # pragma: no cover
    agents_setup = None  # type: ignore[assignment]
    _AGENTS_IMPORT_ERROR = exc
else:
    _AGENTS_IMPORT_ERROR = None

from .vector_store import get_hosted_vector_store_ids, hosted_vector_store_enabled

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


async def ask(
    prompt: str,
    *,
    mode: str = "retrieval",
    metadata: Optional[Dict[str, Any]] = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: Optional[int] = None,
    limit: Optional[int] = None,
    vector_store_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Execute an embedding or retrieval call using the Agents tooling."""

    metadata = dict(metadata or {})
    mode_normalised = (mode or "retrieval").strip().lower()

    if mode_normalised == "embedding":
        embedding = await _generate_embedding(prompt, model=model, dimensions=dimensions, metadata=metadata)
        return {"embedding": embedding, "provider": "openai", "metadata": metadata}

    documents = await _retrieve_documents(
        prompt,
        metadata=metadata,
        limit=limit,
        vector_store_ids=vector_store_ids,
    )
    return {"documents": documents, "provider": "agents", "metadata": metadata}


async def _generate_embedding(
    text: str,
    *,
    model: str,
    dimensions: Optional[int],
    metadata: Dict[str, Any],
) -> List[float]:
    client = AsyncOpenAI()

    try:
        response = await client.embeddings.create(model=model or DEFAULT_EMBEDDING_MODEL, input=[text])
    except Exception as exc:  # pragma: no cover - network failure
        logger.error("Embedding request failed for model %s: %s", model, exc)
        raise

    data = getattr(response, "data", None) or []
    if not data:
        raise RuntimeError("Embedding response did not include any vectors")

    vector = list(getattr(data[0], "embedding", []))
    if not vector:
        raise RuntimeError("Embedding payload missing from OpenAI response")

    coerced = [float(value) for value in vector]
    if isinstance(dimensions, int) and dimensions > 0:
        coerced = _coerce_dimensions(coerced, target=dimensions)

    logger.debug(
        "Generated embedding via OpenAI model=%s dims=%s metadata_keys=%s",
        model,
        len(coerced),
        sorted(metadata.keys()),
    )
    return coerced


async def _retrieve_documents(
    query: str,
    *,
    metadata: Dict[str, Any],
    limit: Optional[int],
    vector_store_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    ids = list(vector_store_ids or get_hosted_vector_store_ids())

    if not ids or not hosted_vector_store_enabled(ids):
        logger.debug("Hosted vector retrieval skipped: no configured vector stores")
        return []

    if agents_setup is None:
        raise RuntimeError(
            "Agents setup helpers unavailable: %s" % (_AGENTS_IMPORT_ERROR or "unknown error")
        )

    try:
        raw_results = await agents_setup.run_file_search_tool(
            query,
            vector_store_ids=ids,
            limit=limit or 5,
            metadata=metadata,
        )
    except Exception as exc:  # pragma: no cover - service failure
        logger.warning("Agents FileSearchTool invocation failed: %s", exc)
        raise

    documents: List[Dict[str, Any]] = []
    for item in raw_results:
        documents.append(_normalise_result(item, metadata))

    logger.debug(
        "Retrieved %s documents via Agents file search", len(documents)
    )
    return documents


def _normalise_result(result: Any, request_metadata: Dict[str, Any]) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {}
    text = ""
    score: Optional[float] = None
    file_id: Optional[str] = None

    if hasattr(result, "attributes"):
        attributes = dict(result.attributes or {})
    elif isinstance(result, dict):
        attributes = dict(result.get("attributes") or result.get("metadata") or {})

    if hasattr(result, "text"):
        text = getattr(result, "text") or ""
    elif isinstance(result, dict):
        text = str(result.get("text", ""))

    if hasattr(result, "score"):
        raw_score = getattr(result, "score")
    else:
        raw_score = result.get("score") if isinstance(result, dict) else None
    try:
        score = float(raw_score) if raw_score is not None else None
    except (TypeError, ValueError):
        score = None

    if hasattr(result, "file_id"):
        file_id = getattr(result, "file_id")
    elif isinstance(result, dict):
        file_id = result.get("file_id")

    metadata = dict(attributes)
    if isinstance(result, dict):
        metadata.update(result.get("metadata") or {})

    if request_metadata:
        metadata.setdefault("request_metadata", request_metadata)

    identifier = metadata.get("memory_id") or file_id or metadata.get("id")
    if not isinstance(identifier, str) or not identifier:
        identifier = _fallback_identifier(metadata)

    payload = {
        "id": identifier,
        "text": text,
        "score": score,
        "metadata": metadata,
    }
    return payload


def _coerce_dimensions(values: List[float], *, target: int) -> List[float]:
    if len(values) == target:
        return values
    if len(values) > target:
        return values[:target]
    padded = list(values)
    padded.extend(0.0 for _ in range(target - len(values)))
    return padded


def _fallback_identifier(metadata: Dict[str, Any]) -> str:
    candidates: Iterable[str] = (
        metadata.get("file_id"),
        metadata.get("id"),
        metadata.get("filename"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return "unknown-document"


__all__ = ["ask"]
