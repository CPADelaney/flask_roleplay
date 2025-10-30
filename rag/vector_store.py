"""Helpers for interacting with hosted vector stores via the Agents backend."""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:  # pragma: no cover - optional during tests
    from agents import setup as agents_setup  # type: ignore
except Exception as exc:  # pragma: no cover
    agents_setup = None  # type: ignore[assignment]
    _AGENTS_IMPORT_ERROR = exc
else:
    _AGENTS_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

__all__ = [
    "get_hosted_vector_store_ids",
    "hosted_vector_store_enabled",
    "legacy_vector_store_enabled",
    "search_hosted_vector_store",
    "upsert_hosted_vector_documents",
]


def _normalize_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def legacy_vector_store_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    if _normalize_bool(os.getenv("ALLOW_LEGACY_EMBEDDINGS")):
        return True

    vector_config = {}
    if isinstance(config, dict):
        vector_config = config.get("vector_store") or {}
    flag = vector_config.get("use_legacy_vector_store")
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, str):
        return _normalize_bool(flag)
    return False


def get_hosted_vector_store_ids(config: Optional[Dict[str, Any]] = None) -> List[str]:
    candidates: List[str] = []

    for env_key in (
        "OPENAI_VECTOR_STORE_IDS",
        "AGENTS_VECTOR_STORE_IDS",
        "HOSTED_VECTOR_STORE_IDS",
    ):
        value = os.getenv(env_key)
        if value:
            candidates.extend(_split_ids(value))

    single = os.getenv("OPENAI_VECTOR_STORE_ID") or os.getenv("AGENTS_VECTOR_STORE_ID")
    if single:
        candidates.extend(_split_ids(single))

    if isinstance(config, dict):
        section = config.get("vector_store") or {}
        for key in ("hosted_vector_store_ids", "vector_store_ids", "hosted_ids"):
            value = section.get(key)
            candidates.extend(_split_ids(value))

    seen: set[str] = set()
    ordered: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return ordered


def hosted_vector_store_enabled(
    configured_ids: Optional[Sequence[str]] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    if _normalize_bool(os.getenv("DISABLE_AGENTS_RAG")):
        return False
    if _normalize_bool(os.getenv("DISABLE_AGENTS_VECTOR_STORE")):
        return False
    if legacy_vector_store_enabled(config=config):
        return False
    ids = list(configured_ids or get_hosted_vector_store_ids(config))
    if not ids:
        return False
    if agents_setup is None:
        logger.debug("Agents setup unavailable for hosted vector store: %s", _AGENTS_IMPORT_ERROR)
        return False
    return True


async def search_hosted_vector_store(
    query: str,
    *,
    vector_store_ids: Optional[Sequence[str]] = None,
    max_results: int = 5,
    attributes: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    ids = list(vector_store_ids or get_hosted_vector_store_ids())
    if not hosted_vector_store_enabled(ids):
        return []
    if agents_setup is None:
        raise RuntimeError("Agents setup helpers unavailable for vector search")

    search_metadata = dict(metadata or {})
    if attributes:
        search_metadata.setdefault("filters", attributes)

    results = await agents_setup.run_file_search_tool(
        query,
        vector_store_ids=ids,
        limit=max(1, int(max_results or 5)),
        metadata=search_metadata,
    )

    normalised: List[Dict[str, Any]] = []
    for result in results:
        normalised.append(_normalise_search_result(result))
    return normalised


async def upsert_hosted_vector_documents(
    documents: Sequence[Dict[str, Any]],
    *,
    vector_store_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[str]:
    if not documents:
        return []
    ids = list(get_hosted_vector_store_ids())
    target = vector_store_id or (ids[0] if ids else None)
    if not target:
        logger.debug("Hosted vector upsert skipped: no vector store IDs configured")
        return []
    if not hosted_vector_store_enabled([target]):
        return []
    if agents_setup is None:
        raise RuntimeError("Agents setup helpers unavailable for vector upsert")

    merged: List[Dict[str, Any]] = []

    for document in documents:
        text = document.get("text") or ""
        if not isinstance(text, str):
            text = str(text)
        doc_metadata = dict(metadata or {})
        doc_metadata.update(document.get("metadata") or {})
        memory_id = document.get("id") or doc_metadata.get("memory_id") or str(uuid.uuid4())
        doc_metadata["memory_id"] = str(memory_id)
        merged.append(
            {
                "id": str(memory_id),
                "text": text,
                "metadata": doc_metadata,
                "filename": document.get("filename"),
            }
        )

    return await agents_setup.upload_text_documents(target, merged)


def _split_ids(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        parts = [str(item).strip() for item in value if str(item).strip()]
    else:
        parts = []
    return [part for part in parts if part]


def _normalise_search_result(result: Any) -> Dict[str, Any]:
    attributes = {}
    if hasattr(result, "attributes"):
        attributes = dict(result.attributes or {})
    elif isinstance(result, dict):
        attributes = dict(result.get("attributes") or result.get("metadata") or {})

    metadata = dict(attributes)
    if isinstance(result, dict):
        metadata.update(result.get("metadata") or {})

    score = None
    candidate = getattr(result, "score", None) if hasattr(result, "score") else metadata.get("score")
    try:
        score = float(candidate) if candidate is not None else None
    except (TypeError, ValueError):
        score = None

    text = getattr(result, "text", None) if hasattr(result, "text") else metadata.get("content")
    if text is None:
        text = ""

    identifier = metadata.get("memory_id") or getattr(result, "file_id", None)
    if not isinstance(identifier, str) or not identifier:
        identifier = metadata.get("id") or str(uuid.uuid4())

    return {
        "id": identifier,
        "score": score,
        "text": text,
        "metadata": metadata,
    }
