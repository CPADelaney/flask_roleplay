"""Simple backend selector that prefers the Agents FileSearchTool implementation."""
from __future__ import annotations

import enum
import logging
import os
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from .ask import ask as agents_ask

logger = logging.getLogger(__name__)

LegacyCallback = Callable[[], Awaitable[Any]]


class BackendPreference(enum.Enum):
    AUTO = "auto"
    AGENTS = "agents"
    LEGACY = "legacy"


def get_configured_backend() -> BackendPreference:
    value = os.getenv("RAG_BACKEND", "").strip().lower()
    if value in {"agent", "agents"}:
        return BackendPreference.AGENTS
    if value in {"legacy", "fallback"}:
        return BackendPreference.LEGACY
    if _normalize_bool(os.getenv("DISABLE_AGENTS_RAG")):
        return BackendPreference.LEGACY
    return BackendPreference.AUTO


async def ask(
    prompt: str,
    *,
    mode: str = "retrieval",
    metadata: Optional[Dict[str, Any]] = None,
    model: str = "text-embedding-3-small",
    dimensions: Optional[int] = None,
    limit: Optional[int] = None,
    legacy_fallback: Optional[LegacyCallback] = None,
    backend: Optional[str] = None,
    vector_store_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Route calls to the preferred backend, falling back to legacy when allowed."""

    metadata = dict(metadata or {})
    preference = _resolve_backend_preference(backend)

    if preference is BackendPreference.LEGACY:
        return await _invoke_legacy(
            prompt,
            mode=mode,
            metadata=metadata,
            legacy_fallback=legacy_fallback,
        )

    try:
        return await agents_ask(
            prompt,
            mode=mode,
            metadata=metadata,
            model=model,
            dimensions=dimensions,
            limit=limit,
            vector_store_ids=list(vector_store_ids) if vector_store_ids is not None else None,
        )
    except Exception as exc:
        logger.warning("Agents backend failed, attempting legacy fallback: %s", exc)
        if preference is BackendPreference.AGENTS:
            raise
        return await _invoke_legacy(
            prompt,
            mode=mode,
            metadata=metadata,
            legacy_fallback=legacy_fallback,
            error=exc,
        )


def _resolve_backend_preference(value: Optional[str]) -> BackendPreference:
    if not value:
        return get_configured_backend()
    value = value.strip().lower()
    if value in {"agent", "agents"}:
        return BackendPreference.AGENTS
    if value in {"legacy", "fallback"}:
        return BackendPreference.LEGACY
    return BackendPreference.AUTO


async def _invoke_legacy(
    prompt: str,
    *,
    mode: str,
    metadata: Dict[str, Any],
    legacy_fallback: Optional[LegacyCallback],
    error: Optional[Exception] = None,
) -> Dict[str, Any]:
    if not _legacy_embeddings_enabled():
        message = (
            "Legacy embedding/retrieval helpers are disabled. Set ENABLE_LEGACY_VECTOR_STORE=1 "
            "to opt back in temporarily."
        )
        if error:
            message = f"{message} Original error: {error}"
        raise RuntimeError(message)

    mode_normalised = (mode or "retrieval").strip().lower()
    if mode_normalised == "embedding":
        raise RuntimeError("Legacy embedding path has been removed; use mode='retrieval' only.")

    documents: List[Dict[str, Any]]
    if legacy_fallback is None:
        documents = []
    else:
        payload = await legacy_fallback()
        documents = _coerce_documents(payload)

    logger.debug("Using legacy retrieval fallback for prompt preview=%s", prompt[:80])
    return {"documents": documents, "provider": "legacy", "metadata": metadata}


def _coerce_documents(payload: Any) -> List[Dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, (list, tuple, set)):
        result: List[Dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                result.append(item)
            else:
                result.append({"value": item})
        return result
    return [{"value": payload}]


def _normalize_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _legacy_embeddings_enabled() -> bool:
    legacy_env = os.getenv("ENABLE_LEGACY_VECTOR_STORE")
    if legacy_env is not None:
        return _normalize_bool(legacy_env)

    fallback_env = os.getenv("ALLOW_LEGACY_EMBEDDINGS")
    if fallback_env is not None:
        return _normalize_bool(fallback_env)

    return False


__all__ = ["ask", "get_configured_backend", "BackendPreference"]
