"""Compatibility layer for retrieval via the Agents backend.

The primary entry point is :func:`ask`, which prefers the Agents
infrastructure when it is available and enabled. When the Agents path is
unavailable, a legacy shim is used instead.  Embedding requests fall back to
local deterministic embeddings (and optionally the legacy OpenAI embeddings
API behind the ``ENABLE_LEGACY_EMBEDDINGS`` feature flag).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

LegacyCallback = Callable[[], Awaitable[Any]]


def _normalize_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _agents_disabled() -> bool:
    return _normalize_bool(os.getenv("DISABLE_AGENTS_RAG"))


def _legacy_openai_enabled() -> bool:
    return _normalize_bool(os.getenv("ENABLE_LEGACY_EMBEDDINGS"))


async def ask(
    prompt: str,
    *,
    mode: str = "retrieval",
    metadata: Optional[Dict[str, Any]] = None,
    model: str = "text-embedding-3-small",
    dimensions: Optional[int] = None,
    limit: Optional[int] = None,
    legacy_fallback: Optional[LegacyCallback] = None,
) -> Dict[str, Any]:
    """Execute a retrieval/embedding request via the Agents backend.

    Parameters
    ----------
    prompt:
        Text payload for the request.
    mode:
        ``"retrieval"`` (default) to obtain contextual documents or
        ``"embedding"`` to request a vector representation.
    metadata:
        Optional metadata that will be forwarded to the backend to aid
        observability/debugging.
    model / dimensions / limit:
        Advisory settings for the backend implementation.  They are ignored
        by the default shim but preserved for compatibility.
    legacy_fallback:
        Async callable to execute when the Agents backend is unavailable.

    Returns
    -------
    Dict[str, Any]
        Normalised response containing either ``"documents"`` or
        ``"embedding"`` depending on ``mode``.
    """

    meta = metadata or {}

    # --- Preferred path: Agents backend -------------------------------------------------
    if mode not in {"retrieval", "embedding"}:
        logger.debug("Unsupported mode %s; falling back to retrieval semantics", mode)
        mode = "retrieval"

    agents_result = await _maybe_call_agents(
        prompt,
        mode=mode,
        metadata=meta,
        model=model,
        dimensions=dimensions,
        limit=limit,
    )
    if agents_result is not None:
        return agents_result

    # --- Legacy shim --------------------------------------------------------------------
    if mode == "embedding":
        embedding, provider = await _legacy_embedding(prompt, model=model, dimensions=dimensions)
        return {"embedding": embedding, "provider": provider, "metadata": meta}

    if legacy_fallback is None:
        logger.debug("No legacy fallback supplied for retrieval; returning empty list")
        documents: List[Dict[str, Any]] = []
    else:
        documents = _coerce_documents(await legacy_fallback())

    return {"documents": documents, "provider": "legacy", "metadata": meta}


async def _maybe_call_agents(
    prompt: str,
    *,
    mode: str,
    metadata: Dict[str, Any],
    model: str,
    dimensions: Optional[int],
    limit: Optional[int],
) -> Optional[Dict[str, Any]]:
    if _agents_disabled():
        return None

    try:
        from agents import ask as agents_ask  # type: ignore
    except Exception:
        return None

    try:
        response = await agents_ask(
            prompt=prompt,
            mode=mode,
            metadata=metadata,
            model=model,
            dimensions=dimensions,
            limit=limit,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Agents ask failed, falling back to legacy shim: %s", exc)
        return None

    try:
        return _normalise_agent_response(response, mode, metadata)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Could not normalise Agents response; falling back: %s", exc)
        return None


def _normalise_agent_response(response: Any, mode: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    provider = "agents"

    if mode == "embedding":
        if isinstance(response, dict) and "embedding" in response:
            embedding = _coerce_embedding(response["embedding"])
            provider = response.get("provider", provider)
        else:
            embedding = _coerce_embedding(response)
        return {"embedding": embedding, "provider": provider, "metadata": metadata}

    # Retrieval path
    if isinstance(response, dict):
        provider = response.get("provider", provider)
        if "documents" in response:
            documents = _coerce_documents(response["documents"])
        elif "data" in response:
            documents = _coerce_documents(response["data"])
        else:
            documents = _coerce_documents(response)
    else:
        documents = _coerce_documents(response)

    return {"documents": documents, "provider": provider, "metadata": metadata}


async def _legacy_embedding(
    prompt: str,
    *,
    model: str,
    dimensions: Optional[int],
) -> tuple[List[float], str]:
    provider = "local"

    if _legacy_openai_enabled():  # pragma: no cover - network / API usage
        try:
            from logic import chatgpt_integration as chatgpt

            client = chatgpt._client_manager.async_client  # type: ignore[attr-defined]
            params: Dict[str, Any] = {
                "model": model,
                "input": prompt.replace("\n", " ").strip() or " ",
                "encoding_format": "float",
            }
            if dimensions:
                params["dimensions"] = dimensions

            response = await client.embeddings.create(**params)
            data = response.data[0].embedding
            embedding = _coerce_embedding(data)
            provider = "legacy-openai"
            return embedding, provider
        except Exception as exc:
            logger.warning("Legacy OpenAI embedding failed, using local fallback: %s", exc)

    # Local deterministic embedding fallback
    from embedding.vector_store import generate_embedding

    embedding = await generate_embedding(prompt)
    return _coerce_embedding(embedding), provider


def _coerce_embedding(value: Any) -> List[float]:
    if isinstance(value, dict) and "embedding" in value:
        value = value["embedding"]

    if isinstance(value, (bytes, bytearray)):
        raise TypeError("Binary embeddings are not supported")

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [float(v) for v in value]

    raise TypeError(f"Unsupported embedding payload: {type(value)!r}")


def _coerce_documents(payload: Any) -> List[Dict[str, Any]]:
    if payload is None:
        return []

    if isinstance(payload, dict):
        iterable: Iterable[Any] = payload.values() if not payload.get("documents") else payload["documents"]
    elif isinstance(payload, (list, tuple, set)):
        iterable = payload
    else:
        # Single item (string/record/etc.)
        iterable = [payload]

    documents: List[Dict[str, Any]] = []
    for item in iterable:
        if item is None:
            continue
        if isinstance(item, dict):
            documents.append(item)
        elif hasattr(item, "_asdict"):
            documents.append(dict(item._asdict()))
        elif hasattr(item, "to_dict") and callable(getattr(item, "to_dict")):
            documents.append(item.to_dict())  # type: ignore[call-arg]
        elif hasattr(item, "model_dump") and callable(getattr(item, "model_dump")):
            documents.append(item.model_dump())  # type: ignore[call-arg]
        else:
            documents.append({"value": item})
    return documents


__all__ = ["ask"]
