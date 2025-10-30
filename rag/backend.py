"""Backend selection helpers for Retrieval-Augmented Generation (RAG)."""

from __future__ import annotations

import enum
import logging
import os
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

LegacyCallback = Callable[[], Awaitable[Any]]


class BackendPreference(enum.Enum):
    """Enumerates the supported RAG backends."""

    AUTO = "auto"
    AGENTS = "agents"
    LEGACY = "legacy"


def get_configured_backend() -> BackendPreference:
    """Return the configured backend preference."""

    value = os.getenv("RAG_BACKEND")
    if value:
        normalized = value.strip().lower()
        if normalized in {"agent", "agents"}:
            return BackendPreference.AGENTS
        if normalized in {"legacy", "fallback"}:
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
) -> Dict[str, Any]:
    """Execute a retrieval/embedding request via the configured backend."""

    meta = dict(metadata or {})

    if mode not in {"retrieval", "embedding"}:
        logger.debug("Unsupported mode %s; defaulting to retrieval", mode)
        mode = "retrieval"

    preference = _resolve_backend_preference(backend)

    if preference is not BackendPreference.LEGACY:
        agents_result = await _call_agents_backend(
            prompt,
            mode=mode,
            metadata=meta,
            model=model,
            dimensions=dimensions,
            limit=limit,
        )
        if agents_result is not None:
            return agents_result
        if preference is BackendPreference.AGENTS:
            logger.warning(
                "Agents backend requested but unavailable; falling back to legacy implementation",
            )

    return await _legacy_backend(
        prompt,
        mode=mode,
        metadata=meta,
        model=model,
        dimensions=dimensions,
        limit=limit,
        legacy_fallback=legacy_fallback,
    )


def _resolve_backend_preference(value: Optional[str]) -> BackendPreference:
    if value is None:
        return get_configured_backend()
    normalized = value.strip().lower()
    if normalized in {"agent", "agents"}:
        return BackendPreference.AGENTS
    if normalized in {"legacy", "fallback"}:
        return BackendPreference.LEGACY
    return BackendPreference.AUTO


async def _call_agents_backend(
    prompt: str,
    *,
    mode: str,
    metadata: Dict[str, Any],
    model: str,
    dimensions: Optional[int],
    limit: Optional[int],
) -> Optional[Dict[str, Any]]:
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
        normalised = _normalise_agent_response(response, mode, metadata)
        _log_agent_invocation(prompt, normalised, mode)
        return normalised
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Could not normalise Agents response; falling back: %s", exc)
        return None


async def _legacy_backend(
    prompt: str,
    *,
    mode: str,
    metadata: Dict[str, Any],
    model: str,
    dimensions: Optional[int],
    limit: Optional[int],
    legacy_fallback: Optional[LegacyCallback],
) -> Dict[str, Any]:
    if mode == "embedding":
        embedding, provider = await _legacy_embedding(prompt, model=model, dimensions=dimensions)
        return {"embedding": embedding, "provider": provider, "metadata": metadata}

    if legacy_fallback is None:
        logger.debug("No legacy fallback supplied for retrieval; returning empty list")
        documents: List[Dict[str, Any]] = []
    else:
        documents = _coerce_documents(await legacy_fallback())

    return {"documents": documents, "provider": "legacy", "metadata": metadata}


def _normalize_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalise_agent_response(response: Any, mode: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    provider = "agents"

    if mode == "embedding":
        if isinstance(response, dict) and "embedding" in response:
            embedding = _coerce_embedding(response["embedding"])
            provider = response.get("provider", provider)
        else:
            embedding = _coerce_embedding(response)
        return {"embedding": embedding, "provider": provider, "metadata": metadata}

    if isinstance(response, dict):
        provider = response.get("provider", provider)
        if "documents" in response:
            documents = _coerce_documents(response["documents"])
        elif "data" in response:
            documents = _coerce_documents(response["data"])
        else:
            documents = _coerce_documents(response)
        merged_metadata = response.get("metadata")
        if isinstance(merged_metadata, dict):
            metadata = {**metadata, **merged_metadata}
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

    if _normalize_bool(os.getenv("ENABLE_LEGACY_EMBEDDINGS")):
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


def _log_agent_invocation(prompt: str, response: Dict[str, Any], mode: str) -> None:
    prompt_text = str(prompt)
    question_preview = prompt_text if len(prompt_text) <= 200 else f"{prompt_text[:197]}..."
    answer_length: int
    if mode == "embedding":
        answer_length = len(response.get("embedding", []))
    else:
        documents = response.get("documents", [])
        answer_length = sum(len(str(doc.get("text") or doc.get("content") or doc)) for doc in documents)

    metadata = response.get("metadata") or {}
    tool_usage = None
    if isinstance(metadata, dict):
        tool_usage = metadata.get("tool_usage")
    if tool_usage is None and isinstance(response, dict):
        tool_usage = response.get("tool_usage")

    logger.info(
        "Agents RAG invocation mode=%s question=%s answer_length=%s tool_usage=%s",
        mode,
        question_preview,
        answer_length,
        tool_usage,
    )


__all__ = ["ask", "get_configured_backend", "BackendPreference"]

