"""Helpers for interacting with hosted vector stores via the Agents backend."""

from __future__ import annotations

import io
import json
import logging
import os
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence

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


def _agents_disabled() -> bool:
    return _normalize_bool(os.getenv("DISABLE_AGENTS_RAG")) or _normalize_bool(
        os.getenv("DISABLE_AGENTS_VECTOR_STORE")
    )


def legacy_vector_store_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    if _normalize_bool(os.getenv("ENABLE_LEGACY_VECTOR_STORE")):
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


def _coerce_str_sequence(value: Any) -> List[str]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            if isinstance(item, str):
                candidate = item.strip()
                if candidate:
                    parts.append(candidate)
    else:
        parts = []

    return [part for part in parts if part]


def get_hosted_vector_store_ids(config: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return configured hosted vector store identifiers."""

    candidates: List[str] = []

    for env_key in (
        "OPENAI_VECTOR_STORE_NAME",
        "AGENTS_VECTOR_STORE_IDS",
        "HOSTED_VECTOR_STORE_IDS",
        "VECTOR_STORE_IDS",
    ):
        env_value = os.getenv(env_key)
        if env_value:
            candidates.extend(_coerce_str_sequence(env_value))

    single_env = os.getenv("AGENTS_VECTOR_STORE_ID") or os.getenv("HOSTED_VECTOR_STORE_ID")
    if single_env:
        candidates.extend(_coerce_str_sequence(single_env))

    if isinstance(config, dict):
        vector_section = config.get("vector_store") or {}
        for key in ("hosted_vector_store_ids", "vector_store_ids", "hosted_ids"):
            value = vector_section.get(key)
            candidates.extend(_coerce_str_sequence(value))

    # Preserve order but drop duplicates while keeping the first occurrence.
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
    if _agents_disabled():
        return False

    if legacy_vector_store_enabled(config=config):
        return False

    ids = list(configured_ids or get_hosted_vector_store_ids(config))
    if not ids:
        return False

    try:
        import agents  # noqa: F401 - ensure package is available
    except Exception:
        logger.debug("Agents package unavailable; hosted vector store disabled", exc_info=True)
        return False

    return True


def _build_filters(attributes: Optional[Dict[str, Any]]) -> Any:
    if not attributes:
        return None

    try:
        from openai.types.responses.file_search_tool import ComparisonFilter, CompoundFilter
    except Exception as exc:  # pragma: no cover - import errors should not propagate
        logger.debug("Unable to import OpenAI filter helpers: %s", exc)
        return None

    filters: List[Any] = []
    for key, value in attributes.items():
        if value is None:
            continue

        if isinstance(value, bool):
            coerced = value
        elif isinstance(value, (int, float)):
            coerced = float(value)
        else:
            coerced = str(value)

        filters.append(ComparisonFilter(key=str(key), type="eq", value=coerced))

    if not filters:
        return None

    if len(filters) == 1:
        return filters[0]

    return CompoundFilter(type="and", filters=filters)


def _decode_metadata(attributes: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {k: v for k, v in attributes.items() if k != "metadata_json"}
    payload = attributes.get("metadata_json")
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            metadata.update(decoded)
    return metadata


def _normalise_search_result(result: Any) -> Dict[str, Any]:
    attributes = dict(getattr(result, "attributes", {}) or {})
    metadata = _decode_metadata(attributes)

    score = getattr(result, "score", None)
    try:
        numeric_score = float(score) if score is not None else None
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        numeric_score = None

    text = getattr(result, "text", None)
    if text is None:
        text = ""

    identifier = metadata.get("memory_id") or getattr(result, "file_id", None)
    if not identifier:
        identifier = metadata.get("filename") or str(uuid.uuid4())

    return {
        "id": identifier,
        "score": numeric_score,
        "text": text,
        "metadata": metadata,
        "attributes": attributes,
        "file_id": getattr(result, "file_id", None),
    }


async def search_hosted_vector_store(
    query: str,
    *,
    vector_store_ids: Optional[Sequence[str]] = None,
    max_results: int = 5,
    attributes: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Search the configured hosted vector store via the Agents runner."""

    ids = list(vector_store_ids or get_hosted_vector_store_ids())
    if not ids:
        logger.debug("Hosted vector search skipped: no vector store IDs configured")
        return []

    if not hosted_vector_store_enabled(ids):
        return []

    try:
        from agents import Agent, FileSearchTool, Runner, RunConfig
        from agents.items import ToolCallItem
        from openai.types.responses import ResponseFileSearchToolCall
    except Exception as exc:  # pragma: no cover - dependency errors
        logger.warning("Agents stack unavailable for hosted vector search: %s", exc)
        return []

    tool = FileSearchTool(
        vector_store_ids=list(ids),
        max_num_results=max(1, int(max_results or 5)),
        include_search_results=True,
        filters=_build_filters(attributes),
    )

    agent = Agent(
        name="MemoryVectorSearch",
        instructions=(
            "Use the file_search tool with the supplied query and then stop. "
            "Return immediately after the tool call."
        ),
        tools=[tool],
    )

    run_kwargs: Dict[str, Any] = {"max_turns": 1}
    if metadata:
        run_kwargs["run_config"] = RunConfig(trace_metadata=dict(metadata))

    try:
        run_result = await Runner.run(agent, query, **run_kwargs)
    except Exception as exc:  # pragma: no cover - network/service failure
        logger.warning("Hosted vector search failed via Agents runner: %s", exc)
        return []

    results: List[Dict[str, Any]] = []

    for item in getattr(run_result, "new_items", []):
        raw = getattr(item, "raw_item", None)
        if isinstance(item, ToolCallItem) and isinstance(raw, ResponseFileSearchToolCall):
            for entry in raw.results or []:
                results.append(_normalise_search_result(entry))

    return results


def _prepare_attributes(metadata: Dict[str, Any]) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {}

    preferred_order: Iterable[str] = (
        "memory_id",
        "user_id",
        "conversation_id",
        "entity_type",
        "collection",
        "timestamp",
    )

    def _coerce_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value)
        if len(text) > 512:
            text = text[:512]
        return text

    # Promote preferred keys first.
    for key in preferred_order:
        if key in metadata and len(attributes) < 15:
            coerced = _coerce_value(metadata[key])
            if coerced is not None:
                attributes[key] = coerced

    # Fill remaining slots with other metadata entries.
    for key, value in metadata.items():
        if len(attributes) >= 15:
            break
        if key in attributes or key in preferred_order or key in {"content", "memory_text"}:
            continue
        coerced = _coerce_value(value)
        if coerced is not None:
            attributes[str(key)] = coerced

    if len(attributes) < 16:
        try:
            payload = json.dumps(metadata, default=str)
        except Exception:  # pragma: no cover - serialization guard
            payload = None
        if payload:
            if len(payload) > 512:
                payload = payload[:512]
            attributes["metadata_json"] = payload

    return attributes


async def upsert_hosted_vector_documents(
    documents: Sequence[Dict[str, Any]],
    *,
    vector_store_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Insert documents into the hosted vector store via the OpenAI Vector Store API."""

    if not documents:
        return []

    vector_store_id = vector_store_id or next(iter(get_hosted_vector_store_ids()), None)
    if not vector_store_id:
        logger.debug("Hosted vector upsert skipped: no vector store ID configured")
        return []

    if not hosted_vector_store_enabled([vector_store_id]):
        return []

    try:
        from openai import AsyncOpenAI
    except Exception as exc:  # pragma: no cover - dependency unavailable
        logger.warning("OpenAI client unavailable for hosted vector upsert: %s", exc)
        return []

    try:
        client = AsyncOpenAI()
    except Exception as exc:  # pragma: no cover - API misconfiguration
        logger.warning("Failed to initialise OpenAI client for vector upsert: %s", exc)
        return []

    stored_ids: List[str] = []

    for document in documents:
        text = document.get("text") or ""
        if not isinstance(text, str):
            text = str(text)

        doc_metadata = dict(document.get("metadata") or {})
        if metadata:
            doc_metadata.update(metadata)

        memory_id = document.get("id") or doc_metadata.get("memory_id")
        if not isinstance(memory_id, str) or not memory_id:
            memory_id = str(uuid.uuid4())
        doc_metadata.setdefault("memory_id", memory_id)

        # Ensure textual content is accessible when retrieving results.
        doc_metadata.setdefault("content", text)

        filename = document.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            filename = f"memory-{memory_id}.txt"

        buffer = io.BytesIO(text.encode("utf-8"))
        buffer.seek(0)

        try:
            uploaded_file = await client.files.create(
                file=(filename, buffer),
                purpose="assistants",
            )

            await client.vector_stores.files.create_and_poll(
                vector_store_id=vector_store_id,
                file_id=uploaded_file.id,
                attributes=_prepare_attributes(doc_metadata),
            )
            stored_ids.append(memory_id)
        except Exception as exc:  # pragma: no cover - API failure path
            logger.warning("Failed to upsert hosted memory %s: %s", memory_id, exc)

    return stored_ids

