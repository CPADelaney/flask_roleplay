"""Minimal helpers for working with the Agents FileSearchTool backend."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from openai import AsyncOpenAI
from openai.types.responses import ResponseFileSearchToolCall

import nyx.gateway.llm_gateway as llm_gateway
from nyx.gateway.llm_gateway import LLMRequest

from . import Agent, FileSearchTool, RunConfig
from .items import ToolCallItem

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FileSearchResult:
    """Container describing a single FileSearchTool match."""

    file_id: str
    text: str
    score: Optional[float]
    attributes: Dict[str, Any]

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self.attributes)


async def ensure_vector_store(
    *,
    client: Optional[AsyncOpenAI] = None,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Return an existing vector store identifier or create one on demand."""

    client = client or AsyncOpenAI()

    if name:
        try:
            listing = await client.beta.vector_stores.list(limit=100)
        except Exception as exc:  # pragma: no cover - network/feature errors
            logger.debug("Vector store listing failed: %s", exc)
            listing = None
        if listing:
            for item in getattr(listing, "data", []):
                if getattr(item, "name", None) == name:
                    return item.id

    try:
        created = await client.beta.vector_stores.create(name=name or "memory-store", metadata=metadata)
    except Exception as exc:  # pragma: no cover - API failure
        logger.error("Unable to create vector store %s: %s", name or "memory-store", exc)
        raise

    return created.id


async def upload_text_documents(
    vector_store_id: str,
    documents: Sequence[Dict[str, Any]],
    *,
    client: Optional[AsyncOpenAI] = None,
    default_filename: str = "memory.txt",
) -> List[str]:
    """Upload textual documents into the specified vector store."""

    if not documents:
        return []

    client = client or AsyncOpenAI()
    stored_ids: List[str] = []

    for document in documents:
        text = document.get("text") or ""
        if not isinstance(text, str):
            text = str(text)
        metadata = {k: v for k, v in (document.get("metadata") or {}).items() if v is not None}
        memory_id = metadata.get("memory_id") or document.get("id")
        if not isinstance(memory_id, str) or not memory_id:
            continue

        filename = document.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            filename = default_filename

        buffer = io.BytesIO(text.encode("utf-8"))
        buffer.seek(0)

        try:
            uploaded = await client.files.create(file=(filename, buffer), purpose="assistants")
            await client.vector_stores.files.create_and_poll(
                vector_store_id=vector_store_id,
                file_id=uploaded.id,
                attributes=_coerce_attributes(metadata, text=text),
            )
            stored_ids.append(memory_id)
        except Exception as exc:  # pragma: no cover - network/service failure
            logger.warning("Failed to upload document %s: %s", memory_id, exc)

    return stored_ids


async def run_file_search_tool(
    query: str,
    *,
    vector_store_ids: Sequence[str],
    limit: int = 5,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[FileSearchResult]:
    """Execute the Agents FileSearchTool and normalise the results."""

    if not vector_store_ids:
        return []

    tool = FileSearchTool(
        vector_store_ids=list(vector_store_ids),
        include_search_results=True,
        max_num_results=max(1, int(limit or 5)),
    )

    agent = Agent(
        name="FileSearchToolRunner",
        instructions="Use the file_search tool with the provided query then stop.",
        tools=[tool],
    )

    run_kwargs: Dict[str, Any] = {"max_turns": 1}
    if metadata:
        run_kwargs["run_config"] = RunConfig(trace_metadata=dict(metadata))

    run = await llm_gateway.execute(
        LLMRequest(
            agent=agent,
            prompt=query,
            runner_kwargs=run_kwargs,
        )
    )
    raw_run = getattr(run, "raw", None)
    results: List[FileSearchResult] = []

    for item in getattr(raw_run, "new_items", []) if raw_run is not None else []:
        raw = getattr(item, "raw_item", None)
        if not isinstance(item, ToolCallItem) or not isinstance(raw, ResponseFileSearchToolCall):
            continue
        for entry in raw.results or []:
            score = getattr(entry, "score", None)
            try:
                score_value: Optional[float] = float(score) if score is not None else None
            except (TypeError, ValueError):
                score_value = None
            attributes = dict(getattr(entry, "attributes", {}) or {})
            result = FileSearchResult(
                file_id=getattr(entry, "file_id", ""),
                text=getattr(entry, "text", "") or "",
                score=score_value,
                attributes=attributes,
            )
            results.append(result)

    return results


def _coerce_attributes(metadata: Dict[str, Any], *, text: str) -> Dict[str, Any]:
    """Convert metadata into the attribute payload accepted by the API."""

    attributes: Dict[str, Any] = {}

    def _coerce_value(value: Any) -> Any:
        if isinstance(value, (bool, int, float)):
            return value
        value_str = str(value)
        return value_str[:512] if len(value_str) > 512 else value_str

    preferred: Iterable[str] = (
        "memory_id",
        "user_id",
        "conversation_id",
        "entity_type",
        "collection",
        "timestamp",
    )

    for key in preferred:
        if key in metadata:
            coerced = _coerce_value(metadata[key])
            attributes[key] = coerced

    for key, value in metadata.items():
        if key in attributes:
            continue
        coerced = _coerce_value(value)
        attributes[key] = coerced

    attributes.setdefault("content", text[:512])
    return attributes


__all__ = [
    "FileSearchResult",
    "ensure_vector_store",
    "run_file_search_tool",
    "upload_text_documents",
]
