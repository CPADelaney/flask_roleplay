"""Read-only access helpers for projection views."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Sequence

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel, Field

from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)


class ReadRequest(BaseModel):
    """Schema for the ``db.read`` tool."""

    view: Literal["scene_context", "entity_cards", "recent_chunks"]
    user_id: int
    conversation_id: int
    limit: Optional[int] = Field(default=None, ge=1, le=200)
    entity_types: Optional[List[str]] = None
    query_text: Optional[str] = None
    embedding: Optional[List[float]] = None


async def _fetch(query: str, params: Sequence[Any]) -> List[Dict[str, Any]]:
    async with get_db_connection_context() as conn:
        rows = await conn.fetch(query, *params)
    return [dict(row) for row in rows]


async def read_scene_context(
    user_id: int,
    conversation_id: int,
    limit: Optional[int] = 1,
) -> List[Dict[str, Any]]:
    params: List[Any] = [user_id, conversation_id]
    query = (
        "SELECT user_id, conversation_id, scene_context "
        "FROM public.v_scene_context "
        "WHERE user_id = $1 AND conversation_id = $2"
    )
    if limit is not None:
        clamped_limit = max(1, min(limit, 200))
        params.append(clamped_limit)
        query += " LIMIT $3"
    return await _fetch(query, params)


async def _read_entity_cards_with_embedding(
    user_id: int,
    conversation_id: int,
    embedding: Sequence[float],
    query_text: Optional[str],
    entity_types: Sequence[str],
    limit: int,
) -> List[Dict[str, Any]]:
    query = """
        SELECT
            entity_type,
            entity_id,
            user_id,
            conversation_id,
            card,
            updated_at,
            CASE
                WHEN embedding IS NULL THEN NULL
                ELSE 1 - (embedding <=> $1::vector)
            END AS vector_score,
            CASE
                WHEN $2::text IS NULL OR $2 = '' THEN 0
                ELSE ts_rank_cd(search_vector, websearch_to_tsquery('english', $2))
            END AS text_score
        FROM public.v_entity_cards
        WHERE user_id = $3
          AND conversation_id = $4
          AND entity_type = ANY($5::text[])
        ORDER BY
            vector_score DESC NULLS LAST,
            text_score DESC,
            updated_at DESC NULLS LAST
        LIMIT $6
    """
    params: List[Any] = [list(embedding), query_text or None, user_id, conversation_id, list(entity_types), limit]
    return await _fetch(query, params)


async def _read_entity_cards_simple(
    user_id: int,
    conversation_id: int,
    query_text: Optional[str],
    entity_types: Sequence[str],
    limit: int,
) -> List[Dict[str, Any]]:
    query = """
        SELECT
            entity_type,
            entity_id,
            user_id,
            conversation_id,
            card,
            updated_at,
            NULL::float8 AS vector_score,
            CASE
                WHEN $3::text IS NULL OR $3 = '' THEN 0
                ELSE ts_rank_cd(search_vector, websearch_to_tsquery('english', $3))
            END AS text_score
        FROM public.v_entity_cards
        WHERE user_id = $1
          AND conversation_id = $2
          AND entity_type = ANY($4::text[])
        ORDER BY
            text_score DESC,
            updated_at DESC NULLS LAST
        LIMIT $5
    """
    params: List[Any] = [user_id, conversation_id, query_text or None, list(entity_types), limit]
    return await _fetch(query, params)


async def read_entity_cards(
    user_id: int,
    conversation_id: int,
    *,
    embedding: Optional[Sequence[float]] = None,
    query_text: Optional[str] = None,
    entity_types: Optional[Sequence[str]] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    types = list(entity_types or ["npc", "location", "memory"])
    limit = max(1, min(limit, 200))
    if embedding is not None:
        return await _read_entity_cards_with_embedding(user_id, conversation_id, embedding, query_text, types, limit)
    return await _read_entity_cards_simple(user_id, conversation_id, query_text, types, limit)


async def read_recent_chunks(
    user_id: int,
    conversation_id: int,
    *,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    limit = max(1, min(limit, 200))
    query = """
        SELECT chunk_id, chunk, occurred_at
        FROM public.v_recent_chunks
        WHERE user_id = $1
          AND conversation_id = $2
        ORDER BY occurred_at DESC NULLS LAST
        LIMIT $3
    """
    params: List[Any] = [user_id, conversation_id, limit]
    return await _fetch(query, params)


async def read_rows(view: str, **kwargs: Any) -> List[Dict[str, Any]]:
    if view == "scene_context":
        return await read_scene_context(kwargs["user_id"], kwargs["conversation_id"], kwargs.get("limit"))
    if view == "entity_cards":
        return await read_entity_cards(
            kwargs["user_id"],
            kwargs["conversation_id"],
            embedding=kwargs.get("embedding"),
            query_text=kwargs.get("query_text"),
            entity_types=kwargs.get("entity_types"),
            limit=kwargs.get("limit", 10),
        )
    if view == "recent_chunks":
        return await read_recent_chunks(
            kwargs["user_id"],
            kwargs["conversation_id"],
            limit=kwargs.get("limit", 5),
        )
    raise ValueError(f"Unsupported view '{view}'")


@function_tool(name_override="db.read")
async def read_tool(ctx: RunContextWrapper, request: ReadRequest) -> Dict[str, Any]:
    del ctx  # The DB layer does its own tracing/metrics.
    rows = await read_rows(
        request.view,
        user_id=request.user_id,
        conversation_id=request.conversation_id,
        limit=request.limit,
        entity_types=request.entity_types,
        query_text=request.query_text,
        embedding=request.embedding,
    )
    return {"rows": rows}

