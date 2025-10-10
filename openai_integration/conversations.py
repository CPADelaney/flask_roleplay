"""Database helpers for managing OpenAI conversation metadata."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from db.connection import get_db_connection_context

# Columns that are returned to callers. Keeping this centralised makes it
# easier to reuse between ``INSERT`` and ``SELECT`` code paths.
_RETURNING_COLUMNS = (
    "id",
    "user_id",
    "conversation_id",
    "openai_assistant_id",
    "openai_thread_id",
    "openai_run_id",
    "openai_response_id",
    "status",
    "last_error",
    "metadata",
    "created_at",
    "updated_at",
)


async def _upsert_conversation(
    conn,
    *,
    user_id: int,
    conversation_id: int,
    openai_assistant_id: str,
    openai_thread_id: str,
    openai_run_id: Optional[str] = None,
    openai_response_id: Optional[str] = None,
    status: str = "pending",
    last_error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Insert or update a conversation row and return the stored record."""

    metadata = metadata or {}

    record = await conn.fetchrow(
        f"""
        INSERT INTO openai_conversations (
            user_id,
            conversation_id,
            openai_assistant_id,
            openai_thread_id,
            openai_run_id,
            openai_response_id,
            status,
            last_error,
            metadata
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (conversation_id) DO UPDATE
        SET
            user_id = EXCLUDED.user_id,
            openai_assistant_id = EXCLUDED.openai_assistant_id,
            openai_thread_id = EXCLUDED.openai_thread_id,
            openai_run_id = EXCLUDED.openai_run_id,
            openai_response_id = EXCLUDED.openai_response_id,
            status = EXCLUDED.status,
            last_error = EXCLUDED.last_error,
            metadata = COALESCE(openai_conversations.metadata, '{{}}'::jsonb) || EXCLUDED.metadata,
            updated_at = NOW()
        RETURNING {', '.join(_RETURNING_COLUMNS)}
        """,
        user_id,
        conversation_id,
        openai_assistant_id,
        openai_thread_id,
        openai_run_id,
        openai_response_id,
        status,
        last_error,
        metadata,
    )

    return dict(record) if record else None


async def create_conversation(
    *,
    user_id: int,
    conversation_id: int,
    openai_assistant_id: str,
    openai_thread_id: str,
    openai_run_id: Optional[str] = None,
    openai_response_id: Optional[str] = None,
    status: str = "pending",
    last_error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    conn=None,
) -> Optional[Dict[str, Any]]:
    """Create or update an OpenAI conversation and return the stored row."""

    if conn is not None:
        return await _upsert_conversation(
            conn,
            user_id=user_id,
            conversation_id=conversation_id,
            openai_assistant_id=openai_assistant_id,
            openai_thread_id=openai_thread_id,
            openai_run_id=openai_run_id,
            openai_response_id=openai_response_id,
            status=status,
            last_error=last_error,
            metadata=metadata,
        )

    async with get_db_connection_context() as db_conn:
        return await _upsert_conversation(
            db_conn,
            user_id=user_id,
            conversation_id=conversation_id,
            openai_assistant_id=openai_assistant_id,
            openai_thread_id=openai_thread_id,
            openai_run_id=openai_run_id,
            openai_response_id=openai_response_id,
            status=status,
            last_error=last_error,
            metadata=metadata,
        )


async def get_or_create_conversation(
    *,
    user_id: int,
    conversation_id: int,
    openai_assistant_id: str,
    openai_thread_id: str,
    openai_run_id: Optional[str] = None,
    openai_response_id: Optional[str] = None,
    status: str = "pending",
    last_error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    conn=None,
) -> Optional[Dict[str, Any]]:
    """Fetch an existing conversation or create it if absent."""

    async def _select_existing(connection) -> Optional[Mapping[str, Any]]:
        record = await connection.fetchrow(
            f"""
            SELECT {', '.join(_RETURNING_COLUMNS)}
            FROM openai_conversations
            WHERE conversation_id = $1
            """,
            conversation_id,
        )
        return record

    if conn is not None:
        existing = await _select_existing(conn)
        if existing:
            return dict(existing)

        return await _upsert_conversation(
            conn,
            user_id=user_id,
            conversation_id=conversation_id,
            openai_assistant_id=openai_assistant_id,
            openai_thread_id=openai_thread_id,
            openai_run_id=openai_run_id,
            openai_response_id=openai_response_id,
            status=status,
            last_error=last_error,
            metadata=metadata,
        )

    async with get_db_connection_context() as db_conn:
        existing = await _select_existing(db_conn)
        if existing:
            return dict(existing)

        return await _upsert_conversation(
            db_conn,
            user_id=user_id,
            conversation_id=conversation_id,
            openai_assistant_id=openai_assistant_id,
            openai_thread_id=openai_thread_id,
            openai_run_id=openai_run_id,
            openai_response_id=openai_response_id,
            status=status,
            last_error=last_error,
            metadata=metadata,
        )
