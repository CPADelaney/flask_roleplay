"""Database helpers for managing OpenAI conversation metadata."""

from __future__ import annotations

import inspect
from typing import Any, AsyncIterator, Callable, Dict, Mapping, Optional

try:  # pragma: no cover - optional dependency guard
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - openai is optional in some test envs
    AsyncOpenAI = None  # type: ignore[assignment]

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

_SCENE_RETURNING_COLUMNS = (
    "id",
    "conversation_id",
    "scene_number",
    "scene_title",
    "scene_summary",
    "scene_state",
    "active_npc_ids",
    "location_reference",
    "tension_level",
    "tags",
    "metadata",
    "is_active",
    "started_at",
    "ended_at",
    "created_at",
    "updated_at",
)

_SELECT_ACTIVE_SCENE_QUERY = f"""
    SELECT {', '.join(_SCENE_RETURNING_COLUMNS)}
    FROM conversation_scenes
    WHERE conversation_id = $1 AND is_active = TRUE
    ORDER BY scene_number DESC
    LIMIT 1
"""


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


async def get_latest_conversation(
    *,
    conversation_id: int,
    user_id: Optional[int] = None,
    conn=None,
) -> Optional[Dict[str, Any]]:
    """Fetch the most recent OpenAI conversation row for a conversation."""

    async def _get(connection):
        params = [conversation_id]
        query = [
            f"SELECT {', '.join(_RETURNING_COLUMNS)}",
            "FROM openai_conversations",
            "WHERE conversation_id = $1",
        ]

        if user_id is not None:
            query.append("AND user_id = $2")
            params.append(user_id)

        query.append("ORDER BY updated_at DESC LIMIT 1")
        record = await connection.fetchrow("\n".join(query), *params)
        return dict(record) if record else None

    if conn is not None:
        return await _get(conn)

    async with get_db_connection_context() as db_conn:
        return await _get(db_conn)


async def _rotate_scene(
    conn,
    *,
    conversation_id: int,
    new_scene: Mapping[str, Any],
    closing_scene: Optional[Mapping[str, Any]] = None,
):
    closing_scene = closing_scene or {}

    active_scene = await conn.fetchrow(
        _SELECT_ACTIVE_SCENE_QUERY,
        conversation_id,
    )

    if active_scene:
        await conn.execute(
            """
            UPDATE conversation_scenes
            SET
                scene_summary = COALESCE($2, scene_summary),
                scene_state = COALESCE($3::jsonb, scene_state),
                metadata = COALESCE(metadata, '{}'::jsonb) || COALESCE($4::jsonb, '{}'::jsonb),
                is_active = FALSE,
                ended_at = COALESCE(ended_at, NOW()),
                updated_at = NOW()
            WHERE id = $1
            """,
            active_scene["id"],
            closing_scene.get("scene_summary"),
            closing_scene.get("scene_state"),
            closing_scene.get("metadata"),
        )

    new_scene_data: Dict[str, Any] = dict(new_scene)

    explicit_scene_number = new_scene_data.get("scene_number")
    if explicit_scene_number is not None:
        scene_number = explicit_scene_number
    elif active_scene:
        scene_number = (active_scene.get("scene_number") or 0) + 1
    else:
        scene_number = 1

    active_npc_ids = new_scene_data.get("active_npc_ids")
    tags = new_scene_data.get("tags")

    if active_npc_ids is None:
        active_npc_ids_param = []
    elif isinstance(active_npc_ids, (str, bytes)):
        active_npc_ids_param = [active_npc_ids]
    else:
        active_npc_ids_param = list(active_npc_ids)

    if tags is None:
        tags_param = []
    elif isinstance(tags, (str, bytes)):
        tags_param = [tags]
    else:
        tags_param = list(tags)

    tension_level = new_scene_data.get("tension_level")
    metadata = new_scene_data.get("metadata")

    inserted_scene = await conn.fetchrow(
        f"""
        INSERT INTO conversation_scenes (
            conversation_id,
            scene_number,
            scene_title,
            scene_summary,
            scene_state,
            active_npc_ids,
            location_reference,
            tension_level,
            tags,
            metadata,
            is_active
        )
        VALUES (
            $1,
            $2,
            $3,
            $4,
            $5::jsonb,
            $6::int[],
            $7,
            $8,
            $9::text[],
            $10::jsonb,
            TRUE
        )
        RETURNING {', '.join(_SCENE_RETURNING_COLUMNS)}
        """,
        conversation_id,
        scene_number,
        new_scene_data.get("scene_title"),
        new_scene_data.get("scene_summary"),
        new_scene_data.get("scene_state") or {},
        active_npc_ids_param,
        new_scene_data.get("location_reference"),
        tension_level if tension_level is not None else 0,
        tags_param,
        metadata or {},
    )

    return dict(inserted_scene) if inserted_scene else None


async def rotate_conversation_scene(
    *,
    conversation_id: int,
    new_scene: Mapping[str, Any],
    closing_scene: Optional[Mapping[str, Any]] = None,
    conn=None,
):
    if conn is not None:
        return await _rotate_scene(
            conn,
            conversation_id=conversation_id,
            new_scene=new_scene,
            closing_scene=closing_scene,
        )

    async with get_db_connection_context() as db_conn:
        return await _rotate_scene(
            db_conn,
            conversation_id=conversation_id,
            new_scene=new_scene,
            closing_scene=closing_scene,
        )


async def get_active_scene(
    *,
    conversation_id: int,
    conn=None,
):
    async def _get(connection):
        record = await connection.fetchrow(
            _SELECT_ACTIVE_SCENE_QUERY,
            conversation_id,
        )
        return dict(record) if record else None

    if conn is not None:
        return await _get(conn)

    async with get_db_connection_context() as db_conn:
        return await _get(db_conn)


class ConversationStreamError(RuntimeError):
    """Raised when streaming a conversation message fails."""


class ConversationManager:
    """High-level helper that manages OpenAI conversations and streaming."""

    def __init__(
        self,
        *,
        client=None,
        client_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Initialise the manager.

        Parameters
        ----------
        client:
            An optional AsyncOpenAI-compatible client. Mainly used for testing.
        client_factory:
            Callable that returns an AsyncOpenAI client when none is provided.
            Defaults to :class:`openai.AsyncOpenAI` when available.
        """

        self._client = client
        if client_factory is not None:
            self._client_factory = client_factory
        elif AsyncOpenAI is not None:
            self._client_factory = AsyncOpenAI
        else:  # pragma: no cover - only triggered if openai is missing
            self._client_factory = None

    def _get_client(self):
        if self._client is None:
            if self._client_factory is None:  # pragma: no cover
                raise RuntimeError("No OpenAI client factory configured")
            self._client = self._client_factory()
        return self._client

    async def create_conversation(self, **kwargs):
        """Proxy to :func:`create_conversation`."""

        return await create_conversation(**kwargs)

    async def get_or_create_conversation(self, **kwargs):
        """Proxy to :func:`get_or_create_conversation`."""

        return await get_or_create_conversation(**kwargs)

    async def rotate_conversation_scene(self, **kwargs):
        """Proxy to :func:`rotate_conversation_scene`."""

        return await rotate_conversation_scene(**kwargs)

    async def get_active_scene(self, **kwargs):
        """Proxy to :func:`get_active_scene`."""

        return await get_active_scene(**kwargs)

    async def send_message(
        self,
        *,
        model: str,
        input: Any,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream a message through the OpenAI Responses client.

        Yields dictionaries that normalise the most common event types from the
        Responses streaming interface. Callers receive incremental deltas via
        ``response.output_text.delta`` events and the final aggregated response
        once ``response.completed`` fires.
        """

        client = self._get_client()
        stream_ctx = client.responses.stream(model=model, input=input, **kwargs)
        final_payload: Optional[Any] = None

        async with stream_ctx as stream:
            async for event in stream:
                event_type = getattr(event, "type", None)

                if event_type == "response.output_text.delta":
                    yield {
                        "type": event_type,
                        "delta": getattr(event, "delta", None),
                    }
                elif event_type == "response.completed":
                    final_payload = stream.get_final_response()
                    if inspect.isawaitable(final_payload):
                        final_payload = await final_payload
                    yield {
                        "type": event_type,
                        "response": final_payload,
                    }
                elif event_type == "response.error":
                    error = getattr(event, "error", None)
                    raise ConversationStreamError(f"OpenAI streaming error: {error}")
                else:
                    yield {
                        "type": event_type,
                        "event": event,
                    }

            if final_payload is None:
                final_payload = stream.get_final_response()
                if inspect.isawaitable(final_payload):
                    final_payload = await final_payload
                if final_payload is not None:
                    yield {
                        "type": "response.completed",
                        "response": final_payload,
                    }
