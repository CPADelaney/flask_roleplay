"""Database helpers for managing OpenAI conversation metadata."""

from __future__ import annotations

import inspect
from typing import Any, AsyncIterator, Callable, Dict, Iterable, Mapping, Optional, Sequence

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

_CHATKIT_RETURNING_COLUMNS = (
    "id",
    "conversation_id",
    "chatkit_assistant_id",
    "chatkit_thread_id",
    "chatkit_run_id",
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


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, bytes):  # pragma: no cover - defensive
        try:
            return value.decode("utf-8").strip() or None
        except UnicodeDecodeError:
            return None
    text = str(value).strip()
    return text or None


def _flatten_non_negotiables(value: Any) -> Sequence[str]:
    if value is None:
        return []

    if isinstance(value, (str, bytes)):
        text = _as_optional_str(value)
        return [text] if text else []

    flattened: list[str] = []

    if isinstance(value, Mapping):
        for item in value.values():
            flattened.extend(_flatten_non_negotiables(item))
        return flattened

    if isinstance(value, Iterable):  # type: ignore[redundant-expr]
        for item in value:
            flattened.extend(_flatten_non_negotiables(item))
        return flattened

    text = _as_optional_str(value)
    return [text] if text else []


def _normalise_scene_seal_payload(data: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(data, Mapping):
        return None

    def _first_present(keys: Sequence[str]) -> Optional[str]:
        for key in keys:
            if key in data:
                value = _as_optional_str(data.get(key))
                if value:
                    return value
        return None

    venue = _first_present(
        (
            "venue",
            "current_venue",
            "currentVenue",
            "CurrentVenue",
            "location",
            "location_name",
            "locationName",
            "CurrentLocation",
            "current_location",
            "currentLocation",
        )
    )

    date = _first_present(
        (
            "date",
            "current_date",
            "currentDate",
            "CurrentDate",
            "time",
            "time_of_day",
            "timeOfDay",
            "CurrentTime",
            "current_time",
            "currentTime",
            "day",
        )
    )

    non_negotiables_source: Any = None
    for key in (
        "non_negotiables",
        "nonNegotiables",
        "NonNegotiables",
        "non_negotiable_rules",
        "nonNegotiableRules",
        "rules",
    ):
        if key in data:
            non_negotiables_source = data.get(key)
            break

    if non_negotiables_source is None:
        constraints = data.get("constraints")
        if isinstance(constraints, Mapping):
            non_negotiables_source = constraints.get("non_negotiables") or constraints.get(
                "nonNegotiables"
            )

    non_negotiables = list(_flatten_non_negotiables(non_negotiables_source))

    if not venue and not date and not non_negotiables:
        return None

    return {
        "venue": venue,
        "date": date,
        "non_negotiables": non_negotiables,
    }


def extract_scene_seal_from_scene(scene: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(scene, Mapping):
        return None

    metadata = scene.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("scene_seal", "sceneSeal", "seal", "sceneSealData"):
            seal = _normalise_scene_seal_payload(metadata.get(key))
            if seal:
                return seal

        merged_candidate = dict(metadata)
        if "location_reference" not in merged_candidate and scene.get("location_reference"):
            merged_candidate["location_reference"] = scene.get("location_reference")
        seal = _normalise_scene_seal_payload(merged_candidate)
        if seal:
            return seal

    scene_state = scene.get("scene_state")
    seal = _normalise_scene_seal_payload(scene_state if isinstance(scene_state, Mapping) else None)
    if seal:
        return seal

    fallback_payload: Dict[str, Any] = {}
    if scene.get("location_reference"):
        fallback_payload["venue"] = scene.get("location_reference")
    for key in ("venue", "date", "non_negotiables", "nonNegotiables"):
        if key in scene:
            fallback_payload[key] = scene[key]

    seal = _normalise_scene_seal_payload(fallback_payload)
    if seal:
        return seal

    return None


def extract_scene_seal_from_updates(updates: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(updates, Mapping):
        return None

    for key in ("scene_seal", "sceneSeal", "sceneSealData"):
        seal = _normalise_scene_seal_payload(updates.get(key))
        if seal:
            return seal

    roleplay_updates = updates.get("roleplay_updates")
    base_payload: Dict[str, Any] = {}
    if isinstance(roleplay_updates, Mapping):
        base_payload.update(roleplay_updates)

    if not base_payload:
        for key in (
            "CurrentLocation",
            "current_location",
            "currentLocation",
            "venue",
            "location",
        ):
            if key in updates:
                base_payload[key] = updates[key]

    metadata = updates.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("scene_seal", "sceneSeal", "seal"):
            candidate = metadata.get(key)
            if isinstance(candidate, Mapping):
                if "venue" not in base_payload and "venue" in candidate:
                    base_payload["venue"] = candidate.get("venue")
                if "date" not in base_payload and "date" in candidate:
                    base_payload["date"] = candidate.get("date")
                if "non_negotiables" not in base_payload and (
                    "non_negotiables" in candidate or "nonNegotiables" in candidate
                ):
                    base_payload["non_negotiables"] = candidate.get("non_negotiables") or candidate.get(
                        "nonNegotiables"
                    )

    constraints = updates.get("session_constraints") or updates.get("constraints")
    if isinstance(constraints, Mapping):
        if "non_negotiables" not in base_payload:
            base_payload["non_negotiables"] = constraints.get("non_negotiables") or constraints.get(
                "nonNegotiables"
            )

    seal = _normalise_scene_seal_payload(base_payload)
    if seal:
        return seal

    return None


def _build_scene_seal_text(venue: Optional[str], date: Optional[str], non_negotiables: Sequence[str]) -> str:
    lines = ["Scene Seal"]
    if venue:
        lines.append(f"Venue: {venue}")
    if date:
        lines.append(f"Date: {date}")

    lines.append("Non-Negotiables:")
    if non_negotiables:
        lines.extend(f"- {rule}" for rule in non_negotiables)
    else:
        lines.append("- None recorded")

    return "\n".join(lines)


async def ensure_scene_seal_item(
    conn,
    *,
    conversation_id: int,
    venue: Optional[str],
    date: Optional[str],
    non_negotiables: Optional[Iterable[str]] = None,
    source: str = "unknown",
) -> Optional[Dict[str, Any]]:
    non_negotiables_list = list(_flatten_non_negotiables(non_negotiables))
    if not venue and not date and not non_negotiables_list:
        return None

    content = {
        "type": "message",
        "role": "system",
        "content": [
            {
                "type": "input_text",
                "text": _build_scene_seal_text(venue, date, non_negotiables_list),
            }
        ],
    }

    metadata = {
        "venue": venue,
        "date": date,
        "non_negotiables": non_negotiables_list,
        "source": source,
    }

    record = await conn.fetchrow(
        """
        INSERT INTO conversations.items (
            conversation_id,
            role,
            item_type,
            content,
            metadata
        )
        VALUES ($1, 'system', 'scene_seal', $2::jsonb, $3::jsonb)
        ON CONFLICT (conversation_id, role, item_type) DO UPDATE
        SET
            content = EXCLUDED.content,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
        RETURNING id, conversation_id, role, item_type, content, metadata, created_at, updated_at
        """,
        conversation_id,
        content,
        metadata,
    )

    return dict(record) if record else None


def _merge_metadata(
    base: Optional[Mapping[str, Any]],
    *,
    openai_conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a mutable metadata dictionary with the OpenAI conversation id."""

    metadata: Dict[str, Any] = dict(base) if isinstance(base, Mapping) else {}
    if openai_conversation_id:
        metadata["openai_conversation_id"] = openai_conversation_id
    return metadata


def _normalise_openai_record(record: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Ensure conversation rows surface metadata and the remote conversation id."""

    if not record:
        return None

    normalised: Dict[str, Any] = dict(record)

    metadata = normalised.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata_dict: Dict[str, Any] = {}
    else:
        metadata_dict = dict(metadata)

    openai_conversation_id = normalised.get("openai_conversation_id") or metadata_dict.get(
        "openai_conversation_id"
    )

    if openai_conversation_id:
        normalised["openai_conversation_id"] = openai_conversation_id
        metadata_dict["openai_conversation_id"] = openai_conversation_id

    normalised["metadata"] = metadata_dict
    return normalised


async def _upsert_conversation(
    conn,
    *,
    user_id: int,
    conversation_id: int,
    openai_assistant_id: str,
    openai_thread_id: str,
    openai_run_id: Optional[str] = None,
    openai_response_id: Optional[str] = None,
    openai_conversation_id: Optional[str] = None,
    status: str = "pending",
    last_error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Insert or update a conversation row and return the stored record."""

    metadata = _merge_metadata(metadata, openai_conversation_id=openai_conversation_id)

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

    return _normalise_openai_record(record)


async def _upsert_chatkit_thread(
    conn,
    *,
    conversation_id: int,
    chatkit_assistant_id: str,
    chatkit_thread_id: str,
    chatkit_run_id: Optional[str] = None,
    status: str = "pending",
    last_error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Insert or update a ChatKit thread row and return the stored record."""

    metadata = metadata or {}

    record = await conn.fetchrow(
        f"""
        INSERT INTO chatkit_threads (
            conversation_id,
            chatkit_assistant_id,
            chatkit_thread_id,
            chatkit_run_id,
            status,
            last_error,
            metadata
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (conversation_id, chatkit_thread_id) DO UPDATE
        SET
            chatkit_assistant_id = EXCLUDED.chatkit_assistant_id,
            chatkit_run_id = EXCLUDED.chatkit_run_id,
            status = EXCLUDED.status,
            last_error = EXCLUDED.last_error,
            metadata = COALESCE(chatkit_threads.metadata, '{{}}'::jsonb) || EXCLUDED.metadata,
            updated_at = NOW()
        RETURNING {', '.join(_CHATKIT_RETURNING_COLUMNS)}
        """,
        conversation_id,
        chatkit_assistant_id,
        chatkit_thread_id,
        chatkit_run_id,
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
    openai_conversation_id: Optional[str] = None,
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
            openai_conversation_id=openai_conversation_id,
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
            openai_conversation_id=openai_conversation_id,
            status=status,
            last_error=last_error,
            metadata=metadata,
        )


async def create_chatkit_thread(
    *,
    conversation_id: int,
    chatkit_assistant_id: str,
    chatkit_thread_id: str,
    chatkit_run_id: Optional[str] = None,
    status: str = "pending",
    last_error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    conn=None,
) -> Optional[Dict[str, Any]]:
    """Create or update a ChatKit thread and return the stored row."""

    if conn is not None:
        return await _upsert_chatkit_thread(
            conn,
            conversation_id=conversation_id,
            chatkit_assistant_id=chatkit_assistant_id,
            chatkit_thread_id=chatkit_thread_id,
            chatkit_run_id=chatkit_run_id,
            status=status,
            last_error=last_error,
            metadata=metadata,
        )

    async with get_db_connection_context() as db_conn:
        return await _upsert_chatkit_thread(
            db_conn,
            conversation_id=conversation_id,
            chatkit_assistant_id=chatkit_assistant_id,
            chatkit_thread_id=chatkit_thread_id,
            chatkit_run_id=chatkit_run_id,
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
    openai_conversation_id: Optional[str] = None,
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
        return _normalise_openai_record(record)

    if conn is not None:
        existing = await _select_existing(conn)
        if existing:
            return _normalise_openai_record(existing)

        return await _upsert_conversation(
            conn,
            user_id=user_id,
            conversation_id=conversation_id,
            openai_assistant_id=openai_assistant_id,
            openai_thread_id=openai_thread_id,
            openai_run_id=openai_run_id,
            openai_response_id=openai_response_id,
            openai_conversation_id=openai_conversation_id,
            status=status,
            last_error=last_error,
            metadata=metadata,
        )

    async with get_db_connection_context() as db_conn:
        existing = await _select_existing(db_conn)
        if existing:
            return existing

        return await _upsert_conversation(
            db_conn,
            user_id=user_id,
            conversation_id=conversation_id,
            openai_assistant_id=openai_assistant_id,
            openai_thread_id=openai_thread_id,
            openai_run_id=openai_run_id,
            openai_response_id=openai_response_id,
            openai_conversation_id=openai_conversation_id,
            status=status,
            last_error=last_error,
            metadata=metadata,
        )


async def get_or_create_chatkit_thread(
    *,
    conversation_id: int,
    chatkit_assistant_id: str,
    chatkit_thread_id: str,
    chatkit_run_id: Optional[str] = None,
    status: str = "pending",
    last_error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    conn=None,
) -> Optional[Dict[str, Any]]:
    """Fetch an existing ChatKit thread or create it if absent."""

    async def _select_existing(connection) -> Optional[Mapping[str, Any]]:
        record = await connection.fetchrow(
            f"""
            SELECT {', '.join(_CHATKIT_RETURNING_COLUMNS)}
            FROM chatkit_threads
            WHERE conversation_id = $1 AND chatkit_thread_id = $2
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            conversation_id,
            chatkit_thread_id,
        )
        return dict(record) if record else None

    if conn is not None:
        existing = await _select_existing(conn)
        if existing:
            return existing

        return await _upsert_chatkit_thread(
            conn,
            conversation_id=conversation_id,
            chatkit_assistant_id=chatkit_assistant_id,
            chatkit_thread_id=chatkit_thread_id,
            chatkit_run_id=chatkit_run_id,
            status=status,
            last_error=last_error,
            metadata=metadata,
        )

    async with get_db_connection_context() as db_conn:
        existing = await _select_existing(db_conn)
        if existing:
            return existing

        return await _upsert_chatkit_thread(
            db_conn,
            conversation_id=conversation_id,
            chatkit_assistant_id=chatkit_assistant_id,
            chatkit_thread_id=chatkit_thread_id,
            chatkit_run_id=chatkit_run_id,
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
        return _normalise_openai_record(record)

    if conn is not None:
        return await _get(conn)

    async with get_db_connection_context() as db_conn:
        return await _get(db_conn)


async def get_latest_chatkit_thread(
    *,
    conversation_id: int,
    conn=None,
) -> Optional[Dict[str, Any]]:
    """Fetch the most recent ChatKit thread row for a conversation."""

    async def _get(connection):
        record = await connection.fetchrow(
            f"""
            SELECT {', '.join(_CHATKIT_RETURNING_COLUMNS)}
            FROM chatkit_threads
            WHERE conversation_id = $1
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            conversation_id,
        )
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

    seal = extract_scene_seal_from_scene(new_scene_data)
    if seal:
        await ensure_scene_seal_item(
            conn,
            conversation_id=conversation_id,
            venue=seal.get("venue"),
            date=seal.get("date"),
            non_negotiables=seal.get("non_negotiables"),
            source="scene_rotation",
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

    async def create_chatkit_thread(self, **kwargs):
        """Proxy to :func:`create_chatkit_thread`."""

        return await create_chatkit_thread(**kwargs)

    async def get_or_create_chatkit_thread(self, **kwargs):
        """Proxy to :func:`get_or_create_chatkit_thread`."""

        return await get_or_create_chatkit_thread(**kwargs)

    async def get_latest_chatkit_thread(self, **kwargs):
        """Proxy to :func:`get_latest_chatkit_thread`."""

        return await get_latest_chatkit_thread(**kwargs)

    def get_client(self):
        """Expose the configured OpenAI client for downstream helpers."""

        return self._get_client()

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
