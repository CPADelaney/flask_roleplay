"""Shared helpers for conflict mode recommendations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import date, datetime
from typing import Any, Dict, Optional

from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

MODE_OPTIMIZER_NAME = "Mode Optimizer"
MODE_OPTIMIZER_MODEL = "gpt-5-nano"
MODE_OPTIMIZER_INSTRUCTIONS = """
Determine optimal integration mode for current context.

Consider:
- Player engagement patterns
- Story progression
- System load
- Narrative needs
- Player preferences

Recommend mode changes to enhance experience.
"""

_MODE_RECOMMENDATION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS integration_mode_recommendations (
    user_id BIGINT NOT NULL,
    conversation_id BIGINT NOT NULL,
    context_signature TEXT NOT NULL,
    recommended_mode TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence DOUBLE PRECISION,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, conversation_id, context_signature)
);
"""

_table_ready = False
_table_lock = asyncio.Lock()


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.hex()
    return str(value)


def normalize_mode_context(context: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return a JSON-safe copy of the context payload."""

    context = context or {}
    serialized = json.dumps(context, default=_json_default)
    return json.loads(serialized)


def compute_context_signature(
    context: Dict[str, Any],
    current_mode: str,
    current_quality: float,
) -> str:
    """Create a deterministic signature for recommendation caching."""

    payload = {
        "mode": current_mode,
        "quality": round(float(current_quality), 3),
        "context": normalize_mode_context(context),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def _ensure_table() -> None:
    global _table_ready
    if _table_ready:
        return
    async with _table_lock:
        if _table_ready:
            return
        async with get_db_connection_context() as conn:
            await conn.execute(_MODE_RECOMMENDATION_TABLE_SQL)
        _table_ready = True


async def store_mode_recommendation(
    user_id: int,
    conversation_id: int,
    context_signature: str,
    recommended_mode: str,
    *,
    source: str,
    confidence: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a recommended mode for reuse."""

    await _ensure_table()
    metadata_json = json.dumps(metadata, default=_json_default) if metadata is not None else None
    async with get_db_connection_context() as conn:
        await conn.execute(
            """
            INSERT INTO integration_mode_recommendations (
                user_id,
                conversation_id,
                context_signature,
                recommended_mode,
                source,
                confidence,
                metadata,
                created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id, conversation_id, context_signature)
            DO UPDATE SET
                recommended_mode = EXCLUDED.recommended_mode,
                source = EXCLUDED.source,
                confidence = EXCLUDED.confidence,
                metadata = EXCLUDED.metadata,
                created_at = CURRENT_TIMESTAMP
            """,
            user_id,
            conversation_id,
            context_signature,
            recommended_mode,
            source,
            confidence,
            metadata_json,
        )


async def fetch_mode_recommendation(
    user_id: int,
    conversation_id: int,
    context_signature: str,
) -> Optional[Dict[str, Any]]:
    """Fetch a cached recommendation if one exists."""

    await _ensure_table()
    async with get_db_connection_context() as conn:
        row = await conn.fetchrow(
            """
            SELECT recommended_mode, source, confidence, metadata, created_at
              FROM integration_mode_recommendations
             WHERE user_id = $1 AND conversation_id = $2 AND context_signature = $3
            """,
            user_id,
            conversation_id,
            context_signature,
        )
    if not row:
        return None
    payload = dict(row)
    metadata = payload.get("metadata")
    if isinstance(metadata, str):
        try:
            payload["metadata"] = json.loads(metadata)
        except json.JSONDecodeError:
            payload["metadata"] = {"raw": metadata}
    return payload


def build_mode_recommendation_prompt(
    current_mode: str,
    current_quality: float,
    context: Dict[str, Any],
) -> str:
    """Create the LLM prompt for mode recommendations."""

    context_json = json.dumps(normalize_mode_context(context), indent=2, sort_keys=True)
    return (
        "Recommend integration mode based on context:\n\n"
        f"Current Mode: {current_mode}\n"
        f"Experience Quality: {current_quality:.1%}\n"
        f"Context: {context_json}\n\n"
        "Available modes:\n"
        "- full_immersion: All systems active\n"
        "- story_focus: Prioritize narrative\n"
        "- social_dynamics: Focus on relationships\n"
        "- background_aware: Emphasize world events\n"
        "- player_centric: Maximum player agency\n"
        "- emergent: Natural pattern emergence\n"
        "- guided: Curated experience\n\n"
        "Which mode would improve the experience? Return just the mode name."
    )


__all__ = [
    "MODE_OPTIMIZER_INSTRUCTIONS",
    "MODE_OPTIMIZER_MODEL",
    "MODE_OPTIMIZER_NAME",
    "build_mode_recommendation_prompt",
    "compute_context_signature",
    "fetch_mode_recommendation",
    "normalize_mode_context",
    "store_mode_recommendation",
]
