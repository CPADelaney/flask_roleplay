"""Database RPC helpers for canonical world updates."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

import asyncpg

from logic.universal_delta import UniversalDelta


logger = logging.getLogger(__name__)


class CanonEventError(RuntimeError):
    """Raised when the canon.apply_event function reports an error."""


def _truncate_for_log(value: Any, limit: int = 200) -> str:
    text = str(value)
    text = text.replace("\n", " ")
    if len(text) > limit:
        return text[: limit - 1] + "\u2026"
    return text


async def write_event(conn: asyncpg.Connection, delta: UniversalDelta) -> Dict[str, Any]:
    """Persist a :class:`UniversalDelta` using ``canon.apply_event``."""

    if conn is None:
        raise CanonEventError("database connection is required")

    payload = delta.model_dump(mode="json", by_alias=True)

    logger.info("write_event invoked for request_id=%s", delta.request_id)

    try:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT canon.apply_event($1::jsonb) AS result", payload
            )
    except asyncpg.PostgresError as exc:  # pragma: no cover - passthrough safety
        logger.info(
            "write_event failed for request_id=%s with database error: %s",
            delta.request_id,
            _truncate_for_log(exc),
        )
        raise CanonEventError(str(exc)) from exc

    if not row or "result" not in row:
        logger.info(
            "write_event failed for request_id=%s: canon.apply_event returned no result",
            delta.request_id,
        )
        raise CanonEventError("canon.apply_event returned no result")

    result = row["result"]
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            logger.info(
                "write_event failed for request_id=%s: canon.apply_event returned invalid JSON",
                delta.request_id,
            )
            raise CanonEventError("canon.apply_event returned invalid JSON text") from exc

    if not isinstance(result, dict):
        logger.info(
            "write_event failed for request_id=%s: canon.apply_event returned non-object",
            delta.request_id,
        )
        raise CanonEventError("canon.apply_event must return a JSON object")

    applied_flag = result.get("applied")
    error_message = result.get("error")
    logger.info(
        "write_event completed for request_id=%s applied=%s error=%s",
        delta.request_id,
        applied_flag,
        _truncate_for_log(error_message, 200) if error_message else None,
    )

    return result
