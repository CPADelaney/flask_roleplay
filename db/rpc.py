"""Database RPC helpers for canonical world updates."""

from __future__ import annotations

import json
from typing import Any, Dict

import asyncpg

from logic.universal_delta import UniversalDelta


class CanonEventError(RuntimeError):
    """Raised when the canon.apply_event function reports an error."""


async def write_event(conn: asyncpg.Connection, delta: UniversalDelta) -> Dict[str, Any]:
    """Persist a :class:`UniversalDelta` using ``canon.apply_event``."""

    if conn is None:
        raise CanonEventError("database connection is required")

    payload = delta.model_dump(mode="json", by_alias=True)

    try:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT canon.apply_event($1::jsonb) AS result", payload
            )
    except asyncpg.PostgresError as exc:  # pragma: no cover - passthrough safety
        raise CanonEventError(str(exc)) from exc

    if not row or "result" not in row:
        raise CanonEventError("canon.apply_event returned no result")

    result = row["result"]
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise CanonEventError("canon.apply_event returned invalid JSON text") from exc

    if not isinstance(result, dict):
        raise CanonEventError("canon.apply_event must return a JSON object")

    return result
