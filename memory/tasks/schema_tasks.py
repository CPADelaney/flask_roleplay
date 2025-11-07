"""Background tasks for memory schema detection."""

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

from memory.schemas import MemorySchemaManager
from nyx.tasks.base import NyxTask, app
from nyx.tasks.utils import run_coro

logger = logging.getLogger(__name__)


def _coerce_int(value: int | str, *, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        logger.warning("Failed to coerce %s to int: %r", field, value)
        raise ValueError(f"{field} must be integer-coercible: {value!r}") from exc


def _coerce_int_list(values: Iterable[int | str] | None, *, field: str) -> List[int]:
    coerced: List[int] = []
    if not values:
        return coerced
    for item in values:
        coerced.append(_coerce_int(item, field=field))
    return coerced


@app.task(
    bind=True,
    base=NyxTask,
    name="memory.tasks.schema_tasks.detect_schema_from_memories",
    queue="background",
    priority=5,
)
def detect_schema_from_memories_task(
    self,
    user_id: int | str,
    conversation_id: int | str,
    entity_type: str,
    entity_id: int | str,
    memory_ids: Sequence[int | str] | None = None,
    tags: Sequence[str] | None = None,
    min_memories: int | str = 3,
):
    """Execute schema detection for a conversation in the background."""

    async def _run():
        manager = MemorySchemaManager(
            user_id=_coerce_int(user_id, field="user_id"),
            conversation_id=_coerce_int(conversation_id, field="conversation_id"),
        )
        return await manager.detect_schema_from_memories(
            entity_type=entity_type,
            entity_id=_coerce_int(entity_id, field="entity_id"),
            memory_ids=_coerce_int_list(memory_ids, field="memory_id"),
            tags=list(tags or []),
            min_memories=_coerce_int(min_memories, field="min_memories"),
            background=False,
        )

    return run_coro(_run())


__all__ = ["detect_schema_from_memories_task"]
