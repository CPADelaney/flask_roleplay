from __future__ import annotations

import asyncio
import logging
from typing import List

from celery import shared_task

from memory.schemas import MemorySchemaManager

logger = logging.getLogger(__name__)


@shared_task(name="memory.tasks.detect_schema_from_memories")
def detect_schema_from_memories_task(
    user_id: int,
    conversation_id: int,
    entity_type: str,
    entity_id: int | str,
    memory_ids: List[int],
    tags: List[str],
    min_memories: int,
) -> dict:
    async def _run():
        mgr = MemorySchemaManager(user_id=user_id, conversation_id=conversation_id)
        return await mgr.detect_schema_from_memories(
            entity_type=entity_type,
            entity_id=entity_id,
            memory_ids=memory_ids or None,
            tags=tags or None,
            min_memories=min_memories or 3,
            conn=None,
            background=False,  # run for real inside the task
        )

    return asyncio.run(_run())
