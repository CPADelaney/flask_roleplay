from __future__ import annotations

from typing import Union

from celery import shared_task

from memory.interference import MemoryInterferenceManager
from nyx.tasks.utils import run_coro


@shared_task(name="memory.tasks.generate_blended_memory")
def generate_blended_memory_task(
    user_id: int,
    conversation_id: int,
    entity_type: str,
    entity_id: Union[int, str],
    memory1_id: int,
    memory2_id: int,
    blend_method: str = "auto",
) -> dict:
    async def _run():
        mgr = MemoryInterferenceManager(user_id=user_id, conversation_id=conversation_id)
        # Use the non-DB-hogging path implemented earlier
        return await mgr.generate_blended_memory(
            entity_type=entity_type,
            entity_id=entity_id,
            memory1_id=memory1_id,
            memory2_id=memory2_id,
            blend_method=blend_method,
            conn=None,
        )

    return run_coro(_run())
