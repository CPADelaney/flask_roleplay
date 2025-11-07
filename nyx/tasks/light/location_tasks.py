from __future__ import annotations

import logging
from typing import Dict, Any

from db.connection import get_db_connection_context
from lore.core.canon import log_canonical_event, get_canon_memory_orchestrator
from lore.core.context import CanonicalContext
from memory.memory_orchestrator import EntityType
from nyx.tasks.base import NyxTask, app
from nyx.tasks.utils import run_coro

logger = logging.getLogger(__name__)

@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.tasks.light.notify_canon_of_location",
    queue="light",
    priority=4,
)
def notify_canon_of_location_task(self, user_id: int, conversation_id: int, location: Dict[str, Any]) -> str:
    """
    Post-commit fan-out: write a tiny canon event (own short DB scope),
    then push memory + vector entry (no pooled conn held).
    """
    async def _run():
        ctx = CanonicalContext(user_id=user_id, conversation_id=conversation_id)
        name = location.get("location_name") or location.get("name") or "Unknown"
        city = location.get("city") or "the world"
        loc_type = (location.get("location_type") or "venue")
        scope = (location.get("scope") or "real")

        # short DB scope for canon event only
        async with get_db_connection_context() as conn:
            await log_canonical_event(
                ctx, conn,
                f"Location '{name}' established in {city}",
                tags=["location", "creation", loc_type, scope],
                significance=7 if loc_type == "city" else 6 if loc_type == "district" else 5,
                persist_memory=True
            )

        # memory + vector fan-out (no DB conn held)
        mo = await get_canon_memory_orchestrator(user_id, conversation_id)
        parts = [name]
        if location.get("district"):
            parts.append(f"in {location['district']}")
        if city:
            parts.append(f", {city}")
        text = " ".join(parts)

        await mo.store_memory(
            entity_type=EntityType.LORE,
            entity_id=0,
            memory_text=text,
            significance=0.8 if loc_type in ("city", "district") else 0.7,
            tags=["location", loc_type, scope],
            metadata={
                "location_id": location.get("id"),
                "location_name": name,
                "location_type": loc_type,
                "city": city,
                "district": location.get("district"),
                "scope": scope,
            }
        )
        await mo.add_to_vector_store(
            text=text,
            metadata={
                "entity_type": "location",
                "location_id": location.get("id"),
                "location_name": name,
                "location_type": loc_type,
                "city": city,
                "district": location.get("district"),
                "is_fictional": bool(location.get("is_fictional", False)),
                "scope": scope,
                "user_id": user_id,
                "conversation_id": conversation_id,
            },
            entity_type="location",
        )

    return run_coro(_run())
