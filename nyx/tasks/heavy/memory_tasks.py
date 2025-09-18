"""Memory pipeline tasks."""

from __future__ import annotations

import logging
from typing import Any, Dict

from celery import shared_task

from nyx.utils.idempotency import idempotent

logger = logging.getLogger(__name__)


def _add_key(payload: Dict[str, Any]) -> str:
    return f"memory:add:{payload.get('conversation_id')}:{payload.get('turn_id')}"


@shared_task(name="nyx.tasks.heavy.memory_tasks.add_and_embed", acks_late=True)
@idempotent(key_fn=lambda payload: _add_key(payload))
def add_and_embed(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """Persist memory text and queue embeddings."""

    if not payload:
        return None

    text = payload.get("text") or ""
    if not text.strip():
        logger.debug("Skipping empty memory payload for turn=%s", payload.get("turn_id"))
        return {"status": "skipped"}

    # Placeholder: real implementation should persist and hand off to the embedding service.
    logger.debug(
        "Memory add queued turn=%s length=%s", payload.get("turn_id"), len(text)
    )
    return {"status": "queued", "turn_id": payload.get("turn_id")}


@shared_task(name="nyx.tasks.heavy.memory_tasks.consolidate_decay", acks_late=True)
def consolidate_decay() -> str:
    """Periodic consolidation/decay placeholder."""

    logger.debug("Memory consolidation tick")
    return "ok"


__all__ = ["add_and_embed", "consolidate_decay"]
