"""Memory pipeline tasks."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict

from nyx.tasks.base import NyxTask, app

from nyx.utils.idempotency import idempotent
from rag.vector_store import hosted_vector_store_enabled, legacy_vector_store_enabled

logger = logging.getLogger(__name__)


def _add_key(payload: Dict[str, Any]) -> str:
    return f"memory:add:{payload.get('conversation_id')}:{payload.get('turn_id')}"


@app.task(bind=True, base=NyxTask, name="nyx.tasks.heavy.memory_tasks.add_and_embed", acks_late=True)
@idempotent(key_fn=lambda payload: _add_key(payload))
def add_and_embed(self, payload: Dict[str, Any]) -> Dict[str, Any] | None:
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


@app.task(bind=True, base=NyxTask, name="nyx.tasks.heavy.memory_tasks.consolidate_decay", acks_late=True)
def consolidate_decay(self) -> str:
    """Periodic consolidation/decay placeholder."""

    logger.debug("Memory consolidation tick")
    return "ok"


@app.task(
    bind=True,
    base=NyxTask,
    name="nyx.tasks.heavy.memory_tasks.hydrate_local_embeddings",
    acks_late=True,
)
def hydrate_local_embeddings(self, user_id: int, conversation_id: int) -> str:
    """Warm-load local embedding assets and replay deferred hydration work."""

    logger.info(
        "Hydrating local embeddings for user_id=%s conversation_id=%s",
        user_id,
        conversation_id,
    )

    if hosted_vector_store_enabled():
        logger.info(
            "Hosted Agents vector store enabled; skipping local hydration for user_id=%s conversation_id=%s",
            user_id,
            conversation_id,
        )
        return "skipped:hosted-vector-store"

    if not legacy_vector_store_enabled():
        logger.info(
            "Legacy vector store disabled; skipping local hydration for user_id=%s conversation_id=%s",
            user_id,
            conversation_id,
        )
        return "skipped:legacy-disabled"

    persist_base = os.getenv("MEMORY_VECTOR_PERSIST_BASE", "./vector_stores")
    vector_store_type = os.getenv("MEMORY_VECTOR_STORE_TYPE", "chroma")
    persist_directory = os.path.join(
        persist_base,
        vector_store_type,
        f"{user_id}_{conversation_id}",
    )

    from memory.chroma_vector_store import (  # lazy import to avoid startup penalties
        ChromaVectorDatabase,
        init_chroma_if_present_else_noop,
    )

    init_chroma_if_present_else_noop(persist_directory)

    async def _initialize_chroma() -> None:
        vector_db = ChromaVectorDatabase(
            persist_directory=persist_directory,
            config={"use_default_embedding_function": False},
        )
        await vector_db.initialize()

    try:
        asyncio.run(_initialize_chroma())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_initialize_chroma())
        finally:
            loop.close()

    logger.info(
        "Queued hydration replay for user_id=%s conversation_id=%s",
        user_id,
        conversation_id,
    )
    return "hydrated"


schedule_heavy_hydration = hydrate_local_embeddings


__all__ = [
    "add_and_embed",
    "consolidate_decay",
    "hydrate_local_embeddings",
    "schedule_heavy_hydration",
]
