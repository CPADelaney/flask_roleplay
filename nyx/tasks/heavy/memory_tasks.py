"""Memory pipeline tasks."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable, Dict, TypeVar

from nyx.tasks.base import NyxTask, app

from nyx.utils.idempotency import idempotent
from rag.vector_store import (
    get_hosted_vector_store_ids,
    hosted_vector_store_enabled,
    legacy_vector_store_enabled,
)

try:  # pragma: no cover - optional config helper
    from memory.memory_config import get_memory_config
except Exception:  # pragma: no cover
    get_memory_config = None  # type: ignore[assignment]

from memory.memory_service import MemoryEmbeddingService

logger = logging.getLogger(__name__)


T = TypeVar("T")


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
def hydrate_local_embeddings(
    self,
    user_id: int,
    conversation_id: int,
    *,
    trace_id: str | None = None,
) -> str:
    """Warm-load local embedding assets and replay deferred hydration work."""

    # Celery instrumentation may inject tracing kwargs (e.g., trace_id); accept and ignore
    # them so the task remains compatible with existing producers.
    _ = trace_id

    logger.info(
        "Hydrating local embeddings for user_id=%s conversation_id=%s",
        user_id,
        conversation_id,
    )

    def _load_memory_config() -> Dict[str, Any]:
        if callable(get_memory_config):
            try:
                config = get_memory_config() or {}
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Failed to load memory config for hydration: %s", exc)
            else:
                if isinstance(config, dict):
                    return config
        return {}

    config = _load_memory_config()

    hosted_ids = get_hosted_vector_store_ids(config)

    if hosted_vector_store_enabled(hosted_ids, config=config):
        logger.info(
            "Hosted Agents vector store enabled; skipping local hydration for user_id=%s conversation_id=%s",
            user_id,
            conversation_id,
        )
        return "skipped:hosted-vector-store"

    if not legacy_vector_store_enabled(config):
        logger.info(
            "Legacy vector store disabled; skipping local hydration for user_id=%s conversation_id=%s",
            user_id,
            conversation_id,
        )
        return "skipped:legacy-disabled"

    def _run_coroutine(factory: Callable[[], Awaitable[T]]) -> T:
        try:
            return asyncio.run(factory())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(factory())
            finally:
                loop.close()

    vector_config = config.get("vector_store") if isinstance(config, dict) else {}
    if not isinstance(vector_config, dict):
        vector_config = {}

    embedding_config = config.get("embedding") if isinstance(config, dict) else {}
    if not isinstance(embedding_config, dict):
        embedding_config = {}

    vector_store_type = str(
        vector_config.get("type")
        or os.getenv("MEMORY_VECTOR_STORE_TYPE")
        or "chroma"
    ).lower()

    embedding_model = str(
        embedding_config.get("type")
        or os.getenv("MEMORY_EMBEDDING_TYPE")
        or "openai"
    ).lower()

    service = MemoryEmbeddingService(
        user_id=user_id,
        conversation_id=conversation_id,
        vector_store_type=vector_store_type,
        embedding_model=embedding_model,
        config=config or None,
    )

    result = _run_coroutine(service.hydrate_legacy_vector_store)

    logger.info(
        "Hydration result for user_id=%s conversation_id=%s: %s",
        user_id,
        conversation_id,
        result,
    )
    return result


schedule_heavy_hydration = hydrate_local_embeddings


__all__ = [
    "add_and_embed",
    "consolidate_decay",
    "hydrate_local_embeddings",
    "schedule_heavy_hydration",
]
