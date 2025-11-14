from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict, Optional, Tuple

from nyx import settings
from nyx.nyx_agent.context import NyxContext

logger = logging.getLogger(__name__)

_context_warm_promises: Dict[Tuple[int, int], asyncio.Future] = {}


async def _await_orchestrator_with_retry(
    context: NyxContext,
    orchestrator: str,
    *,
    attempts: int = 3,
    delay: float = 0.5,
) -> bool:
    """Wait for an orchestrator to become ready with retry semantics."""

    for attempt in range(1, attempts + 1):
        try:
            is_ready = await context.await_orchestrator(orchestrator)
        except Exception:
            logger.exception(
                "Error awaiting orchestrator '%s' on attempt %s", orchestrator, attempt
            )
            is_ready = False

        if is_ready:
            return True

        if attempt < attempts:
            logger.warning(
                "Orchestrator '%s' not ready on attempt %s/%s; retrying",
                orchestrator,
                attempt,
                attempts,
            )
            await asyncio.sleep(delay * attempt)

    logger.error(
        "Orchestrator '%s' failed to become ready after %s attempts",
        orchestrator,
        attempts,
    )
    return False


async def warm_user_context_cache(
    user_id: int,
    conversation_id: int,
    *,
    redis_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Warm the Nyx user context cache for a given conversation.

    By default this function executes the minimal warm path, skipping expensive
    orchestrator bootstraps. Enable ``settings.CONFLICT_EAGER_WARMUP`` (set
    ``NYX_CONFLICT_EAGER_WARMUP=1``) to opt into the legacy eager warm that fully
    initializes memory/lore orchestrators and builds a context bundle.
    """

    key = f"ctx:warmed:{user_id}:{conversation_id}"
    promise_key = (user_id, conversation_id)
    logger.info(
        "Starting context cache warm-up for user_id=%s conversation_id=%s",
        user_id,
        conversation_id,
    )

    if redis_client is not None:
        try:
            cached_value = redis_client.get(key)
        except Exception:
            logger.exception("Redis get failed for context warm cache key=%s", key)
        else:
            if cached_value:
                logger.info(
                    "Context cache already warmed for user_id=%s conversation_id=%s",
                    user_id,
                    conversation_id,
                )
                return {
                    "status": "cached",
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                }

    existing_future = _context_warm_promises.get(promise_key)
    if existing_future is not None:
        logger.info(
            "Awaiting in-flight context warm for user_id=%s conversation_id=%s",
            user_id,
            conversation_id,
        )
        return await existing_future

    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    _context_warm_promises[promise_key] = future

    try:
        context = NyxContext(user_id=user_id, conversation_id=conversation_id)

        eager_conflict = settings.CONFLICT_EAGER_WARMUP

        await context.initialize(warm_minimal=not eager_conflict)

        if eager_conflict:
            logger.info(
                "Eager warm enabled via settings.CONFLICT_EAGER_WARMUP for user_id=%s conversation_id=%s",
                user_id,
                conversation_id,
            )
            for orchestrator in ("memory", "lore"):
                ready = await _await_orchestrator_with_retry(context, orchestrator)
                if not ready:
                    raise RuntimeError(
                        f"Orchestrator '{orchestrator}' did not become ready during warm-up"
                    )

            await context.build_context_for_input("", None)
            warm_result: Dict[str, Any] = {
                "status": "warmed",
                "mode": "full",
                "user_id": user_id,
                "conversation_id": conversation_id,
            }
        else:
            warm_result = await context.warm_minimal_context()
    except Exception as exc:
        logger.exception(
            "Failed context cache warm-up for user_id=%s conversation_id=%s: %s",
            user_id,
            conversation_id,
            exc,
        )
        if not future.done():
            future.set_exception(exc)
        if _context_warm_promises.get(promise_key) is future:
            _context_warm_promises.pop(promise_key, None)
        raise

    result = {
        "status": warm_result.get("status", "warmed"),
        "user_id": user_id,
        "conversation_id": conversation_id,
    }
    if "mode" in warm_result:
        result["mode"] = warm_result["mode"]
    if not future.done():
        future.set_result(result)
    if _context_warm_promises.get(promise_key) is future:
        _context_warm_promises.pop(promise_key, None)

    if redis_client is not None:
        try:
            redis_client.setex(key, 600, "1")
        except Exception:
            logger.exception("Redis setex failed for context warm cache key=%s", key)

    logger.info(
        "Successfully warmed context cache for user_id=%s conversation_id=%s",
        user_id,
        conversation_id,
    )
    return result


__all__ = ["warm_user_context_cache", "_context_warm_promises"]
