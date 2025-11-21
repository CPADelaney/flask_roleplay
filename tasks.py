# tasks.py

from __future__ import annotations

import os
import json
import logging
import asyncio
import asyncpg
import datetime
import redis
import time
import traceback
import re
import hashlib
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from nyx.tasks.celery_app import app as celery_app
from celery.signals import task_revoked, worker_process_init
from celery import shared_task
from billiard.exceptions import SoftTimeLimitExceeded
from agents import trace, custom_span, RunContextWrapper
from agents.tracing import get_current_trace
from context.cache_warmup import warm_user_context_cache
from infra.cache import get_redis_client
from nyx.tasks.background.world_tasks import enqueue_background_universal_updates

# --- DB utils (async loop + connection mgmt) ---
from db.connection import (
    get_db_connection_context,
    get_db_connection_sync,
    run_async_in_worker_loop,
)
from nyx.conversation.store import ConversationStore

# --- LLM + NPC + memory integration (unchanged external modules) ---
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from new_game_agent import NewGameAgent
from npcs.npc_learning_adaptation import NPCLearningManager
from memory.memory_nyx_integration import run_maintenance_through_nyx
from lore.systems.regional_culture import RegionalCultureSystem

# --- Core NyxBrain + checkpointing ---
from nyx.core.brain.base import NyxBrain
from nyx.core.brain.checkpointing_agent import CheckpointingPlannerAgent

# --- New scene-scoped SDK (lazy singleton) ---
from nyx.nyx_agent_sdk import NyxAgentSDK, NyxSDKConfig

from psycopg2 import errors as psycopg2_errors

logger = logging.getLogger(__name__)
_conversation_store = ConversationStore()

# Define DSN (optional sanity check)
DB_DSN = os.getenv("DB_DSN", "postgresql://user:password@host:port/database")
if not DB_DSN:
    logger.error("DB_DSN environment variable not set for Celery tasks!")

# --- Application Readiness Flag (worker-local) ---
_APP_INITIALIZED = False
_LAST_INIT_CHECK_TIME = 0.0
_INIT_CHECK_INTERVAL = 30  # seconds

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
try:
    redis_publisher = redis.from_url(REDIS_URL)
    # Test the connection at startup
    redis_publisher.ping()
    logger.info("Redis publisher connected successfully for Celery tasks.")
except Exception as e:
    logger.critical(f"CRITICAL: Celery worker could not connect to Redis publisher at {REDIS_URL}. Tasks that publish results will fail. Error: {e}")
    redis_publisher = None


CONFLICT_CACHE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS CulturalConflictAnalysisCache (
    nation1_id INTEGER NOT NULL,
    nation2_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    summary JSONB,
    last_error TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (nation1_id, nation2_id)
);
"""

CONFLICT_CACHE_STALE_AFTER = datetime.timedelta(hours=6)
CONFLICT_CACHE_PENDING_RETRY_AFTER = datetime.timedelta(minutes=15)
CONFLICT_CACHE_FAILURE_RETRY_AFTER = datetime.timedelta(hours=1)


def set_app_initialized():
    """Call from main.py AFTER successful NyxBrain (or app) init."""
    global _APP_INITIALIZED
    _APP_INITIALIZED = True
    logger.info("Application initialization status set to True for Celery tasks.")


async def is_app_initialized() -> bool:
    """
    Workers can't see the in-process flag of your web process.
    Treat the app as 'ready' if DB is reachable (cached for a short interval).
    """
    global _APP_INITIALIZED, _LAST_INIT_CHECK_TIME
    if _APP_INITIALIZED:
        return True

    now = time.time()
    if now - _LAST_INIT_CHECK_TIME < _INIT_CHECK_INTERVAL:
        return False

    _LAST_INIT_CHECK_TIME = now
    try:
        async with get_db_connection_context() as conn:
            await conn.fetchval("SELECT 1")
        _APP_INITIALIZED = True
        return True
    except Exception:
        return False


def serialize_for_celery(obj: Any) -> Any:
    """Make pydantic/objects JSON-serializable for Celery results."""

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Enum):
        return serialize_for_celery(obj.value)

    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {str(k): serialize_for_celery(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [serialize_for_celery(v) for v in obj]

    if hasattr(obj, "model_dump"):
        try:
            dumped = obj.model_dump(mode="json")
        except TypeError:
            dumped = obj.model_dump()
        return serialize_for_celery(dumped)

    if hasattr(obj, "dict"):
        return serialize_for_celery(obj.dict())

    if hasattr(obj, "__dict__"):
        return serialize_for_celery(vars(obj))

    return str(obj)


_SENSITIVE_TOKEN_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"bearer\s+[A-Za-z0-9._-]{10,}", re.IGNORECASE),
    re.compile(r"[A-Fa-f0-9]{32,}"),
]


def _redact_sensitive_text(text: str) -> str:
    """Best-effort redaction for secrets that may appear in logs."""

    if not text:
        return text

    redacted = text
    for pattern in _SENSITIVE_TOKEN_PATTERNS:
        redacted = pattern.sub("<redacted>", redacted)
    return redacted


def get_preset_id(d: Dict[str, Any]) -> Optional[str]:
    """Extract preset story ID from various possible keys."""
    return d.get("preset_story_id") or d.get("story_id") or d.get("presetStoryId")


def _schedule_universal_updates(
    *,
    user_id: int,
    conversation_id: int,
    updates: Optional[Dict[str, Any]],
    request_id: Optional[str],
    source: str,
) -> None:
    if not updates:
        return

    payload = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "updates": updates,
        "request_id": request_id,
        "source": source,
    }
    scheduled = enqueue_background_universal_updates(payload)
    if scheduled:
        logger.info(
            "[UniversalUpdate] Scheduled async apply for conv=%s request_id=%s source=%s",
            conversation_id,
            request_id,
            source,
        )
    else:
        logger.warning(
            "[UniversalUpdate] Failed to enqueue updates for conv=%s request_id=%s",  # pragma: no cover
            conversation_id,
            request_id,
        )


def _normalize_nation_pair(nation1_id: int, nation2_id: int) -> Tuple[int, int]:
    n1, n2 = int(nation1_id), int(nation2_id)
    return (n1, n2) if n1 <= n2 else (n2, n1)


async def _ensure_conflict_cache_table() -> None:
    async with get_db_connection_context() as conn:
        await conn.execute(CONFLICT_CACHE_TABLE_SQL)


async def _get_conflict_cache_row(n1: int, n2: int) -> Optional[asyncpg.Record]:
    async with get_db_connection_context() as conn:
        return await conn.fetchrow(
            """
            SELECT status, summary, last_error, updated_at
            FROM CulturalConflictAnalysisCache
            WHERE nation1_id = $1 AND nation2_id = $2
            """,
            n1,
            n2,
        )


async def _set_conflict_cache_pending(n1: int, n2: int) -> None:
    async with get_db_connection_context() as conn:
        await conn.execute(
            """
            INSERT INTO CulturalConflictAnalysisCache (nation1_id, nation2_id, status, updated_at, last_error)
            VALUES ($1, $2, 'pending', NOW(), NULL)
            ON CONFLICT (nation1_id, nation2_id)
            DO UPDATE SET status = 'pending', updated_at = NOW(), last_error = NULL
            """,
            n1,
            n2,
        )


async def _update_conflict_cache(
    n1: int,
    n2: int,
    status: str,
    summary_json: Optional[str],
    error: Optional[str],
) -> None:
    async with get_db_connection_context() as conn:
        await conn.execute(
            """
            UPDATE CulturalConflictAnalysisCache
               SET status = $3,
                   summary = $4::jsonb,
                   last_error = $5,
                   updated_at = NOW()
             WHERE nation1_id = $1 AND nation2_id = $2
            """,
            n1,
            n2,
            status,
            summary_json,
            error,
        )


async def _refresh_conflict_pair(
    rcs: RegionalCultureSystem,
    nation1_id: int,
    nation2_id: int,
    force: bool = False,
) -> Dict[str, Any]:
    await _ensure_conflict_cache_table()
    n1, n2 = _normalize_nation_pair(nation1_id, nation2_id)

    existing = await _get_conflict_cache_row(n1, n2)
    now = datetime.datetime.utcnow()

    if existing and not force:
        updated_at = existing.get("updated_at") if existing else None
        status = existing.get("status") if existing else None

        if status == "ready" and updated_at and updated_at > now - CONFLICT_CACHE_STALE_AFTER:
            return {"nation1_id": n1, "nation2_id": n2, "status": "skipped", "reason": "fresh"}
        if status == "pending" and updated_at and updated_at > now - CONFLICT_CACHE_PENDING_RETRY_AFTER:
            return {"nation1_id": n1, "nation2_id": n2, "status": "skipped", "reason": "recently_pending"}
        if status == "failed" and updated_at and updated_at > now - CONFLICT_CACHE_FAILURE_RETRY_AFTER:
            return {"nation1_id": n1, "nation2_id": n2, "status": "skipped", "reason": "recent_failure"}

    await _set_conflict_cache_pending(n1, n2)

    try:
        analysis = await rcs.detect_cultural_conflicts(n1, n2)
    except Exception as exc:
        logger.exception(
            "Error computing cultural conflict analysis for nations %s and %s", n1, n2
        )
        await _update_conflict_cache(n1, n2, "failed", None, str(exc))
        return {
            "nation1_id": n1,
            "nation2_id": n2,
            "status": "failed",
            "error": str(exc),
        }

    if not analysis or not isinstance(analysis, dict):
        await _update_conflict_cache(n1, n2, "failed", None, "Unknown error")
        return {
            "nation1_id": n1,
            "nation2_id": n2,
            "status": "failed",
            "error": "Unknown error",
        }

    if analysis.get("error"):
        error_msg = analysis.get("error")
        await _update_conflict_cache(n1, n2, "failed", None, error_msg)
        return {
            "nation1_id": n1,
            "nation2_id": n2,
            "status": "failed",
            "error": error_msg,
        }

    summary_json = json.dumps(analysis)
    await _update_conflict_cache(n1, n2, "ready", summary_json, None)
    return {
        "nation1_id": n1,
        "nation2_id": n2,
        "status": "ready",
        "severity": analysis.get("severity_level"),
    }


async def _gather_conflict_pairs(limit: int) -> List[Tuple[int, int]]:
    await _ensure_conflict_cache_table()
    async with get_db_connection_context() as conn:
        try:
            rows = await conn.fetch(
                """
                WITH pairs AS (
                    SELECT LEAST(nation1_id, nation2_id) AS n1,
                           GREATEST(nation1_id, nation2_id) AS n2
                      FROM InternationalRelations
                     WHERE nation1_id IS NOT NULL AND nation2_id IS NOT NULL
                )
                SELECT DISTINCT p.n1 AS nation1_id,
                                p.n2 AS nation2_id,
                                cca.status,
                                cca.updated_at
                  FROM pairs p
             LEFT JOIN CulturalConflictAnalysisCache cca
                    ON cca.nation1_id = p.n1 AND cca.nation2_id = p.n2
                 WHERE cca.updated_at IS NULL
                    OR cca.status IN ('pending', 'failed')
                    OR cca.updated_at < NOW() - INTERVAL '6 hours'
                 LIMIT $1
                """,
                limit,
            )
        except asyncpg.exceptions.UndefinedTableError:
            return []
    return [(_normalize_nation_pair(r["nation1_id"], r["nation2_id"])) for r in rows]


@shared_task(bind=True, max_retries=2, default_retry_delay=300)
def refresh_cultural_conflict_cache(
    self,
    nation1_id: int,
    nation2_id: int,
    force: bool = False,
):
    """Compute and cache cultural conflict analysis for a nation pair."""

    async def _run() -> Dict[str, Any]:
        if not await is_app_initialized():
            return {"status": "deferred"}

        rcs = RegionalCultureSystem(user_id=0, conversation_id=0)
        await rcs.ensure_initialized()
        return await _refresh_conflict_pair(rcs, nation1_id, nation2_id, force=force)

    return run_async_in_worker_loop(_run())


@shared_task(bind=True, max_retries=1, default_retry_delay=900)
def refresh_all_cultural_conflict_caches(
    self,
    limit: int = 10,
    force: bool = False,
):
    """Refresh cultural conflict caches for active nation pairs."""

    async def _run() -> Dict[str, Any]:
        if not await is_app_initialized():
            return {"status": "deferred"}

        pairs = await _gather_conflict_pairs(limit)
        if not pairs:
            return {"status": "noop", "processed": 0}

        rcs = RegionalCultureSystem(user_id=0, conversation_id=0)
        await rcs.ensure_initialized()

        results = []
        for n1, n2 in pairs:
            results.append(await _refresh_conflict_pair(rcs, n1, n2, force=force))

        return {"status": "completed", "processed": len(results), "details": results}

    return run_async_in_worker_loop(_run())


@shared_task(name="tasks.warm_user_context_cache_task")
def warm_user_context_cache_task(user_id: int, conversation_id: int):
    """Warm the Nyx user context cache for a given conversation."""

    return run_async_in_worker_loop(
        warm_user_context_cache(
            user_id=user_id,
            conversation_id=conversation_id,
            redis_client=redis_publisher,
        )
    )


# === Nyx SDK lazy singleton ====================================================

_SDK: Optional[NyxAgentSDK] = None
_SDK_LOCK = asyncio.Lock()


async def _get_nyx_sdk() -> NyxAgentSDK:
    global _SDK
    if _SDK is not None:
        return _SDK

    async with _SDK_LOCK:
        if _SDK is not None:
            return _SDK

        sdk_config = NyxSDKConfig(
            request_timeout_seconds=55.0,  # Warmed at worker init; keep headroom for calls
            retry_on_failure=True,
            enable_telemetry=True,
        )
        instance = NyxAgentSDK(sdk_config)
        await instance.initialize_agent()
        _SDK = instance
        return _SDK


@worker_process_init.connect(weak=False)
def warm_nyx_sdk(**kwargs) -> None:
    """Warm the Nyx SDK singleton when a Celery worker process starts."""

    try:
        run_async_in_worker_loop(_get_nyx_sdk())
        logger.info("Nyx SDK warmed during worker init.")
    except Exception:
        logger.warning("Nyx SDK warm-up during worker init failed", exc_info=True)


# === Small DB helpers ==========================================================

async def _get_user_id_for_conversation(conversation_id: int) -> int:
    try:
        async with get_db_connection_context() as conn:
            uid = await conn.fetchval(
                "SELECT user_id FROM conversations WHERE id=$1", conversation_id
            )
            return int(uid) if uid is not None else 0
    except Exception:
        return 0


async def _get_user_conv_for_npc(npc_id: int) -> Tuple[int, int]:
    """
    Try to infer (user_id, conversation_id) for an NPC.
    """
    try:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                """
                SELECT user_id, conversation_id
                  FROM NPCStats
                 WHERE npc_id=$1
                 LIMIT 1
                """,
                npc_id,
            )
            if row:
                return int(row["user_id"]), int(row["conversation_id"])
    except Exception:
        pass
    return (0, 0)


# === Simple test task ==========================================================

@celery_app.task
def test_task():
    logger.info("Executing test task!")
    return "Hello from test task!"


# === Background chat (uses new NyxAgentSDK) ====================================

@celery_app.task(
    bind=True,
    name="tasks.background_chat_task_with_memory",
    # Give the task more headroom. Soft limit gets a graceful TimeoutError we can handle.
    soft_time_limit=180,
    time_limit=210,
    acks_late=True,
)
def background_chat_task_with_memory(
    self,
    conversation_id: int,
    user_input: str,
    user_id: int,
    universal_update: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
):
    """
    Enhanced background chat task that runs the full Nyx Agent pipeline and
    publishes the final result to a Redis channel for the web server to stream.
    
    This function is executed by a Celery worker and is decoupled from the web server process.
    """
    async def run_chat_and_publish():
        """Inner async function to run the core logic."""
        full_response_payload = {}
        with trace(workflow_name="background_chat_task_celery"):
            logger.info(
                "[BG Task %s] Starting for user %s, request_id=%s",
                conversation_id,
                user_id,
                request_id,
            )
            try:
                # === Step-level timeout budget ===
                SOFT_BUDGET = 165.0  # seconds (soft_limit=180 -> leave buffer now that SDK warms earlier)
                STEP_MAX = {
                    "agg": 20.0,
                    "updates": 20.0,
                    "mem": 25.0,
                    "sdk": 70.0,
                }
                t0 = time.time()

                def time_left() -> float:
                    return max(5.0, SOFT_BUDGET - (time.time() - t0))

                # 1) Build aggregator context
                from logic.aggregator_sdk import get_aggregated_roleplay_context

                aggregator_data = await asyncio.wait_for(
                    get_aggregated_roleplay_context(user_id, conversation_id, "Chase"),
                    timeout=min(STEP_MAX["agg"], time_left()),
                )

                # Fetch recent turns, but do NOT treat timeout as fatal
                try:
                    recent_turns = await asyncio.wait_for(
                        _conversation_store.fetch_recent_turns(
                            user_id=user_id,
                            conversation_id=conversation_id,
                        ),
                        timeout=min(8.0, time_left()),
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logger.warning(
                        "[BG Task %s] recent_turns fetch timed out/cancelled; proceeding without history",
                        conversation_id,
                    )
                    recent_turns = []

                context = {
                    "location": aggregator_data.get("currentRoleplay", {}).get("CurrentLocation", "Unknown"),
                    "time_of_day": aggregator_data.get("timeOfDay", "Morning"),
                    "player_name": aggregator_data.get("playerName", "Chase"),
                    "npc_present": aggregator_data.get("npcsPresent", []),
                    "aggregator_data": aggregator_data,
                    "recent_turns": recent_turns,
                }

                # 2) Apply universal updates if provided
                if universal_update:
                    _schedule_universal_updates(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        updates=universal_update,
                        request_id=request_id,
                        source="background_chat_task_with_memory",
                    )

                # 3) Enrich context with memories
                try:
                    from memory.memory_integration import enrich_context_with_memories

                    logger.info("[BG Task %s] Enriching context with memories...", conversation_id)
                    context = await asyncio.wait_for(
                        enrich_context_with_memories(
                            user_id=user_id,
                            conversation_id=conversation_id,
                            user_input=user_input,
                            context=context,
                        ),
                        timeout=min(STEP_MAX["mem"], time_left()),
                    )
                    logger.info("[BG Task %s] Context enriched with memories.", conversation_id)
                except asyncio.TimeoutError:
                    logger.warning(
                        "[BG Task %s] Memory enrichment timed out; continuing without extra memory context",
                        conversation_id,
                    )
                except Exception as memory_err:
                    logger.error(
                        "[BG Task %s] Error enriching context with memories: %s",
                        conversation_id,
                        memory_err,
                        exc_info=True,
                    )
                    # Non-critical; just continue.

                # 4) Invoke Nyx SDK for the main response
                sdk = await _get_nyx_sdk()
                logger.info("[BG Task %s] Processing input with Nyx SDK...", conversation_id)

                nyx_resp = await asyncio.wait_for(
                    sdk.process_user_input(
                        message=user_input,
                        conversation_id=str(conversation_id),
                        user_id=str(user_id),
                        metadata=context,
                    ),
                    timeout=min(STEP_MAX["sdk"], time_left()),
                )

                redacted_resp_repr = _redact_sensitive_text(repr(nyx_resp))
                logger.info(
                    "[BG Task %s] Nyx SDK response resolved (conversation_id=%s request_id=%s): %s",
                    conversation_id,
                    conversation_id,
                    request_id,
                    redacted_resp_repr,
                )

                logger.info("[BG Task %s] Nyx SDK processing complete.", conversation_id)

                if not nyx_resp or not nyx_resp.success:
                    if nyx_resp:
                        error_msg = (
                            nyx_resp.error
                            or (nyx_resp.metadata or {}).get("error")
                            or "Unknown SDK error"
                        )
                    else:
                        error_msg = "Empty SDK response"
                    raise Exception(error_msg)

                # 5) Prepare the final success payload for publishing
                message_content = nyx_resp.narrative.strip() if nyx_resp.narrative else "…"

                full_response_payload = {
                    "conversation_id": conversation_id,
                    "full_text": message_content,
                    "request_id": request_id,
                    "success": True,
                    "metadata": serialize_for_celery(nyx_resp.metadata),
                }

            except Exception as e:
                logger.error(
                    "[BG Task %s] Critical Error in task execution: %s",
                    conversation_id,
                    str(e),
                    exc_info=True,
                )
                # Prepare a structured error payload for publishing
                full_response_payload = {
                    "conversation_id": conversation_id,
                    "request_id": request_id,
                    "success": False,
                    "error": (
                        "I encountered an issue while processing your request. "
                        f"Please try again. (Details: {str(e)})"
                    ),
                }

            finally:
                # 6) ALWAYS publish the result (success or failure) to the Redis channel
                if full_response_payload and redis_publisher:
                    channel = "chat-responses"
                    try:
                        logger.info(
                            "[BG Task %s] Preparing payload for publish (request_id=%s): %r",
                            conversation_id,
                            request_id,
                            full_response_payload,
                        )
                        safe_payload = serialize_for_celery(full_response_payload)
                        payload_json = json.dumps(safe_payload)
                        redis_publisher.publish(channel, payload_json)
                        logger.info(
                            "[BG Task %s] Published result to '%s' for request_id=%s",
                            conversation_id,
                            channel,
                            request_id,
                        )
                    except Exception as pub_err:
                        logger.critical(
                            "[BG Task %s] FAILED to publish result to Redis for request_id=%s. Error: %s",
                            conversation_id,
                            request_id,
                            pub_err,
                            exc_info=True,
                        )
                elif not redis_publisher:
                    logger.critical(
                        "[BG Task] Redis publisher is not available. Cannot publish task result."
                    )

    # Execute the async function within the Celery worker's event loop
    try:
        return run_async_in_worker_loop(run_chat_and_publish())
    except SoftTimeLimitExceeded:
        # Publish a graceful timeout to the client so the UI never hangs on a dead task.
        if redis_publisher:
            try:
                payload = {
                    "conversation_id": conversation_id,
                    "request_id": request_id,
                    "success": False,
                    "error": (
                        "The server is still crunching. I hit a time limit in the "
                        "background; please retry."
                    ),
                    "error_type": "SoftTimeLimitExceeded",
                }
                redis_publisher.publish("chat-responses", json.dumps(payload))
                logger.warning(
                    "[BG Task %s] SoftTimeLimitExceeded – published timeout notice",
                    conversation_id,
                )
            except Exception:
                logger.exception(
                    "[BG Task %s] Failed to publish timeout notice", conversation_id
                )
        # Re-raise so Celery records the timeout
        raise


# === Memory embed/retrieval/analyze ============================================

@celery_app.task
def process_memory_embedding_task(user_id: int, conversation_id: int, message_text: str, entity_type: str = "memory", metadata: Optional[Dict[str, Any]] = None):
    logger.info(f"Processing memory embedding for user {user_id}, conversation {conversation_id}")

    async def process_memory_async():
        from memory.memory_integration import add_memory_from_message
        try:
            memory_id = await add_memory_from_message(
                user_id=user_id,
                conversation_id=conversation_id,
                message_text=message_text,
                entity_type=entity_type,
                metadata=metadata
            )
            return {"success": True, "memory_id": memory_id, "message": f"Processed memory for user {user_id}, conversation {conversation_id}"}
        except Exception as e:
            logger.error(f"Error processing memory: {e}")
            return {"success": False, "error": str(e)}

    return run_async_in_worker_loop(process_memory_async())


@celery_app.task
def retrieve_memories_task(user_id: int, conversation_id: int, query_text: str, entity_types: Optional[List[str]] = None, top_k: int = 5):
    logger.info(f"Retrieving memories for user {user_id}, conversation {conversation_id}, query: {query_text[:50]}...")

    async def retrieve_memories_async():
        from memory.memory_integration import retrieve_relevant_memories
        try:
            memories = await retrieve_relevant_memories(
                user_id=user_id,
                conversation_id=conversation_id,
                query_text=query_text,
                entity_types=entity_types,
                top_k=top_k
            )
            return {"success": True, "memories": memories, "message": f"Retrieved {len(memories)} memories"}
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return {"success": False, "memories": [], "error": str(e)}

    return run_async_in_worker_loop(retrieve_memories_async())


@celery_app.task
def analyze_with_memory_task(user_id: int, conversation_id: int, query_text: str, entity_types: Optional[List[str]] = None, top_k: int = 5):
    logger.info(f"Analyzing query with memories for user {user_id}, conversation {conversation_id}, query: {query_text[:50]}...")

    async def analyze_with_memory_async():
        from memory.memory_integration import analyze_with_memory
        try:
            result = await analyze_with_memory(
                user_id=user_id,
                conversation_id=conversation_id,
                query_text=query_text,
                entity_types=entity_types,
                top_k=top_k
            )
            return {"success": True, "result": result, "message": "Analyzed query with memories"}
        except Exception as e:
            logger.error(f"Error analyzing with memories: {e}")
            return {"success": False, "result": None, "error": str(e)}

    return run_async_in_worker_loop(analyze_with_memory_async())


@celery_app.task
def memory_maintenance_task():
    """Periodic memory maintenance."""
    logger.info("Running memory system maintenance task")

    async def run_maintenance():
        try:
            from memory.maintenance import MemoryMaintenance
            maintenance = MemoryMaintenance()
            should_run = await maintenance.should_run_cleanup()
            if should_run:
                cleanup_stats = await maintenance.cleanup_old_memories()
                logger.info(f"Memory cleanup completed: {cleanup_stats}")
                maintenance.last_cleanup = datetime.datetime.now()

                from memory.memory_integration import cleanup_memory_services, cleanup_memory_retrievers
                await cleanup_memory_services()
                await cleanup_memory_retrievers()

                return {"success": True, "message": "Memory system maintenance completed", "cleanup_stats": cleanup_stats, "cleanup_performed": True}
            else:
                logger.info("Skipping memory cleanup - conditions not met")
                return {"success": True, "message": "Maintenance checked - cleanup not needed", "cleanup_performed": False}
        except Exception as e:
            logger.error(f"Error during memory maintenance: {e}", exc_info=True)
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    return run_async_in_worker_loop(run_maintenance())


# === NPC learning cycle ========================================================

@celery_app.task
def run_npc_learning_cycle_task():
    """Periodic NPC learning cycle for active conversations."""
    logger.info("Starting NPC learning cycle task via Celery Beat.")

    async def run_learning_cycle():
        with trace(workflow_name="npc_learning_cycle_celery"):
            processed_conversations = 0
            try:
                async with get_db_connection_context() as conn:
                    convs = await conn.fetch(
                        """
                        SELECT id, user_id
                          FROM conversations
                         WHERE last_active > NOW() - INTERVAL '1 day'
                        """
                    )
                    if not convs:
                        logger.info("No recent conversations found for NPC learning.")
                        return {"status": "success", "processed_conversations": 0}

                    for conv_row in convs:
                        conv_id = conv_row["id"]
                        user_id = conv_row["user_id"]
                        try:
                            npc_rows = await conn.fetch(
                                """
                                SELECT npc_id FROM NPCStats
                                 WHERE user_id=$1 AND conversation_id=$2
                                """,
                                user_id,
                                conv_id,
                            )
                            npc_ids = [row["npc_id"] for row in npc_rows]

                            if npc_ids:
                                manager = NPCLearningManager(user_id, conv_id)
                                await manager.initialize()
                                await manager.run_regular_adaptation_cycle(npc_ids)
                                logger.info(f"Learning cycle completed for conversation {conv_id}: {len(npc_ids)} NPCs")
                                processed_conversations += 1
                            else:
                                logger.info(f"No NPCs found for learning cycle in conversation {conv_id}.")
                        except Exception as e_inner:
                            logger.error(f"Error in NPC learning cycle for conv {conv_id}: {e_inner}", exc_info=True)

            except Exception as e_outer:
                logger.error(f"Critical error in NPC learning scheduler task: {e_outer}", exc_info=True)
                return {"status": "error", "message": str(e_outer)}

            logger.info(f"NPC learning cycle task finished. Processed {processed_conversations} conversations.")
            return {"status": "success", "processed_conversations": processed_conversations}

    return run_async_in_worker_loop(run_learning_cycle())


# === New game creation =========================================================

@celery_app.task
def process_new_game_task(user_id: int, conversation_data: Dict[str, Any]):
    """Process new (or preset) game creation - completes ALL operations before returning."""
    logger.info("CELERY – process_new_game_task called")

    async def run_new_game():
        with trace(workflow_name="process_new_game_celery_task"):
            try:
                logger.info("[NG] payload keys=%s, preset_id=%s",
                            list(conversation_data.keys()) if isinstance(conversation_data, dict) else type(conversation_data),
                            (conversation_data or {}).get("preset_story_id"))
                
                # Validate user_id
                try:
                    user_id_int = int(user_id)
                    logger.info(f"[NG] Validated user_id: {user_id_int}")
                except Exception:
                    logger.error(f"Invalid user_id: {user_id}")
                    return {"status": "failed", "error": "Invalid user_id"}

                # Validate conversation_data
                if not isinstance(conversation_data, dict):
                    logger.error("conversation_data is not a dict")
                    return {"status": "failed", "error": "Invalid conversation_data"}

                # Validate conversation_id if provided
                conv_id = conversation_data.get("conversation_id")
                if conv_id is not None:
                    try:
                        conversation_data["conversation_id"] = int(conv_id)
                        logger.info(f"[NG] Validated conversation_id: {conversation_data['conversation_id']}")
                    except Exception:
                        logger.error(f"Invalid conversation_id: {conv_id}")
                        return {"status": "failed", "error": "Invalid conversation_id"}

                preset_story_id = get_preset_id(conversation_data)
                logger.info(f"[NG] Preset story ID: {preset_story_id or 'None (dynamic)'}")

                # Create agent and context
                agent = NewGameAgent()
                from lore.core.context import CanonicalContext
                ctx = CanonicalContext(user_id_int, conversation_data.get("conversation_id", 0))

                try:
                    # Process game creation with timeout
                    # The agent now handles EVERYTHING including finalization
                    logger.info(f"[NG] Starting complete game creation (timeout: 600s)")
                    
                    if preset_story_id:
                        logger.info(f"[NG] Processing preset story: {preset_story_id}")
                        result = await asyncio.wait_for(
                            agent.process_preset_game_direct(ctx, conversation_data, preset_story_id),
                            timeout=600.0
                        )
                    else:
                        logger.info(f"[NG] Processing dynamic game creation")
                        result = await asyncio.wait_for(
                            agent.process_new_game(ctx, conversation_data),
                            timeout=600.0
                        )
                    
                    logger.info(f"[NG] Agent processing complete, result type: {type(result)}")

                    # Extract conversation_id for verification
                    def _get(attr, default=None):
                        return getattr(result, attr, default) if hasattr(result, attr) else result.get(attr, default) if isinstance(result, dict) else default

                    conv_id_final = conversation_data.get("conversation_id") or _get("conversation_id")
                    
                    if conv_id_final is None:
                        logger.warning("[NG POST] No conversation_id found after pipeline")
                        return serialize_for_celery(result)
                    
                    # Verify the conversation is actually ready
                    try:
                        async with get_db_connection_context(timeout=5.0) as conn:
                            status = await conn.fetchval(
                                "SELECT status FROM conversations WHERE id=$1 AND user_id=$2",
                                conv_id_final,
                                user_id_int
                            )
                            
                            if status != 'ready':
                                logger.warning(
                                    f"[NG POST] Conversation {conv_id_final} status is '{status}', "
                                    f"expected 'ready'. Agent may not have completed properly."
                                )
                            else:
                                logger.info(f"[NG POST] Verified conversation {conv_id_final} is ready")
                    except Exception as verify_err:
                        logger.error(f"[NG POST] Failed to verify conversation status: {verify_err}")

                    logger.info(f"[NG COMPLETE] Task successfully completed for conversation {conv_id_final}")
                    return serialize_for_celery(result)

                except asyncio.TimeoutError:
                    logger.exception("[NG TIMEOUT] Game creation timed out after 600 seconds")
                    
                    conv_id_fail = conversation_data.get("conversation_id")
                    if conv_id_fail:
                        try:
                            async with get_db_connection_context() as conn:
                                await conn.execute(
                                    """
                                    UPDATE conversations
                                       SET status='failed',
                                           conversation_name='New Game - Creation Timeout'
                                     WHERE id=$1 AND user_id=$2
                                    """,
                                    conv_id_fail,
                                    user_id_int,
                                )
                        except Exception as update_err:
                            logger.error(f"Failed to update conversation status: {update_err}")
                    
                    return {
                        "status": "failed",
                        "error": "Game creation timed out after 10 minutes",
                        "error_type": "TimeoutError",
                        "conversation_id": conv_id_fail,
                    }

                except Exception as e:
                    logger.exception("[NG ERROR] Critical error in process_new_game_task")

                    conv_id_fail = conversation_data.get("conversation_id")
                    if conv_id_fail:
                        try:
                            async with get_db_connection_context() as conn:
                                await conn.execute(
                                    """
                                    UPDATE conversations
                                       SET status='failed',
                                           conversation_name='New Game - Task Failed'
                                     WHERE id=$1 AND user_id=$2
                                    """,
                                    conv_id_fail,
                                    user_id_int,
                                )
                        except Exception as update_err:
                            logger.error(f"Failed to update conversation status: {update_err}")

                    return {
                        "status": "failed",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "conversation_id": conv_id_fail,
                    }

            except Exception as outer_e:
                logger.exception("[NG OUTER ERROR] Outer exception in run_new_game wrapper")
                return {
                    "status": "failed",
                    "error": str(outer_e),
                    "error_type": type(outer_e).__name__,
                }

    return run_async_in_worker_loop(run_new_game())

@celery_app.task(name="tasks.update_edge_case_scan", max_retries=1)
def update_edge_case_scan(user_id: int, conversation_id: int):
    """
    Celery task to run the full edge case scan in the background and cache the results.
    This is the core background job for the Edge Case subsystem.
    """
    logger.info(f"Starting background edge case scan for user={user_id}, conv={conversation_id}")

    async def do_scan():
        from logic.conflict_system.conflict_synthesizer import get_synthesizer, SubsystemType
        
        synthesizer = await get_synthesizer(user_id, conversation_id)
        edge_handler = synthesizer._subsystems.get(SubsystemType.EDGE_HANDLER)
        
        if edge_handler:
            await edge_handler.perform_full_scan_and_cache()
            return {"status": "success", "message": "Edge case scan completed and cached."}
        else:
            logger.error(f"Could not find Edge Case Subsystem for user={user_id}, conv={conversation_id}")
            return {"status": "error", "message": "Edge Case Subsystem not found."}

    try:
        result = run_async_in_worker_loop(do_scan())
        logger.info(f"Finished edge case scan for user={user_id}, conv={conversation_id}. Result: {result}")
        return result
    except Exception as e:
        logger.exception(f"Critical error in update_edge_case_scan task for user={user_id}")
        return {"status": "error", "error": str(e)}


@celery_app.task(name="tasks.update_tension_bundle_cache", max_retries=1)
def update_tension_bundle_cache(user_id: int, conversation_id: int, scene_context: Dict[str, Any]):
    """
    Celery task to generate and cache the full tension bundle for a specific scene.
    This is the core background job for the Tension subsystem's performance fix.
    """
    logger.info(f"Starting background tension bundle generation for user={user_id}, conv={conversation_id}")

    async def do_generation():
        from logic.conflict_system.conflict_synthesizer import get_synthesizer, SubsystemType
        
        synthesizer = await get_synthesizer(user_id, conversation_id)
        tension_system = synthesizer._subsystems.get(SubsystemType.TENSION)
        
        if tension_system:
            await tension_system.perform_bundle_generation_and_cache(scene_context)
            return {"status": "success", "message": "Tension bundle generated and cached."}
        else:
            logger.error(f"Could not find Tension Subsystem for user={user_id}, conv={conversation_id}")
            return {"status": "error", "message": "Tension Subsystem not found."}

    try:
        result = run_async_in_worker_loop(do_generation())
        logger.info(f"Finished tension bundle generation for user={user_id}, conv={conversation_id}. Result: {result}")
        return result
    except Exception as e:
        logger.exception(f"Critical error in update_tension_bundle_cache task for user={user_id}")
        return {"status": "error", "error": str(e)}


@celery_app.task(name="tasks.periodic_edge_case_maintenance")
def periodic_edge_case_maintenance():
    """
    Periodic task (for Celery Beat) to trigger edge case scans for recently
    active conversations, ensuring system health is monitored proactively.
    """
    logger.info("Starting periodic edge case maintenance.")

    async def do_maintenance():
        processed_count = 0
        try:
            async with get_db_connection_context() as conn:
                # Find conversations that have had recent activity
                conversations = await conn.fetch("""
                    SELECT DISTINCT user_id, id as conversation_id
                    FROM conversations
                    WHERE last_active > NOW() - INTERVAL '24 hours'
                    LIMIT 100
                """)
            
            if not conversations:
                logger.info("No active conversations found for edge case maintenance.")
                return {"status": "success", "triggered_scans": 0}
            
            for conv in conversations:
                try:
                    # Re-use the main task to avoid duplicating logic
                    update_edge_case_scan.delay(conv['user_id'], conv['conversation_id'])
                    processed_count += 1
                    await asyncio.sleep(0.5) # Stagger tasks slightly
                except Exception as e:
                    logger.error(f"Failed to trigger scan for {conv['user_id']}:{conv['conversation_id']}: {e}")
            
            return {"status": "success", "triggered_scans": processed_count}
        except Exception as e:
            logger.error(f"Periodic edge case maintenance failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    return run_async_in_worker_loop(do_maintenance())

# Signal handling for revoked new game tasks to avoid leaving conversations stuck
def _handle_process_new_game_task_revoked(
    sender=None,
    request=None,
    terminated=None,
    signum=None,
    expired=None,
    **kwargs,
):
    """Mark conversations as failed if the new game task is revoked."""
    try:
        expected_name = getattr(
            process_new_game_task, "name", "tasks.process_new_game_task"
        )
        task_name = (
            getattr(request, "name", None)
            or getattr(request, "task", None)
            or getattr(sender, "name", None)
            or sender
        )

        if task_name != expected_name:
            return

        args = list(getattr(request, "args", ()) or ())
        kwargs_data = dict(getattr(request, "kwargs", {}) or {})

        user_id = args[0] if len(args) >= 1 else kwargs_data.get("user_id")
        conversation_payload = args[1] if len(args) >= 2 else kwargs_data.get(
            "conversation_data"
        )

        conversation_id = None
        if isinstance(conversation_payload, dict):
            conversation_id = conversation_payload.get("conversation_id")

        if conversation_id is None:
            conversation_id = kwargs_data.get("conversation_id")

        if conversation_id is None:
            logger.warning(
                "task_revoked received for process_new_game_task without conversation_id"
            )
            return

        try:
            conversation_id = int(conversation_id)
        except (TypeError, ValueError):
            logger.warning(
                "task_revoked received invalid conversation_id: %s", conversation_id
            )
            return

        if user_id is not None:
            try:
                user_id = int(user_id)
            except (TypeError, ValueError):
                logger.warning(
                    "task_revoked received invalid user_id: %s", user_id
                )
                user_id = None

        async def _mark_failed():
            async with get_db_connection_context() as conn:
                if user_id is not None:
                    await conn.execute(
                        """
                        UPDATE conversations
                           SET status='failed',
                               conversation_name='New Game - Task Failed'
                         WHERE id=$1 AND user_id=$2
                        """,
                        conversation_id,
                        user_id,
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE conversations
                           SET status='failed',
                               conversation_name='New Game - Task Failed'
                         WHERE id=$1
                        """,
                        conversation_id,
                    )

        run_async_in_worker_loop(_mark_failed())
    except Exception:
        logger.exception(
            "Failed to update conversation after process_new_game_task revoke"
        )


# Register signal handler eagerly so workers always apply failure state
task_revoked.connect(_handle_process_new_game_task_revoked, weak=False)


# === NPC creation ===============================================================

@celery_app.task
def create_npcs_task(user_id: int, conversation_id: int, count: int = 10):
    """Create NPCs asynchronously."""
    logger.info(f"Starting create_npcs_task for user={user_id}, conv={conversation_id}, count={count}")

    async def create_npcs_async():
        async with get_db_connection_context() as conn:
            # Environment description
            row_env = await conn.fetchrow(
                """
                SELECT value
                  FROM CurrentRoleplay
                 WHERE user_id=$1 AND conversation_id=$2 AND key='EnvironmentDesc'
                """,
                user_id,
                conversation_id,
            )
            environment_desc = row_env["value"] if row_env else "A fallback environment description"
            logger.debug(f"Fetched environment_desc: {environment_desc[:100]}...")

            # Calendar day names
            row_cal = await conn.fetchrow(
                """
                SELECT value
                  FROM CurrentRoleplay
                 WHERE user_id=$1 AND conversation_id=$2 AND key='CalendarNames'
                """,
                user_id,
                conversation_id,
            )
            if row_cal and row_cal["value"]:
                try:
                    cal_data = json.loads(row_cal["value"])
                    day_names = cal_data.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode CalendarNames JSON for conv {conversation_id}, using defaults.")
                    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            else:
                logger.info(f"No CalendarNames found for conv {conversation_id}, using defaults.")
                day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            logger.debug(f"Using day_names: {day_names}")

            from npcs.npc_creation import spawn_multiple_npcs
            npc_ids = await spawn_multiple_npcs(
                user_id=user_id,
                conversation_id=conversation_id,
                environment_desc=environment_desc,
                day_names=day_names,
                count=count
            )

            return {
                "message": f"Successfully created {len(npc_ids) if npc_ids else 0} NPCs (via Nyx governance)",
                "npc_ids": npc_ids or [],
            }

    try:
        final_info = run_async_in_worker_loop(create_npcs_async())
        logger.info(f"Finished create_npcs_task successfully for user={user_id}, conv={conversation_id}. NPCs: {final_info.get('npc_ids')}")
        return final_info
    except Exception as e:
        logger.exception(f"Error in create_npcs_task for user={user_id}, conv={conversation_id}")
        return {"status": "failed", "error": str(e)}


@celery_app.task
def ensure_npc_pool_task(user_id: int, conversation_id: int, target_count: int = 5, source: Optional[str] = None):
    """Ensure at least ``target_count`` unintroduced NPCs exist for the conversation."""

    logger.info(
        "Ensuring NPC pool for user=%s, conversation=%s, target=%s", user_id, conversation_id, target_count
    )

    async def ensure_pool_async():
        async with get_db_connection_context() as conn:
            existing = await conn.fetchval(
                """
                SELECT COUNT(*)
                  FROM NPCStats
                 WHERE user_id=$1
                   AND conversation_id=$2
                   AND COALESCE(introduced, FALSE) = FALSE
                """,
                user_id,
                conversation_id,
            )

            payload = {
                "status": "in_progress",
                "target": target_count,
                "existing": existing,
                "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
            }
            if source:
                payload["source"] = source

            await conn.execute(
                """
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'NPCPoolStatus', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = EXCLUDED.value
                """,
                user_id,
                conversation_id,
                json.dumps(payload),
            )

        needed = max(int(target_count) - int(existing), 0)
        npc_ids: List[int] = []

        if needed > 0:
            logger.info(
                "NPC pool short by %s characters for conversation %s. Spawning via handler...",
                needed,
                conversation_id,
            )

            from npcs.new_npc_creation import NPCCreationHandler

            handler = NPCCreationHandler()
            ctx = type("NPCContext", (), {
                "context": {
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                }
            })()
            npc_ids = await handler.spawn_multiple_npcs(ctx, count=needed)

        async with get_db_connection_context() as conn:
            final_count = await conn.fetchval(
                """
                SELECT COUNT(*)
                  FROM NPCStats
                 WHERE user_id=$1
                   AND conversation_id=$2
                   AND COALESCE(introduced, FALSE) = FALSE
                """,
                user_id,
                conversation_id,
            )

            payload = {
                "status": "ready" if final_count >= target_count else "partial",
                "target": target_count,
                "created": len(npc_ids),
                "final_count": final_count,
                "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
            }
            if source:
                payload["source"] = source

            await conn.execute(
                """
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'NPCPoolStatus', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = EXCLUDED.value
                """,
                user_id,
                conversation_id,
                json.dumps(payload),
            )

        return {
            "status": payload["status"],
            "created": len(npc_ids),
            "final_count": final_count,
            "npc_ids": npc_ids,
        }

    try:
        result = run_async_in_worker_loop(ensure_pool_async())
        logger.info(
            "NPC pool ensure task complete for conversation %s. Status=%s, final_count=%s",
            conversation_id,
            result.get("status"),
            result.get("final_count"),
        )
        return result
    except Exception as exc:
        logger.exception(
            "Error ensuring NPC pool for user=%s conversation=%s", user_id, conversation_id
        )

        async def mark_failure():
            failure_payload = {
                "status": "failed",
                "target": target_count,
                "error": str(exc),
                "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
            }
            if source:
                failure_payload["source"] = source

            async with get_db_connection_context() as conn:
                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'NPCPoolStatus', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    json.dumps(failure_payload),
                )

        try:
            run_async_in_worker_loop(mark_failure())
        except Exception:
            logger.exception("Failed to record NPC pool failure state for conversation %s", conversation_id)

        return {"status": "failed", "error": str(exc)}


@celery_app.task
def generate_lore_background_task(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """Generate lore asynchronously so the new-game flow can finish quickly."""

    logger.info(
        "Starting background lore generation for user=%s conversation=%s",
        user_id,
        conversation_id,
    )

    async def run_lore_generation():
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        status_payload = {
            "status": "in_progress",
            "updated_at": timestamp,
        }

        try:
            async with get_db_connection_context() as conn:
                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'LoreGenerationStatus', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    json.dumps(status_payload),
                )
        except Exception:
            logger.exception(
                "Failed to mark lore generation in progress for conversation %s",
                conversation_id,
            )

        try:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT value FROM CurrentRoleplay
                    WHERE user_id=$1 AND conversation_id=$2 AND key='EnvironmentDesc'
                    """,
                    user_id,
                    conversation_id,
                )
                environment_desc = row["value"] if row else "A mysterious environment with hidden layers of complexity."

                npc_rows = await conn.fetch(
                    """
                    SELECT npc_id FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2
                    """,
                    user_id,
                    conversation_id,
                )
                npc_ids = [r["npc_id"] for r in npc_rows] if npc_rows else []

            from lore.core.lore_system import LoreSystem

            lore_system = await LoreSystem.get_instance(user_id, conversation_id)
            base_context = {
                "user_id": user_id,
                "conversation_id": conversation_id,
            }
            lore_ctx = RunContextWrapper(context=base_context)
            lore_ctx.user_id = user_id
            lore_ctx.conversation_id = conversation_id

            lore_result = await lore_system.generate_complete_lore(
                lore_ctx, environment_desc
            )

            if npc_ids:
                logger.info(
                    "Integrating generated lore with %s NPCs for conversation %s",
                    len(npc_ids),
                    conversation_id,
                )
                for npc_id in npc_ids:
                    faction_affiliations: List[str] = []
                    async with get_db_connection_context() as conn:
                        npc_row = await conn.fetchrow(
                            """
                            SELECT affiliations
                            FROM NPCStats
                            WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
                            """,
                            npc_id,
                            user_id,
                            conversation_id,
                        )
                        if npc_row and npc_row["affiliations"]:
                            affiliations_data = npc_row["affiliations"]
                            if isinstance(affiliations_data, str):
                                try:
                                    affiliations_data = json.loads(affiliations_data)
                                except json.JSONDecodeError:
                                    affiliations_data = []
                            if isinstance(affiliations_data, list):
                                faction_affiliations = affiliations_data

                    npc_context = dict(base_context)
                    npc_context["npc_id"] = npc_id
                    npc_ctx = RunContextWrapper(context=npc_context)
                    npc_ctx.user_id = user_id
                    npc_ctx.conversation_id = conversation_id
                    npc_ctx.npc_id = npc_id

                    await lore_system.initialize_npc_lore_knowledge(
                        npc_ctx,
                        npc_id,
                        cultural_background="common",
                        faction_affiliations=faction_affiliations,
                    )

            factions = len(lore_result.get("factions", []))
            cultural = len(lore_result.get("cultural_elements", []))
            locations = len(lore_result.get("locations", []))
            summary = (
                f"Generated {factions} factions, {cultural} cultural elements, and {locations} locations"
            )

            completion_payload = {
                "status": "completed",
                "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
                "factions": factions,
                "cultural_elements": cultural,
                "locations": locations,
            }

            async with get_db_connection_context() as conn:
                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'LoreSummary', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    summary,
                )

                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'LoreGenerationStatus', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    json.dumps(completion_payload),
                )

            logger.info(
                "Background lore generation completed for conversation %s",
                conversation_id,
            )

            return {
                "status": "completed",
                "lore_summary": summary,
                "factions": factions,
                "cultural_elements": cultural,
                "locations": locations,
            }

        except Exception as exc:  # pragma: no cover - defensive safeguard
            logger.exception(
                "Background lore generation failed for conversation %s", conversation_id
            )

            failure_summary = f"Failed to generate lore: {exc}"
            failure_payload = {
                "status": "failed",
                "failed_at": datetime.datetime.utcnow().isoformat() + "Z",
                "error": str(exc),
            }

            async with get_db_connection_context() as conn:
                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'LoreSummary', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    failure_summary,
                )

                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'LoreGenerationStatus', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    json.dumps(failure_payload),
                )

            return {
                "status": "failed",
                "error": str(exc),
            }

    return run_async_in_worker_loop(run_lore_generation())

@shared_task(bind=True, max_retries=2, default_retry_delay=90)
def update_scene_conflict_context(self, user_id: int, conversation_id: int, scene_info: Dict[str, Any], cache_key: str):
    """
    Celery task to generate a full, rich conflict context for a scene in the background.
    """
    try:
        logger.info(f"Starting background conflict context generation for key: {cache_key}")

        async def _generate_context():
            from logic.conflict_system.conflict_synthesizer import get_synthesizer
            from logic.conflict_system.signal_tasks import generate_scene_conflict_context

            synthesizer = await get_synthesizer(user_id, conversation_id)

            return await generate_scene_conflict_context(
                synthesizer,
                user_id,
                conversation_id,
                scene_info,
            )

        full_context = run_async_in_worker_loop(_generate_context())

        try:
            client = get_redis_client()
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "Redis client unavailable while caching conflict context for key %s: %s",
                cache_key,
                exc,
            )
        else:
            try:
                client.set(cache_key, json.dumps(full_context), ex=600)
            except (redis.RedisError, TypeError, ValueError) as exc:
                logger.warning(
                    "Failed to cache conflict context for key %s in Redis: %s",
                    cache_key,
                    exc,
                )
            else:
                logger.info(
                    f"Successfully cached full conflict context for key: {cache_key}"
                )

    except Exception as exc:
        logger.error(f"Background conflict context generation failed for key {cache_key}: {exc}", exc_info=True)
        raise self.retry(exc=exc)


@celery_app.task
def generate_initial_conflict_task(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """Generate the initial conflict outside the critical path."""

    logger.info(
        "Starting background conflict generation for user=%s conversation=%s",
        user_id,
        conversation_id,
    )

    async def run_conflict_generation():
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        status_payload = {
            "status": "in_progress",
            "updated_at": timestamp,
        }

        try:
            async with get_db_connection_context() as conn:
                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'InitialConflictStatus', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    json.dumps(status_payload),
                )
        except Exception:
            logger.exception(
                "Failed to mark conflict generation in progress for conversation %s",
                conversation_id,
            )

        try:
            async with get_db_connection_context() as conn:
                npc_count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2
                    """,
                    user_id,
                    conversation_id,
                )

            npc_count = int(npc_count or 0)

            if npc_count < 3:
                summary = "No initial conflict - insufficient NPCs"
                skip_payload = {
                    "status": "skipped",
                    "npc_count": npc_count,
                    "reason": "insufficient_npcs",
                    "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
                }

                async with get_db_connection_context() as conn:
                    await conn.execute(
                        """
                        INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                        VALUES ($1, $2, 'InitialConflictSummary', $3)
                        ON CONFLICT (user_id, conversation_id, key)
                        DO UPDATE SET value = EXCLUDED.value
                        """,
                        user_id,
                        conversation_id,
                        summary,
                    )

                    await conn.execute(
                        """
                        INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                        VALUES ($1, $2, 'InitialConflictStatus', $3)
                        ON CONFLICT (user_id, conversation_id, key)
                        DO UPDATE SET value = EXCLUDED.value
                        """,
                        user_id,
                        conversation_id,
                        json.dumps(skip_payload),
                    )

                logger.info(
                    "Skipping conflict generation for conversation %s due to insufficient NPCs",
                    conversation_id,
                )

                return {
                    "status": "skipped",
                    "reason": "insufficient_npcs",
                    "npc_count": npc_count,
                }

            from logic.conflict_system.conflict_integration import ConflictSystemIntegration

            conflict_ctx = RunContextWrapper(context={
                "user_id": user_id,
                "conversation_id": conversation_id,
            })
            conflict_ctx.user_id = user_id
            conflict_ctx.conversation_id = conversation_id

            conflict_integration = await ConflictSystemIntegration.get_instance(
                user_id,
                conversation_id,
            )
            await conflict_integration.initialize()

            initial_conflict = await conflict_integration.generate_conflict(
                conflict_ctx,
                {
                    "conflict_type": "major",
                    "intensity": "medium",
                    "player_involvement": "indirect",
                },
            )

            # ADD THIS DEBUG LOGGING
            logger.info(f"[CONFLICT DEBUG] Full response: {json.dumps(initial_conflict, default=str, indent=2)}")
            logger.info(f"[CONFLICT DEBUG] Response keys: {list(initial_conflict.keys()) if isinstance(initial_conflict, dict) else 'Not a dict'}")

            if initial_conflict is None:
                summary = "No initial conflict - generation returned None"
            elif not isinstance(initial_conflict, dict):
                summary = "No initial conflict - invalid response type"
            elif not initial_conflict.get("success", False):
                summary = f"No initial conflict - {initial_conflict.get('message', 'Unknown error')}"
            else:
                raw_result = initial_conflict.get("raw_result", {})
                resolved_name: Optional[str] = None
                logger.info(f"[CONFLICT DEBUG] raw_result keys: {list(raw_result.keys()) if isinstance(raw_result, dict) else 'Not a dict'}")
                logger.info(f"[CONFLICT DEBUG] raw_result content: {json.dumps(raw_result, default=str, indent=2)}")

                if isinstance(raw_result, dict):
                    conflict_name = raw_result.get("conflict_name")
                    if isinstance(conflict_name, str) and conflict_name.strip():
                        resolved_name = conflict_name.strip()

                if not resolved_name:
                    conflict_details = initial_conflict.get("conflict_details")
                    if conflict_details and isinstance(conflict_details, dict):
                        conflict_name = conflict_details.get("name")
                        if isinstance(conflict_name, str) and conflict_name.strip():
                            resolved_name = conflict_name.strip()

                if not resolved_name:
                    conflict_id = initial_conflict.get("conflict_id")
                    if conflict_id:
                        fetched_name: Optional[str] = None
                        try:
                            async with get_db_connection_context() as conn:
                                fetched_name = await conn.fetchval(
                                    """
                                    SELECT conflict_name
                                    FROM Conflicts
                                    WHERE id=$1
                                    """,
                                    conflict_id,
                                )
                        except Exception:  # pragma: no cover - defensive safeguard
                            logger.exception(
                                "Failed to fetch conflict name for conflict_id=%s", conflict_id
                            )

                        if isinstance(fetched_name, str):
                            normalized_name = " ".join(fetched_name.split())
                            if normalized_name:
                                resolved_name = normalized_name

                summary = resolved_name or "Unnamed Conflict"

            success = not summary.startswith("No initial conflict -")
            completion_payload = {
                "status": "completed" if success else "completed_no_conflict",
                "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
                "npc_count": npc_count,
            }

            async with get_db_connection_context() as conn:
                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'InitialConflictSummary', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    summary,
                )

                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'InitialConflictStatus', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    json.dumps(completion_payload),
                )

            logger.info(
                "Background conflict generation completed for conversation %s",
                conversation_id,
            )

            return {
                "status": completion_payload["status"],
                "initial_conflict": summary,
            }

        except Exception as exc:  # pragma: no cover - defensive safeguard
            logger.exception(
                "Background conflict generation failed for conversation %s",
                conversation_id,
            )

            failure_summary = f"No initial conflict - exception occurred ({exc})"
            failure_payload = {
                "status": "failed",
                "failed_at": datetime.datetime.utcnow().isoformat() + "Z",
                "error": str(exc),
            }

            async with get_db_connection_context() as conn:
                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'InitialConflictSummary', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    failure_summary,
                )

                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'InitialConflictStatus', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    json.dumps(failure_payload),
                )

            return {
                "status": "failed",
                "error": str(exc),
            }

    return run_async_in_worker_loop(run_conflict_generation())


# === GPT opening line ===========================================================

@celery_app.task
def get_gpt_opening_line_task(conversation_id: int, aggregator_text: str, opening_user_prompt: str):
    """Generate GPT opening line. Synchronous because it calls a sync SDK client."""
    logger.info(f"Async GPT task: Calling GPT for opening line for conv_id={conversation_id}.")

    gpt_reply_dict = get_chatgpt_response(
        conversation_id=conversation_id,
        aggregator_text=aggregator_text,
        user_input=opening_user_prompt
    )

    nyx_text = None
    if isinstance(gpt_reply_dict, dict):
        nyx_text = gpt_reply_dict.get("response")
        if gpt_reply_dict.get("type") == "function_call" or not nyx_text:
            nyx_text = None
    else:
        logger.error(f"get_chatgpt_response returned unexpected type: {type(gpt_reply_dict)}")
        gpt_reply_dict = {}

    if nyx_text is None:
        logger.warning("Async GPT task: GPT returned function call, no text, or error. Retrying without function calls.")
        try:
            client = get_openai_client()
            forced_messages = [
                {"role": "system", "content": aggregator_text},
                {"role": "user", "content": "No function calls. Produce only a text narrative.\n\n" + opening_user_prompt}
            ]
            fallback_response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
                messages=forced_messages,
                temperature=0.7,
            )
            fallback_text = fallback_response.choices[0].message.content.strip()
            nyx_text = fallback_text if fallback_text else "[No text returned from fallback GPT]"
            gpt_reply_dict["response"] = nyx_text
            gpt_reply_dict["type"] = "fallback"
        except Exception as e:
            logger.exception("Error during GPT fallback call.")
            gpt_reply_dict["response"] = "[Error during fallback GPT call]"
            gpt_reply_dict["type"] = "error"
            gpt_reply_dict["error"] = str(e)

    try:
        result_json = json.dumps(gpt_reply_dict)
        logger.info(f"GPT opening line task completed for conv_id={conversation_id}.")
        return result_json
    except (TypeError, OverflowError) as e:
        logger.error(f"Failed to serialize GPT response to JSON: {e}. Response: {gpt_reply_dict}")
        return json.dumps({"status": "error", "message": "Failed to serialize GPT response", "original_response_type": str(type(gpt_reply_dict))})


# === Nyx memory maintenance (governed) =========================================

@celery_app.task
def nyx_memory_maintenance_task():
    """Run Nyx memory maintenance for recent conversations."""
    logger.info("Starting Nyx memory maintenance task (via governance)")

    async def process_all_conversations():
        with trace(workflow_name="nyx_memory_maintenance_celery"):
            processed_count = 0
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(
                    """
                    SELECT DISTINCT user_id, conversation_id
                      FROM NyxMemories
                     WHERE is_archived = FALSE
                       AND timestamp > NOW() - INTERVAL '30 days'
                    """
                )
                if not rows:
                    logger.info("No conversations found with recent Nyx memories to maintain")
                    return {"status": "success", "conversations_processed": 0}

                for row in rows:
                    user_id = row["user_id"]
                    conversation_id = row["conversation_id"]
                    try:
                        await run_maintenance_through_nyx(
                            user_id=user_id,
                            conversation_id=conversation_id,
                            entity_type="nyx",
                            entity_id=0
                        )
                        processed_count += 1
                        logger.info(f"Completed governed memory maintenance for user={user_id}, conv={conversation_id}")
                    except Exception as e:
                        logger.error(f"Governed maintenance error user={user_id}, conv={conversation_id}: {e}", exc_info=True)
                    await asyncio.sleep(0.1)

                return {"status": "success", "conversations_processed": processed_count}

    try:
        result = run_async_in_worker_loop(process_all_conversations())
        logger.info(f"Nyx memory maintenance task completed: {result}")
        return result
    except Exception as e:
        logger.exception("Critical error in nyx_memory_maintenance_task")
        return {"status": "error", "error": str(e)}


# === Performance monitoring / aggregation / cleanup ============================

@celery_app.task
def monitor_nyx_performance_task():
    """Monitor Nyx agent performance and log issues."""
    logger.info("Starting Nyx performance monitoring task")

    async def run_monitoring():
        if not await is_app_initialized():
            logger.info("Application not initialized. Skipping performance monitoring.")
            return {"status": "skipped", "reason": "App not initialized"}

        monitored_count = 0
        issues_found: List[Dict[str, Any]] = []

        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(
                    """
                    SELECT DISTINCT c.id, c.user_id
                      FROM conversations c
                      JOIN messages m ON m.conversation_id = c.id
                     WHERE m.created_at > NOW() - INTERVAL '1 hour'
                     GROUP BY c.id, c.user_id
                    """
                )
                for row in rows:
                    user_id = row["user_id"]
                    conversation_id = row["id"]
                    try:
                        perf_row = await conn.fetchrow(
                            """
                            SELECT metrics, error_log
                              FROM performance_metrics
                             WHERE user_id=$1 AND conversation_id=$2
                             ORDER BY created_at DESC
                             LIMIT 1
                            """,
                            user_id,
                            conversation_id,
                        )
                        if perf_row and perf_row["metrics"]:
                            metrics = json.loads(perf_row["metrics"])

                            if metrics.get("memory_usage", 0) > 600:
                                issues_found.append({"type": "high_memory", "user_id": user_id, "conversation_id": conversation_id, "value": metrics["memory_usage"]})

                            if metrics.get("error_rates", {}).get("total", 0) > 50:
                                issues_found.append({"type": "high_errors", "user_id": user_id, "conversation_id": conversation_id, "value": metrics["error_rates"]["total"]})

                            response_times = metrics.get("response_times", [])
                            if response_times and len(response_times) > 5:
                                avg_time = sum(response_times) / len(response_times)
                                if avg_time > 3.0:
                                    issues_found.append({"type": "slow_response", "user_id": user_id, "conversation_id": conversation_id, "value": avg_time})

                        monitored_count += 1
                    except Exception as e:
                        logger.error(f"Error monitoring performance for {user_id}/{conversation_id}: {e}")

                if issues_found:
                    logger.warning(f"Performance issues found: {json.dumps(issues_found)}")

            return {"status": "success", "conversations_monitored": monitored_count, "issues_found": len(issues_found), "issues": issues_found}
        except Exception as e:
            logger.exception("Error in Nyx performance monitoring task")
            return {"status": "error", "error": str(e)}

    return run_async_in_worker_loop(run_monitoring())


@celery_app.task
def aggregate_learning_metrics_task():
    """Aggregate learning metrics across the last day."""
    logger.info("Starting learning metrics aggregation task")

    async def run_aggregation():
        if not await is_app_initialized():
            return {"status": "skipped", "reason": "App not initialized"}

        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(
                    """
                    SELECT user_id, conversation_id, metrics, learned_patterns
                      FROM learning_metrics
                     WHERE created_at > NOW() - INTERVAL '1 day'
                     ORDER BY created_at DESC
                    """
                )

                total_patterns = 0
                pattern_success_rates: List[float] = []

                for row in rows:
                    if row["metrics"]:
                        metrics = json.loads(row["metrics"])
                        rate = metrics.get("adaptation_success_rate", 0.0)
                        if rate > 0:
                            pattern_success_rates.append(rate)

                    if row["learned_patterns"]:
                        patterns = json.loads(row["learned_patterns"])
                        total_patterns += len(patterns)

                avg_adaptation_rate = sum(pattern_success_rates) / len(pattern_success_rates) if pattern_success_rates else 0.0

                logger.info(
                    "Learning metrics - Total patterns: %s, Avg adaptation rate: %.2f%%, Active conversations: %s",
                    total_patterns,
                    100 * avg_adaptation_rate,
                    len(rows),
                )

                return {
                    "status": "success",
                    "total_patterns_learned": total_patterns,
                    "average_adaptation_rate": avg_adaptation_rate,
                    "active_learning_conversations": len(rows),
                }
        except Exception as e:
            logger.exception("Error in learning metrics aggregation")
            return {"status": "error", "error": str(e)}

    return run_async_in_worker_loop(run_aggregation())


@celery_app.task
def cleanup_old_performance_data_task():
    """Clean up old performance/learning/scenario data to keep DB lean."""
    logger.info("Starting performance data cleanup task")

    async def run_cleanup():
        try:
            async with get_db_connection_context() as conn:
                perf_result = await conn.execute(
                    "DELETE FROM performance_metrics WHERE created_at < NOW() - INTERVAL '7 days'"
                )
                perf_deleted = int(perf_result.split()[-1]) if perf_result else 0

                learn_result = await conn.execute(
                    "DELETE FROM learning_metrics WHERE created_at < NOW() - INTERVAL '30 days'"
                )
                learn_deleted = int(learn_result.split()[-1]) if learn_result else 0

                scenario_result = await conn.execute(
                    "DELETE FROM scenario_states WHERE created_at < NOW() - INTERVAL '3 days'"
                )
                scenario_deleted = int(scenario_result.split()[-1]) if scenario_result else 0

                logger.info(
                    "Cleanup complete - Performance: %s, Learning: %s, Scenarios: %s",
                    perf_deleted,
                    learn_deleted,
                    scenario_deleted,
                )

                return {
                    "status": "success",
                    "performance_metrics_deleted": perf_deleted,
                    "learning_metrics_deleted": learn_deleted,
                    "scenario_states_deleted": scenario_deleted,
                }
        except Exception as e:
            logger.exception("Error in cleanup task")
            return {"status": "error", "error": str(e)}

    return run_async_in_worker_loop(run_cleanup())

@shared_task(bind=True, max_retries=2, default_retry_delay=60)
def generate_and_cache_mpf_lore(
    self,
    user_id: int,
    conversation_id: int,
    foundation_data: Optional[Dict[str, Any]] = None,
):
    """Generate and cache Matriarchal Power Framework lore and transforms."""

    def _store_status_sync(
        status: str,
        principles=None,
        expressions=None,
        constraints=None,
        last_error=None,
        transformed_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        conn = get_db_connection_sync()
        try:
            with conn.cursor() as cur:
                json_principles = json.dumps(principles) if principles is not None else None
                json_expressions = json.dumps(expressions) if expressions is not None else None
                json_constraints = json.dumps(constraints) if constraints is not None else None
                json_transforms = (
                    json.dumps(transformed_fields) if transformed_fields is not None else None
                )

                def _execute_insert(include_transforms: bool) -> None:
                    if include_transforms:
                        cur.execute(
                            """
                            INSERT INTO MatriarchalLoreCache
                              (user_id, conversation_id, principles, expressions, constraints, transformed_fields, generation_status, last_updated)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                            ON CONFLICT (user_id, conversation_id) DO UPDATE SET
                              principles = EXCLUDED.principles,
                              expressions = EXCLUDED.expressions,
                              constraints = EXCLUDED.constraints,
                              transformed_fields = EXCLUDED.transformed_fields,
                              generation_status = EXCLUDED.generation_status,
                              last_updated = NOW();
                            """,
                            (
                                user_id,
                                conversation_id,
                                json_principles,
                                json_expressions,
                                json_constraints,
                                json_transforms,
                                status,
                            ),
                        )
                    else:
                        cur.execute(
                            """
                            INSERT INTO MatriarchalLoreCache
                              (user_id, conversation_id, principles, expressions, constraints, generation_status, last_updated)
                            VALUES (%s, %s, %s, %s, %s, %s, NOW())
                            ON CONFLICT (user_id, conversation_id) DO UPDATE SET
                              principles = EXCLUDED.principles,
                              expressions = EXCLUDED.expressions,
                              constraints = EXCLUDED.constraints,
                              generation_status = EXCLUDED.generation_status,
                              last_updated = NOW();
                            """,
                            (
                                user_id,
                                conversation_id,
                                json_principles,
                                json_expressions,
                                json_constraints,
                                status,
                            ),
                        )

                def _execute_status_update(include_transforms: bool) -> None:
                    if include_transforms:
                        cur.execute(
                            """
                            UPDATE MatriarchalLoreCache
                               SET generation_status = %s,
                                   transformed_fields = %s,
                                   last_updated = NOW()
                             WHERE user_id = %s AND conversation_id = %s;
                            """,
                            (status, json_transforms, user_id, conversation_id),
                        )
                    else:
                        cur.execute(
                            """
                            UPDATE MatriarchalLoreCache
                               SET generation_status = %s,
                                   last_updated = NOW()
                             WHERE user_id = %s AND conversation_id = %s;
                            """,
                            (status, user_id, conversation_id),
                        )

                try:
                    if status in ("complete", "partial"):
                        _execute_insert(include_transforms=True)
                    else:
                        _execute_status_update(include_transforms=True)
                except psycopg2_errors.UndefinedColumn:
                    if status in ("complete", "partial"):
                        _execute_insert(include_transforms=False)
                    else:
                        _execute_status_update(include_transforms=False)
            conn.commit()
        finally:
            conn.close()

    async def _generate_async():
        from lore.lore_orchestrator import get_lore_orchestrator

        orchestrator = await get_lore_orchestrator(user_id, conversation_id)
        core_principles, power_expressions, hierarchical_constraints = await asyncio.gather(
            orchestrator.mpf_generate_core_principles(),
            orchestrator.mpf_generate_power_expressions(limit=10),
            orchestrator.mpf_generate_hierarchical_constraints(),
            return_exceptions=True,
        )

        transformed_fields: Optional[Dict[str, Any]] = None
        if foundation_data:
            try:
                transformed_payload = await orchestrator.mpf_apply_power_lens(foundation_data)
            except Exception as exc:  # pragma: no cover - defensive log only
                logger.error("MPF lens transformation failed: %s", exc)
            else:
                if isinstance(transformed_payload, dict):
                    transformed_fields = {}
                    for key, original in foundation_data.items():
                        if not isinstance(original, str) or not original.strip():
                            continue
                        new_value = transformed_payload.get(key)
                        if not isinstance(new_value, str) or not new_value.strip():
                            continue
                        transformed_fields[key] = {
                            "transformed": new_value,
                            "source_hash": hashlib.sha256(original.encode("utf-8")).hexdigest(),
                        }

        def _ok(x):
            return not isinstance(x, Exception) and x is not None

        if isinstance(core_principles, Exception):
            logger.error("MPF core principles failed: %s", core_principles)
            core = None
        else:
            core = core_principles.dict() if _ok(core_principles) else None

        if isinstance(power_expressions, Exception):
            logger.error("MPF power expressions failed: %s", power_expressions)
            exprs = None
        else:
            exprs = [e.dict() for e in power_expressions] if _ok(power_expressions) else None

        if isinstance(hierarchical_constraints, Exception):
            logger.error("MPF hierarchical constraints failed: %s", hierarchical_constraints)
            cons = None
        else:
            cons = hierarchical_constraints.dict() if _ok(hierarchical_constraints) else None

        return core, exprs, cons, transformed_fields

    logger.info("Starting MPF lore generation for user=%s convo=%s", user_id, conversation_id)

    try:
        principles, expressions, constraints, transformed_fields = run_async_in_worker_loop(
            _generate_async()
        )

        if not principles and not expressions and not constraints:
            _store_status_sync("retrying", transformed_fields=transformed_fields)
            raise self.retry(exc=RuntimeError("All MPF sub-tasks failed"))

        status = "complete" if all([principles, expressions, constraints]) else "partial"
        _store_status_sync(
            status,
            principles,
            expressions,
            constraints,
            transformed_fields=transformed_fields,
        )
        logger.info("Cached MPF lore (%s) for user=%s convo=%s", status, user_id, conversation_id)

    except Retry:
        raise
    except Exception as exc:
        logger.exception(
            "MPF lore generation failed for user=%s convo=%s", user_id, conversation_id
        )
        _store_status_sync("failed", transformed_fields=None)
        raise self.retry(exc=exc) if self.request.retries < self.max_retries else exc


# === Split-brain sweep and merge ===============================================

async def find_split_brain_nyxes() -> List[str]:
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(minutes=30)
    async with get_db_connection_context() as conn:
        rows = await conn.fetch(
            """
            SELECT nyx_id, COUNT(DISTINCT instance_id) as instance_count
              FROM nyx_brain_checkpoints
             WHERE checkpoint_time > $1 AND nyx_id = $2
             GROUP BY nyx_id
            HAVING COUNT(DISTINCT instance_id) > 1
            """,
            cutoff,
            os.getenv("NYX_ID", "nyx_v1"),
        )
    return [row["nyx_id"] for row in rows]


async def perform_sweep_and_merge_for_id(nyx_id: str) -> bool:
    logger.info(f"Attempting merge for potentially split Nyx: {nyx_id}")
    try:
        brain = await NyxBrain.get_instance(0, 0, nyx_id=nyx_id)
        if not getattr(brain, "initialized", False):
            logger.warning(f"Skipping merge for {nyx_id}: Brain instance not initialized.")
            return False
        success = await brain.restore_entity_from_distributed_checkpoints()
        if success:
            logger.info(f"Successfully processed/merged state for Nyx: {nyx_id}")
            return True
        logger.warning(f"State restoration/merge returned no action or failed for Nyx: {nyx_id}")
        return False
    except Exception as e:
        logger.error(f"Failed to merge {nyx_id}: {e}", exc_info=True)
        return False


@celery_app.task
def sweep_and_merge_nyx_split_brains():
    """Periodically merge split-brain Nyx instances (if any)."""
    logger.info("Checking application readiness for split-brain sweep...")

    async def run_sweep():
        if not await is_app_initialized():
            logger.info("Application not initialized yet. Skipping split-brain sweep task.")
            return {"status": "skipped", "reason": "App not initialized"}

        logger.info("Application initialized. Starting split-brain Nyx sweep-and-merge task.")
        merged_count = 0
        failed_count = 0
        try:
            split_nyxes = await find_split_brain_nyxes()
            if not split_nyxes:
                logger.info("No split-brain Nyx instances found requiring merge.")
            else:
                logger.info(f"Found potentially split Nyx IDs: {split_nyxes}")
                for nyx_id in split_nyxes:
                    success = await perform_sweep_and_merge_for_id(nyx_id)
                    if success:
                        merged_count += 1
                    else:
                        failed_count += 1
                    await asyncio.sleep(1)

            logger.info(f"Sweep-and-merge completed. Merged: {merged_count}, Failed/Skipped: {failed_count}.")
            return {"status": "success", "merged": merged_count, "failed_or_skipped": failed_count}
        except Exception as e:
            logger.exception("Sweep-and-merge task failed critically.")
            return {"status": "error", "error": str(e)}

    return run_async_in_worker_loop(run_sweep())


# === LLM periodic checkpointing ===============================================

@celery_app.task
def run_llm_periodic_checkpoint_task(user_id: int, conversation_id: int):
    """Run LLM-driven checkpointing periodically."""
    nyx_id = os.getenv("NYX_ID", "nyx_v1")
    logger.info(f"Starting LLM periodic checkpoint task for NyxBrain {user_id}-{conversation_id} (NyxID: {nyx_id})...")

    async def run_checkpoint():
        if not await is_app_initialized():
            logger.info(f"App not initialized yet. Skipping LLM checkpoint for {user_id}-{conversation_id}.")
            return {"status": "skipped", "reason": "App not initialized"}

        try:
            brain_instance = await NyxBrain.get_instance(
                user_id, conversation_id, nyx_id=nyx_id if user_id == 0 and conversation_id == 0 else None
            )
            if not brain_instance or not getattr(brain_instance, "initialized", False):
                logger.warning(f"Brain instance not ready for {user_id}-{conversation_id}. Skipping checkpoint.")
                return {"status": "skipped", "reason": "Brain instance not ready"}

            current_state = await brain_instance.gather_checkpoint_state(event="periodic_llm_scheduled")

            planner_agent = CheckpointingPlannerAgent()
            checkpoint_plan = await planner_agent.recommend_checkpoint(current_state, brain_instance_for_context=brain_instance)

            if checkpoint_plan and checkpoint_plan.get("to_save"):
                data_to_save = checkpoint_plan["to_save"]  # {"field": {"value": ..., "why_saved": ...}}
                justifications = {k: v.get("why_saved", "N/A") for k, v in data_to_save.items()}
                skipped = checkpoint_plan.get("skip_fields", [])

                await brain_instance.save_planned_checkpoint(
                    event="periodic",
                    data_to_save=data_to_save,
                    justifications=justifications,
                    skipped=skipped
                )
                logger.info(f"LLM periodic checkpoint saved for {user_id}-{conversation_id}.")
                return {"status": "success", "saved_fields": len(data_to_save), "skipped_fields": len(skipped)}

            logger.info(f"Checkpoint planner recommended skipping save for {user_id}-{conversation_id}.")
            return {"status": "success", "saved_fields": 0, "skipped_fields": checkpoint_plan.get("skip_fields", ["No plan generated"])}

        except Exception as e:
            logger.exception(f"Error during LLM periodic checkpoint task for {user_id}-{conversation_id}")
            return {"status": "error", "error": str(e)}

    return run_async_in_worker_loop(run_checkpoint())


# === Handlers for SDK-enqueued maintenance tasks ===============================

@celery_app.task(name="memory.consolidate")
def memory_consolidate_task(conversation_id: str, recent_memories: Optional[List[Dict[str, Any]]] = None):
    """Consolidate recent memories for a conversation."""
    async def _run():
        try:
            conv_id = int(conversation_id)
            user_id = await _get_user_id_for_conversation(conv_id)
            await run_maintenance_through_nyx(
                user_id=user_id,
                conversation_id=conv_id,
                entity_type="conversation",
                entity_id=0
            )
            return {"ok": True, "conversation_id": conv_id}
        except Exception as e:
            logger.exception("memory.consolidate failed")
            return {"ok": False, "error": str(e)}
    return run_async_in_worker_loop(_run())


@celery_app.task
def lore_evolution_task(user_id: int, conversation_id: int, event_description: str, affected_location_id: Optional[int] = None):
    """
    REVISED: Background task to evolve lore based on a specific event.
    """
    async def _evolve_lore():
        with trace(workflow_name="lore_evolution_background"):
            from lore.lore_orchestrator import get_lore_orchestrator
            
            orchestrator = await get_lore_orchestrator(user_id, conversation_id)
            ctx = orchestrator._create_mock_context()
            
            evolution_result = await orchestrator.evolve_world_with_event(
                ctx,
                event_description=event_description,
                affected_location_id=affected_location_id
            )
            
            logger.info(f"Lore evolved for conversation {conversation_id} due to event: '{event_description[:50]}...'")
            return serialize_for_celery(evolution_result)
            
    try:
        return run_async_in_worker_loop(_evolve_lore())
    except Exception as e:
        logger.error(f"Lore evolution task failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}


@celery_app.task
def quick_setup_world_task(user_id: int, conversation_id: int, world_description: str):
    """
    NEW: Background task for the very heavy quick_setup_world function.
    """
    async def _setup_world():
        with trace(workflow_name="quick_setup_world_background"):
            from lore.lore_orchestrator import get_lore_orchestrator
            orchestrator = await get_lore_orchestrator(user_id, conversation_id)
            
            setup_result = await orchestrator.quick_setup_world(world_description)
            
            logger.info(f"Quick world setup completed for conversation {conversation_id}.")
            return serialize_for_celery(setup_result)
            
    try:
        return run_async_in_worker_loop(_setup_world())
    except Exception as e:
        logger.error(f"Quick world setup task failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

@celery_app.task
def update_conflict_tensions_task(params):
    """
    Background task to update conflict states and recalculate tensions.
    """
    async def update_tensions():
        with trace(workflow_name="conflict_tension_update"):
            try:
                active_conflicts = params.get('active_conflicts', [])
                
                # Get a single synthesizer for efficiency
                from logic.conflict_system.conflict_synthesizer import get_synthesizer
                
                # Need at least one conflict to get user/conversation context
                if active_conflicts:
                    # Get context from first conflict
                    async with get_db_connection_context() as conn:
                        first_conflict = await conn.fetchrow(
                            """SELECT user_id, conversation_id 
                               FROM Conflicts WHERE conflict_id = $1""",
                            active_conflicts[0]
                        )
                        if not first_conflict:
                            return {"status": "error", "error": "No conflicts found"}
                        
                        user_id = first_conflict['user_id']
                        conversation_id = first_conflict['conversation_id']
                else:
                    # Get any active conflicts
                    async with get_db_connection_context() as conn:
                        result = await conn.fetchrow(
                            """SELECT user_id, conversation_id 
                               FROM Conflicts 
                               WHERE is_active = true 
                               LIMIT 1"""
                        )
                        if not result:
                            return {"status": "success", "message": "No active conflicts"}
                        
                        user_id = result['user_id']
                        conversation_id = result['conversation_id']
                
                # Get synthesizer
                synthesizer = await get_synthesizer(user_id, conversation_id)
                
                # Process state sync to update tensions via signal
                from logic.conflict_system.signal_tasks import dispatch_tension_update_signal

                await dispatch_tension_update_signal(
                    synthesizer,
                    user_id,
                    conversation_id,
                )
                
                return {
                    "status": "success",
                    "message": "Tension update triggered"
                }
                
            except Exception as e:
                logger.error(f"Conflict tension update error: {e}", exc_info=True)
                return {"status": "error", "error": str(e)}
    
    return run_async_in_worker_loop(update_tensions())


@celery_app.task
def process_universal_updates_task(params):
    """
    Background task to process universal world state updates.
    """
    async def process_updates():
        with trace(workflow_name="universal_updates_background"):
            try:
                response_data = params.get('response', {})
                conversation_id = params.get('conversation_id')
                
                if not conversation_id:
                    return {"status": "error", "error": "No conversation_id provided"}
                
                # Get user_id from conversation
                async with get_db_connection_context() as conn:
                    result = await conn.fetchrow(
                        """SELECT user_id FROM CurrentRoleplay 
                           WHERE conversation_id = $1 AND key = 'UserId'""",
                        conversation_id
                    )
                    if result and result['user_id']:
                        user_id = result['user_id']
                    else:
                        # Try getting from NPCStats as fallback
                        result = await conn.fetchrow(
                            "SELECT DISTINCT user_id FROM NPCStats WHERE conversation_id = $1",
                            conversation_id
                        )
                        if not result:
                            return {"status": "error", "error": "Conversation not found"}
                        user_id = result['user_id']
                
                def _coerce_int(value: Any) -> Optional[int]:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return None

                def _extract_int_from_string(value: Any) -> Optional[int]:
                    if not isinstance(value, str):
                        return None
                    match = re.search(r"-?\d+", value)
                    if not match:
                        return None
                    try:
                        return int(match.group(0))
                    except ValueError:
                        return None

                def _ensure_dict(value: Any) -> Dict[str, Any]:
                    if isinstance(value, dict):
                        return value
                    if isinstance(value, str):
                        try:
                            parsed = json.loads(value)
                            if isinstance(parsed, dict):
                                return parsed
                        except Exception:
                            return {}
                    return {}

                def _extract_location_fields(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
                    location_name: Optional[str] = None
                    location_id: Optional[int] = None

                    def _update_from_payload(payload: Any) -> None:
                        nonlocal location_name, location_id
                        if isinstance(payload, dict):
                            location_name = location_name or payload.get("name") or payload.get("value") or payload.get("slug") or payload.get("location")
                            location_id = location_id or _coerce_int(payload.get("id") or payload.get("location_id") or payload.get("scene_id"))
                        elif isinstance(payload, (int, float)):
                            location_id = location_id or _coerce_int(payload)
                        elif isinstance(payload, str):
                            if payload.strip():
                                location_name = location_name or payload.strip()

                    for key in (
                        "current_location",
                        "location",
                        "scene_location",
                        "venue",
                        "currentVenue",
                    ):
                        if key in record:
                            _update_from_payload(record.get(key))

                    if not location_name:
                        for key in ("current_location_name", "location_name", "scene_name"):
                            value = record.get(key)
                            if isinstance(value, str) and value.strip():
                                location_name = value.strip()
                                break

                    for key in ("current_location_id", "location_id", "scene_id"):
                        if location_id is None and key in record:
                            location_id = _coerce_int(record.get(key))

                    return location_name, location_id

                def _merge_npc_update(
                    npc_updates: Dict[int, Dict[str, Any]],
                    npc_id: Optional[int],
                    location_name: Optional[str],
                    location_id: Optional[int],
                ) -> None:
                    if npc_id is None:
                        return
                    entry = npc_updates.setdefault(npc_id, {"npc_id": npc_id})
                    if location_name and not entry.get("current_location"):
                        entry["current_location"] = location_name
                    if location_id is not None and entry.get("location_id") is None:
                        entry["location_id"] = location_id

                def _collect_npc_updates_from_value(value: Any, npc_updates: Dict[int, Dict[str, Any]]) -> None:
                    if isinstance(value, list):
                        for item in value:
                            _collect_npc_updates_from_value(item, npc_updates)
                        return
                    if isinstance(value, dict):
                        candidate = value.get("value") if isinstance(value.get("value"), dict) else value
                        candidate = candidate or {}

                        possible_npc_id = (
                            _coerce_int(value.get("npc_id"))
                            or _coerce_int(candidate.get("npc_id"))
                            or _coerce_int(candidate.get("id"))
                            or _coerce_int(candidate.get("npcId"))
                        )
                        if possible_npc_id is None:
                            possible_npc_id = _extract_int_from_string(value.get("key"))

                        loc_name, loc_id = _extract_location_fields(candidate)
                        if loc_name is None and loc_id is None and isinstance(candidate.get("npc"), dict):
                            loc_name, loc_id = _extract_location_fields(candidate["npc"])

                        if possible_npc_id is not None and (loc_name or loc_id is not None):
                            _merge_npc_update(npc_updates, possible_npc_id, loc_name, loc_id)

                        for nested in candidate.values():
                            if isinstance(nested, (dict, list)):
                                _collect_npc_updates_from_value(nested, npc_updates)

                def _coerce_level_change(value: Any) -> Optional[int]:
                    if value is None:
                        return None
                    if isinstance(value, (int,)):
                        return value if value != 0 else None
                    try:
                        parsed = float(value)
                    except (TypeError, ValueError):
                        return None
                    if abs(parsed) < 1e-6:
                        return None
                    return int(round(parsed))

                def _extract_level_delta(candidate: Dict[str, Any]) -> Optional[int]:
                    for key in ("level_change", "delta", "change", "relationship_delta"):
                        delta = _coerce_level_change(candidate.get(key))
                        if delta is not None:
                            return delta

                    aggregate = 0
                    found = False
                    for key in (
                        "trust_change",
                        "respect_change",
                        "attraction_change",
                        "bond_change",
                        "power_change",
                        "power_dynamic_change",
                        "affection_change",
                    ):
                        val = _coerce_level_change(candidate.get(key))
                        if val is not None:
                            aggregate += val
                            found = True
                    if found and aggregate != 0:
                        return aggregate
                    return None

                def _collect_social_links_from_value(value: Any, social_link_map: Dict[Tuple[str, int, str, int], Dict[str, Any]]) -> None:
                    if isinstance(value, list):
                        for item in value:
                            _collect_social_links_from_value(item, social_link_map)
                        return
                    if not isinstance(value, dict):
                        return

                    candidate = value.get("value") if isinstance(value.get("value"), dict) else value
                    candidate = candidate or {}

                    delta = _extract_level_delta(candidate)
                    if delta is None:
                        for nested in candidate.values():
                            if isinstance(nested, (dict, list)):
                                _collect_social_links_from_value(nested, social_link_map)
                        return

                    entity1_id = (
                        _coerce_int(candidate.get("entity1_id"))
                        or _coerce_int(candidate.get("source_id"))
                        or _coerce_int(candidate.get("from_id"))
                        or _coerce_int(candidate.get("npc_id"))
                        or _extract_int_from_string(candidate.get("key"))
                    )
                    entity2_id = (
                        _coerce_int(candidate.get("entity2_id"))
                        or _coerce_int(candidate.get("target_id"))
                        or _coerce_int(candidate.get("to_id"))
                        or _coerce_int(candidate.get("player_id"))
                    )
                    entity1_type = candidate.get("entity1_type") or candidate.get("source_type") or candidate.get("from_type")
                    entity2_type = candidate.get("entity2_type") or candidate.get("target_type") or candidate.get("to_type")

                    if entity1_id is None:
                        return
                    if entity2_id is None:
                        entity2_id = _coerce_int(user_id)
                    if entity2_id is None:
                        return

                    entity1_type = str(entity1_type or "npc")
                    entity2_type = str(entity2_type or "player")

                    key = (entity1_type, entity1_id, entity2_type, entity2_id)
                    entry = social_link_map.setdefault(
                        key,
                        {
                            "entity1_type": entity1_type,
                            "entity1_id": entity1_id,
                            "entity2_type": entity2_type,
                            "entity2_id": entity2_id,
                            "level_change": 0,
                        },
                    )
                    entry["level_change"] += delta
                    if "group_context" not in entry:
                        context = candidate.get("group_context") or candidate.get("context")
                        if context:
                            entry["group_context"] = context
                    if "new_event" not in entry:
                        new_event = candidate.get("new_event") or candidate.get("event") or candidate.get("description")
                        if new_event:
                            entry["new_event"] = new_event

                updates_to_apply: Dict[str, Any] = {}

                narrative = response_data.get("narrative")
                if isinstance(narrative, str) and narrative.strip():
                    updates_to_apply["narrative"] = narrative.strip()

                metadata = response_data.get("metadata") if isinstance(response_data.get("metadata"), dict) else {}
                world_state = response_data.get("world_state") if isinstance(response_data.get("world_state"), dict) else {}

                scene_scope = _ensure_dict(metadata.get("scene_scope"))
                roleplay_updates: Dict[str, Any] = {}

                location_name = scene_scope.get("location_name") or scene_scope.get("location")
                location_id = scene_scope.get("location_id") or scene_scope.get("scene_id")

                if not location_name or location_id is None:
                    world_location = world_state.get("location") or world_state.get("current_location")
                    if isinstance(world_location, dict):
                        location_name = location_name or world_location.get("name") or world_location.get("value")
                        location_id = location_id or _coerce_int(world_location.get("id") or world_location.get("location_id"))
                    elif isinstance(world_location, str):
                        location_name = location_name or world_location.strip()

                if not location_name and isinstance(world_state.get("location_data"), str):
                    candidate_location = world_state["location_data"].strip()
                    if candidate_location:
                        location_name = candidate_location

                if location_name:
                    roleplay_updates["CurrentLocation"] = location_name

                scene_payload: Dict[str, Any] = {}
                if location_id is not None:
                    scene_payload["id"] = location_id
                if location_name:
                    scene_payload["name"] = location_name
                if scene_payload:
                    try:
                        roleplay_updates["CurrentScene"] = json.dumps({"location": scene_payload})
                    except Exception:
                        pass

                current_time = world_state.get("current_time")
                if isinstance(current_time, dict):
                    time_of_day = current_time.get("time_of_day")
                    if isinstance(time_of_day, dict):
                        time_of_day = time_of_day.get("value") or time_of_day.get("name")
                    if isinstance(time_of_day, str) and time_of_day.strip():
                        roleplay_updates["CurrentTime"] = time_of_day.strip()

                if roleplay_updates:
                    updates_to_apply["roleplay_updates"] = roleplay_updates

                npc_update_map: Dict[int, Dict[str, Any]] = {}
                for key, value in metadata.items():
                    if isinstance(value, (list, dict)) and "npc" in key.lower():
                        _collect_npc_updates_from_value(value, npc_update_map)

                _collect_npc_updates_from_value(world_state.get("active_npcs"), npc_update_map)
                _collect_npc_updates_from_value(world_state.get("npc_updates"), npc_update_map)
                _collect_npc_updates_from_value(world_state.get("npc_schedules"), npc_update_map)

                npc_updates = [entry for entry in npc_update_map.values() if any(k in entry for k in ("current_location", "location_id"))]
                if npc_updates:
                    updates_to_apply["npc_updates"] = npc_updates

                social_link_map: Dict[Tuple[str, int, str, int], Dict[str, Any]] = {}
                for key, value in metadata.items():
                    if isinstance(value, (list, dict)) and ("relationship" in key.lower() or "social" in key.lower()):
                        _collect_social_links_from_value(value, social_link_map)

                _collect_social_links_from_value(world_state.get("relationship_updates"), social_link_map)
                _collect_social_links_from_value(world_state.get("relationship_states"), social_link_map)
                _collect_social_links_from_value(world_state.get("pending_relationship_events"), social_link_map)

                social_links = [entry for entry in social_link_map.values() if entry.get("level_change")]
                if social_links:
                    updates_to_apply["social_links"] = social_links

                has_narrative = bool(updates_to_apply.get("narrative"))
                has_npc = bool(updates_to_apply.get("npc_updates"))
                has_social = bool(updates_to_apply.get("social_links"))
                player_location = updates_to_apply.get("roleplay_updates", {}) if isinstance(updates_to_apply.get("roleplay_updates"), dict) else {}
                has_player_location = any(
                    key in player_location for key in ("CurrentLocation", "current_location", "CurrentScene")
                )

                if not any((has_narrative, has_npc, has_social, has_player_location)):
                    logger.info(
                        "Skipping universal updates for conversation %s: no canonical fields derived",
                        conversation_id,
                    )
                    return {
                        "status": "skipped",
                        "conversation_id": conversation_id,
                        "reason": "no canonical updates",
                    }

                # The actual implementation uses UniversalUpdaterContext
                from logic.universal_updater_agent import (
                    UniversalUpdaterContext,
                    apply_universal_updates_async,
                )

                updater_context = UniversalUpdaterContext(user_id, conversation_id)
                await updater_context.initialize()

                async with get_db_connection_context() as conn:
                    await apply_universal_updates_async(
                        updater_context,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        updates=updates_to_apply,
                        conn=conn,
                    )

                logger.info(f"Universal updates applied for conversation {conversation_id}")

                return {
                    "status": "success",
                    "conversation_id": conversation_id,
                    "updates_applied": len(updates_to_apply)
                }
                
            except Exception as e:
                logger.error(f"Universal update error: {e}", exc_info=True)
                return {"status": "error", "error": str(e)}
    
    return run_async_in_worker_loop(process_updates())


@celery_app.task  
def npc_background_think_task(params):
    """
    Individual NPC background processing - update snapshot and relationships.
    """
    async def npc_think():
        with trace(workflow_name="npc_background_think"):
            try:
                npc_id = params.get('npc_id')
                if not npc_id:
                    return {"status": "error", "error": "No NPC ID provided"}
                
                # Get NPC's context
                async with get_db_connection_context() as conn:
                    npc_data = await conn.fetchrow("""
                        SELECT user_id, conversation_id, current_location
                        FROM NPCStats 
                        WHERE npc_id = $1
                    """, npc_id)
                    
                    if not npc_data:
                        return {"status": "error", "error": f"NPC {npc_id} not found"}
                
                from npcs.npc_orchestrator import NPCOrchestrator
                
                orchestrator = NPCOrchestrator(
                    npc_data['user_id'], 
                    npc_data['conversation_id']
                )
                await orchestrator.initialize()
                
                # Refresh NPC snapshot (this updates cache)
                snapshot = await orchestrator.get_npc_snapshot(
                    npc_id, 
                    force_refresh=True, 
                    light=False  # Full snapshot for background processing
                )
                
                # Run perception update if NPC is in a location
                if npc_data['current_location']:
                    await orchestrator.update_npc_perception(
                        npc_id, 
                        npc_data['current_location']
                    )
                
                # Update relationship dynamics
                await orchestrator.update_npc_relationships(npc_id)
                
                # Check for scheming opportunities
                if snapshot.dominance > 70 or snapshot.cruelty > 60:
                    # This NPC might be scheming
                    async with get_db_connection_context() as conn:
                        await conn.execute("""
                            UPDATE NPCStats 
                            SET scheming_level = LEAST(scheming_level + 5, 100),
                                last_updated = NOW()
                            WHERE npc_id = $1
                        """, npc_id)
                
                return {
                    "status": "success",
                    "npc_id": npc_id,
                    "npc_name": snapshot.name,
                    "status": snapshot.status,
                    "scheming": snapshot.scheming_level > 50
                }
                
            except Exception as e:
                logger.error(f"NPC think error for {params}: {e}", exc_info=True)
                return {"status": "error", "error": str(e)}
    
    return run_async_in_worker_loop(npc_think())


@celery_app.task(name="npc.background_think")
def npc_background_think_task(npc_id: int, context: Optional[Dict[str, Any]] = None):
    """Run a lightweight background cognition pass for an NPC."""
    async def _run():
        try:
            uid, cid = await _get_user_conv_for_npc(int(npc_id))
            mgr = NPCLearningManager(uid, cid)
            await mgr.initialize()
            await mgr.run_targeted_reflection([int(npc_id)])
            return {"ok": True, "npc_id": int(npc_id)}
        except Exception as e:
            logger.exception("npc.background_think failed")
            return {"ok": False, "error": str(e)}
    return run_async_in_worker_loop(_run())


@celery_app.task(name="lore.evolve")
def lore_evolve_task(affected_entities: Optional[List[str]] = None):
    """Placeholder hook for lore evolution (wire into your lore system)."""
    async def _run():
        try:
            # TODO: integrate with your lore evolution pipeline
            return {"ok": True, "affected": affected_entities or []}
        except Exception as e:
            logger.exception("lore.evolve failed")
            return {"ok": False, "error": str(e)}
    return run_async_in_worker_loop(_run())


@celery_app.task(name="conflict.update_tensions")
def conflict_update_tensions_task(active_conflicts: Optional[List[int]] = None):
    """Update conflict tensions. Implement routing if you can map conflict → (user, conversation)."""
    async def _run():
        try:
            updated = 0
            for _ in (active_conflicts or []):
                updated += 1
                # TODO: route to your conflict system if you can resolve scope per conflict_id
            return {"ok": True, "updated": updated}
        except Exception as e:
            logger.exception("conflict.update_tensions failed")
            return {"ok": False, "error": str(e)}
    return run_async_in_worker_loop(_run())


@celery_app.task(name="world.update_universal")
def world_update_universal_task(conversation_id: str, response: Dict[str, Any]):
    """Apply universal world updates for a conversation."""
    async def _run():
        try:
            conv_id = int(conversation_id)
            user_id = await _get_user_id_for_conversation(conv_id)
            from logic.universal_updater_agent import (
                UniversalUpdaterContext,
                apply_universal_updates_async,
            )

            updater_context = UniversalUpdaterContext(user_id, conv_id)
            await updater_context.initialize()

            async with get_db_connection_context() as conn:
                await apply_universal_updates_async(
                    updater_context,
                    user_id=user_id,
                    conversation_id=conv_id,
                    updates=response,
                    conn=conn,
                )
            return {"ok": True, "conversation_id": conv_id}
        except Exception as e:
            logger.exception("world.update_universal failed")
            return {"ok": False, "error": str(e)}
    return run_async_in_worker_loop(_run())


# Ensure celery_app is correctly exposed if imported elsewhere
app = celery_app
