# tasks.py

from __future__ import annotations

import os
import json
import logging
import asyncio
import asyncpg
import datetime
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from celery_config import celery_app
from agents import trace, custom_span
from agents.tracing import get_current_trace

# --- DB utils (async loop + connection mgmt) ---
from db.connection import get_db_connection_context, run_async_in_worker_loop

# --- LLM + NPC + memory integration (unchanged external modules) ---
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from new_game_agent import NewGameAgent
from npcs.npc_learning_adaptation import NPCLearningManager
from memory.memory_nyx_integration import run_maintenance_through_nyx

# --- Core NyxBrain + checkpointing ---
from nyx.core.brain.base import NyxBrain
from nyx.core.brain.checkpointing_agent import CheckpointingPlannerAgent

# --- New scene-scoped SDK (lazy singleton) ---
from nyx.nyx_agent_sdk import NyxAgentSDK
from nyx.nyx_agent_sdk.config import NyxConfig

logger = logging.getLogger(__name__)

# Define DSN (optional sanity check)
DB_DSN = os.getenv("DB_DSN", "postgresql://user:password@host:port/database")
if not DB_DSN:
    logger.error("DB_DSN environment variable not set for Celery tasks!")

# --- Application Readiness Flag (worker-local) ---
_APP_INITIALIZED = False
_LAST_INIT_CHECK_TIME = 0.0
_INIT_CHECK_INTERVAL = 30  # seconds


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
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


def get_preset_id(d: Dict[str, Any]) -> Optional[str]:
    """Extract preset story ID from various possible keys."""
    return d.get("preset_story_id") or d.get("story_id") or d.get("presetStoryId")


# === Nyx SDK lazy singleton ====================================================

_SDK: Optional[NyxAgentSDK] = None


async def _get_nyx_sdk() -> NyxAgentSDK:
    global _SDK
    if _SDK is None:
        _SDK = NyxAgentSDK(NyxConfig())
        await _SDK.initialize_agent()
    return _SDK


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

@celery_app.task
def background_chat_task_with_memory(conversation_id: int, user_input: str, user_id: int, universal_update: Optional[Dict[str, Any]] = None):
    """
    Enhanced background chat task that includes memory retrieval and scene-scoped SDK.
    Runs async flow on the worker's persistent loop.
    """
    async def run_chat_task():
        with trace(workflow_name="background_chat_task_celery"):
            logger.info(f"[BG Task {conversation_id}] Starting for user {user_id}")
            try:
                # 1) Build aggregator context
                from logic.aggregator import get_aggregated_roleplay_context
                aggregator_data = await get_aggregated_roleplay_context(user_id, conversation_id, "Chase")

                context = {
                    "location": aggregator_data.get("currentRoleplay", {}).get("CurrentLocation", "Unknown"),
                    "time_of_day": aggregator_data.get("timeOfDay", "Morning"),
                    "player_name": aggregator_data.get("playerName", "Chase"),
                    "npc_present": aggregator_data.get("npcsPresent", []),
                    "aggregator_data": aggregator_data,
                }

                # 2) Optional universal updates
                if universal_update:
                    logger.info(f"[BG Task {conversation_id}] Applying universal updates...")
                    try:
                        from logic.universal_updater import apply_universal_updates_async
                        async with get_db_connection_context() as conn:
                            await apply_universal_updates_async(
                                user_id,
                                conversation_id,
                                universal_update,
                                conn
                            )
                        logger.info(f"[BG Task {conversation_id}] Applied universal updates.")
                        # Refresh
                        aggregator_data = await get_aggregated_roleplay_context(user_id, conversation_id, context["player_name"])
                        context["aggregator_data"] = aggregator_data
                    except Exception as update_err:
                        logger.error(f"[BG Task {conversation_id}] Error applying universal updates: {update_err}", exc_info=True)
                        return {"error": "Failed to apply world updates"}

                # 3) Enrich context with memories
                try:
                    from memory.memory_integration import enrich_context_with_memories
                    logger.info(f"[BG Task {conversation_id}] Enriching context with memories...")
                    context = await enrich_context_with_memories(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        user_input=user_input,
                        context=context
                    )
                    logger.info(f"[BG Task {conversation_id}] Context enriched with memories.")
                except Exception as memory_err:
                    logger.error(f"[BG Task {conversation_id}] Error enriching context with memories: {memory_err}", exc_info=True)

                # 4) Invoke Nyx SDK (scene-scoped)
                sdk = await _get_nyx_sdk()
                logger.info(f"[BG Task {conversation_id}] Processing input with Nyx SDK...")
                nyx_resp = await sdk.process_user_input(
                    message=user_input,
                    conversation_id=str(conversation_id),
                    user_id=str(user_id),
                    metadata=context
                )
                logger.info(f"[BG Task {conversation_id}] Nyx SDK processing complete.")

                if not nyx_resp or not nyx_resp.success:
                    error_msg = (nyx_resp.metadata or {}).get("error", "Unknown SDK error") if nyx_resp else "Empty SDK response"
                    logger.error(f"[BG Task {conversation_id}] Nyx SDK failed: {error_msg}")
                    return {"error": error_msg}

                message_content = nyx_resp.narrative or ""

                # 5) Store response in DB + add memory
                try:
                    async with get_db_connection_context() as conn:
                        await conn.execute(
                            """INSERT INTO messages (conversation_id, sender, content, created_at)
                               VALUES ($1, $2, $3, NOW())""",
                            conversation_id, "Nyx", message_content
                        )
                    logger.info(f"[BG Task {conversation_id}] Stored Nyx response to DB.")

                    try:
                        from memory.memory_integration import add_memory_from_message
                        memory_id = await add_memory_from_message(
                            user_id=user_id,
                            conversation_id=conversation_id,
                            message_text=message_content,
                            entity_type="memory",
                            metadata={"source": "ai_response", "importance": 0.7}
                        )
                        logger.info(f"[BG Task {conversation_id}] Added AI response as memory {memory_id}")
                    except Exception as memory_err:
                        logger.error(f"[BG Task {conversation_id}] Error adding memory: {memory_err}", exc_info=True)

                except Exception as db_err:
                    logger.error(f"[BG Task {conversation_id}] DB Error storing Nyx response: {db_err}", exc_info=True)

                return serialize_for_celery({
                    "success": True,
                    "message": message_content,
                    "conversation_id": conversation_id,
                    "metadata": serialize_for_celery(nyx_resp.metadata),
                })

            except Exception as e:
                logger.error(f"[BG Task {conversation_id}] Critical Error: {str(e)}", exc_info=True)
                return {"error": f"Server error processing message: {str(e)}"}

    return run_async_in_worker_loop(run_chat_task())


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

@celery_app.task(expires=3600)
def process_new_game_task(user_id: int, conversation_data: Dict[str, Any]):
    """Process new (or preset) game creation asynchronously."""
    logger.info("CELERY – process_new_game_task called")

    async def run_new_game():
        with trace(workflow_name="process_new_game_celery_task"):
            try:
                logger.info("[NG] payload keys=%s, preset_id=%s",
                            list(conversation_data.keys()) if isinstance(conversation_data, dict) else type(conversation_data),
                            (conversation_data or {}).get("preset_story_id"))
                user_id_int = int(user_id)
            except Exception:
                logger.error(f"Invalid user_id: {user_id}")
                return {"status": "failed", "error": "Invalid user_id"}

            if not isinstance(conversation_data, dict):
                logger.error("conversation_data is not a dict")
                return {"status": "failed", "error": "Invalid conversation_data"}

            conv_id = conversation_data.get("conversation_id")
            if conv_id is not None:
                try:
                    conversation_data["conversation_id"] = int(conv_id)
                except Exception:
                    logger.error(f"Invalid conversation_id: {conv_id}")
                    return {"status": "failed", "error": "Invalid conversation_id"}

            preset_story_id = get_preset_id(conversation_data)

            agent = NewGameAgent()
            from lore.core.context import CanonicalContext
            ctx = CanonicalContext(user_id_int, conversation_data.get("conversation_id", 0))

            try:
                if preset_story_id:
                    logger.info(f"Preset path triggered (story_id={preset_story_id})")
                    result = await agent.process_preset_game_direct(ctx, conversation_data, preset_story_id)
                else:
                    result = await agent.process_new_game(ctx, conversation_data)

                def _get(attr, default=None):
                    return getattr(result, attr, default) if hasattr(result, attr) else result.get(attr, default) if isinstance(result, dict) else default

                conv_id_final = conversation_data.get("conversation_id") or _get("conversation_id")
                if conv_id_final is None:
                    logger.warning("No conversation_id found after pipeline; cannot mark ready.")
                conv_name = _get("conversation_name", "New Game")
                opening_text = _get("opening_narrative") or _get("opening_message") or "[World initialized]"

                try:
                    async with get_db_connection_context() as conn:
                        if conv_id_final is not None:
                            await conn.execute(
                                """
                                UPDATE conversations
                                   SET status='ready',
                                       conversation_name=$3
                                 WHERE id=$1 AND user_id=$2
                                """,
                                conv_id_final,
                                user_id_int,
                                conv_name,
                            )
                            exists = await conn.fetchval(
                                """
                                SELECT 1 FROM messages
                                 WHERE conversation_id=$1 AND sender='Nyx'
                                 LIMIT 1
                                """,
                                conv_id_final,
                            )
                            if not exists:
                                await conn.execute(
                                    """
                                    INSERT INTO messages (conversation_id, sender, content, created_at)
                                    VALUES ($1, 'Nyx', $2, NOW())
                                    """,
                                    conv_id_final,
                                    opening_text,
                                )
                                logger.info("Inserted opening Nyx message for conversation %s", conv_id_final)
                except Exception as upd_err:
                    logger.error("Failed to finalize conversation row: %s", upd_err, exc_info=True)

                return serialize_for_celery(result)

            except Exception as e:
                logger.exception("Critical error in process_new_game_task")

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

    return run_async_in_worker_loop(run_new_game())


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

# tasks.py

@celery_app.task
def lore_evolution_task(params):
    """
    Background task to evolve lore based on events.
    """
    async def evolve_lore():
        with trace(workflow_name="lore_evolution_background"):
            try:
                affected_entities = params.get('affected_entities', [])
                
                if not affected_entities:
                    logger.info("No entities to evolve, skipping lore evolution")
                    return {"status": "success", "message": "No entities to process"}
                
                processed_count = 0
                
                for entity in affected_entities:
                    try:
                        user_id = entity.get('user_id')
                        conversation_id = entity.get('conversation_id')
                        event_description = entity.get('event', 'World state evolution')
                        
                        if not user_id or not conversation_id:
                            continue
                        
                        # Use the lore orchestrator's actual method
                        from lore.lore_orchestrator import get_lore_orchestrator
                        
                        orchestrator = await get_lore_orchestrator(user_id, conversation_id)
                        
                        # Create mock context for the governance decorator
                        ctx = orchestrator._create_mock_context()
                        
                        # Use the actual evolve_world_with_event method
                        evolution_result = await orchestrator.evolve_world_with_event(
                            ctx,
                            event_description=event_description,
                            affected_location_id=entity.get('location_id')
                        )
                        
                        if evolution_result:
                            logger.info(f"Lore evolved for entity {entity}: {evolution_result.get('affected_elements', [])}")
                            processed_count += 1
                    
                    except Exception as e:
                        logger.error(f"Failed to evolve lore for entity {entity}: {e}")
                
                return {
                    "status": "success",
                    "entities_processed": processed_count
                }
                
            except Exception as e:
                logger.error(f"Lore evolution error: {e}", exc_info=True)
                return {"status": "error", "error": str(e)}
    
    return run_async_in_worker_loop(evolve_lore())


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
                
                # Process state sync to update tensions
                from logic.conflict_system.conflict_synthesizer import SystemEvent, EventType, SubsystemType
                
                event = SystemEvent(
                    event_id=f"tension_update_{time.time()}",
                    event_type=EventType.STATE_SYNC,
                    source_subsystem=SubsystemType.BACKGROUND,
                    payload={'update_tensions': True},
                    target_subsystems={SubsystemType.TENSION},
                    requires_response=False
                )
                
                await synthesizer.emit_event(event)
                
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
                
                # Build universal update structure
                updates_to_apply = {}
                
                # Extract relevant updates from response
                if 'world_state' in response_data:
                    updates_to_apply['world_state'] = response_data['world_state']
                
                if 'emergent_events' in response_data:
                    updates_to_apply['emergent_events'] = response_data['emergent_events']
                
                if 'npc_dialogue' in response_data:
                    updates_to_apply['npc_updates'] = response_data['npc_dialogue']
                
                # The actual implementation uses UniversalUpdaterContext
                from logic.universal_updater_agent import UniversalUpdaterContext, apply_universal_updates_async
                
                try:
                    updater_context = UniversalUpdaterContext(user_id, conversation_id)
                    await updater_context.initialize()
                    
                    await apply_universal_updates_async(
                        updater_context,
                        user_id,
                        conversation_id,
                        updates_to_apply,
                        None  # Connection will be created internally
                    )
                    
                    logger.info(f"Universal updates applied for conversation {conversation_id}")
                    
                    return {
                        "status": "success",
                        "conversation_id": conversation_id,
                        "updates_applied": len(updates_to_apply)
                    }
                except ImportError:
                    # Fallback to simpler updater if available
                    from logic.universal_updater import apply_universal_updates_async
                    
                    async with get_db_connection_context() as conn:
                        await apply_universal_updates_async(
                            user_id,
                            conversation_id,
                            updates_to_apply,
                            conn
                        )
                    
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
            from logic.universal_updater import apply_universal_updates_async
            async with get_db_connection_context() as conn:
                await apply_universal_updates_async(
                    user_id=user_id,
                    conversation_id=conv_id,
                    universal_update=response,
                    conn=conn
                )
            return {"ok": True, "conversation_id": conv_id}
        except Exception as e:
            logger.exception("world.update_universal failed")
            return {"ok": False, "error": str(e)}
    return run_async_in_worker_loop(_run())


# Ensure celery_app is correctly exposed if imported elsewhere
app = celery_app
