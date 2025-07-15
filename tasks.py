# tasks.py

import os
import json
import logging
import asyncio
import asyncpg
import datetime
from celery_config import celery_app
from functools import wraps
import time # Import time for potential delays
from agents import trace, custom_span
from agents.tracing import get_current_trace
import traceback

# Import your helper functions and task logic
from npcs.new_npc_creation import NPCCreationHandler
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from new_game_agent import NewGameAgent
from npcs.npc_learning_adaptation import NPCLearningManager
from memory.memory_nyx_integration import run_maintenance_through_nyx
from db.connection import get_db_connection_context

# --- Core NyxBrain and Checkpointing ---
from nyx.core.brain.base import NyxBrain
from nyx.core.brain.checkpointing_agent import CheckpointingPlannerAgent

_WORKER_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(_WORKER_LOOP)

logger = logging.getLogger(__name__)

# Define DSN globally or ensure it's available
DB_DSN = os.getenv("DB_DSN", "postgresql://user:password@host:port/database")
if not DB_DSN:
    logger.error("DB_DSN environment variable not set for Celery tasks!")

# --- Application Readiness Flag ---
# This is a simple in-memory flag. For production, you might use Redis,
# a database flag, or an external health check endpoint.
_APP_INITIALIZED = False
_LAST_INIT_CHECK_TIME = 0
_INIT_CHECK_INTERVAL = 30 # Check every 30 seconds

def set_app_initialized():
    """Call this from main.py AFTER successful NyxBrain initialization."""
    global _APP_INITIALIZED
    _APP_INITIALIZED = True
    logger.info("Application initialization status set to True for Celery tasks.")

async def is_app_initialized():
    """Checks if the application is initialized (with caching)."""
    global _APP_INITIALIZED, _LAST_INIT_CHECK_TIME
    now = time.time()

    # If already marked as initialized, return True
    if _APP_INITIALIZED:
        return True

    # Check if enough time has passed since the last check or if never checked
    if now - _LAST_INIT_CHECK_TIME < _INIT_CHECK_INTERVAL:
        return False # Return cached False value

    _LAST_INIT_CHECK_TIME = now # Update last check time

    return False

def serialize_for_celery(obj):
    """
    Convert Pydantic models or other objects to JSON-serializable format for Celery.
    
    Args:
        obj: The object to serialize (Pydantic model, dict, or other)
        
    Returns:
        JSON-serializable representation of the object
    """
    if hasattr(obj, 'model_dump'):
        # Pydantic v2
        return obj.model_dump()
    elif hasattr(obj, 'dict'):
        # Pydantic v1
        return obj.dict()
    elif hasattr(obj, '__dict__'):
        # Generic object with __dict__
        return obj.__dict__
    else:
        # Already serializable (dict, list, str, etc.)
        return obj



# Helper decorator to run async functions in Celery tasks
def async_task(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if _WORKER_LOOP.is_closed():
            raise RuntimeError("Worker event-loop was closed unexpectedly")
        return _WORKER_LOOP.run_until_complete(func(*args, **kwargs))
    return wrapper

@celery_app.task
def test_task():
    """Simple test task to verify Celery is working."""
    logger.info("Executing test task!")
    return "Hello from test task!"

# Convert async tasks to use the async_task wrapper
@celery_app.task
@async_task
async def background_chat_task_with_memory(conversation_id, user_input, user_id, universal_update=None):
    """
    Enhanced background chat task that includes memory retrieval.
    """
    with trace(workflow_name="background_chat_task_celery"):
        logger.info(f"[BG Task {conversation_id}] Starting for user {user_id}")
        try:
            # Get aggregator context
            from logic.aggregator import get_aggregated_roleplay_context
            aggregator_data = await get_aggregated_roleplay_context(user_id, conversation_id, "Chase")
    
            context = {
                "location": aggregator_data.get("currentRoleplay", {}).get("CurrentLocation", "Unknown"),
                "time_of_day": aggregator_data.get("timeOfDay", "Morning"),
                "player_name": aggregator_data.get("playerName", "Chase"),
                "npc_present": aggregator_data.get("npcsPresent", []),
                "aggregator_data": aggregator_data
            }
    
            # Apply universal update if provided
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
                    # Refresh aggregator data post-update
                    aggregator_data = await get_aggregated_roleplay_context(user_id, conversation_id, context["player_name"])
                    context["aggregator_data"] = aggregator_data
                except Exception as update_err:
                    logger.error(f"[BG Task {conversation_id}] Error applying universal updates: {update_err}", exc_info=True)
                    return {"error": "Failed to apply world updates"}
    
            # Enrich context with relevant memories
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
                # Continue without memories if error occurs
    
            # Process the user_input with OpenAI-enhanced Nyx agent
            from nyx.nyx_agent_sdk import process_user_input_with_openai
            logger.info(f"[BG Task {conversation_id}] Processing input with Nyx agent...")
            response = await process_user_input_with_openai(user_id, conversation_id, user_input, context)
            logger.info(f"[BG Task {conversation_id}] Nyx agent processing complete.")
    
            if not response or not response.get("success", False):
                error_msg = response.get("error", "Unknown error from Nyx agent") if response else "Empty response from Nyx agent"
                logger.error(f"[BG Task {conversation_id}] Nyx agent failed: {error_msg}")
                return {"error": error_msg}
    
            # Extract the message content
            message_content = response.get("message", "")
            if not message_content and "function_args" in response:
                message_content = response["function_args"].get("narrative", "")
            
            # Store the Nyx response in DB
            try:
                async with get_db_connection_context() as conn:
                    await conn.execute(
                        """INSERT INTO messages (conversation_id, sender, content, created_at)
                           VALUES ($1, $2, $3, NOW())""",
                        conversation_id, "Nyx", message_content
                    )
                logger.info(f"[BG Task {conversation_id}] Stored Nyx response to DB.")
                
                # Add AI response as a memory
                try:
                    from memory.memory_integration import add_memory_from_message
                    
                    memory_id = await add_memory_from_message(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        message_text=message_content,
                        entity_type="memory",
                        metadata={
                            "source": "ai_response",
                            "importance": 0.7  # Higher importance for AI responses
                        }
                    )
                    logger.info(f"[BG Task {conversation_id}] Added AI response as memory {memory_id}")
                except Exception as memory_err:
                    logger.error(f"[BG Task {conversation_id}] Error adding memory: {memory_err}", exc_info=True)
                    
            except Exception as db_err:
                logger.error(f"[BG Task {conversation_id}] DB Error storing Nyx response: {db_err}", exc_info=True)
    
            return {
                "success": True,
                "message": message_content,
                "conversation_id": conversation_id
            }
    
        except Exception as e:
            logger.error(f"[BG Task {conversation_id}] Critical Error: {str(e)}", exc_info=True)
            return {"error": f"Server error processing message: {str(e)}"}

# Memory System Celery Tasks
@celery_app.task
def process_memory_embedding_task(user_id, conversation_id, message_text, entity_type="memory", metadata=None):
    """
    Celery task to process a memory embedding asynchronously.
    """
    from memory.memory_integration import process_memory_task
    
    logger.info(f"Processing memory embedding for user {user_id}, conversation {conversation_id}")
    
    # Call the async task wrapper
    result = process_memory_task(user_id, conversation_id, message_text, entity_type)
    
    return result

@celery_app.task
def retrieve_memories_task(user_id, conversation_id, query_text, entity_types=None, top_k=5):
    """
    Celery task to retrieve relevant memories.
    """
    from memory.memory_integration import memory_celery_task
    
    logger.info(f"Retrieving memories for user {user_id}, conversation {conversation_id}, query: {query_text[:50]}...")
    
    # Define async function
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
            
            return {
                "success": True,
                "memories": memories,
                "message": f"Successfully retrieved {len(memories)} memories"
            }
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return {
                "success": False,
                "memories": [],
                "error": str(e)
            }
    
    # Create and use the task wrapper
    wrapper = memory_celery_task(retrieve_memories_async)
    result = wrapper()
    
    return result

@celery_app.task
def analyze_with_memory_task(user_id, conversation_id, query_text, entity_types=None, top_k=5):
    """
    Celery task to analyze a query with relevant memories.
    """
    from memory.memory_integration import memory_celery_task
    
    logger.info(f"Analyzing query with memories for user {user_id}, conversation {conversation_id}, query: {query_text[:50]}...")
    
    # Define async function
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
            
            return {
                "success": True,
                "result": result,
                "message": "Successfully analyzed query with memories"
            }
        except Exception as e:
            logger.error(f"Error analyzing with memories: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }
    
    # Create and use the task wrapper
    wrapper = memory_celery_task(analyze_with_memory_async)
    result = wrapper()
    
    return result

@celery_app.task
def memory_maintenance_task():
    """
    Celery task to perform maintenance on the memory system.
    This task should be scheduled to run periodically.
    """
    from memory.memory_integration import memory_celery_task
    
    logger.info("Running memory system maintenance task")
    
    # Define async function
    async def memory_maintenance_async():
        from memory.memory_integration import cleanup_memory_services, cleanup_memory_retrievers
        
        try:
            # Run any needed maintenance tasks
            
            # Finally, clean up resources
            await cleanup_memory_services()
            await cleanup_memory_retrievers()
            
            return {
                "success": True,
                "message": "Memory system maintenance completed"
            }
        except Exception as e:
            logger.error(f"Error during memory maintenance: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # Create and use the task wrapper
    wrapper = memory_celery_task(memory_maintenance_async)
    result = wrapper()
    
    return result

# Fixed version of the NPC learning cycle task
@celery_app.task
def run_npc_learning_cycle_task():
    """
    Celery task to run the NPC learning cycle periodically for active conversations.
    """
    with trace(workflow_name="npc_learning_cycle_celery"):
        logger.info("Starting NPC learning cycle task via Celery Beat.")
        
        async def run_learning_cycle():
            processed_conversations = 0
            try:
                # Use the async context manager for DB connection
                async with get_db_connection_context() as conn:
                    # Find recent conversations
                    convs = await conn.fetch("""
                        SELECT id, user_id
                        FROM conversations
                        WHERE last_active > NOW() - INTERVAL '1 day'
                    """)
    
                    if not convs:
                        logger.info("No recent conversations found for NPC learning.")
                        return {"status": "success", "processed_conversations": 0}
    
                    for conv_row in convs:
                        conv_id = conv_row['id']
                        user_id = conv_row['user_id']
                        try:
                            # Fetch NPCs for this conversation
                            npc_rows = await conn.fetch("""
                                SELECT npc_id FROM NPCStats
                                WHERE user_id=$1 AND conversation_id=$2
                            """, user_id, conv_id)
    
                            npc_ids = [row['npc_id'] for row in npc_rows]
    
                            if npc_ids:
                                # Run the learning logic
                                manager = NPCLearningManager(user_id, conv_id)
                                await manager.initialize()
                                await manager.run_regular_adaptation_cycle(npc_ids)
                                logger.info(f"Learning cycle completed for conversation {conv_id}: {len(npc_ids)} NPCs")
                                processed_conversations += 1
                            else:
                                logger.info(f"No NPCs found for learning cycle in conversation {conv_id}.")
    
                        except Exception as e_inner:
                            logger.error(f"Error in NPC learning cycle for conv {conv_id}: {e_inner}", exc_info=True)
                            # Continue to the next conversation
    
            except Exception as e_outer:
                logger.error(f"Critical error in NPC learning scheduler task: {e_outer}", exc_info=True)
                return {"status": "error", "message": str(e_outer)}
    
            logger.info(f"NPC learning cycle task finished. Processed {processed_conversations} conversations.")
            return {"status": "success", "processed_conversations": processed_conversations}
        
        # Run the async function in the sync task
        return asyncio.run(run_learning_cycle())


@celery_app.task(expires=3600)
@async_task
async def process_new_game_task(user_id, conversation_data):
    """
    Celery task to process new game creation.
    
    Args:
        user_id: The user ID for the new game
        conversation_data: Dict containing conversation setup data
        
    Returns:
        Dict with game creation results (JSON-serializable)
    """
    with trace(workflow_name="process_new_game_celery_task"):
        logger.info("CELERY – process_new_game_task called")
        agent = NewGameAgent()
        conversation_id = None
    
        try:
            # Call the agent's process_new_game method
            from lore.core.context import CanonicalContext  # or create inline
            ctx = CanonicalContext(user_id, conversation_data.get('conversation_id', 0))
            result = await agent.process_new_game(ctx, conversation_data)
            
            # Convert the Pydantic model result to a JSON-serializable dict
            serialized_result = serialize_for_celery(result)
            
            logger.info(f"Successfully created new game for user_id={user_id}, "
                       f"conversation_id={serialized_result.get('conversation_id')}")
            
            return serialized_result
            
        except Exception as e:
            logger.exception(f"[DEBUG] Critical error in process_new_game_task for user_id={user_id}")
            
            # Try to update conversation status to failed
            conversation_id = conversation_data.get("conversation_id")
            if conversation_id:
                try:
                    # Don't use asyncio.run() here - we're already in an async context
                    async with get_db_connection_context() as conn:
                        await conn.execute("""
                            UPDATE conversations 
                            SET status='failed', 
                                conversation_name='New Game - Task Failed'
                            WHERE id=$1 AND user_id=$2
                        """, conversation_id, user_id)
                        logger.info(f"Updated conversation {conversation_id} status to 'failed'")
                except Exception as update_error:
                    logger.error(f"[DEBUG] Failed to update conversation status: {update_error}")
            
            # Return a serializable error structure
            error_result = {
                "status": "failed", 
                "error": str(e),
                "error_type": type(e).__name__,
                "conversation_id": conversation_id
            }
            
            logger.error(f"Returning error result: {error_result}")
            return error_result
        
@celery_app.task
def create_npcs_task(user_id, conversation_id, count=10):
    """Celery task to create NPCs using async logic."""
    logger.info(f"Starting create_npcs_task for user={user_id}, conv={conversation_id}, count={count}")

    async def create_npcs_async():
        # Use the async context manager
        async with get_db_connection_context() as conn:
            # Fetch environment_desc from DB
            row_env = await conn.fetchrow("""
                SELECT value
                FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='EnvironmentDesc'
            """, user_id, conversation_id)
            environment_desc = row_env["value"] if row_env else "A fallback environment description"
            logger.debug(f"Fetched environment_desc: {environment_desc[:100]}...")

            # Fetch 'CalendarNames' from DB
            row_cal = await conn.fetchrow("""
                SELECT value
                FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='CalendarNames'
            """, user_id, conversation_id)
            if row_cal and row_cal["value"]:
                try:
                    cal_data = json.loads(row_cal["value"])
                    day_names = cal_data.get("days", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode CalendarNames JSON for conv {conversation_id}, using defaults.")
                    day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            else:
                logger.info(f"No CalendarNames found for conv {conversation_id}, using defaults.")
                day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            logger.debug(f"Using day_names: {day_names}")

            # Spawn NPCs - assuming this function exists in your codebase
            from npcs.npc_creation import spawn_multiple_npcs
            npc_ids = await spawn_multiple_npcs(
                user_id=user_id,
                conversation_id=conversation_id,
                environment_desc=environment_desc,
                day_names=day_names,
                count=count
            )

            return {
                "message": f"Successfully created {len(npc_ids)} NPCs (via Nyx governance)",
                "npc_ids": npc_ids
            }

    try:
        # Run the async main function within the synchronous Celery task
        final_info = asyncio.run(create_npcs_async())
        logger.info(f"Finished create_npcs_task successfully for user={user_id}, conv={conversation_id}. NPCs: {final_info.get('npc_ids')}")
        return final_info
    except Exception as e:
        logger.exception(f"Error in create_npcs_task for user={user_id}, conv={conversation_id}")
        return {"status": "failed", "error": str(e)}

@celery_app.task
def get_gpt_opening_line_task(conversation_id, aggregator_text, opening_user_prompt):
    """
    Generate the GPT opening line.
    This task calls the GPT API (or fallback) and returns a JSON-encoded reply.
    """
    logger.info(f"Async GPT task: Calling GPT for opening line for conv_id={conversation_id}.")

    # First attempt: normal GPT call
    gpt_reply_dict = get_chatgpt_response(
        conversation_id=conversation_id,
        aggregator_text=aggregator_text,
        user_input=opening_user_prompt
    )

    # Check if a fallback is needed
    nyx_text = None
    if isinstance(gpt_reply_dict, dict):
        nyx_text = gpt_reply_dict.get("response")
        if gpt_reply_dict.get("type") == "function_call" or not nyx_text:
            nyx_text = None # Force fallback
    else:
        logger.error(f"get_chatgpt_response returned unexpected type: {type(gpt_reply_dict)}")
        gpt_reply_dict = {} # Initialize for fallback

    if nyx_text is None:
        logger.warning("Async GPT task: GPT returned function call, no text, or error. Retrying without function calls.")
        try:
            client = get_openai_client()
            forced_messages = [
                {"role": "system", "content": aggregator_text},
                {"role": "user", "content": "No function calls. Produce only a text narrative.\n\n" + opening_user_prompt}
            ]
            fallback_response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-nano"),
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

    # Ensure the result is always JSON serializable
    try:
        result_json = json.dumps(gpt_reply_dict)
        logger.info(f"GPT opening line task completed for conv_id={conversation_id}.")
        return result_json
    except (TypeError, OverflowError) as e:
        logger.error(f"Failed to serialize GPT response to JSON: {e}. Response: {gpt_reply_dict}")
        # Return a serializable error
        return json.dumps({"status": "error", "message": "Failed to serialize GPT response", "original_response_type": str(type(gpt_reply_dict))})

@celery_app.task
def nyx_memory_maintenance_task():
    """Celery task for Nyx memory maintenance using asyncpg."""
    with trace(workflow_name="nyx_memory_maintenance_celery"):
        logger.info("Starting Nyx memory maintenance task (via governance)")
    
        async def process_all_conversations():
            processed_count = 0
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT DISTINCT user_id, conversation_id
                    FROM NyxMemories
                    WHERE is_archived = FALSE
                    AND timestamp > NOW() - INTERVAL '30 days'
                """)
    
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
    
                    await asyncio.sleep(0.1) # Small delay to prevent hammering
    
                return {
                    "status": "success",
                    "conversations_processed": processed_count
                }
    
        try:
            # Run the async logic
            result = asyncio.run(process_all_conversations())
            logger.info(f"Nyx memory maintenance task completed: {result}")
            return result
        except Exception as e:
            logger.exception("Critical error in nyx_memory_maintenance_task")
            return {"status": "error", "error": str(e)}

@celery_app.task
@async_task
async def monitor_nyx_performance_task():
    """
    Periodic task to monitor Nyx agent performance across active conversations.
    Collects metrics and triggers cleanup if needed.
    """
    logger.info("Starting Nyx performance monitoring task")
    
    if not await is_app_initialized():
        logger.info("Application not initialized. Skipping performance monitoring.")
        return {"status": "skipped", "reason": "App not initialized"}
    
    monitored_count = 0
    issues_found = []
    
    try:
        async with get_db_connection_context() as conn:
            # Find active conversations
            rows = await conn.fetch("""
                SELECT DISTINCT c.id, c.user_id
                FROM conversations c
                JOIN messages m ON m.conversation_id = c.id
                WHERE m.created_at > NOW() - INTERVAL '1 hour'
                GROUP BY c.id, c.user_id
            """)
            
            for row in rows:
                user_id = row['user_id']
                conversation_id = row['id']
                
                try:
                    # Get latest performance metrics
                    perf_row = await conn.fetchrow("""
                        SELECT metrics, error_log
                        FROM performance_metrics
                        WHERE user_id = $1 AND conversation_id = $2
                        ORDER BY created_at DESC
                        LIMIT 1
                    """, user_id, conversation_id)
                    
                    if perf_row and perf_row['metrics']:
                        metrics = json.loads(perf_row['metrics'])
                        
                        # Check for issues
                        if metrics.get('memory_usage', 0) > 600:  # 600MB threshold
                            issues_found.append({
                                'type': 'high_memory',
                                'user_id': user_id,
                                'conversation_id': conversation_id,
                                'value': metrics['memory_usage']
                            })
                        
                        if metrics.get('error_rates', {}).get('total', 0) > 50:
                            issues_found.append({
                                'type': 'high_errors',
                                'user_id': user_id,
                                'conversation_id': conversation_id,
                                'value': metrics['error_rates']['total']
                            })
                        
                        # Calculate average response time
                        response_times = metrics.get('response_times', [])
                        if response_times and len(response_times) > 5:
                            avg_time = sum(response_times) / len(response_times)
                            if avg_time > 3.0:  # 3 second threshold
                                issues_found.append({
                                    'type': 'slow_response',
                                    'user_id': user_id,
                                    'conversation_id': conversation_id,
                                    'value': avg_time
                                })
                    
                    monitored_count += 1
                    
                except Exception as e:
                    logger.error(f"Error monitoring performance for {user_id}/{conversation_id}: {e}")
            
            # Log aggregated metrics
            if issues_found:
                logger.warning(f"Performance issues found: {json.dumps(issues_found)}")
                
                # Could trigger alerts or auto-remediation here
                # For example, send to monitoring system or trigger cleanup tasks
        
        return {
            "status": "success",
            "conversations_monitored": monitored_count,
            "issues_found": len(issues_found),
            "issues": issues_found
        }
        
    except Exception as e:
        logger.exception("Error in Nyx performance monitoring task")
        return {"status": "error", "error": str(e)}


@celery_app.task
@async_task
async def aggregate_learning_metrics_task():
    """
    Periodic task to aggregate learning metrics across all Nyx instances.
    Useful for understanding system-wide learning patterns.
    """
    logger.info("Starting learning metrics aggregation task")
    
    if not await is_app_initialized():
        return {"status": "skipped", "reason": "App not initialized"}
    
    try:
        async with get_db_connection_context() as conn:
            # Get recent learning metrics
            rows = await conn.fetch("""
                SELECT user_id, conversation_id, metrics, learned_patterns
                FROM learning_metrics
                WHERE created_at > NOW() - INTERVAL '1 day'
                ORDER BY created_at DESC
            """)
            
            # Aggregate metrics
            total_patterns = 0
            avg_adaptation_rate = 0.0
            pattern_success_rates = []
            
            for row in rows:
                if row['metrics']:
                    metrics = json.loads(row['metrics'])
                    adaptation_rate = metrics.get('adaptation_success_rate', 0.0)
                    if adaptation_rate > 0:
                        pattern_success_rates.append(adaptation_rate)
                
                if row['learned_patterns']:
                    patterns = json.loads(row['learned_patterns'])
                    total_patterns += len(patterns)
            
            if pattern_success_rates:
                avg_adaptation_rate = sum(pattern_success_rates) / len(pattern_success_rates)
            
            # Store aggregated metrics (could go to a monitoring system)
            logger.info(f"Learning metrics - Total patterns: {total_patterns}, "
                       f"Avg adaptation rate: {avg_adaptation_rate:.2%}, "
                       f"Active conversations: {len(rows)}")
            
            return {
                "status": "success",
                "total_patterns_learned": total_patterns,
                "average_adaptation_rate": avg_adaptation_rate,
                "active_learning_conversations": len(rows)
            }
            
    except Exception as e:
        logger.exception("Error in learning metrics aggregation")
        return {"status": "error", "error": str(e)}


@celery_app.task
@async_task
async def cleanup_old_performance_data_task():
    """
    Periodic task to clean up old performance and learning data.
    Keeps the database lean while preserving important patterns.
    """
    logger.info("Starting performance data cleanup task")
    
    try:
        async with get_db_connection_context() as conn:
            # Clean up old performance metrics (keep last 7 days)
            perf_result = await conn.execute("""
                DELETE FROM performance_metrics
                WHERE created_at < NOW() - INTERVAL '7 days'
            """)
            perf_deleted = int(perf_result.split()[-1]) if perf_result else 0
            
            # Clean up old learning metrics (keep last 30 days)
            learn_result = await conn.execute("""
                DELETE FROM learning_metrics
                WHERE created_at < NOW() - INTERVAL '30 days'
            """)
            learn_deleted = int(learn_result.split()[-1]) if learn_result else 0
            
            # Clean up old scenario states (keep last 3 days)
            scenario_result = await conn.execute("""
                DELETE FROM scenario_states
                WHERE created_at < NOW() - INTERVAL '3 days'
            """)
            scenario_deleted = int(scenario_result.split()[-1]) if scenario_result else 0
            
            logger.info(f"Cleanup complete - Performance: {perf_deleted}, "
                       f"Learning: {learn_deleted}, Scenarios: {scenario_deleted}")
            
            return {
                "status": "success",
                "performance_metrics_deleted": perf_deleted,
                "learning_metrics_deleted": learn_deleted,
                "scenario_states_deleted": scenario_deleted
            }
            
    except Exception as e:
        logger.exception("Error in cleanup task")
        return {"status": "error", "error": str(e)}

# --- Utility Functions ---
async def find_split_brain_nyxes():
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(minutes=30) # Check last 30 mins
    async with get_db_connection_context() as conn:
        rows = await conn.fetch("""
            SELECT nyx_id, COUNT(DISTINCT instance_id) as instance_count
            FROM nyx_brain_checkpoints
            WHERE checkpoint_time > $1 AND nyx_id = $2
            GROUP BY nyx_id
            HAVING COUNT(DISTINCT instance_id) > 1
        """, cutoff, os.getenv("NYX_ID", "nyx_v1")) # Filter by current NYX_ID
    return [row["nyx_id"] for row in rows]

async def perform_sweep_and_merge_for_id(nyx_id: str):
    """Performs merge for a specific nyx_id."""
    logger.info(f"Attempting merge for potentially split Nyx: {nyx_id}")
    try:
        # Get the potentially multiple instances/states via get_instance logic or load checkpoints directly
        # For simplicity, we get the 'main' instance and trigger its restore/merge logic
        # Assuming user_id/conv_id 0/0 is used for the global Nyx ID instance tracking
        brain = await NyxBrain.get_instance(0, 0, nyx_id=nyx_id)

        # Check if brain is initialized before attempting restore
        if not brain.initialized:
             logger.warning(f"Skipping merge for {nyx_id}: Corresponding brain instance is not initialized.")
             return False

        success = await brain.restore_entity_from_distributed_checkpoints()
        if success:
            logger.info(f"Successfully processed/merged state for Nyx: {nyx_id}")
            return True
        else:
            logger.warning(f"State restoration/merge process indicated no action or failure for Nyx: {nyx_id}")
            return False
    except Exception as e:
        logger.error(f"Failed to merge {nyx_id}: {e}", exc_info=True)
        return False

# --- Modified Sweep Task ---
@celery_app.task
@async_task
async def sweep_and_merge_nyx_split_brains():
    """
    Celery task for periodically merging split-brain Nyx instances.
    Checks if the main application is initialized before running.
    """
    logger.info("Checking application readiness for split-brain sweep...")
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
                if success: merged_count += 1
                else: failed_count += 1
                await asyncio.sleep(1) # Small delay between merges

        logger.info(f"Sweep-and-merge task completed. Merged: {merged_count}, Failed/Skipped: {failed_count}.")
        return {"status": "success", "merged": merged_count, "failed_or_skipped": failed_count}
    except Exception as e:
        logger.exception("Sweep-and-merge task failed critically.")
        return {"status": "error", "error": str(e)}

# --- NEW LLM Checkpointing Task ---
@celery_app.task
@async_task # Use decorator for the async logic
async def run_llm_periodic_checkpoint_task(user_id: int, conversation_id: int):
    """
    Celery task for periodically running the LLM-driven checkpointing.
    """
    nyx_id = os.getenv("NYX_ID", "nyx_v1") # Or determine based on user/conv if needed
    logger.info(f"Starting LLM periodic checkpoint task for NyxBrain {user_id}-{conversation_id} (NyxID: {nyx_id})...")

    if not await is_app_initialized():
        logger.info(f"Application not initialized yet. Skipping LLM checkpoint for {user_id}-{conversation_id}.")
        return {"status": "skipped", "reason": "App not initialized"}

    try:
        # Get the specific brain instance
        # Use nyx_id if you have a global instance per NYX_ID, otherwise use user/conv
        brain_instance = await NyxBrain.get_instance(user_id, conversation_id, nyx_id=nyx_id if user_id == 0 and conversation_id == 0 else None)

        if not brain_instance or not brain_instance.initialized:
            logger.warning(f"Could not get initialized NyxBrain instance for {user_id}-{conversation_id}. Skipping checkpoint.")
            return {"status": "skipped", "reason": "Brain instance not ready"}

        # 1. Gather current state
        logger.debug(f"Gathering state for {user_id}-{conversation_id}...")
        current_state = await brain_instance.gather_checkpoint_state(event="periodic_llm_scheduled")

        # 2. Get plan from agent
        logger.debug(f"Requesting checkpoint plan for {user_id}-{conversation_id}...")
        planner_agent = CheckpointingPlannerAgent() # Create agent instance
        # Pass brain_instance as context if planner's tools need it, else None
        checkpoint_plan = await planner_agent.recommend_checkpoint(current_state, brain_instance_for_context=brain_instance)

        # 3. Save based on plan
        if checkpoint_plan and checkpoint_plan.get("to_save"):
            logger.debug(f"Saving planned checkpoint for {user_id}-{conversation_id}...")
            # Extract data correctly from the plan structure
            data_to_save = checkpoint_plan["to_save"] # This is {"field": {"value": ..., "why_saved": ...}}
            justifications = {k: v.get("why_saved", "N/A") for k, v in data_to_save.items()}
            skipped = checkpoint_plan.get("skip_fields", [])

            await brain_instance.save_planned_checkpoint( # Call method on brain instance
                event="periodic", # Base event type
                data_to_save=data_to_save, # Pass the structured dict
                justifications=justifications,
                skipped=skipped
            )
            logger.info(f"LLM periodic checkpoint saved for {user_id}-{conversation_id}.")
            return {"status": "success", "saved_fields": len(data_to_save), "skipped_fields": len(skipped)}
        else:
            logger.info(f"Checkpoint planner recommended skipping save for {user_id}-{conversation_id}.")
            return {"status": "success", "saved_fields": 0, "skipped_fields": checkpoint_plan.get("skip_fields", ["No plan generated"])}

    except Exception as e:
        logger.exception(f"Error during LLM periodic checkpoint task for {user_id}-{conversation_id}")
        return {"status": "error", "error": str(e)}

# Ensure celery_app is correctly configured if needed elsewhere
app = celery_app
