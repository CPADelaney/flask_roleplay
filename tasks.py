# tasks.py

import os
import json
import logging
import asyncio
import asyncpg
import datetime
from celery_config import celery_app
from functools import wraps

# Import your helper functions and task logic
from npcs.new_npc_creation import NPCCreationHandler
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from new_game_agent import NewGameAgent
from npcs.npc_learning_adaptation import NPCLearningManager
from memory.memory_nyx_integration import run_maintenance_through_nyx
from db.connection import get_db_connection_context

from nyx.core.brain.base import NyxBrain

logger = logging.getLogger(__name__)

# Define DSN globally or ensure it's available
DB_DSN = os.getenv("DB_DSN", "postgresql://user:password@host:port/database")
if not DB_DSN:
    logger.error("DB_DSN environment variable not set for Celery tasks!")

# Helper decorator to run async functions in Celery tasks
def async_task(func):
    """Decorator to run async functions in synchronous Celery tasks."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
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
    logger.info(f"[BG Task {conversation_id}] Starting for user {user_id}")
    try:
        # Get aggregator context
        from logic.aggregator import get_aggregated_roleplay_context
        aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, "Chase")

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
                aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, context["player_name"])
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

@celery_app.task
def process_new_game_task(user_id, conversation_data):
    """
    Celery task to run heavy game startup processing.
    This function uses the NewGameAgent to create a new game with Nyx governance.
    """
    logger.info(f"Starting process_new_game_task for user_id={user_id}")
    try:
        # Create a NewGameAgent instance
        agent = NewGameAgent()

        # Run the async method using asyncio.run
        result = asyncio.run(agent.process_new_game(user_id, conversation_data))
        logger.info(f"Completed processing new game for user_id={user_id}. Result: {result}")
        return result
    except Exception as e:
        logger.exception(f"Error in process_new_game_task for user_id={user_id}")
        # Return a serializable error structure
        return {"status": "failed", "error": str(e)}

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
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
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

async def find_split_brain_nyxes():
    """
    Find all nyx_id's with >1 recent checkpoints (i.e., split-brain state).
    """
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(minutes=30)
    async with get_db_connection_context() as conn:
        rows = await conn.fetch("""
            SELECT nyx_id, count(*)
            FROM nyx_brain_checkpoints
            WHERE checkpoint_time > $1
            GROUP BY nyx_id
            HAVING count(distinct instance_id) > 1
        """, cutoff)
    return [row["nyx_id"] for row in rows]

async def perform_sweep_and_merge():
    """
    For each split-brain Nyx, call the merge/restore logic.
    """
    split_nyxes = await find_split_brain_nyxes()
    if not split_nyxes:
        logger.info("No splits found.")
        return

    logger.info(f"Found split-brain Nyxes: {split_nyxes}")
    for nyx_id in split_nyxes:
        try:
            brain = await NyxBrain.get_instance(0, 0, nyx_id=nyx_id)
            await brain.restore_entity_from_distributed_checkpoints()
            logger.info(f"Successfully merged split-brain Nyx: {nyx_id}")
        except Exception as e:
            logger.error(f"Failed to merge {nyx_id}: {e}", exc_info=True)

@celery_app.task
def sweep_and_merge_nyx_split_brains():
    """
    Celery task for periodically merging split-brain Nyx instances.
    """
    logger.info("Starting split-brain Nyx sweep-and-merge task")
    try:
        asyncio.run(perform_sweep_and_merge())
        logger.info("Sweep-and-merge task completed successfully.")
        return {"status": "success"}
    except Exception as e:
        logger.exception("Sweep-and-merge task failed")
        return {"status": "error", "error": str(e)}

@celery_app.task
def memory_embedding_consolidation_task():
    """
    Celery task to consolidate memory embeddings.
    """
    logger.info("Starting memory embedding consolidation task")
    
    async def consolidate_embeddings():
        try:
            from memory.memory_integration import consolidate_memory_embeddings
            
            result = await consolidate_memory_embeddings()
            logger.info(f"Memory embedding consolidation completed: {result}")
            return result
        except Exception as e:
            logger.exception("Memory embedding consolidation failed")
            return {"status": "error", "error": str(e)}
    
    try:
        return asyncio.run(consolidate_embeddings())
    except Exception as e:
        logger.exception("Memory embedding consolidation task failed")
        return {"status": "error", "error": str(e)}

# Assign celery_app to 'app' if needed for discovery
app = celery_app
