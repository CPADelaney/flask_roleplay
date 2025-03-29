import os
import json
import logging
import asyncio
import asyncpg # Import asyncpg
from celery_config import celery_app # Import our dedicated Celery app

# Import your helper functions and task logic
from npcs.new_npc_creation import spawn_multiple_npcs_enhanced, spawn_multiple_npcs_through_nyx
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from new_game_agent import NewGameAgent
from npcs.npc_learning_adaptation import NPCLearningManager
from memory.memory_nyx_integration import run_maintenance_through_nyx
from db.connection import get_async_db_connection, get_db_connection_context # Import async context manager

logger = logging.getLogger(__name__)

# Define DSN globally or ensure it's available
DB_DSN = os.getenv("DB_DSN", "postgresql://user:password@host:port/database")
if not DB_DSN:
    logger.error("DB_DSN environment variable not set for Celery tasks!")

@celery_app.task
def test_task():
    logger.info("Executing test task!")
    return "Hello from dummy task!"

# --- New Task for NPC Learning Cycle ---
@celery_app.task
async def run_npc_learning_cycle_task():
    """
    Celery task to run the NPC learning cycle periodically for active conversations.
    Uses asyncpg for database access.
    """
    logger.info("Starting NPC learning cycle task via Celery Beat.")
    processed_conversations = 0
    try:
        # Use the async context manager for DB connection
        async with get_db_connection_context() as conn:
            # Find recent conversations (adjust interval as needed)
            convs = await conn.fetch("""
                SELECT id, user_id
                FROM conversations
                WHERE last_active > NOW() - INTERVAL '1 day'
            """) # Assuming 'last_active' column exists

            if not convs:
                logger.info("No recent conversations found for NPC learning.")
                return {"status": "success", "processed_conversations": 0}

            for conv_row in convs:
                conv_id = conv_row['id']
                user_id = conv_row['user_id']
                try:
                    # Fetch NPCs for this conversation using the same connection
                    npc_rows = await conn.fetch("""
                        SELECT npc_id FROM NPCStats
                        WHERE user_id=$1 AND conversation_id=$2
                    """, user_id, conv_id)

                    npc_ids = [row['npc_id'] for row in npc_rows]

                    if npc_ids:
                        # Run the learning logic (ensure NPCLearningManager uses asyncpg)
                        # Make NPCLearningManager accept an existing connection or pool if possible
                        # Or ensure it creates its own async connections internally
                        manager = NPCLearningManager(user_id, conv_id)
                        await manager.initialize() # Assuming this sets up async resources if needed
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
        # Depending on the error, you might want to raise it to trigger Celery retry mechanisms
        # raise self.retry(exc=e_outer, countdown=60)
        return {"status": "error", "message": str(e_outer)}

    logger.info(f"NPC learning cycle task finished. Processed {processed_conversations} conversations.")
    return {"status": "success", "processed_conversations": processed_conversations}


@celery_app.task
def process_new_game_task(user_id, conversation_data):
    """
    Celery task to run heavy game startup processing.
    This function uses the NewGameAgent to create a new game with Nyx governance.
    (Assumes NewGameAgent handles its own async DB correctly)
    """
    logger.info(f"Starting process_new_game_task for user_id={user_id}")
    try:
        # Create a NewGameAgent instance
        agent = NewGameAgent()

        # Run the async method using asyncio.run (appropriate for sync Celery task calling async code)
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

    async def main():
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

            # Spawn NPCs using your new approach
            # Ensure spawn_multiple_npcs_through_nyx is async and handles its own DB or accepts conn
            npc_ids = await spawn_multiple_npcs_through_nyx(
                user_id=user_id,
                conversation_id=conversation_id,
                environment_desc=environment_desc,
                day_names=day_names,
                count=count
                # Pass conn if the function accepts it: connection=conn
            )

            return {
                "message": f"Successfully created {len(npc_ids)} NPCs (via Nyx governance)",
                "npc_ids": npc_ids
            }

    try:
        # Run the async main function within the synchronous Celery task
        final_info = asyncio.run(main())
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
    (Assumes get_chatgpt_response is synchronous or handled appropriately)
    """
    logger.info(f"Async GPT task: Calling GPT for opening line for conv_id={conversation_id}.")

    # First attempt: normal GPT call
    # Ensure get_chatgpt_response doesn't block excessively if it's synchronous
    gpt_reply_dict = get_chatgpt_response(
        conversation_id=conversation_id,
        aggregator_text=aggregator_text,
        user_input=opening_user_prompt
    )

    # Check if a fallback is needed (handle potential errors in get_chatgpt_response)
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
                model=os.getenv("OPENAI_MODEL", "gpt-4o"), # Use env var for model
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
        # Use the async context manager
        async with get_db_connection_context() as conn:
            # Same query to find relevant user_id + conversation_id
            rows = await conn.fetch("""
                SELECT DISTINCT user_id, conversation_id
                FROM NyxMemories -- Or unified_memories if that's the target
                WHERE is_archived = FALSE
                AND timestamp > NOW() - INTERVAL '30 days'
                -- Add other conditions as necessary
            """)

            if not rows:
                logger.info("No conversations found with recent Nyx memories to maintain")
                return {"status": "success", "conversations_processed": 0}

            for row in rows:
                user_id = row["user_id"]
                conversation_id = row["conversation_id"]

                try:
                    # Ensure run_maintenance_through_nyx is async
                    # and handles its DB needs or accepts the connection
                    await run_maintenance_through_nyx(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        entity_type="nyx", # Adjust as needed
                        entity_id=0 # Adjust as needed
                        # Pass conn if accepted: connection=conn
                    )
                    processed_count += 1
                    logger.info(f"Completed governed memory maintenance for user={user_id}, conv={conversation_id}")
                except Exception as e:
                    logger.error(f"Governed maintenance error user={user_id}, conv={conversation_id}: {e}", exc_info=True)
                    # Continue with the next conversation

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

# Assign celery_app to 'app' if needed for discovery, although explicit -A tasks should work
app = celery_app
