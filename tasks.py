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

async def background_chat_task_with_memory(conversation_id, user_input, user_id, universal_update=None):
    """
    Enhanced background chat task that includes memory retrieval.
    """
    global socketio
    if not socketio:
        logger.error("SocketIO instance not available in background_chat_task")
        return

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
                socketio.emit('error', {'error': 'Failed to apply world updates.'}, room=str(conversation_id))
                return

        # NEW: Enrich context with relevant memories
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
            socketio.emit('error', {'error': error_msg}, room=str(conversation_id))
            return

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
            
            # NEW: Store as memory as well
            try:
                from memory.memory_integration import add_memory_from_message
                
                # Add AI response as a memory
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
                # Continue even if memory storage fails
                
        except Exception as db_err:
            logger.error(f"[BG Task {conversation_id}] DB Error storing Nyx response: {db_err}", exc_info=True)

        # Check if we should generate an image
        should_generate = response.get("generate_image", False)
        if "function_args" in response and "image_generation" in response["function_args"]:
            img_settings = response["function_args"]["image_generation"]
            should_generate = should_generate or img_settings.get("generate", False)

        # Generate image if needed
        if should_generate:
            logger.info(f"[BG Task {conversation_id}] Image generation triggered.")
            try:
                img_data = {
                    "narrative": message_content,
                    "image_generation": response.get("function_args", {}).get("image_generation", {
                        "generate": True, "priority": "medium", "focus": "balanced",
                        "framing": "medium_shot", "reason": "Narrative moment"
                    })
                }
                res = await generate_roleplay_image_from_gpt(img_data, user_id, conversation_id)

                if res and "image_urls" in res and res["image_urls"]:
                    image_url = res["image_urls"][0]
                    prompt_used = res.get('prompt_used', '')
                    reason = img_data["image_generation"].get("reason", "Narrative moment")
                    logger.info(f"[BG Task {conversation_id}] Image generated: {image_url}")
                    socketio.emit('image', {
                        'image_url': image_url, 'prompt_used': prompt_used, 'reason': reason
                    }, room=str(conversation_id))
                else:
                    logger.warning(f"[BG Task {conversation_id}] Image generation task ran but produced no valid URLs.")
            except Exception as img_err:
                logger.error(f"[BG Task {conversation_id}] Error generating image: {img_err}", exc_info=True)

        # Stream the text tokens
        if message_content:
            logger.debug(f"[BG Task {conversation_id}] Streaming tokens...")
            chunk_size = 5
            delay = 0.01
            for i in range(0, len(message_content), chunk_size):
                token = message_content[i:i+chunk_size]
                socketio.emit('new_token', {'token': token}, room=str(conversation_id))
                await asyncio.sleep(delay)

            socketio.emit('done', {'full_text': message_content}, room=str(conversation_id))
            logger.info(f"[BG Task {conversation_id}] Finished streaming response.")
        else:
            logger.warning(f"[BG Task {conversation_id}] No message content to stream.")
            socketio.emit('done', {'full_text': ''}, room=str(conversation_id))

    except Exception as e:
        logger.error(f"[BG Task {conversation_id}] Critical Error: {str(e)}", exc_info=True)
        socketio.emit('error', {'error': f"Server error processing message: {str(e)}"}, room=str(conversation_id))

# --- Memory System Celery Tasks ---

@celery_app.task
def process_memory_embedding_task(user_id, conversation_id, message_text, entity_type="memory", metadata=None):
    """
    Celery task to process a memory embedding asynchronously.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        message_text: Message text
        entity_type: Entity type (memory, npc, location, narrative)
        metadata: Optional metadata
        
    Returns:
        Dictionary with task result
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
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        query_text: Query text
        entity_types: List of entity types to search
        top_k: Number of results to return
        
    Returns:
        Dictionary with task result
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
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        query_text: Query text
        entity_types: List of entity types to search
        top_k: Number of results to return
        
    Returns:
        Dictionary with task result
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
