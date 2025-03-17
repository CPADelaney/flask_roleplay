# tasks.py
import os
import json
import logging
import asyncio
from celery_config import celery_app  # Import our dedicated Celery app

# Import your helper functions and task logic
from logic.npc_creation import spawn_multiple_npcs, spawn_single_npc
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from new_game_agent import NewGameAgent

@celery_app.task
def test_task():
    return "Hello from dummy task!"

@celery_app.task
def process_new_game_task(user_id, conversation_data):
    """
    Celery task to run heavy game startup processing.
    This function uses the NewGameAgent to create a new game with Nyx governance.
    """
    try:
        # Create a NewGameAgent instance
        agent = NewGameAgent()
        
        # Call the process_new_game method
        result = asyncio.run(agent.process_new_game(user_id, conversation_data))
        logging.info("Completed processing new game for user_id=%s", user_id)
        return result
    except Exception as e:
        logging.exception("Error in process_new_game_task for user_id=%s", user_id)
        return {"status": "failed", "error": str(e)}

@celery_app.task
def create_npcs_task(user_id, conversation_id, count=10):
    import asyncio
    from db.connection import get_async_db_connection
    from logic.npc_creation import spawn_multiple_npcs

    async def main():
        # 1) Get an async connection
        conn = await get_async_db_connection()

        # 2) Fetch environment_desc from DB
        row_env = await conn.fetchrow("""
            SELECT value
            FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='EnvironmentDesc'
        """, user_id, conversation_id)
        environment_desc = row_env["value"] if row_env else "A fallback environment description"

        # 3) Fetch 'CalendarNames' from DB
        row_cal = await conn.fetchrow("""
            SELECT value
            FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='CalendarNames'
        """, user_id, conversation_id)
        if row_cal:
            cal_data = json.loads(row_cal["value"] or "{}")
            day_names = cal_data.get("days", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        else:
            day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

        # 4) Spawn NPCs using your new approach
        npc_ids = await spawn_multiple_npcs(
            user_id=user_id,
            conversation_id=conversation_id,
            environment_desc=environment_desc,
            day_names=day_names,
            count=count
        )

        await conn.close()

        # 5) Return info
        return {
            "message": f"Successfully created {len(npc_ids)} NPCs",
            "npc_ids": npc_ids,
            "day_names": day_names
        }

    final_info = asyncio.run(main())
    return final_info

@celery_app.task
def get_gpt_opening_line_task(conversation_id, aggregator_text, opening_user_prompt):
    """
    Generate the GPT opening line.
    This task calls the GPT API (or fallback) and returns a JSON-encoded reply.
    """
    logging.info("Async GPT task: Calling GPT for opening line.")
    
    # First attempt: normal GPT call
    gpt_reply_dict = get_chatgpt_response(
        conversation_id=conversation_id,
        aggregator_text=aggregator_text,
        user_input=opening_user_prompt
    )
    
    nyx_text = gpt_reply_dict.get("response")
    if gpt_reply_dict.get("type") == "function_call" or not nyx_text:
        logging.info("Async GPT task: GPT returned a function call or no text. Retrying without function calls.")
        client = get_openai_client()
        forced_messages = [
            {"role": "system", "content": aggregator_text},
            {"role": "user", "content": "No function calls. Produce only a text narrative.\n\n" + opening_user_prompt}
        ]
        fallback_response = client.chat.completions.create(
            model="gpt-4o",
            messages=forced_messages,
            temperature=0.7,
        )
        fallback_text = fallback_response.choices[0].message.content.strip()
        nyx_text = fallback_text if fallback_text else "[No text returned from GPT]"
        gpt_reply_dict["response"] = nyx_text
        gpt_reply_dict["type"] = "fallback"
        
    return json.dumps(gpt_reply_dict)

@celery_app.task
def nyx_memory_maintenance_task():
    """
    Celery task to perform regular maintenance on Nyx's memory system.
    - Consolidates related memories into patterns
    - Applies memory decay to older memories
    - Archives unimportant memories
    - Updates narrative arcs based on accumulated experiences
    Should run daily for optimal performance.
    """
    import asyncio
    import asyncpg
    import os
    from logic.nyx_memory_manager import perform_memory_maintenance
    
    logging.info("Starting Nyx memory maintenance task")
    
    async def process_all_conversations():
        dsn = os.getenv("DB_DSN")
        if not dsn:
            logging.error("DB_DSN environment variable not set")
            return {"status": "error", "error": "DB_DSN not set"}
            
        conn = None
        try:
            conn = await asyncpg.connect(dsn)
            
            # Get active conversations
            rows = await conn.fetch("""
                SELECT DISTINCT user_id, conversation_id
                FROM NyxMemories
                WHERE is_archived = FALSE
                AND timestamp > NOW() - INTERVAL '30 days'
            """)
            
            if not rows:
                logging.info("No conversations found with Nyx memories to maintain")
                return {"status": "success", "conversations_processed": 0}
            
            processed_count = 0
            for row in rows:
                user_id = row["user_id"]
                conversation_id = row["conversation_id"]
                
                try:
                    await perform_memory_maintenance(user_id, conversation_id)
                    processed_count += 1
                    logging.info(f"Memory maintenance completed for user_id={user_id}, conversation_id={conversation_id}")
                except Exception as e:
                    logging.error(f"Error in memory maintenance for user_id={user_id}, conversation_id={conversation_id}: {str(e)}")
                    
                # Brief pause between processing to avoid overloading the database
                await asyncio.sleep(0.5)
                
            return {
                "status": "success", 
                "conversations_processed": processed_count
            }
                
        except Exception as e:
            logging.error(f"Error in nyx_memory_maintenance_task: {str(e)}")
            return {"status": "error", "error": str(e)}
        finally:
            if conn:
                await conn.close()
    
    # Run the async function and return the result
    result = asyncio.run(process_all_conversations())
    logging.info(f"Nyx memory maintenance task completed: {result}")
    return result
