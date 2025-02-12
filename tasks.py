# tasks.py
import os
import json
import logging
import asyncio
# We do NOT create a new Celery() here; we import the existing one from main.py
from main import celery_app

from logic.npc_creation import spawn_and_refine_npcs_with_relationships
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from game_processing import async_process_new_game


@celery_app.task
def test_task():
    return "Hello from dummy task!"


@celery_app.task
def process_new_game_task(user_id, conversation_data):
    """
    Celery task to run the heavy game startup processing.
    This function runs the asynchronous helper using asyncio.run().
    """
    try:
        result = asyncio.run(async_process_new_game(user_id, conversation_data))
        logging.info("Completed processing new game for user_id=%s", user_id)
        return result
    except Exception as e:
        logging.exception("Error in process_new_game_task for user_id=%s", user_id)
        return {"status": "failed", "error": str(e)}


@celery_app.task
def create_npcs_task(user_id, conversation_id, count=10):
    import asyncio

    # We'll provide placeholder environment info & day names,
    # or load them from DB if needed:
    environment_desc = "A fallback environment description"
    day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    # We call the async function using `asyncio.run(...)` inside the Celery task.
    from logic.npc_creation import spawn_and_refine_npcs_with_relationships
    from db.connection import get_async_db_connection  # or whichever async connection you use

    async def main():
        # 1) Open an async DB connection
        conn = await get_async_db_connection()

        # 2) Spawn the requested # of NPCs
        result = await spawn_and_refine_npcs_with_relationships(
            user_id=user_id,
            conversation_id=conversation_id,
            environment_desc=environment_desc,
            day_names=day_names,
            conn=conn,
            count=count
        )

        # 3) Clean up
        await conn.close()
        return result

    # Actually run the async logic
    final_info = asyncio.run(main())

    # `final_info` is whatever spawn_and_refine_npcs_with_relationships returns,
    # e.g. {"message": "Spawned 10 NPCs and assigned relationships."}
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
