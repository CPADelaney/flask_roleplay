# tasks.py
import os
import json
import logging
import asyncio
import asyncpg
from main import celery_app
from logic.npc_creation import spawn_multiple_npcs, spawn_single_npc
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from game_processing import async_process_new_game
from db.connection import get_db_connection
from logic.universal_updater import apply_universal_updates

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
    from db.connection import get_async_db_connection
    from logic.npc_creation import spawn_multiple_npcs

    async def main():
        # 1) Get an async connection (if needed)
        conn = await get_async_db_connection()

        # 2) Fetch environment_desc from DB
        row_env = await conn.fetchrow("""
            SELECT value
            FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='EnvironmentDesc'
        """, user_id, conversation_id)
        environment_desc = row_env["value"] if row_env else "A fallback environment description"

        # 3) Fetch CalendarNames for day names
        row_cal = await conn.fetchrow("""
            SELECT value
            FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='CalendarNames'
        """, user_id, conversation_id)
        if row_cal:
            cal_data = json.loads(row_cal["value"] or "{}")
            day_names = cal_data.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        else:
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        # 4) Spawn NPCs
        npc_ids = await spawn_multiple_npcs(
            user_id=user_id,
            conversation_id=conversation_id,
            environment_desc=environment_desc,
            day_names=day_names,
            count=count
        )
        await conn.close()
        return {
            "message": f"Successfully created {len(npc_ids)} NPCs",
            "npc_ids": npc_ids,
            "day_names": day_names
        }

    return asyncio.run(main())


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
def process_storybeat_task(user_id, conversation_id, aggregator_text, user_input):
    """
    Celery task that generates the narrative and a complete state update in one GPT call,
    then applies the updates to the database and stores the narrative.
    """
    async def main():
        try:
            # Call GPT once to get the narrative and the complete update payload.
            response = get_chatgpt_response(conversation_id, aggregator_text, user_input)
            # We assume GPT returns a function_call response with our update.
            narrative = response.get("response") or "[No narrative generated]"
            update_payload = {}
            if response.get("type") == "function_call":
                update_payload = response.get("function_args", {})
            
            # Apply the update payload if provided.
            if update_payload:
                dsn = os.getenv("DB_DSN")
                async_conn = await asyncpg.connect(dsn=dsn)
                result = await apply_universal_updates(user_id, conversation_id, update_payload, async_conn)
                await async_conn.close()
                logging.info("State update result: %s", result)
            
            # Insert the narrative into the messages table.
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO messages (conversation_id, sender, content) VALUES (%s, %s, %s)",
                (conversation_id, "assistant", narrative)
            )
            conn.commit()
            cur.close()
            conn.close()
            
            return {"status": "success", "narrative": narrative}
        except Exception as e:
            logging.exception("Error in process_storybeat_task:")
            return {"status": "failed", "error": str(e)}
    
    return asyncio.run(main())

