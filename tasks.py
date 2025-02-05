# tasks.py
import json
import logging
from celery import Celery
from logic.npc_creation import create_npc
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client

# Configure the Celery app.
# Adjust the broker URL (e.g., using Redis or RabbitMQ) as needed.
celery_app = Celery('tasks', broker='redis://redis:6379/0')

@celery_app.task
def create_npcs_task(user_id, conversation_id, count=10):
    """Create a given number of NPCs asynchronously."""
    npc_ids = []
    for i in range(count):
        new_id = create_npc(user_id=user_id, conversation_id=conversation_id, introduced=False)
        logging.info(f"Created NPC {i+1}/{count} with ID: {new_id}")
        npc_ids.append(new_id)
    return npc_ids

@celery_app.task
def get_gpt_opening_line_task(conversation_id, aggregator_text, opening_user_prompt):
    """
    Generate the GPT opening line.
    This task calls the GPT API (or its fallback) and returns a JSON-encoded reply.
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
            {"role": "user", "content": (
                "No function calls for the introduction. Produce only text narrative.\n\n" +
                opening_user_prompt)}
        ]
        fallback_response = client.chat.completions.create(
            model="gpt-4o",
            messages=forced_messages,
            temperature=0.7,
        )
        fallback_text = fallback_response.choices[0].message.content.strip()
        nyx_text = fallback_text if fallback_text else "[No text returned from GPT]"
        # Update the reply dictionary to indicate fallback was used.
        gpt_reply_dict["response"] = nyx_text
        gpt_reply_dict["type"] = "fallback"
    return json.dumps(gpt_reply_dict)
