# logic/gpt_utils.py
import logging
import asyncio
from logic.chatgpt_integration import get_chatgpt_response

async def spaced_gpt_call(conversation_id, context, prompt, delay=1.0):
    logging.info("Waiting for %.1f seconds before calling GPT (conversation_id=%s)", delay, conversation_id)
    await asyncio.sleep(delay)
    logging.info("Calling GPT with conversation_id=%s", conversation_id)
    try:
        result = await asyncio.to_thread(get_chatgpt_response, conversation_id, context, prompt)
    except Exception as e:
        logging.exception("Error calling GPT for conversation_id=%s: %s", conversation_id, e)
        raise
    logging.info("GPT returned response: %s", result)
    return result

