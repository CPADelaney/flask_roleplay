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
    
def safe_json_loads(s, max_trim=100):
    """
    Attempt to parse a JSON string.
    If it fails, iteratively trim off the last character (up to max_trim times)
    until parsing succeeds. Returns the parsed JSON or an empty dict.
    """
    original = s
    for i in range(max_trim):
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            # Log the error and trim one character off the end.
            s = s[:-1]
    logging.error("Could not safely parse JSON after trimming %d characters. Original string: %s", max_trim, original)
    return {}
