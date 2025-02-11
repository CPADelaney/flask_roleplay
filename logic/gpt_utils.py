# logic/gpt_utils.py
import logging
import asyncio
from logic.chatgpt_integration import get_chatgpt_response

import asyncio
import logging
import time
from logic.chatgpt_integration import get_chatgpt_response

async def spaced_gpt_call(
    conversation_id,
    context,
    prompt,
    delay=1.0,
    max_retries=5
):
    """
    Calls get_chatgpt_response with an initial 'delay' wait and
    exponential backoff on 429 Too Many Requests errors.
    
    Args:
      conversation_id (int): conversation context
      context (str): aggregator or environment text
      prompt (str): user/system prompt
      delay (float): initial delay before first call
      max_retries (int): maximum number of attempts on 429 errors
      
    Returns:
      dict: GPT response structure from get_chatgpt_response
    """
    attempt = 1
    wait_time = delay

    while attempt <= max_retries:
        logging.info(
            "spaced_gpt_call: attempt %d/%d for conversation_id=%s. Waiting %.1f sec, then calling GPT...",
            attempt, max_retries, conversation_id, wait_time
        )
        await asyncio.sleep(wait_time)
        
        try:
            # Actually make the GPT call in a separate thread.
            result = await asyncio.to_thread(get_chatgpt_response, conversation_id, context, prompt)
            logging.info("GPT returned response on attempt %d: %s", attempt, result)
            return result  # success
        except Exception as e:
            # If we see '429' or 'RateLimit' in the exception, treat it as a rate-limit scenario
            err_str = str(e).lower()
            if '429' in err_str or 'ratelimit' in err_str or 'rate limit' in err_str:
                logging.warning("Got a 429/rate-limit error on attempt %d: %s", attempt, e)
                if attempt < max_retries:
                    # exponential backoff
                    attempt += 1
                    wait_time *= 2
                    continue
                else:
                    logging.error("Max retries reached. Re-raising the rate-limit error.")
                    raise
            else:
                # Some other error
                logging.error("Non-429 error from GPT call: %s", e, exc_info=True)
                raise

    raise RuntimeError("spaced_gpt_call ended unexpectedly without returning or raising.")  # safety
