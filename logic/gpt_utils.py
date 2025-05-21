# logic/gpt_utils.py - Enhanced with robust JSON parsing and async support

import json
import re
import logging
import asyncio

from logic.chatgpt_integration import get_chatgpt_response


def is_rate_limit_error(exception_obj: Exception) -> bool:
    """
    Checks if an exception is likely a rate-limit or 429 error.
    """
    err_str = str(exception_obj).lower()
    return ('429' in err_str) or ('ratelimit' in err_str) or ('rate limit' in err_str)


async def spaced_gpt_call(conversation_id, context, prompt, delay=1.0, max_retries=5):
    """
    Calls get_chatgpt_response with an initial 'delay' wait and
    exponential backoff on 429 Too Many Requests errors.
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
            # Now calling the async function directly
            result = await get_chatgpt_response(conversation_id, context, prompt)
            logging.info("GPT returned response on attempt %d: %s", attempt, result)
            return result  # success

        except Exception as e:
            if is_rate_limit_error(e):
                logging.warning("Got a 429/rate-limit error on attempt %d: %s", attempt, e)
                if attempt < max_retries:
                    attempt += 1
                    wait_time *= 2
                    continue
                else:
                    logging.error("Max retries reached. Re-raising the rate-limit error.")
                    raise
            else:
                logging.error("Non-429 error from GPT call: %s", e, exc_info=True)
                raise

    raise RuntimeError("spaced_gpt_call ended unexpectedly without returning or raising.")


def parse_json_str(text: str) -> dict:
    """
    Safely parse a string into JSON. Tries removing code fences, 
    attempts direct loads, then scanning with regex for a JSON block.
    Returns a dict or empty {} if all attempts fail.
    """
    if not text or not isinstance(text, str):
        return {}

    # Remove triple-backtick fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines.pop(0)
        if lines and lines[-1].startswith("```"):
            lines.pop()
        text = "\n".join(lines).strip()

    # Attempt direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt to extract first {...} block from the text via regex
    match = re.search(r'(\{[\s\S]*\})', text)
    if match:
        snippet = match.group(1)
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    # Attempt naive replacement of fancy quotes => standard quotes
    fix_quotes = text.replace(""", '"').replace(""", '"').replace("'", "'")
    try:
        return json.loads(fix_quotes)
    except json.JSONDecodeError:
        pass

    return {}


async def call_gpt_json(conversation_id, context, prompt, model="gpt-4.1-nano", temperature=0.7, max_retries=2) -> dict:
    """
    Calls GPT with the given context and prompt, attempting to parse valid JSON from the response.
    If it fails, tries multiple fallback methods. Returns a Python dict or empty {}.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"[call_gpt_json] Attempt {attempt}/{max_retries}")
            response = await spaced_gpt_call(conversation_id, context, prompt, delay=1.0)
            # Possibly a "function_call" type
            if response.get("type") == "function_call":
                return response.get("function_args", {})
            else:
                raw_text = response.get("response", "").strip()
                parsed = parse_json_str(raw_text)
                if parsed:
                    return parsed
                else:
                    logging.warning(f"[call_gpt_json] GPT returned malformed JSON attempt {attempt}.")
        except Exception as e:
            logging.error(f"[call_gpt_json] Error calling GPT or parsing: {e}")

    return {}


def safe_int(val, default=1):
    """Converts val to int, returning default on fail."""
    try:
        return int(val)
    except:
        return default


def normalize_smart_quotes(text):
    """
    Replace smart quotes with straight quotes for JSON or general text compatibility.
    """
    if not text or not isinstance(text, str):
        return text
    return (text
            .replace("'", "'").replace("'", "'")
            .replace(""", '"').replace(""", '"'))

def extract_json_from_text(text: str) -> dict:
    """
    If you still want a direct 'JSON object in text' extraction, 
    you can use parse_json_str directly or keep this as a fallback.
    """
    # For backward compatibility if needed
    return parse_json_str(text)
