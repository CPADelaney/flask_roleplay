# logic/gpt_utils.py - Enhanced with robust JSON parsing

import json
import re
import logging
import asyncio
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

    raise RuntimeError("spaced_gpt_call ended unexpectedly without returning or raising.")

async def call_gpt_json(
    conversation_id, 
    context, 
    prompt, 
    model="gpt-4o", 
    temperature=0.7, 
    max_retries=2
) -> dict:
    """
    Calls GPT with the given context and prompt, attempting to parse valid JSON from the response.
    If it fails, tries multiple fallback methods. Returns a Python dict or empty {}.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"[call_gpt_json] Attempt {attempt}/{max_retries}")
            response = await spaced_gpt_call(conversation_id, context, prompt, delay=1.0)
            
            # Check response type
            if response.get("type") == "function_call":
                # If we received a function call, return its arguments
                return response.get("function_args", {})
            else:
                # If text response, try to parse JSON
                raw_text = response.get("response", "").strip()
                
                # Basic check if it starts/ends with braces
                parsed = attempt_json_parse(raw_text)
                if parsed is not None:
                    return parsed
                
                logging.warning(f"[call_gpt_json] GPT returned malformed JSON attempt {attempt}. Trying fallback extraction.")
                
                fallback = extract_json_from_text(raw_text)
                if fallback is not None:
                    return fallback
        except Exception as e:
            logging.error(f"[call_gpt_json] Error calling GPT or parsing: {e}")
    
    # If all fails:
    return {}

def attempt_json_parse(text: str) -> dict:
    """Try direct JSON parse; return None on fail."""
    if not text or not isinstance(text, str):
        return None
        
    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        
    if not text.startswith("{") or not text.endswith("}"):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def extract_json_from_text(text: str) -> dict:
    """
    Looks for a JSON object in `text` using regex,
    tries to parse it. Return a dict or None.
    """
    if not text or not isinstance(text, str):
        return None
        
    json_match = re.search(r'(\{[\s\S]*\})', text)
    if json_match:
        snippet = json_match.group(1)
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None
    return None

def safe_int(val, default=1):
    """
    Attempts to convert val to int.
    If val is None or invalid, returns default.
    """
    try:
        return int(val)
    except:
        return default

def normalize_smart_quotes(text):
    """Replace smart quotes with straight quotes for JSON compatibility."""
    if not text or not isinstance(text, str):
        return text
    return (text.replace("'", "'")
               .replace("'", "'")
               .replace(""", '"')
               .replace(""", '"'))
