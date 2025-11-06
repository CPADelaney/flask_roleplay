# logic/gpt_utils.py - Enhanced with robust JSON parsing and async support

import json
import re
import logging
import asyncio
from typing import Any, Dict, Optional

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
            result = await get_chatgpt_response(
                conversation_id,
                context,
                prompt,
                use_nyx_integration=False,
            )
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

    normalized_text = normalize_smart_quotes(text)

    # Attempt direct parse
    try:
        return json.loads(normalized_text)
    except json.JSONDecodeError:
        pass

    # Attempt to extract first {...} block from the text via regex
    match = re.search(r'(\{[\s\S]*\})', normalized_text)
    if match:
        snippet = match.group(1)
        snippet = normalize_smart_quotes(snippet)
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    return {}


def parse_json_from_response(raw_text: str) -> Optional[Dict[str, Any]]:
    """Robustly extracts and parses a JSON object from a raw LLM response string."""
    if not raw_text:
        return None
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
    if match:
        json_str = match.group(1)
    else:
        json_str = raw_text
    start_brace = json_str.find('{')
    start_bracket = json_str.find('[')
    if start_brace == -1:
        start_index = start_bracket
    elif start_bracket == -1:
        start_index = start_brace
    else:
        start_index = min(start_brace, start_bracket)
    if start_index == -1:
        return None
    end_brace = json_str.rfind('}')
    end_bracket = json_str.rfind(']')
    end_index = max(end_brace, end_bracket)
    if end_index == -1:
        return None
    json_str = json_str[start_index : end_index + 1]
    json_str = re.sub(r",\s*([\}\]])", r"\1", json_str)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.warning(f"[parse_json_from_response] Final JSON parsing failed after cleaning: {e}")
        return None


def _extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first top-level JSON object from text using brace balance scanning."""
    if not text:
        return None

    start_idx = text.find("{")
    if start_idx < 0:
        return None

    depth = 0
    for idx in range(start_idx, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start_idx : idx + 1].strip()
                candidate = re.sub(r",\s*}", "}", candidate)
                candidate = re.sub(r",\s*]", "]", candidate)
                return candidate

    return None


async def call_gpt_json(
    conversation_id: str,
    context: str,
    prompt: str,
    model: str = "gpt-5-nano",
    max_retries: int = 2
) -> Dict[str, Any]:
    """
    Calls GPT for world-building, forcing a direct JSON response.
    This is the function that your lore generator should be using.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"[call_gpt_json] Attempt {attempt}/{max_retries}")
            
            logging.info(
                f"[call_gpt_json] Sending request to model '{model}' with context '{context}'.\n"
                f"--- PROMPT START ---\n{prompt}\n--- PROMPT END ---"
            )

            # --- KEY CHANGE: This function now *always* calls get_chatgpt_response with force_json_response=True ---
            response = await get_chatgpt_response(
                conversation_id=int(conversation_id),
                aggregator_text=context, # Pass context as aggregator text
                user_input=prompt,
                use_nyx_integration=False, # We don't want the full Nyx pipeline here
                force_json_response=True # THIS IS THE FIX
            )

            raw_text = response.get("response", "").strip()

            logging.info("RAW RESPONSE FROM GPT: ---BEGIN---\n%s\n---END---", raw_text)

            if not raw_text:
                logging.warning(f"[call_gpt_json] GPT returned an empty response string on attempt {attempt}.")
                continue

            parsed_json = parse_json_from_response(raw_text)
            if parsed_json is not None:
                return parsed_json

            cleaned = _extract_first_json_object(raw_text)
            if cleaned:
                parsed_json = parse_json_from_response(cleaned)
                if parsed_json is not None:
                    return parsed_json

            logging.warning(
                "[call_gpt_json] Failed to parse valid JSON; returning partial payload with status flag"
            )
            return {
                "_status": "partial_parse_failed",
                "_raw": raw_text[:4000],
            }

        except Exception as e:
            logging.error(f"[call_gpt_json] An unexpected error occurred on attempt {attempt}: {e}", exc_info=True)

    logging.error(f"[call_gpt_json] All {max_retries} attempts failed. Returning empty dict.")
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
    translation_table = {
        ord("\u2018"): "'",  # LEFT SINGLE QUOTATION MARK
        ord("\u2019"): "'",  # RIGHT SINGLE QUOTATION MARK
        ord("\u201A"): "'",  # SINGLE LOW-9 QUOTATION MARK
        ord("\u201B"): "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
        ord("\u201C"): '"',  # LEFT DOUBLE QUOTATION MARK
        ord("\u201D"): '"',  # RIGHT DOUBLE QUOTATION MARK
        ord("\u201E"): '"',  # DOUBLE LOW-9 QUOTATION MARK
        ord("\u201F"): '"',  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    }
    return text.translate(translation_table)

def extract_json_from_text(text: str) -> dict:
    """
    If you still want a direct 'JSON object in text' extraction, 
    you can use parse_json_str directly or keep this as a fallback.
    """
    # For backward compatibility if needed
    return parse_json_str(text)
