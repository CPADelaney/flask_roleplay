# logic/gpt_helpers.py

import json
import logging
import copy
from db.connection import get_db_connection
from logic.gpt_utils import spaced_gpt_call  # or spaced_gpt_call_with_retry

# The keys we consider "required" for a complete NPC
REQUIRED_KEYS = ["likes", "dislikes", "hobbies", "affiliations", "schedule"]

async def adjust_npc_complete(npc_data, environment_desc, conversation_id, immersive_days=None, max_retries=3):
    """
    Adapt NPC details in multiple partial GPT calls if necessary. 
    Instead of requiring GPT to return everything at once, we gather partial 
    data and only re-ask for missing keys.

    npc_data: dict with partial info about the NPC (e.g. likes, etc.)
    environment_desc: the broader environment/story setting.
    conversation_id: for logging or GPT call context.
    immersive_days: list of custom day names for schedule
    max_retries: how many times we keep asking GPT for missing keys

    Returns: dict with the keys ["likes","dislikes","hobbies","affiliations","schedule"] 
             (any missing keys are filled with empty lists/objects).
    """

    if immersive_days is None:
        immersive_days = ["Sol", "Luna", "Terra", "Vesta", "Mercury", "Venus", "Mars"]

    # We'll keep merging data into 'refined_npc'
    refined_npc = copy.deepcopy(npc_data)

    # Figure out which keys are initially missing or empty
    missing_keys = set(REQUIRED_KEYS)
    # If refined_npc already has any required keys with non-empty data, remove them from missing
    for key in list(missing_keys):
        # If it exists and isn't empty, consider it "filled"
        val = refined_npc.get(key, None)
        if val:  # e.g. if likes is already a non-empty list
            missing_keys.remove(key)

    retry_count = 0
    while missing_keys and retry_count < max_retries:
        missing_list_str = ", ".join(missing_keys)
        # Build a prompt *only* for the missing pieces.
        # We also show the partial data we already have, 
        # so GPT can keep it in context.
        prompt = f"""
We have an NPC in a femdom environment. We already have partial data:
{json.dumps(refined_npc, indent=2)}

We are missing these keys: {missing_list_str}.

We want a JSON object containing ONLY these missing keys. 
If you can't provide some key, set it to an empty array ([]) or empty object ({{}}).

Immersive day names for schedule: {', '.join(immersive_days)}

Return strictly JSON, no extra text or function calls.
"""

        logging.info(f"[adjust_npc_complete] Attempt={retry_count+1}, missing_keys={missing_keys}")
        reply = await spaced_gpt_call(
            conversation_id=conversation_id,
            context=environment_desc,
            prompt=prompt,
            delay=1.0,
            max_retries=3  # or however many times you want to handle 429
        )

        # Check what we got back
        if reply.get("type") == "function_call":
            # GPT returned a function call
            function_args = reply.get("function_args", {})
            # Merge only the missing keys from function_args
            for mk in list(missing_keys):
                if mk in function_args:
                    refined_npc[mk] = function_args[mk]
                    missing_keys.remove(mk)

        elif reply.get("response"):
            # Plain text JSON
            response_text = reply["response"].strip()
            # remove fences
            if response_text.startswith("```"):
                lines = response_text.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()

            try:
                parsed_data = json.loads(response_text)
                # Merge only the missing keys
                for mk in list(missing_keys):
                    if mk in parsed_data:
                        refined_npc[mk] = parsed_data[mk]
                        missing_keys.remove(mk)
            except Exception as e:
                logging.warning(f"Failed to parse JSON in partial refine: {e}")

        retry_count += 1

    # If still missing anything after all attempts, fill with defaults
    for mk in missing_keys:
        # e.g. empty list or object
        if mk == "schedule":
            refined_npc[mk] = {}
        else:
            refined_npc[mk] = []

    # refined_npc now has all required keys
    return refined_npc

def fetch_npc_name(user_id, conversation_id, npc_id) -> str:
    """
    Returns the 'npc_name' from NPCStats for the given npc_id/user_id/conversation_id,
    or None if not found.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT npc_name
        FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        LIMIT 1
    """, (user_id, conversation_id, npc_id))
    row = cursor.fetchone()
    conn.close()

    if row:
        return row[0]  # the npc_name
    return None

