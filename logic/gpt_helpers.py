# logic/gpt_helpers.py
from logic.gpt_utils import spaced_gpt_call  # Import from the separate GPT utils module
import httpx
import json
import logging
import os
import asyncio

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# The JSON schema as explained:
npc_schema = {
    "name": "NPCDetails",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "likes": {
                "type": "array",
                "items": {"type": "string"}
            },
            "dislikes": {
                "type": "array",
                "items": {"type": "string"}
            },
            "hobbies": {
                "type": "array",
                "items": {"type": "string"}
            },
            "affiliations": {
                "type": "array",
                "items": {"type": "string"}
            },
            "schedule": {
                "type": "object",
                "description": "A dictionary keyed by day name, each day is {Morning,Afternoon,Evening,Night}",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "Morning": {"type": "string"},
                        "Afternoon": {"type": "string"},
                        "Evening": {"type": "string"},
                        "Night": {"type": "string"}
                    },
                    "required": ["Morning","Afternoon","Evening","Night"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["likes","dislikes","hobbies","affiliations","schedule"],
        "additionalProperties": False
    }
}

async def adjust_npc_complete(npc_data, environment_desc, conversation_id, immersive_days=None):
    """
    Adapt NPC details by combining preference adjustments and schedule/affiliation generation
    into one GPT call using Structured Outputs and JSON schema.

    The GPT call ensures we get exactly these keys:
      - likes (array of strings)
      - dislikes (array of strings)
      - hobbies (array of strings)
      - affiliations (array of strings)
      - schedule (object keyed by day names, each day has Morning,Afternoon,Evening,Night)

    Returns a dict with the five fields. If the model refuses or something fails,
    we return fallback from npc_data.
    """
    # Default day names if none provided
    if immersive_days is None:
        immersive_days = ["Sol","Luna","Terra","Vesta","Mercury","Venus","Mars"]

    archetype_summary = npc_data.get("archetype_summary", "")
    likes = npc_data.get("likes", [])
    dislikes = npc_data.get("dislikes", [])
    hobbies = npc_data.get("hobbies", [])

    # Build your system prompt or message
    # We'll just do a single "system" style message for clarity:
    system_text = (
        "You are adjusting an NPC for a femdom environment. "
        "Given the existing likes, dislikes, and hobbies, produce an updated set that fits. "
        "Also produce 'affiliations' and a 'schedule'. "
        "Use these day names for the schedule: "
        f"{', '.join(immersive_days)}. "
        "Schedule must have keys for each day, each day has Morning/Afternoon/Evening/Night. "
        "Return only JSON adhering to the schema, no extra keys or text."
    )

    # The user message might pass in more context if needed. Let's keep it simple:
    user_text = (
        f"NPC Archetype Summary: {archetype_summary}\n\n"
        f"Likes: {likes}\nDislikes: {dislikes}\nHobbies: {hobbies}\n"
        f"Environment: {environment_desc}\n\n"
        "Please produce final JSON with keys: likes, dislikes, hobbies, affiliations, schedule."
    )

    request_body = {
        "model": "gpt-4o-2024-08-06",  # or a newer model that supports structured outputs
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text}
        ],
        # The key part: response_format specifying the JSON schema
        "response_format": {
            "type": "json_schema",
            "json_schema": npc_schema
        },
        # Possibly set "strict" at top-level if your doc version requires it. 
        # But we've included "strict": true inside the schema definition above.
    }

    logging.info("Sending structured outputs request for NPC. Prompt: %s", user_text)

    # Attempt up to 3 retries if the model refuses or fails
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json=request_body,
                    timeout=120
                )
                resp.raise_for_status()
                data = resp.json()

            # data is the entire completion. We want data["choices"][0]["message"].
            choice = data["choices"][0]["message"]
            # If the model refused, the "refusal" field will appear:
            if "refusal" in choice:
                logging.warning(f"NPC schema attempt {attempt+1}: model refused: {choice['refusal']}")
                continue  # try again or break out, your call

            # Otherwise, the structured JSON is in `choice["parsed"]`
            if "parsed" not in choice:
                logging.warning(f"NPC schema attempt {attempt+1}: no 'parsed' found in message? Full: {choice}")
                continue

            parsed = choice["parsed"]
            # e.g. { "likes": [...], "dislikes": [...], "hobbies": [...], "affiliations": [...], "schedule": {...}}
            return parsed

        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error: {e} attempt {attempt+1}")
        except Exception as e:
            logging.error(f"Error parsing or calling GPT for NPC. Attempt {attempt+1}: {e}")

    # If we reach here, all attempts failed or it kept refusing => fallback
    logging.error("NPC structured call failed after retries. Returning fallback from npc_data.")
    return {
        "likes": npc_data.get("likes", []),
        "dislikes": npc_data.get("dislikes", []),
        "hobbies": npc_data.get("hobbies", []),
        "affiliations": npc_data.get("affiliations", []),
        "schedule": npc_data.get("schedule", {})
    }
