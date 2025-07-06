# logic/calendar.py

"""
REFACTORED: All database writes now go through canon
"""

import json
import logging
from openai import AsyncOpenAI
import asyncio
import asyncpg
from db.connection import get_db_connection_context
from logic.chatgpt_integration import get_openai_client, build_message_history, safe_json_loads
from lore.core import canon

# Configure logging as needed.
logging.basicConfig(level=logging.INFO)

async def get_openai_client() -> AsyncOpenAI:
    """
    Centralised factory so every call site gets the *same* AsyncOpenAI instance.
    Keeps your API key / base_url in one place.
    """
    # Re-use a singleton if you want; demonstration keeps it simple.
    return AsyncOpenAI()                # or AsyncOpenAI(base_url=..., api_key=...)

# ---------------------------------------------------------------------------
async def get_chatgpt_response_no_function(
    conversation_id: int,
    aggregator_text: str,
    user_input: str,
) -> dict:
    """
    Fire a call to the Responses endpoint and return the plain-text output.
    """

    client: AsyncOpenAI = await get_openai_client()   # ⭐ now awaited because the
                                                      #   factory could become async
                                                      #   (e.g., to fetch creds)

    # prepare the request payload ------------------------------------------
    messages = await build_message_history(
        conversation_id, aggregator_text, user_input, limit=15
    )

    # call the Responses API -----------------------------------------------
    response = await client.responses.create(
        model="gpt-4.1-nano",   # or "gpt-4.1-nano" if you really need that tier
        input=messages         # Responses API uses the *single* `input` field
    )                          # Streaming? add stream=True and iterate.

    # extract what we need --------------------------------------------------
    response_text = response.output_text
    tokens_used   = response.usage.total_tokens        # guaranteed to exist

    return {
        "type": "text",
        "response": response_text,
        "tokens_used": tokens_used,
    }
    
async def generate_calendar_names(environment_desc, conversation_id):
    """
    Use GPT to generate immersive calendar names for the in-game time system.
    
    The GPT prompt asks for:
      - "year_name": a creative name for the overall year,
      - "months": an array of 12 unique month names,
      - "days": an array of 7 unique day names.
    
    Returns a dictionary with these keys.
    """
    prompt = (
        "Based on the following environment description, generate an immersive and thematic naming scheme for the in-game calendar. "
        "Keep in mind the context is 'femdom daily-life sim roleplaying game' and the names should reflect this. "
        "Ensure names are creative and unique, and are rooted in the universe and history of the setting. "
        "Your response should be in JSON format with exactly the following keys:\n"
        "  - \"year_name\": a creative name for the overall year (e.g., 'The Age of Ember', 'The Silver Cycle'),\n"
        "  - \"months\": an array of 12 creative and unique month names,\n"
        "  - \"days\": an array of 7 creative and unique day names for the week.\n\n"
        "Environment description: " + environment_desc + "\n\n"
        "Return only the JSON object with no additional explanation."
    )
    
    logging.info("Calling GPT for calendar names with prompt:\n%s", prompt)
    # Use the no-function variant for calendar names.
    gpt_response = await get_chatgpt_response_no_function(conversation_id, environment_desc, prompt)
    logging.info("GPT calendar naming response: %s", gpt_response)
    
    calendar_names = {}
    try:
        response_text = gpt_response.get("response", "").strip()
        # Remove markdown code fences if present.
        if response_text.startswith("```"):
            lines = response_text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            response_text = "\n".join(lines).strip()
        calendar_names = json.loads(response_text)
    except Exception as e:
        logging.error("Failed to parse calendar names JSON: %s", e, exc_info=True)
        # Fallback to a default naming scheme if GPT fails.
        calendar_names = {
            "year_name": "The Eternal Cycle",
            "months": ["Aurora", "Blaze", "Crimson", "Dusk", "Ember", "Frost", "Gleam", "Haze", "Iris", "Jade", "Knell", "Lumen"],
            "days": ["Sol", "Luna", "Terra", "Vesta", "Mercury", "Venus", "Mars"]
        }
    
    return calendar_names

async def store_calendar_names(user_id: int, conversation_id: int, calendar_names: dict, conn: asyncpg.Connection):
    """
    REFACTORED: Uses canon to store calendar names in CurrentRoleplay
    Connection is passed from calling function.
    """
    try:
        # Convert data to JSON string
        value_json = json.dumps(calendar_names)
        
        # Create a context object for canon
        ctx = type('obj', (object,), {'user_id': user_id, 'conversation_id': conversation_id})
        
        # Use canon to update CurrentRoleplay
        await canon.update_current_roleplay(
            ctx, conn, user_id, conversation_id, 
            'CalendarNames', value_json
        )

        logging.info(f"Stored CalendarNames successfully for user {user_id}, convo {conversation_id}.")

    except Exception as e:
        logging.exception(
            f"Unexpected error storing calendar names for user {user_id}, convo {conversation_id}: {e}"
        )

async def update_calendar_names(user_id, conversation_id, environment_desc) -> dict:
    """
    REFACTORED: Generates immersive calendar names based on the provided environment description,
    stores them using canon, and returns the resulting dictionary.
    """
    calendar_names = await generate_calendar_names(environment_desc, conversation_id)
    
    # Get connection from context manager and pass it to store function
    async with get_db_connection_context() as conn:
        await store_calendar_names(user_id, conversation_id, calendar_names, conn)
    
    return calendar_names

async def load_calendar_names(user_id, conversation_id):
    """
    Retrieves the calendar names (year_name, months, days) 
    from CurrentRoleplay where key='CalendarNames'.
    Returns a dict with keys 'year_name', 'months', and 'days'.
    """
    try:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT value 
                FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='CalendarNames'
                LIMIT 1
            """, user_id, conversation_id)
            
            if row:
                try:
                    return json.loads(row['value'])
                except json.JSONDecodeError as e:
                    logging.warning("Calendar JSON invalid, returning fallback.")
    except Exception as e:
        logging.error(f"Error loading calendar names: {e}")
    
    # Fallback if not found or invalid
    return {
        "year_name": "The Eternal Cycle",
        "months": [
            "Aurora", "Blaze", "Crimson", "Dusk",
            "Ember", "Frost", "Gleam", "Haze",
            "Iris", "Jade", "Knell", "Lumen"
        ],
        "days": ["Sol", "Luna", "Terra", "Vesta", "Mercury", "Venus", "Mars"]
    }
