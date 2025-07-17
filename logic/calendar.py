# logic/calendar.py

"""
REFACTORED: All database writes now go through canon
"""

import json
import logging
from typing import Dict
import asyncio
import asyncpg
from db.connection import get_db_connection_context
from logic.chatgpt_integration import get_async_openai_client, build_message_history, safe_json_loads
from lore.core import canon
from agents import RunContextWrapper

# Configure logging as needed.
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
async def get_chatgpt_response_no_function(
    conversation_id: int,
    aggregator_text: str,
    user_input: str,
) -> dict:
    """
    Fire a call to the Responses endpoint and return the plain-text output.
    """

    client = await get_async_openai_client()   # Use the existing function

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
        "  - \"months\": an array of exactly 12 creative and unique month names,\n"
        "  - \"days\": an array of exactly 7 creative and unique day names for the week.\n\n"
        "IMPORTANT: Ensure your JSON is valid - all array elements must be separated by commas.\n"
        "Environment description: " + environment_desc + "\n\n"
        "Return only the JSON object with no additional explanation or markdown formatting."
    )
    
    logging.info("Calling GPT for calendar names with prompt:\n%s", prompt)
    # Use the no-function variant for calendar names.
    gpt_response = await get_chatgpt_response_no_function(conversation_id, environment_desc, prompt)
    logging.info("GPT calendar naming response: %s", gpt_response)
    
    calendar_names = {}
    try:
        response_text = gpt_response.get("response", "").strip()
        
        # Remove markdown code fences if present
        if response_text.startswith("```"):
            lines = response_text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            response_text = "\n".join(lines).strip()
        
        # Fix common JSON issues
        response_text = fix_common_json_issues(response_text)
        
        calendar_names = json.loads(response_text)
        
        # Validate the structure
        if not isinstance(calendar_names.get("months"), list) or len(calendar_names["months"]) != 12:
            raise ValueError("Invalid months array - must have exactly 12 months")
        if not isinstance(calendar_names.get("days"), list) or len(calendar_names["days"]) != 7:
            raise ValueError("Invalid days array - must have exactly 7 days")
        if not calendar_names.get("year_name"):
            raise ValueError("Missing year_name")
            
    except Exception as e:
        logging.error("Failed to parse calendar names JSON: %s", e, exc_info=True)
        logging.error("Raw response text: %s", response_text if 'response_text' in locals() else 'N/A')
        
        # Fallback to a default naming scheme if GPT fails
        calendar_names = {
            "year_name": "The Eternal Cycle",
            "months": ["Aurora", "Blaze", "Crimson", "Dusk", "Ember", "Frost", 
                      "Gleam", "Haze", "Iris", "Jade", "Knell", "Lumen"],
            "days": ["Sol", "Luna", "Terra", "Vesta", "Mercury", "Venus", "Mars"]
        }
    
    return calendar_names


def fix_common_json_issues(json_str: str) -> str:
    """
    Fix common JSON formatting issues from LLM outputs.
    """
    import re
    
    # Fix missing commas between array elements
    # Pattern: "element1"\n"element2" -> "element1",\n"element2"
    json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
    
    # Fix missing commas between array elements on same line
    # Pattern: "element1" "element2" -> "element1", "element2"
    json_str = re.sub(r'"\s+"', '", "', json_str)
    
    # Fix trailing commas before closing brackets
    json_str = re.sub(r',\s*\]', ']', json_str)
    json_str = re.sub(r',\s*\}', '}', json_str)
    
    # Fix missing quotes around keys (simple cases)
    # Pattern: key: "value" -> "key": "value"
    json_str = re.sub(r'(\w+):\s*"', r'"\1": "', json_str)
    
    # Ensure arrays have proper formatting
    # Fix cases like ["item1""item2"] -> ["item1","item2"]
    json_str = re.sub(r'"\s*"', '", "', json_str)
    
    return json_str

async def store_calendar_names(user_id: int, conversation_id: int, calendar_names: dict, conn: asyncpg.Connection):
    """
    REFACTORED: Uses canon to store calendar names in CurrentRoleplay
    Connection is passed from calling function.
    """
    try:
        # Convert data to JSON string
        value_json = json.dumps(calendar_names)
        
        # Create a context object for canon - UPDATE THIS
        ctx = RunContextWrapper(context={
            'user_id': user_id,
            'conversation_id': conversation_id
        })
        
        # Use canon to update CurrentRoleplay
        await canon.update_current_roleplay(
            ctx, conn, 
            'CalendarNames', value_json
        )
    except Exception as e:
        logging.error(f"Error storing calendar names: {e}")
        raise  # Re-raise the exception after logging

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
