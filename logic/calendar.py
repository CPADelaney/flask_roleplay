# logic/calendar.py

import json
import logging
import openai
from db.connection import get_db_connection
from logic.chatgpt_integration import get_openai_client, build_message_history, safe_json_loads

# Configure logging as needed.
logging.basicConfig(level=logging.INFO)

def get_chatgpt_response_no_function(conversation_id: int, aggregator_text: str, user_input: str) -> dict:
    """
    A local version of the GPT call that does not enforce a function call.
    Returns a plain text response.
    """
    client = get_openai_client()
    messages = build_message_history(conversation_id, aggregator_text, user_input, limit=15)
    response = client.chat.completions.create(
         model="gpt-4o",
         messages=messages,
         temperature=0.2,
         max_tokens=4000,
         frequency_penalty=0.0
    )
    msg = response.choices[0].message
    tokens_used = response.usage.total_tokens

    return {
         "type": "text",
         "response": msg.content,
         "tokens_used": tokens_used
    }

def generate_calendar_names(environment_desc, conversation_id):
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
    gpt_response = get_chatgpt_response_no_function(conversation_id, environment_desc, prompt)
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

def store_calendar_names(user_id, conversation_id, calendar_names):
    """
    Stores the generated calendar names in the CurrentRoleplay table
    under the key 'CalendarNames'. This ensures consistency throughout your game.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        value = json.dumps(calendar_names)
        cursor.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES (%s, %s, 'CalendarNames', %s)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value = EXCLUDED.value
        """, (user_id, conversation_id, value))
        conn.commit()
        logging.info("Stored CalendarNames successfully.")
    except Exception as e:
        logging.error("Failed to store calendar names: %s", e, exc_info=True)
    finally:
        cursor.close()
        conn.close()

async def update_calendar_names(user_id, conversation_id, environment_desc) -> dict:
    """
    Generates immersive calendar names based on the provided environment description,
    stores them, and returns the resulting dictionary.
    """
    calendar_names = generate_calendar_names(environment_desc, conversation_id)
    store_calendar_names(user_id, conversation_id, calendar_names)
    return calendar_names

def load_calendar_names(user_id, conversation_id):
    """
    Retrieves the calendar names (year_name, months, days) 
    from CurrentRoleplay where key='CalendarNames'.
    Returns a dict with keys 'year_name', 'months', and 'days'.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT value 
        FROM CurrentRoleplay
        WHERE user_id=%s AND conversation_id=%s AND key='CalendarNames'
        LIMIT 1
    """, (user_id, conversation_id))
    row = cur.fetchone()
    cur.close()
    conn.close()
    
    if row:
        try:
            return json.loads(row[0])
        except json.JSONDecodeError as e:
            logging.warning("Calendar JSON invalid, returning fallback.")
    
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
