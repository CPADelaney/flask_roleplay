# logic/calendar.py

import json
import logging
from db.connection import get_db_connection
from logic.chatgpt_integration import get_chatgpt_response

# Configure logging as needed.
logging.basicConfig(level=logging.INFO)

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
        "Your response should be in JSON format with exactly the following keys:\n"
        "  - \"year_name\": a creative name for the overall year (e.g., 'The Age of Ember', 'The Silver Cycle'),\n"
        "  - \"months\": an array of 12 unique month names,\n"
        "  - \"days\": an array of 7 unique day names for the week.\n\n"
        "Environment description: " + environment_desc + "\n\n"
        "Return only the JSON object with no additional explanation."
    )
    
    logging.info("Calling GPT for calendar names with prompt:\n%s", prompt)
    gpt_response = get_chatgpt_response(conversation_id, environment_desc, prompt)
    logging.info("GPT calendar naming response: %s", gpt_response)
    
    try:
        response_text = gpt_response.get("response", "").strip()
        # If GPT wraps the response in markdown code fences, remove them.
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

def update_calendar_names(user_id, conversation_id, environment_desc):
    """
    Generates immersive calendar names based on the provided environment description,
    stores them, and returns the resulting dictionary.
    """
    calendar_names = generate_calendar_names(environment_desc, conversation_id)
    store_calendar_names(user_id, conversation_id, calendar_names)
    return calendar_names
