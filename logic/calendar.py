# logic/calendar.py

import json
import logging
import asyncpg  # or your DB library, if you're doing async. 
               # If you're using psycopg2, adapt store_calendar_names to your approach.

from db.connection import get_db_connection
from logic.chatgpt_integration import get_chatgpt_response

logging.basicConfig(level=logging.INFO)

async def generate_calendar_names(environment_desc: str, conversation_id: int) -> dict:
    """
    Asks GPT to generate an immersive calendar naming scheme:
      - 'year_name': A creative name for the overall year/era (string),
      - 'months': An array of 12 creative month names (list[str]),
      - 'days': An array of 7 creative day names for the week (list[str]).
    
    Returns a dict with exactly those keys. If GPT fails or we can't parse, 
    returns a fallback dictionary.
    """

    prompt = (
        "Based on the following environment description, generate an immersive and thematic naming scheme for the in-game calendar. "
        "Context is a 'femdom daily-life sim roleplaying game' so names should reflect a powerful female-dominated world. "
        "Ensure names are creative, unique, and rooted in the universe's history. "
        "Your response should be JSON **with exactly** these keys:\n"
        '  "year_name": a creative name for the overall year (e.g., "The Age of Ember"),\n'
        '  "months": an array of 12 creative month names,\n'
        '  "days": an array of 7 creative day names.\n\n'
        "Do not include any additional explanation or text, and do not wrap it in markdown code fences.\n"
        "Environment description: " + environment_desc + "\n\n"
        "Return only the JSON object with no extra text."
    )

    logging.info("Calling GPT for calendar names with prompt:\n%s", prompt)

    # 1) Call GPT (await needed if get_chatgpt_response is async)
    response = await get_chatgpt_response(conversation_id, environment_desc, prompt)
    logging.info("GPT raw response for calendar naming: %s", response)

    # 2) Attempt to parse GPT's output
    #    (We handle function_call vs. normal text.)
    try:
        if response.get("type") == "function_call":
            # GPT returned a function call with .function_args
            fn_args = response.get("function_args", {})
            year_name = fn_args.get("year_name", "The Eternal Cycle")
            months = fn_args.get("months", [])
            days = fn_args.get("days", [])
            # Basic fallback if keys are missing or not lists
            if not isinstance(months, list) or len(months) != 12:
                months = ["Aurora", "Blaze", "Crimson", "Dusk", "Ember", "Frost", 
                          "Gleam", "Haze", "Iris", "Jade", "Knell", "Lumen"]
            if not isinstance(days, list) or len(days) != 7:
                days = ["Sol", "Luna", "Terra", "Vesta", "Mercury", "Venus", "Mars"]

            calendar_names = {
                "year_name": year_name,
                "months": months,
                "days": days
            }

        else:
            # GPT returned normal text. It's in response["response"] possibly
            response_text = response.get("response", "").strip()

            # remove code fences if present
            if response_text.startswith("```"):
                lines = response_text.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()

            # Now parse JSON
            calendar_names = json.loads(response_text)

            # Basic fallback checks
            if "year_name" not in calendar_names:
                calendar_names["year_name"] = "The Eternal Cycle"
            if "months" not in calendar_names or not isinstance(calendar_names["months"], list) \
               or len(calendar_names["months"]) != 12:
                calendar_names["months"] = ["Aurora","Blaze","Crimson","Dusk","Ember","Frost",
                                            "Gleam","Haze","Iris","Jade","Knell","Lumen"]
            if "days" not in calendar_names or not isinstance(calendar_names["days"], list) \
               or len(calendar_names["days"]) != 7:
                calendar_names["days"] = ["Sol","Luna","Terra","Vesta","Mercury","Venus","Mars"]

    except Exception as e:
        logging.error("Failed to parse GPT's calendar JSON: %s", e, exc_info=True)
        calendar_names = {
            "year_name": "The Eternal Cycle",
            "months": ["Aurora","Blaze","Crimson","Dusk","Ember","Frost",
                       "Gleam","Haze","Iris","Jade","Knell","Lumen"],
            "days": ["Sol","Luna","Terra","Vesta","Mercury","Venus","Mars"]
        }

    logging.info("Final calendar_names: %s", calendar_names)
    return calendar_names


async def store_calendar_names(user_id: int, conversation_id: int, calendar_names: dict):
    """
    Stores the generated calendar names in the CurrentRoleplay table
    under the key 'CalendarNames'. This ensures consistency throughout the game.
    Using asyncpg or your async DB approach. 
    """
    conn = await asyncpg.connect(dsn="YOUR_DSN_HERE")  # Or get_db_connection if it's async.
    try:
        value = json.dumps(calendar_names)
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'CalendarNames', $3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value = EXCLUDED.value
        """, user_id, conversation_id, value)
        logging.info("Stored CalendarNames successfully.")
    except Exception as e:
        logging.error("Failed to store calendar names: %s", e, exc_info=True)
    finally:
        await conn.close()


async def update_calendar_names(user_id: int, conversation_id: int, environment_desc: str) -> dict:
    """
    Generates immersive calendar names based on the environment description,
    stores them, and returns the resulting dictionary.
    """
    # 1) Generate with GPT
    calendar_names = await generate_calendar_names(environment_desc, conversation_id)
    # 2) Store in DB
    await store_calendar_names(user_id, conversation_id, calendar_names)
    return calendar_names
