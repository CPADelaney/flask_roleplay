# logic/calendar.py
import os
import json
import logging
import httpx
import asyncio
import asyncpg

logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

calendar_schema = {
    "name": "CalendarNames",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "year_name": {
                "type": "string",
                "description": "Creative or thematic name for the year/era"
            },
            "months": {
                "type": "array",
                "description": "Exactly 12 unique month names",
                "items": {"type": "string"}
            },
            "days": {
                "type": "array",
                "description": "Exactly 7 day names",
                "items": {"type": "string"}
            }
        },
        "required": ["year_name","months","days"],
        "additionalProperties": False
    }
}

async def generate_calendar_names(environment_desc: str, conversation_id: int) -> dict:
    """
    Calls the GPT-4o structured outputs endpoint to get a strictly valid JSON object with:
      - year_name (str)
      - months (array of 12 str)
      - days   (array of 7 str)

    If the model refuses or fails, we return a fallback dict.
    """
    # system instructions
    system_text = (
        "You are generating a custom calendar in a femdom daily-life sim environment. "
        "The names should be creative, unique, and reflect this particular setting and its history. "
        "Return only valid JSON that matches the schema, no extra keys or text."
    )

    # user instructions
    user_text = (
        f"Environment description: {environment_desc}\n\n"
        "Please produce an object with exactly 'year_name','months','days' keys.\n"
        "No explanation or markup, just the JSON."
    )

    # We'll prepare the request
    request_body = {
        "model": "gpt-4o",  # or a new gpt-4o variant that supports structured outputs
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": calendar_schema
        }
    }

    logging.info("Sending structured outputs request for calendar. Prompt:\n%s", user_text)

    # Attempt up to 3 times in case of refusal
    max_retries = 3
    fallback = {
        "year_name": "The Eternal Cycle",
        "months": ["Aurora","Blaze","Crimson","Dusk","Ember","Frost",
                   "Gleam","Haze","Iris","Jade","Knell","Lumen"],
        "days": ["Sol","Luna","Terra","Vesta","Mercury","Venus","Mars"]
    }

    for attempt in range(3, max_retries+1):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json=request_body
                )
                resp.raise_for_status()
                data = resp.json()

            choice = data["choices"][0]["message"]
            # If the model refuses
            if "refusal" in choice:
                logging.warning(f"Calendar schema attempt {attempt}: model refused => {choice['refusal']}")
                continue

            # Otherwise, we expect "parsed" in choice
            if "parsed" in choice:
                parsed = choice["parsed"]
                # e.g. { "year_name":"...", "months":[...], "days":[...] }
                logging.info("Parsed calendar data: %s", parsed)
                return parsed

            logging.warning(f"Attempt {attempt} => no refusal, but no 'parsed' object? Full: {choice}")

        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP status error attempt {attempt}: {e}")
        except Exception as e:
            logging.error(f"Error retrieving or parsing GPT structured result attempt {attempt}: {e}")

    # If we got here, all attempts failed or refused => fallback
    logging.error("Unable to get structured calendar after retries. Returning fallback.")
    return fallback

async def store_calendar_names(user_id: int, conversation_id: int, calendar_names: dict):
    """
    Stores the generated calendar names in the CurrentRoleplay table
    under 'CalendarNames' key using asyncpg or your approach.
    """
    dsn = "YOUR_DSN_HERE"
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'CalendarNames', $3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, json.dumps(calendar_names))
        logging.info("CalendarNames stored successfully for user=%s, conversation=%s", user_id, conversation_id)
    except Exception as e:
        logging.error("Error storing calendar: %s", e, exc_info=True)
    finally:
        await conn.close()

async def update_calendar_names(user_id: int, conversation_id: int, environment_desc: str) -> dict:
    """
    Generates immersive calendar names (structured), stores them, and returns the final dict.
    """
    calendar_data = await generate_calendar_names(environment_desc, conversation_id)
    await store_calendar_names(user_id, conversation_id, calendar_data)
    return calendar_data
