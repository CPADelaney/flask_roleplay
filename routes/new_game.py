import logging
import json
import random
import time
import requests.exceptions  # for Timeout exceptions
from flask import Blueprint, request, jsonify, session
import openai
from openai.error import RateLimitError, APIError, ServiceUnavailableError, Timeout

from db.connection import get_db_connection
from routes.settings_routes import insert_missing_settings, generate_mega_setting_logic
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from logic.npc_creation import create_npc
from logic.aggregator import get_aggregated_roleplay_context
from routes.story_routes import build_aggregator_text

new_game_bp = Blueprint('new_game_bp', __name__)

# -------------------------------
# Retry decorator for API calls
# -------------------------------
def retry_on_exception(max_retries=5, backoff_factor=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (RateLimitError, APIError, ServiceUnavailableError, Timeout, requests.exceptions.Timeout) as e:
                    wait = backoff_factor ** attempt
                    logging.warning(f"API error: {e}. Retrying in {wait} seconds (attempt {attempt + 1}/{max_retries}).")
                    time.sleep(wait)
            raise Exception("Max retries exceeded for API call.")
        return wrapper
    return decorator

# -----------------------------------------
# Helper function to stream chat completion
# -----------------------------------------
@retry_on_exception(max_retries=5, backoff_factor=2)
def stream_chat_completion(client, messages, model="gpt-4o", temperature=0.7, timeout=120, max_tokens=150, frequency_penalty=0.0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        stream=True  # enable streaming responses
    )
    full_response = ""
    # Iterate over the streamed chunks and accumulate the text.
    for chunk in response:
        if 'choices' in chunk:
            for choice in chunk['choices']:
                delta = choice.get('delta', {})
                content = delta.get('content', '')
                full_response += content
    return full_response.strip()

@new_game_bp.route('/start_new_game', methods=['POST'])
def start_new_game():
    """
    Creates or re-initializes a game scenario *for the current user, in a specific conversation*.
      - If no conversation_id is provided, we first generate the environment snippet
        and pass it into GPT to produce a robust scenario name & short quest summary.
      - Then create or reuse the conversation.
      - Clear old game data, reset 'Chase' stats, spawn unintroduced NPCs, etc.
      - Store the environment in CurrentRoleplay.
      - Finally, call GPT for an in-character "Nyx" intro, then return the conversation + environment details.
    """

    logging.info("=== START: /start_new_game CALLED ===")

    # 1) Confirm user is logged in
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 2) Parse request JSON
        data = request.get_json() or {}
        logging.info(f"Raw incoming JSON data: {data}")
        if "params" in data:
            data = data["params"]
            logging.info(f"After unwrapping 'params': {data}")

        conversation_id = data.get("conversation_id")

        # === NEW: Generate environment up front so we have a snippet for scenario naming ===
        logging.info("Generating new mega setting via generate_mega_setting_logic()")
        mega_data = generate_mega_setting_logic()
        if "error" in mega_data:
            mega_data["mega_name"] = "No environment available"

        environment_name = mega_data["mega_name"]
        environment_desc = (
            "An eclectic realm combining monstrous societies, futuristic tech, "
            "and archaic ruins floating across the sky. Strange energies swirl, "
            "revealing hidden rituals and uncharted opportunities."
        )
        logging.info(f"Environment snippet for naming:\nName: {environment_name}\nDesc: {environment_desc}")

        # If no conversation_id, ask GPT for scenario name & quest, using environment snippet
        scenario_name = "New Game"  # fallback
        quest_blurb = ""
        if not conversation_id:
            scenario_name, quest_blurb = gpt_generate_scenario_name_and_quest(environment_name, environment_desc)
            logging.info(f"GPT scenario_name={scenario_name}, quest_blurb={quest_blurb}")

        # 3) Create or reuse conversation
        if not conversation_id:
            # 1) Create new conversation
            cursor.execute("""
                INSERT INTO conversations (user_id, conversation_name)
                VALUES (%s, %s)
                RETURNING id
            """, (user_id, scenario_name))
            conversation_id = cursor.fetchone()[0]
            conn.commit()
            logging.info(f"Created new conversation_id={conversation_id} for user_id={user_id}, name={scenario_name}")
        else:
            # 2) If conversation is provided, ensure it belongs to user
            cursor.execute("SELECT id FROM conversations WHERE id=%s AND user_id=%s",
                           (conversation_id, user_id))
            row = cursor.fetchone()
            if not row:
                return jsonify({"error": f"Conversation {conversation_id} not found or unauthorized"}), 403
            logging.info(f"Using existing conversation_id={conversation_id} for user_id={user_id}")
        
        # 3) Clear old data FIRST
        logging.info(f"Clearing data for user_id={user_id}, conversation_id={conversation_id}")
        cursor.execute("DELETE FROM Events        WHERE user_id=%s AND conversation_id=%s", (user_id, conversation_id))
        cursor.execute("DELETE FROM PlannedEvents WHERE user_id=%s AND conversation_id=%s", (user_id, conversation_id))
        cursor.execute("DELETE FROM PlayerInventory WHERE user_id=%s AND conversation_id=%s", (user_id, conversation_id))
        cursor.execute("DELETE FROM Quests        WHERE user_id=%s AND conversation_id=%s", (user_id, conversation_id))
        cursor.execute("DELETE FROM NPCStats      WHERE user_id=%s AND conversation_id=%s", (user_id, conversation_id))
        cursor.execute("DELETE FROM Locations     WHERE user_id=%s AND conversation_id=%s", (user_id, conversation_id))
        cursor.execute("DELETE FROM SocialLinks   WHERE user_id=%s AND conversation_id=%s", (user_id, conversation_id))
        cursor.execute("DELETE FROM CurrentRoleplay WHERE user_id=%s AND conversation_id=%s", (user_id, conversation_id))
        conn.commit()
        
        # 5) Now that everything is cleared, re-insert environment data, including MegaSettingModifiers
        
        # Actually define combined_modifiers_json
        combined_modifiers_json = json.dumps(mega_data["stat_modifiers"])
        
        cursor.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES (%s, %s, 'MegaSettingModifiers', %s)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, (user_id, conversation_id, combined_modifiers_json))
        conn.commit()

        # 5) Insert missing settings if needed
        logging.info("Calling insert_missing_settings()")
        insert_missing_settings()

        # 6) Reset or create 'Chase'
        logging.info(f"Resetting PlayerStats for user_id={user_id}, conversation_id={conversation_id}, keep only 'Chase'.")
        cursor.execute("""
            DELETE FROM PlayerStats
            WHERE user_id=%s AND conversation_id=%s AND player_name <> 'Chase'
        """, (user_id, conversation_id))

        # Check if 'Chase' row exists
        cursor.execute("""
            SELECT id FROM PlayerStats
            WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
        """, (user_id, conversation_id))
        chase_row = cursor.fetchone()

        if chase_row:
            logging.info("Updating existing 'Chase' stats.")
            cursor.execute('''
                UPDATE PlayerStats
                SET corruption=10,
                    confidence=60,
                    willpower=50,
                    obedience=20,
                    dependency=10,
                    lust=15,
                    mental_resilience=55,
                    physical_endurance=40
                WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
            ''', (user_id, conversation_id))
        else:
            logging.info("Inserting fresh row for 'Chase'.")
            cursor.execute('''
                INSERT INTO PlayerStats (
                  user_id, conversation_id, player_name,
                  corruption, confidence, willpower,
                  obedience, dependency, lust,
                  mental_resilience, physical_endurance
                )
                VALUES (%s, %s, 'Chase', 10, 60, 50, 20, 10, 15, 55, 40)
            ''', (user_id, conversation_id))

        conn.commit()
        logging.info("PlayerStats reset complete.")

        # 7) Spawn 10 new unintroduced NPCs
        logging.info("Spawning 10 new unintroduced NPCs.")
        for i in range(3):
            new_id = create_npc(
                user_id=user_id,
                conversation_id=conversation_id,
                introduced=False
            )
            logging.info(f"Created new unintroduced NPC {i+1}/10, ID={new_id}")

        # 8) Store environment in CurrentRoleplay
        logging.info(f"Storing environment name & quest in CurrentRoleplay: {environment_name}")
        cursor.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES (%s, %s, 'CurrentSetting', %s)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, (user_id, conversation_id, environment_name))

        if quest_blurb.strip():
            cursor.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES (%s, %s, 'MainQuest', %s)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value=EXCLUDED.value
            """, (user_id, conversation_id, quest_blurb))

        conn.commit()

        chase_schedule = {
            "Monday": {
                "Morning": "Wake at small inn",
                "Afternoon": "Work",
                "Evening": "Meetup with hobby group",
                "Night": "Inn room rest"
            },
            "Tuesday": {
                "Morning": "Physical training",
                "Afternoon": "Study mystical texts",
                "Evening": "Free time",
                "Night": "Return to inn"
            },
            "Wednesday": {
                "Morning": "Wake at small inn",
                "Afternoon": "Guild errands",
                "Evening": "Meetup with hobby group",
                "Night": "Inn room rest"
            },
            "Thursday": {
                "Morning": "Physical training",
                "Afternoon": "Work",
                "Evening": "Meetup with hobby group",
                "Night": "Return to inn"
            },
            "Friday": {
                "Morning": "Wake at small inn",
                "Afternoon": "Guild errands",
                "Evening": "Leisure time",
                "Night": "Inn room rest"
            },
            "Saturday": {
                "Morning": "Sleep in",
                "Afternoon": "Work",
                "Evening": "Free time",
                "Night": "Return to inn"
            },
            "Sunday": {
                "Morning": "Physical training",
                "Afternoon": "Work",
                "Evening": "Meetup with hobby group",
                "Night": "Return to inn"
            }
        }
        chase_role = (
            "Chase is one of the only men in this world of dominant females. "
            "He scrapes by on odd jobs, forging bonds with the realm’s formidable denizens."
        )

        success_msg = f"New game started. Environment={environment_name}, conversation_id={conversation_id}"
        logging.info(f"Success! Returning 200 with message: {success_msg}")

        # 9) aggregator_data => from aggregator
        aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, "Chase")
        aggregator_text = build_aggregator_text(aggregator_data)
        
        # 10) GPT call for the game’s opening line
        opening_user_prompt = (
            "Begin the scenario now, Nyx. Greet Chase with your sadistic, mocking style, "
            "briefly recount the new environment’s background or history from the aggregator data, "
            "and announce that Day 1 has just begun. "
            "Describe where the player is that morning (look at their schedule). "
            "Reference the player's role (if relevant), "
            "and (only if the main character has already met them) highlight a couple of newly introduced NPCs. "
            "If there's a main quest mentioned, hint at it ominously. "
            "Stay fully in character, with no disclaimers or system explanations. "
            "Conclude with a menacing or teasing invitation for Chase to proceed."
        )
        
        gpt_reply_dict = get_chatgpt_response(
            conversation_id=conversation_id, 
            aggregator_text=get_aggregated_roleplay_context(user_id, conversation_id, "Chase"),
            user_input=opening_user_prompt
        )
        nyx_text = gpt_reply_dict.get("response")

        if gpt_reply_dict["type"] == "function_call" or not nyx_text:
            logging.info("GPT tried a function call or returned no text for the intro. Re-calling without function calls.")
            client = get_openai_client()
            forced_messages = [
                {
                    "role": "system",
                    "content": get_aggregated_roleplay_context(user_id, conversation_id, "Chase")
                },
                {
                    "role": "user",
                    "content": "No function calls for the introduction. Produce only text narrative.\n\n" + opening_user_prompt
                }
            ]
            try:
                # Use our streaming helper with error handling/retries.
                fallback_text = stream_chat_completion(
                    client,
                    forced_messages,
                    model="gpt-4o",
                    temperature=0.7,
                    timeout=120,
                    max_tokens=150  # adjust as needed
                )
            except Exception as e:
                logging.exception("Error during fallback streaming call:")
                fallback_text = "[No text returned from GPT]"
            nyx_text = fallback_text if fallback_text else "[No text returned from GPT]"

        # 13) Store the final text into DB
        structured_json_str = json.dumps(gpt_reply_dict)
        cursor.execute(
            """
            INSERT INTO messages (conversation_id, sender, content, structured_content)
            VALUES (%s, %s, %s, %s)
            """,
            (conversation_id, "Nyx", nyx_text, structured_json_str)
        )
        conn.commit()

        # 14) Return data, including conversation history
        cursor.execute("""
            SELECT sender, content, created_at
            FROM messages
            WHERE conversation_id=%s
            ORDER BY id ASC
        """, (conversation_id,))
        rows = cursor.fetchall()
        conversation_history = [{
            "sender": r[0],
            "content": r[1],
            "created_at": r[2].isoformat()
        } for r in rows]

        success_msg = f"New game started. Environment={environment_name}, conversation_id={conversation_id}"
        logging.info(f"Success! Returning 200 with message: {success_msg}")
        return jsonify({
            "message": success_msg,
            "scenario_name": scenario_name,
            "environment_name": environment_name,
            "environment_desc": environment_desc,
            "chase_schedule": chase_schedule,
            "chase_role": chase_role,
            "conversation_id": conversation_id,
            "messages": conversation_history
        }), 200

    except Exception as e:
        logging.exception("Error in /start_new_game:")
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
        logging.info("=== END: /start_new_game ===")


def gpt_generate_scenario_name_and_quest(env_name: str, env_desc: str):
    client = get_openai_client()
    unique_token = f"{random.randint(1000,9999)}_{int(time.time())}"
    forbidden_words = ["mistress", "darkness", "manor", "chains", "twilight"]

    system_instructions = f"""
    You are setting up a new femdom daily-life sim scenario with a main quest.
    Environment name: {env_name}
    Environment desc: {env_desc}
    Unique token: {unique_token}

    Please produce:
    1) A single line starting with 'ScenarioName:' followed by a short, creative (1–8 words) name 
       that draws from the environment above. 
       Avoid cliche words like {', '.join(forbidden_words)}.
    2) Then one or two lines summarizing the main quest.

    Ensure uniqueness by using the provided token.
    """
    messages = [{"role": "system", "content": system_instructions}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.9,
        max_tokens=120,
        frequency_penalty=0.3,
        timeout=120
    )
    msg = response.choices[0].message.content.strip()
    logging.info(f"[gpt_generate_scenario_name_and_quest] Raw GPT output: {msg}")
    scenario_name = "New Game"
    quest_blurb = ""
    for line in msg.splitlines():
        line = line.strip()
        if line.lower().startswith("scenarioname:"):
            scenario_name = line.split(":", 1)[1].strip()
        else:
            quest_blurb += line + " "
    return scenario_name.strip(), quest_blurb.strip()
