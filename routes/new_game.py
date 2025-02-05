# routes/new_game.py

import logging
import json
import random
import time
from flask import Blueprint, request, jsonify, session
import openai

from db.connection import get_db_connection
from routes.settings_routes import insert_missing_settings, generate_mega_setting_logic
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from logic.npc_creation import create_npc
from logic.aggregator import get_aggregated_roleplay_context
from routes.story_routes import build_aggregator_text

# Import the Celery tasks
from tasks import create_npcs_task, get_gpt_opening_line_task

new_game_bp = Blueprint('new_game_bp', __name__)

@new_game_bp.route('/start_new_game', methods=['POST'])
def start_new_game():
    """
    Creates or re-initializes a game scenario.
    (Documentation omitted for brevity.)
    """
    logging.info("=== START: /start_new_game CALLED ===")
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Parse and unwrap incoming JSON.
        data = request.get_json() or {}
        logging.info(f"Raw incoming JSON data: {data}")
        if "params" in data:
            data = data["params"]
            logging.info(f"After unwrapping 'params': {data}")
        conversation_id = data.get("conversation_id")

        # 1) Generate environment snippet.
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

        # 2) If no conversation_id, generate scenario name & quest via GPT.
        scenario_name = "New Game"  # fallback
        quest_blurb = ""
        if not conversation_id:
            scenario_name, quest_blurb = gpt_generate_scenario_name_and_quest(environment_name, environment_desc)
            logging.info(f"GPT scenario_name={scenario_name}, quest_blurb={quest_blurb}")

        # 3) Create or validate conversation.
        if not conversation_id:
            cursor.execute("""
                INSERT INTO conversations (user_id, conversation_name)
                VALUES (%s, %s)
                RETURNING id
            """, (user_id, scenario_name))
            conversation_id = cursor.fetchone()[0]
            conn.commit()
            logging.info(f"Created new conversation_id={conversation_id} for user_id={user_id}, name={scenario_name}")
        else:
            cursor.execute("SELECT id FROM conversations WHERE id=%s AND user_id=%s",
                           (conversation_id, user_id))
            row = cursor.fetchone()
            if not row:
                return jsonify({"error": f"Conversation {conversation_id} not found or unauthorized"}), 403
            logging.info(f"Using existing conversation_id={conversation_id} for user_id={user_id}")

        # 4) Clear old game data.
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

        # 5) Insert environment data.
        combined_modifiers_json = json.dumps(mega_data["stat_modifiers"])
        cursor.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES (%s, %s, 'MegaSettingModifiers', %s)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, (user_id, conversation_id, combined_modifiers_json))
        conn.commit()

        logging.info("Calling insert_missing_settings()")
        insert_missing_settings()

        # 6) Reset or create 'Chase' in PlayerStats.
        logging.info(f"Resetting PlayerStats for user_id={user_id}, conversation_id={conversation_id}, keep only 'Chase'.")
        cursor.execute("""
            DELETE FROM PlayerStats
            WHERE user_id=%s AND conversation_id=%s AND player_name <> 'Chase'
        """, (user_id, conversation_id))
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

        # 7) Offload NPC creation as an asynchronous task.
        logging.info("Dispatching async task to spawn 10 new unintroduced NPCs.")
        create_npcs_task.delay(user_id, conversation_id, 10)

        # 8) Store environment data (CurrentSetting and MainQuest) in CurrentRoleplay.
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
            "Monday": {"Morning": "Wake at small inn", "Afternoon": "Work", "Evening": "Meetup with hobby group", "Night": "Inn room rest"},
            "Tuesday": {"Morning": "Physical training", "Afternoon": "Study mystical texts", "Evening": "Free time", "Night": "Return to inn"},
            "Wednesday": {"Morning": "Wake at small inn", "Afternoon": "Guild errands", "Evening": "Meetup with hobby group", "Night": "Inn room rest"},
            "Thursday": {"Morning": "Physical training", "Afternoon": "Work", "Evening": "Meetup with hobby group", "Night": "Return to inn"},
            "Friday": {"Morning": "Wake at small inn", "Afternoon": "Guild errands", "Evening": "Leisure time", "Night": "Inn room rest"},
            "Saturday": {"Morning": "Sleep in", "Afternoon": "Work", "Evening": "Free time", "Night": "Return to inn"},
            "Sunday": {"Morning": "Physical training", "Afternoon": "Work", "Evening": "Meetup with hobby group", "Night": "Return to inn"}
        }
        chase_role = (
            "Chase is one of the only men in this world of dominant females. "
            "He scrapes by on odd jobs, forging bonds with the realm’s formidable denizens."
        )

        success_msg = f"New game started. Environment={environment_name}, conversation_id={conversation_id}"
        logging.info(f"Success! {success_msg}")

        # 9) Aggregate roleplay context.
        aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, "Chase")
        aggregator_text = build_aggregator_text(aggregator_data)

        # 10) Prepare the opening prompt.
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

        # 11) Offload the GPT call to get the opening line.
        logging.info("Dispatching async task to generate GPT opening line.")
        gpt_result = get_gpt_opening_line_task.delay(conversation_id, aggregator_text, opening_user_prompt)
        try:
            # Wait for the async task to complete. Adjust the timeout as needed.
            gpt_reply_json = gpt_result.get(timeout=30)
            gpt_reply_dict = json.loads(gpt_reply_json)
        except Exception as e:
            logging.error("Error in async GPT call: %s", e)
            gpt_reply_dict = {"response": "[Error retrieving GPT response]", "type": "error"}
        nyx_text = gpt_reply_dict.get("response", "[No text returned from GPT]")

        # 12) Store the GPT reply in messages.
        structured_json_str = json.dumps(gpt_reply_dict)
        cursor.execute(
            """
            INSERT INTO messages (conversation_id, sender, content, structured_content)
            VALUES (%s, %s, %s, %s)
            """,
            (conversation_id, "Nyx", nyx_text, structured_json_str)
        )
        conn.commit()

        # 13) Fetch conversation history.
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
    """
    Calls GPT for a short scenario name (1–8 words)
    and a short main quest (1-2 lines),
    referencing the environment name/desc.
    Returns (scenario_name, quest_blurb).
    """
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

    The conversation name must be unique; do not reuse names from older scenarios 
    (you can ensure uniqueness using the token or environment cues).
    Keep it thematically relevant to a femdom environment.
    """
    messages = [{"role": "system", "content": system_instructions}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.9,
        max_tokens=120,
        frequency_penalty=0.3
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
