import logging
import json             # Explicit import for json
import random           # For randomness
import time             # For time-based token
from flask import Blueprint, request, jsonify, session
import openai
from db.connection import get_db_connection
from routes.settings_routes import insert_missing_settings, generate_mega_setting_logic
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from logic.npc_creation import create_npc
from logic.aggregator import get_aggregated_roleplay_context
from routes.story_routes import build_aggregator_text

new_game_bp = Blueprint('new_game_bp', __name__)

@new_game_bp.route('/start_new_game', methods=['POST'])
def start_new_game():
    """
    Creates or re-initializes a game scenario *for the current user, in a specific conversation*.
      - If no conversation_id is provided, we call gpt_generate_scenario_name_and_quest()
        to produce a unique name & short quest summary, then create a new conversation row.
      - Otherwise, we reuse the conversation_id (only if owned by this user).
      - We clear old game data from relevant tables.
      - We reset or create the player's 'Chase' stats.
      - We spawn 10 unintroduced NPCs.
      - We set a new environment in CurrentRoleplay.
      - Then we call GPT for an in-character "Nyx" intro message referencing aggregator data.
      - We return the entire conversation + environment details to the front end.
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

        # If no conversation_id, ask GPT for scenario name & short quest
        scenario_name = "New Game"  # fallback name
        if not conversation_id:
            scenario_name, quest_blurb = gpt_generate_scenario_name_and_quest()
            logging.info(f"GPT scenario_name={scenario_name}, quest_blurb={quest_blurb}")

        # 3) Create or reuse conversation
        if not conversation_id:
            # Create a new conversation row with the GPT scenario_name
            cursor.execute("""
                INSERT INTO conversations (user_id, conversation_name)
                VALUES (%s, %s)
                RETURNING id
            """, (user_id, scenario_name))
            conversation_id = cursor.fetchone()[0]
            logging.info(f"Created new conversation_id={conversation_id} for user_id={user_id}, name={scenario_name}")
            conn.commit()
        else:
            # Verify conversation belongs to user
            cursor.execute("""
                SELECT id FROM conversations
                WHERE id=%s AND user_id=%s
            """, (conversation_id, user_id))
            row = cursor.fetchone()
            if not row:
                logging.error(f"Conversation {conversation_id} not found or not owned by user {user_id}.")
                return jsonify({"error": "Conversation not found or unauthorized"}), 403
            logging.info(f"Using existing conversation_id={conversation_id} for user_id={user_id}")

        # 4) Clear old game data
        logging.info(f"Deleting old game state for user_id={user_id}, conversation_id={conversation_id}.")
        cursor.execute("DELETE FROM Events        WHERE user_id=%s AND conversation_id=%s", (user_id, conversation_id))
        cursor.execute("DELETE FROM PlannedEvents WHERE user_id=%s AND conversation_id=%s", (user_id, conversation_id))
        cursor.execute("DELETE FROM PlayerInventory WHERE user_id=%s AND conversation_id=%s", (user_id, conversation_id))
        cursor.execute("DELETE FROM Quests        WHERE user_id=%s AND conversation_id=%s", (user_id, conversation_id))
        cursor.execute("DELETE FROM NPCStats      WHERE user_id=%s AND conversation_id=%s", (user_id, conversation_id))
        cursor.execute("DELETE FROM Locations     WHERE user_id=%s AND conversation_id=%s", (user_id, conversation_id))
        cursor.execute("DELETE FROM SocialLinks   WHERE user_id=%s AND conversation_id=%s", (user_id, conversation_id))
        cursor.execute("DELETE FROM CurrentRoleplay WHERE user_id=%s AND conversation_id=%s", (user_id, conversation_id))
        conn.commit()
        logging.info("Per-conversation tables cleared successfully.")

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
        for i in range(10):
            new_id = create_npc(
                user_id=user_id,
                conversation_id=conversation_id,
                introduced=False
            )
            logging.info(f"Created new unintroduced NPC {i+1}/10, ID={new_id}")

        # 8) Generate environment & store in CurrentRoleplay
        logging.info("Generating new mega setting via generate_mega_setting_logic()")
        mega_data = generate_mega_setting_logic()
        if "error" in mega_data:
            mega_data["mega_name"] = "No environment available"

        environment_name = mega_data["mega_name"]
        logging.info(f"Setting CurrentSetting to: {environment_name}")

        cursor.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES (%s, %s, 'CurrentSetting', %s)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, (user_id, conversation_id, environment_name))
        conn.commit()

        environment_desc = (
            "An eclectic realm combining monstrous societies, futuristic tech, "
            "and archaic ruins floating across the sky. Strange energies swirl, "
            "revealing hidden rituals and uncharted opportunities."
        )
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
            "Reference the player's role or schedule (if relevant), "
            "and highlight a couple of newly introduced NPCs in this realm if the main character already knows them. "
            "If there's a main quest mentioned, hint at it ominously. "
            "Stay fully in character, with no disclaimers or system explanations. "
            "Conclude with a menacing or teasing invitation for Chase to proceed."
        )
        gpt_reply_dict = get_chatgpt_response(conversation_id, aggregator_text, opening_user_prompt)
    
        # 11) Store GPT reply as “Nyx” message
        nyx_text = gpt_reply_dict.get("response", "Welcome to your new domain.")
        structured_json_str = json.dumps(gpt_reply_dict)
        cursor.execute("""
            INSERT INTO messages (conversation_id, sender, content, structured_content)
            VALUES (%s, %s, %s, %s)
        """, (conversation_id, "Nyx", nyx_text, structured_json_str))
        conn.commit()
    
        # 12) Return data, including conversation history
        cursor.execute("""
            SELECT sender, content, created_at
            FROM messages
            WHERE conversation_id=%s
            ORDER BY id ASC
        """, (conversation_id,))
        rows = cursor.fetchall()
        conversation_history = []
        for r in rows:
            conversation_history.append({
                "sender": r[0],
                "content": r[1],
                "created_at": r[2].isoformat()
            })

        return jsonify({
            "message": success_msg,
            "scenario_name": scenario_name,
            "environment_name": environment_name,
            "environment_desc": environment_desc,
            "chase_schedule": chase_schedule,
            "chase_role": chase_role,
            "conversation_id": conversation_id,
            "messages": conversation_history  # <-- Ensure the comma is present
        }), 200

    except Exception as e:
        logging.exception("Error in /start_new_game:")
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
        logging.info("=== END: /start_new_game ===")


def gpt_generate_scenario_name_and_quest():
    """
    Calls GPT for a short scenario name (1–8 words)
    and a short main quest (1-2 lines).
    Returns (scenario_name, quest_blurb).
    """
    client = get_openai_client()

    # Add a random token to encourage unique naming
    unique_token = f"{random.randint(1000,9999)}_{int(time.time())}"

    system_instructions = f"""
    You are setting up a new femdom daily-life sim scenario with a main quest. 
    To ensure uniqueness, note token: {unique_token}.

    Please produce:
    1) A single line starting with 'ScenarioName:' followed by a short (1–8 words) name. 
    2) Then one or two lines summarizing the main quest.

    Example:
    ScenarioName: Chains of Twilight
    The main quest: retrieve the midnight relic from the Coven...
    """

    messages = [
        {"role": "system", "content": system_instructions}
    ]

    # Increase temperature for more variation
    response = client.chat.completions.create(
        model="o1",
        reasoning_effort="medium",
        messages=[
            {
                {"role": "system", "content": system_prompt}
            }
        ],
        temperature=0.7,
        max_tokens=100,
        frequency_penalty=0.3
    )

    msg = response.choices[0].message.content.strip()
    logging.info(f"[gpt_generate_scenario_name_and_quest] Raw GPT output: {msg}")

    scenario_name = "New Game"
    quest_blurb = ""

    lines = msg.splitlines()
    for line in lines:
        line = line.strip()
        if line.lower().startswith("scenarioname:"):
            scenario_name = line.split(":", 1)[1].strip()
        else:
            quest_blurb += line + " "

    return scenario_name, quest_blurb.strip()
