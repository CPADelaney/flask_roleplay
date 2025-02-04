import logging
from flask import Blueprint, request, jsonify, session
import random
from db.connection import get_db_connection
import openai
from routes.settings_routes import insert_missing_settings, generate_mega_setting_logic
from logic.chatgpt_integration import get_chatgpt_response
from logic.npc_creation import create_npc
from logic.chatgpt_integration import get_openai_client 

# You likely need to import these two from your aggregator logic:
# from logic.aggregator import get_aggregated_roleplay_context, build_aggregator_text

new_game_bp = Blueprint('new_game_bp', __name__)

@new_game_bp.route('/start_new_game', methods=['POST'])
def start_new_game():
    """
    Creates or re-initializes a game scenario *for the current user, in a specific conversation*.
    ...
    """

    logging.info("=== START: /start_new_game CALLED ===")

    # 1) Confirm user is logged in
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 2) Parse input
        data = request.get_json() or {}
        logging.info(f"Raw incoming JSON data: {data}")
        if "params" in data:
            data = data["params"]
            logging.info(f"After unwrapping 'params': {data}")

        conversation_id = data.get("conversation_id")

        # If no conversation_id, ask GPT for scenario name
        scenario_name = "New Game"  # fallback
        if not conversation_id:
            scenario_name, quest_blurb = gpt_generate_scenario_name_and_quest()
            # e.g., scenario_name might be "Chains of Dusk"

        # Now handle conversation creation or reuse
        if not conversation_id:
            # 3) Create a new conversation row
            cursor.execute("""
                INSERT INTO conversations (user_id, conversation_name)
                VALUES (%s, %s)
                RETURNING id
            """, (user_id, scenario_name))
            conversation_id = cursor.fetchone()[0]
            logging.info(f"Created new conversation_id={conversation_id} for user_id={user_id}, name={scenario_name}")
            conn.commit()
        else:
            # Verify the conversation belongs to this user
            cursor.execute("""
                SELECT id FROM conversations
                WHERE id=%s AND user_id=%s
            """, (conversation_id, user_id))
            row = cursor.fetchone()
            if not row:
                logging.error(f"Conversation {conversation_id} not found or not owned by user {user_id}.")
                return jsonify({"error": "Conversation not found or unauthorized"}), 403
            logging.info(f"Using existing conversation_id={conversation_id} for user_id={user_id}")

        # 4) Clear data from old game state
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

        # 6) Reset or create 'Chase' in PlayerStats
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
        # (you need to have from logic.aggregator import get_aggregated_roleplay_context, build_aggregator_text)
        aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, "Chase")
        aggregator_text = build_aggregator_text(aggregator_data)
    
        # 10) GPT call for the game’s opening line
        # You can expand this prompt to mention day1, setting history, npc intros, etc.
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
    
        # 11) If it’s normal text, store it as a “Nyx” message
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
            "messages": conversation_history  # <-- add missing comma
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

    system_instructions = """
    You are setting up a new femdom daily-life sim scenario with a main quest. 
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

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        temperature=0.2,
        max_tokens=100,
        frequency_penalty=0.0
    )

    msg = response.choices[0].message.content.strip()

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
