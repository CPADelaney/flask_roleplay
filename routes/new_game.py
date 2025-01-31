import logging
from flask import Blueprint, request, jsonify, session
import random
from db.connection import get_db_connection
from routes.settings_routes import insert_missing_settings, generate_mega_setting_logic
from logic.npc_creation import create_npc

new_game_bp = Blueprint('new_game_bp', __name__)

@new_game_bp.route('/start_new_game', methods=['POST'])
def start_new_game():
    """
    Creates or re-initializes a game scenario *for the current user, in a specific conversation*.
    - Accepts optional 'conversation_id' in JSON. If not provided, we create a new conversation row.
    - Clears data in relevant tables (Events, NPCStats, etc.) where user_id=this user AND conversation_id=this conversation.
    - Resets or creates 'Chase' in PlayerStats for (user_id, conversation_id).
    - Spawns 10 new unintroduced NPCs for (user_id, conversation_id).
    - Sets a new environment in CurrentRoleplay for (user_id, conversation_id).
    - Returns environment info + a sample schedule + conversation_id.
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

        # 3) Determine conversation_id
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            # Create a new conversation row for this user
            cursor.execute("""
                INSERT INTO conversations (user_id, conversation_name)
                VALUES (%s, %s)
                RETURNING id
            """, (user_id, "New Game"))
            conversation_id = cursor.fetchone()[0]
            logging.info(f"Created new conversation_id={conversation_id} for user_id={user_id}")
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

        # 4) Clear data for this user+conversation from events, NPCStats, etc.
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

        # 5) Insert missing settings if needed (if your 'Settings' is global, you can just call it once globally).
        logging.info("Calling insert_missing_settings() (global or local).")
        insert_missing_settings()  # or if it's global, it doesn't need user_id, conversation_id

        # 6) Reset or create the 'Chase' PlayerStats for user_id+conversation_id
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

        # 7) Spawn 10 new unintroduced NPCs in (user_id, conversation_id)
        logging.info("Spawning 10 new unintroduced NPCs.")
        from logic.npc_creation import create_npc  # ensure it can handle user_id/conversation_id
        for i in range(10):
            new_id = create_npc(
                user_id=user_id,
                conversation_id=conversation_id,
                introduced=False
            )
            logging.info(f"Created new unintroduced NPC {i+1}/10, ID={new_id}")

        # 8) Optionally re-insert missing settings again if you want? (unclear if needed)
        # insert_missing_settings() 

        # 9) Generate environment & store in CurrentRoleplay
        logging.info("Generating new mega setting via generate_mega_setting_logic()")
        mega_data = generate_mega_setting_logic(user_id, conversation_id)
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

        # Provide environment info
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

        return jsonify({
            "message": success_msg,
            "environment_name": environment_name,
            "environment_desc": environment_desc,
            "chase_schedule": chase_schedule,
            "chase_role": chase_role,
            "conversation_id": conversation_id
        }), 200

    except Exception as e:
        logging.exception("Error in /start_new_game:")
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
        logging.info("=== END: /start_new_game ===")
