# routes/new_game.py

import logging
from flask import Blueprint, request, jsonify
import random
from db.connection import get_db_connection
from routes.settings_routes import insert_missing_settings, generate_mega_setting_logic
from logic.npc_creation import create_npc

new_game_bp = Blueprint('new_game', __name__)

@new_game_bp.route('/start_new_game', methods=['POST'])
def start_new_game():
    """
    Clears out Settings, CurrentRoleplay, etc.
    Only 'Chase' remains in PlayerStats (default stats).
    Then re-inserts missing settings, spawns 10 unintroduced NPCs,
    and sets a new environment in CurrentRoleplay.
    Also returns environment name/desc, a sample schedule for Chase,
    and a short role summary for immediate grounding.
    """

    logging.info("=== START: /start_new_game CALLED ===")
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 1) Get request data, handle "params" if GPT front-end
        data = request.get_json() or {}
        logging.info(f"Raw incoming JSON data: {data}")
        if "params" in data:
            data = data["params"]
            logging.info(f"After unwrapping 'params': {data}")

        # 2) Clear relevant tables
        logging.info("Deleting old game state.")
        cursor.execute("DELETE FROM Events;")
        cursor.execute("DELETE FROM PlannedEvents;")
        cursor.execute("DELETE FROM PlayerInventory;")
        cursor.execute("DELETE FROM Quests;")
        cursor.execute("DELETE FROM NPCStats;")
        cursor.execute("DELETE FROM Locations;")
        cursor.execute("DELETE FROM SocialLinks;")
        cursor.execute("DELETE FROM CurrentRoleplay;")
        conn.commit()
        logging.info("Tables cleared successfully.")

        # 3) Reinsert default settings
        logging.info("Calling insert_missing_settings()")
        insert_missing_settings()
        logging.info("insert_missing_settings() completed.")

        # 4) Reset PlayerStats to keep only 'Chase'
        logging.info("Resetting PlayerStats to keep only 'Chase'.")
        cursor.execute("DELETE FROM PlayerStats WHERE player_name != 'Chase';")
        cursor.execute("SELECT id FROM PlayerStats WHERE player_name = 'Chase';")
        chase_row = cursor.fetchone()

        if chase_row:
            logging.info("Updating existing 'Chase' stats.")
            cursor.execute('''
                UPDATE PlayerStats
                SET corruption = 10,
                    confidence = 60,
                    willpower = 50,
                    obedience = 20,
                    dependency = 10,
                    lust = 15,
                    mental_resilience = 55,
                    physical_endurance = 40
                WHERE player_name = 'Chase'
            ''')
        else:
            logging.info("Inserting fresh row for 'Chase'.")
            cursor.execute('''
                INSERT INTO PlayerStats (
                  player_name, corruption, confidence, willpower, obedience,
                  dependency, lust, mental_resilience, physical_endurance
                )
                VALUES ('Chase', 10, 60, 50, 20, 10, 15, 55, 40)
            ''')
        conn.commit()
        logging.info("PlayerStats reset complete.")

        # 5) Spawn 10 new unintroduced NPCs
        logging.info("Spawning 10 new unintroduced NPCs via create_npc(introduced=False).")
        for i in range(10):
            new_id = create_npc(introduced=False)
            logging.info(f"Created new unintroduced NPC {i+1}/10, ID={new_id}")

        # 6) Re-insert missing settings (again, just to ensure no corners missed)
        logging.info("Calling insert_missing_settings() again.")
        insert_missing_settings()

        # 7) Generate environment & store in CurrentRoleplay
        logging.info("Generating new mega setting via generate_mega_setting_logic()")
        mega_data = generate_mega_setting_logic()
        if "error" in mega_data:
            logging.info("mega_data contained an error, forcing mega_name to 'No environment available'")
            mega_data["mega_name"] = "No environment available"

        environment_name = mega_data["mega_name"]
        logging.info(f"Setting CurrentSetting to: {environment_name}")
        cursor.execute("""
            INSERT INTO CurrentRoleplay (key, value)
            VALUES ('CurrentSetting', %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """, (environment_name,))
        conn.commit()

        # 8) Provide environment description, a sample weekly schedule for Chase, and a short role summary
        environment_desc = (
            "An eclectic realm combining monstrous societies, futuristic tech, "
            "and archaic ruins floating across the sky. Strange energies swirl, "
            "revealing hidden rituals and uncharted opportunities."
        )
        chase_schedule = {
            "Monday": {
                "Morning": "Wake at small inn",
                "Afternoon": "Guild errands",
                "Evening": "Leisure time",
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
                "Evening": "Leisure time",
                "Night": "Inn room rest"
            },
            "Thursday": {
                "Morning": "Physical training",
                "Afternoon": "Study mystical texts",
                "Evening": "Free time",
                "Night": "Return to inn"
            },            
            "Friday": {
                "Morning": "Wake at small inn",
                "Afternoon": "Guild errands",
                "Evening": "Leisure time",
                "Night": "Inn room rest"
            },
            "Saturday": {
                "Morning": "Physical training",
                "Afternoon": "Study mystical texts",
                "Evening": "Free time",
                "Night": "Return to inn"
            },
            "Sunday": {
                "Morning": "Physical training",
                "Afternoon": "Study mystical texts",
                "Evening": "Free time",
                "Night": "Return to inn"
            },            
        }
        chase_role = (
            "Chase is one of the only men in this world of dominant females. "
            "He scrapes by on odd jobs, forging bonds with the realm’s formidable denizens."
        )

        # 9) Return final message + environment data
        success_msg = (
            "New game started with 10 unintroduced NPCs. "
            f"New environment = {environment_name}"
        )
        logging.info(f"Success! Returning 200 with message: {success_msg}")

        return jsonify({
            "message": success_msg,
            "environment_name": environment_name,
            "environment_desc": environment_desc,
            "chase_schedule": chase_schedule,
            "chase_role": chase_role
        }), 200

    except Exception as e:
        logging.exception("Error in /start_new_game:")
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
        logging.info("=== END: /start_new_game ===")
