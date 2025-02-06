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

new_game_bp = Blueprint('new_game_bp', __name__)

@new_game_bp.route('/start_new_game', methods=['POST'])
def start_new_game():
    logging.info("=== START: /start_new_game CALLED ===")

    # 1. Confirm the user is logged in.
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        # 2. Parse the incoming JSON (unwrap "params" if present).
        data = request.get_json() or {}
        if "params" in data:
            data = data["params"]
        conversation_id = data.get("conversation_id")

        # 3. Generate the environment snippet.
        mega_data = generate_mega_setting_logic()
        if "error" in mega_data:
            mega_data["mega_name"] = "No environment available"
        environment_name = mega_data["mega_name"]
        environment_desc = (
            "An eclectic realm combining monstrous societies, futuristic tech, "
            "and archaic ruins floating across the sky. Strange energies swirl, "
            "revealing hidden rituals and uncharted opportunities."
        )

        # 4. Generate scenario name and quest if no conversation ID is provided.
        scenario_name, quest_blurb = ("New Game", "")
        if not conversation_id:
            scenario_name, quest_blurb = gpt_generate_scenario_name_and_quest(environment_name, environment_desc)

        # 5. Create or reuse the conversation.
        if not conversation_id:
            cursor.execute(
                """
                INSERT INTO conversations (user_id, conversation_name)
                VALUES (%s, %s)
                RETURNING id
                """,
                (user_id, scenario_name)
            )
            conversation_id = cursor.fetchone()[0]
            # Commit immediately so that subsequent operations (like create_npc) can see the new conversation.
            conn.commit()
            logging.info(f"Created new conversation_id={conversation_id} for user_id={user_id}, name={scenario_name}")
        else:
            cursor.execute(
                "SELECT id FROM conversations WHERE id=%s AND user_id=%s",
                (conversation_id, user_id)
            )
            if not cursor.fetchone():
                return jsonify({"error": f"Conversation {conversation_id} not found or unauthorized"}), 403
            logging.info(f"Using existing conversation_id={conversation_id} for user_id={user_id}")

        # 6. Clear old data from several tables.
        tables_to_clear = [
            "Events", "PlannedEvents", "PlayerInventory", "Quests",
            "NPCStats", "Locations", "SocialLinks", "CurrentRoleplay"
        ]
        for table in tables_to_clear:
            cursor.execute(f"DELETE FROM {table} WHERE user_id=%s AND conversation_id=%s", (user_id, conversation_id))
        logging.info(f"Cleared data for user_id={user_id}, conversation_id={conversation_id}")

        # 7. Insert environment data into CurrentRoleplay.
        roleplay_entries = {
            "MegaSettingModifiers": json.dumps(mega_data.get("stat_modifiers", {})),
            "CurrentSetting": environment_name
        }
        if quest_blurb.strip():
            roleplay_entries["MainQuest"] = quest_blurb.strip()
        for key, value in roleplay_entries.items():
            cursor.execute(
                """
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value=EXCLUDED.value
                """,
                (user_id, conversation_id, key, value)
            )

        # 8. Ensure all necessary settings exist.
        logging.info("Calling insert_missing_settings()")
        insert_missing_settings()

        # 9. Reset or create 'Chase' in PlayerStats.
        cursor.execute(
            """
            DELETE FROM PlayerStats
            WHERE user_id=%s AND conversation_id=%s AND player_name <> 'Chase'
            """,
            (user_id, conversation_id)
        )
        cursor.execute(
            """
            SELECT id FROM PlayerStats
            WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
            """,
            (user_id, conversation_id)
        )
        if cursor.fetchone():
            cursor.execute(
                '''
                UPDATE PlayerStats
                SET corruption=10, confidence=60, willpower=50, obedience=20,
                    dependency=10, lust=15, mental_resilience=55, physical_endurance=40
                WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
                ''',
                (user_id, conversation_id)
            )
            logging.info("Updated existing 'Chase' stats.")
        else:
            cursor.execute(
                '''
                INSERT INTO PlayerStats (
                  user_id, conversation_id, player_name,
                  corruption, confidence, willpower,
                  obedience, dependency, lust,
                  mental_resilience, physical_endurance
                )
                VALUES (%s, %s, 'Chase', 10, 60, 50, 20, 10, 15, 55, 40)
                ''',
                (user_id, conversation_id)
            )
            logging.info("Inserted fresh row for 'Chase'.")

        # 10. Spawn new NPCs (note: the loop comment mentioned 10 but here only 3 are created).
        for i in range(2):
            npc_id = create_npc(user_id=user_id, conversation_id=conversation_id, introduced=False)
            logging.info(f"Spawned NPC {i+1}/3, ID={npc_id}")

        # 11. Define the player's schedule and role.
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

        # 12. Get aggregated roleplay context and generate the opening narrative.
        aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, "Chase")
        aggregator_text = build_aggregator_text(aggregator_data)
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

        gpt_reply = get_chatgpt_response(
            conversation_id=conversation_id,
            aggregator_text=aggregator_text,
            user_input=opening_user_prompt
        )
        nyx_text = gpt_reply.get("response")

        # 13. If GPT returns a function call or empty text, retry with a forced text-only prompt.
        if gpt_reply.get("type") == "function_call" or not nyx_text:
            logging.info("GPT returned a function call or empty response; retrying without function calls.")
            client = get_openai_client()
            forced_messages = [
                {"role": "system", "content": aggregator_text},
                {"role": "user", "content": f"No function calls for the introduction. Produce only text narrative.\n\n{opening_user_prompt}"}
            ]
            fallback_response = client.chat.completions.create(
                model="gpt-4o",
                messages=forced_messages,
                temperature=0.7,
                timeout=120 
            )
            fallback_text = fallback_response.choices[0].message.content.strip()
            nyx_text = fallback_text if fallback_text else "[No text returned from GPT]"

        # 14. Store the final GPT response into the messages table.
        structured_json_str = json.dumps(gpt_reply)
        cursor.execute(
            """
            INSERT INTO messages (conversation_id, sender, content, structured_content)
            VALUES (%s, %s, %s, %s)
            """,
            (conversation_id, "Nyx", nyx_text, structured_json_str)
        )

        # 15. Retrieve conversation history.
        cursor.execute(
            """
            SELECT sender, content, created_at
            FROM messages
            WHERE conversation_id=%s
            ORDER BY id ASC
            """,
            (conversation_id,)
        )
        conversation_history = [
            {"sender": r[0], "content": r[1], "created_at": r[2].isoformat()}
            for r in cursor.fetchall()
        ]

        conn.commit()
        success_msg = f"New game started. Environment={environment_name}, conversation_id={conversation_id}"
        logging.info(f"Success! {success_msg}")

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
    Calls GPT to produce a short scenario name and a quest summary.
    Returns a tuple: (scenario_name, quest_blurb).
    """
    client = get_openai_client()
    unique_token = f"{random.randint(1000, 9999)}_{int(time.time())}"
    forbidden_words = ["mistress", "darkness", "manor", "chains", "twilight"]

    system_instructions = (
        f"You are setting up a new femdom daily-life sim scenario with a main quest.\n"
        f"Environment name: {env_name}\n"
        f"Environment desc: {env_desc}\n"
        f"Unique token: {unique_token}\n\n"
        "Please produce:\n"
        "1) A single line starting with 'ScenarioName:' followed by a short, creative (1–8 words) name "
        f"that draws from the environment above. Avoid cliche words like {', '.join(forbidden_words)}.\n"
        "2) Then one or two lines summarizing the main quest.\n\n"
        "The conversation name must be unique; do not reuse names from older scenarios "
        "(you can ensure uniqueness using the token or environment cues). "
        "Keep it thematically relevant to a fantasy/femdom environment, "
        "but do not use overly repeated phrases like 'Mistress of Darkness' or 'Chains of Twilight.'"
    )
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
