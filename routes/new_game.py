import logging
import json
import random
import time
import asyncio
from flask import Blueprint, request, jsonify, session
import openai
import asyncpg

from routes.settings_routes import insert_missing_settings, generate_mega_setting_logic
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from logic.npc_creation import create_npc
from logic.aggregator import get_aggregated_roleplay_context
from routes.story_routes import build_aggregator_text

# Replace this with your actual Railway DSN.
DB_DSN = "postgresql://postgres:gUAfzAPnULbYOAvZeaOiwuKLLebutXEY@monorail.proxy.rlwy.net:24727/railway"

new_game_bp = Blueprint('new_game_bp', __name__)

@new_game_bp.route('/start_new_game', methods=['POST'])
async def start_new_game():
    """
    Asynchronously creates or re-initializes a game scenario.
    This version uses asyncpg for PostgreSQL access and offloads synchronous GPT and other functions via asyncio.to_thread().
    """
    logging.info("=== START: /start_new_game CALLED ===")
    
    # 1) Confirm user is logged in.
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    try:
        # 2) Parse request JSON.
        data = request.get_json() or {}
        logging.info(f"Raw incoming JSON data: {data}")
        if "params" in data:
            data = data["params"]
            logging.info(f"After unwrapping 'params': {data}")
        conversation_id = data.get("conversation_id")

        # 3) Generate environment snippet using a synchronous function offloaded to a thread.
        logging.info("Generating new mega setting via generate_mega_setting_logic()")
        mega_data = await asyncio.to_thread(generate_mega_setting_logic)
        if "error" in mega_data:
            mega_data["mega_name"] = "No environment available"
        environment_name = mega_data["mega_name"]
        environment_desc = (
            "An eclectic realm combining monstrous societies, futuristic tech, "
            "and archaic ruins floating across the sky. Strange energies swirl, "
            "revealing hidden rituals and uncharted opportunities."
        )
        logging.info(f"Environment snippet:\nName: {environment_name}\nDesc: {environment_desc}")

        # 4) If no conversation_id, call GPT for a scenario name & quest summary.
        scenario_name = "New Game"  # fallback value
        quest_blurb = ""
        if not conversation_id:
            scenario_name, quest_blurb = await asyncio.to_thread(
                gpt_generate_scenario_name_and_quest, environment_name, environment_desc
            )
            logging.info(f"GPT produced scenario_name={scenario_name}, quest_blurb={quest_blurb}")

        # 5) Connect to PostgreSQL using asyncpg.
        conn = await asyncpg.connect(dsn=DB_DSN)
        try:
            # Create or reuse the conversation.
            if not conversation_id:
                row = await conn.fetchrow("""
                    INSERT INTO conversations (user_id, conversation_name)
                    VALUES ($1, $2)
                    RETURNING id
                """, user_id, scenario_name)
                conversation_id = row["id"]
                logging.info(f"Created new conversation_id={conversation_id} for user_id={user_id}, name={scenario_name}")
            else:
                row = await conn.fetchrow("""
                    SELECT id FROM conversations WHERE id=$1 AND user_id=$2
                """, conversation_id, user_id)
                if not row:
                    return jsonify({"error": f"Conversation {conversation_id} not found or unauthorized"}), 403
                logging.info(f"Using existing conversation_id={conversation_id} for user_id={user_id}")

            # 6) Clear old data from relevant tables.
            logging.info(f"Clearing data for user_id={user_id}, conversation_id={conversation_id}")
            tables_to_clear = [
                "Events", "PlannedEvents", "PlayerInventory", "Quests",
                "NPCStats", "Locations", "SocialLinks", "CurrentRoleplay"
            ]
            for table in tables_to_clear:
                await conn.execute(f"DELETE FROM {table} WHERE user_id=$1 AND conversation_id=$2", user_id, conversation_id)

            # 7) Insert environment data (MegaSettingModifiers) into CurrentRoleplay.
            combined_modifiers_json = json.dumps(mega_data.get("stat_modifiers", {}))
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'MegaSettingModifiers', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value=EXCLUDED.value
            """, user_id, conversation_id, combined_modifiers_json)

            # 8) Insert missing settings (offloaded to a thread).
            logging.info("Calling insert_missing_settings()")
            await asyncio.to_thread(insert_missing_settings)

            # 9) Reset or create 'Chase' in PlayerStats.
            logging.info(f"Resetting PlayerStats for user_id={user_id}, conversation_id={conversation_id}, keeping only 'Chase'.")
            await conn.execute("""
                DELETE FROM PlayerStats
                WHERE user_id=$1 AND conversation_id=$2 AND player_name <> 'Chase'
            """, user_id, conversation_id)

            chase_row = await conn.fetchrow("""
                SELECT id FROM PlayerStats
                WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
            """, user_id, conversation_id)

            if chase_row:
                logging.info("Updating existing 'Chase' stats.")
                await conn.execute("""
                    UPDATE PlayerStats
                    SET corruption=10, confidence=60, willpower=50, obedience=20,
                        dependency=10, lust=15, mental_resilience=55, physical_endurance=40
                    WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
                """, user_id, conversation_id)
            else:
                logging.info("Inserting fresh row for 'Chase'.")
                await conn.execute("""
                    INSERT INTO PlayerStats (
                      user_id, conversation_id, player_name,
                      corruption, confidence, willpower,
                      obedience, dependency, lust,
                      mental_resilience, physical_endurance
                    )
                    VALUES ($1, $2, 'Chase', 10, 60, 50, 20, 10, 15, 55, 40)
                """, user_id, conversation_id)

            # 10) Spawn 10 new unintroduced NPCs.
            logging.info("Spawning 10 new unintroduced NPCs.")
            for i in range(10):
                new_id = await asyncio.to_thread(create_npc, user_id=user_id, conversation_id=conversation_id, introduced=False)
                logging.info(f"Created NPC {i+1}/10, ID={new_id}")

            # 11) Store environment name & main quest in CurrentRoleplay.
            logging.info(f"Storing environment name in CurrentRoleplay: {environment_name}")
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'CurrentSetting', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value=EXCLUDED.value
            """, user_id, conversation_id, environment_name)

            if quest_blurb.strip():
                await conn.execute("""
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'MainQuest', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value=EXCLUDED.value
                """, user_id, conversation_id, quest_blurb)

            # Build chase schedule and role.
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

            # 12) Get aggregated roleplay context and build aggregator text.
            aggregator_data = await asyncio.to_thread(get_aggregated_roleplay_context, user_id, conversation_id, "Chase")
            aggregator_text = await asyncio.to_thread(build_aggregator_text, aggregator_data)

            # 13) Prepare the GPT prompt for the game’s opening narrative.
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

            # 14) Call GPT for the opening line.
            gpt_reply_dict = await asyncio.to_thread(get_chatgpt_response, conversation_id, aggregator_text, opening_user_prompt)
            nyx_text = gpt_reply_dict.get("response")
            if gpt_reply_dict.get("type") == "function_call" or not nyx_text:
                logging.info("GPT attempted a function call or returned no text; retrying without function calls.")
                client = get_openai_client()
                forced_messages = [
                    {"role": "system", "content": aggregator_text},
                    {"role": "user", "content": f"No function calls for the introduction. Produce only text narrative.\n\n{opening_user_prompt}"}
                ]
                fallback_response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model="gpt-4o",
                    messages=forced_messages,
                    temperature=0.7
                )
                fallback_text = fallback_response.choices[0].message.content.strip()
                nyx_text = fallback_text if fallback_text else "[No text returned from GPT]"

            # 15) Store the GPT response in the messages table.
            structured_json_str = json.dumps(gpt_reply_dict)
            await conn.execute("""
                INSERT INTO messages (conversation_id, sender, content, structured_content)
                VALUES ($1, $2, $3, $4)
            """, conversation_id, "Nyx", nyx_text, structured_json_str)

            # 16) Retrieve conversation history.
            rows = await conn.fetch("""
                SELECT sender, content, created_at
                FROM messages
                WHERE conversation_id=$1
                ORDER BY id ASC
            """, conversation_id)
            conversation_history = []
            for row in rows:
                conversation_history.append({
                    "sender": row["sender"],
                    "content": row["content"],
                    "created_at": row["created_at"].isoformat()
                })

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

        except Exception as db_e:
            logging.exception("Error during DB operations in /start_new_game:")
            raise db_e
        finally:
            await conn.close()

    except Exception as e:
        logging.exception("Error in /start_new_game:")
        return jsonify({"error": str(e)}), 500
    finally:
        logging.info("=== END: /start_new_game ===")


def gpt_generate_scenario_name_and_quest(env_name: str, env_desc: str):
    """
    Synchronous helper that calls GPT to produce a short scenario name and a quest summary.
    Returns a tuple: (scenario_name, quest_blurb).
    """
    client = get_openai_client()
    unique_token = f"{random.randint(1000, 9999)}_{int(time.time())}"
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
    Keep it thematically relevant to a fantasy/femdom environment, 
    but do not use overly repeated phrases like 'Mistress of Darkness' or 'Chains of Twilight.'
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
