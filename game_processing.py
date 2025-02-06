# game_processing.py
import logging
import json
import random
import time
import asyncio
import asyncpg
from routes.settings_routes import insert_missing_settings, generate_mega_setting_logic
from logic.npc_creation import create_npc
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client
from logic.aggregator import get_aggregated_roleplay_context
from routes.story_routes import build_aggregator_text

# Use your Railway DSN (update as needed)
DB_DSN = "postgresql://postgres:gUAfzAPnULbYOAvZeaOiwuKLLebutXEY@monorail.proxy.rlwy.net:24727/railway"

# Helper: spaced-out GPT call (if needed)
async def spaced_gpt_call(conversation_id, context, prompt, delay=1.0):
    await asyncio.sleep(delay)
    return await asyncio.to_thread(get_chatgpt_response, conversation_id, context, prompt)

async def async_process_new_game(user_id, conversation_data):
    """
    This function encapsulates all the heavy processing for starting a new game.
    It performs GPT calls, database updates, etc. It returns a dictionary of results.
    """
    # If your incoming JSON has a "conversation_id" (or nested in "params"), extract it.
    provided_conversation_id = conversation_data.get("conversation_id")
    
    conn = await asyncpg.connect(dsn=DB_DSN)
    try:
        # 1. Create or validate conversation.
        if not provided_conversation_id:
            preliminary_name = "New Game"
            row = await conn.fetchrow("""
                INSERT INTO conversations (user_id, conversation_name)
                VALUES ($1, $2)
                RETURNING id
            """, user_id, preliminary_name)
            conversation_id = row["id"]
            logging.info(f"Pre-created conversation_id={conversation_id} for user_id={user_id}")
        else:
            conversation_id = provided_conversation_id
            row = await conn.fetchrow("""
                SELECT id FROM conversations WHERE id=$1 AND user_id=$2
            """, conversation_id, user_id)
            if not row:
                raise Exception(f"Conversation {conversation_id} not found or unauthorized")
        
        # 2. Generate environment snippet and history.
        mega_data = await asyncio.to_thread(generate_mega_setting_logic)
        if "error" in mega_data:
            mega_data["mega_name"] = "No environment available"
        environment_name = mega_data["mega_name"]
        base_environment_desc = (
            "An eclectic realm combining monstrous societies, futuristic tech, "
            "and archaic ruins floating across the sky. Strange energies swirl, "
            "revealing hidden rituals and uncharted opportunities."
        )
        history_prompt = (
            "Based on the following environment description, generate a brief, evocative history "
            "of this setting. Explain its origins, major past events, and its current state so that the narrative is well grounded. "
            "Include notable NPCs, important locations (including details about the town), and key cultural information such as holidays, festivals, and beliefs. "
            "\nEnvironment description: " + base_environment_desc
        )
        history_reply = await spaced_gpt_call(conversation_id, base_environment_desc, history_prompt)
        environment_history = history_reply.get("response", "").strip()
        if not environment_history:
            environment_history = "Ancient legends speak of forgotten gods and lost civilizations that once shaped this realm."
        environment_desc = f"{base_environment_desc}\n\nHistory: {environment_history}"
        
        # 3. Store EnvironmentDesc in CurrentRoleplay.
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'EnvironmentDesc', $3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, environment_desc)
        
        # 4. Generate and store notable Events.
        events_prompt = (
            "Based on the following environment description, generate a JSON array of notable events and holidays in this setting. "
            "Each event should be an object with keys 'name' and 'description' describing the event briefly. "
            "\nEnvironment description: " + environment_desc
        )
        events_reply = await spaced_gpt_call(conversation_id, environment_desc, events_prompt)
        events_response = events_reply.get("response", "").strip() if events_reply else ""
        try:
            events_json = json.loads(events_response)
        except Exception as e:
            logging.warning("Failed to parse events JSON, using fallback.", exc_info=e)
            events_json = []
        for event in events_json:
            event_name = event.get("name", "Unnamed Event")
            event_desc = event.get("description", "")
            await conn.execute("""
                INSERT INTO Events (user_id, conversation_id, event_name, event_description)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT DO NOTHING
            """, user_id, conversation_id, event_name, event_desc)
        
        # 5. Generate and store notable Locations.
        locations_prompt = (
            "Based on the following environment description, generate a JSON array of notable locations in this setting. "
            "Each location should be an object with keys 'name' and 'description' providing a brief overview of the location. "
            "\nEnvironment description: " + environment_desc
        )
        locations_reply = await spaced_gpt_call(conversation_id, environment_desc, locations_prompt)
        locations_response = locations_reply.get("response", "").strip()
        try:
            locations_json = json.loads(locations_response)
        except Exception as e:
            logging.warning("Failed to parse locations JSON, using fallback.", exc_info=e)
            locations_json = []
        for loc in locations_json:
            loc_name = loc.get("name", "Unnamed Location")
            loc_desc = loc.get("description", "")
            await conn.execute("""
                INSERT INTO Locations (user_id, conversation_id, location_name, location_description)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT DO NOTHING
            """, user_id, conversation_id, loc_name, loc_desc)
        
        # 6. If the conversation was just created, generate scenario name and quest summary.
        scenario_name = "New Game"
        quest_blurb = ""
        if not provided_conversation_id:
            # Here we call a synchronous helper; run it in a thread.
            scenario_name, quest_blurb = await asyncio.to_thread(gpt_generate_scenario_name_and_quest, environment_name, environment_desc)
            await conn.execute("""
                UPDATE conversations SET conversation_name=$1 WHERE id=$2
            """, scenario_name, conversation_id)
        
        # 7. Re-open connection for subsequent steps.
        await conn.close()
        conn = await asyncpg.connect(dsn=DB_DSN)
        
        # 8. Verify conversation exists (if provided).
        if provided_conversation_id:
            row = await conn.fetchrow("""
                SELECT id FROM conversations WHERE id=$1 AND user_id=$2
            """, conversation_id, user_id)
            if not row:
                await conn.close()
                raise Exception(f"Conversation {conversation_id} not found or unauthorized")
        
        # 9. Clear old game data.
        tables_to_clear = [
            "Events", "PlannedEvents", "PlayerInventory", "Quests",
            "NPCStats", "Locations", "SocialLinks", "CurrentRoleplay"
        ]
        for table in tables_to_clear:
            await conn.execute(f"DELETE FROM {table} WHERE user_id=$1 AND conversation_id=$2", user_id, conversation_id)
        
        # 10. Insert environment data (MegaSettingModifiers).
        modifiers_json = json.dumps(mega_data.get("stat_modifiers", {}))
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'MegaSettingModifiers', $3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, modifiers_json)
        
        # 11. Insert missing settings.
        await asyncio.to_thread(insert_missing_settings)
        
        # 12. Reset or create 'Chase' in PlayerStats.
        await conn.execute("""
            DELETE FROM PlayerStats
            WHERE user_id=$1 AND conversation_id=$2 AND player_name <> 'Chase'
        """, user_id, conversation_id)
        chase_row = await conn.fetchrow("""
            SELECT id FROM PlayerStats
            WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
        """, user_id, conversation_id)
        if chase_row:
            await conn.execute("""
                UPDATE PlayerStats
                SET corruption=10, confidence=60, willpower=50, obedience=20,
                    dependency=10, lust=15, mental_resilience=55, physical_endurance=40
                WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
            """, user_id, conversation_id)
        else:
            await conn.execute("""
                INSERT INTO PlayerStats (
                  user_id, conversation_id, player_name,
                  corruption, confidence, willpower,
                  obedience, dependency, lust,
                  mental_resilience, physical_endurance
                )
                VALUES ($1, $2, 'Chase', 10, 60, 50, 20, 10, 15, 55, 40)
            """, user_id, conversation_id)
        
        # 13. Store environment name as 'CurrentSetting'.
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'CurrentSetting', $3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, environment_name)
        
        # 14. Generate and store PlayerRole.
        player_role_prompt = (
            "Based on the current environment and setting, generate a concise description of Chase's typical day "
            "(e.g., career/daily life). The description should reflect how his role fits into this world of dominant females. "
            "In real life, Chase is a 31 year old data analyst, but this does not necessarily mean it will be the same. "
            "Career can be anything (student, etc.), but make sure it fits and makes sense within setting context."
        )
        player_role_reply = await spaced_gpt_call(conversation_id, environment_desc, player_role_prompt)
        player_role_text = player_role_reply.get("response", "Chase works a standard office job, barely scraping by.")
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'PlayerRole', $3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, player_role_text)
        
        # 15. Generate and store MainQuest.
        main_quest_prompt = (
            "Based on the current environment and the fact that Chase is one of the only men in this world of dominant females, "
            "generate a short summary of the main quest he is about to undertake. The quest should be intriguing and mysterious, "
            "hinting at challenges ahead without revealing too much."
        )
        main_quest_reply = await spaced_gpt_call(conversation_id, environment_desc, main_quest_prompt)
        main_quest_text = main_quest_reply.get("response", "Embark on a mysterious quest that challenges everything Chase thought he knew.")
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'MainQuest', $3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, main_quest_text)
        
        # 16. Generate and store ChaseSchedule.
        schedule_prompt = (
            "Based on the current environment and Chase's role, generate a detailed weekly schedule for Chase. "
            "Format the schedule as valid JSON with keys for each day of the week (e.g., Monday, Tuesday, etc.)."
        )
        schedule_reply = await spaced_gpt_call(conversation_id, environment_desc, schedule_prompt)
        chase_schedule_generated = schedule_reply.get("response", "{}")
        try:
            chase_schedule = json.loads(chase_schedule_generated)
        except Exception:
            chase_schedule = {
                "Monday": {"Morning": "Wake at a cozy inn, have a quick breakfast",
                           "Afternoon": "Head to work at the local data office",
                           "Evening": "Attend a casual meetup with friends",
                           "Night": "Return to the inn for rest"},
                "Tuesday": {"Morning": "Jog along the city walls, enjoy the sunrise",
                            "Afternoon": "Study mystical texts at the library",
                            "Evening": "Work on personal creative projects",
                            "Night": "Return to the inn and unwind"},
                "Wednesday": {"Morning": "Wake at the inn and enjoy a hearty breakfast",
                              "Afternoon": "Run errands and visit the guild",
                              "Evening": "Attend a community dinner",
                              "Night": "Head back to the inn for some rest"},
                "Thursday": {"Morning": "Do light training at the local gym",
                             "Afternoon": "Work at the office",
                             "Evening": "Meet with friends at a nearby tavern",
                             "Night": "Return home for sleep"},
                "Friday": {"Morning": "Wake up at the inn",
                           "Afternoon": "Wrap up work and relax",
                           "Evening": "Attend a small social gathering",
                           "Night": "Take a leisurely late night stroll"},
                "Saturday": {"Morning": "Sleep in and enjoy a lazy start",
                             "Afternoon": "Explore the bustling market",
                             "Evening": "Watch a local performance",
                             "Night": "Return to the inn to wind down"},
                "Sunday": {"Morning": "Take an early walk in the park",
                           "Afternoon": "Reflect on the week and plan ahead",
                           "Evening": "Have a light dinner with friends",
                           "Night": "Enjoy some quiet time before sleep"}
            }
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'ChaseSchedule', $3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, json.dumps(chase_schedule))
        
        # 17. Build aggregated context.
        aggregator_data = await asyncio.to_thread(get_aggregated_roleplay_context, user_id, conversation_id, "Chase")
        aggregator_text = await asyncio.to_thread(build_aggregator_text, aggregator_data)
        
        # 18. Call GPT for the opening narrative using the aggregated context.
        opening_user_prompt = (
            "Begin the scenario now, Nyx. Greet Chase with your sadistic, mocking style, avoiding clichéd phrases. "
            "Format your greeting using Markdown sections. Briefly recount the new environment’s background from the aggregator data, "
            "and announce that Monday morning has just begun. Describe where Chase is that morning by referencing the schedule, "
            "the player's role, and hint at the mysterious main quest. "
            "Stay fully in character and conclude with a teasing invitation for Chase to proceed."
        )
        gpt_reply_dict = await spaced_gpt_call(conversation_id, aggregator_text, opening_user_prompt)
        nyx_text = gpt_reply_dict.get("response")
        if gpt_reply_dict.get("type") == "function_call" or not nyx_text:
            logging.info("GPT attempted a function call or returned no text; retrying without function calls.")
            client = get_openai_client()
            forced_messages = [
                {"role": "system", "content": aggregator_text},
                {"role": "user", "content": f"No function calls. Produce only a text narrative.\n\n{opening_user_prompt}"}
            ]
            fallback_response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o",
                messages=forced_messages,
                temperature=0.7
            )
            fallback_text = fallback_response.choices[0].message.content.strip()
            nyx_text = fallback_text if fallback_text else "[No text returned from GPT]"
        
        # 19. Store the GPT response in messages.
        structured_json_str = json.dumps(gpt_reply_dict)
        await conn.execute("""
            INSERT INTO messages (conversation_id, sender, content, structured_content)
            VALUES ($1, $2, $3, $4)
        """, conversation_id, "Nyx", nyx_text, structured_json_str)
        
        # 20. Retrieve conversation history.
        rows = await conn.fetch("""
            SELECT sender, content, created_at
            FROM messages
            WHERE conversation_id=$1
            ORDER BY id ASC
        """, conversation_id)
        conversation_history = [{
            "sender": row["sender"],
            "content": row["content"],
            "created_at": row["created_at"].isoformat()
        } for row in rows]
        
        success_msg = f"New game started. Environment={environment_name}, conversation_id={conversation_id}"
        logging.info(success_msg)
        return {
            "message": success_msg,
            "scenario_name": scenario_name,
            "environment_name": environment_name,
            "environment_desc": environment_desc,
            "chase_schedule": chase_schedule,
            "chase_role": player_role_text,
            "conversation_id": conversation_id,
            "messages": conversation_history
        }
         
    except Exception as e:
         logging.exception("Error in async_process_new_game:")
         return {"error": str(e)}
    finally:
         logging.info("=== END: async_process_new_game ===")
         try:
             await conn.close()
         except Exception:
             pass

def gpt_generate_scenario_name_and_quest(env_name: str, env_desc: str):
    """
    Synchronous helper that calls GPT for a short scenario name and a quest summary.
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

async def spawn_npcs():
    """
    Spawns NPCs for a given conversation.
    Expects JSON with a 'conversation_id'.
    Spawns 5 NPCs concurrently using asyncio.gather.
    """
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json() or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id"}), 400

    conn = await asyncpg.connect(dsn=DB_DSN)
    try:
        row = await conn.fetchrow("SELECT id FROM conversations WHERE id=$1 AND user_id=$2", conversation_id, user_id)
        if not row:
            return jsonify({"error": "Conversation not found or unauthorized"}), 403

        spawn_tasks = [
            asyncio.to_thread(create_npc, user_id=user_id, conversation_id=conversation_id, introduced=False)
            for _ in range(5)
        ]
        spawned_npc_ids = await asyncio.gather(*spawn_tasks)
        logging.info(f"Spawned NPCs concurrently: {spawned_npc_ids}")
        return jsonify({"message": "NPCs spawned", "npc_ids": spawned_npc_ids}), 200
    except Exception as e:
        logging.exception("Error in /spawn_npcs:")
        return jsonify({"error": str(e)}), 500
    finally:
        await conn.close()
