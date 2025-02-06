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

# ---------------------------------------------------------------------
# Helper to query the CurrentRoleplay table for a stored value.
async def get_stored_value(conn, user_id, conversation_id, key):
    row = await conn.fetchrow(
        "SELECT value FROM CurrentRoleplay WHERE user_id=$1 AND conversation_id=$2 AND key=$3",
        user_id, conversation_id, key
    )
    if row:
        return row["value"]
    return None

# ---------------------------------------------------------------------
# Helper: spaced-out GPT call.
async def spaced_gpt_call(conversation_id, context, prompt, delay=1.0):
    logging.info("Waiting for %.1f seconds before calling GPT (conversation_id=%s)", delay, conversation_id)
    await asyncio.sleep(delay)
    logging.info("Calling GPT with conversation_id=%s", conversation_id)
    result = await asyncio.to_thread(get_chatgpt_response, conversation_id, context, prompt)
    logging.info("GPT returned response: %s", result)
    return result

# ---------------------------------------------------------------------
# Universal update function (asynchronous version)
async def apply_universal_update(user_id, conversation_id, update_data, conn):
    """
    Processes the universal_update payload, inserting or updating DB records.
    This function has been converted to async/await using asyncpg.
    """
    logging.info("Applying universal update for conversation_id=%s with data: %s", conversation_id, update_data)
    
    # 1) roleplay_updates
    rp_updates = update_data.get("roleplay_updates", {})
    logging.info("[apply_universal_update] roleplay_updates: %s", rp_updates)
    for key, val in rp_updates.items():
        if isinstance(val, (dict, list)):
            stored_val = json.dumps(val)
        else:
            stored_val = str(val)
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value = EXCLUDED.value
        """, user_id, conversation_id, key, stored_val)
        logging.info("Inserted/Updated CurrentRoleplay => key=%s, value=%s", key, val)
    
    # 2) npc_creations
    npc_creations = update_data.get("npc_creations", [])
    logging.info("[apply_universal_update] npc_creations: %s", npc_creations)
    for npc_data in npc_creations:
        name = npc_data.get("npc_name", "Unnamed NPC")
        introduced = npc_data.get("introduced", False)
        arche = npc_data.get("archetypes", [])
        dom = npc_data.get("dominance", 0)
        cru = npc_data.get("cruelty", 0)
        clos = npc_data.get("closeness", 0)
        tru = npc_data.get("trust", 0)
        resp = npc_data.get("respect", 0)
        inten = npc_data.get("intensity", 0)
        hbs = npc_data.get("hobbies", [])
        pers = npc_data.get("personality_traits", [])
        lks = npc_data.get("likes", [])
        dlks = npc_data.get("dislikes", [])
        affil = npc_data.get("affiliations", [])
        sched = npc_data.get("schedule", {})
        mem = npc_data.get("memory", [])
        if isinstance(mem, str):
            mem = [mem]
        monica_lvl = npc_data.get("monica_level", 0)
        sex = npc_data.get("sex", None)
        
        row = await conn.fetchrow("""
            SELECT npc_id FROM NPCStats
            WHERE user_id=$1 AND conversation_id=$2 AND LOWER(npc_name)=$3
            LIMIT 1
        """, user_id, conversation_id, name.lower())
        if row:
            logging.info("Skipping NPC creation, '%s' already exists.", name)
            continue
        logging.info("Creating NPC: %s, introduced=%s, dominance=%s, cruelty=%s", name, introduced, dom, cru)
        await conn.execute("""
            INSERT INTO NPCStats (
                user_id, conversation_id, npc_name, introduced,
                archetypes, dominance, cruelty, closeness, trust, respect, intensity,
                hobbies, personality_traits, likes, dislikes,
                affiliations, schedule, memory, monica_level, sex
            )
            VALUES (
                $1, $2, $3, $4,
                $5, $6, $7, $8, $9, $10, $11,
                $12, $13, $14, $15,
                $16, $17, $18, $19, $20
            )
        """, user_id, conversation_id, name, introduced, json.dumps(arche),
           dom, cru, clos, tru, resp, inten,
           json.dumps(hbs), json.dumps(pers), json.dumps(lks), json.dumps(dlks),
           json.dumps(affil), json.dumps(sched), json.dumps(mem),
           monica_lvl, sex)
        logging.info("NPC creation insert complete for %s", name)
    
    # 3) npc_updates
    npc_updates = update_data.get("npc_updates", [])
    logging.info("[apply_universal_update] npc_updates: %s", npc_updates)
    for up in npc_updates:
        npc_id = up.get("npc_id")
        if not npc_id:
            logging.warning("Skipping npc_update: missing npc_id.")
            continue
        fields_map = {
            "npc_name": "npc_name",
            "introduced": "introduced",
            "dominance": "dominance",
            "cruelty": "cruelty",
            "closeness": "closeness",
            "trust": "trust",
            "respect": "respect",
            "intensity": "intensity",
            "monica_level": "monica_level",
            "sex": "sex"
        }
        set_clauses = []
        set_vals = []
        for field_key, db_col in fields_map.items():
            if field_key in up:
                set_clauses.append(f"{db_col} = $%d" % (len(set_vals) + 1))
                set_vals.append(up[field_key])
        if set_clauses:
            set_str = ", ".join(set_clauses)
            set_vals.extend([npc_id, user_id, conversation_id])
            query = f"""
                UPDATE NPCStats
                SET {set_str}
                WHERE npc_id=$%d AND user_id=$%d AND conversation_id=$%d
            """ % (len(set_vals)-2, len(set_vals)-1, len(set_vals))
            logging.info("Updating NPC %s with %s", npc_id, set_clauses)
            await conn.execute(query, *set_vals)
            logging.info("Updated NPC %s", npc_id)
        if "memory" in up:
            new_mem_entries = up["memory"]
            if isinstance(new_mem_entries, str):
                new_mem_entries = [new_mem_entries]
            logging.info("Appending memory to NPC %s: %s", npc_id, new_mem_entries)
            await conn.execute("""
                UPDATE NPCStats
                SET memory = COALESCE(memory, '[]'::jsonb) || to_jsonb($1::text)
                WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
            """, new_mem_entries, npc_id, user_id, conversation_id)
        if "schedule" in up:
            new_schedule = up["schedule"]
            logging.info("Overwriting schedule for NPC %s: %s", npc_id, new_schedule)
            await conn.execute("""
                UPDATE NPCStats
                SET schedule=$1
                WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
            """, json.dumps(new_schedule), npc_id, user_id, conversation_id)
        if "schedule_updates" in up:
            partial_sched = up["schedule_updates"]
            logging.info("Merging schedule_updates for NPC %s: %s", npc_id, partial_sched)
            row = await conn.fetchrow("""
                SELECT schedule FROM NPCStats
                WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
            """, npc_id, user_id, conversation_id)
            if row:
                existing_schedule = row[0] or {}
                for day_key, times_map in partial_sched.items():
                    if day_key not in existing_schedule:
                        existing_schedule[day_key] = {}
                    existing_schedule[day_key].update(times_map)
                await conn.execute("""
                    UPDATE NPCStats
                    SET schedule=$1
                    WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
                """, json.dumps(existing_schedule), npc_id, user_id, conversation_id)
    
    # 4) character_stat_updates
    char_update = update_data.get("character_stat_updates", {})
    logging.info("[apply_universal_update] character_stat_updates: %s", char_update)
    if char_update:
        p_name = char_update.get("player_name", "Chase")
        stats = char_update.get("stats", {})
        stat_map = {
            "corruption": "corruption",
            "confidence": "confidence",
            "willpower": "willpower",
            "obedience": "obedience",
            "dependency": "dependency",
            "lust": "lust",
            "mental_resilience": "mental_resilience",
            "physical_endurance": "physical_endurance"
        }
        set_clauses = []
        set_vals = []
        for k, col in stat_map.items():
            if k in stats:
                set_clauses.append(f"{col}=$%d" % (len(set_vals)+1))
                set_vals.append(stats[k])
        if set_clauses:
            set_str = ", ".join(set_clauses)
            set_vals.extend([p_name, user_id, conversation_id])
            query = f"""
                UPDATE PlayerStats
                SET {set_str}
                WHERE player_name=$%d AND user_id=$%d AND conversation_id=$%d
            """ % (len(set_vals)-2, len(set_vals)-1, len(set_vals))
            logging.info("Updating player stats for %s: %s", p_name, stats)
            await conn.execute(query, *set_vals)
    
    # 5) relationship_updates
    rel_updates = update_data.get("relationship_updates", [])
    logging.info("[apply_universal_update] relationship_updates: %s", rel_updates)
    for r in rel_updates:
        npc_id = r.get("npc_id")
        if not npc_id:
            logging.warning("Skipping relationship update: missing npc_id.")
            continue
        aff_list = r.get("affiliations", None)
        if aff_list is not None:
            logging.info("Updating affiliations for NPC %s: %s", npc_id, aff_list)
            await conn.execute("""
                UPDATE NPCStats
                SET affiliations=$1
                WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
            """, json.dumps(aff_list), npc_id, user_id, conversation_id)
    
    # 6) shared_memory_updates
    shared_memory_updates = update_data.get("shared_memory_updates", [])
    logging.info("[apply_universal_update] shared_memory_updates: %s", shared_memory_updates)
    for sm_update in shared_memory_updates:
        npc_id = sm_update.get("npc_id")
        relationship = sm_update.get("relationship")
        if not npc_id or not relationship:
            logging.warning("Skipping shared memory update: missing npc_id or relationship data.")
            continue
        row = await conn.fetchrow("""
            SELECT npc_name FROM NPCStats
            WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
        """, npc_id, user_id, conversation_id)
        if not row:
            logging.warning("Shared memory update: NPC with id %s not found.", npc_id)
            continue
        npc_name = row[0]
        from logic.memory import get_shared_memory
        shared_memory_text = get_shared_memory(relationship, npc_name)
        logging.info("Generated shared memory for NPC %s: %s", npc_id, shared_memory_text)
        await conn.execute("""
            UPDATE NPCStats
            SET memory = COALESCE(memory, '[]'::jsonb) || to_jsonb($1::text)
            WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
        """, shared_memory_text, npc_id, user_id, conversation_id)
        logging.info("Appended shared memory to NPC %s", npc_id)
    
    # 7) npc_introductions
    npc_intros = update_data.get("npc_introductions", [])
    logging.info("[apply_universal_update] npc_introductions: %s", npc_intros)
    for intro in npc_intros:
        nid = intro.get("npc_id")
        if nid:
            logging.info("Marking NPC %s as introduced", nid)
            await conn.execute("""
                UPDATE NPCStats
                SET introduced=TRUE
                WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
            """, nid, user_id, conversation_id)
    
    # 8) location_creations
    loc_creations = update_data.get("location_creations", [])
    logging.info("[apply_universal_update] location_creations: %s", loc_creations)
    for loc in loc_creations:
        loc_name = loc.get("location_name", "Unnamed")
        desc = loc.get("description", "")
        open_hours = loc.get("open_hours", [])
        row = await conn.fetchrow("""
            SELECT id FROM Locations
            WHERE user_id=$1 AND conversation_id=$2 AND LOWER(location_name)=$3
            LIMIT 1
        """, user_id, conversation_id, loc_name.lower())
        if row:
            logging.info("Skipping location creation, '%s' already exists.", loc_name)
            continue
        logging.info("Inserting location => location_name=%s, description=%s, open_hours=%s", loc_name, desc, open_hours)
        await conn.execute("""
            INSERT INTO Locations (user_id, conversation_id, location_name, description, open_hours)
            VALUES ($1, $2, $3, $4, $5)
        """, user_id, conversation_id, loc_name, desc, json.dumps(open_hours))
        logging.info("Inserted location: %s", loc_name)
    
    # 9) event_list_updates
    event_updates = update_data.get("event_list_updates", [])
    logging.info("[apply_universal_update] event_list_updates: %s", event_updates)
    for ev in event_updates:
        if "npc_id" in ev and "day" in ev and "time_of_day" in ev:
            npc_id = ev["npc_id"]
            day = ev["day"]
            tod = ev["time_of_day"]
            ov_loc = ev.get("override_location", "Unknown")
            row = await conn.fetchrow("""
                SELECT event_id FROM PlannedEvents
                WHERE user_id=$1 AND conversation_id=$2
                  AND npc_id=$3 AND day=$4 AND time_of_day=$5
                LIMIT 1
            """, user_id, conversation_id, npc_id, day, tod)
            if row:
                logging.info("Skipping planned event creation; day=%s, time_of_day=%s, npc=%s already exists.", day, tod, npc_id)
                continue
            logging.info("Inserting PlannedEvent => npc_id=%s, day=%s, time_of_day=%s, override_location=%s", npc_id, day, tod, ov_loc)
            await conn.execute("""
                INSERT INTO PlannedEvents (user_id, conversation_id, npc_id, day, time_of_day, override_location)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, user_id, conversation_id, npc_id, day, tod, ov_loc)
        else:
            ev_name = ev.get("event_name", "UnnamedEvent")
            ev_desc = ev.get("description", "")
            ev_start = ev.get("start_time", "TBD Start")
            ev_end = ev.get("end_time", "TBD End")
            ev_loc = ev.get("location", "Unknown")
            row = await conn.fetchrow("""
                SELECT id FROM Events
                WHERE user_id=$1 AND conversation_id=$2
                  AND LOWER(event_name)=$3
                LIMIT 1
            """, user_id, conversation_id, ev_name.lower())
            if row:
                logging.info("Skipping event creation; '%s' already exists.", ev_name)
                continue
            logging.info("Inserting Event => %s, location=%s, times=%s-%s", ev_name, ev_loc, ev_start, ev_end)
            await conn.execute("""
                INSERT INTO Events (user_id, conversation_id, event_name, description, start_time, end_time, location)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, user_id, conversation_id, ev_name, ev_desc, ev_start, ev_end, ev_loc)
    
    # 10) inventory_updates
    inv_updates = update_data.get("inventory_updates", {})
    logging.info("[apply_universal_update] inventory_updates: %s", inv_updates)
    if inv_updates:
        p_n = inv_updates.get("player_name", "Chase")
        added = inv_updates.get("added_items", [])
        removed = inv_updates.get("removed_items", [])
        for item in added:
            if isinstance(item, dict):
                item_name = item.get("item_name", "Unnamed")
                item_desc = item.get("item_description", "")
                item_fx   = item.get("item_effect", "")
                category  = item.get("category", "")
                logging.info("Adding item for %s: %s", p_n, item_name)
                await conn.execute("""
                    INSERT INTO PlayerInventory (user_id, conversation_id, player_name, item_name, item_description, item_effect, category, quantity)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, 1)
                    ON CONFLICT (user_id, conversation_id, player_name, item_name)
                    DO UPDATE SET quantity = PlayerInventory.quantity + 1
                """, user_id, conversation_id, p_n, item_name, item_desc, item_fx, category)
            elif isinstance(item, str):
                logging.info("Adding item (string) for %s: %s", p_n, item)
                await conn.execute("""
                    INSERT INTO PlayerInventory (user_id, conversation_id, player_name, item_name, quantity)
                    VALUES ($1, $2, $3, $4, 1)
                    ON CONFLICT (user_id, conversation_id, player_name, item_name)
                    DO UPDATE SET quantity = PlayerInventory.quantity + 1
                """, user_id, conversation_id, p_n, item)
        for item in removed:
            if isinstance(item, dict):
                i_name = item.get("item_name")
                if i_name:
                    logging.info("Removing item for %s: %s", p_n, i_name)
                    await conn.execute("""
                        DELETE FROM PlayerInventory
                        WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3 AND item_name=$4
                    """, user_id, conversation_id, p_n, i_name)
            elif isinstance(item, str):
                logging.info("Removing item for %s: %s", p_n, item)
                await conn.execute("""
                    DELETE FROM PlayerInventory
                    WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3 AND item_name=$4
                """, user_id, conversation_id, p_n, item)
    
    # 11) quest_updates
    quest_updates = update_data.get("quest_updates", [])
    logging.info("[apply_universal_update] quest_updates: %s", quest_updates)
    for qu in quest_updates:
        qid = qu.get("quest_id")
        status = qu.get("status", "In Progress")
        detail = qu.get("progress_detail", "")
        qgiver = qu.get("quest_giver", "")
        reward = qu.get("reward", "")
        qname  = qu.get("quest_name", None)
        if qid:
            logging.info("Updating Quest %s: status=%s, detail=%s", qid, status, detail)
            await conn.execute("""
                UPDATE Quests
                SET status=$1, progress_detail=$2, quest_giver=$3, reward=$4
                WHERE quest_id=$5 AND user_id=$6 AND conversation_id=$7
            """, status, detail, qgiver, reward, qid, user_id, conversation_id)
            row = await conn.fetchrow("""
                SELECT 1 FROM Quests WHERE quest_id=$1 AND user_id=$2 AND conversation_id=$3
            """, qid, user_id, conversation_id)
            if not row:
                logging.info("No existing quest with ID %s, inserting new.", qid)
                await conn.execute("""
                    INSERT INTO Quests (user_id, conversation_id, quest_id, quest_name, status, progress_detail, quest_giver, reward)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, user_id, conversation_id, qid, qname, status, detail, qgiver, reward)
        else:
            logging.info("Inserting new quest: %s, status=%s", qname, status)
            await conn.execute("""
                INSERT INTO Quests (user_id, conversation_id, quest_name, status, progress_detail, quest_giver, reward)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, user_id, conversation_id, qname or "Unnamed Quest", status, detail, qgiver, reward)
    
    logging.info("=== [apply_universal_update] Success! ===")
    await conn.commit()
    return {"message": "Universal update successful"}

# ---------------------------------------------------------------------
# Main game processing function
async def async_process_new_game(user_id, conversation_data):
    logging.info("=== Starting async_process_new_game for user_id=%s with conversation_data=%s ===", user_id, conversation_data)
    
    # Step 1: Create or validate conversation.
    provided_conversation_id = conversation_data.get("conversation_id")
    conn = await asyncpg.connect(dsn=DB_DSN)
    try:
        if not provided_conversation_id:
            preliminary_name = "New Game"
            logging.info("No conversation_id provided; creating a new conversation for user_id=%s", user_id)
            row = await conn.fetchrow("""
                INSERT INTO conversations (user_id, conversation_name)
                VALUES ($1, $2)
                RETURNING id
            """, user_id, preliminary_name)
            conversation_id = row["id"]
            logging.info("Created new conversation with id=%s for user_id=%s", conversation_id, user_id)
        else:
            conversation_id = provided_conversation_id
            logging.info("Validating provided conversation_id=%s for user_id=%s", conversation_id, user_id)
            row = await conn.fetchrow("""
                SELECT id FROM conversations WHERE id=$1 AND user_id=$2
            """, conversation_id, user_id)
            if not row:
                raise Exception(f"Conversation {conversation_id} not found or unauthorized")
            logging.info("Validated existing conversation with id=%s", conversation_id)
        
        # Step 2: Dynamically generate environment components, a setting name, and a cohesive description.
        logging.info("Generating mega setting logic for conversation_id=%s", conversation_id)
        mega_data = await asyncio.to_thread(generate_mega_setting_logic)
        unique_envs = mega_data.get("unique_environments", [])
        if not unique_envs or len(unique_envs) == 0:
            unique_envs = [
                "A sprawling cyberpunk metropolis under siege by monstrous clans",
                "Floating archaic ruins steeped in ancient rituals",
                "Futuristic tech hubs that blend magic and machinery"
            ]
        
        # --- Generate a dynamic setting name ---
        name_prompt = "Given the following environment components:\n"
        for i, env in enumerate(unique_envs):
            name_prompt += f"Component {i+1}: {env}\n"
        name_prompt += (
            "Describe how these components come together to form a cohesive world and generate a short, creative name for the overall setting. "
            "Return only the name."
        )
        logging.info("Calling GPT for dynamic setting name with prompt: %s", name_prompt)
        setting_name_reply = await spaced_gpt_call(conversation_id, "", name_prompt)
        dynamic_setting_name = setting_name_reply.get("response", "")
        if dynamic_setting_name:
            dynamic_setting_name = dynamic_setting_name.strip()
        else:
            stored_setting = await get_stored_value(conn, user_id, conversation_id, "CurrentSetting")
            if stored_setting:
                dynamic_setting_name = stored_setting
            else:
                logging.warning("GPT returned no setting name and none stored; using fallback.")
                dynamic_setting_name = "Default Setting Name"
        logging.info("Generated dynamic setting name: %s", dynamic_setting_name)
        environment_name = dynamic_setting_name
        
        # --- Generate a cohesive environment description ---
        env_desc_prompt = "Using the following environment components:\n"
        for i, env in enumerate(unique_envs):
            env_desc_prompt += f"Component {i+1}: {env}\n"
        env_desc_prompt += (
            "Describe in a cohesive narrative how these components combine to form a unique, dynamic world."
        )
        logging.info("Calling GPT for dynamic environment description with prompt: %s", env_desc_prompt)
        env_desc_reply = await spaced_gpt_call(conversation_id, "", env_desc_prompt)
        base_environment_desc = env_desc_reply.get("response")
        if base_environment_desc is None or not base_environment_desc.strip():
            stored_env_desc = await get_stored_value(conn, user_id, conversation_id, "EnvironmentDesc")
            if stored_env_desc:
                base_environment_desc = stored_env_desc
            else:
                logging.warning("GPT returned no dynamic environment description and none stored; using fallback text.")
                base_environment_desc = (
                    "An eclectic realm combining monstrous societies, futuristic tech, "
                    "and archaic ruins floating across the sky. Strange energies swirl, "
                    "revealing hidden rituals and uncharted opportunities."
                )
        else:
            base_environment_desc = base_environment_desc.strip()
        
        # --- Generate the setting history based on the dynamic description ---
        history_prompt = (
            "Based on the following environment description, generate a brief, evocative history "
            "of this setting. Explain its origins, major past events, and its current state so that the narrative is well grounded. "
            "Include notable NPCs, important locations (with details about the town), and key cultural information such as holidays, festivals, and beliefs. "
            "\nEnvironment description: " + base_environment_desc
        )
        logging.info("Calling GPT for environment history with prompt: %s", history_prompt)
        history_reply = await spaced_gpt_call(conversation_id, base_environment_desc, history_prompt)
        if history_reply.get("type") == "function_call":
            logging.info("GPT returned a function call for environment history. Processing function call.")
            function_args = history_reply.get("function_args", {})
            await apply_universal_update(user_id, conversation_id, function_args, conn)
            stored_env_desc = await get_stored_value(conn, user_id, conversation_id, "EnvironmentDesc")
            if stored_env_desc:
                environment_history = stored_env_desc
            else:
                environment_history = "World context updated via GPT function call."
        else:
            response_text = history_reply.get("response")
            if response_text is None or not response_text.strip():
                stored_env_desc = await get_stored_value(conn, user_id, conversation_id, "EnvironmentDesc")
                if stored_env_desc:
                    environment_history = stored_env_desc
                else:
                    logging.warning("GPT returned no response text for environment history and none stored; using fallback text.")
                    environment_history = "Ancient legends speak of forgotten gods and lost civilizations that once shaped this realm."
            else:
                environment_history = response_text.strip()
        
        environment_desc = f"{base_environment_desc}\n\nHistory: {environment_history}"
        logging.info("Constructed environment description: %s", environment_desc)
        
        # Step 3: Store EnvironmentDesc in CurrentRoleplay.
        logging.info("Storing EnvironmentDesc in CurrentRoleplay for conversation_id=%s", conversation_id)
        await conn.execute("""
            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
            VALUES ($1, $2, 'EnvironmentDesc', $3)
            ON CONFLICT (user_id, conversation_id, key)
            DO UPDATE SET value=EXCLUDED.value
        """, user_id, conversation_id, environment_desc)
        
        # (Steps 4 through 20 remain unchanged; see the full code above.)
        # For brevity, they are identical to the previously provided implementation.
        # ...
        # (Include your steps for generating events, locations, scenario name, quest summary,
        #  inserting missing settings, resetting PlayerStats, storing CurrentSetting, generating
        #  PlayerRole, MainQuest, ChaseSchedule, building aggregated context, calling GPT for the opening narrative,
        #  storing the GPT response in messages, and retrieving conversation history.)
        
        # For this example, we assume that the remainder of the code (steps 4-20) remains unchanged.
        # At the end, return the results:
        success_msg = f"New game started. Environment={environment_name}, conversation_id={conversation_id}"
        logging.info(success_msg)
        # (Assuming conversation_history, chase_schedule, player_role_text, scenario_name, quest_blurb are defined in steps 4-16.)
        return {
            "message": success_msg,
            "scenario_name": scenario_name,
            "environment_name": environment_name,
            "environment_desc": environment_desc,
            # "chase_schedule": chase_schedule,
            # "chase_role": player_role_text,
            "conversation_id": conversation_id,
            # "messages": conversation_history
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

# ---------------------------------------------------------------------
# Helper function for generating scenario name and quest summary.
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

# ---------------------------------------------------------------------
# spawn_npcs remains unchanged.
async def spawn_npcs():
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
        logging.info("Spawned NPCs concurrently: %s", spawned_npc_ids)
        return jsonify({"message": "NPCs spawned", "npc_ids": spawned_npc_ids}), 200
    except Exception as e:
        logging.exception("Error in /spawn_npcs:")
        return jsonify({"error": str(e)}), 500
    finally:
        await conn.close()
