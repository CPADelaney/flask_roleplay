import json
import logging
import asyncpg
from datetime import datetime, date  # Merged import from version 2

from logic.npc_creation import (
    create_npc_partial,
    insert_npc_stub_into_db,
    recalc_npc_stats_with_new_archetypes,
    spawn_multiple_npcs_enhanced,
    create_and_refine_npc
)
from logic.social_links import (
    get_social_link, create_social_link,
    update_link_type_and_level, add_link_event
)

# If needed (shared_memory handling & image gen):
from logic.memory_logic import get_shared_memory
from routes.ai_image_generator import generate_roleplay_image_from_gpt

#############################
# Helper: normalize curly quotes
#############################
def normalize_smart_quotes_inplace(obj):
    """Recursively replace curly quotes/apostrophes with straight ASCII quotes."""
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            obj[k] = normalize_smart_quotes_inplace(obj[k])
        return obj
    elif isinstance(obj, list):
        return [normalize_smart_quotes_inplace(x) for x in obj]
    elif isinstance(obj, str):
        return (
            obj.replace("’", "'")
               .replace("‘", "'")
               .replace("“", '"')
               .replace("”", '"')
        )
    else:
        return obj


async def apply_universal_updates_async(user_id, conversation_id, data, conn) -> dict:
    """
    Applies universal data updates from a GPT or user-provided payload. 
    Incorporates features from both code versions:
      - Normalizes curly quotes
      - Updates NPCs (creation, stats, partial schedules, memory)
      - Updates roleplay states, relationships, location/event/quest data
      - Adds activity, perk, inventory changes
      - Handles optional journaling, image generation, plus partial refinements
      - Supports new `shared_memory_updates` merges and partial schedule merges
      - Calls `refine_npc_final_data(...)` to finalize NPC changes.
    """
    try:
        # 1) Normalize curly quotes
        data = normalize_smart_quotes_inplace(data)
        logging.info("=== [apply_universal_updates_async] Incoming data (after normalization) ===")
        logging.info(json.dumps(data, indent=2))

        data_user_id = data.get("user_id", user_id)
        data_conv_id = data.get("conversation_id", conversation_id)
        if not data_user_id or not data_conv_id:
            logging.error("Missing user_id or conversation_id.")
            return {"error": "Missing user_id or conversation_id"}

        # Default day names / environment if not provided
        day_names = data.get("day_names", ["Commandday","Bindday","Chastiseday","Overlorday","Submissday","Whipday","Obeyday"])
        environment_desc = data.get("environment_desc", "A corporate gothic environment")

        # We'll track NPCs that need a final "refine" pass
        npcs_to_refine = set()

        # --------------------------------------------------------------------------------
        # 2) Process roleplay_updates (including possible MainQuest / PlayerRole merges)
        # --------------------------------------------------------------------------------
        rp_updates = data.get("roleplay_updates", {})

        # If older code used "MainQuest"/"PlayerRole" top-level keys, unify into rp_updates
        main_quest = data.get("MainQuest")
        player_role = data.get("PlayerRole")
        if main_quest:
            rp_updates["MainQuest"] = main_quest
        if player_role:
            rp_updates["PlayerRole"] = player_role

        logging.info(f"[apply_universal_updates_async] roleplay_updates: {rp_updates}")
        for key, val in rp_updates.items():
            stored_val = json.dumps(val) if isinstance(val, (dict, list)) else str(val)
            await conn.execute(
                """
                INSERT INTO CurrentRoleplay(user_id, conversation_id, key, value)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = EXCLUDED.value
                """,
                user_id, conversation_id, key, stored_val
            )
            logging.info(f"  Insert/Update CurrentRoleplay => key={key}, value={val}")

        # --------------------------------------------------------------------------------
        # 3) Process npc_creations
        # --------------------------------------------------------------------------------
        npc_creations = data.get("npc_creations", [])
        logging.info(f"[apply_universal_updates_async] npc_creations => count={len(npc_creations)}")
        for npc_data in npc_creations:
            name = npc_data.get("npc_name", "Unnamed NPC")

            # Skip if we already have an NPC with the same name
            row = await conn.fetchrow(
                """
                SELECT npc_id FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND LOWER(npc_name)=$3
                LIMIT 1
                """,
                user_id, conversation_id, name.lower()
            )
            if row:
                logging.info(f"Skipping creation. NPC '{name}' exists (npc_id={row['npc_id']}).")
                continue

            # Create partial NPC from provided data
            partial_npc = create_npc_partial(
                user_id=user_id,
                conversation_id=conversation_id,
                sex=npc_data.get("sex", "female"),
                environment_desc=npc_data.get("environment_desc", environment_desc)
            )

            # Merge override fields
            override_keys = [
                "npc_name", "introduced", "sex", "dominance", "cruelty", "closeness",
                "trust", "respect", "intensity", "archetypes", "archetype_summary",
                "archetype_extras_summary", "likes", "dislikes", "hobbies",
                "personality_traits", "age", "birthdate"
            ]
            for key in override_keys:
                if key in npc_data:
                    partial_npc[key] = npc_data[key]

            logging.info(f"Inserting new NPC '{partial_npc['npc_name']}'")
            new_npc_id = await insert_npc_stub_into_db(partial_npc, user_id, conversation_id)
            npcs_to_refine.add(new_npc_id)

        # --------------------------------------------------------------------------------
        # 4) Process npc_updates (stats, archetypes, memory, schedules, partial schedules)
        # --------------------------------------------------------------------------------
        npc_updates = data.get("npc_updates", [])
        logging.info(f"[apply_universal_updates_async] npc_updates: {npc_updates}")
        for up in npc_updates:
            npc_id = up.get("npc_id")
            if not npc_id:
                logging.warning("Skipping npc_update: missing npc_id.")
                continue

            # Basic field maps
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
                    set_clauses.append(f"{db_col} = ${len(set_vals) + 1}")
                    set_vals.append(up[field_key])

            # If user changes stats or name, we update them
            if set_clauses:
                set_str = ", ".join(set_clauses)
                set_vals += [npc_id, user_id, conversation_id]
                await conn.execute(
                    f"""
                    UPDATE NPCStats
                    SET {set_str}
                    WHERE npc_id=${len(set_vals)-2}
                      AND user_id=${len(set_vals)-1}
                      AND conversation_id=${len(set_vals)}
                    """,
                    *set_vals
                )

            # If "archetypes" changed, store it & recalc
            if "archetypes" in up:
                await conn.execute(
                    """
                    UPDATE NPCStats
                    SET archetypes=$1
                    WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
                    """,
                    json.dumps(up["archetypes"]), npc_id, user_id, conversation_id
                )
                await recalc_npc_stats_with_new_archetypes(user_id, conversation_id, npc_id)
                npcs_to_refine.add(npc_id)

            # Append new memory entries
            if "memory" in up:
                new_mem = up["memory"]
                if isinstance(new_mem, str):
                    new_mem = [new_mem]
                await conn.execute(
                    """
                    UPDATE NPCStats
                    SET memory = COALESCE(memory, '[]'::jsonb) || to_jsonb($1::json)
                    WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
                    """,
                    json.dumps(new_mem), npc_id, user_id, conversation_id
                )
                npcs_to_refine.add(npc_id)

            # Full schedule overwrite
            if "schedule" in up:
                await conn.execute(
                    """
                    UPDATE NPCStats
                    SET schedule=$1
                    WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
                    """,
                    json.dumps(up["schedule"]), npc_id, user_id, conversation_id
                )
                npcs_to_refine.add(npc_id)

            # Partial schedule update (merged in from Version 2)
            if "schedule_updates" in up:
                partial_sched = up["schedule_updates"]
                row = await conn.fetchrow(
                    """
                    SELECT schedule
                    FROM NPCStats
                    WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
                    """,
                    npc_id, user_id, conversation_id
                )
                if row:
                    existing_schedule = row["schedule"] or {}
                    # Merge partial updates
                    for day_key, times_map in partial_sched.items():
                        if day_key not in existing_schedule:
                            existing_schedule[day_key] = {}
                        existing_schedule[day_key].update(times_map)
                    await conn.execute(
                        """
                        UPDATE NPCStats
                        SET schedule=$1
                        WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
                        """,
                        json.dumps(existing_schedule), npc_id, user_id, conversation_id
                    )
                npcs_to_refine.add(npc_id)

        # --------------------------------------------------------------------------------
        # 5) Store "ChaseSchedule" in CurrentRoleplay if present
        # --------------------------------------------------------------------------------
        chase_sched = data.get("ChaseSchedule")
        if chase_sched:
            await conn.execute(
                """
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'ChaseSchedule', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value=EXCLUDED.value
                """,
                user_id, conversation_id, json.dumps(chase_sched)
            )

        # --------------------------------------------------------------------------------
        # 5.5) (New from Version 2) Process shared_memory_updates
        # --------------------------------------------------------------------------------
        shared_memory_updates = data.get("shared_memory_updates", [])
        for sm_update in shared_memory_updates:
            npc_id = sm_update.get("npc_id")
            relationship = sm_update.get("relationship")
            if not npc_id or not relationship:
                logging.warning("Skipping shared_memory_update: missing npc_id or relationship.")
                continue
            row = await conn.fetchrow(
                """
                SELECT npc_name FROM NPCStats
                WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
                """,
                npc_id, user_id, conversation_id
            )
            if not row:
                logging.warning(f"Shared memory update: NPC with id {npc_id} not found.")
                continue

            npc_name = row["npc_name"]
            # Example usage: retrieve some shared memory text from your logic...
            shared_memory_text = get_shared_memory(user_id, conversation_id, relationship, npc_name)
            await conn.execute(
                """
                UPDATE NPCStats
                SET memory = COALESCE(memory, '[]'::jsonb) || to_jsonb($1::text)
                WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
                """,
                shared_memory_text, npc_id, user_id, conversation_id
            )
            npcs_to_refine.add(npc_id)

        # --------------------------------------------------------------------------------
        # 6) Process character_stat_updates
        # --------------------------------------------------------------------------------
        char_update = data.get("character_stat_updates", {})
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
                    set_clauses.append(f"{col} = ${len(set_vals) + 1}")
                    set_vals.append(stats[k])
            if set_clauses:
                set_str = ", ".join(set_clauses)
                set_vals += [p_name, user_id, conversation_id]
                await conn.execute(
                    f"""
                    UPDATE PlayerStats
                    SET {set_str}
                    WHERE player_name=${len(set_vals)-2}
                      AND user_id=${len(set_vals)-1}
                      AND conversation_id=${len(set_vals)}
                    """,
                    *set_vals
                )

        # --------------------------------------------------------------------------------
        # 7) Process relationship_updates (append affiliations)
        # --------------------------------------------------------------------------------
        rel_updates = data.get("relationship_updates", [])
        for r in rel_updates:
            npc_id = r.get("npc_id")
            if not npc_id:
                continue
            aff_list = r.get("affiliations")
            if aff_list:
                await conn.execute(
                    """
                    UPDATE NPCStats
                    SET affiliations = COALESCE(affiliations, '[]'::jsonb) || to_jsonb($1::json)
                    WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
                    """,
                    json.dumps(aff_list), npc_id, user_id, conversation_id
                )
                npcs_to_refine.add(npc_id)

        # --------------------------------------------------------------------------------
        # 8) Process npc_introductions
        # --------------------------------------------------------------------------------
        npc_intros = data.get("npc_introductions", [])
        for intro in npc_intros:
            nid = intro.get("npc_id")
            if nid:
                await conn.execute(
                    """
                    UPDATE NPCStats
                    SET introduced=TRUE
                    WHERE npc_id=$1 AND user_id=$2 AND conversation_id=$3
                    """,
                    nid, user_id, conversation_id
                )

        # --------------------------------------------------------------------------------
        # 9) Process location_creations
        # --------------------------------------------------------------------------------
        loc_creations = data.get("location_creations", [])
        for loc in loc_creations:
            loc_name = loc.get("location_name", "Unnamed")
            desc = loc.get("description", "")
            open_hours = loc.get("open_hours", [])
            row = await conn.fetchrow(
                """
                SELECT id FROM Locations
                WHERE user_id=$1 AND conversation_id=$2 AND LOWER(location_name)=$3
                LIMIT 1
                """,
                user_id, conversation_id, loc_name.lower()
            )
            if not row:
                await conn.execute(
                    """
                    INSERT INTO Locations (user_id, conversation_id, location_name, description, open_hours)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    user_id, conversation_id, loc_name, desc, json.dumps(open_hours)
                )

        # --------------------------------------------------------------------------------
        # 10) Process event_list_updates (including optional fantasy_level check)
        # --------------------------------------------------------------------------------
        event_updates = data.get("event_list_updates", [])
        for ev in event_updates:
            if "npc_id" in ev and "day" in ev and "time_of_day" in ev:
                npc_id = ev["npc_id"]
                year = ev.get("year", 1)
                month = ev.get("month", 1)
                day = ev["day"]
                tod = ev["time_of_day"]
                ov_loc = ev.get("override_location", "Unknown")
                fantasy_level = ev.get("fantasy_level", "realistic")  # from Version 1

                row = await conn.fetchrow(
                    """
                    SELECT event_id FROM PlannedEvents
                    WHERE user_id=$1 AND conversation_id=$2
                      AND npc_id=$3 AND year=$4 AND month=$5 AND day=$6 AND time_of_day=$7
                    LIMIT 1
                    """,
                    user_id, conversation_id, npc_id, year, month, day, tod
                )
                if not row:
                    await conn.execute(
                        """
                        INSERT INTO PlannedEvents (
                            user_id, conversation_id, npc_id, year, month, day, 
                            time_of_day, override_location
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                        user_id, conversation_id, npc_id, year, month, day, tod, ov_loc
                    )

                # (Version 1) If surreal, log a "fantasy" moment in PlayerJournal
                if fantasy_level == "surreal":
                    await conn.execute(
                        """
                        INSERT INTO PlayerJournal (
                            user_id, conversation_id, entry_type, entry_text, 
                            fantasy_flag, intensity_level
                        )
                        VALUES ($1, $2, 'moment', $3, TRUE, 4)
                        """,
                        user_id, conversation_id, f"Surreal event: {ev.get('description', 'Unnamed event')}"
                    )

        # --------------------------------------------------------------------------------
        # 11) Process inventory_updates
        # --------------------------------------------------------------------------------
        inv_updates = data.get("inventory_updates", {})
        if inv_updates:
            p_name = inv_updates.get("player_name", "Chase")
            added = inv_updates.get("added_items", [])
            removed = inv_updates.get("removed_items", [])

            for item in added:
                if isinstance(item, dict):
                    item_name = item.get("item_name", "Unnamed")
                    item_desc = item.get("item_description", "")
                    item_fx = item.get("item_effect", "")
                    category = item.get("category", "")
                    await conn.execute(
                        """
                        INSERT INTO PlayerInventory (
                            user_id, conversation_id, player_name, item_name, 
                            item_description, item_effect, category, quantity
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, 1)
                        ON CONFLICT (user_id, conversation_id, player_name, item_name)
                        DO UPDATE SET quantity = PlayerInventory.quantity + 1
                        """,
                        user_id, conversation_id, p_name, item_name, item_desc, item_fx, category
                    )
                elif isinstance(item, str):
                    await conn.execute(
                        """
                        INSERT INTO PlayerInventory (
                            user_id, conversation_id, player_name, item_name, quantity
                        )
                        VALUES ($1, $2, $3, $4, 1)
                        ON CONFLICT (user_id, conversation_id, player_name, item_name)
                        DO UPDATE SET quantity = PlayerInventory.quantity + 1
                        """,
                        user_id, conversation_id, p_name, item
                    )

            for item in removed:
                i_name = item.get("item_name") if isinstance(item, dict) else item
                if i_name:
                    await conn.execute(
                        """
                        DELETE FROM PlayerInventory
                        WHERE user_id=$1 AND conversation_id=$2
                          AND player_name=$3 AND item_name=$4
                        """,
                        user_id, conversation_id, p_name, i_name
                    )

        # --------------------------------------------------------------------------------
        # 12) Process quest_updates
        # --------------------------------------------------------------------------------
        quest_updates = data.get("quest_updates", [])
        for qu in quest_updates:
            qid = qu.get("quest_id")
            status = qu.get("status", "In Progress")
            detail = qu.get("progress_detail", "")
            qgiver = qu.get("quest_giver", "")
            reward = qu.get("reward", "")
            qname = qu.get("quest_name", None)
            if qid:
                await conn.execute(
                    """
                    UPDATE Quests
                    SET status=$1, progress_detail=$2, quest_giver=$3, reward=$4
                    WHERE quest_id=$5 AND user_id=$6 AND conversation_id=$7
                    """,
                    status, detail, qgiver, reward, qid, user_id, conversation_id
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO Quests (
                        user_id, conversation_id, quest_name, status, progress_detail,
                        quest_giver, reward
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    user_id, conversation_id, qname or "Unnamed Quest",
                    status, detail, qgiver, reward
                )

        # --------------------------------------------------------------------------------
        # 13) Process social_links
        # --------------------------------------------------------------------------------
        rel_links = data.get("social_links", [])
        for link_data in rel_links:
            e1_type = link_data.get("entity1_type")
            e1_id = link_data.get("entity1_id")
            e2_type = link_data.get("entity2_type")
            e2_id = link_data.get("entity2_id")
            if not (e1_type and e1_id and e2_type and e2_id):
                continue

            existing_link = get_social_link(e1_type, e1_id, e2_type, e2_id, user_id, conversation_id)
            if not existing_link:
                link_type = link_data.get("link_type", "neutral")
                lvl_change = link_data.get("level_change", 0)
                new_link_id = create_social_link(
                    e1_type, e1_id, e2_type, e2_id,
                    link_type=link_type, link_level=lvl_change,
                    user_id=user_id, conversation_id=conversation_id
                )
                if "group_context" in link_data:
                    await conn.execute(
                        """
                        UPDATE SocialLinks
                        SET group_context=$1
                        WHERE link_id=$2
                        """,
                        link_data["group_context"], new_link_id
                    )
                if "new_event" in link_data:
                    add_link_event(new_link_id, link_data["new_event"])
            else:
                ltype = link_data.get("link_type")
                lvl_change = link_data.get("level_change", 0)
                update_link_type_and_level(
                    existing_link["link_id"],
                    new_type=ltype, level_change=lvl_change,
                    user_id=user_id, conversation_id=conversation_id
                )
                if "group_context" in link_data:
                    await conn.execute(
                        """
                        UPDATE SocialLinks
                        SET group_context=$1
                        WHERE link_id=$2
                        """,
                        link_data["group_context"], existing_link["link_id"]
                    )
                if "new_event" in link_data:
                    add_link_event(existing_link["link_id"], link_data["new_event"])

        # --------------------------------------------------------------------------------
        # 14) Process perk_unlocks
        # --------------------------------------------------------------------------------
        perk_unlocks = data.get("perk_unlocks", [])
        for perk in perk_unlocks:
            perk_name = perk.get("perk_name")
            perk_desc = perk.get("perk_description", "")
            perk_fx = perk.get("perk_effect", "")
            player_name = perk.get("player_name", "Chase")
            if perk_name:
                await conn.execute(
                    """
                    INSERT INTO PlayerPerks (
                        user_id, conversation_id, player_name, perk_name,
                        perk_description, perk_effect
                    )
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT DO NOTHING
                    """,
                    user_id, conversation_id, player_name, perk_name, perk_desc, perk_fx
                )

        # --------------------------------------------------------------------------------
        # 15) Process activity_updates (Version 1)
        # --------------------------------------------------------------------------------
        activity_updates = data.get("activity_updates", [])
        for act in activity_updates:
            act_name = act.get("activity_name")
            if not act_name:
                continue
            purpose = json.dumps(act.get("purpose", {"description": "Unnamed activity", "fantasy_level": "realistic"}))
            stat_integration = json.dumps(act.get("stat_integration", {}))
            intensity_tier = act.get("intensity_tier", 0)
            setting_variant = act.get("setting_variant", "")
            row = await conn.fetchrow(
                """
                SELECT id FROM Activities
                WHERE name=$1
                LIMIT 1
                """,
                act_name
            )
            if not row:
                await conn.execute(
                    """
                    INSERT INTO Activities (
                        name, purpose, stat_integration, intensity_tiers, setting_variants
                    )
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    act_name, purpose, stat_integration,
                    json.dumps({"tier": intensity_tier}),
                    json.dumps([setting_variant])
                )

        # --------------------------------------------------------------------------------
        # 16) Process journal_updates (Version 1)
        # --------------------------------------------------------------------------------
        journal_updates = data.get("journal_updates", [])
        for ju in journal_updates:
            entry_type = ju.get("entry_type")
            entry_text = ju.get("entry_text")
            fantasy_flag = ju.get("fantasy_flag", False)
            intensity_level = ju.get("intensity_level", 0)
            if entry_type and entry_text:
                await conn.execute(
                    """
                    INSERT INTO PlayerJournal (
                        user_id, conversation_id, entry_type, entry_text,
                        fantasy_flag, intensity_level
                    )
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    user_id, conversation_id, entry_type, entry_text, fantasy_flag, intensity_level
                )

        # --------------------------------------------------------------------------------
        # 16.5) Process image_generation (Version 1)
        # --------------------------------------------------------------------------------
        img_gen = data.get("image_generation", {})
        if img_gen.get("generate", False):
            logging.info(f"Triggering image generation: {img_gen}")
            # Example "simulate GPT response"
            gpt_response = {"scene_data": {"narrative": data.get("narrative", "")}}
            image_result = await generate_roleplay_image_from_gpt(gpt_response, user_id, conversation_id)
            if "image_urls" in image_result and image_result["image_urls"]:
                # Store last image in NPCVisualAttributes for each updated/created NPC
                for npc in data.get("npc_updates", []) + data.get("npc_creations", []):
                    npc_id = npc.get("npc_id")
                    if npc_id:
                        await conn.execute(
                            """
                            UPDATE NPCVisualAttributes
                            SET last_generated_image=$1, updated_at=CURRENT_TIMESTAMP
                            WHERE npc_id=$2 AND user_id=$3 AND conversation_id=$4
                            """,
                            image_result["image_urls"][0], npc_id, user_id, conversation_id
                        )

        # --------------------------------------------------------------------------------
        # 17) Refine NPCs if needed (merged logic from Version 2)
        # --------------------------------------------------------------------------------
        for nid in npcs_to_refine:
            logging.info(f"[apply_universal_updates_async] refine_npc_final_data on NPC {nid}")
            # Function presumably defined elsewhere:
            await refine_npc_final_data(user_id, conversation_id, nid, day_names, environment_desc)

        logging.info("=== [apply_universal_updates_async] Success ===")
        return {"message": "Universal update successful"}

    except Exception as e:
        logging.exception("[apply_universal_updates_async] Error")
        return {"error": str(e)}


# Keep the alias
apply_universal_updates = apply_universal_updates_async
