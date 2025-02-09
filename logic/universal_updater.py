# logic/universal_updater.py

import json
import logging  # <-- Make sure you have import logging
from db.connection import get_db_connection
import asyncpg
from logic.social_links import (
    get_social_link, create_social_link,
    update_link_type_and_level, add_link_event
)

async def apply_universal_updates_async(data: dict) -> dict:
    """
    Asynchronously processes the universal_update payload, inserting or updating DB records.
    This version uses asyncpg's API (without a synchronous cursor).
    """
    logging.info("=== [apply_universal_updates_async] Incoming data ===")
    logging.info(json.dumps(data, indent=2))
    
    conn = await asyncpg.connect(dsn=DB_DSN)
    try:
        user_id = data.get("user_id")
        conv_id = data.get("conversation_id")
        if not user_id or not conv_id:
            logging.error("Missing user_id or conversation_id in universal_update data.")
            return {"error": "Missing user_id or conversation_id in universal_update"}

        # 1) Process roleplay_updates
        rp_updates = data.get("roleplay_updates", {})
        logging.info(f"[apply_universal_updates_async] roleplay_updates: {rp_updates}")
        for key, val in rp_updates.items():
            stored_val = json.dumps(val) if isinstance(val, (dict, list)) else str(val)
            await conn.execute("""
                INSERT INTO CurrentRoleplay(user_id, conversation_id, key, value)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, conversation_id, key) DO UPDATE SET value = EXCLUDED.value
            """, user_id, conv_id, key, stored_val)
            logging.info(f"  Insert/Update CurrentRoleplay => key={key}, value={val}")

        # 2) Process npc_creations (example for one block)
        npc_creations = data.get("npc_creations", [])
        logging.info(f"[apply_universal_updates_async] npc_creations: {npc_creations}")
        for npc_data in npc_creations:
            name = npc_data.get("npc_name", "Unnamed NPC")
            introduced = npc_data.get("introduced", False)
            # ... (extract other fields as in your original code) ...
            arche_json = json.dumps(npc_data.get("archetypes", []))  # example
            # Check if NPC already exists:
            row = await conn.fetchrow("""
                SELECT npc_id FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND LOWER(npc_name)=$3
                LIMIT 1
            """, user_id, conv_id, name.lower())
            if row:
                logging.info(f"  Skipping NPC creation, '{name}' already exists.")
                continue
            logging.info(f"  Creating NPC: {name}, introduced={introduced}")
            await conn.execute("""
                INSERT INTO NPCStats (
                    user_id, conversation_id,
                    npc_name, introduced, sex,
                    dominance, cruelty, closeness, trust, respect, intensity,
                    archetypes, archetype_summary, archetype_extras_summary,
                    physical_description, memory, monica_level,
                    age, birthdate
                )
                VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8, $9, $10, $11,
                    $12, $13, $14, $15,
                    '[]'::jsonb, 0,
                    $16, $17
                )
            """, 
            # Pass the appropriate values extracted from npc_data
            user_id, conv_id, name, introduced, npc_data.get("sex", "female").lower(),
            npc_data.get("dominance", 0), npc_data.get("cruelty", 0), npc_data.get("closeness", 0),
            npc_data.get("trust", 0), npc_data.get("respect", 0), npc_data.get("intensity", 0),
            arche_json, npc_data.get("archetype_summary", ""), npc_data.get("archetype_extras_summary", ""),
            npc_data.get("physical_description", ""),
            npc_data.get("age", None), npc_data.get("birthdate", None)  # You will update these later
            )

        # ---------------------------------------------------------------------
        # 3) npc_updates
        # ---------------------------------------------------------------------
        npc_updates = data.get("npc_updates", [])
        logging.info(f"[apply_universal_updates] npc_updates: {npc_updates}")
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
                    set_clauses.append(f"{db_col} = %s")
                    set_vals.append(up[field_key])

            if set_clauses:
                set_str = ", ".join(set_clauses)
                set_vals += [npc_id, user_id, conv_id]
                query = f"""
                    UPDATE NPCStats
                    SET {set_str}
                    WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
                """
                logging.info(f"  Updating NPC {npc_id}: {set_clauses}")
                cursor.execute(query, tuple(set_vals))
                logging.info(f"  NPC {npc_id} rowcount = {cursor.rowcount}")

            # Memory merges
            if "memory" in up:
                new_mem_entries = up["memory"]
                if isinstance(new_mem_entries, str):
                    new_mem_entries = [new_mem_entries]
                logging.info(f"  Appending memory to NPC {npc_id}: {new_mem_entries}")
                cursor.execute("""
                    UPDATE NPCStats
                    SET memory = COALESCE(memory, '[]'::jsonb) || to_jsonb(%s)
                    WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
                """, (new_mem_entries, npc_id, user_id, conv_id))

            # Overwrite entire schedule if "schedule" is present
            if "schedule" in up:
                new_schedule = up["schedule"]
                logging.info(f"  Overwriting schedule for NPC {npc_id}: {new_schedule}")
                cursor.execute("""
                    UPDATE NPCStats
                    SET schedule=%s
                    WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
                """, (json.dumps(new_schedule), npc_id, user_id, conv_id))

            # Partial merges if "schedule_updates" is present
            if "schedule_updates" in up:
                partial_sched = up["schedule_updates"]
                logging.info(f"  Merging schedule_updates for NPC {npc_id}: {partial_sched}")
                cursor.execute("""
                    SELECT schedule
                    FROM NPCStats
                    WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
                """, (npc_id, user_id, conv_id))
                row = cursor.fetchone()
                if row:
                    existing_schedule = row[0] or {}
                    for day_key, times_map in partial_sched.items():
                        if day_key not in existing_schedule:
                            existing_schedule[day_key] = {}
                        existing_schedule[day_key].update(times_map)
                    cursor.execute("""
                        UPDATE NPCStats
                        SET schedule=%s
                        WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
                    """, (json.dumps(existing_schedule), npc_id, user_id, conv_id))

        # ---------------------------------------------------------------------
        # 4) character_stat_updates
        # ---------------------------------------------------------------------
        char_update = data.get("character_stat_updates", {})
        logging.info(f"[apply_universal_updates] character_stat_updates: {char_update}")
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
                    set_clauses.append(f"{col}=%s")
                    set_vals.append(stats[k])
            if set_clauses:
                set_str = ", ".join(set_clauses)
                set_vals += [p_name, user_id, conv_id]
                logging.info(f"  Updating player stats for {p_name}: {stats}")
                cursor.execute(
                    f"""
                    UPDATE PlayerStats
                    SET {set_str}
                    WHERE player_name=%s AND user_id=%s AND conversation_id=%s
                    """,
                    tuple(set_vals)
                )

        # ---------------------------------------------------------------------
        # 5) relationship_updates
        # ---------------------------------------------------------------------
        rel_updates = data.get("relationship_updates", [])
        logging.info(f"[apply_universal_updates] relationship_updates: {rel_updates}")
        for r in rel_updates:
            npc_id = r.get("npc_id")
            if not npc_id:
                logging.warning("Skipping relationship update: no npc_id.")
                continue
            aff_list = r.get("affiliations", None)
            if aff_list is not None:
                logging.info(f"  Updating affiliations for NPC {npc_id}: {aff_list}")
                cursor.execute("""
                    UPDATE NPCStats
                    SET affiliations=%s
                    WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
                """, (json.dumps(aff_list), npc_id, user_id, conv_id))

        # ---------------------------------------------------------------------
        # 5.5) Shared Memory Updates for Pre-existing Relationships
        # ---------------------------------------------------------------------
        shared_memory_updates = data.get("shared_memory_updates", [])
        logging.info(f"[apply_universal_updates] shared_memory_updates: {shared_memory_updates}")
        for sm_update in shared_memory_updates:
            npc_id = sm_update.get("npc_id")
            relationship = sm_update.get("relationship")
            if not npc_id or not relationship:
                logging.warning("Skipping shared memory update: missing npc_id or relationship data.")
                continue

            # Retrieve the NPC's name so the shared memory can reference it
            cursor.execute("""
                SELECT npc_name FROM NPCStats
                WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
            """, (npc_id, user_id, conv_id))
            row = cursor.fetchone()
            if not row:
                logging.warning(f"Shared memory update: NPC with id {npc_id} not found.")
                continue
            npc_name = row[0]

            # Import and call get_shared_memory to generate the memory text.
            from logic.memory import get_shared_memory
            shared_memory_text = get_shared_memory(user_id, conversation_id, relationship, npc_name)
            logging.info(f"Generated shared memory for NPC {npc_id}: {shared_memory_text}")

            cursor.execute("""
                UPDATE NPCStats
                SET memory = COALESCE(memory, '[]'::jsonb) || to_jsonb(%s::text)
                WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
            """, (shared_memory_text, npc_id, user_id, conv_id))
            logging.info(f"Appended shared memory to NPC {npc_id}")

        # ---------------------------------------------------------------------
        # 6) npc_introductions
        # ---------------------------------------------------------------------
        npc_intros = data.get("npc_introductions", [])
        logging.info(f"[apply_universal_updates] npc_introductions: {npc_intros}")
        for intro in npc_intros:
            nid = intro.get("npc_id")
            if nid:
                logging.info(f"  Marking NPC {nid} as introduced")
                cursor.execute("""
                    UPDATE NPCStats
                    SET introduced=TRUE
                    WHERE npc_id=%s AND user_id=%s AND conversation_id=%s
                """, (nid, user_id, conv_id))

        # ---------------------------------------------------------------------
        # 7) location_creations
        # ---------------------------------------------------------------------
        loc_creations = data.get("location_creations", [])
        logging.info(f"[apply_universal_updates] location_creations: {loc_creations}")
        for loc in loc_creations:
            loc_name = loc.get("location_name", "Unnamed")
            desc = loc.get("description", "")
            open_hours = loc.get("open_hours", [])

            cursor.execute("""
                SELECT id
                FROM Locations
                WHERE user_id=%s AND conversation_id=%s
                  AND LOWER(location_name)=%s
                LIMIT 1
            """, (user_id, conv_id, loc_name.lower()))
            existing_location = cursor.fetchone()
            if existing_location:
                logging.info(f"  Skipping location creation, '{loc_name}' already exists.")
                continue

            logging.info(f"  Inserting location => location_name={loc_name}, description={desc}, open_hours={open_hours}")
            cursor.execute("""
                INSERT INTO Locations (user_id, conversation_id, location_name, description, open_hours)
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, conv_id, loc_name, desc, json.dumps(open_hours)))
            logging.info(f"  Inserted location: {loc_name}. cursor.rowcount={cursor.rowcount}")

        # ---------------------------------------------------------------------
        # 8) event_list_updates => normal Events or PlannedEvents
        # ---------------------------------------------------------------------
        event_updates = data.get("event_list_updates", [])
        logging.info(f"[apply_universal_updates_async] event_list_updates: {event_updates}")
        for ev in event_updates:
            if "npc_id" in ev and "day" in ev and "time_of_day" in ev:
                npc_id = ev["npc_id"]
                year = ev.get("year", 1)
                month = ev.get("month", 1)
                day = ev["day"]
                tod = ev["time_of_day"]
                ov_loc = ev.get("override_location", "Unknown")
                row = await conn.fetchrow("""
                    SELECT event_id FROM PlannedEvents
                    WHERE user_id=$1 AND conversation_id=$2
                      AND npc_id=$3 AND year=$4 AND month=$5 AND day=$6 AND time_of_day=$7
                    LIMIT 1
                """, user_id, conv_id, npc_id, year, month, day, tod)
                if row:
                    logging.info(f"  Skipping planned event creation; already exists.")
                    continue
                logging.info(f"  Inserting PlannedEvent for npc_id={npc_id}")
                await conn.execute("""
                    INSERT INTO PlannedEvents (user_id, conversation_id, npc_id, year, month, day, time_of_day, override_location)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, user_id, conv_id, npc_id, year, month, day, tod, ov_loc)
            else:
                # Process global events similarly using await conn.fetchrow/execute
                pass

        # ---------------------------------------------------------------------
        # 9) inventory_updates
        # ---------------------------------------------------------------------
        inv_updates = data.get("inventory_updates", {})
        logging.info(f"[apply_universal_updates] inventory_updates: {inv_updates}")
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
                    logging.info(f"  Adding item for {p_n}: {item_name}")
                    cursor.execute("""
                        INSERT INTO PlayerInventory (
                            user_id, conversation_id,
                            player_name, item_name, item_description, item_effect, category, quantity
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, 1)
                        ON CONFLICT (user_id, conversation_id, player_name, item_name)
                        DO UPDATE SET quantity = PlayerInventory.quantity + 1
                    """, (user_id, conv_id, p_n, item_name, item_desc, item_fx, category))
                elif isinstance(item, str):
                    logging.info(f"  Adding item (string) for {p_n}: {item}")
                    cursor.execute("""
                        INSERT INTO PlayerInventory (
                            user_id, conversation_id,
                            player_name, item_name, quantity
                        )
                        VALUES (%s, %s, %s, %s, 1)
                        ON CONFLICT (user_id, conversation_id, player_name, item_name)
                        DO UPDATE SET quantity = PlayerInventory.quantity + 1
                    """, (user_id, conv_id, p_n, item))

            for item in removed:
                if isinstance(item, dict):
                    i_name = item.get("item_name")
                    if i_name:
                        logging.info(f"  Removing item for {p_n}: {i_name}")
                        cursor.execute("""
                            DELETE FROM PlayerInventory
                            WHERE user_id=%s AND conversation_id=%s
                              AND player_name=%s AND item_name=%s
                        """, (user_id, conv_id, p_n, i_name))
                elif isinstance(item, str):
                    logging.info(f"  Removing item for {p_n}: {item}")
                    cursor.execute("""
                        DELETE FROM PlayerInventory
                        WHERE user_id=%s AND conversation_id=%s
                          AND player_name=%s AND item_name=%s
                    """, (user_id, conv_id, p_n, item))

        # ---------------------------------------------------------------------
        # 10) quest_updates
        # ---------------------------------------------------------------------
        quest_updates = data.get("quest_updates", [])
        logging.info(f"[apply_universal_updates] quest_updates: {quest_updates}")
        for qu in quest_updates:
            qid = qu.get("quest_id")
            status = qu.get("status", "In Progress")
            detail = qu.get("progress_detail", "")
            qgiver = qu.get("quest_giver", "")
            reward = qu.get("reward", "")
            qname  = qu.get("quest_name", None)

            if qid:
                logging.info(f"  Updating Quest {qid}: status={status}, detail={detail}")
                cursor.execute("""
                    UPDATE Quests
                    SET status=%s, progress_detail=%s, quest_giver=%s, reward=%s
                    WHERE quest_id=%s AND user_id=%s AND conversation_id=%s
                """, (status, detail, qgiver, reward, qid, user_id, conv_id))
                if cursor.rowcount == 0:
                    logging.info(f"  No existing quest with ID {qid}, inserting new.")
                    cursor.execute("""
                        INSERT INTO Quests (
                            user_id, conversation_id, quest_id,
                            quest_name, status, progress_detail,
                            quest_giver, reward
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (user_id, conv_id, qid, qname, status, detail, qgiver, reward))
            else:
                logging.info(f"  Inserting new quest: {qname}, status={status}")
                cursor.execute("""
                    INSERT INTO Quests (
                        user_id, conversation_id,
                        quest_name, status, progress_detail,
                        quest_giver, reward
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (user_id, conv_id, qname or "Unnamed Quest", status, detail, qgiver, reward))

        # ---------------------------------------------------------------------
        # 11) social_links
        # ---------------------------------------------------------------------
        rel_links = data.get("social_links", [])
        logging.info(f"[apply_universal_updates] social_links: {rel_links}")
        for link_data in rel_links:
            e1_type = link_data.get("entity1_type")
            e1_id   = link_data.get("entity1_id")
            e2_type = link_data.get("entity2_type")
            e2_id   = link_data.get("entity2_id")
            if not e1_type or not e1_id or not e2_type or not e2_id:
                logging.warning(f"Skipping social link creation, missing entity info: {link_data}")
                continue

            existing_link = get_social_link(e1_type, e1_id, e2_type, e2_id, user_id, conv_id)
            if not existing_link:
                link_type = link_data.get("link_type", "neutral")
                lvl_change = link_data.get("level_change", 0)
                logging.info(f"  Creating new social link: {e1_type}({e1_id}) <-> {e2_type}({e2_id}), type={link_type}")
                new_link_id = create_social_link(
                    e1_type, e1_id,
                    e2_type, e2_id,
                    link_type=link_type,
                    link_level=lvl_change,
                    user_id=user_id,
                    conversation_id=conv_id
                )
                if "new_event" in link_data:
                    add_link_event(new_link_id, link_data["new_event"])
            else:
                ltype = link_data.get("link_type")
                lvl_change = link_data.get("level_change", 0)
                logging.info(f"  Updating social link {existing_link['link_id']}: new_type={ltype}, level_change={lvl_change}")
                update_link_type_and_level(
                    existing_link["link_id"],
                    new_type=ltype,
                    level_change=lvl_change,
                    user_id=user_id,
                    conversation_id=conv_id
                )
                if "new_event" in link_data:
                    add_link_event(existing_link["link_id"], link_data["new_event"])

        # ---------------------------------------------------------------------
        # 12) perk_unlocks
        # ---------------------------------------------------------------------
        perk_unlocks = data.get("perk_unlocks", [])
        logging.info(f"[apply_universal_updates] perk_unlocks: {perk_unlocks}")
        for perk in perk_unlocks:
            perk_name = perk.get("perk_name")
            perk_desc = perk.get("perk_description", "")
            perk_fx   = perk.get("perk_effect", "")
            player_name = perk.get("player_name", "Chase")
            if perk_name:
                logging.info(f"  Inserting perk {perk_name} for player {player_name}")
                cursor.execute("""
                    INSERT INTO PlayerPerks (
                        user_id, conversation_id,
                        player_name, perk_name, perk_description, perk_effect
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (
                    user_id, conv_id,
                    player_name, perk_name, perk_desc, perk_fx
                ))

        conn.commit()
        logging.info("=== [apply_universal_updates] Success! ===")
        return {"message": "Universal update successful"}

    except Exception as e:
        logging.exception("[apply_universal_updates] Error encountered")
        conn.rollback()
        return {"error": str(e)}
    finally:
        cursor.close()
        conn.close()
apply_universal_updates = apply_universal_updates_async
