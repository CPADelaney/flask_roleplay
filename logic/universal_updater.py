# logic/universal_updater.py

import json
from db.connection import get_db_connection
from logic.social_links import (
    get_social_link, create_social_link,
    update_link_type_and_level, add_link_event
)

def apply_universal_updates(data: dict):
    """
    Applies the 'universal update' logic to the database using the JSON structure
    described in data. This handles NPC creations/updates, location creations,
    roleplay state changes, events, inventory, quests, social links, perk unlocks,
    and full/partial schedule changes for NPCs.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 1) roleplay_updates
        #    GPT can store environment name/desc, a "ChaseSchedule", or any other global data here.
        roleplay_updates = data.get("roleplay_updates", {})
        for key, value in roleplay_updates.items():
            # If "value" is a list or dict (e.g. schedule), store as JSON;
            # else store as a string
            if isinstance(value, (dict, list)):
                val_to_store = json.dumps(value)
            else:
                val_to_store = str(value)

            cursor.execute("""
                INSERT INTO CurrentRoleplay (key, value)
                VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value
            """, (key, val_to_store))

        # 2) npc_creations (brand-new NPCs)
        npc_creations = data.get("npc_creations", [])
        for npc_data in npc_creations:
            name = npc_data.get("npc_name", "Unnamed NPC")
            introduced = npc_data.get("introduced", False)

            # Archetypes is optional JSON
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

            cursor.execute("""
                INSERT INTO NPCStats (
                    npc_name, introduced,
                    archetypes,
                    dominance, cruelty, closeness, trust, respect, intensity,
                    hobbies, personality_traits, likes, dislikes,
                    affiliations, schedule, memory, monica_level
                )
                VALUES (
                    %s, %s,
                    %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
            """, (
                name, introduced,
                json.dumps(arche),
                dom, cru, clos, tru, resp, inten,
                json.dumps(hbs), json.dumps(pers),
                json.dumps(lks), json.dumps(dlks),
                json.dumps(affil), json.dumps(sched),
                json.dumps(mem),
                monica_lvl
            ))

        # 3) npc_updates (existing NPC changes, including schedule merges)
        npc_updates = data.get("npc_updates", [])
        for up in npc_updates:
            npc_id = up.get("npc_id")
            if not npc_id:
                continue

            # Basic columns to update directly
            fields_map = {
                "npc_name": "npc_name",
                "introduced": "introduced",
                "dominance": "dominance",
                "cruelty": "cruelty",
                "closeness": "closeness",
                "trust": "trust",
                "respect": "respect",
                "intensity": "intensity",
                "monica_level": "monica_level"
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
                set_vals.append(npc_id)
                query = f"UPDATE NPCStats SET {set_str} WHERE npc_id=%s"
                cursor.execute(query, tuple(set_vals))

            # Memory merges: append new entries
            if "memory" in up:
                new_mem_entries = up["memory"]
                if isinstance(new_mem_entries, str):
                    new_mem_entries = [new_mem_entries]
                cursor.execute("""
                    UPDATE NPCStats
                    SET memory = COALESCE(memory, '[]'::jsonb) || to_jsonb(%s)
                    WHERE npc_id = %s
                """, (new_mem_entries, npc_id))

            # Overwrite entire schedule if "schedule" is present
            if "schedule" in up:
                new_schedule = up["schedule"]
                cursor.execute("""
                    UPDATE NPCStats
                    SET schedule = %s
                    WHERE npc_id = %s
                """, (json.dumps(new_schedule), npc_id))

            # Partial merges if "schedule_updates" is present
            if "schedule_updates" in up:
                partial_sched = up["schedule_updates"]  # e.g. { "Wednesday": { "Afternoon": "Dentist" } }
                cursor.execute("SELECT schedule FROM NPCStats WHERE npc_id=%s", (npc_id,))
                row = cursor.fetchone()
                if row:
                    existing_schedule = row[0] or {}
                    for day_key, times_map in partial_sched.items():
                        if day_key not in existing_schedule:
                            existing_schedule[day_key] = {}
                        existing_schedule[day_key].update(times_map)

                    cursor.execute("""
                        UPDATE NPCStats
                        SET schedule = %s
                        WHERE npc_id=%s
                    """, (json.dumps(existing_schedule), npc_id))

        # 4) character_stat_updates (PlayerStats changes)
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
                    set_clauses.append(f"{col} = %s")
                    set_vals.append(stats[k])
            if set_clauses:
                set_str = ", ".join(set_clauses)
                set_vals.append(p_name)
                cursor.execute(
                    f"UPDATE PlayerStats SET {set_str} WHERE player_name=%s",
                    tuple(set_vals)
                )

        # 5) relationship_updates (NPC affiliations array)
        rel_updates = data.get("relationship_updates", [])
        for r in rel_updates:
            npc_id = r.get("npc_id")
            if not npc_id:
                continue
            aff_list = r.get("affiliations", None)
            if aff_list is not None:
                cursor.execute("""
                    UPDATE NPCStats
                    SET affiliations = %s
                    WHERE npc_id = %s
                """, (json.dumps(aff_list), npc_id))

        # 6) npc_introductions
        npc_intros = data.get("npc_introductions", [])
        for intro in npc_intros:
            nid = intro.get("npc_id")
            if nid:
                cursor.execute("""
                    UPDATE NPCStats
                    SET introduced = TRUE
                    WHERE npc_id=%s
                """, (nid,))

        # 7) location_creations
        location_creations = data.get("location_creations", [])
        for loc in location_creations:
            loc_name = loc.get("location_name", "Unnamed")
            desc = loc.get("description", "")
            open_hours = loc.get("open_hours", [])
            cursor.execute("""
                INSERT INTO Locations (name, description, open_hours)
                VALUES (%s, %s, %s)
            """, (loc_name, desc, json.dumps(open_hours)))

        # 8) event_list_updates
        event_updates = data.get("event_list_updates", [])
        for ev in event_updates:
            ev_name = ev.get("event_name", "UnnamedEvent")
            ev_desc = ev.get("description", "")
            ev_start = ev.get("start_time", "TBD Start")
            ev_end = ev.get("end_time", "TBD End")
            ev_loc = ev.get("location", "Unknown")
            cursor.execute("""
                INSERT INTO Events (event_name, description, start_time, end_time, location)
                VALUES (%s, %s, %s, %s, %s)
            """, (ev_name, ev_desc, ev_start, ev_end, ev_loc))

        # 9) inventory_updates
        inv_updates = data.get("inventory_updates", {})
        if inv_updates:
            p_n = inv_updates.get("player_name", "Chase")
            added = inv_updates.get("added_items", [])
            removed = inv_updates.get("removed_items", [])

            # Add items
            for item in added:
                if isinstance(item, dict):
                    item_name = item.get("item_name", "Unnamed")
                    item_desc = item.get("item_description", "")
                    item_fx   = item.get("item_effect", "")
                    category  = item.get("category", "")
                    cursor.execute("""
                        INSERT INTO PlayerInventory
                            (player_name, item_name, item_description, item_effect, category, quantity)
                        VALUES (%s, %s, %s, %s, %s, 1)
                        ON CONFLICT (player_name, item_name) DO UPDATE
                            SET quantity = PlayerInventory.quantity + 1
                    """, (p_n, item_name, item_desc, item_fx, category))
                elif isinstance(item, str):
                    cursor.execute("""
                        INSERT INTO PlayerInventory
                            (player_name, item_name, quantity)
                        VALUES (%s, %s, 1)
                        ON CONFLICT (player_name, item_name) DO UPDATE
                            SET quantity = PlayerInventory.quantity + 1
                    """, (p_n, item))

            # Remove items
            for item in removed:
                if isinstance(item, dict):
                    i_name = item.get("item_name")
                    if i_name:
                        cursor.execute("""
                            DELETE FROM PlayerInventory
                            WHERE player_name=%s AND item_name=%s
                        """, (p_n, i_name))
                elif isinstance(item, str):
                    cursor.execute("""
                        DELETE FROM PlayerInventory
                        WHERE player_name=%s AND item_name=%s
                    """, (p_n, item))

        # 10) quest_updates
        quest_updates = data.get("quest_updates", [])
        for qu in quest_updates:
            qid = qu.get("quest_id")
            status = qu.get("status", "In Progress")
            detail = qu.get("progress_detail", "")
            if qid:
                cursor.execute("""
                    UPDATE Quests
                    SET status=%s, progress_detail=%s
                    WHERE quest_id=%s
                """, (status, detail, qid))

        # 11) social_links (NPC<->NPC or player<->NPC relationships)
        rel_links = data.get("social_links", [])
        for link_data in rel_links:
            e1_type = link_data.get("entity1_type")
            e1_id   = link_data.get("entity1_id")
            e2_type = link_data.get("entity2_type")
            e2_id   = link_data.get("entity2_id")
            if not e1_type or not e1_id or not e2_type or not e2_id:
                continue

            existing = get_social_link(e1_type, e1_id, e2_type, e2_id)
            if not existing:
                new_link_id = create_social_link(
                    e1_type, e1_id, e2_type, e2_id,
                    link_type=link_data.get("link_type", "neutral"),
                    link_level=link_data.get("level_change", 0)
                )
                if "new_event" in link_data:
                    add_link_event(new_link_id, link_data["new_event"])
            else:
                ltype = link_data.get("link_type")
                lvl_change = link_data.get("level_change", 0)
                update_link_type_and_level(existing["link_id"], new_type=ltype, level_change=lvl_change)
                if "new_event" in link_data:
                    add_link_event(existing["link_id"], link_data["new_event"])

        # 12) perk_unlocks (one-time perks/skills)
        perk_unlocks = data.get("perk_unlocks", [])
        for perk in perk_unlocks:
            perk_name = perk.get("perk_name")
            perk_desc = perk.get("perk_description", "")
            perk_fx   = perk.get("perk_effect", "")

            # We'll default the player_name if not in the perk object
            player_name = data.get("player_name", "Chase")

            if perk_name:
                cursor.execute("""
                    INSERT INTO PlayerPerks (player_name, perk_name, perk_description, perk_effect)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (player_name, perk_name) DO NOTHING
                """, (player_name, perk_name, perk_desc, perk_fx))

        conn.commit()
        return {"message": "Universal update successful"}

    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        conn.close()
