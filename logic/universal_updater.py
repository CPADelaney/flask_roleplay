# logic/universal_updater.py

import json
from db.connection import get_db_connection
from logic.social_links import (
    get_social_link, create_social_link,
    update_link_type_and_level, add_link_event)

def apply_universal_updates(data: dict):
    """
    Applies the 'universal update' logic to the database using the JSON structure
    described in data. This replaces the duplicate logic found in:
      - routes/universal_update.py
      - new_game.py (next_storybeat, etc.)
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 1) roleplay_updates
        roleplay_updates = data.get("roleplay_updates", {})
        for key, value in roleplay_updates.items():
            cursor.execute("""
                INSERT INTO CurrentRoleplay (key, value)
                VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value
            """, (key, json.dumps(value) if isinstance(value, (dict, list)) else str(value)))

        # 2) npc_creations
        npc_creations = data.get("npc_creations", [])
        for npc_data in npc_creations:
            name = npc_data.get("npc_name", "Unnamed NPC")
            introduced = npc_data.get("introduced", False)
        
            # New field:
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
        
            # If memory is an array of strings, do something like:
            mem = npc_data.get("memory", [])
            # If memory can be a single string, convert it to an array 
            # or handle it differently. Example:
            if isinstance(mem, str):
                mem = [mem]
        
            monica_lvl = npc_data.get("monica_level", 0)
        
            cursor.execute("""
                INSERT INTO NPCStats (
                    npc_name, introduced,
                    archetypes,                -- ADDED
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
                json.dumps(arche),          # Archetypes as JSON
                dom, cru, clos, tru, resp, inten,
                json.dumps(hbs), json.dumps(pers),
                json.dumps(lks), json.dumps(dlks),
                json.dumps(affil), json.dumps(sched),
                json.dumps(mem),            # If memory is a JSON array
                monica_lvl
            ))

        npc_updates = data.get("npc_updates", [])
        for up in npc_updates:
            npc_id = up.get("npc_id")
            if not npc_id:
                continue
        
            # We'll handle memory separately if present
            # This map covers normal columns that we just set = %s
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
            }
        
            set_clauses = []
            set_vals = []
        
            # 1) Gather normal fields (NOT memory)
            for field_key, db_col in fields_map.items():
                if field_key in up:
                    set_clauses.append(f"{db_col} = %s")
                    set_vals.append(up[field_key])
        
            # 2) If we have normal fields, build the dynamic UPDATE
            if set_clauses:
                set_str = ", ".join(set_clauses)
                set_vals.append(npc_id)
                query = f"UPDATE NPCStats SET {set_str} WHERE npc_id=%s"
                cursor.execute(query, tuple(set_vals))
        
            # 3) Now handle "memory" if it exists in up (append to the existing JSON array)
            if "memory" in up:
                new_mem_entries = up["memory"]
        
                # If the user is sending a single string, wrap it in a list
                # If they're sending an array of strings, just use it as is.
                if isinstance(new_mem_entries, str):
                    new_mem_entries = [new_mem_entries]
        
                # new_mem_entries should now be a list of strings
                # We'll append them to existing memory with COALESCE(..., '[]') || to_jsonb(...)
                cursor.execute("""
                    UPDATE NPCStats
                    SET memory = COALESCE(memory, '[]'::jsonb) || to_jsonb(%s)
                    WHERE npc_id = %s
                """, (new_mem_entries, npc_id))
        
                # 4) character_stat_updates
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

        # 5) relationship_updates
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
        
            # Add these lines:
            ev_start = ev.get("start_time", "TBD Start")    # or "Day X, Morning"
            ev_end = ev.get("end_time", "TBD End")
            ev_loc = ev.get("location", "Unknown")
        
            # Now insert all columns
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

            # For added items
            for item in added:
                # Distinguish if it's a dict or a simple string
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
                    # Old fallback if item is just a string
                    cursor.execute("""
                        INSERT INTO PlayerInventory
                            (player_name, item_name, quantity)
                        VALUES (%s, %s, 1)
                        ON CONFLICT (player_name, item_name) DO UPDATE
                            SET quantity = PlayerInventory.quantity + 1
                    """, (p_n, item))

            # For removed items
            for item in removed:
                if isinstance(item, dict):
                    # If you want to allow removing by item dict,
                    # you'll need to decide how to handle item_name, etc.
                    i_name = item.get("item_name")
                    if i_name:
                        cursor.execute("""
                            DELETE FROM PlayerInventory
                            WHERE player_name = %s AND item_name = %s
                        """, (p_n, i_name))
                elif isinstance(item, str):
                    cursor.execute("""
                        DELETE FROM PlayerInventory
                        WHERE player_name = %s AND item_name = %s
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


        # 11) Social Links (NPC↔NPC or player↔NPC relationships)
        rel_links = data.get("social_links", [])
        for link_data in rel_links:
            e1_type = link_data.get("entity1_type")
            e1_id   = link_data.get("entity1_id")
            e2_type = link_data.get("entity2_type")
            e2_id   = link_data.get("entity2_id")
            if not e1_type or not e1_id or not e2_type or not e2_id:
                continue  # skip invalid

            # Check if a link already exists
            existing = get_social_link(e1_type, e1_id, e2_type, e2_id)
            if not existing:
                # Create
                new_link_id = create_social_link(
                    e1_type, e1_id, e2_type, e2_id,
                    link_type=link_data.get("link_type", "neutral"),
                    link_level=link_data.get("level_change", 0)  # or 0 if no level
                )
                # Optionally add an event
                if "new_event" in link_data:
                    add_link_event(new_link_id, link_data["new_event"])
            else:
                # Possibly update link_type/level
                ltype = link_data.get("link_type")
                lvl_change = link_data.get("level_change", 0)
                update_link_type_and_level(existing["link_id"], new_type=ltype, level_change=lvl_change)

                # If there's a new event text
                if "new_event" in link_data:
                    add_link_event(existing["link_id"], link_data["new_event"])

        conn.commit()
        return {"message": "Universal update successful"}
        
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        conn.close()
