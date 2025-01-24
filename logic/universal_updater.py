# logic/universal_updater.py

import json
from db.connection import get_db_connection

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
            dom = npc_data.get("dominance", 0)
            cru = npc_data.get("cruelty", 0)
            clos = npc_data.get("closeness", 0)
            tru = npc_data.get("trust", 0)
            resp = npc_data.get("respect", 0)
            inten = npc_data.get("intensity", 0)
            occ = npc_data.get("occupation", "")
            hbs = npc_data.get("hobbies", [])
            pers = npc_data.get("personality_traits", [])
            lks = npc_data.get("likes", [])
            dlks = npc_data.get("dislikes", [])
            affil = npc_data.get("affiliations", [])
            sched = npc_data.get("schedule", {})
            mem = npc_data.get("memory", "")
            monica_lvl = npc_data.get("monica_level", 0)
            introduced = npc_data.get("introduced", False)

            cursor.execute("""
                INSERT INTO NPCStats (
                    npc_name, introduced,
                    dominance, cruelty, closeness, trust, respect, intensity,
                    occupation, hobbies, personality_traits, likes, dislikes,
                    affiliations, schedule, memory, monica_level
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s)
            """, (
                name, introduced,
                dom, cru, clos, tru, resp, inten,
                occ, json.dumps(hbs), json.dumps(pers),
                json.dumps(lks), json.dumps(dlks),
                json.dumps(affil), json.dumps(sched), mem, monica_lvl
            ))

        # 3) npc_updates
        npc_updates = data.get("npc_updates", [])
        for up in npc_updates:
            npc_id = up.get("npc_id")
            if not npc_id:
                continue

            # Build a dynamic set of fields for the UPDATE
            fields_map = {
                "dominance": "dominance",
                "cruelty": "cruelty",
                "closeness": "closeness",
                "trust": "trust",
                "respect": "respect",
                "intensity": "intensity",
                "memory": "memory",
                "monica_level": "monica_level"
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
            cursor.execute("""
                INSERT INTO Events (event_name, description)
                VALUES (%s, %s)
            """, (ev_name, ev_desc))

        # 9) inventory_updates
        inv_updates = data.get("inventory_updates", {})
        if inv_updates:
            p_n = inv_updates.get("player_name", "Chase")
            added = inv_updates.get("added_items", [])
            removed = inv_updates.get("removed_items", [])
            for item in added:
                cursor.execute("""
                    INSERT INTO PlayerInventory (player_name, item_name)
                    VALUES (%s, %s)
                    ON CONFLICT (player_name, item_name) DO UPDATE
                        SET quantity = PlayerInventory.quantity + 1
                """, (p_n, item))
            for item in removed:
                # Remove or decrement quantity
                # This example just deletes the row
                cursor.execute("""
                    DELETE FROM PlayerInventory
                    WHERE player_name=%s AND item_name=%s
                    LIMIT 1
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

        conn.commit()
        return {"message": "Universal update successful"}
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        conn.close()
