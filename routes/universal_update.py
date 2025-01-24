# routes/universal_update.py

from flask import Blueprint, request, jsonify
from db.connection import get_db_connection
import json

universal_bp = Blueprint("universal_bp", __name__)

@universal_bp.route("/universal_update", methods=["POST"])
def universal_update():
    """
    Endpoint to handle universal updates in a single JSON payload.
    This can update:
      - Setting and day/time
      - NPC creation
      - NPC stat updates
      - NPC memory, monica_level
      - NPC introduction
      - NPC relationships/affiliations
      - location creation
      - event list updates
      - inventory updates
      - quest updates
      - etc.
    """
    try:
        data = request.get_json() or {}

        conn = get_db_connection()
        cursor = conn.cursor()

        # 1) Handle roleplay_updates
        roleplay_updates = data.get("roleplay_updates", {})
        # e.g. time_of_day, CurrentSetting, UsedSettings, etc.
        for key, value in roleplay_updates.items():
            # Upsert each into CurrentRoleplay
            cursor.execute("""
                INSERT INTO CurrentRoleplay (key, value)
                VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE 
                  SET value = EXCLUDED.value
            """, (key, json.dumps(value) if isinstance(value, dict) else str(value)))

        # 2) NPC creation
        npc_creations = data.get("npc_creations", [])
        for npc_data in npc_creations:
            # minimal required fields:
            npc_name = npc_data.get("npc_name", "Unnamed NPC")
            dominance = npc_data.get("dominance", 0)
            cruelty = npc_data.get("cruelty", 0)
            closeness = npc_data.get("closeness", 0)
            trust = npc_data.get("trust", 0)
            respect = npc_data.get("respect", 0)
            intensity = npc_data.get("intensity", 0)
            occupation = npc_data.get("occupation", "")
            hobbies = npc_data.get("hobbies", [])
            personality_traits = npc_data.get("personality_traits", [])
            likes = npc_data.get("likes", [])
            dislikes = npc_data.get("dislikes", [])
            affiliations = npc_data.get("affiliations", [])
            schedule = npc_data.get("schedule", {})
            memory = npc_data.get("memory", "")
            monica_level = npc_data.get("monica_level", 0)
            introduced = npc_data.get("introduced", False)

            cursor.execute("""
                INSERT INTO NPCStats (
                    npc_name, 
                    dominance, cruelty, closeness, trust, respect, intensity,
                    occupation, hobbies, personality_traits, likes, dislikes,
                    affiliations, schedule, memory, monica_level, introduced
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, 
                        %s, %s, %s, %s, %s, 
                        %s, %s, %s, %s)
            """, (
                npc_name, 
                dominance, cruelty, closeness, trust, respect, intensity,
                occupation, json.dumps(hobbies), json.dumps(personality_traits),
                json.dumps(likes), json.dumps(dislikes),
                json.dumps(affiliations), json.dumps(schedule),
                memory, monica_level, introduced
            ))

        # 3) NPC updates (existing NPCs)
        npc_updates = data.get("npc_updates", [])
        for up in npc_updates:
            npc_id = up["npc_id"]
            # We'll build a dynamic SET clause
            set_clauses = []
            set_values = []

            # We check which fields are present
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
            for field_key, db_col in fields_map.items():
                if field_key in up:
                    set_clauses.append(f"{db_col} = %s")
                    set_values.append(up[field_key])

            # build final query if any
            if set_clauses:
                set_str = ", ".join(set_clauses)
                set_values.append(npc_id)  # for the WHERE
                query = f"UPDATE NPCStats SET {set_str} WHERE npc_id=%s"
                cursor.execute(query, tuple(set_values))

        # 4) Character stat updates
        char_update = data.get("character_stat_updates", {})
        if char_update:
            pn = char_update.get("player_name", "Chase")
            new_stats = char_update.get("stats", {})
            # build dynamic SET
            set_clauses = []
            set_values = []
            stat_fields_map = {
                "corruption": "corruption",
                "confidence": "confidence",
                "willpower": "willpower",
                "obedience": "obedience",
                "dependency": "dependency",
                "lust": "lust",
                "mental_resilience": "mental_resilience",
                "physical_endurance": "physical_endurance"
            }
            for sf_key, db_col in stat_fields_map.items():
                if sf_key in new_stats:
                    set_clauses.append(f"{db_col} = %s")
                    set_values.append(new_stats[sf_key])

            if set_clauses:
                set_str = ", ".join(set_clauses)
                set_values.append(pn)
                query = f"UPDATE PlayerStats SET {set_str} WHERE player_name=%s"
                cursor.execute(query, tuple(set_values))

        # 5) Relationship updates (affiliations, etc.)
        rel_updates = data.get("relationship_updates", [])
        for rel in rel_updates:
            # e.g. { npc_id, affiliations: [...] }
            # or could store separate "relationship" table
            npc_id = rel.get("npc_id")
            aff_list = rel.get("affiliations", None)
            if aff_list is not None:
                cursor.execute("""
                    UPDATE NPCStats
                    SET affiliations = %s
                    WHERE npc_id = %s
                """, (json.dumps(aff_list), npc_id))

            # Could also handle npc→npc relationships or store them in a separate table, etc.

        # 6) NPC introductions
        npc_introductions = data.get("npc_introductions", [])
        for intro in npc_introductions:
            # e.g. { "npc_id": 9 }
            npc_id = intro["npc_id"]
            cursor.execute("""
                UPDATE NPCStats
                SET introduced = TRUE
                WHERE npc_id = %s
            """, (npc_id,))

        # 7) Location creations
        location_creations = data.get("location_creations", [])
        for loc in location_creations:
            # You might have a separate table, e.g. "Locations"
            loc_name = loc.get("location_name", "Unnamed Location")
            desc = loc.get("description", "")
            open_hours = loc.get("open_hours", [])
            # Example insert
            cursor.execute("""
                INSERT INTO Locations (name, description, open_hours)
                VALUES (%s, %s, %s)
            """, (loc_name, desc, json.dumps(open_hours)))

        # 8) Event list updates
        event_list_updates = data.get("event_list_updates", [])
        for ev in event_list_updates:
            # e.g. { event_name, description }
            # Maybe store in 'Events' table
            ev_name = ev.get("event_name", "Unnamed Event")
            ev_desc = ev.get("description", "")
            cursor.execute("""
                INSERT INTO Events (event_name, description)
                VALUES (%s, %s)
            """, (ev_name, ev_desc))

        # 9) Inventory updates
        inv_updates = data.get("inventory_updates", {})
        if inv_updates:
            # example structure
            # "player_name": "Chase",
            # "added_items": [...],
            # "removed_items": [...]
            player_n = inv_updates.get("player_name", "Chase")
            added_items = inv_updates.get("added_items", [])
            removed_items = inv_updates.get("removed_items", [])
            # up to you if you store in a PlayerInventory table. 
            # e.g.:
            for item in added_items:
                cursor.execute("""
                    INSERT INTO PlayerInventory (player_name, item_name)
                    VALUES (%s, %s)
                """, (player_n, item))
            for item in removed_items:
                cursor.execute("""
                    DELETE FROM PlayerInventory 
                    WHERE player_name=%s AND item_name=%s
                    LIMIT 1
                """, (player_n, item))
            # or do more complicated logic

        # 10) Quest updates
        quest_updates = data.get("quest_updates", [])
        for quest in quest_updates:
            # e.g. { quest_id, status, progress_detail }
            q_id = quest.get("quest_id")
            status = quest.get("status", "In Progress")
            detail = quest.get("progress_detail", "")
            # update or insert into Quests table
            cursor.execute("""
                UPDATE Quests
                SET status=%s, progress_detail=%s
                WHERE quest_id=%s
            """, (status, detail, q_id))

        conn.commit()
        conn.close()

        return jsonify({"message": "Universal update successful"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
