# routes/story_routes.py

from flask import Blueprint, request, jsonify
import logging
import json

from logic.activities_logic import filter_activities_for_npc, build_short_summary
from logic.stats_logic import update_player_stats
from logic.meltdown_logic import meltdown_dialog_gpt, record_meltdown_dialog
from logic.aggregator import get_aggregated_roleplay_context
from routes.meltdown import remove_meltdown_npc
from db.connection import get_db_connection
from routes.settings_routes import generate_mega_setting_route
from logic.time_cycle import advance_time_and_update, get_current_daytime
from logic.inventory_logic import add_item_to_inventory, remove_item_from_inventory, get_player_inventory


story_bp = Blueprint("story_bp", __name__)


@story_bp.route("/next_storybeat", methods=["POST"])
def next_storybeat():
    """
    Handles the main story logic, processes user input, and returns story context.
    Also automatically applies any 'universal update' data that GPT or the front-end
    passes in (like new NPCs, changed stats, location creation, etc.).
    """
    try:
        data = request.get_json() or {}
        logging.info(f"Request Data: {data}")

        # (1) FIRST: If we have a 'universal_update' key in the incoming JSON,
        #    we apply it to the database
        universal_data = data.get("universal_update", {})
        if universal_data:
            logging.info("Applying universal update from payload.")
            apply_universal_updates(universal_data)
        else:
            logging.info("No universal update data found, skipping DB updates aside from meltdown or user triggers.")

        # (2) Now handle meltdown removal, forced obedience, environment generation, etc.
        player_name = data.get("player_name", "Chase")
        user_input = data.get("user_input", "")
        logging.info(f"Player: {player_name}, User Input: {user_input}")

        meltdown_forced_removal = False
        user_lower = user_input.lower()

        if "obedience=100" in user_lower:
            logging.info("Setting obedience to 100 for player.")
            force_obedience_to_100(player_name)

        if "remove meltdown" in user_lower:
            logging.info("Attempting to remove meltdown NPCs")
            remove_meltdown_npc(force=True)
            meltdown_forced_removal = True

        mega_setting_name_if_generated = None
        if "generate environment" in user_lower or "mega setting" in user_lower:
            logging.info("Generating new mega setting")
            mega_setting_name_if_generated = generate_mega_setting_logic()
            logging.info(f"New Mega Setting: {mega_setting_name_if_generated}")

        meltdown_newly_triggered = False  # placeholder
        removed_npcs_list = []
        new_npc_data = None

        # (3) Fetch aggregator data again after universal updates
        aggregator_data = get_aggregated_roleplay_context(player_name)
        logging.info(f"Aggregator Data: {aggregator_data}")

        # Possibly pick meltdown level or npc_archetypes from aggregator
        npc_archetypes = []
        meltdown_level = 0
        setting_str = None

        current_rp = aggregator_data.get("currentRoleplay", {})
        if "CurrentSetting" in current_rp:
            setting_str = current_rp["CurrentSetting"]

        npc_list = aggregator_data.get("npcStats", [])
        if npc_list:
            # Example: if first NPC name contains 'giant', assume 'Giantess' archetype
            npc_archetypes = ["Giantess"] if "giant" in npc_list[0].get("npc_name", "").lower() else []
            meltdown_level = 0

        user_stats = aggregator_data.get("playerStats", {})

        # (4) Get possible activity suggestions
        chosen_activities = filter_activities_for_npc(
            npc_archetypes=npc_archetypes,
            meltdown_level=meltdown_level,
            user_stats=user_stats,
            setting=setting_str or ""
        )
        lines_for_gpt = [build_short_summary(act) for act in chosen_activities]
        aggregator_data["activitySuggestions"] = lines_for_gpt

        # Check meltdown flavor
        meltdown_flavor = check_for_meltdown_flavor()

        changed_stats = {"obedience": 100} if "obedience=100" in user_lower else {}

        updates_dict = {
            "meltdown_triggered": meltdown_newly_triggered,
            "meltdown_removed": meltdown_forced_removal,
            "new_mega_setting": mega_setting_name_if_generated,
            "updated_player_stats": changed_stats,
            "removed_npc_ids": removed_npcs_list,
            "added_npc": new_npc_data,
            "plot_event": None,
        }
        logging.info(f"Updates Dict: {updates_dict}")

        # (5) Possibly advance the time one step if that's your design
        # (If user input doesn't mention time skip, do 1 increment, etc.)
        time_spent = 1
        new_day, new_phase = advance_time_and_update(increment=time_spent)

        # (6) Re-fetch aggregator in case anything changed again
        aggregator_data = get_aggregated_roleplay_context(player_name)
        story_output = build_aggregator_text(aggregator_data, meltdown_flavor)

        # Return final scenario text + updates
        return jsonify({
            "story_output": story_output,
            "updates": {
                "current_day": new_day,
                "time_of_day": new_phase
            }
        }), 200

    except Exception as e:
        logging.exception("Error in next_storybeat")
        return jsonify({"error": str(e)}), 500


def apply_universal_updates(universal_data):
    """
    Applies the "universal" style database updates in-line, so that
    any GPT or user-provided changes are immediately reflected.

    This is effectively the same logic as a universal update endpoint,
    but in a function here so we can call it from next_storybeat.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) roleplay_updates (timeOfDay, CurrentSetting, UsedSettings, etc.)
    roleplay_updates = universal_data.get("roleplay_updates", {})
    for key, value in roleplay_updates.items():
        cursor.execute("""
            INSERT INTO CurrentRoleplay (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """, (key, json.dumps(value) if isinstance(value, dict) else str(value)))

    # 2) npc_creations
    npc_creations = universal_data.get("npc_creations", [])
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
                npc_name, introduced, dominance, cruelty, closeness, trust, respect, intensity,
                occupation, hobbies, personality_traits, likes, dislikes,
                affiliations, schedule, memory, monica_level
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s)
        """, (
            name, introduced, dom, cru, clos, tru, resp, inten,
            occ, json.dumps(hbs), json.dumps(pers), json.dumps(lks), json.dumps(dlks),
            json.dumps(affil), json.dumps(sched), mem, monica_lvl
        ))

    # 3) npc_updates
    npc_updates = universal_data.get("npc_updates", [])
    for up in npc_updates:
        npc_id = up.get("npc_id")
        if not npc_id:
            continue
        set_clauses = []
        set_values = []
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
        for k, col in fields_map.items():
            if k in up:
                set_clauses.append(f"{col} = %s")
                set_values.append(up[k])

        if set_clauses:
            set_str = ", ".join(set_clauses)
            set_values.append(npc_id)
            query = f"UPDATE NPCStats SET {set_str} WHERE npc_id=%s"
            cursor.execute(query, tuple(set_values))

    # 4) character_stat_updates
    char_update = universal_data.get("character_stat_updates", {})
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
    rel_updates = universal_data.get("relationship_updates", [])
    for r in rel_updates:
        npc_id = r.get("npc_id")
        if not npc_id:
            continue
        # e.g. affiliations
        aff_list = r.get("affiliations", None)
        if aff_list is not None:
            cursor.execute("""
                UPDATE NPCStats
                SET affiliations = %s
                WHERE npc_id = %s
            """, (json.dumps(aff_list), npc_id))

    # 6) npc_introductions
    npc_intros = universal_data.get("npc_introductions", [])
    for intro in npc_intros:
        npc_id = intro.get("npc_id")
        if npc_id:
            cursor.execute("""
                UPDATE NPCStats
                SET introduced = TRUE
                WHERE npc_id=%s
            """, (npc_id,))

    # 7) location_creations
    location_creations = universal_data.get("location_creations", [])
    for loc in location_creations:
        loc_name = loc.get("location_name", "Unnamed")
        desc = loc.get("description", "")
        open_hours = loc.get("open_hours", [])
        # you need a Locations table: (id, name, description, open_hours JSON, etc.)
        cursor.execute("""
            INSERT INTO Locations (name, description, open_hours)
            VALUES (%s, %s, %s)
        """, (loc_name, desc, json.dumps(open_hours)))

    # 8) event_list_updates
    event_updates = universal_data.get("event_list_updates", [])
    for ev in event_updates:
        ev_name = ev.get("event_name", "UnnamedEvent")
        ev_desc = ev.get("description", "")
        # you need an Events table
        cursor.execute("""
            INSERT INTO Events (event_name, description)
            VALUES (%s, %s)
        """, (ev_name, ev_desc))

    # 9) inventory_updates
    inv_updates = universal_data.get("inventory_updates", {})
    if inv_updates:
        p_n = inv_updates.get("player_name", "Chase")
        added = inv_updates.get("added_items", [])
        removed = inv_updates.get("removed_items", [])
        for item in added:
            cursor.execute("""
                INSERT INTO PlayerInventory (player_name, item_name)
                VALUES (%s, %s)
            """, (p_n, item))
        for item in removed:
            cursor.execute("""
                DELETE FROM PlayerInventory
                WHERE player_name=%s AND item_name=%s
                LIMIT 1
            """, (p_n, item))

    # 10) quest_updates
    quest_updates = universal_data.get("quest_updates", [])
    for qu in quest_updates:
        quest_id = qu.get("quest_id")
        status = qu.get("status", "In Progress")
        detail = qu.get("progress_detail", "")
        # you need a Quests table: (quest_id PK, status, progress_detail, etc.)
        cursor.execute("""
            UPDATE Quests
            SET status=%s, progress_detail=%s
            WHERE quest_id=%s
        """, (status, detail, quest_id))

    conn.commit()
    conn.close()


def force_obedience_to_100(player_name):
    """
    Simple example that updates the player's obedience to 100
    through your stats logic or direct DB logic.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE PlayerStats
            SET obedience=100
            WHERE player_name=%s
        """, (player_name,))
        conn.commit()
    except:
        conn.rollback()
    finally:
        conn.close()


def check_for_meltdown_flavor():
    """
    Fetch meltdown NPCs from the DB. If any meltdown NPC is present,
    generate meltdown line or add special text.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT npc_id, npc_name, monica_level
        FROM NPCStats
        WHERE monica_level > 0
        ORDER BY monica_level DESC
    """)
    meltdown_npcs = cursor.fetchall()
    conn.close()

    if not meltdown_npcs:
        return ""

    # Take the top meltdown NPC
    npc_id, npc_name, meltdown_level = meltdown_npcs[0]
    meltdown_line = meltdown_dialog_gpt(npc_name, meltdown_level)
    record_meltdown_dialog(npc_id, meltdown_line)

    return f"[Meltdown NPC {npc_name} - meltdown_level={meltdown_level}]: {meltdown_line}"


def build_aggregator_text(aggregator_data, meltdown_flavor=""):
    """
    Merges aggregator_data into user-friendly text for the front-end GPT to use.
    """
    lines = []
    day = aggregator_data.get("day", "1")
    tod = aggregator_data.get("timeOfDay", "Morning")
    player_stats = aggregator_data.get("playerStats", {})
    npc_stats = aggregator_data.get("npcStats", [])
    current_rp = aggregator_data.get("currentRoleplay", {})

    lines.append("=== PLAYER STATS ===")
    lines.append(f"=== DAY {day}, {tod.upper()} ===")

    if player_stats:
        lines.append(
            f"Name: {player_stats.get('name', 'Unknown')}\n"
            f"Corruption: {player_stats.get('corruption', 0)}, "
            f"Confidence: {player_stats.get('confidence', 0)}, "
            f"Willpower: {player_stats.get('willpower', 0)}, "
            f"Obedience: {player_stats.get('obedience', 0)}, "
            f"Dependency: {player_stats.get('dependency', 0)}, "
            f"Lust: {player_stats.get('lust', 0)}, "
            f"MentalResilience: {player_stats.get('mental_resilience', 0)}, "
            f"PhysicalEndurance: {player_stats.get('physical_endurance', 0)}"
        )
    else:
        lines.append("(No player stats found)")

    # NPC Stats
    lines.append("\n=== NPC STATS ===")
    if npc_stats:
        for npc in npc_stats:
            lines.append(
                f"NPC: {npc.get('npc_name','Unnamed')} | "
                f"Dom={npc.get('dominance',0)}, Cru={npc.get('cruelty',0)}, "
                f"Close={npc.get('closeness',0)}, Trust={npc.get('trust',0)}, "
                f"Respect={npc.get('respect',0)}, Int={npc.get('intensity',0)}"
            )
            occupation = npc.get("occupation", "Unemployed?")
            hobbies = npc.get("hobbies", [])
            personality = npc.get("personality_traits", [])
            likes = npc.get("likes", [])
            dislikes = npc.get("dislikes", [])

            lines.append(f"  Occupation: {occupation}")
            lines.append(f"  Hobbies: {', '.join(hobbies)}" if hobbies else "  Hobbies: None")
            lines.append(f"  Personality: {', '.join(personality)}" if personality else "  Personality: None")
            lines.append(f"  Likes: {', '.join(likes)} | Dislikes: {', '.join(dislikes)}\n")
    else:
        lines.append("(No NPCs found)")

    universal_update = data.get("universal_update", {})
    
    inventory_updates = universal_update.get("inventory_updates", {})
    
    if inventory_updates:
        # Suppose "inventory_updates" is something like:
        # {
        #   "player_name": "Chase",
        #   "added_items": ["Potion", "Rope"],
        #   "removed_items": ["Expired Key"]
        # }
        p_name = inventory_updates.get("player_name", "Chase")
    
        added = inventory_updates.get("added_items", [])
        removed = inventory_updates.get("removed_items", [])
    
        # For each item to add
        for item_name in added:
            add_item_to_inventory(p_name, item_name, 
                                  description="???",   # or fetch from a reference table
                                  effect="???", 
                                  category="misc", 
                                  quantity=1)
    
        # For each item to remove
        for item_name in removed:
            remove_item_from_inventory(p_name, item_name, quantity=1)

    # Current Roleplay
    lines.append("\n=== CURRENT ROLEPLAY ===")
    if current_rp:
        for k, v in current_rp.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append("(No current roleplay data)")

    # Potential Activities
    if "activitySuggestions" in aggregator_data:
        lines.append("\n=== NPC POTENTIAL ACTIVITIES ===")
        for suggestion in aggregator_data["activitySuggestions"]:
            lines.append(f"- {suggestion}")
        lines.append("NPC can adopt, combine, or ignore these ideas in line with her personality.\n")

    # Meltdown flavor
    if meltdown_flavor:
        lines.append("\n=== MELTDOWN NPC MESSAGE ===")
        lines.append(meltdown_flavor)

    return "\n".join(lines)
