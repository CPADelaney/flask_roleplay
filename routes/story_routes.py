# routes/story_routes.py

from flask import Blueprint, request, jsonify
import logging
import json

from db.connection import get_db_connection
from logic.activities_logic import filter_activities_for_npc, build_short_summary
from logic.stats_logic import update_player_stats
#from logic.meltdown_logic import check_and_inject_meltdown
from logic.aggregator import get_aggregated_roleplay_context
#from routes.meltdown import remove_meltdown_npc
from routes.settings_routes import generate_mega_setting_logic
from logic.universal_updater import apply_universal_updates
from logic.time_cycle import advance_time_and_update
from logic.inventory_logic import (
    add_item_to_inventory, remove_item_from_inventory, get_player_inventory
)

story_bp = Blueprint("story_bp", __name__)

@story_bp.route("/next_storybeat", methods=["POST"])
def next_storybeat():
    """
    Handles the main story logic, processes user input, and returns story context.
    Also automatically applies any 'universal_update' data that GPT or the front-end
    passes in (like new NPCs, changed stats, location creation, etc.).
    """
    try:
        # 1) Get the request JSON
        data = request.get_json() or {}

        # 2) If the GPT front-end nests everything under "params", unwrap it
        if "params" in data:
            data = data["params"]

        # 3) Extract fields
        player_name = data.get("player_name", "Chase")
        user_input = data.get("user_input", "")
        logging.info(f"Player: {player_name}, User Input: {user_input}")

        # 4) Apply universal update if provided
        universal_data = data.get("universal_update", {})
        if universal_data:
            logging.info("Applying universal update from payload.")
            update_result = apply_universal_updates(universal_data)
            if "error" in update_result:
                return jsonify(update_result), 500
        else:
            logging.info("No universal update data found, skipping DB updates aside from meltdown or user triggers.")

        # 5) Handle user input triggers (like forcing obedience=100)
        user_lower = user_input.lower()
        if "obedience=100" in user_lower:
            logging.info("Setting obedience to 100 for player.")
            force_obedience_to_100(player_name)

        # 6) Fetch aggregator data
        aggregator_data = get_aggregated_roleplay_context(player_name)
        logging.info(f"Aggregator Data: {aggregator_data}")

        # Possibly generate a mega setting if forced or missing
        mega_setting_name_if_generated = None
        current_setting = aggregator_data["currentRoleplay"].get("CurrentSetting")
        if ("generate environment" in user_lower or "mega setting" in user_lower):
            if current_setting and "force" not in user_lower:
                logging.info(f"Already have environment '{current_setting}'. Skipping new generation.")
            else:
                logging.info("Generating new mega setting (forced or none set).")
                mega_data = generate_mega_setting_logic()
                mega_setting_name_if_generated = mega_data.get("mega_name", "No environment")
                logging.info(f"New Mega Setting: {mega_setting_name_if_generated}")

        # Prepare placeholders for other logic
        removed_npcs_list = []
        new_npc_data = None
   #     meltdown_level = 0
        setting_str = current_setting if current_setting else ""
        npc_list = aggregator_data.get("npcStats", [])
        npc_archetypes = []

        if npc_list:
            # Example: if the first NPC name has "giant" in it
            if "giant" in npc_list[0].get("npc_name", "").lower():
                npc_archetypes = ["Giantess"]

        user_stats = aggregator_data.get("playerStats", {})

        # 7) Suggest possible activities
        chosen_activities = filter_activities_for_npc(
            npc_archetypes=npc_archetypes,
     #       meltdown_level=meltdown_level,
            user_stats=user_stats,
            setting=setting_str
        )
        lines_for_gpt = [build_short_summary(act) for act in chosen_activities]
        aggregator_data["activitySuggestions"] = lines_for_gpt

        # 8) Track internal changes (just an example if you changed stats above)
        changed_stats = {"obedience": 100} if "obedience=100" in user_lower else {}
        updates_dict = {
            "new_mega_setting": mega_setting_name_if_generated,
            "updated_player_stats": changed_stats,
            "removed_npc_ids": removed_npcs_list,
            "added_npc": new_npc_data,
            "plot_event": None
        }

        logging.info(f"Updates Dict: {updates_dict}")

        # 9) Conditionally advance time
        # if "advance_time" is True in the JSON, we increment the day/time
        advance_time = data.get("advance_time", False)
        if advance_time:
            logging.info("Advancing time by 1 block.")
            new_day, new_phase = advance_time_and_update(increment=1)
        else:
            logging.info("Skipping time advancement, using aggregator data's existing day/time.")
            # Keep the aggregator's current day/time
            new_day = aggregator_data.get("day", 1)
            new_phase = aggregator_data.get("timeOfDay", "Morning")

        # 10) Re-fetch aggregator if you want the final updated state
        aggregator_data = get_aggregated_roleplay_context(player_name)
        story_output = build_aggregator_text(aggregator_data)

        # 11) Return final scenario text and updates
        return jsonify({
            "story_output": story_output,
            "updates": {
                "current_day": new_day,
                "time_of_day": new_phase,
                "internal_changes": updates_dict
            }
        }), 200

    except Exception as e:
        logging.exception("Error in next_storybeat")
        return jsonify({"error": str(e)}), 500


def force_obedience_to_100(player_name):
    """
    Simple direct approach to set player's Obedience=100.
    Alternatively, you could place this in stats_logic.py or use update_player_stats.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE PlayerStats
            SET obedience = 100
            WHERE player_name = %s
        """, (player_name,))
        conn.commit()
    except:
        conn.rollback()
    finally:
        conn.close()


def build_aggregator_text(aggregator_data):
    """
    Merges aggregator_data into user-friendly text for your front-end or GPT usage.
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

    lines.append("\n=== NPC STATS ===")
    if npc_stats:
        for npc in npc_stats:
            lines.append(
                f"NPC: {npc.get('npc_name','Unnamed')} | "
                f"Dom={npc.get('dominance',0)}, Cru={npc.get('cruelty',0)}, "
                f"Close={npc.get('closeness',0)}, Trust={npc.get('trust',0)}, "
                f"Respect={npc.get('respect',0)}, Int={npc.get('intensity',0)}"
            )
            hobbies = npc.get("hobbies", [])
            personality = npc.get("personality_traits", [])
            likes = npc.get("likes", [])
            dislikes = npc.get("dislikes", [])

            lines.append(f"  Hobbies: {', '.join(hobbies)}" if hobbies else "  Hobbies: None")
            lines.append(f"  Personality: {', '.join(personality)}" if personality else "  Personality: None")
            lines.append(f"  Likes: {', '.join(likes)} | Dislikes: {', '.join(dislikes)}\n")
    else:
        lines.append("(No NPCs found)")

    lines.append("\n=== CURRENT ROLEPLAY ===")
    if current_rp:
        for k, v in current_rp.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append("(No current roleplay data)")

    env_desc = current_rp.get("EnvironmentDescription")
    if env_desc:
        lines.append(f"\nENVIRONMENT: {env_desc}")

    if "activitySuggestions" in aggregator_data:
        lines.append("\n=== NPC POTENTIAL ACTIVITIES ===")
        for suggestion in aggregator_data["activitySuggestions"]:
            lines.append(f"- {suggestion}")
        lines.append("NPCs can adopt, combine, or ignore these ideas in line with their personality.\n")

    return "\n".join(lines)
