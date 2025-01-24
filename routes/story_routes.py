# routes/story_routes.py

from flask import Blueprint, request, jsonify
import logging
import json

from db.connection import get_db_connection
from logic.activities_logic import filter_activities_for_npc, build_short_summary
from logic.stats_logic import update_player_stats
from logic.meltdown_logic import check_and_inject_meltdown  # Central meltdown synergy
from logic.aggregator import get_aggregated_roleplay_context
from routes.meltdown import remove_meltdown_npc  # If you keep that route
from routes.settings_routes import generate_mega_setting_logic  # if you want to generate new environment
from logic.universal_updater import apply_universal_updates  # Single universal update
from logic.time_cycle import advance_time_and_update
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

        # 1) If there's a 'universal_update' block, apply it
        universal_data = data.get("universal_update", {})
        if universal_data:
            logging.info("Applying universal update from payload.")
            update_result = apply_universal_updates(universal_data)
            if "error" in update_result:
                # Log or handle error from universal update
                logging.warning(f"Universal update error: {update_result['error']}")
        else:
            logging.info("No universal update data found, skipping DB updates aside from meltdown or user triggers.")

        # 2) Handle user input triggers (like meltdown removal, forced obedience, environment generation)
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
            remove_meltdown_npc(force=True)  # calls meltdown removal route logic
            meltdown_forced_removal = True

        mega_setting_name_if_generated = None
        if "generate environment" in user_lower or "mega setting" in user_lower:
            logging.info("Generating new mega setting")
            mega_data = generate_mega_setting_logic()
            mega_setting_name_if_generated = mega_data.get("mega_name", "No environment")
            logging.info(f"New Mega Setting: {mega_setting_name_if_generated}")

        # We'll track meltdown or new NPC data, etc.
        meltdown_newly_triggered = False
        removed_npcs_list = []
        new_npc_data = None

        # 3) Fetch aggregator data after universal updates
        aggregator_data = get_aggregated_roleplay_context(player_name)
        logging.info(f"Aggregator Data: {aggregator_data}")

        # Possibly parse meltdown level or NPC archetypes from aggregator
        npc_archetypes = []
        meltdown_level = 0
        setting_str = None

        current_rp = aggregator_data.get("currentRoleplay", {})
        if "CurrentSetting" in current_rp:
            setting_str = current_rp["CurrentSetting"]

        npc_list = aggregator_data.get("npcStats", [])
        if npc_list:
            # Example: if first NPC name has "giant", assume "Giantess" archetype
            if "giant" in npc_list[0].get("npc_name", "").lower():
                npc_archetypes = ["Giantess"]
            meltdown_level = 0  # or do something else if meltdown is relevant

        user_stats = aggregator_data.get("playerStats", {})

        # 4) Suggest possible activities based on current stats, meltdown, setting, etc.
        chosen_activities = filter_activities_for_npc(
            npc_archetypes=npc_archetypes,
            meltdown_level=meltdown_level,
            user_stats=user_stats,
            setting=setting_str or ""
        )
        lines_for_gpt = [build_short_summary(act) for act in chosen_activities]
        aggregator_data["activitySuggestions"] = lines_for_gpt

        # 4b) Now we do meltdown synergy with the centralized meltdown check
        meltdown_flavor = check_and_inject_meltdown()
        meltdown_newly_triggered = bool(meltdown_flavor)

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

        # 5) Possibly advance the time
        time_spent = 1
        new_day, new_phase = advance_time_and_update(increment=time_spent)

        # 6) Re-fetch aggregator in case anything changed again
        aggregator_data = get_aggregated_roleplay_context(player_name)
        # Build final textual summary
        story_output = build_aggregator_text(aggregator_data, meltdown_flavor)

        # Return final scenario text + some updates
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


def build_aggregator_text(aggregator_data, meltdown_flavor=""):
    """
    Merges aggregator_data into user-friendly text for your front-end or GPT usage.
    If meltdown_flavor is non-empty, append it at the end.
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

    # CurrentRoleplay
    lines.append("\n=== CURRENT ROLEPLAY ===")
    if current_rp:
        for k, v in current_rp.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append("(No current roleplay data)")

    # If aggregator_data has "activitySuggestions" from filter_activities_for_npc
    if "activitySuggestions" in aggregator_data:
        lines.append("\n=== NPC POTENTIAL ACTIVITIES ===")
        for suggestion in aggregator_data["activitySuggestions"]:
            lines.append(f"- {suggestion}")
        lines.append("NPC can adopt, combine, or ignore these ideas in line with their personality.\n")

    # meltdown flavor if present
    if meltdown_flavor:
        lines.append("\n=== MELTDOWN NPC MESSAGE ===")
        lines.append(meltdown_flavor)

    return "\n".join(lines)
