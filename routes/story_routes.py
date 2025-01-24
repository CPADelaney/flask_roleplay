# routes/story_routes.py

from flask import Blueprint, request, jsonify
import logging

# (A) NEW: import your activities logic
from logic.activities_logic import filter_activities_for_npc, build_short_summary

from logic.stats_logic import update_player_stats  # if you want direct calls
from logic.meltdown_logic import meltdown_dialog_gpt, record_meltdown_dialog
from logic.aggregator import get_aggregated_roleplay_context
from routes.meltdown import remove_meltdown_npc
from db.connection import get_db_connection
from routes.settings_routes import generate_mega_setting_route  # or your logic version

story_bp = Blueprint("story_bp", __name__)

@story_bp.route("/next_storybeat", methods=["POST"])
def next_storybeat():
    """
    Handles the main story logic, processes user input, and returns story context.
    Debugging and error handling are included to capture any issues.
    """
    try:
        # Get and log request data
        data = request.get_json() or {}
        logging.info(f"Request Data: {data}")

        player_name = data.get("player_name", "Chase")
        user_input = data.get("user_input", "")
        logging.info(f"Player: {player_name}, User Input: {user_input}")

        user_lower = user_input.lower()

        # -----------------------------------------------------------------
        # 1) Check meltdown removal, forced obedience, or other triggers
        # -----------------------------------------------------------------
        meltdown_forced_removal = False
        if "obedience=100" in user_lower:
            logging.info("Setting obedience to 100")
            force_obedience_to_100(player_name)

        if "remove meltdown" in user_lower:
            logging.info("Attempting to remove meltdown NPCs")
            remove_meltdown_npc(force=True)
            meltdown_forced_removal = True

        # -----------------------------------------------------------------
        # 2) Possibly generate a new environment if user_input asks for it
        # -----------------------------------------------------------------
        mega_setting_name_if_generated = None
        if "generate environment" in user_lower or "mega setting" in user_lower:
            logging.info("Generating new mega setting")
            mega_setting_name_if_generated = generate_mega_setting_logic()
            logging.info(f"New Mega Setting: {mega_setting_name_if_generated}")
        
        # Track meltdown triggering
        meltdown_newly_triggered = False  # Example placeholder
        # Track removed NPC IDs and any new NPCs
        removed_npcs_list = []
        new_npc_data = None  # Example placeholder

        # -----------------------------------------------------------------
        # 3) Fetch the aggregated context from DB
        # -----------------------------------------------------------------
        logging.info("Fetching aggregated roleplay context")
        aggregator_data = get_aggregated_roleplay_context(player_name)
        logging.info(f"Aggregator Data: {aggregator_data}")

        # (A) NEW: Decide how to gather NPC archetype + meltdown level, setting, user stats
        # In your aggregator_data, you might have something like aggregator_data["npcStats"]
        # We'll do a naive example:

        npc_archetypes = []
        meltdown_level = 0
        setting_str = None

        # Possibly read from aggregator_data's currentRoleplay
        current_rp = aggregator_data.get("currentRoleplay", {})
        if "CurrentSetting" in current_rp:
            setting_str = current_rp["CurrentSetting"]  # e.g. "Urban Life" or "High Society"

        # If we have multiple NPC stats, we might pick the "dominant" NPC or top meltdown
        # This is up to you to define; here's just a sample:
        npc_list = aggregator_data.get("npcStats", [])
        if npc_list:
            # For demonstration, let's just look at the first NPC:
            # If you actually store .archetypes in the DB or in aggregator_data, we can read it.
            # We'll just assume no archetypes for now or a placeholder:
            npc_archetypes = ["Giantess"] if "giant" in npc_list[0].get("npc_name","").lower() else []

            # meltdown_level is if we store it in aggregator_data or from meltdown_npc
            # We'll guess monica_level is stored in aggregator_data somewhere
            meltdown_level = 0  # or aggregator_data["someKey"]

        # For user_stats, we do:
        user_stats = aggregator_data.get("playerStats", {})

        # (A) NEW: Actually fetch activity suggestions
        chosen_activities = filter_activities_for_npc(
            npc_archetypes=npc_archetypes,
            meltdown_level=meltdown_level,
            user_stats=user_stats,
            setting=setting_str or ""
        )

        # Summarize them
        lines_for_gpt = [build_short_summary(act) for act in chosen_activities]

        # We'll store them in aggregator_data so that build_aggregator_text can display them
        aggregator_data["activitySuggestions"] = lines_for_gpt

        # -----------------------------------------------------------------
        # 4) (Optional) If meltdown NPCs exist, add meltdown flavor
        # -----------------------------------------------------------------
        logging.info("Checking for meltdown flavor")
        meltdown_flavor = check_for_meltdown_flavor()
        logging.info(f"Meltdown Flavor: {meltdown_flavor}")

        # -----------------------------------------------------------------
        # 5) Convert aggregator_data into final "story_output" text
        # -----------------------------------------------------------------
        logging.info("Building story output")
        story_output = build_aggregator_text(aggregator_data, meltdown_flavor)
        logging.info(f"Story Output: {story_output}")

        # -----------------------------------------------------------------
        # 6) Build the updates object
        # -----------------------------------------------------------------
        changed_stats = {"obedience": 100} if "obedience=100" in user_lower else {}

        updates_dict = {
            "meltdown_triggered": meltdown_newly_triggered,
            "meltdown_removed": meltdown_forced_removal,
            "new_mega_setting": mega_setting_name_if_generated,
            "updated_player_stats": changed_stats,
            "removed_npc_ids": removed_npcs_list,
            "added_npc": new_npc_data,
            "plot_event": None,  # or other meaningful value
        }
        logging.info(f"Updates Dict: {updates_dict}")

        # -----------------------------------------------------------------
        # 7) Return final scenario text plus updates
        # -----------------------------------------------------------------
        return jsonify({
            "story_output": story_output,
            "updates": updates_dict
        }), 200

    except Exception as e:
        # Log the error with a traceback
        logging.exception("An error occurred in /story/next_storybeat")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


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
    Optionally fetch meltdown NPCs from the DB.
    If any meltdown NPC is present, generate a meltdown line or add special text.

    Returns a string or empty if no meltdown is active.
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

    # Return meltdown flavor text to be appended or embedded in aggregator text
    return f"[Meltdown NPC {npc_name} - meltdown_level={meltdown_level}]: {meltdown_line}"


def build_aggregator_text(aggregator_data, meltdown_flavor=""):
    """
    Merges aggregator_data into user-friendly text for the front-end GPT to use.
    This is the 'final scenario text' or 'context block' your custom GPT can fetch.
    """

    player_stats = aggregator_data.get("playerStats", {})
    npc_stats = aggregator_data.get("npcStats", [])
    current_rp = aggregator_data.get("currentRoleplay", {})

    lines = []

    # 1) Summarize Player Stats
    lines.append("=== PLAYER STATS ===")
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

    # 2) Summarize NPC Stats
    lines.append("\n=== NPC STATS ===")
    if npc_stats:
        for npc in npc_stats:
            lines.append(
                f"NPC: {npc.get('npc_name','Unnamed')} | "
                f"Dom={npc.get('dominance',0)}, Cru={npc.get('cruelty',0)}, "
                f"Close={npc.get('closeness',0)}, Trust={npc.get('trust',0)}, "
                f"Respect={npc.get('respect',0)}, Int={npc.get('intensity',0)}"
            )
            
            # Now add occupation, hobbies, personality, etc.
            occupation = npc.get("occupation", "Unemployed?")
            hobbies = npc.get("hobbies", [])
            personality = npc.get("personality_traits", [])
            likes = npc.get("likes", [])
            dislikes = npc.get("dislikes", [])

            lines.append(f"  Occupation: {occupation}")
            lines.append(f"  Hobbies: {', '.join(hobbies)}" if hobbies else "  Hobbies: None")
            lines.append(f"  Personality: {', '.join(personality)}" if personality else "  Personality: None")
            lines.append(f"  Likes: {', '.join(likes)} | Dislikes: {', '.join(dislikes)}")
            lines.append("")  # blank line for spacing
    else:
        lines.append("(No NPCs found)")


    # 3) Current Roleplay Data
    lines.append("\n=== CURRENT ROLEPLAY ===")
    if current_rp:
        for k, v in current_rp.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append("(No current roleplay data)")

    # (A) NEW: If we have activity suggestions
    if "activitySuggestions" in aggregator_data:
        lines.append("\n=== NPC POTENTIAL ACTIVITIES ===")
        for suggestion in aggregator_data["activitySuggestions"]:
            lines.append(f"- {suggestion}")
        # Tiny instruction so GPT can spontaneously use them or not
        lines.append("NPC can adopt, combine, or ignore these ideas in line with her personality.\n")

    # 4) meltdown flavor
    if meltdown_flavor:
        lines.append("\n=== MELTDOWN NPC MESSAGE ===")
        lines.append(meltdown_flavor)

    # 5) Final joined text
    final_text = "\n".join(lines)
    return final_text
