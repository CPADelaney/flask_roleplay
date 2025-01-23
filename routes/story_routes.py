from flask import Blueprint, request, jsonify
#
# Import your logic pieces here:
#
from logic.stats_logic import update_player_stats  # if you want direct calls
from logic.meltdown_logic import remove_meltdown_npc, meltdown_dialog_gpt, record_meltdown_dialog
from logic.aggregator import get_aggregated_roleplay_context
from db.connection import get_db_connection
# If you have a "generate_mega_setting_route" or a direct function:
# from routes.settings_routes import generate_mega_setting_route
# or if you have a direct function:
# from .some_logic_module import generate_mega_setting_logic

story_bp = Blueprint("story_bp", __name__)

@story_bp.route("/next_storybeat", methods=["POST"])
def next_storybeat():
    """
    POST /story/next_storybeat
    --------------------------
    Expects JSON body: {"player_name": "...", "user_input": "..."}.
    
    Steps:
      1) Parse user_input for meltdown removal or stat changes.
      2) Possibly update DB or generate new environment.
      3) Build aggregator context from DB (the big dict).
      4) (Optional) Add meltdown flavor if meltdown NPCs are present.
      5) Convert aggregator_data into final story_output text.
      6) Construct an updates_dict to reflect any changes.
      7) Return JSON with {"story_output": story_output, "updates": updates_dict}.
    """
    data = request.get_json() or {}
    player_name = data.get("player_name", "Chase")
    user_input = data.get("user_input", "")

    user_lower = user_input.lower()

    # -----------------------------------------------------------------
    # 1) Check meltdown removal, forced obedience, or other triggers
    # -----------------------------------------------------------------
    meltdown_forced_removal = False
    if "obedience=100" in user_lower:
        # Example: directly set the player's obedience to 100
        force_obedience_to_100(player_name)

    if "remove meltdown" in user_lower:
        # meltdown removal attempt
        remove_meltdown_npc(force=True)
        meltdown_forced_removal = True

    # -----------------------------------------------------------------
    # 2) Possibly generate a new environment if user_input asks for it
    # -----------------------------------------------------------------
    mega_setting_name_if_generated = None
    if "generate environment" in user_lower or "mega setting" in user_lower:
        # Example function call that merges random settings from DB
        mega_setting_name_if_generated = generate_mega_setting_logic()
        # e.g. "All-Girls College + Space Station"
    
    # (If you introduced a meltdown_newly_triggered somewhere, track it)
    meltdown_newly_triggered = False  # or compute from stat thresholds

    # Suppose we also track removed NPC IDs and new NPC
    removed_npcs_list = []
    new_npc_data = None  # depends on your logic

    # -----------------------------------------------------------------
    # 3) Fetch the aggregated context from DB
    # -----------------------------------------------------------------
    aggregator_data = get_aggregated_roleplay_context(player_name)
    # aggregator_data is typically a dict:
    # {
    #   "playerStats": {...},
    #   "npcStats": [...],
    #   "currentRoleplay": {...}
    # }

    # -----------------------------------------------------------------
    # 4) (Optional) If meltdown NPCs exist, add meltdown flavor
    # -----------------------------------------------------------------
    meltdown_flavor = check_for_meltdown_flavor()

    # -----------------------------------------------------------------
    # 5) Convert aggregator_data into final "story_output" text
    # -----------------------------------------------------------------
    story_output = build_aggregator_text(aggregator_data, meltdown_flavor)

    # -----------------------------------------------------------------
    # 6) Build the updates object
    # -----------------------------------------------------------------
    # You can pick any keys you want to return
    # For demonstration:
    changed_stats = {"obedience": 100} if "obedience=100" in user_lower else {}

    updates_dict = {
        "meltdown_triggered": meltdown_newly_triggered,
        "meltdown_removed": meltdown_forced_removal,
        "new_mega_setting": mega_setting_name_if_generated,
        "updated_player_stats": changed_stats,
        "removed_npc_ids": removed_npcs_list,
        "added_npc": new_npc_data,
        "plot_event": None,  # or "Some meltdown event"
    }

    # -----------------------------------------------------------------
    # 7) Return final scenario text plus updates
    # -----------------------------------------------------------------
    return jsonify({
        "story_output": story_output,
        "updates": updates_dict
    }), 200


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


def generate_mega_setting_logic():
    """
    Example stub that calls your /settings/generate_mega_setting route internally
    or directly runs the code to create a 'mega setting.'
    
    Return the newly created environment name or summary.
    """
    # If you have a direct function from `routes.settings_routes`:
    #   response = generate_mega_setting_route()  # typically returns JSON
    #   ...
    #   return response["mega_name"]

    # Or if it's purely internal logic, do something like:
    #   new_env = "All-Girls College + Space Station"
    #   store it in CurrentRoleplay ...
    #   return new_env
    return "All-Girls College + Space Station"  # placeholder


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
    # Optionally record it:
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

    # Build something plain-text or structured; your call.
    # Below is a straightforward, readable example:
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
    else:
        lines.append("(No NPCs found)")

    # 3) Current Roleplay Data
    lines.append("\n=== CURRENT ROLEPLAY ===")
    if current_rp:
        for k, v in current_rp.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append("(No current roleplay data)")

    # 4) If meltdown flavor is present, append it
    if meltdown_flavor:
        lines.append("\n=== MELTDOWN NPC MESSAGE ===")
        lines.append(meltdown_flavor)

    # 5) Final joined text
    final_text = "\n".join(lines)
    return final_text
