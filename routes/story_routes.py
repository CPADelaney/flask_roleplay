import logging
from flask import Blueprint, request, jsonify, session
from db.connection import get_db_connection
from logic.universal_updater import apply_universal_updates
from logic.aggregator import get_aggregated_roleplay_context
from logic.time_cycle import advance_time_and_update

# Additional imports from your snippet
from logic.activities_logic import filter_activities_for_npc, build_short_summary
from logic.stats_logic import update_player_stats
# from logic.meltdown_logic import check_and_inject_meltdown  # if/when needed
# from routes.meltdown import remove_meltdown_npc             # if/when needed
from routes.settings_routes import generate_mega_setting_logic
from logic.inventory_logic import add_item_to_inventory, remove_item_from_inventory, get_player_inventory
from logic.chatgpt_integration import get_chatgpt_response

story_bp = Blueprint("story_bp", __name__)

@story_bp.route("/next_storybeat", methods=["POST"])
def next_storybeat():
    """
    Handles the main story logic:
    - Validates user login
    - Creates/updates conversation, stores user message
    - Applies universal updates
    - Checks triggers (obedience=100, meltdown, environment generation, etc.)
    - Fetches aggregator data, suggests activities
    - Calls GPT with aggregator text + user input
    - Stores GPT reply in conversation
    - Optionally advances time
    - Returns entire conversation plus GPT response
    """
    try:
        # 1) Ensure the user is logged in
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        # 2) Parse incoming JSON
        data = request.get_json() or {}
        user_input = data.get("user_input", "").strip()
        conv_id = data.get("conversation_id")  # might be None if new
        player_name = data.get("player_name", "Unknown")

        if not user_input:
            return jsonify({"error": "No user_input provided"}), 400

        # 3) Create or reuse conversation
        conn = get_db_connection()
        cur = conn.cursor()

        if not conv_id:
            # Create new conversation for this user
            cur.execute("""
                INSERT INTO conversations (user_id, conversation_name)
                VALUES (%s, %s) RETURNING id
            """, (user_id, "New Chat"))
            conv_id = cur.fetchone()[0]
            conn.commit()

        # 4) Store the user’s message in DB
        cur.execute("""
            INSERT INTO messages (conversation_id, sender, content)
            VALUES (%s, %s, %s)
        """, (conv_id, "user", user_input))
        conn.commit()

        logging.info(f"Player: {player_name}, User Input: {user_input}")

        # 5) Apply universal update if provided
        universal_data = data.get("universal_update", {})
        if universal_data:
            logging.info("Applying universal update from payload.")
            update_result = apply_universal_updates(universal_data)
            if "error" in update_result:
                cur.close()
                conn.close()
                return jsonify(update_result), 500
        else:
            logging.info("No universal update data found.")

        # 6) Handle user input triggers
        user_lower = user_input.lower()
        if "obedience=100" in user_lower:
            logging.info("Forcing obedience=100 for player.")
            force_obedience_to_100(player_name)

        # Example meltdown logic (if needed):
        # meltdown_triggered = check_and_inject_meltdown(player_name, user_lower)
        # if meltdown_triggered:
        #     # Possibly remove meltdown NPC or do other meltdown steps
        #     remove_meltdown_npc(...)

        # 7) Fetch aggregator data to reflect any new changes
        aggregator_data = get_aggregated_roleplay_context(player_name)
        logging.info(f"Aggregator Data: {aggregator_data}")

        # Possibly generate a mega setting if forced or missing
        mega_setting_name_if_generated = None
        current_setting = aggregator_data["currentRoleplay"].get("CurrentSetting")
        if ("generate environment" in user_lower or "mega setting" in user_lower):
            if current_setting and "force" not in user_lower:
                logging.info(f"Already have environment '{current_setting}', skipping new generation.")
            else:
                logging.info("Generating new mega setting.")
                mega_data = generate_mega_setting_logic()
                mega_setting_name_if_generated = mega_data.get("mega_name", "No environment")
                logging.info(f"New Mega Setting: {mega_setting_name_if_generated}")

        # 8) Possibly suggest NPC activities
        removed_npcs_list = []
        new_npc_data = None
        setting_str = current_setting if current_setting else ""
        npc_list = aggregator_data.get("npcStats", [])
        npc_archetypes = []

        if npc_list:
            # Example: detect "giant" in the first NPC name
            if "giant" in npc_list[0].get("npc_name", "").lower():
                npc_archetypes = ["Giantess"]

        user_stats = aggregator_data.get("playerStats", {})
        chosen_activities = filter_activities_for_npc(
            npc_archetypes=npc_archetypes,
            user_stats=user_stats,
            setting=setting_str
        )
        lines_for_gpt = [build_short_summary(act) for act in chosen_activities]
        aggregator_data["activitySuggestions"] = lines_for_gpt

        # 9) Track internal changes or updates for reference
        changed_stats = {"obedience": 100} if "obedience=100" in user_lower else {}
        updates_dict = {
            "new_mega_setting": mega_setting_name_if_generated,
            "updated_player_stats": changed_stats,
            "removed_npc_ids": removed_npcs_list,
            "added_npc": new_npc_data,
            "plot_event": None
        }
        logging.info(f"Updates Dict: {updates_dict}")

        # 10) Optionally advance time
        if data.get("advance_time", False):
            logging.info("Advancing time by 1 block.")
            new_day, new_phase = advance_time_and_update(increment=1)
        else:
            logging.info("Not advancing time.")
            new_day = aggregator_data.get("day", 1)
            new_phase = aggregator_data.get("timeOfDay", "Morning")

        # 11) Re-fetch aggregator to get final updated state (optional)
        aggregator_data = get_aggregated_roleplay_context(player_name)

        # Build aggregator text from final aggregator_data
        aggregator_text = build_aggregator_text(aggregator_data)

        # Combine aggregator text with user input to feed GPT
        # This ensures GPT sees the entire updated world context plus the new user command
        combined_input = f"{aggregator_text}\n\nUser Input: {user_input}"

        # 12) Call GPT
        gpt_reply = get_chatgpt_response(combined_input, model="gpt-4")

        # 13) Store GPT's message
        cur.execute("""
            INSERT INTO messages (conversation_id, sender, content)
            VALUES (%s, %s, %s)
        """, (conv_id, "GPT", gpt_reply))
        conn.commit()

        # 14) Gather full conversation for response
        cur.execute("""
            SELECT sender, content, created_at
            FROM messages
            WHERE conversation_id = %s
            ORDER BY id ASC
        """, (conv_id,))
        rows = cur.fetchall()
        conversation_history = []
        for row in rows:
            conversation_history.append({
                "sender": row[0],
                "content": row[1],
                "created_at": row[2].isoformat()
            })

        cur.close()
        conn.close()

        # 15) Return final story output (GPT reply), conversation ID, entire messages, plus any data you want
        return jsonify({
            "conversation_id": conv_id,
            "story_output": gpt_reply,  # The main GPT text
            "messages": conversation_history,
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
        cursor.close()
        conn.close()


def build_aggregator_text(aggregator_data):
    """
    Merges aggregator_data into user-friendly text for GPT.
    Expand or rewrite to suit your aggregator structure.
    """
    lines = []
    day = aggregator_data.get("day", 1)
    tod = aggregator_data.get("timeOfDay", "Morning")
    player_stats = aggregator_data.get("playerStats", {})
    npc_stats = aggregator_data.get("npcStats", [])
    current_rp = aggregator_data.get("currentRoleplay", {})

    lines.append(f"=== DAY {day}, {tod.upper()} ===")

    # Player stats
    if player_stats:
        lines.append("=== PLAYER STATS ===")
        lines.append(
            f"Name: {player_stats.get('name','Unknown')}, "
            f"Corruption: {player_stats.get('corruption',0)}, "
            f"Confidence: {player_stats.get('confidence',0)}, "
            f"Willpower: {player_stats.get('willpower',0)}, "
            f"Obedience: {player_stats.get('obedience',0)}, "
            f"Dependency: {player_stats.get('dependency',0)}, "
            f"Lust: {player_stats.get('lust',0)}, "
            f"MentalResilience: {player_stats.get('mental_resilience',0)}, "
            f"PhysicalEndurance: {player_stats.get('physical_endurance',0)}"
        )
    else:
        lines.append("No player stats found.")

    # NPC stats
    lines.append("\n=== NPC STATS ===")
    if npc_stats:
        for npc in npc_stats:
            stats_line = (
                f"NPC: {npc.get('npc_name','Unnamed')} | "
                f"Dom={npc.get('dominance',0)}, Cru={npc.get('cruelty',0)}, "
                f"Close={npc.get('closeness',0)}, Trust={npc.get('trust',0)}, "
                f"Respect={npc.get('respect',0)}, Int={npc.get('intensity',0)}"
            )
            lines.append(stats_line)
            
            hobbies = npc.get("hobbies", [])
            personality = npc.get("personality_traits", [])
            likes = npc.get("likes", [])
            dislikes = npc.get("dislikes", [])
            
            lines.append(f"Hobbies: {', '.join(hobbies)}" if hobbies else "Hobbies: None")
            lines.append(f"Personality: {', '.join(personality)}" if personality else "Personality: None")
            lines.append(f"Likes: {', '.join(likes)} | Dislikes: {', '.join(dislikes)}\n")
    else:
        lines.append("(No NPCs found)")

    # Current roleplay
    lines.append("\n=== CURRENT ROLEPLAY ===")
    if current_rp:
        for k, v in current_rp.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append("(No current roleplay data)")

    # Potential activities
    if "activitySuggestions" in aggregator_data:
        lines.append("\n=== NPC POTENTIAL ACTIVITIES ===")
        for suggestion in aggregator_data["activitySuggestions"]:
            lines.append(f"- {suggestion}")
        lines.append("NPCs can adopt, combine, or ignore these ideas.\n")

    return "\n".join(lines)
