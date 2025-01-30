import logging
from flask import Blueprint, request, jsonify, session
from db.connection import get_db_connection
from logic.universal_updater import apply_universal_updates
from logic.aggregator import get_aggregated_roleplay_context
from logic.time_cycle import advance_time_and_update
from logic.activities_logic import filter_activities_for_npc, build_short_summary
# from logic.stats_logic import update_player_stats  # if needed for direct stat updates
from routes.settings_routes import generate_mega_setting_logic
from logic.inventory_logic import add_item_to_inventory, remove_item_from_inventory, get_player_inventory
from logic.chatgpt_integration import get_chatgpt_response

story_bp = Blueprint("story_bp", __name__)

@story_bp.route("/next_storybeat", methods=["POST"])
def next_storybeat():
    """
    Handles the main story logic, now fully scoped by user_id + conversation_id:
    1) Validates user login
    2) Creates or reuses conversation row (belongs to user_id)
    3) Stores user's message
    4) Applies universal_update
    5) aggregator_data -> used by GPT
    6) Optionally generate environment or set obedience=100
    7) Optionally advance time
    8) Calls GPT with aggregator_text + user_input
    9) Stores GPT reply
    10) Returns entire conversation + GPT output
    """
    try:
        # 1) Ensure user is logged in
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        # 2) Parse JSON
        data = request.get_json() or {}
        user_input = data.get("user_input", "").strip()
        conv_id = data.get("conversation_id")  # might be None if new
        player_name = data.get("player_name", "Chase")

        if not user_input:
            return jsonify({"error": "No user_input provided"}), 400

        # Acquire DB connection
        conn = get_db_connection()
        cur = conn.cursor()

        # 3) If no conversation_id, create one for this user
        if not conv_id:
            # Create new conversation
            cur.execute("""
                INSERT INTO conversations (user_id, conversation_name)
                VALUES (%s, %s)
                RETURNING id
            """, (user_id, "New Chat"))
            conv_id = cur.fetchone()[0]
            conn.commit()
        else:
            # Check that conversation belongs to user
            cur.execute("SELECT user_id FROM conversations WHERE id=%s", (conv_id,))
            row = cur.fetchone()
            if not row:
                conn.close()
                return jsonify({"error": f"Conversation {conv_id} not found"}), 404
            if row[0] != user_id:
                conn.close()
                return jsonify({"error": f"Conversation {conv_id} not owned by this user"}), 403

        # 4) Store the user's message in DB
        cur.execute("""
            INSERT INTO messages (conversation_id, sender, content)
            VALUES (%s, %s, %s)
        """, (conv_id, "user", user_input))
        conn.commit()

        logging.info(f"[next_storybeat] user_id={user_id}, conv_id={conv_id}, player_name={player_name}, input={user_input}")

        # 5) If there's universal_update data, apply it
        universal_data = data.get("universal_update", {})
        if universal_data:
            # Attach user_id & conversation_id so universal_updater can scope
            universal_data["user_id"] = user_id
            universal_data["conversation_id"] = conv_id
            logging.info("[next_storybeat] Applying universal update from payload.")
            update_result = apply_universal_updates(universal_data)
            if "error" in update_result:
                cur.close()
                conn.close()
                return jsonify(update_result), 500
        else:
            logging.info("[next_storybeat] No universal update data found.")

        # 6) Check triggers in user_input
        user_lower = user_input.lower()
        if "obedience=100" in user_lower:
            logging.info("[next_storybeat] Forcing obedience=100 for player.")
            force_obedience_to_100(user_id, conv_id, player_name)

        # Possibly meltdown logic, environment generation triggers, etc.

        # 7) aggregator_data => from aggregator scoping
        aggregator_data = get_aggregated_roleplay_context(user_id, conv_id)
        logging.info(f"[next_storybeat] aggregator_data keys: {aggregator_data.keys()}")

        # Possibly generate new environment
        mega_setting_name_if_generated = None
        current_setting = aggregator_data["currentRoleplay"].get("CurrentSetting")
        if "generate environment" in user_lower or "mega setting" in user_lower:
            if current_setting and "force" not in user_lower:
                logging.info(f"[next_storybeat] Already have environment '{current_setting}', skipping generation.")
            else:
                logging.info("[next_storybeat] Generating new mega setting.")
                # We'll call user-scoped version of generate_mega_setting_logic if you prefer
                # Or if Settings is global, you just call generate_mega_setting_logic() with no scoping
                mega_data = generate_mega_setting_logic(user_id, conv_id)
                mega_setting_name_if_generated = mega_data.get("mega_name", "No environment")
                logging.info(f"[next_storybeat] New Mega Setting: {mega_setting_name_if_generated}")

        # 8) Suggest possible NPC activities
        npc_list = aggregator_data.get("npcStats", [])
        user_stats = aggregator_data.get("playerStats", {})
        setting_str = current_setting if current_setting else ""
        npc_archetypes = []

        if npc_list:
            # Example: detect "giant" in first NPC name
            if "giant" in npc_list[0].get("npc_name","").lower():
                npc_archetypes = ["Giantess"]

        chosen_activities = filter_activities_for_npc(
            npc_archetypes=npc_archetypes,
            user_stats=user_stats,
            setting=setting_str
        )
        lines_for_gpt = [build_short_summary(act) for act in chosen_activities]
        aggregator_data["activitySuggestions"] = lines_for_gpt

        # 9) Track changes
        changed_stats = {"obedience": 100} if "obedience=100" in user_lower else {}
        updates_dict = {
            "new_mega_setting": mega_setting_name_if_generated,
            "updated_player_stats": changed_stats,
            "removed_npc_ids": [],
            "added_npc": None,
            "plot_event": None
        }
        logging.info(f"[next_storybeat] updates_dict = {updates_dict}")

        # 10) Possibly advance time
        if data.get("advance_time", False):
            logging.info("[next_storybeat] Advancing time by 1 block.")
            new_day, new_phase = advance_time_and_update(user_id, conv_id, increment=1)
        else:
            logging.info("[next_storybeat] Not advancing time.")
            new_day = aggregator_data.get("day", 1)
            new_phase = aggregator_data.get("timeOfDay", "Morning")

        # 11) Re-fetch aggregator for final updated state
        aggregator_data = get_aggregated_roleplay_context(user_id, conv_id, player_name)

        # Build aggregator text from aggregator_data
        aggregator_text = build_aggregator_text(aggregator_data)

        combined_input = f"{aggregator_text}\n\nUser Input: {user_input}"

        # 12) Call GPT
        gpt_reply = get_chatgpt_response(combined_input, model="gpt-4")

        # 13) Store GPT message
        cur.execute("""
            INSERT INTO messages (conversation_id, sender, content)
            VALUES (%s, %s, %s)
        """, (conv_id, "GPT", gpt_reply))
        conn.commit()

        # 14) Gather full conversation
        cur.execute("""
            SELECT sender, content, created_at
            FROM messages
            WHERE conversation_id=%s
            ORDER BY id ASC
        """, (conv_id,))
        rows = cur.fetchall()
        conversation_history = []
        for r in rows:
            conversation_history.append({
                "sender": r[0],
                "content": r[1],
                "created_at": r[2].isoformat()
            })

        cur.close()
        conn.close()

        # 15) Return final
        return jsonify({
            "conversation_id": conv_id,
            "story_output": gpt_reply,
            "messages": conversation_history,
            "updates": {
                "current_day": new_day,
                "time_of_day": new_phase,
                "internal_changes": updates_dict
            }
        }), 200

    except Exception as e:
        logging.exception("[next_storybeat] Error")
        return jsonify({"error": str(e)}), 500


def force_obedience_to_100(user_id, conversation_id, player_name):
    """
    A direct approach to set player's Obedience=100 
    for user_id + conversation_id + player_name.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE PlayerStats
            SET obedience=100
            WHERE user_id=%s
              AND conversation_id=%s
              AND player_name=%s
        """, (user_id, conversation_id, player_name))
        conn.commit()
    except:
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


def build_aggregator_text(aggregator_data):
    """
    Merges aggregator_data into user-friendly text for GPT.
    The rest is your existing logic, unchanged except no scoping needed here.
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
