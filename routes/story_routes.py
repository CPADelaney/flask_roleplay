# routes/story_routes.py

import logging
import openai
import os
import json
from db.connection import get_db_connection
from logic.prompts import SYSTEM_PROMPT
from flask import Blueprint, request, jsonify, session
from db.connection import get_db_connection
from logic.universal_updater import apply_universal_updates
from logic.aggregator import get_aggregated_roleplay_context
from logic.time_cycle import advance_time_and_update
from logic.activities_logic import filter_activities_for_npc, build_short_summary
from routes.settings_routes import generate_mega_setting_logic
from logic.inventory_logic import add_item_to_inventory, remove_item_from_inventory
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client, build_message_history

story_bp = Blueprint("story_bp", __name__)

FUNCTION_SCHEMAS = [
    {
        "name": "get_npc_details",
        "description": "Retrieve full or partial NPC info from NPCStats by npc_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "npc_id": {"type": "number"}
            },
            "required": ["npc_id"]
        }
    },
    {
        "name": "get_quest_details",
        "description": "Retrieve quest info from the Quests table by quest_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "quest_id": {"type": "number"}
            },
            "required": ["quest_id"]
        }
    },
    {
        "name": "get_location_details",
        "description": "Retrieve a location’s info by location_id or location_name.",
        "parameters": {
            "type": "object",
            "properties": {
                "location_id": {"type": "number"},
                "location_name": {"type": "string"}
            }
        }
    },
    {
        "name": "get_event_details",
        "description": "Retrieve event info by event_id from the Events table.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {"type": "number"}
            },
            "required": ["event_id"]
        }
    },
    {
        "name": "get_inventory_item",
        "description": "Lookup a specific item in the player's inventory by item_name.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {"type": "string"}
            },
            "required": ["item_name"]
        }
    },
    {
        "name": "get_intensity_tiers",
        "description": "Retrieve the entire IntensityTiers data (key features, etc.).",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_plot_triggers",
        "description": "Retrieve the entire list of PlotTriggers (with stage_name, description, etc.).",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_interactions",
        "description": "Retrieve all Interactions from the Interactions table (detailed_rules, etc.).",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]
def fetch_npc_details(user_id, conversation_id, npc_id):
    """
    Retrieve full or partial NPC info from NPCStats for a given npc_id.
    Adjust the columns or JSON as needed.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Example: Grab select columns from NPCStats
    cursor.execute("""
        SELECT npc_name,
               introduced,
               dominance, cruelty, closeness,
               trust, respect, intensity,
               affiliations, schedule,
               current_location
        FROM NPCStats
        WHERE user_id=%s
          AND conversation_id=%s
          AND npc_id=%s
        LIMIT 1
    """, (user_id, conversation_id, npc_id))
    row = cursor.fetchone()
    
    if not row:
        cursor.close()
        conn.close()
        return {"error": f"No NPC found with npc_id={npc_id}"}
    
    (nname, intro, dom, cru, clos, tru, resp, inten, affil, sched, cloc) = row
    
    # Convert JSONB columns to Python structures if they are stored as JSON
    # e.g., "affiliations" or "schedule"
    if affil is None:
        affil = []
    if sched is None:
        sched = {}
    
    npc_data = {
        "npc_id": npc_id,
        "npc_name": nname,
        "introduced": intro,
        "dominance": dom,
        "cruelty": cru,
        "closeness": clos,
        "trust": tru,
        "respect": resp,
        "intensity": inten,
        "affiliations": affil,
        "schedule": sched,
        "current_location": cloc
    }
    
    cursor.close()
    conn.close()
    
    return npc_data


def fetch_quest_details(user_id, conversation_id, quest_id):
    """
    Retrieve quest info from the Quests table by quest_id.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT quest_name, status, progress_detail, quest_giver, reward
        FROM Quests
        WHERE user_id=%s
          AND conversation_id=%s
          AND quest_id=%s
        LIMIT 1
    """, (user_id, conversation_id, quest_id))
    
    row = cursor.fetchone()
    if not row:
        cursor.close()
        conn.close()
        return {"error": f"No quest found with quest_id={quest_id}"}
    
    (qname, qstatus, qdetail, qgiver, qreward) = row
    quest_data = {
        "quest_id": quest_id,
        "quest_name": qname,
        "status": qstatus,
        "progress_detail": qdetail,
        "quest_giver": qgiver,
        "reward": qreward
    }
    
    cursor.close()
    conn.close()
    
    return quest_data


def fetch_location_details(user_id, conversation_id, location_id=None, location_name=None):
    """
    Retrieve location info by location_id or location_name from the Locations table.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if location_id:
        cursor.execute("""
            SELECT location_name, description, open_hours
            FROM Locations
            WHERE user_id=%s
              AND conversation_id=%s
              AND id=%s
            LIMIT 1
        """, (user_id, conversation_id, location_id))
    elif location_name:
        cursor.execute("""
            SELECT location_name, description, open_hours
            FROM Locations
            WHERE user_id=%s
              AND conversation_id=%s
              AND location_name=%s
            LIMIT 1
        """, (user_id, conversation_id, location_name))
    else:
        return {"error": "No location_id or location_name provided"}
    
    row = cursor.fetchone()
    if not row:
        cursor.close()
        conn.close()
        return {"error": "No matching location found"}
    
    (lname, ldesc, lhours) = row
    if lhours is None:
        lhours = []
    
    loc_data = {
        "location_name": lname,
        "description": ldesc,
        "open_hours": lhours
    }
    
    cursor.close()
    conn.close()
    
    return loc_data


def fetch_event_details(user_id, conversation_id, event_id):
    """
    Retrieve an event's info by event_id from the Events table.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT event_name, description, start_time, end_time, location
        FROM Events
        WHERE user_id=%s
          AND conversation_id=%s
          AND id=%s
        LIMIT 1
    """, (user_id, conversation_id, event_id))
    
    row = cursor.fetchone()
    if not row:
        cursor.close()
        conn.close()
        return {"error": f"No event found with id={event_id}"}
    
    (ename, edesc, stime, etime, eloc) = row
    event_data = {
        "event_id": event_id,
        "event_name": ename,
        "description": edesc,
        "start_time": stime,
        "end_time": etime,
        "location": eloc
    }
    
    cursor.close()
    conn.close()
    
    return event_data


def fetch_inventory_item(user_id, conversation_id, item_name):
    """
    Lookup a specific item in the player's inventory by item_name.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT player_name, item_description, item_effect, quantity, category
        FROM PlayerInventory
        WHERE user_id=%s
          AND conversation_id=%s
          AND item_name=%s
        LIMIT 1
    """, (user_id, conversation_id, item_name))
    
    row = cursor.fetchone()
    if not row:
        cursor.close()
        conn.close()
        return {"error": f"No item named '{item_name}' found in inventory"}
    
    (pname, idesc, ifx, qty, cat) = row
    
    item_data = {
        "item_name": item_name,
        "player_name": pname,
        "item_description": idesc,
        "item_effect": ifx,
        "quantity": qty,
        "category": cat
    }
    
    cursor.close()
    conn.close()
    
    return item_data


def fetch_intensity_tiers():
    """
    If the model calls 'get_intensity_tiers' (which is purely global),
    we can simply fetch them from IntensityTiers. No user_id scoping required.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT tier_name, key_features, activity_examples, permanent_effects
        FROM IntensityTiers
        ORDER BY id
    """)
    rows = cursor.fetchall()
    
    all_tiers = []
    for (tname, kfeat, aex, peff) in rows:
        if kfeat is None:
            kfeat = []
        if aex is None:
            aex = []
        if peff is None:
            peff = {}
        all_tiers.append({
            "tier_name": tname,
            "key_features": kfeat,
            "activity_examples": aex,
            "permanent_effects": peff
        })
    
    cursor.close()
    conn.close()
    return all_tiers


def fetch_plot_triggers():
    """
    If the model calls 'get_plot_triggers',
    we can fetch from PlotTriggers table (also global).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT trigger_name, stage_name, description,
               key_features, stat_dynamics, examples, triggers
        FROM PlotTriggers
        ORDER BY id
    """)
    rows = cursor.fetchall()
    
    triggers = []
    for r in rows:
        (tname, stg, desc, kfeat, sdyn, ex, trigz) = r
        # parse JSON if needed
        if not kfeat: kfeat = []
        if not sdyn: sdyn = []
        if not ex: ex = []
        if not trigz: trigz = {}
        triggers.append({
            "trigger_name": tname,
            "stage_name": stg,
            "description": desc,
            "key_features": kfeat,
            "stat_dynamics": sdyn,
            "examples": ex,
            "triggers": trigz
        })
    
    cursor.close()
    conn.close()
    return triggers


def fetch_interactions():
    """
    If model calls 'get_interactions'.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT interaction_name, detailed_rules, task_examples, agency_overrides
        FROM Interactions
        ORDER BY id
    """)
    rows = cursor.fetchall()
    result = []
    for (iname, drules, texamples, aover) in rows:
        if not drules: drules = {}
        if not texamples: texamples = {}
        if not aover: aover = {}
        result.append({
            "interaction_name": iname,
            "detailed_rules": drules,
            "task_examples": texamples,
            "agency_overrides": aover
        })
    cursor.close()
    conn.close()
    return result

story_bp = Blueprint("story_bp", __name__)

@story_bp.route("/next_storybeat", methods=["POST"])
def next_storybeat():
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        data = request.get_json() or {}
        user_input = data.get("user_input", "").strip()
        conv_id = data.get("conversation_id")
        player_name = data.get("player_name", "Chase")

        if not user_input:
            return jsonify({"error": "No user_input provided"}), 400

        conn = get_db_connection()
        cur = conn.cursor()

        # 1) Possibly create a new conversation
        if not conv_id:
            cur.execute(
                """
                INSERT INTO conversations (user_id, conversation_name)
                VALUES (%s, %s)
                RETURNING id
                """,
                (user_id, "New Chat")
            )
            conv_id = cur.fetchone()[0]
            conn.commit()
        else:
            # Validate conversation ownership
            cur.execute("SELECT user_id FROM conversations WHERE id=%s", (conv_id,))
            row = cur.fetchone()
            if not row:
                conn.close()
                return jsonify({"error": f"Conversation {conv_id} not found"}), 404
            if row[0] != user_id:
                conn.close()
                return jsonify({"error": f"Conversation {conv_id} not owned by this user"}), 403

        # 2) Insert user message
        cur.execute(
            """
            INSERT INTO messages (conversation_id, sender, content)
            VALUES (%s, %s, %s)
            """,
            (conv_id, "user", user_input)
        )
        conn.commit()

        # 3) Possibly apply universal updates if posted in the request
        universal_data = data.get("universal_update", {})
        if universal_data:
            from logic.universal_updater import apply_universal_updates
            universal_data["user_id"] = user_id
            universal_data["conversation_id"] = conv_id
            update_result = apply_universal_updates(universal_data)
            if "error" in update_result:
                cur.close()
                conn.close()
                return jsonify(update_result), 500

        # 4) Possibly advance time
        from logic.time_cycle import advance_time_and_update
        aggregator_data = get_aggregated_roleplay_context(user_id, conv_id, player_name)
        if data.get("advance_time", False):
            new_day, new_phase = advance_time_and_update(user_id, conv_id, increment=1)
            aggregator_data = get_aggregated_roleplay_context(user_id, conv_id, player_name)
        else:
            new_day = aggregator_data.get("day", 1)
            new_phase = aggregator_data.get("timeOfDay", "Morning")

        # -- This uses aggregator_data["aggregator_text"] that your aggregator builds,
        #    which is intentionally short & updated in build_changes_summary() + make_minimal_scene_info().
        #    It includes a rolling "GlobalSummary" plus a short snippet.
        aggregator_text = aggregator_data.get("aggregator_text", "")
        if not aggregator_text:
            aggregator_text = "No aggregator text available."

        logging.debug("[next_storybeat] aggregator_text:\n%s", aggregator_text)

        # 5) Attempt up to 3 function calls from GPT
        final_text = None
        structured_json_str = None

        for attempt in range(3):
            logging.debug("[next_storybeat] GPT attempt #%d with user_input=%r", attempt, user_input)

            # Call GPT
            gpt_reply_dict = get_chatgpt_response(
                conversation_id=conv_id,
                aggregator_text=aggregator_text,
                user_input=user_input
            )
            logging.debug("[next_storybeat] GPT reply (attempt %d): %s", attempt, gpt_reply_dict)

            if gpt_reply_dict["type"] == "function_call":
                fn_name = gpt_reply_dict["function_name"]
                fn_args = gpt_reply_dict["function_args"] or {}

                # Example function calls:
                if fn_name == "get_npc_details":
                    data_out = fetch_npc_details(user_id, conv_id, fn_args["npc_id"])
                    function_msg = {
                        "role": "function",
                        "name": fn_name,
                        "content": json.dumps(data_out)
                    }
                    # Re-invoke GPT
                    gpt_reply_dict = get_chatgpt_response(
                        conversation_id=conv_id,
                        aggregator_text=aggregator_text,
                        user_input=user_input,
                    )
                    logging.debug("[next_storybeat] GPT reply after function '%s': %s", fn_name, gpt_reply_dict)

                    if gpt_reply_dict["type"] == "function_call":
                        # If it calls another function, let loop continue
                        continue
                    else:
                        final_text = gpt_reply_dict["response"]
                        structured_json_str = json.dumps(gpt_reply_dict)
                        break

                elif fn_name == "get_quest_details":
                    data_out = fetch_quest_details(user_id, conv_id, fn_args["quest_id"])
                    function_msg = {
                        "role": "function",
                        "name": fn_name,
                        "content": json.dumps(data_out)
                    }
                    gpt_reply_dict = get_chatgpt_response(
                        conversation_id=conv_id,
                        aggregator_text=aggregator_text,
                        user_input=user_input,
                    )
                    logging.debug("[next_storybeat] GPT reply after function '%s': %s", fn_name, gpt_reply_dict)

                    if gpt_reply_dict["type"] == "function_call":
                        continue
                    else:
                        final_text = gpt_reply_dict["response"]
                        structured_json_str = json.dumps(gpt_reply_dict)
                        break

                elif fn_name == "apply_universal_update":
                    fn_args["user_id"] = user_id
                    fn_args["conversation_id"] = conv_id

                    data_out = apply_universal_updates(fn_args)
                    function_msg = {
                        "role": "function",
                        "name": fn_name,
                        "content": json.dumps(data_out)
                    }
                    gpt_reply_dict = get_chatgpt_response(
                        conversation_id=conv_id,
                        aggregator_text=aggregator_text,
                        user_input=user_input,
                    )
                    logging.debug("[next_storybeat] GPT reply after function '%s': %s", fn_name, gpt_reply_dict)

                    if gpt_reply_dict["type"] == "function_call":
                        continue
                    else:
                        final_text = gpt_reply_dict["response"]
                        structured_json_str = json.dumps(gpt_reply_dict)
                        break

                elif fn_name == "get_location_details":
                    data_out = fetch_location_details(
                        user_id, conv_id,
                        fn_args.get("location_id"),
                        fn_args.get("location_name")
                    )
                    function_msg = {
                        "role": "function",
                        "name": fn_name,
                        "content": json.dumps(data_out)
                    }
                    gpt_reply_dict = get_chatgpt_response(
                        conversation_id=conv_id,
                        aggregator_text=aggregator_text,
                        user_input=user_input,
                    )
                    logging.debug("[next_storybeat] GPT reply after function '%s': %s", fn_name, gpt_reply_dict)

                    if gpt_reply_dict["type"] == "function_call":
                        continue
                    else:
                        final_text = gpt_reply_dict["response"]
                        structured_json_str = json.dumps(gpt_reply_dict)
                        break

                # ... Additional recognized function names ...
                else:
                    final_text = f"Function call '{fn_name}' is not recognized."
                    structured_json_str = json.dumps(gpt_reply_dict)
                    break

            else:
                # GPT returned normal text on the first try
                final_text = gpt_reply_dict["response"]
                structured_json_str = json.dumps(gpt_reply_dict)
                break

        if not final_text:
            final_text = "[No final text returned after repeated function calls]"
            structured_json_str = json.dumps({"attempts": "exceeded"})

        # 6) Store GPT's final text as a message
        cur.execute(
            """
            INSERT INTO messages (conversation_id, sender, content, structured_content)
            VALUES (%s, %s, %s, %s)
            """,
            (conv_id, "Nyx", final_text, structured_json_str)
        )
        conn.commit()

        # 7) Gather entire conversation for return
        cur.execute(
            """
            SELECT sender, content, created_at
            FROM messages
            WHERE conversation_id=%s
            ORDER BY id ASC
            """,
            (conv_id,)
        )
        rows = cur.fetchall()
        conversation_history = [
            {"sender": r[0], "content": r[1], "created_at": r[2].isoformat()}
            for r in rows
        ]

        cur.close()
        conn.close()

        return jsonify({
            "conversation_id": conv_id,
            "story_output": final_text,
            "messages": conversation_history,
            "updates": {
                "current_day": new_day,
                "time_of_day": new_phase
            }
        }), 200

    except Exception as e:
        logging.exception("[next_storybeat] Error")
        return jsonify({"error": str(e)}), 500



def gather_rule_knowledge():
    """
    Fetch or build short text summaries from rule_enforcement.py logic,
    plus data from the PlotTriggers, IntensityTiers, Interactions tables.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) PlotTriggers
    # We'll store each row's data so we can present them later
    # Adjust the column names to match your actual schema
    cursor.execute("""
        SELECT trigger_name, stage_name, description, key_features, stat_dynamics, examples, triggers
        FROM PlotTriggers
    """)
    trig_list = []
    for row in cursor.fetchall():
        (trig_name, stage, desc, kfeat, sdyn, ex, trigz) = row
        # parse JSON if needed
        try:
            kfeat = json.loads(kfeat) if kfeat else []
        except:
            kfeat = []
        try:
            sdyn = json.loads(sdyn) if sdyn else []
        except:
            sdyn = []
        try:
            ex = json.loads(ex) if ex else []
        except:
            ex = []
        try:
            trigz = json.loads(trigz) if trigz else {}
        except:
            trigz = {}

        trig_list.append({
            "title": trig_name,         # e.g. "Early Stage", "Mid-Stage Escalation"
            "stage": stage,
            "description": desc,
            "key_features": kfeat,
            "stat_dynamics": sdyn,
            "examples": ex,
            "triggers": trigz
        })

    # 2) Intensity Tiers
    cursor.execute("""
        SELECT tier_name, key_features, activity_examples, permanent_effects
        FROM IntensityTiers
    """)
    tier_list = []
    for row in cursor.fetchall():
        tname, kfeat, aex, peff = row
        try:
            kfeat = json.loads(kfeat) if kfeat else []
        except:
            kfeat = []
        try:
            aex = json.loads(aex) if aex else []
        except:
            aex = []
        try:
            peff = json.loads(peff) if peff else {}
        except:
            peff = {}

        tier_list.append({
            "tier_name": tname,
            "key_features": kfeat,
            "activity_examples": aex,
            "permanent_effects": peff
        })

    # 3) Interactions
    cursor.execute("""
        SELECT interaction_name, detailed_rules, task_examples, agency_overrides
        FROM Interactions
    """)
    interactions_list = []
    for row in cursor.fetchall():
        iname, drules, tex, aov = row
        try:
            drules = json.loads(drules) if drules else {}
        except:
            drules = {}
        try:
            tex = json.loads(tex) if tex else {}
        except:
            tex = {}
        try:
            aov = json.loads(aov) if aov else {}
        except:
            aov = {}
        interactions_list.append({
            "interaction_name": iname,
            "detailed_rules": drules,
            "task_examples": tex,
            "agency_overrides": aov
        })

    conn.close()

    # Basic textual summary from your doc
    rule_enforcement_summary = (
        "Conditions are parsed (e.g. 'Lust > 90 or Dependency > 80'), "
        "evaluated against stats, and if true, an effect like 'Locks Independent Choices' is applied. "
        "This can raise Obedience, trigger punishments, meltdown synergy, or endgame events."
    )

    # Optionally reintroduce meltdown info
 #   meltdown_info = (
  #      "Meltdowns occur if certain conditions are met. The meltdown logic triggers GPT to produce meltdown dialog, etc."
   # )

    return {
        "rule_enforcement_summary": rule_enforcement_summary,
        "plot_triggers": trig_list,
        "intensity_tiers": tier_list,
        "interactions": interactions_list
 #       "meltdown_synergy": meltdown_info
    }


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


def build_aggregator_text(aggregator_data, rule_knowledge=None):
    """
    Merges aggregator_data into user-friendly text for GPT,
    including events, social links, inventory, etc.
    If rule_knowledge is provided, appends advanced rule data as well.
    """
    lines = []
    # Pull values from aggregator_data with defaults to avoid KeyErrors
    day = aggregator_data.get("day", 1)
    tod = aggregator_data.get("timeOfDay", "Morning")
    player_stats = aggregator_data.get("playerStats", {})
    npc_stats = aggregator_data.get("npcStats", [])
    current_rp = aggregator_data.get("currentRoleplay", {})
    social_links = aggregator_data.get("socialLinks", [])
    player_perks = aggregator_data.get("playerPerks", [])
    inventory = aggregator_data.get("inventory", [])
    events_list = aggregator_data.get("events", [])
    planned_events_list = aggregator_data.get("plannedEvents", [])
    game_rules_list = aggregator_data.get("gameRules", [])
    quests_list = aggregator_data.get("quests", [])
    stat_definitions_list = aggregator_data.get("statDefinitions", [])

    # 1) Day/Time
    lines.append(f"=== DAY {day}, {tod.upper()} ===")

    # 2) Player Stats
    lines.append("\n=== PLAYER STATS ===")
    if player_stats:
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

    # 3) NPC Stats (only introduced NPCs)
    lines.append("\n=== NPC STATS ===")
    introduced_npcs = [npc for npc in npc_stats if npc.get("introduced") is True]
    if introduced_npcs:
        for npc in introduced_npcs:
            lines.append(
                f"NPC: {npc.get('npc_name','Unnamed')} "
                f"| Sex={npc.get('sex','Unknown')} "
                f"| Archetypes={npc.get('archetypes',[])} "
                f"| Dom={npc.get('dominance',0)}, Cru={npc.get('cruelty',0)}, "
                f"Close={npc.get('closeness',0)}, Trust={npc.get('trust',0)}, "
                f"Respect={npc.get('respect',0)}, Int={npc.get('intensity',0)}"
            )

            hobbies = npc.get("hobbies", [])
            personality = npc.get("personality_traits", [])
            likes = npc.get("likes", [])
            dislikes = npc.get("dislikes", [])

            lines.append(f"  Hobbies: {', '.join(hobbies)}" if hobbies else "  Hobbies: None")
            lines.append(f"  Personality: {', '.join(personality)}" if personality else "  Personality: None")
            lines.append(f"  Likes: {', '.join(likes)} | Dislikes: {', '.join(dislikes)}")

            npc_memory = npc.get("memory", [])
            if npc_memory:
                if isinstance(npc_memory, list):
                    lines.append(f"  Memory: {npc_memory}")
                else:
                    lines.append(f"  Memory: {npc_memory}")
            else:
                lines.append("  Memory: (None)")

            affiliations = npc.get("affiliations", [])
            lines.append(f"  Affiliations: {', '.join(affiliations)}" if affiliations else "  Affiliations: None")

            schedule = npc.get("schedule", {})
            if schedule:
                schedule_json = json.dumps(schedule, indent=2)
                lines.append("  Schedule:")
                for line in schedule_json.splitlines():
                    lines.append("    " + line)
            else:
                lines.append("  Schedule: (None)")

            current_loc = npc.get("current_location", "Unknown")
            lines.append(f"  Current Location: {current_loc}\n")
    else:
        lines.append("(No NPCs found)")

    # 4) Current roleplay data
    lines.append("\n=== CURRENT ROLEPLAY ===")
    if current_rp:
        for k, v in current_rp.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append("(No current roleplay data)")

    # 5) Potential activities
    if "activitySuggestions" in aggregator_data:
        lines.append("\n=== NPC POTENTIAL ACTIVITIES ===")
        for suggestion in aggregator_data["activitySuggestions"]:
            lines.append(f"- {suggestion}")
        lines.append("NPCs can adopt, combine, or ignore these ideas.\n")

    # 6) Social Links
    lines.append("\n=== SOCIAL LINKS ===")
    if social_links:
        for link in social_links:
            lines.append(
                f"Link {link['link_id']}: "
                f"{link['entity1_type']}({link['entity1_id']}) <-> {link['entity2_type']}({link['entity2_id']}); "
                f"Type={link['link_type']}, Level={link['link_level']}"
            )
            history = link.get("link_history", [])
            if history:
                lines.append(f"  History: {history}")
    else:
        lines.append("(No social links found)")

    # 7) Player Perks
    lines.append("\n=== PLAYER PERKS ===")
    if player_perks:
        for perk in player_perks:
            lines.append(
                f"Perk: {perk['perk_name']} | Desc: {perk['perk_description']} | Effect: {perk['perk_effect']}"
            )
    else:
        lines.append("(No perks found)")

    # 8) Inventory
    lines.append("\n=== INVENTORY ===")
    if inventory:
        for item in inventory:
            lines.append(
                f"{item['player_name']}'s Item: {item['item_name']} (x{item['quantity']}) - "
                f"{item.get('item_description','No desc')} "
                f"[Effect: {item.get('item_effect','none')}], Category: {item.get('category','misc')}"
            )
    else:
        lines.append("(No inventory items found)")

    # 9) Events
    lines.append("\n=== EVENTS ===")
    if events_list:
        for ev in events_list:
            lines.append(
                f"Event #{ev['event_id']}: {ev['event_name']} @ {ev['location']}, "
                f"{ev['start_time']}-{ev['end_time']} | {ev['description']}"
            )
    else:
        lines.append("(No events found)")

    # 10) Planned Events
    lines.append("\n=== PLANNED EVENTS ===")
    if planned_events_list:
        for pev in planned_events_list:
            lines.append(
                f"PlannedEvent #{pev['event_id']}: NPC {pev['npc_id']} on Day {pev['day']} {pev['time_of_day']} "
                f"@ {pev['override_location']}"
            )
    else:
        lines.append("(No planned events found)")

    # 11) Quests
    lines.append("\n=== QUESTS ===")
    if quests_list:
        for q in quests_list:
            lines.append(
                f"Quest #{q['quest_id']}: {q['quest_name']} [Status: {q['status']}] - {q['progress_detail']}. "
                f"Giver={q['quest_giver']}, Reward={q['reward']}"
            )
    else:
        lines.append("(No quests found)")

    # 12) Game Rules
    lines.append("\n=== GAME RULES ===")
    if game_rules_list:
        for gr in game_rules_list:
            lines.append(
                f"Rule: {gr['rule_name']} => If({gr['condition']}), then({gr['effect']})"
            )
    else:
        lines.append("(No game rules found)")

    # 13) Stat Definitions
    lines.append("\n=== STAT DEFINITIONS ===")
    if stat_definitions_list:
        for sd in stat_definitions_list:
            lines.append(
                f"{sd['stat_name']} [{sd['range_min']}..{sd['range_max']}]: {sd['definition']}; "
                f"Effects={sd['effects']}; Triggers={sd['progression_triggers']}"
            )
    else:
        lines.append("(No stat definitions found)")

    # 14) If you want advanced rule knowledge appended
    if rule_knowledge:
        lines.append("\n=== ADVANCED RULE ENFORCEMENT & KNOWLEDGE ===")

        # Summary
        lines.append("\nRule Enforcement Summary:")
        lines.append(rule_knowledge.get("rule_enforcement_summary", "(No info)"))

        # Plot Triggers
        plot_trigs = rule_knowledge.get("plot_triggers", [])
        if plot_trigs:
            lines.append("\n-- PLOT TRIGGERS --")
            for trig in plot_trigs:
                lines.append(f"Trigger Name: {trig['title']}")
                lines.append(f"Stage: {trig['stage']}")
                lines.append(f"Description: {trig['description']}")
                lines.append(f"Key Features: {trig['key_features']}")
                lines.append(f"Stat Dynamics: {trig['stat_dynamics']}")
                if trig.get("examples"):
                    lines.append(f"  Examples: {json.dumps(trig['examples'])}")
                if trig.get("triggers"):
                    lines.append(f"  Additional Triggers: {json.dumps(trig['triggers'])}")
                lines.append("")  # blank line
        else:
            lines.append("No plot triggers found.")

        # Intensity Tiers
        tiers = rule_knowledge.get("intensity_tiers", [])
        if tiers:
            lines.append("\n-- INTENSITY TIERS --")
            for tier in tiers:
                lines.append(f"{tier['tier_name']}")
                lines.append(f"  Key Features: {json.dumps(tier['key_features'])}")
                lines.append(f"  Activities: {json.dumps(tier['activity_examples'])}")
                lines.append(f"  Permanent Effects: {json.dumps(tier['permanent_effects'])}\n")
        else:
            lines.append("No intensity tiers found.")

        # Interactions
        interactions = rule_knowledge.get("interactions", [])
        if interactions:
            lines.append("\n-- INTERACTIONS --")
            for intr in interactions:
                lines.append(f"Interaction Name: {intr['interaction_name']}")
                lines.append(f"Detailed Rules: {json.dumps(intr['detailed_rules'])}")
                lines.append(f"Task Examples: {json.dumps(intr['task_examples'])}")
                lines.append(f"Agency Overrides: {json.dumps(intr['agency_overrides'])}\n")
        else:
            lines.append("No interactions data found.")

  #      meltdown_info = rule_knowledge.get("meltdown_synergy", None)
  #      if meltdown_info:
  #          lines.append("\nMeltdown Synergy Info:")
   #         lines.append(meltdown_info)

    # Return final string
    return "\n".join(lines)
