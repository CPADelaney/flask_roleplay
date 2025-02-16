#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete module: routes/story_routes.py
"""

import logging
import json
import os
import asyncio
from datetime import datetime, date
from flask import Blueprint, request, jsonify, session

# Import your DB and logic modules
from db.connection import get_db_connection
from logic.universal_updater import apply_universal_updates  # async function
from logic.npc_creation import spawn_multiple_npcs, spawn_single_npc
from logic.aggregator import get_aggregated_roleplay_context
from logic.time_cycle import advance_time_and_update
from logic.inventory_logic import add_item_to_inventory, remove_item_from_inventory
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client, build_message_history
from routes.settings_routes import generate_mega_setting_logic
from logic.activities_logic import filter_activities_for_npc, build_short_summary

story_bp = Blueprint("story_bp", __name__)

# -------------------------------------------------------------------
# FUNCTION SCHEMAS (for ChatGPT function calls)
# -------------------------------------------------------------------
FUNCTION_SCHEMAS = [
    {
        "name": "get_npc_details",
        "description": "Retrieve full or partial NPC info from NPCStats by npc_id.",
        "parameters": {
            "type": "object",
            "properties": {"npc_id": {"type": "number"}},
            "required": ["npc_id"]
        }
    },
    {
        "name": "get_quest_details",
        "description": "Retrieve quest info from the Quests table by quest_id.",
        "parameters": {
            "type": "object",
            "properties": {"quest_id": {"type": "number"}},
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
            "properties": {"event_id": {"type": "number"}},
            "required": ["event_id"]
        }
    },
    {
        "name": "get_inventory_item",
        "description": "Lookup a specific item in the player's inventory by item_name.",
        "parameters": {
            "type": "object",
            "properties": {"item_name": {"type": "string"}},
            "required": ["item_name"]
        }
    },
    {
        "name": "get_intensity_tiers",
        "description": "Retrieve the entire IntensityTiers data (key features, etc.).",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_plot_triggers",
        "description": "Retrieve the entire list of PlotTriggers (with stage_name, description, etc.).",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_interactions",
        "description": "Retrieve all Interactions from the Interactions table (detailed_rules, etc.).",
        "parameters": {"type": "object", "properties": {}}
    }
]

# -------------------------------------------------------------------
# HELPER FUNCTIONS (synchronous versions remain; if needed, consider wrapping in asyncio.to_thread)
# -------------------------------------------------------------------

def fetch_npc_details(user_id, conversation_id, npc_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT npc_name, introduced, dominance, cruelty, closeness,
               trust, respect, intensity, affiliations, schedule, current_location
        FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        LIMIT 1
    """, (user_id, conversation_id, npc_id))
    row = cursor.fetchone()
    if not row:
        cursor.close()
        conn.close()
        return {"error": f"No NPC found with npc_id={npc_id}"}
    (nname, intro, dom, cru, clos, tru, resp, inten, affil, sched, cloc) = row
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
        "affiliations": affil if affil is not None else [],
        "schedule": sched if sched is not None else {},
        "current_location": cloc
    }
    cursor.close()
    conn.close()
    return npc_data

def fetch_quest_details(user_id, conversation_id, quest_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT quest_name, status, progress_detail, quest_giver, reward
        FROM Quests
        WHERE user_id=%s AND conversation_id=%s AND quest_id=%s
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
    conn = get_db_connection()
    cursor = conn.cursor()
    if location_id:
        cursor.execute("""
            SELECT location_name, description, open_hours
            FROM Locations
            WHERE user_id=%s AND conversation_id=%s AND id=%s
            LIMIT 1
        """, (user_id, conversation_id, location_id))
    elif location_name:
        cursor.execute("""
            SELECT location_name, description, open_hours
            FROM Locations
            WHERE user_id=%s AND conversation_id=%s AND location_name=%s
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
    loc_data = {
        "location_name": lname,
        "description": ldesc,
        "open_hours": lhours if lhours is not None else []
    }
    cursor.close()
    conn.close()
    return loc_data

def fetch_event_details(user_id, conversation_id, event_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT event_name, description, start_time, end_time, location, year, month, day, time_of_day
        FROM Events
        WHERE user_id=%s AND conversation_id=%s AND id=%s
        LIMIT 1
    """, (user_id, conversation_id, event_id))
    row = cursor.fetchone()
    if not row:
        cursor.close()
        conn.close()
        return {"error": f"No event found with id={event_id}"}
    (ename, edesc, stime, etime, eloc, eyear, emonth, eday, etod) = row
    event_data = {
        "event_id": event_id,
        "event_name": ename,
        "description": edesc,
        "start_time": stime,
        "end_time": etime,
        "location": eloc,
        "year": eyear,
        "month": emonth,
        "day": eday,
        "time_of_day": etod
    }
    cursor.close()
    conn.close()
    return event_data

def fetch_inventory_item(user_id, conversation_id, item_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT player_name, item_description, item_effect, quantity, category
        FROM PlayerInventory
        WHERE user_id=%s AND conversation_id=%s AND item_name=%s
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
        all_tiers.append({
            "tier_name": tname,
            "key_features": kfeat if kfeat is not None else [],
            "activity_examples": aex if aex is not None else [],
            "permanent_effects": peff if peff is not None else {}
        })
    cursor.close()
    conn.close()
    return all_tiers

def fetch_plot_triggers():
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
        triggers.append({
            "title": tname,
            "stage": stg,
            "description": desc,
            "key_features": kfeat if kfeat is not None else [],
            "stat_dynamics": sdyn if sdyn is not None else [],
            "examples": ex if ex is not None else [],
            "triggers": trigz if trigz is not None else {}
        })
    cursor.close()
    conn.close()
    return triggers

def fetch_interactions():
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
        result.append({
            "interaction_name": iname,
            "detailed_rules": drules if drules is not None else {},
            "task_examples": texamples if texamples is not None else {},
            "agency_overrides": aover if aover is not None else {}
        })
    cursor.close()
    conn.close()
    return result

# -------------------------------------------------------------------
# ROUTE DEFINITION
# -------------------------------------------------------------------

@story_bp.route("/next_storybeat", methods=["POST"])
async def next_storybeat():
    try:
        logging.info("Starting next_storybeat")
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        data = request.get_json() or {}
        user_input = data.get("user_input", "").strip()
        conv_id = data.get("conversation_id")
        player_name = data.get("player_name", "Chase")

        if not user_input:
            return jsonify({"error": "No user_input provided"}), 400

        # Get a DB connection in a thread to avoid blocking the event loop.
        conn = await asyncio.to_thread(get_db_connection)
        cur = conn.cursor()

        # 1) Create (or validate) a conversation.
        if not conv_id:
            cur.execute(
                "INSERT INTO conversations (user_id, conversation_name) VALUES (%s, %s) RETURNING id",
                (user_id, "New Chat")
            )
            conv_id = cur.fetchone()[0]
            conn.commit()
        else:
            cur.execute("SELECT user_id FROM conversations WHERE id=%s", (conv_id,))
            row = cur.fetchone()
            if not row:
                conn.close()
                return jsonify({"error": f"Conversation {conv_id} not found"}), 404
            if row[0] != user_id:
                conn.close()
                return jsonify({"error": f"Conversation {conv_id} not owned by this user"}), 403

        # 1.5) Check unintroduced NPC count; if fewer than 2, spawn 3 more.
        cur.execute("""
            SELECT COUNT(*) FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND introduced=FALSE
        """, (user_id, conv_id))
        count = cur.fetchone()[0]
        aggregator_data = get_aggregated_roleplay_context(user_id, conv_id, player_name)
        env_desc = aggregator_data.get("currentRoleplay", {}).get("EnvironmentDesc", "A default environment description.")
        calendar = aggregator_data.get("calendar", {})
        day_names = calendar.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        if count < 2:
            logging.info("Only %d unintroduced NPC(s) found; generating 3 more.", count)
            await asyncio.to_thread(spawn_multiple_npcs, user_id, conv_id, env_desc, day_names, count=3)

        # 2) Insert the user message.
        cur.execute(
            "INSERT INTO messages (conversation_id, sender, content) VALUES (%s, %s, %s)",
            (conv_id, "user", user_input)
        )
        conn.commit()

        # 3) (Optional) Create a dominant NPC (e.g., Nyx) – commented out.
        # nyx_npc_id = await asyncio.to_thread(spawn_single_npc, user_id, conv_id, env_desc, day_names)
        # cur.execute("UPDATE NPCStats SET npc_name=%s WHERE npc_id=%s", ("Nyx", nyx_npc_id))
        # conn.commit()

        # 4) Run universal updates if provided.
        universal_data = data.get("universal_update", {})
        if universal_data:
            universal_data["user_id"] = user_id
            universal_data["conversation_id"] = conv_id

            async def run_univ_update():
                import asyncpg
                dsn = os.getenv("DB_DSN")
                async_conn = await asyncpg.connect(dsn=dsn)
                result = await apply_universal_updates(user_id, conv_id, universal_data, async_conn)
                await async_conn.close()
                return result

            update_result = await run_univ_update()
            if "error" in update_result:
                cur.close()
                conn.close()
                return jsonify(update_result), 500

        # 5) Possibly advance time and rebuild aggregator context.
        if data.get("advance_time", False):
            new_year, new_month, new_day, new_phase = advance_time_and_update(user_id, conv_id, increment=1)
            aggregator_data = get_aggregated_roleplay_context(user_id, conv_id, player_name)
        else:
            new_year = aggregator_data.get("year", 1)
            new_month = aggregator_data.get("month", 1)
            new_day = aggregator_data.get("day", 1)
            new_phase = aggregator_data.get("timeOfDay", "Morning")

        aggregator_text = aggregator_data.get("aggregator_text", "No aggregator text available.")

        # Append additional NPC context.
        cur.execute("""
            SELECT npc_name, memory, archetype_extras_summary
            FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND introduced=TRUE
        """, (user_id, conv_id))
        npc_context_rows = cur.fetchall()
        npc_context_summary = ""
        for row in npc_context_rows:
            npc_name, memory_json, extras_summary = row
            mem_text = ""
            if memory_json:
                try:
                    mem_list = json.loads(memory_json)
                    mem_text = " ".join(mem_list)
                except Exception:
                    mem_text = str(memory_json)
            extra_text = extras_summary if extras_summary else ""
            npc_context_summary += f"{npc_name}: {mem_text} {extra_text}\n"
        if npc_context_summary:
            aggregator_text += "\nNPC Context:\n" + npc_context_summary
        logging.debug("Aggregator text prepared: %s", aggregator_text)

        # 6) Attempt up to 3 ChatGPT function calls with a timeout.
        final_text = None
        structured_json_str = None

        for attempt in range(3):
            logging.debug("GPT attempt #%d with user_input: %r", attempt, user_input)
            try:
                gpt_reply_dict = await asyncio.wait_for(
                    asyncio.to_thread(get_chatgpt_response, conv_id, aggregator_text, user_input),
                    timeout=30
                )
            except asyncio.TimeoutError:
                logging.error("ChatGPT call timed out on attempt %d", attempt)
                continue

            logging.debug("GPT reply received on attempt %d: %s", attempt, gpt_reply_dict)

            if gpt_reply_dict.get("type") == "function_call":
                fn_name = gpt_reply_dict.get("function_name")
                fn_args = gpt_reply_dict.get("function_args", {})

                # Execute the requested function locally.
                if fn_name == "get_npc_details":
                    data_out = fetch_npc_details(user_id, conv_id, fn_args.get("npc_id"))
                elif fn_name == "get_quest_details":
                    data_out = fetch_quest_details(user_id, conv_id, fn_args.get("quest_id"))
                elif fn_name == "get_location_details":
                    data_out = fetch_location_details(
                        user_id, conv_id,
                        fn_args.get("location_id"),
                        fn_args.get("location_name")
                    )
                else:
                    data_out = {"error": f"Function call '{fn_name}' not recognized."}

                function_response_text = json.dumps(data_out)
                logging.debug("Function response for %s: %s", fn_name, function_response_text)

                aggregator_text += (
                    f"\n\n[Function Response Received for {fn_name}: {function_response_text}]\n"
                    "Please use the above function response and the previous context to generate a final narrative response. "
                    "Do not issue further function calls."
                )
                await asyncio.sleep(1)
                continue
            else:
                final_text = gpt_reply_dict.get("response")
                structured_json_str = json.dumps(gpt_reply_dict)
                break

        if not final_text:
            final_text = "[No final text returned after repeated function calls]"
            structured_json_str = json.dumps({"attempts": "exceeded"})

        # 7) Store GPT's final message as a new message.
        cur.execute(
            """
            INSERT INTO messages (conversation_id, sender, content, structured_content)
            VALUES (%s, %s, %s, %s)
            """,
            (conv_id, "assistant", final_text, structured_json_str)
        )
        conn.commit()
        cur.close()
        conn.close()

        logging.info("Returning final response.")
        return jsonify({
            "response": final_text,
            "aggregator_text": aggregator_text,
            "conversation_id": conv_id,
            "updates": {
                "year": new_year,
                "month": new_month,
                "day": new_day,
                "time_of_day": new_phase
            }
        })

    except Exception as e:
        logging.exception("Error in next_storybeat")
        return jsonify({"error": str(e)}), 500
        
