# routes/story_routes.py (revised)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete module: routes/story_routes.py with IntegratedNPCSystem integration
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
from logic.npc_creation import spawn_multiple_npcs_enhanced, create_and_refine_npc
from logic.aggregator import get_aggregated_roleplay_context
from logic.time_cycle import get_current_time, should_advance_time, nightly_maintenance
from logic.inventory_logic import add_item_to_inventory, remove_item_from_inventory
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client, build_message_history
from routes.settings_routes import generate_mega_setting_logic
from logic.gpt_image_decision import should_generate_image_for_response
from routes.ai_image_generator import generate_roleplay_image_from_gpt

# Import IntegratedNPCSystem - PRIMARY CHANGE
from logic.fully_integrated_npc_system import IntegratedNPCSystem

# Import new enhanced modules
from logic.stats_logic import get_player_current_tier, check_for_combination_triggers, apply_stat_change, apply_activity_effects

from logic.npc_creation import process_daily_npc_activities, check_for_mask_slippage, detect_relationship_stage_changes
from logic.narrative_progression import get_current_narrative_stage, check_for_personal_revelations, check_for_narrative_moments,check_for_npc_revelations, add_dream_sequence, add_moment_of_clarity
from logic.social_links import get_relationship_dynamic_level, update_relationship_dynamic, check_for_relationship_crossroads, check_for_relationship_ritual, get_relationship_summary, apply_crossroads_choice
from logic.addiction_system import check_addiction_levels, update_addiction_level, process_addiction_effects, get_addiction_status, get_addiction_label

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
        "description": "Retrieve a location's info by location_id or location_name.",
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
    },
    {
        "name": "get_relationship_summary",
        "description": "Retrieve a summary of the relationship between two entities",
        "parameters": {
            "type": "object",
            "properties": {
                "entity1_type": {"type": "string"},
                "entity1_id": {"type": "number"},
                "entity2_type": {"type": "string"},
                "entity2_id": {"type": "number"}
            },
            "required": ["entity1_type", "entity1_id", "entity2_type", "entity2_id"]
        }
    },
    {
        "name": "get_narrative_stage",
        "description": "Retrieve the current narrative stage for a player",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_addiction_status",
        "description": "Retrieve the current addiction status for a player",
        "parameters": {"type": "object", "properties": {}}
    }
]

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------

def fetch_npc_details(user_id, conversation_id, npc_id):
    """
    Retrieve full or partial NPC info from NPCStats for a given npc_id.
    """
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
    """
    Retrieve quest info from the Quests table by quest_id.
    """
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
    """
    Retrieve location info by location_id or location_name from the Locations table.
    """
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
    """
    Retrieve an event's info by event_id from the Events table.
    """
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
    """
    Lookup a specific item in the player's inventory by item_name.
    """
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
    """
    Retrieve the entire IntensityTiers data.
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
    """
    Retrieve the entire list of PlotTriggers.
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
    """
    Retrieve all Interactions from the Interactions table.
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
        result.append({
            "interaction_name": iname,
            "detailed_rules": drules if drules is not None else {},
            "task_examples": texamples if texamples is not None else {},
            "agency_overrides": aover if aover is not None else {}
        })
    cursor.close()
    conn.close()
    return result

# NEW FUNCTION: Get nearby NPCs for interactions
async def get_nearby_npcs(user_id, conversation_id, location=None):
    """
    Get NPCs that are at the current location or nearby
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if location:
        # Get NPCs at the specific location
        cursor.execute("""
            SELECT npc_id, npc_name, current_location, dominance, cruelty
            FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s 
            AND current_location=%s
            LIMIT 5
        """, (user_id, conversation_id, location))
    else:
        # Get any NPCs (prioritize ones that are introduced)
        cursor.execute("""
            SELECT npc_id, npc_name, current_location, dominance, cruelty
            FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s
            ORDER BY introduced DESC
            LIMIT 5
        """, (user_id, conversation_id))
        
    nearby_npcs = []
    for row in cursor.fetchall():
        npc_id, npc_name, current_location, dominance, cruelty = row
        nearby_npcs.append({
            "npc_id": npc_id,
            "npc_name": npc_name,
            "current_location": current_location,
            "dominance": dominance,
            "cruelty": cruelty
        })
    
    cursor.close()
    conn.close()
    
    return nearby_npcs

# -------------------------------------------------------------------
# ROUTE DEFINITION
# -------------------------------------------------------------------

@story_bp.route("/next_storybeat", methods=["POST"])
async def next_storybeat():
    """Improved next_storybeat with better resource cleanup."""
    conn = None
    cur = None
    npc_system = None
    
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        data = request.get_json() or {}
        user_input = data.get("user_input", "").strip()
        conv_id = data.get("conversation_id")
        player_name = data.get("player_name", "Chase")
        requested_activity = data.get("activity_type")

        if not user_input:
            return jsonify({"error": "No user_input provided"}), 400

        # 1) Validate or create conversation with proper resource management
        conn = get_db_connection()
        cur = conn.cursor()
        
        try:
            if not conv_id:
                cur.execute("""
                    INSERT INTO conversations (user_id, conversation_name)
                    VALUES (%s, %s) RETURNING id
                """, (user_id, "New Chat"))
                conv_id = cur.fetchone()[0]
                conn.commit()
            else:
                cur.execute("SELECT user_id FROM conversations WHERE id=%s", (conv_id,))
                row = cur.fetchone()
                if not row:
                    return jsonify({"error": f"Conversation {conv_id} not found"}), 404
                if row[0] != user_id:
                    return jsonify({"error": "Conversation not owned by user"}), 403
        except Exception as db_error:
            # Handle database errors properly
            if conn:
                conn.rollback()
            logger.error(f"Database error: {db_error}")
            return jsonify({"error": f"Database error: {str(db_error)}"}), 500

        # Initialize IntegratedNPCSystem with proper cleanup
        npc_system = IntegratedNPCSystem(user_id, conv_id)

        # 1.5) Possibly spawn more NPCs if too few introduced
        cur.execute("""
            SELECT COUNT(*) FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND introduced=FALSE
        """, (user_id, conv_id))
        unintroduced_count = cur.fetchone()[0]

        aggregator_data = get_aggregated_roleplay_context(user_id, conv_id, player_name)
        env_desc = aggregator_data.get("currentRoleplay", {}).get("EnvironmentDesc", "")
        day_names = aggregator_data.get("calendar", {}).get("days", ["Monday","Tuesday","..."])

        if unintroduced_count < 2:
            # Use IntegratedNPCSystem to create multiple NPCs
            await npc_system.create_multiple_npcs(env_desc, day_names, count=3)

        # 2) Insert user message
        cur.execute("""
            INSERT INTO messages (conversation_id, sender, content)
            VALUES (%s, %s, %s)
        """, (conv_id, "user", user_input))
        conn.commit()

        # 3) Process universal updates
        universal_data = data.get("universal_update")
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

        # 4) NPC Interactions using IntegratedNPCSystem
        context = {
            "location": aggregator_data.get("currentRoleplay", {}).get("CurrentLocation", "Unknown"),
            "time_of_day": aggregator_data.get("timeOfDay", "Morning")
        }
        
        # Determine activity type from user input if not explicitly provided
        activity_type = requested_activity
        if not activity_type:
            # Use process_activity to classify the user input
            activity_result = await npc_system.process_player_activity(user_input, context)
            activity_type = activity_result.get("activity_type", "quick_chat")
        
        # Get nearby NPCs for interaction
        nearby_npcs = await get_nearby_npcs(user_id, conv_id, context["location"])
        
        # Process NPC interactions
        npc_responses = []
        for nearby_npc in nearby_npcs[:3]:  # Limit to 3 NPCs for performance
            interaction_result = await npc_system.handle_npc_interaction(
                npc_id=nearby_npc["npc_id"],
                interaction_type="defiant_response" if "no" in user_input.lower() or "won't" in user_input.lower() else 
                               "submissive_response" if "yes" in user_input.lower() or "okay" in user_input.lower() else
                               "flirtatious_remark" if any(word in user_input.lower() for word in ["cute", "pretty", "hot", "sexy"]) else
                               "extended_conversation",
                player_input=user_input,
                context=context
            )
            
            # Format response for display
            if interaction_result:
                npc_name = nearby_npc["npc_name"]
                response_data = {
                    "npc_id": nearby_npc["npc_id"],
                    "npc_name": npc_name,
                    "action": f"reacts to your {activity_type}",
                    "result": {
                        "outcome": interaction_result.get("events", [{}])[0].get("text", 
                                 f"{npc_name} observes your actions with interest.")
                    },
                    "stat_changes": interaction_result.get("stat_changes", {})
                }
                npc_responses.append(response_data)
                
                # Record memories created
                memories_created = interaction_result.get("memories_created", [])
                for memory_text in memories_created:
                    await npc_system.add_memory_to_npc(
                        npc_id=nearby_npc["npc_id"],
                        memory_text=memory_text,
                        tags=["player_interaction", interaction_type]
                    )

        # 5) Check time advancement using IntegratedNPCSystem
        old_year, old_month, old_day, old_phase = await npc_system.get_current_game_time()
        
        # Determine if time should advance
        time_result = {
            "time_advanced": False,
            "would_advance": False,
            "periods": 0,
            "current_time": old_phase,
            "confirm_needed": False
        }
        
        if data.get("confirm_time_advance", False):
            # Actually do time advance with IntegratedNPCSystem
            time_result = await npc_system.advance_time_with_activity(activity_type)
        else:
            # Check if it would advance time
            activity_info = await npc_system.process_player_activity(user_input, context)
            if activity_info.get("time_advanced", False):
                time_result = {
                    "time_advanced": False,
                    "would_advance": True,
                    "periods": 1,  # Default to 1 period
                    "current_time": old_phase,
                    "confirm_needed": True
                }

        # 6) If time advanced and new_day > old_day and new_time == "Morning", do nightly maintenance
        if time_result.get("time_advanced", False):
            new_time = time_result.get("new_time", old_phase)
            new_day = time_result.get("new_day", old_day)
            
            if new_time == "Morning" and new_day > old_day:
                # End-of-day => call memory fade
                await nightly_maintenance(user_id, conv_id)
                logging.info("[next_storybeat] Ran nightly maintenance for day rollover.")

            # Process addiction exposures if time advanced
            if time_result.get("time_advanced", False):
                for adtype in ["socks", "feet", "sweat", "ass", "scent"]:
                    if "exposed_to_" + adtype in data:
                        await update_addiction_level(user_id, conv_id, player_name, adtype, 0.2)

        # 7) Relationship crossroads check using IntegratedNPCSystem
        crossroads_event = None
        crossroads_result = {}
        
        if data.get("check_crossroads", False):
            # Use IntegratedNPCSystem to check for relationship events
            relationship_events = await npc_system.check_for_relationship_events()
            if relationship_events:
                for event in relationship_events:
                    if event.get("type") == "relationship_crossroads":
                        crossroads_event = event.get("data")
                        break
                    
            # Handle crossroads choice if provided
            if data.get("crossroads_choice") is not None and data.get("crossroads_name") and data.get("link_id"):
                choice_result = await npc_system.apply_crossroads_choice(
                    int(data["link_id"]),
                    data["crossroads_name"],
                    int(data["crossroads_choice"])
                )
                
                if isinstance(choice_result, dict) and "error" in choice_result:
                    crossroads_result = {"error": choice_result["error"]}
                else:
                    crossroads_result = {
                        "choice_applied": True,
                        "outcome": choice_result.get("outcome_text", "Your choice was processed.")
                    }

        # 8) Build aggregator text with NPC responses
        aggregator_text = build_aggregator_text(aggregator_data)
        
        npc_response_text = ""
        for resp in npc_responses:
            npc_name = resp.get("npc_name", "An NPC")
            action = resp.get("action", "does something")
            outcome = resp.get("result", {}).get("outcome", "")
            npc_response_text += f"{npc_name} {action}. {outcome}\n"
            
        if npc_response_text:
            aggregator_text += "\n\n=== NPC RESPONSES ===\n" + npc_response_text

        # Add addiction context
        addiction_status = await get_addiction_status(user_id, conv_id, player_name)
        if addiction_status and addiction_status.get("has_addictions"):
            from logic.addiction_system import get_addiction_label
            lines = ["\n\n=== ADDICTION STATUS ==="]
            for adtp, lvl in addiction_status.get("addiction_levels", {}).items():
                if lvl > 0:
                    lines.append(f"{adtp.capitalize()}: {get_addiction_label(lvl)}")
            aggregator_text += "\n".join(lines)

        # 9) GPT response
        response_data = get_chatgpt_response(conv_id, aggregator_text, user_input)
        image_result = None
        
        if response_data["type"] == "function_call":
            # function_args => possibly do universal updates
            ai_response = response_data["function_args"].get("narrative", "")
            try:
                if response_data["function_args"]:
                    import asyncpg
                    from logic.universal_updater import apply_universal_updates_async
                    async def apply_updates():
                        dsn = os.getenv("DB_DSN")
                        c2 = await asyncpg.connect(dsn=dsn)
                        try:
                            await apply_universal_updates_async(
                                user_id, conv_id,
                                response_data["function_args"],
                                c2
                            )
                        finally:
                            await c2.close()
                    await apply_updates()
            except Exception as ex:
                logging.error(f"[next_storybeat] Universal update error: {ex}")

            # Possibly do image generation
            should_gen, reason = should_generate_image_for_response(user_id, conv_id, response_data["function_args"])
            if should_gen:
                image_result = generate_roleplay_image_from_gpt(
                    response_data["function_args"],
                    user_id,
                    conv_id
                )
        else:
            ai_response = response_data.get("response", "")

        # 10) Store GPT response
        cur.execute("""
            INSERT INTO messages (conversation_id, sender, content)
            VALUES (%s, %s, %s)
        """, (conv_id, "Nyx", ai_response))
        conn.commit()
        cur.close()
        conn.close()

        # 11) Prepare final JSON
        final_resp = {
            "message": ai_response,
            "time_result": time_result,
            "confirm_needed": time_result.get("would_advance", False) and not data.get("confirm_time_advance", False),
            "npc_responses": [
                {
                    "npc_id": r.get("npc_id"),
                    "npc_name": r.get("npc_name", "An NPC"),
                    "action": r.get("action", ""),
                    "result": r.get("result", {}).get("outcome", "")
                } for r in npc_responses
            ]
        }
        
        if addiction_status:
            final_resp["addiction_effects"] = await process_addiction_effects(user_id, conv_id, player_name, addiction_status)

        # If relationship crossroads
        if crossroads_event:
            final_resp["crossroads_event"] = crossroads_event
        if data.get("crossroads_choice") is not None:
            final_resp["crossroads_result"] = crossroads_result

        # Include image if generated
        if image_result and "image_urls" in image_result:
            final_resp["image"] = {
                "image_url": image_result["image_urls"][0],
                "prompt_used": image_result.get("prompt_used", ""),
                "reason": reason
            }

        # Possibly add narrative_stage or anything else
        narrative_stage = await get_current_narrative_stage(user_id, conv_id)
        if narrative_stage:
            final_resp["narrative_stage"] = narrative_stage.name

        return jsonify(final_resp)

        if cur:
            cur.close()
        if conn:
            conn.close()
            
        return jsonify(final_resp)

    except Exception as e:
        # Proper cleanup in error case
        logger.exception("[next_storybeat] Error")
        return jsonify({"error": str(e)}), 500
        
    finally:
        # Ensure all resources are properly cleaned up
        if cur:
            try:
                cur.close()
            except Exception as cur_error:
                logger.error(f"Error closing cursor: {cur_error}")
                
        if conn:
            try:
                conn.close()
            except Exception as conn_error:
                logger.error(f"Error closing connection: {conn_error}")
        
        # Clean up NPC system resources if needed
        if npc_system:
            try:
                # Close any owned resources
                if hasattr(npc_system, '_pool') and npc_system._pool:
                    await npc_system._pool.close()
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up NPC system: {cleanup_error}")


@story_bp.route("/relationship_summary", methods=["GET"])
async def get_relationship_details():
    """
    Get a summary of the relationship between two entities using IntegratedNPCSystem
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
            
        conversation_id = request.args.get("conversation_id")
        entity1_type = request.args.get("entity1_type")
        entity1_id = request.args.get("entity1_id")
        entity2_type = request.args.get("entity2_type")
        entity2_id = request.args.get("entity2_id")
        
        if not all([conversation_id, entity1_type, entity1_id, entity2_type, entity2_id]):
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Use IntegratedNPCSystem for relationship summary
        npc_system = IntegratedNPCSystem(user_id, int(conversation_id))
        relationship = await npc_system.get_relationship(
            entity1_type, int(entity1_id),
            entity2_type, int(entity2_id)
        )
        
        if not relationship:
            # Try alternative relationship configuration
            relationship = await npc_system.get_relationship(
                entity2_type, int(entity2_id),
                entity1_type, int(entity1_id)
            )
        
        if not relationship:
            return jsonify({"error": "Relationship not found"}), 404
            
        return jsonify(relationship)
        
    except Exception as e:
        logging.exception("[get_relationship_details] Error")
        return jsonify({"error": str(e)}), 500

@story_bp.route("/addiction_status", methods=["GET"])
async def addiction_status():
    """
    Get the current addiction status for a player
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
            
        conversation_id = request.args.get("conversation_id")
        player_name = request.args.get("player_name", "Chase")
        
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
            
        status = await get_addiction_status(user_id, int(conversation_id), player_name)
        return jsonify(status)
        
    except Exception as e:
        logging.exception("[addiction_status] Error")
        return jsonify({"error": str(e)}), 500

@story_bp.route("/apply_crossroads_choice", methods=["POST"])
async def apply_choice():
    """
    Apply a choice in a relationship crossroads using IntegratedNPCSystem
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
            
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        link_id = data.get("link_id")
        crossroads_name = data.get("crossroads_name")
        choice_index = data.get("choice_index")
        
        if not all([conversation_id, link_id, crossroads_name, choice_index is not None]):
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Use IntegratedNPCSystem to apply crossroads choice
        npc_system = IntegratedNPCSystem(user_id, int(conversation_id))
        result = await npc_system.apply_crossroads_choice(
            int(link_id), crossroads_name, int(choice_index)
        )
        
        if isinstance(result, dict) and "error" in result:
            return jsonify({"error": result["error"]}), 400
            
        return jsonify(result)
        
    except Exception as e:
        logging.exception("[apply_choice] Error")
        return jsonify({"error": str(e)}), 500

@story_bp.route("/generate_multi_npc_scene", methods=["POST"])
async def generate_scene():
    """
    Generate a scene with multiple NPCs interacting
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
            
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        npc_ids = data.get("npc_ids", [])
        location = data.get("location")
        include_player = data.get("include_player", True)
        
        if not conversation_id or not npc_ids:
            return jsonify({"error": "Missing required parameters"}), 400
            
        # Generate scene using IntegratedNPCSystem
        npc_system = IntegratedNPCSystem(user_id, int(conversation_id))
        scene = await npc_system.generate_multi_npc_scene(
            npc_ids, location, include_player
        )
        
        return jsonify(scene)
        
    except Exception as e:
        logging.exception("[generate_scene] Error")
        return jsonify({"error": str(e)}), 500

@story_bp.route("/generate_overheard_conversation", methods=["POST"])
async def generate_conversation():
    """
    Generate a conversation between NPCs that the player can overhear
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
            
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        npc_ids = data.get("npc_ids", [])
        topic = data.get("topic")
        about_player = data.get("about_player", False)
        
        if not conversation_id or not npc_ids or len(npc_ids) < 2:
            return jsonify({"error": "Missing required parameters"}), 400
            
        # Generate conversation using IntegratedNPCSystem
        npc_system = IntegratedNPCSystem(user_id, int(conversation_id))
        conversation = await npc_system.generate_overheard_conversation(
            npc_ids, topic, about_player
        )
        
        return jsonify(conversation)
        
    except Exception as e:
        logging.exception("[generate_conversation] Error")
        return jsonify({"error": str(e)}), 500

@story_bp.route("/end_of_day", methods=["POST"])
async def end_of_day():
    """
    Example route that the front-end calls when 
    the player chooses to finish the day or go to sleep. 
    This triggers our nightly_maintenance across all NPCs.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json() or {}
        conv_id = data.get("conversation_id")
        if not conv_id:
            return jsonify({"error": "Missing conversation_id"}), 400

        # Initialize IntegratedNPCSystem
        npc_system = IntegratedNPCSystem(user_id, int(conv_id))
        
        # Get current time to validate we're at night
        year, month, day, time_of_day = await npc_system.get_current_game_time()
        
        # Advance to next day if not already at night
        if time_of_day != "Night":
            await npc_system.set_game_time(year, month, day, "Night")
            
        # Call nightly maintenance for memory fading, summarizing
        await nightly_maintenance(user_id, int(conv_id))

        return jsonify({"status": "Nightly maintenance complete"})
    except Exception as e:
        logging.exception("Error in end_of_day route")
        return jsonify({"error": str(e)}), 500

def gather_rule_knowledge():
    """
    Fetch short summaries from PlotTriggers, IntensityTiers, and Interactions.
    Returns a dictionary with various rule knowledge components.
    
    Optimized for better error handling and performance.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Use a dictionary to store all rule data
        rule_data = {
            "plot_triggers": [],
            "intensity_tiers": [],
            "interactions": []
        }

        # 1) PlotTriggers - fetch all in one go
        cursor.execute("""
            SELECT trigger_name, stage_name, description, key_features, stat_dynamics, examples, triggers
            FROM PlotTriggers
        """)
        
        for row in cursor.fetchall():
            trig_name, stage, desc, kfeat, sdyn, ex, trigz = row
            
            # Safe JSON parsing with fallback to empty values
            try:
                key_features = json.loads(kfeat) if kfeat else []
            except (json.JSONDecodeError, TypeError):
                logging.warning(f"Error parsing key_features for trigger {trig_name}")
                key_features = []
                
            try:
                stat_dynamics = json.loads(sdyn) if sdyn else []
            except (json.JSONDecodeError, TypeError):
                logging.warning(f"Error parsing stat_dynamics for trigger {trig_name}")
                stat_dynamics = []
                
            try:
                examples = json.loads(ex) if ex else []
            except (json.JSONDecodeError, TypeError):
                logging.warning(f"Error parsing examples for trigger {trig_name}")
                examples = []
                
            try:
                triggers = json.loads(trigz) if trigz else {}
            except (json.JSONDecodeError, TypeError):
                logging.warning(f"Error parsing triggers for trigger {trig_name}")
                triggers = {}
            
            rule_data["plot_triggers"].append({
                "title": trig_name,
                "stage": stage,
                "description": desc,
                "key_features": key_features,
                "stat_dynamics": stat_dynamics,
                "examples": examples,
                "triggers": triggers
            })

        # 2) Intensity Tiers - fetch all in one go
        cursor.execute("""
            SELECT tier_name, key_features, activity_examples, permanent_effects
            FROM IntensityTiers
        """)
        
        for row in cursor.fetchall():
            tname, kfeat, aex, peff = row
            
            # Safe JSON parsing with fallback to empty values
            try:
                key_features = json.loads(kfeat) if kfeat else []
            except (json.JSONDecodeError, TypeError):
                logging.warning(f"Error parsing key_features for tier {tname}")
                key_features = []
                
            try:
                activity_examples = json.loads(aex) if aex else []
            except (json.JSONDecodeError, TypeError):
                logging.warning(f"Error parsing activity_examples for tier {tname}")
                activity_examples = []
                
            try:
                permanent_effects = json.loads(peff) if peff else {}
            except (json.JSONDecodeError, TypeError):
                logging.warning(f"Error parsing permanent_effects for tier {tname}")
                permanent_effects = {}
            
            rule_data["intensity_tiers"].append({
                "tier_name": tname,
                "key_features": key_features,
                "activity_examples": activity_examples,
                "permanent_effects": permanent_effects
            })

        # 3) Interactions - fetch all in one go
        cursor.execute("""
            SELECT interaction_name, detailed_rules, task_examples, agency_overrides
            FROM Interactions
        """)
        
        for row in cursor.fetchall():
            iname, drules, tex, aov = row
            
            # Safe JSON parsing with fallback to empty values
            try:
                detailed_rules = json.loads(drules) if drules else {}
            except (json.JSONDecodeError, TypeError):
                logging.warning(f"Error parsing detailed_rules for interaction {iname}")
                detailed_rules = {}
                
            try:
                task_examples = json.loads(tex) if tex else {}
            except (json.JSONDecodeError, TypeError):
                logging.warning(f"Error parsing task_examples for interaction {iname}")
                task_examples = {}
                
            try:
                agency_overrides = json.loads(aov) if aov else {}
            except (json.JSONDecodeError, TypeError):
                logging.warning(f"Error parsing agency_overrides for interaction {iname}")
                agency_overrides = {}
            
            rule_data["interactions"].append({
                "interaction_name": iname,
                "detailed_rules": detailed_rules,
                "task_examples": task_examples,
                "agency_overrides": agency_overrides
            })

        cursor.close()
        
        # Add the rule enforcement summary
        rule_data["rule_enforcement_summary"] = (
            "Conditions are parsed (e.g. 'Lust > 90 or Dependency > 80') and evaluated against stats. "
            "If true, effects such as 'Locks Independent Choices' are applied, raising Obedience, triggering punishments, "
            "or even ending the game."
        )
        
        return rule_data
    
    except Exception as e:
        logging.error(f"Error in gather_rule_knowledge: {str(e)}", exc_info=True)
        # Return minimal data structure to avoid crashes
        return {
            "rule_enforcement_summary": "Error fetching rule data.",
            "plot_triggers": [],
            "intensity_tiers": [],
            "interactions": []
        }
    finally:
        if conn:
            conn.close()

def force_obedience_to_100(user_id, conversation_id, player_name):
    """
    Directly set player's Obedience to 100.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE PlayerStats
            SET obedience=100
            WHERE user_id=%s AND conversation_id=%s AND player_name=%s
        """, (user_id, conversation_id, player_name))
        conn.commit()
    except Exception as ex:
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


def build_aggregator_text(aggregator_data, rule_knowledge=None):
    """
    Merge aggregator_data into a text summary for ChatGPT.
    Optimized for better error handling, performance, and readability.
    """
    # Initialize with a list and join once at the end instead of doing string concatenation
    lines = []
    
    # Get basic time data with defaults
    year = aggregator_data.get("year", 1)
    month = aggregator_data.get("month", 1)
    day = aggregator_data.get("day", 1)
    tod = aggregator_data.get("timeOfDay", "Morning")
    
    # Add main sections with error handling
    try:
        lines.append(f"=== YEAR {year}, MONTH {month}, DAY {day}, {tod.upper()} ===")
        
        # Player Stats Section
        lines.append("\n=== PLAYER STATS ===")
        player_stats = aggregator_data.get("playerStats", {})
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
        
        # NPC Stats Section - Process in batches for efficiency
        lines.append("\n=== NPC STATS ===")
        
        # Get introduced NPCs, checking array and introduced flag explicitly
        npc_stats = aggregator_data.get("introducedNPCs", [])
        if not npc_stats and "npcStats" in aggregator_data:
            # Fallback to filtering from npcStats if introducedNPCs isn't available
            npc_stats = [npc for npc in aggregator_data.get("npcStats", []) 
                         if npc.get("introduced") is True]
            
        if npc_stats:
            for npc in npc_stats:
                try:
                    # Basic NPC info line
                    npc_name = npc.get('npc_name', 'Unnamed')
                    sex = npc.get('sex', 'Unknown')
                    dom = npc.get('dominance', 0)
                    cru = npc.get('cruelty', 0)
                    clos = npc.get('closeness', 0)
                    trust = npc.get('trust', 0)
                    resp = npc.get('respect', 0)
                    inten = npc.get('intensity', 0)
                    
                    # Format archetype info
                    arch_summary = str(npc.get('archetype_summary', []))
                    extras_summary = str(npc.get('archetype_extras_summary', []))
                    
                    lines.append(
                        f"NPC: {npc_name} | Sex={sex} | "
                        f"Archetypes={arch_summary} | Extras={extras_summary} | "
                        f"Dom={dom}, Cru={cru}, Close={clos}, Trust={trust}, "
                        f"Respect={resp}, Int={inten}"
                    )
                    
                    # Add detailed trait information
                    hobbies = npc.get("hobbies", [])
                    personality = npc.get("personality_traits", [])
                    likes = npc.get("likes", [])
                    dislikes = npc.get("dislikes", [])
                    
                    # Safe joins with explicit string conversion
                    hobbies_str = ", ".join(str(h) for h in hobbies) if hobbies else "None"
                    personality_str = ", ".join(str(p) for p in personality) if personality else "None"
                    likes_str = ", ".join(str(l) for l in likes) if likes else "None"
                    dislikes_str = ", ".join(str(d) for d in dislikes) if dislikes else "None"
                    
                    lines.append(f"  Hobbies: {hobbies_str}")
                    lines.append(f"  Personality: {personality_str}")
                    lines.append(f"  Likes: {likes_str} | Dislikes: {dislikes_str}")
                    
                    # Memory
                    memory = npc.get("memory", [])
                    mem_str = str(memory) if memory else "(None)"
                    lines.append(f"  Memory: {mem_str}")
                    
                    # Affiliations
                    affiliations = npc.get("affiliations", [])
                    affil_str = ", ".join(str(a) for a in affiliations) if affiliations else "None"
                    lines.append(f"  Affiliations: {affil_str}")
                    
                    # Schedule - use compact formatting
                    schedule = npc.get("schedule", {})
                    if schedule:
                        # Use more compact single-line format for schedules
                        schedule_summary = []
                        for day, times in schedule.items():
                            day_summary = f"{day}: "
                            time_parts = []
                            for time_period, activity in times.items():
                                if activity:
                                    time_parts.append(f"{time_period}={activity}")
                            day_summary += ", ".join(time_parts)
                            schedule_summary.append(day_summary)
                        
                        lines.append("  Schedule: " + "; ".join(schedule_summary))
                    else:
                        lines.append("  Schedule: (None)")
                    
                    # Current location
                    current_loc = npc.get("current_location", "Unknown")
                    lines.append(f"  Current Location: {current_loc}\n")
                
                except Exception as e:
                    logging.warning(f"Error formatting NPC {npc.get('npc_name', 'unknown')}: {str(e)}")
                    lines.append(f"  [Error processing NPC data: {str(e)}]")
        else:
            lines.append("(No NPCs found)")
        
        # Current Roleplay Section
        lines.append("\n=== CURRENT ROLEPLAY ===")
        current_rp = aggregator_data.get("currentRoleplay", {})
        if current_rp:
            # Only include key currentRoleplay entries that fit in the context window
            important_keys = [
                "EnvironmentDesc", "CurrentSetting", "PlayerRole", "MainQuest", 
                "CurrentYear", "CurrentMonth", "CurrentDay", "TimeOfDay"
            ]
            
            # First add important keys
            for key in important_keys:
                if key in current_rp:
                    lines.append(f"{key}: {current_rp[key]}")
            
            # Then add other keys (excluding very large or less important ones)
            exclude_keys = important_keys + ["ChaseSchedule", "MegaSettingModifiers", "CalendarNames", "GlobalSummary"]
            for key, value in current_rp.items():
                if key not in exclude_keys:
                    # Truncate very long values
                    if isinstance(value, str) and len(value) > 500:
                        lines.append(f"{key}: {value[:500]}... [truncated]")
                    else:
                        lines.append(f"{key}: {value}")
            
            # Add Chase schedule in a condensed format
            if "ChaseSchedule" in current_rp:
                chase_schedule = current_rp["ChaseSchedule"]
                if chase_schedule:
                    lines.append("\n--- Chase Schedule ---")
                    for day, activities in chase_schedule.items():
                        day_line = f"{day}: "
                        time_activities = []
                        for time_slot, activity in activities.items():
                            time_activities.append(f"{time_slot}={activity}")
                        day_line += ", ".join(time_activities)
                        lines.append(day_line)
        else:
            lines.append("(No current roleplay data)")
        
        # Activity Suggestions
        if "activitySuggestions" in aggregator_data and aggregator_data["activitySuggestions"]:
            lines.append("\n=== NPC POTENTIAL ACTIVITIES ===")
            for suggestion in aggregator_data["activitySuggestions"]:
                lines.append(f"- {suggestion}")
            lines.append("NPCs can adopt, combine, or ignore these ideas.\n")
        
        # Social Links Section
        lines.append("\n=== SOCIAL LINKS ===")
        social_links = aggregator_data.get("socialLinks", [])
        if social_links:
            for link in social_links:
                lines.append(
                    f"Link {link.get('link_id', '?')}: "
                    f"{link.get('entity1_type', '?')}({link.get('entity1_id', '?')}) <-> "
                    f"{link.get('entity2_type', '?')}({link.get('entity2_id', '?')}); "
                    f"Type={link.get('link_type', '?')}, Level={link.get('link_level', '?')}"
                )
                history = link.get("link_history", [])
                if history:
                    lines.append(f"  History: {history}")
        else:
            lines.append("(No social links found)")
        
        # Other sections - keep these briefer for better token efficiency
        # Player Perks
        lines.append("\n=== PLAYER PERKS ===")
        player_perks = aggregator_data.get("playerPerks", [])
        if player_perks:
            for perk in player_perks:
                lines.append(
                    f"Perk: {perk.get('perk_name', '?')} | "
                    f"Desc: {perk.get('perk_description', '?')} | "
                    f"Effect: {perk.get('perk_effect', '?')}"
                )
        else:
            lines.append("(No perks found)")
        
        # Inventory - only show important details
        lines.append("\n=== INVENTORY ===")
        inventory = aggregator_data.get("inventory", [])
        if inventory:
            for item in inventory:
                lines.append(
                    f"{item.get('player_name', 'Unknown')}'s Item: {item.get('item_name', 'Unknown')} "
                    f"(x{item.get('quantity', 1)}) - {item.get('item_description', 'No desc')}"
                )
        else:
            lines.append("(No inventory items found)")
        
        # Events - keep concise
        lines.append("\n=== EVENTS ===")
        events_list = aggregator_data.get("events", [])
        if events_list:
            # Only show the 5 most relevant events
            for ev in events_list[:5]:
                lines.append(
                    f"Event #{ev.get('event_id', '?')}: {ev.get('event_name', 'Unknown')} on "
                    f"{ev.get('year', 1)}/{ev.get('month', 1)}/{ev.get('day', 1)} "
                    f"{ev.get('time_of_day', 'Morning')} @ {ev.get('location', 'Unknown')}"
                )
        else:
            lines.append("(No events found)")
        
        # Planned Events - keep concise
        lines.append("\n=== PLANNED EVENTS ===")
        planned_events_list = aggregator_data.get("plannedEvents", [])
        if planned_events_list:
            for pev in planned_events_list[:5]:  # Only show up to 5
                lines.append(
                    f"PlannedEvent #{pev.get('event_id', '?')}: NPC {pev.get('npc_id', '?')} on "
                    f"{pev.get('year', 1)}/{pev.get('month', 1)}/{pev.get('day', 1)} "
                    f"{pev.get('time_of_day', 'Morning')} @ {pev.get('override_location', 'Unknown')}"
                )
        else:
            lines.append("(No planned events found)")
        
        # Quests
        lines.append("\n=== QUESTS ===")
        quests_list = aggregator_data.get("quests", [])
        if quests_list:
            for q in quests_list:
                lines.append(
                    f"Quest #{q.get('quest_id', '?')}: {q.get('quest_name', 'Unknown')} "
                    f"[Status: {q.get('status', 'Unknown')}] - {q.get('progress_detail', '')}"
                )
        else:
            lines.append("(No quests found)")
        
        # Game Rules - more concise
        lines.append("\n=== GAME RULES ===")
        game_rules_list = aggregator_data.get("gameRules", [])
        if game_rules_list:
            for gr in game_rules_list[:10]:  # Limit to 10 most important rules
                lines.append(f"Rule: {gr.get('rule_name', '?')} => If({gr.get('condition', '?')}), then({gr.get('effect', '?')})")
        else:
            lines.append("(No game rules found)")
        
        # Stat Definitions - brief summary
        lines.append("\n=== STAT DEFINITIONS ===")
        stat_definitions_list = aggregator_data.get("statDefinitions", [])
        if stat_definitions_list:
            for sd in stat_definitions_list[:8]:  # Only include most relevant stats
                lines.append(
                    f"{sd.get('stat_name', '?')} [{sd.get('range_min', 0)}..{sd.get('range_max', 100)}]: "
                    f"{sd.get('definition', 'No definition')}"
                )
        else:
            lines.append("(No stat definitions found)")
        
        # Add rule knowledge if provided
        if rule_knowledge:
            lines.append("\n=== ADVANCED RULE ENFORCEMENT & KNOWLEDGE ===")
            lines.append("\nRule Enforcement Summary:")
            lines.append(rule_knowledge.get("rule_enforcement_summary", "(No info)"))
            
            # Add plot triggers - limited to most important ones
            plot_trigs = rule_knowledge.get("plot_triggers", [])
            if plot_trigs:
                lines.append("\n-- PLOT TRIGGERS --")
                for trig in plot_trigs[:3]:  # Only include top 3 triggers
                    lines.append(f"Trigger: {trig.get('title', '?')} - {trig.get('description', '?')}")
            
            # Add intensity tiers - summarized
            tiers = rule_knowledge.get("intensity_tiers", [])
            if tiers:
                lines.append("\n-- INTENSITY TIERS --")
                for tier in tiers:
                    lines.append(f"{tier.get('tier_name', '?')}: {', '.join(tier.get('key_features', []))[:150]}")
            
            # Add interactions - summarized
            interactions = rule_knowledge.get("interactions", [])
            if interactions:
                lines.append("\n-- INTERACTIONS --")
                for intr in interactions[:3]:  # Only include top 3
                    lines.append(f"Interaction: {intr.get('interaction_name', '?')}")
    
    except Exception as e:
        logging.error(f"Error building aggregator text: {str(e)}", exc_info=True)
        # Add error information to prevent complete failure
        lines.append(f"\n[Error occurred while building context: {str(e)}]")
        # Make sure we still return basic information
        if not lines:
            lines = [
                "=== BASIC CONTEXT ===",
                f"Year: {year}, Month: {month}, Day: {day}, Time: {tod}",
                "Error occurred while building full context."
            ]

    return "\n".join(lines)
