# routes/story_routes.py 

import logging
import json
import os
import asyncio
import time
import random
import contextlib
from datetime import datetime, date, timedelta
from quart import Blueprint, request, jsonify, session
from logic.conflict_system.conflict_integration import ConflictSystemIntegration
from logic.activity_analyzer import ActivityAnalyzer
from logic.npc_narrative_progression import check_for_npc_revelation

# Import utility modules
from utils.db_helpers import db_transaction, with_transaction, handle_database_operation, fetch_row_async, fetch_all_async, execute_async
from utils.performance import PerformanceTracker, timed_function, STATS
from utils.caching import NPC_CACHE, LOCATION_CACHE, AGGREGATOR_CACHE, TIME_CACHE, COMPUTATION_CACHE, cache
from utils.performance import timed_function

# Import core logic modules
from db.connection import get_db_connection_context
from logic.universal_updater_agent import apply_universal_updates
from logic.aggregator_sdk import get_aggregated_roleplay_context
from logic.time_cycle import get_current_time, should_advance_time, nightly_maintenance
from logic.inventory_system_sdk import InventoryContext
from logic.chatgpt_integration import get_chatgpt_response, get_openai_client, build_message_history
from logic.resource_management import ResourceManager
from routes.settings_routes import generate_mega_setting_logic
from logic.gpt_image_decision import should_generate_image_for_response
from routes.ai_image_generator import generate_roleplay_image_from_gpt
from lore.core.lore_system import LoreSystem

# Import IntegratedNPCSystem
from logic.fully_integrated_npc_system import IntegratedNPCSystem
from npcs.new_npc_creation import NPCCreationHandler, RunContextWrapper

# Import enhanced modules
from logic.stats_logic import get_player_current_tier, check_for_combination_triggers, apply_stat_change, apply_activity_effects

from logic.social_links import get_relationship_dynamic_level, update_relationship_dynamic, check_for_relationship_crossroads, check_for_relationship_ritual, get_relationship_summary, apply_crossroads_choice

from logic.addiction_system_sdk import check_addiction_levels, update_addiction_level, process_addiction_effects, get_addiction_status, get_addiction_label

from nyx.nyx_agent_sdk import process_user_input

from typing import List, Dict, Any, Optional

from lore.lore_generator import DynamicLoreGenerator

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

async def fetch_npc_details_async(user_id, conversation_id, npc_id):
    """
    Retrieve full or partial NPC info from NPCStats for a given npc_id.
    Async version with connection pooling.
    """
    # Create cache key
    cache_key = f"npc_details:{user_id}:{conversation_id}:{npc_id}"
    
    # Check cache first
    cached_result = NPC_CACHE.get(cache_key)
    if cached_result:
        return cached_result
    
    # Query database if not in cache
    query = """
        SELECT npc_name, introduced, dominance, cruelty, closeness,
               trust, respect, intensity, affiliations, schedule, current_location
        FROM NPCStats
        WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
        LIMIT 1
    """
    
    row = await fetch_row_async(query, user_id, conversation_id, npc_id)
    
    if not row:
        return {"error": f"No NPC found with npc_id={npc_id}"}
    
    # Process row data
    npc_data = {
        "npc_id": npc_id,
        "npc_name": row["npc_name"],
        "introduced": row["introduced"],
        "dominance": row["dominance"],
        "cruelty": row["cruelty"],
        "closeness": row["closeness"],
        "trust": row["trust"],
        "respect": row["respect"],
        "intensity": row["intensity"],
        "affiliations": row["affiliations"] if row["affiliations"] is not None else [],
        "schedule": row["schedule"] if row["schedule"] is not None else {},
        "current_location": row["current_location"]
    }
    
    # Cache the result
    NPC_CACHE.set(cache_key, npc_data, 30)  # TTL: 30 seconds
    
    return npc_data

async def fetch_quest_details_async(user_id, conversation_id, quest_id):
    """
    Retrieve quest info from the Quests table by quest_id.
    Async version with connection pooling.
    """
    # Create cache key
    cache_key = f"quest_details:{user_id}:{conversation_id}:{quest_id}"
    
    # Check cache first
    cached_result = NPC_CACHE.get(cache_key)
    if cached_result:
        return cached_result
        
    query = """
        SELECT quest_name, status, progress_detail, quest_giver, reward
        FROM Quests
        WHERE user_id=$1 AND conversation_id=$2 AND quest_id=$3
        LIMIT 1
    """
    
    row = await fetch_row_async(query, user_id, conversation_id, quest_id)
    
    if not row:
        return {"error": f"No quest found with quest_id={quest_id}"}
    
    # Process row data
    quest_data = {
        "quest_id": quest_id,
        "quest_name": row["quest_name"],
        "status": row["status"],
        "progress_detail": row["progress_detail"],
        "quest_giver": row["quest_giver"],
        "reward": row["reward"]
    }
    
    # Cache the result
    NPC_CACHE.set(cache_key, quest_data, 60)  # TTL: 60 seconds
    
    return quest_data

async def fetch_location_details_async(user_id, conversation_id, location_id=None, location_name=None):
    """
    Retrieve location info by location_id or location_name from the Locations table.
    Async version with connection pooling.
    """
    # Create cache key
    cache_key = f"location_details:{user_id}:{conversation_id}:{location_id or ''}:{location_name or ''}"
    
    # Check cache first
    cached_result = LOCATION_CACHE.get(cache_key)
    if cached_result:
        return cached_result
    
    if location_id:
        query = """
            SELECT location_name, description, open_hours
            FROM Locations
            WHERE user_id=$1 AND conversation_id=$2 AND id=$3
            LIMIT 1
        """
        row = await fetch_row_async(query, user_id, conversation_id, location_id)
    elif location_name:
        query = """
            SELECT location_name, description, open_hours
            FROM Locations
            WHERE user_id=$1 AND conversation_id=$2 AND location_name=$3
            LIMIT 1
        """
        row = await fetch_row_async(query, user_id, conversation_id, location_name)
    else:
        return {"error": "No location_id or location_name provided"}
    
    if not row:
        return {"error": "No matching location found"}
    
    # Process row data
    loc_data = {
        "location_name": row["location_name"],
        "description": row["description"],
        "open_hours": row["open_hours"] if row["open_hours"] is not None else []
    }
    
    # Cache the result
    LOCATION_CACHE.set(cache_key, loc_data, 120)  # TTL: 2 minutes
    
    return loc_data

async def fetch_event_details_async(user_id, conversation_id, event_id):
    """
    Retrieve an event's info by event_id from the Events table.
    Async version with connection pooling.
    """
    query = """
        SELECT event_name, description, start_time, end_time, location, 
               year, month, day, time_of_day
        FROM Events
        WHERE user_id=$1 AND conversation_id=$2 AND id=$3
        LIMIT 1
    """
    
    row = await fetch_row_async(query, user_id, conversation_id, event_id)
    
    if not row:
        return {"error": f"No event found with id={event_id}"}
    
    # Process row data
    event_data = {
        "event_id": event_id,
        "event_name": row["event_name"],
        "description": row["description"],
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "location": row["location"],
        "year": row["year"],
        "month": row["month"],
        "day": row["day"],
        "time_of_day": row["time_of_day"]
    }
    
    return event_data

async def fetch_inventory_item_async(user_id, conversation_id, item_name):
    """
    Lookup a specific item in the player's inventory by item_name.
    Async version with connection pooling.
    """
    query = """
        SELECT player_name, item_description, item_effect, quantity, category
        FROM PlayerInventory
        WHERE user_id=$1 AND conversation_id=$2 AND item_name=$3
        LIMIT 1
    """
    
    row = await fetch_row_async(query, user_id, conversation_id, item_name)
    
    if not row:
        return {"error": f"No item named '{item_name}' found in inventory"}
    
    # Process row data
    item_data = {
        "item_name": item_name,
        "player_name": row["player_name"],
        "item_description": row["item_description"],
        "item_effect": row["item_effect"],
        "quantity": row["quantity"],
        "category": row["category"]
    }
    
    return item_data

async def fetch_intensity_tiers_async():
    """
    Retrieve the entire IntensityTiers data.
    Async version with connection pooling.
    """
    # Check cache first
    cached_result = AGGREGATOR_CACHE.get("intensity_tiers")
    if cached_result:
        return cached_result
    
    query = """
        SELECT tier_name, key_features, activity_examples, permanent_effects
        FROM IntensityTiers
        ORDER BY id
    """
    
    rows = await fetch_all_async(query)
    
    all_tiers = []
    for row in rows:
        try:
            key_features = json.loads(row["key_features"]) if isinstance(row["key_features"], str) else row["key_features"] or []
        except (json.JSONDecodeError, TypeError):
            key_features = []
            
        try:
            activity_examples = json.loads(row["activity_examples"]) if isinstance(row["activity_examples"], str) else row["activity_examples"] or []
        except (json.JSONDecodeError, TypeError):
            activity_examples = []
            
        try:
            permanent_effects = json.loads(row["permanent_effects"]) if isinstance(row["permanent_effects"], str) else row["permanent_effects"] or {}
        except (json.JSONDecodeError, TypeError):
            permanent_effects = {}
        
        all_tiers.append({
            "tier_name": row["tier_name"],
            "key_features": key_features,
            "activity_examples": activity_examples,
            "permanent_effects": permanent_effects
        })
    
    # Cache the result
    AGGREGATOR_CACHE.set("intensity_tiers", all_tiers, 300)  # TTL: 5 minutes
    
    return all_tiers

async def fetch_plot_triggers_async():
    """
    Retrieve the entire list of PlotTriggers.
    Async version with connection pooling.
    """
    # Check cache first
    cached_result = AGGREGATOR_CACHE.get("plot_triggers")
    if cached_result:
        return cached_result
    
    query = """
        SELECT trigger_name, stage_name, description,
               key_features, stat_dynamics, examples, triggers
        FROM PlotTriggers
        ORDER BY id
    """
    
    rows = await fetch_all_async(query)
    
    triggers = []
    for row in rows:
        try:
            key_features = json.loads(row["key_features"]) if isinstance(row["key_features"], str) else row["key_features"] or []
        except (json.JSONDecodeError, TypeError):
            key_features = []
            
        try:
            stat_dynamics = json.loads(row["stat_dynamics"]) if isinstance(row["stat_dynamics"], str) else row["stat_dynamics"] or []
        except (json.JSONDecodeError, TypeError):
            stat_dynamics = []
            
        try:
            examples = json.loads(row["examples"]) if isinstance(row["examples"], str) else row["examples"] or []
        except (json.JSONDecodeError, TypeError):
            examples = []
            
        try:
            triggers_data = json.loads(row["triggers"]) if isinstance(row["triggers"], str) else row["triggers"] or {}
        except (json.JSONDecodeError, TypeError):
            triggers_data = {}
        
        triggers.append({
            "title": row["trigger_name"],
            "stage": row["stage_name"],
            "description": row["description"],
            "key_features": key_features,
            "stat_dynamics": stat_dynamics,
            "examples": examples,
            "triggers": triggers_data
        })
    
    # Cache the result
    AGGREGATOR_CACHE.set("plot_triggers", triggers, 300)  # TTL: 5 minutes
    
    return triggers

async def fetch_interactions_async():
    """
    Retrieve all Interactions from the Interactions table.
    Async version with connection pooling.
    """
    # Check cache first
    cached_result = AGGREGATOR_CACHE.get("interactions")
    if cached_result:
        return cached_result
    
    query = """
        SELECT interaction_name, detailed_rules, task_examples, agency_overrides
        FROM Interactions
        ORDER BY id
    """
    
    rows = await fetch_all_async(query)
    
    result = []
    for row in rows:
        try:
            detailed_rules = json.loads(row["detailed_rules"]) if isinstance(row["detailed_rules"], str) else row["detailed_rules"] or {}
        except (json.JSONDecodeError, TypeError):
            detailed_rules = {}
            
        try:
            task_examples = json.loads(row["task_examples"]) if isinstance(row["task_examples"], str) else row["task_examples"] or {}
        except (json.JSONDecodeError, TypeError):
            task_examples = {}
            
        try:
            agency_overrides = json.loads(row["agency_overrides"]) if isinstance(row["agency_overrides"], str) else row["agency_overrides"] or {}
        except (json.JSONDecodeError, TypeError):
            agency_overrides = {}
        
        result.append({
            "interaction_name": row["interaction_name"],
            "detailed_rules": detailed_rules,
            "task_examples": task_examples,
            "agency_overrides": agency_overrides
        })
    
    # Cache the result
    AGGREGATOR_CACHE.set("interactions", result, 300)  # TTL: 5 minutes
    
    return result

async def get_lore_system(user_id: int, conversation_id: int):
    """
    Get an initialized instance of the LoreSystem.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Initialized LoreSystem instance
    """
    lore_system = LoreSystem.get_instance(user_id, conversation_id)
    await lore_system.initialize()
    return lore_system

@timed_function(name="get_nearby_npcs")
@cache.cached(timeout=300)  # 5-minute cache
async def get_nearby_npcs(user_id: int, conversation_id: int, location: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get nearby NPCs with caching and performance tracking.
    
    Args:
        user_id: The user's ID
        conversation_id: The current conversation ID
        location: Optional location to filter NPCs by
        
    Returns:
        List[Dict[str, Any]]: List of nearby NPCs with their details
    """
    try:
        # Get NPC system instance
        npc_system = await IntegratedNPCSystem.get_instance(user_id, conversation_id)
        
        # Get base NPC list
        npcs = await npc_system.get_nearby_npcs(location)
        
        # Enhance with additional data
        enhanced_npcs = []
        for npc in npcs:
            # Get speech patterns
            speech_patterns = await npc.get_speech_patterns()
            
            # Get relationship status
            relationship = await get_relationship_dynamic_level(user_id, npc['id'])
            
            # Get activity status
            activity = await process_daily_npc_activities(npc['id'])
            
            enhanced_npcs.append({
                **npc,
                'speech_patterns': speech_patterns,
                'relationship': relationship,
                'activity_status': activity
            })
        
        return enhanced_npcs
        
    except Exception as e:
        logging.error(f"Error getting nearby NPCs: {e}", exc_info=True)
        return []

async def process_universal_updates(universal_data):
    """Process universal updates with async connection."""
    operation_name = "process_universal_updates"
    
    # Record the start time for performance tracking
    start_time = time.time()
    
    try:
        async with get_db_connection_context() as conn:
            result = await apply_universal_updates(
                universal_data["user_id"],
                universal_data["conversation_id"],
                universal_data,
                conn
            )
            
            # Record performance metrics
            elapsed_time = time.time() - start_time
            STATS.record_interaction_time(elapsed_time * 1000)  # Convert to ms
            logging.info(f"{operation_name} completed in {elapsed_time:.3f}s")
            
            return result
    except Exception as e:
        # Record error and elapsed time
        elapsed_time = time.time() - start_time
        STATS.record_error(f"{operation_name}_error")
        logging.error(f"Error in {operation_name} after {elapsed_time:.3f}s: {e}")
        return {"error": str(e)}

@timed_function(name="process_npc_interactions_batch")
async def process_npc_interactions_batch(npc_system, nearby_npcs, user_input, activity_type, context):
    """Process NPC interactions in efficient batches."""
    if not nearby_npcs:
        return []
    
    # Determine appropriate interaction type based on input content
    interaction_type = determine_interaction_type(user_input, activity_type)
    
    # Process in batches of 3 for better performance
    batch_size = 3  # Should be 3 by default based on config.py
    npc_responses = []
    
    # Process each batch concurrently
    for i in range(0, len(nearby_npcs), batch_size):
        batch = nearby_npcs[i:i+batch_size]
        
        # Create tasks for concurrent processing
        batch_tasks = [
            npc_system.handle_npc_interaction(
                npc_id=npc["npc_id"],
                interaction_type=interaction_type,
                player_input=user_input,
                context=context
            ) for npc in batch
        ]
        
        # Execute batch concurrently
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Process results
        for npc, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                logging.error(f"Error processing interaction with NPC {npc['npc_id']}: {result}")
                continue
                
            # Format response for display
            if result:
                npc_name = npc["npc_name"]
                response_data = {
                    "npc_id": npc["npc_id"],
                    "npc_name": npc_name,
                    "action": f"reacts to your {activity_type}",
                    "result": {
                        "outcome": get_response_outcome(result, npc_name, activity_type)
                    },
                    "stat_changes": result.get("stat_changes", {})
                }
                npc_responses.append(response_data)
    
    return npc_responses

def determine_interaction_type(user_input, activity_type):
    """Determine the most appropriate interaction type based on user input."""
    lower_input = user_input.lower()
    
    if "no" in lower_input or "won't" in lower_input or "refuse" in lower_input:
        return "defiant_response"
    elif "yes" in lower_input or "okay" in lower_input or "sure" in lower_input:
        return "submissive_response"
    elif any(word in lower_input for word in ["cute", "pretty", "hot", "sexy", "beautiful"]):
        return "flirtatious_remark"
    elif activity_type in ["talk", "question", "conversation"]:
        return "extended_conversation"
    else:
        return "standard_interaction"

def get_response_outcome(result, npc_name, activity_type):
    """Format the response outcome for display."""
    if isinstance(result, dict) and "response" in result:
        return result["response"]
    elif isinstance(result, dict) and "result" in result and "outcome" in result["result"]:
        return result["result"]["outcome"]
    else:
        # Fallback response
        return f"{npc_name} acknowledges your {activity_type}"

@timed_function(name="process_time_advancement")
async def process_time_advancement(npc_system, activity_type, data, current_time=None):
    """Process time advancement with proper verification."""
    if current_time is None:
        # Get current time if not provided
        current_time = await npc_system.get_current_game_time()
        
    old_year, old_month, old_day, old_phase = current_time
    
    # Default time result
    time_result = {
        "time_advanced": False,
        "would_advance": False,
        "periods": 0,
        "current_time": old_phase,
        "confirm_needed": False
    }
    
    # Handle direct confirmation
    if data.get("confirm_time_advance", False):
        # Actually perform time advance
        time_result = await npc_system.advance_time_with_activity(activity_type)
        
        # If time advanced to a new day's morning, run maintenance
        if time_result.get("time_advanced", False):
            new_time = time_result.get("new_time", {})
            if new_time.get("time_of_day") == "Morning" and new_time.get("day") > old_day:
                await nightly_maintenance(npc_system.user_id, npc_system.conversation_id)
                logging.info("[next_storybeat] Ran nightly maintenance for day rollover.")
    else:
        # Check if it would advance time
        would_advance = await npc_system.would_advance_time(activity_type)
        if would_advance:
            time_result = {
                "time_advanced": False,
                "would_advance": True,
                "periods": 1,  # Default to 1 period
                "current_time": old_phase,
                "confirm_needed": True
            }
    
    return time_result

@timed_function(name="process_relationship_events")
async def process_relationship_events(npc_system, data):
    """Process relationship events and crossroads with enhanced dynamics."""
    result = {
        "event": None,
        "result": None
    }
    
    # Check for crossroads events (significant relationship moments)
    if data.get("check_crossroads", False):
        events = await npc_system.check_for_relationship_events()
        if events:
            for event in events:
                if event.get("type") == "relationship_crossroads":
                    result["event"] = event.get("data")
                    break
    
    # Check for relationship rituals (shared activities that boost bonds)
    if data.get("check_rituals", False):
        rituals = await npc_system.check_for_relationship_rituals()
        if rituals:
            result["rituals"] = rituals
    
    # Process crossroads choice if provided
    if data.get("crossroads_choice") is not None and data.get("crossroads_name") and data.get("link_id"):
        choice_result = await npc_system.apply_crossroads_choice(
            int(data["link_id"]),
            data["crossroads_name"],
            int(data["crossroads_choice"])
        )
        
        if isinstance(choice_result, dict) and "error" in choice_result:
            result["result"] = {"error": choice_result["error"]}
        else:
            result["result"] = {
                "choice_applied": True,
                "outcome": choice_result.get("outcome_text", "Your choice was processed.")
            }
            
            # Check for relationship level changes after choice
            if choice_result.get("new_level") and choice_result.get("old_level"):
                level_change = choice_result["new_level"] - choice_result["old_level"]
                if abs(level_change) >= 10:  # Significant change
                    result["result"]["significant_change"] = True
                    result["result"]["level_change"] = level_change
    
    return result

@timed_function(name="process_user_message")
async def process_user_message(user_id, conv_id, user_input):
    """Process and store user message."""
    async with db_transaction() as conn:
        await conn.execute("""
            INSERT INTO messages (conversation_id, sender, content)
            VALUES ($1, $2, $3)
        """, conv_id, "user", user_input)
        return {"status": "stored"}

@timed_function(name="process_npc_responses")
async def process_npc_responses(npc_system, user_input, context):
    """Process NPC responses to user input."""
    # Get nearby NPCs
    nearby_npcs = await get_nearby_npcs(
        npc_system.user_id, 
        npc_system.conversation_id, 
        context.get("location")
    )
    
    if not nearby_npcs:
        return []
    
    # Determine activity type with enhanced detection
    activity_result = await npc_system.process_player_activity(user_input, context)
    activity_type = activity_result.get("activity_type", "conversation")
    
    # Process NPC interactions in batches
    return await process_npc_interactions_batch(
        npc_system, 
        nearby_npcs, 
        user_input, 
        activity_type,
        context
    )

@timed_function(name="process_ai_response_with_nyx")
async def process_ai_response_with_nyx(user_id, conv_id, user_input, context, aggregator_data):
    """
    Process AI response using the Nyx agent instead of direct GPT calls.
    
    Args:
        user_id: User ID
        conv_id: Conversation ID
        user_input: User's input message
        context: Context dictionary
        aggregator_data: Aggregated context data
        
    Returns:
        Tuple of (final_response, image_result)
    """
    # Get Nyx governance to enhance context with lore and conflicts
    try:
        # Get central governance
        from nyx.integrate import get_central_governance
        governance = await get_central_governance(user_id, conv_id)
        
        # Enhance context with lore and conflicts
        enhanced_context = await governance.enhance_context_with_lore(context)
        
        # Add any specific location-based lore if available
        if "location" in context and context["location"]:
            from lore.core.lore_system import LoreSystem
            lore_system = LoreSystem.get_instance(user_id, conv_id)
            await lore_system.initialize()
            location_context = await lore_system.get_location_lore_context(context["location"])
            enhanced_context["location_lore_context"] = location_context
        
        logging.info(f"Enhanced context with lore for user {user_id}, conversation {conv_id}")
        
        # Use the enhanced context instead of the original
        context = enhanced_context
    except Exception as e:
        logging.error(f"Error enhancing context with lore: {e}", exc_info=True)
        # Continue with original context if enhancement fails
    
    # Initialize Nyx agent
    from nyx.nyx_agent import NyxAgent
    nyx_agent = NyxAgent(user_id, conv_id)
    
    # Process with Nyx agent
    response_data = await nyx_agent.process_input(
        user_input,
        context=context
    )
    
    final_response = response_data.get("text", "")
    
    # Check for image generation
    image_result = None
    should_generate = response_data.get("generate_image", False)
    
    if should_generate:
        try:
            # Generate image based on the response
            image_result = await generate_roleplay_image_from_gpt(
                {
                    "narrative": final_response,
                    "image_generation": {
                        "generate": True,
                        "priority": "medium",
                        "focus": "balanced",
                        "framing": "medium_shot",
                        "reason": "Narrative moment"
                    }
                },
                user_id,
                conv_id
            )
        except Exception as e:
            logging.error(f"Error generating image: {e}")
    
    return final_response, image_result

def format_npc_responses(npc_responses):
    """Format NPC responses for the AI context."""
    if not npc_responses:
        return ""
    
    response_text = []
    for resp in npc_responses:
        npc_name = resp.get("npc_name", "NPC")
        result = resp.get("result", {})
        outcome = result.get("outcome", "reacts")
        
        response_text.append(f"{npc_name}: {outcome}")
    
    return "\n".join(response_text)

def format_npc_responses_for_client(npc_responses):
    """Format NPC responses for the client."""
    if not npc_responses:
        return []
    
    client_responses = []
    for resp in npc_responses:
        client_resp = {
            "npc_id": resp.get("npc_id"),
            "npc_name": resp.get("npc_name", "NPC"),
            "response": resp.get("result", {}).get("outcome", "reacts"),
            "stat_changes": resp.get("stat_changes", {})
        }
        client_responses.append(client_resp)
    
    return client_responses

def format_addiction_status(addiction_status):
    """Format addiction status for context."""
    if not addiction_status or not addiction_status.get("has_addictions"):
        return ""
    
    result = "\n\n=== ADDICTION STATUS ===\n"
    for addiction, details in addiction_status.get("addictions", {}).items():
        label = details.get("label", "Unknown")
        level = details.get("level", 0)
        result += f"{addiction}: {label} (Level {level})\n"
    
    return result

async def cleanup_resources(resources):
    """Comprehensive cleanup of all resources."""
    # If there's an NPC system, ensure any connections are properly closed
    if resources.get("npc_system"):
        await resources["npc_system"].close()

async def check_npc_availability(user_id, conv_id):
    """Check NPC availability with error handling."""
    try:
        async with get_db_connection_context() as conn:
            query = """
                SELECT COUNT(*) FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND introduced=FALSE
            """
            count = await conn.fetchval(query, user_id, conv_id)
            return [count]
    except Exception as e:
        logging.error(f"Error checking NPC availability: {e}")
        return [0]  # Default to needing NPCs

async def manage_npc_memory_lifecycle(npc_system):
    """
    Comprehensive memory lifecycle management for NPCs.
    Implements the strategies from TO_DO.TXT.
    """
    tracker = PerformanceTracker("memory_lifecycle")
    
    results = {
        "pruned_count": 0,
        "decayed_count": 0,
        "consolidated_count": 0,
        "archived_count": 0,
        "emotional_tagging_count": 0  # New counter
    }
    
    try:
        # Run memory maintenance for all NPCs
        tracker.start_phase("memory_maintenance")
        maintenance_result = await npc_system.run_memory_maintenance()
        tracker.end_phase()
        
        # Apply emotional tagging - memories with strong emotion last longer
        tracker.start_phase("emotional_tagging")
        tagging_result = await npc_system.apply_emotional_tagging()
        results["emotional_tagging_count"] = tagging_result.get("tagged_count", 0)
        tracker.end_phase()
        
        # Consolidate repetitive memories
        tracker.start_phase("consolidate_memories")
        consolidate_result = await npc_system.consolidate_repetitive_memories()
        results["consolidated_count"] = consolidate_result.get("consolidated_count", 0)
        tracker.end_phase()
        
        # Process memory decay - consider intelligence factor for decay rate
        tracker.start_phase("memory_decay")
        decay_result = await npc_system.apply_memory_decay()
        results["decayed_count"] = decay_result.get("decayed_count", 0)
        tracker.end_phase()
        
        # Archive old memories
        tracker.start_phase("archive_memories")
        archive_result = await npc_system.archive_stale_memories()
        results["archived_count"] = archive_result.get("archived_count", 0)
        tracker.end_phase()
        
        # Record performance metrics
        results["performance_metrics"] = tracker.get_metrics()
        return results
        
    except Exception as e:
        logging.error(f"Error in memory lifecycle management: {e}")
        tracker.end_phase()
        return {
            "error": str(e),
            "performance_metrics": tracker.get_metrics()
        }

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
        logging.error(f"Error building aggregator text: {str(e)}")
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

# -------------------------------------------------------------------
# ROUTE DEFINITION
# -------------------------------------------------------------------

@story_bp.route("/player/resources", methods=["GET"])
@timed_function
async def get_player_resources():
    """
    Get the current player resources.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
            
        conversation_id = request.args.get("conversation_id")
        player_name = request.args.get("player_name", "Chase")
        
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        # Create cache key
        cache_key = f"resources:{user_id}:{conversation_id}:{player_name}"
        
        # Check cache first
        cached_result = NPC_CACHE.get(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        # Get resources from ResourceManager
        resource_manager = ResourceManager(user_id, int(conversation_id), player_name)
        resources = await resource_manager.get_resources()
        vitals = await resource_manager.get_vitals()
        
        # Combine resources and vitals for a comprehensive resource state
        result = {
            "resources": resources,
            "vitals": vitals
        }
        
        # Cache the result
        NPC_CACHE.set(cache_key, result, 30)  # TTL: 30 seconds
        
        return jsonify(result)
        
    except Exception as e:
        logging.exception("[get_player_resources] Error")
        return jsonify({"error": str(e)}), 500

@story_bp.route("/player/resources/modify", methods=["POST"])
@timed_function
async def modify_player_resources():
    """
    Modify player resources (money, supplies, influence, hunger, energy).
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
            
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        player_name = data.get("player_name", "Chase")
        
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        # Get resource and amount to modify
        resource_type = data.get("resource_type")
        amount = data.get("amount")
        source = data.get("source", "manual")
        description = data.get("description", "")
        
        if not resource_type or amount is None:
            return jsonify({"error": "Missing resource_type or amount parameters"}), 400
        
        # Initialize ResourceManager
        resource_manager = ResourceManager(user_id, int(conversation_id), player_name)
        
        # Modify the specified resource
        result = None
        if resource_type == "money":
            result = await resource_manager.modify_money(amount, source, description)
        elif resource_type == "supplies":
            result = await resource_manager.modify_supplies(amount, source, description)
        elif resource_type == "influence":
            result = await resource_manager.modify_influence(amount, source, description)
        elif resource_type == "hunger":
            result = await resource_manager.modify_hunger(amount, source, description)
        elif resource_type == "energy":
            result = await resource_manager.modify_energy(amount, source, description)
        else:
            return jsonify({"error": f"Invalid resource_type: {resource_type}"}), 400
        
        # Clear cache
        NPC_CACHE.remove(f"resources:{user_id}:{conversation_id}:{player_name}")
        
        return jsonify(result)
        
    except Exception as e:
        logging.exception("[modify_player_resources] Error")
        return jsonify({"error": str(e)}), 500

@story_bp.route("/player/daily-income", methods=["POST"])
@timed_function
async def process_daily_income():
    """
    Process daily income for the player.
    This gets called during nightly maintenance.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
            
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        # Initialize ResourceManager
        resource_manager = ResourceManager(user_id, int(conversation_id))
        
        # Base daily income
        money_income = 10
        supplies_income = 5
        influence_income = 2
        
        # Get player stats to modify income based on skills/stats
        # For example, higher confidence might increase influence gain
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT confidence, willpower FROM PlayerStats
                WHERE user_id=$1 AND conversation_id=$2 AND player_name='Chase'
            """, user_id, conversation_id)
            
            if row:
                confidence, willpower = row["confidence"], row["willpower"]
                # Adjust income based on stats
                influence_bonus = max(0, (confidence - 50) // 10)
                money_bonus = max(0, (willpower - 50) // 10)
                
                money_income += money_bonus
                influence_income += influence_bonus
        
        # Apply income
        money_result = await resource_manager.modify_money(
            money_income, "daily_income", "Daily income"
        )
        
        supplies_result = await resource_manager.modify_supplies(
            supplies_income, "daily_income", "Daily supplies"
        )
        
        influence_result = await resource_manager.modify_influence(
            influence_income, "daily_income", "Daily influence gain"
        )
        
        # Recover some energy overnight
        energy_result = await resource_manager.modify_energy(
            20, "rest", "Overnight rest"  # Recover 20 energy overnight
        )
        
        # Decrease hunger (negative value = more hungry)
        hunger_result = await resource_manager.modify_hunger(
            -15, "metabolism", "Overnight metabolism"  # Lose 15 hunger overnight
        )
        
        return jsonify({
            "money": money_result,
            "supplies": supplies_result,
            "influence": influence_result,
            "energy": energy_result,
            "hunger": hunger_result
        })
        
    except Exception as e:
        logging.exception("[process_daily_income] Error")
        return jsonify({"error": str(e)}), 500

@story_bp.route("/analyze_activity", methods=["POST"])
@timed_function
async def analyze_activity_endpoint():
    """
    Analyze an activity to determine its resource effects without applying them.
    This can be used for showing the player what will happen before they confirm.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
            
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        activity_text = data.get("activity_text")
        
        if not conversation_id or not activity_text:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Initialize activity analyzer
        activity_analyzer = ActivityAnalyzer(user_id, int(conversation_id))
        
        # Analyze the activity (but don't apply effects yet)
        activity_analysis = await activity_analyzer.analyze_activity(
            activity_text,
            apply_effects=False
        )
        
        return jsonify({
            "analysis": activity_analysis,
            "message": "Activity analyzed successfully"
        })
        
    except Exception as e:
        logging.exception("[analyze_activity_endpoint] Error")
        return jsonify({"error": str(e)}), 500

@story_bp.route("/perform_activity", methods=["POST"])
@timed_function
async def perform_activity_endpoint():
    """
    Perform a specific activity and apply its resource effects.
    This endpoint allows activities outside the main conversation flow.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
            
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        activity_text = data.get("activity_text")
        
        if not conversation_id or not activity_text:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Initialize activity analyzer
        activity_analyzer = ActivityAnalyzer(user_id, int(conversation_id))
        
        # Analyze and apply the activity effects
        activity_result = await activity_analyzer.analyze_activity(
            activity_text,
            apply_effects=True
        )
        
        # Get updated resources
        resource_manager = ResourceManager(user_id, int(conversation_id))
        current_resources = await resource_manager.get_resources()
        current_vitals = await resource_manager.get_vitals()
        
        # Build the response
        response = {
            "activity_result": activity_result,
            "current_resources": current_resources,
            "current_vitals": current_vitals,
            "message": "Activity performed successfully"
        }
        
        # Check for special conditions based on resource states
        if current_vitals.get("hunger", 100) < 20:
            response["warnings"] = ["You're getting very hungry."]
        if current_vitals.get("energy", 100) < 20:
            response["warnings"] = response.get("warnings", []) + ["You're getting very tired."]
        
        return jsonify(response)
        
    except Exception as e:
        logging.exception("[perform_activity_endpoint] Error")
        return jsonify({"error": str(e)}), 500

@story_bp.route("/currency_info", methods=["GET"])
@timed_function
async def get_currency_info():
    """
    Get the currency system information for the current game.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
            
        conversation_id = request.args.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        # Create cache key
        cache_key = f"currency:{user_id}:{conversation_id}"
        
        # Check cache first
        cached_result = NPC_CACHE.get(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        # Get currency system
        from logic.currency_generator import CurrencyGenerator
        currency_generator = CurrencyGenerator(user_id, int(conversation_id))
        currency_system = await currency_generator.get_currency_system()
        
        # Cache the result
        NPC_CACHE.set(cache_key, currency_system, 3600)  # TTL: 1 hour (currency doesn't change often)
        
        return jsonify(currency_system)
        
    except Exception as e:
        logging.exception("[get_currency_info] Error")
        return jsonify({"error": str(e)}), 500

@story_bp.route("/next_storybeat", methods=["POST"])
async def next_storybeat():
    """Enhanced storybeat endpoint with better resource management and parallel processing."""
    tracker = PerformanceTracker("next_storybeat")
    tracker.start_phase("initialization")

    STATS.record_request("/next_storybeat")
    
    response = {}
    resources = {
        "npc_system": None
    }

    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        data = request.get_json() or {}
        user_input = data.get("user_input", "").strip()
        conv_id = data.get("conversation_id")
        player_name = data.get("player_name", "Chase")

        # Initialize NPC system
        resources["npc_system"] = IntegratedNPCSystem(user_id, conv_id)
        
        tracker.end_phase()  # initialization phase

        # 1) Store the user message
        tracker.start_phase("store_message")
        await process_user_message(user_id, conv_id, user_input)
        tracker.end_phase()

        # 2) Build aggregator context
        tracker.start_phase("get_context")
        aggregator_data = await get_aggregated_roleplay_context(user_id, conv_id, player_name)
        context = {
            "location": aggregator_data.get("currentRoleplay", {}).get("CurrentLocation", "Unknown"),
            "time_of_day": aggregator_data.get("timeOfDay", "Morning"),
            "player_input": user_input,
            "player_name": player_name,
            # You can pass the entire aggregator if desired:
            "aggregator_data": aggregator_data
        }
        tracker.end_phase()

        # 3) Parallel tasks: check NPC availability, current time, nearby NPCs
        tracker.start_phase("parallel_tasks")
        tasks = [
            check_npc_availability(user_id, conv_id),
            resources["npc_system"].get_current_game_time(),
            # possibly other tasks
        ]
        npc_count_result, current_time = await asyncio.gather(*tasks)
        tracker.end_phase()

        # 4) Possibly spawn new NPCs if needed
        tracker.start_phase("spawn_npcs")
        unintroduced_count = npc_count_result[0] if npc_count_result else 0
        if unintroduced_count < 2:
            try:
                # Create NPCCreationHandler
                npc_handler = NPCCreationHandler()
                
                # Create context wrapper
                ctx = RunContextWrapper({
                    "user_id": user_id,
                    "conversation_id": conv_id
                })
                
                # Get environment description
                env_desc = aggregator_data.get("currentRoleplay", {}).get("EnvironmentDesc", 
                                                                  "A default environment.")
                
                # Spawn NPCs directly
                npc_ids = await npc_handler.spawn_multiple_npcs(ctx, count=3)
                logging.info(f"Generated new NPCs: {npc_ids}")
            except Exception as e:
                logging.error(f"Error spawning NPCs: {e}")
        tracker.end_phase()

        # 5) Process universal updates if present
        if data.get("universal_update"):
            tracker.start_phase("universal_updates")
            universal_data = data["universal_update"]
            universal_data["user_id"] = user_id
            universal_data["conversation_id"] = conv_id

            update_result = await process_universal_updates(universal_data)
            if update_result.get("error"):
                return jsonify(update_result), 500
            tracker.end_phase()

        # 6) (Optional) NPC interactions if you want them before Nyx's response
        tracker.start_phase("npc_interactions")
        npc_responses = await process_npc_responses(
            resources["npc_system"], 
            user_input, 
            context
        )
        tracker.end_phase()
        
        # 6.5) Check for NPC-specific revelations based on interactions
        tracker.start_phase("npc_revelations")
        npc_revelations = []
        if npc_responses:
            # Check for revelations for each NPC we just interacted with
            for npc_resp in npc_responses[:3]:  # Limit to top 3 to avoid spam
                npc_id = npc_resp.get("npc_id")
                if npc_id:
                    revelation = await check_for_npc_revelation(
                        user_id, 
                        conv_id, 
                        npc_id
                    )
                    if revelation:
                        npc_revelations.append(revelation)
                        logging.info(f"NPC revelation triggered for {revelation['npc_name']}: {revelation['stage']}")
        tracker.end_phase()
        
        # Add revelations to response if any occurred
        if npc_revelations:
            # Add to the context dict that gets passed to Nyx
            context["narrative_hints"] = {
                "npc_revelations": [
                    {
                        "npc_name": rev["npc_name"],
                        "stage": rev["stage"],
                        "revelation_type": "player_realization"
                    }
                    for rev in npc_revelations
                ],
                "tone_guidance": "The player is beginning to understand the true nature of their relationships. Incorporate subtle, ominous hints about the NPCs' control without being explicit."
            }

        if "narrative_context" not in aggregator_data:
            aggregator_data["narrative_context"] = {}
        
        aggregator_data["narrative_context"]["active_revelations"] = [
            f"{rev['npc_name']} - {rev['stage']}" for rev in npc_revelations
        ]

        # 7) Time advancement
        tracker.start_phase("time_advancement")
        action_type = npc_responses[0].get("action_type", "conversation") if npc_responses else "conversation"
        time_result = await process_time_advancement(
            resources["npc_system"],
            action_type,
            data,
            current_time
        )
        tracker.end_phase()

        # Process resources based on activity
        tracker.start_phase("resource_processing")
        try:
            activity_type = action_type if "action_type" in locals() else "conversation"
            activity_analyzer = ActivityAnalyzer(user_id, conv_id)
            
            # If we have a specific activity description from the user, use that
            # Otherwise, use the general activity type
            activity_description = data.get("activity_description", user_input)
            if not activity_description or len(activity_description) < 5:
                activity_description = f"{activity_type}"
            
            # Analyze the activity and apply effects
            activity_analysis = await activity_analyzer.analyze_activity(
                activity_description,
                apply_effects=True
            )
            
            # Get updated resources
            resource_manager = ResourceManager(user_id, conv_id)
            current_resources = await resource_manager.get_resources()
            current_vitals = await resource_manager.get_vitals()
            
            # Add results to response
            response["activity_effects"] = {
                "activity_type": activity_analysis["activity_type"],
                "activity_details": activity_analysis["activity_details"],
                "effects": activity_analysis["effects"],
                "description": activity_analysis["description"],
                "flags": activity_analysis.get("flags", {})
            }
            response["current_resources"] = current_resources
            response["current_vitals"] = current_vitals
            
        except Exception as e:
            logging.error(f"Error processing resources: {e}")
            response["resource_error"] = str(e)
        tracker.end_phase()
    
        # Process conflicts
        tracker.start_phase("conflicts")
        try:
            conflict_integration = ConflictSystemIntegration(user_id, conv_id)
            
            # Check for active conflicts
            active_conflicts = await conflict_integration.get_active_conflicts()
            
            # If time advanced, run daily update
            conflict_update = None
            if time_result.get("time_advanced", False):
                conflict_update = await conflict_integration.run_daily_update()
            
            # Process activity impact on conflicts
            impact_result = None
            activity_type = action_type if "action_type" in locals() else "conversation"
            impact_result = await conflict_integration.process_activity_for_conflict_impact(
                activity_type, 
                user_input
            )
            
            # Add conflict info to response
            response["conflicts"] = {
                "active": active_conflicts,
                "updates": conflict_update,
                "impact": impact_result
            }
        except Exception as e:
            logging.error(f"Error processing conflicts: {e}")
            response["conflicts"] = {"error": str(e)}
        tracker.end_phase()

        # 8) Relationship events & crossroads
        tracker.start_phase("relationship_events")
        crossroads_data = await process_relationship_events(
            resources["npc_system"], data
        )
        tracker.end_phase()

        # 8.5 Process conflict evolution based on player action
        tracker.start_phase("conflict_evolution")
        if active_conflicts:
            # Notify conflict system of player action
            await on_player_major_action(
                user_id,
                conversation_id,
                "player_input",  # action type
                {
                    "description": user_input,
                    "involved_npcs": [resp.npc_id for resp in npc_responses],
                    "location": current_location
                }
            )
            
            # Generate story beats for active conflicts affected by this action
            from story_agent.tools import generate_conflict_beat, ConflictBeatGenerationParams
            
            for conflict in active_conflicts[:2]:  # Process top 2 conflicts to avoid overload
                # Check if any responding NPCs are stakeholders in this conflict
                conflict_stakeholder_ids = {
                    s['npc_id'] for s in conflict.get('stakeholders', []) 
                    if s.get('entity_type') == 'npc'
                }
                
                involved_npc_ids = [
                    resp.npc_id for resp in npc_responses 
                    if resp.npc_id in conflict_stakeholder_ids
                ]
                
                if involved_npc_ids or conflict.get('player_involvement', {}).get('involvement_level') != 'none':
                    # Generate a story beat for this conflict
                    beat_params = ConflictBeatGenerationParams(
                        conflict_id=conflict['conflict_id'],
                        recent_action=user_input,
                        involved_npcs=involved_npc_ids
                    )
                    
                    beat_ctx = RunContextWrapper(context={
                        'user_id': user_id,
                        'conversation_id': conversation_id
                    })
                    
                    beat_result = await generate_conflict_beat(beat_ctx, beat_params)
                    
                    if beat_result.get('success'):
                        logger.info(f"Generated conflict beat for {conflict['conflict_name']}")
                        
                        # Store beat result for potential inclusion in response
                        if 'conflict_beats' not in response:
                            response['conflict_beats'] = []
                            
                        response['conflict_beats'].append({
                            'conflict_name': conflict['conflict_name'],
                            'beat': beat_result['generated_beat']['beat_description'],
                            'impact': beat_result['generated_beat']['impact_summary']
                        })
        tracker.end_phase()

        # 9) *** Call your Nyx agent instead of direct GPT. ***
        tracker.start_phase("ai_response")
        # This calls your Agents-based function with aggregator_data in 'context'
        agent_output = await process_user_input(
            user_id=user_id,
            conversation_id=conv_id,
            user_input=user_input,
            context_data=context
        )
        tracker.end_phase()

        # The agent_output is e.g.:
        # {
        #   "message": "...",
        #   "generate_image": True/False,
        #   "image_prompt": "...",
        #   "tension_level": ...,
        #   ...
        # }

        final_response = agent_output.get("message", "")
        generate_img_flag = agent_output.get("generate_image", False)
        image_prompt = agent_output.get("image_prompt", "")

        # 10) Store the final Nyx message in messages table
        tracker.start_phase("store_ai_response")
        async with db_transaction() as conn:
            await conn.execute("""
                INSERT INTO messages (conversation_id, sender, content)
                VALUES ($1, $2, $3)
            """, conv_id, "Nyx", final_response)
        tracker.end_phase()

        # 11) Memory maintenance scheduling
        tracker.start_phase("memory_maintenance")
        maintenance_key = f"last_maintenance:{user_id}:{conv_id}"
        last_maintenance = TIME_CACHE.get(maintenance_key)
        now_time = time.time()

        elapsed_time = float('inf') if last_maintenance is None else now_time - last_maintenance
        should_run_maintenance = (elapsed_time > 1800) or (random.random() < 0.1)

        if should_run_maintenance:
            asyncio.create_task(manage_npc_memory_lifecycle(resources["npc_system"]))
            TIME_CACHE.set(maintenance_key, now_time, 3600)
            logging.info(f"Scheduled memory lifecycle management (elapsed: {elapsed_time:.1f}s)")
        tracker.end_phase()

        # 12) If the agent asked for an image, generate it
        tracker.start_phase("image_generation")
        image_result = None
        if generate_img_flag:
            try:
                # You can pass agent_output["image_prompt"] or final_response
                # or a combination.
                generation_data = {
                    "narrative": final_response,
                    "image_generation": {
                        "generate": True,
                        "priority": "medium",
                        "focus": "balanced",
                        "framing": "medium_shot",
                        "reason": "Nyx requested image"
                    }
                }
                image_result = await generate_roleplay_image_from_gpt(
                    generation_data, user_id, conv_id
                )
            except Exception as e:
                logging.error(f"Image generation error: {e}")
        tracker.end_phase()

        # 13) Build final JSON
        tracker.start_phase("build_response")
        response.update({
            "message": final_response,
            "time_result": time_result,
            "confirm_needed": time_result.get("would_advance", False) and not data.get("confirm_time_advance", False),
            "npc_responses": format_npc_responses_for_client(npc_responses),
            "performance_metrics": tracker.get_metrics()
        })
        
        # Optionally, you might want to track that revelations occurred for debugging/analytics
        if npc_revelations:
            response["_debug"] = {
                "revelations_triggered": len(npc_revelations),
                "revelation_npcs": [rev["npc_name"] for rev in npc_revelations]
            } if os.getenv("FLASK_ENV") == "development" else None

        # Optional: If agent provided tension level or environment changes, etc.
        if "tension_level" in agent_output:
            response["tension_level"] = agent_output["tension_level"]
        if "environment_update" in agent_output:
            response["environment_update"] = agent_output["environment_update"]

        # Add addiction status if relevant
        addiction_status = await get_addiction_status(user_id, conv_id, player_name)
        if addiction_status and addiction_status.get("has_addictions"):
            response["addiction_effects"] = await process_addiction_effects(
                user_id, conv_id, player_name, addiction_status
            )

        # Add crossroads event data if any
        if crossroads_data.get("event"):
            response["crossroads_event"] = crossroads_data["event"]
        if crossroads_data.get("result"):
            response["crossroads_result"] = crossroads_data["result"]

        # If we got an image, attach it
        if image_result and "image_urls" in image_result:
            response["image"] = {
                "image_url": image_result["image_urls"][0],
                "prompt_used": image_result.get("prompt_used", ""),
                "reason": image_result.get("reason", "")
            }

        # Possibly add the narrative stage
        narrative_stage = await get_current_narrative_stage(user_id, conv_id)
        if narrative_stage:
            response["narrative_stage"] = narrative_stage.name

        # In dev mode, show cache stats
        if os.getenv("FLASK_ENV") == "development":
            response["cache_stats"] = {
                "npc_cache": NPC_CACHE.stats(),
                "location_cache": LOCATION_CACHE.stats()
            }
        tracker.end_phase()

        return jsonify(response)

    except Exception as e:
        if tracker.current_phase:
            tracker.end_phase()
        STATS.record_error(type(e).__name__)
        logging.exception("[next_storybeat] Error")

        return jsonify({
            "error": str(e),
            "performance": tracker.get_metrics()
        }), 500

    finally:
        await cleanup_resources(resources)

@contextlib.asynccontextmanager
async def npc_system_context(user_id, conv_id):
    """Context manager for NPC system resources."""
    resources = {"npc_system": None}
    try:
        resources["npc_system"] = IntegratedNPCSystem(user_id, conv_id)
        yield resources
    finally:
        if resources.get("npc_system"):
            await resources["npc_system"].close()

@story_bp.route("/relationship_summary", methods=["GET"])
@timed_function
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
        
        # Create cache key
        cache_key = f"relationship:{conversation_id}:{entity1_type}:{entity1_id}:{entity2_type}:{entity2_id}"
        
        # Check cache first
        cached_result = NPC_CACHE.get(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
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
        
        # Cache the result
        NPC_CACHE.set(cache_key, relationship, 60)  # TTL: 60 seconds
            
        return jsonify(relationship)
        
    except Exception as e:
        logging.exception("[get_relationship_details] Error")
        return jsonify({"error": str(e)}), 500

@story_bp.route("/relationships", methods=["GET"])
@timed_function
async def get_relationships():
    """
    Get relationships for a character
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
            
        conversation_id = request.args.get("conversation_id")
        entity_type = request.args.get("entity_type", "player")
        entity_id = request.args.get("entity_id")
        
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        # If entity_type is player, get player id
        if entity_type == "player" and not entity_id:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow("""
                    SELECT id FROM PlayerStats
                    WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
                """, user_id, conversation_id)
                if row:
                    entity_id = row["id"]
        
        if not entity_id:
            return jsonify({"error": "No entity_id provided or found"}), 400
        
        # Create cache key
        cache_key = f"relationships:{user_id}:{conversation_id}:{entity_type}:{entity_id}"
        
        # Check cache first
        cached_result = NPC_CACHE.get(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        # Get all relationships for this entity
        async with get_db_connection_context() as conn:
            # Get relationships where entity is entity1
            rows1 = await conn.fetch("""
                SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id,
                       link_type, link_level, link_history, active
                FROM SocialLinks
                WHERE user_id = $1 AND conversation_id = $2
                AND entity1_type = $3 AND entity1_id = $4
                AND active = true
            """, user_id, conversation_id, entity_type, entity_id)
            
            # Get relationships where entity is entity2
            rows2 = await conn.fetch("""
                SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id,
                       link_type, link_level, link_history, active
                FROM SocialLinks
                WHERE user_id = $1 AND conversation_id = $2
                AND entity2_type = $3 AND entity2_id = $4
                AND active = true
            """, user_id, conversation_id, entity_type, entity_id)
            
            # Combine and format relationships
            relationships = []
            
            for row in rows1:
                # For each relationship where entity is entity1
                rel = {
                    "link_id": row["link_id"],
                    "entity_type": row["entity2_type"],
                    "entity_id": row["entity2_id"],
                    "link_type": row["link_type"],
                    "link_level": row["link_level"],
                    "history": row["link_history"] if row["link_history"] else []
                }
                
                # Get entity name for entity2
                if row["entity2_type"] == "npc":
                    npc_row = await conn.fetchrow("""
                        SELECT npc_name FROM NPCStats
                        WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                    """, user_id, conversation_id, row["entity2_id"])
                    if npc_row:
                        rel["entity_name"] = npc_row["npc_name"]
                elif row["entity2_type"] == "player":
                    player_row = await conn.fetchrow("""
                        SELECT player_name FROM PlayerStats
                        WHERE user_id = $1 AND conversation_id = $2 AND id = $3
                    """, user_id, conversation_id, row["entity2_id"])
                    if player_row:
                        rel["entity_name"] = player_row["player_name"]
                
                relationships.append(rel)
            
            for row in rows2:
                # For each relationship where entity is entity2
                rel = {
                    "link_id": row["link_id"],
                    "entity_type": row["entity1_type"],
                    "entity_id": row["entity1_id"],
                    "link_type": row["link_type"],
                    "link_level": row["link_level"],
                    "history": row["link_history"] if row["link_history"] else []
                }
                
                # Get entity name for entity1
                if row["entity1_type"] == "npc":
                    npc_row = await conn.fetchrow("""
                        SELECT npc_name FROM NPCStats
                        WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                    """, user_id, conversation_id, row["entity1_id"])
                    if npc_row:
                        rel["entity_name"] = npc_row["npc_name"]
                elif row["entity1_type"] == "player":
                    player_row = await conn.fetchrow("""
                        SELECT player_name FROM PlayerStats
                        WHERE user_id = $1 AND conversation_id = $2 AND id = $3
                    """, user_id, conversation_id, row["entity1_id"])
                    if player_row:
                        rel["entity_name"] = player_row["player_name"]
                
                relationships.append(rel)
            
            result = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "relationships": relationships
            }
            
            # Cache the result
            NPC_CACHE.set(cache_key, result, 60)  # TTL: 60 seconds
            
            return jsonify(result)
        
    except Exception as e:
        logging.exception("[get_relationships] Error")
        return jsonify({"error": str(e)}), 500

@story_bp.route("/addiction_status", methods=["GET"])
@timed_function
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
        
        # Create cache key
        cache_key = f"addiction:{user_id}:{conversation_id}:{player_name}"
        
        # Check cache first
        cached_result = NPC_CACHE.get(cache_key)
        if cached_result:
            return jsonify(cached_result)
            
        status = await get_addiction_status(user_id, int(conversation_id), player_name)
        
        # Cache the result
        NPC_CACHE.set(cache_key, status, 60)  # TTL: 60 seconds
        
        return jsonify(status)
        
    except Exception as e:
        logging.exception("[addiction_status] Error")
        return jsonify({"error": str(e)}), 500

@story_bp.route("/apply_crossroads_choice", methods=["POST"])
@timed_function
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
            
        # Clear relationship cache
        NPC_CACHE.remove_pattern("relationship:")
            
        return jsonify(result)
        
    except Exception as e:
        logging.exception("[apply_choice] Error")
        return jsonify({"error": str(e)}), 500

@story_bp.route("/generate_multi_npc_scene", methods=["POST"])
@timed_function(name="generate_scene")
async def generate_scene():
    """Generate a multi-NPC scene with enhanced integration."""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        conv_id = data.get('conversation_id')
        location = data.get('location')
        npc_ids = data.get('npc_ids', [])
        
        # Get story context
        story_context = await get_story_context(user_id, conv_id)
        
        # Get active conflicts
        conflict_system = await get_conflict_system(user_id, conv_id)
        active_conflicts = await conflict_system.get_active_conflicts()
        
        # Get relevant lore
        lore_system = await get_lore_system(user_id, conv_id)
        lore_context = await lore_system.get_narrative_elements(story_context.get('current_narrative_id'))
        
        # Get NPC system
        npc_system = await get_npc_system(user_id, conv_id)
        
        # Generate scene with enhanced context
        scene = await npc_system.generate_scene(
            location=location,
            npc_ids=npc_ids,
            context={
                'story_context': story_context,
                'active_conflicts': active_conflicts,
                'lore_context': lore_context
            }
        )
        
        return {
            'success': True,
            'scene': scene,
            'story_context': story_context,
            'active_conflicts': active_conflicts
        }
        
    except Exception as e:
        logger.error(f"Error generating scene: {e}")
        return {'success': False, 'error': str(e)}

@story_bp.route("/generate_overheard_conversation", methods=["POST"])
@timed_function(name="generate_conversation")
async def generate_conversation():
    """Generate overheard conversation with enhanced integration."""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        conv_id = data.get('conversation_id')
        npc_ids = data.get('npc_ids', [])
        
        # Get story context
        story_context = await get_story_context(user_id, conv_id)
        
        # Get active conflicts
        conflict_system = await get_conflict_system(user_id, conv_id)
        active_conflicts = await conflict_system.get_active_conflicts()
        
        # Get relevant lore
        lore_system = await get_lore_system(user_id, conv_id)
        lore_context = await lore_system.get_narrative_elements(story_context.get('current_narrative_id'))
        
        # Get NPC system
        npc_system = await get_npc_system(user_id, conv_id)
        
        # Generate conversation with enhanced context
        conversation = await npc_system.generate_conversation(
            npc_ids=npc_ids,
            context={
                'story_context': story_context,
                'active_conflicts': active_conflicts,
                'lore_context': lore_context
            }
        )
        
        return {
            'success': True,
            'conversation': conversation,
            'story_context': story_context,
            'active_conflicts': active_conflicts
        }
        
    except Exception as e:
        logger.error(f"Error generating conversation: {e}")
        return {'success': False, 'error': str(e)}

@story_bp.route("/events", methods=["GET"])
@timed_function
async def get_events():
    """
    Get events happening in the game world.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
            
        conversation_id = request.args.get("conversation_id")
        filter_type = request.args.get("filter", "upcoming")  # upcoming, past, all
        
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        # Get current time to filter events
        async with get_db_connection_context() as conn:
            current_time = await get_current_time_data(conn, user_id, conversation_id)
            
            # Query events
            if filter_type == "upcoming":
                # Get events in the future
                rows = await conn.fetch("""
                    SELECT id, event_name, description, start_time, end_time, location,
                           year, month, day, time_of_day
                    FROM Events
                    WHERE user_id = $1 AND conversation_id = $2
                    AND (year > $3 
                         OR (year = $3 AND month > $4)
                         OR (year = $3 AND month = $4 AND day > $5)
                         OR (year = $3 AND month = $4 AND day = $5 AND 
                             CASE 
                                WHEN time_of_day = 'Morning' THEN 1
                                WHEN time_of_day = 'Afternoon' THEN 2
                                WHEN time_of_day = 'Evening' THEN 3
                                WHEN time_of_day = 'Night' THEN 4
                             END >
                             CASE 
                                WHEN $6 = 'Morning' THEN 1
                                WHEN $6 = 'Afternoon' THEN 2
                                WHEN $6 = 'Evening' THEN 3
                                WHEN $6 = 'Night' THEN 4
                             END
                            )
                        )
                    ORDER BY year, month, day, 
                        CASE 
                            WHEN time_of_day = 'Morning' THEN 1
                            WHEN time_of_day = 'Afternoon' THEN 2
                            WHEN time_of_day = 'Evening' THEN 3
                            WHEN time_of_day = 'Night' THEN 4
                        END
                """, user_id, conversation_id, current_time["year"], current_time["month"], 
                    current_time["day"], current_time["time_of_day"])
            elif filter_type == "past":
                # Get events in the past
                rows = await conn.fetch("""
                    SELECT id, event_name, description, start_time, end_time, location,
                           year, month, day, time_of_day
                    FROM Events
                    WHERE user_id = $1 AND conversation_id = $2
                    AND (year < $3 
                         OR (year = $3 AND month < $4)
                         OR (year = $3 AND month = $4 AND day < $5)
                         OR (year = $3 AND month = $4 AND day = $5 AND 
                             CASE 
                                WHEN time_of_day = 'Morning' THEN 1
                                WHEN time_of_day = 'Afternoon' THEN 2
                                WHEN time_of_day = 'Evening' THEN 3
                                WHEN time_of_day = 'Night' THEN 4
                             END <
                             CASE 
                                WHEN $6 = 'Morning' THEN 1
                                WHEN $6 = 'Afternoon' THEN 2
                                WHEN $6 = 'Evening' THEN 3
                                WHEN $6 = 'Night' THEN 4
                             END
                            )
                        )
                    ORDER BY year DESC, month DESC, day DESC, 
                        CASE 
                            WHEN time_of_day = 'Morning' THEN 1
                            WHEN time_of_day = 'Afternoon' THEN 2
                            WHEN time_of_day = 'Evening' THEN 3
                            WHEN time_of_day = 'Night' THEN 4
                        END DESC
                """, user_id, conversation_id, current_time["year"], current_time["month"], 
                    current_time["day"], current_time["time_of_day"])
            else:
                # Get all events
                rows = await conn.fetch("""
                    SELECT id, event_name, description, start_time, end_time, location,
                           year, month, day, time_of_day
                    FROM Events
                    WHERE user_id = $1 AND conversation_id = $2
                    ORDER BY year, month, day, 
                        CASE 
                            WHEN time_of_day = 'Morning' THEN 1
                            WHEN time_of_day = 'Afternoon' THEN 2
                            WHEN time_of_day = 'Evening' THEN 3
                            WHEN time_of_day = 'Night' THEN 4
                        END
                """, user_id, conversation_id)
            
            # Format events
            events = []
            for row in rows:
                events.append({
                    "event_id": row["id"],
                    "event_name": row["event_name"],
                    "description": row["description"],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "location": row["location"],
                    "year": row["year"],
                    "month": row["month"],
                    "day": row["day"],
                    "time_of_day": row["time_of_day"],
                    "is_current": (
                        row["year"] == current_time["year"] and
                        row["month"] == current_time["month"] and
                        row["day"] == current_time["day"] and
                        row["time_of_day"] == current_time["time_of_day"]
                    )
                })
            
            return jsonify({
                "events": events,
                "current_time": current_time,
                "filter": filter_type
            })
        
    except Exception as e:
        logging.exception("[get_events] Error")
        return jsonify({"error": str(e)}), 500

@story_bp.route("/end_of_day", methods=["POST"])
@timed_function
async def end_of_day():
    """
    Process end of day actions, including:
    - Nightly maintenance for NPCs
    - Memory lifecycle management
    - Daily income and resource changes
    - Conflict updates
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

        # Run comprehensive memory lifecycle management
        maintenance_result = await manage_npc_memory_lifecycle(npc_system)

        # Process daily income and resource changes
        resource_manager = ResourceManager(user_id, int(conv_id))

        # Base daily income
        money_income = 10
        supplies_income = 5
        influence_income = 2

        # Apply income
        money_result = await resource_manager.modify_money(
            money_income, "daily_income", "Daily income"
        )

        supplies_result = await resource_manager.modify_supplies(
            supplies_income, "daily_income", "Daily supplies"
        )

        influence_result = await resource_manager.modify_influence(
            influence_income, "daily_income", "Daily influence gain"
        )

        # Recover some energy overnight
        energy_result = await resource_manager.modify_energy(
            20, "rest", "Overnight rest"  # Recover 20 energy
        )

        # Decrease hunger (negative = more hungry)
        hunger_result = await resource_manager.modify_hunger(
            -15, "metabolism", "Overnight metabolism"
        )

        # Process conflict daily updates
        conflict_integration = ConflictSystemIntegration(user_id, int(conv_id))
        conflict_update = await conflict_integration.run_daily_update()

        # Clear caches for new day
        NPC_CACHE.clear()
        LOCATION_CACHE.clear()
        AGGREGATOR_CACHE.clear()
        TIME_CACHE.clear()

        resource_changes = {
            "money": money_result,
            "supplies": supplies_result,
            "influence": influence_result,
            "energy": energy_result,
            "hunger": hunger_result
        }

        return jsonify({
            "status": "Nightly maintenance complete",
            "memory_maintenance": maintenance_result,
            "resource_changes": resource_changes,
            "conflict_update": conflict_update,
            "current_resources": await resource_manager.get_resources(),
            "current_vitals": await resource_manager.get_vitals()
        })

    except Exception as e:
        logger.error(f"Error during end_of_day: {e}", exc_info=True)
        return jsonify({"error": "Server error", "details": str(e)}), 500

@timed_function(name="process_user_message")
async def process_user_message(user_id, conv_id, user_input):
    """
    Process and store user message.
    
    This is a helper function, not a route handler.
    
    Args:
        user_id: User ID
        conv_id: Conversation ID
        user_input: User's message
        
    Returns:
        Dictionary with status
    """
    async with db_transaction() as conn:
        await conn.execute("""
            INSERT INTO messages (conversation_id, sender, content)
            VALUES ($1, $2, $3)
        """, conv_id, "user", user_input)
        return {"status": "stored"}

async def create_npc_with_new_handler(user_id, conv_id, env_desc, archetype_names=None):
    """
    Create an NPC with the new NPCCreationHandler
    
    This is a helper function, not a route handler.
    
    Args:
        user_id: User ID
        conv_id: Conversation ID
        env_desc: Environment description
        archetype_names: Optional list of archetype names
        
    Returns:
        Dictionary with NPC data or error
    """
    try:
        # Create NPCCreationHandler
        npc_handler = NPCCreationHandler()
        
        # Create NPC
        npc_result = await npc_handler.create_npc_with_context(
            environment_desc=env_desc,
            archetype_names=archetype_names,
            user_id=user_id,
            conversation_id=conv_id
        )
        
        # Convert to dict format
        return {
            "npc_id": npc_result.npc_id,
            "npc_name": npc_result.npc_name,
            "physical_description": npc_result.physical_description,
            "personality": npc_result.personality.dict() if hasattr(npc_result.personality, 'dict') else npc_result.personality,
            "stats": npc_result.stats.dict() if hasattr(npc_result.stats, 'dict') else npc_result.stats,
            "archetypes": npc_result.archetypes.dict() if hasattr(npc_result.archetypes, 'dict') else npc_result.archetypes,
            "schedule": npc_result.schedule,
            "memories": npc_result.memories,
            "current_location": npc_result.current_location
        }
    except Exception as e:
        logging.error(f"Error creating NPC: {e}")
        return {"error": str(e)}
