# story_agent/tools.py

"""
Organized tools for the Story Director agent.

This module organizes tools into logical categories:
- story_tools: Tools for general story state and progression
- conflict_tools: Tools for managing conflicts
- resource_tools: Tools for resource management
- narrative_tools: Tools for narrative elements
"""

import logging
import json
import asyncio
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field

from agents import function_tool, RunContextWrapper

from db.connection import get_db_connection
from logic.narrative_progression import (
    get_current_narrative_stage, 
    check_for_personal_revelations,
    check_for_narrative_moments,
    check_for_npc_revelations,
    add_dream_sequence,
    add_moment_of_clarity,
    NARRATIVE_STAGES
)
from logic.social_links_agentic import (
    get_social_link,
    get_relationship_summary,
    check_for_relationship_crossroads,
    check_for_relationship_ritual,
    apply_crossroads_choice
)

logger = logging.getLogger(__name__)

# ----- Story State Tools -----

@function_tool
async def get_story_state(ctx) -> Dict[str, Any]:
    """
    Get the current state of the story, including active conflicts, narrative stage, 
    resources, and any pending narrative events.
    
    Returns:
        A dictionary containing the current story state
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    conflict_manager = context.conflict_manager
    resource_manager = context.resource_manager
    
    try:
        # Get current narrative stage
        narrative_stage = await get_current_narrative_stage(user_id, conversation_id)
        stage_info = None
        if narrative_stage:
            stage_info = {
                "name": narrative_stage.name,
                "description": narrative_stage.description
            }
        
        # Get active conflicts
        active_conflicts = await conflict_manager.get_active_conflicts()
        conflict_infos = []
        for conflict in active_conflicts:
            conflict_infos.append({
                "conflict_id": conflict['conflict_id'],
                "conflict_name": conflict['conflict_name'],
                "conflict_type": conflict['conflict_type'],
                "description": conflict['description'],
                "phase": conflict['phase'],
                "progress": conflict['progress'],
                "faction_a_name": conflict['faction_a_name'],
                "faction_b_name": conflict['faction_b_name']
            })
        
        # Get key NPCs
        key_npcs = await get_key_npcs(ctx, limit=5)
        
        # Get player resources and vitals
        resources = await resource_manager.get_resources()
        vitals = await resource_manager.get_vitals()
        
        # Format currency for display
        formatted_money = await resource_manager.get_formatted_money()
        
        resource_status = {
            "money": resources.get('money', 0),
            "supplies": resources.get('supplies', 0),
            "influence": resources.get('influence', 0),
            "energy": vitals.get('energy', 0),
            "hunger": vitals.get('hunger', 0),
            "formatted_money": formatted_money
        }
        
        # Check for narrative events
        narrative_events = []
        
        # Personal revelations
        personal_revelation = await check_for_personal_revelations(user_id, conversation_id)
        if personal_revelation:
            narrative_events.append({
                "event_type": "personal_revelation",
                "content": personal_revelation,
                "should_present": True,
                "priority": 8
            })
        
        # Narrative moments
        narrative_moment = await check_for_narrative_moments(user_id, conversation_id)
        if narrative_moment:
            narrative_events.append({
                "event_type": "narrative_moment",
                "content": narrative_moment,
                "should_present": True,
                "priority": 9
            })
        
        # NPC revelations
        npc_revelation = await check_for_npc_revelations(user_id, conversation_id)
        if npc_revelation:
            narrative_events.append({
                "event_type": "npc_revelation",
                "content": npc_revelation,
                "should_present": True,
                "priority": 7
            })
        
        # Check for relationship events
        crossroads = await check_for_relationship_crossroads(user_id, conversation_id)
        ritual = await check_for_relationship_ritual(user_id, conversation_id)
        
        # Generate key observations based on current state
        key_observations = []
        
        # If at a higher corruption stage, add observation
        if narrative_stage and narrative_stage.name in ["Creeping Realization", "Veil Thinning", "Full Revelation"]:
            key_observations.append(f"Player has progressed to {narrative_stage.name} stage, indicating significant corruption")
        
        # If multiple active conflicts, note this
        if len(conflict_infos) > 2:
            key_observations.append(f"Player is juggling {len(conflict_infos)} active conflicts, which may be overwhelming")
        
        # If any major or catastrophic conflicts, highlight them
        major_conflicts = [c for c in conflict_infos if c["conflict_type"] in ["major", "catastrophic"]]
        if major_conflicts:
            conflict_names = ", ".join([c["conflict_name"] for c in major_conflicts])
            key_observations.append(f"Major conflicts in progress: {conflict_names}")
        
        # If resources are low, note this
        if resource_status["money"] < 30:
            key_observations.append("Player is low on money, which may limit conflict involvement options")
        
        if resource_status["energy"] < 30:
            key_observations.append("Player energy is low, which may affect capability in conflicts")
        
        if resource_status["hunger"] < 30:
            key_observations.append("Player is hungry, which may distract from conflict progress")
        
        # Determine overall story direction
        story_direction = ""
        if narrative_stage:
            if narrative_stage.name == "Innocent Beginning":
                story_direction = "Introduce subtle hints of control dynamics while maintaining a veneer of normalcy"
            elif narrative_stage.name == "First Doubts":
                story_direction = "Create situations that highlight inconsistencies in NPC behavior, raising questions"
            elif narrative_stage.name == "Creeping Realization":
                story_direction = "NPCs should be more open about their manipulative behavior, testing boundaries"
            elif narrative_stage.name == "Veil Thinning":
                story_direction = "Dominant characters should drop pretense more frequently, openly directing the player"
            elif narrative_stage.name == "Full Revelation":
                story_direction = "The true nature of relationships should be explicit, with NPCs acknowledging their control"
        
        return {
            "narrative_stage": stage_info,
            "active_conflicts": conflict_infos,
            "narrative_events": narrative_events,
            "key_npcs": key_npcs,
            "resources": resource_status,
            "key_observations": key_observations,
            "relationship_crossroads": crossroads,
            "relationship_ritual": ritual,
            "story_direction": story_direction,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting story state: {str(e)}", exc_info=True)
        return {
            "error": f"Failed to get story state: {str(e)}",
            "narrative_stage": None,
            "active_conflicts": [],
            "narrative_events": [],
            "key_npcs": [],
            "resources": {},
            "last_updated": datetime.now().isoformat()
        }

@function_tool
async def get_key_npcs(ctx, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get the key NPCs in the current game state, ordered by importance.
    
    Args:
        limit: Maximum number of NPCs to return
        
    Returns:
        List of NPC information dictionaries
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get NPCs ordered by dominance (a proxy for importance)
        cursor.execute("""
            SELECT npc_id, npc_name, dominance, cruelty, closeness, trust, respect
            FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND introduced=TRUE
            ORDER BY dominance DESC
            LIMIT %s
        """, (user_id, conversation_id, limit))
        
        npcs = []
        for row in cursor.fetchall():
            npc_id, npc_name, dominance, cruelty, closeness, trust, respect = row
            
            # Get relationship with player
            relationship = await get_relationship_summary(
                user_id, conversation_id, 
                "player", user_id, "npc", npc_id
            )
            
            dynamics = {}
            if relationship and 'dynamics' in relationship:
                dynamics = relationship['dynamics']
            
            npcs.append({
                "npc_id": npc_id,
                "npc_name": npc_name,
                "dominance": dominance,
                "cruelty": cruelty,
                "closeness": closeness,
                "trust": trust,
                "respect": respect,
                "relationship_dynamics": dynamics
            })
        
        return npcs
    except Exception as e:
        logger.error(f"Error fetching key NPCs: {str(e)}", exc_info=True)
        return []
    finally:
        cursor.close()
        conn.close()

@function_tool
async def get_narrative_stages(ctx) -> List[Dict[str, str]]:
    """
    Get information about all narrative stages in the game.
    
    Returns:
        List of narrative stages with their descriptions
    """
    stages = []
    for stage in NARRATIVE_STAGES:
        stages.append({
            "name": stage.name,
            "description": stage.description
        })
    return stages

@function_tool
async def analyze_narrative_and_activity(
    ctx,
    narrative_text: str,
    player_activity: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive analysis of narrative text and player activity to determine
    impacts on conflicts, resources, and story progression.
    
    Args:
        narrative_text: The narrative description
        player_activity: Optional specific player activity description
        
    Returns:
        Comprehensive analysis results
    """
    context = ctx.context
    conflict_manager = context.conflict_manager
    
    try:
        # Start with conflict analysis
        conflict_analysis = await conflict_manager.add_conflict_to_narrative(narrative_text)
        
        results = {
            "conflict_analysis": conflict_analysis,
            "activity_effects": None,
            "relationship_impacts": [],
            "resource_changes": {},
            "conflict_progression": []
        }
        
        # If player activity is provided, analyze it
        if player_activity:
            activity_analyzer = context.activity_analyzer
            activity_effects = await activity_analyzer.analyze_activity(
                player_activity, apply_effects=False
            )
            results["activity_effects"] = activity_effects
            
            # Check if this activity might progress any conflicts
            active_conflicts = await conflict_manager.get_active_conflicts()
            
            for conflict in active_conflicts:
                # Simple relevance check - see if keywords from conflict appear in activity
                conflict_keywords = [
                    conflict['conflict_name'],
                    conflict['faction_a_name'],
                    conflict['faction_b_name']
                ]
                
                relevant = any(keyword.lower() in player_activity.lower() for keyword in conflict_keywords if keyword)
                
                if relevant:
                    # Determine an appropriate progress increment
                    progress_increment = 5  # Default increment
                    
                    if "actively" in player_activity.lower() or "directly" in player_activity.lower():
                        progress_increment = 10
                    
                    if conflict['conflict_type'] == "major":
                        progress_increment = progress_increment * 0.5  # Major conflicts progress slower
                    elif conflict['conflict_type'] == "minor":
                        progress_increment = progress_increment * 1.5  # Minor conflicts progress faster
                    
                    # Add to results
                    results["conflict_progression"].append({
                        "conflict_id": conflict['conflict_id'],
                        "conflict_name": conflict['conflict_name'],
                        "is_relevant": True,
                        "suggested_progress_increment": progress_increment
                    })
        
        return results
    except Exception as e:
        logger.error(f"Error analyzing narrative and activity: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "conflict_analysis": {"conflict_generated": False},
            "activity_effects": None,
            "relationship_impacts": [],
            "resource_changes": {},
            "conflict_progression": []
        }

# ----- Conflict Tools -----

@function_tool
async def generate_conflict(ctx, conflict_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a new conflict of the specified type, or determine the appropriate type
    based on current game state if none specified.
    
    Args:
        conflict_type: Optional type of conflict to generate (major, minor, standard, catastrophic)
        
    Returns:
        Information about the generated conflict
    """
    context = ctx.context
    conflict_manager = context.conflict_manager
    
    try:
        conflict = await conflict_manager.generate_conflict(conflict_type)
        
        return {
            "conflict_id": conflict['conflict_id'],
            "conflict_name": conflict['conflict_name'],
            "conflict_type": conflict['conflict_type'],
            "description": conflict['description'],
            "success": True,
            "message": "Conflict generated successfully"
        }
    except Exception as e:
        logger.error(f"Error generating conflict: {str(e)}", exc_info=True)
        return {
            "conflict_id": 0,
            "conflict_name": "",
            "conflict_type": conflict_type or "unknown",
            "description": "",
            "success": False,
            "message": f"Failed to generate conflict: {str(e)}"
        }

@function_tool
async def update_conflict_progress(
    ctx, 
    conflict_id: int, 
    progress_increment: float
) -> Dict[str, Any]:
    """
    Update the progress of a conflict.
    
    Args:
        conflict_id: ID of the conflict to update
        progress_increment: Amount to increment the progress (0-100)
        
    Returns:
        Updated conflict information
    """
    context = ctx.context
    conflict_manager = context.conflict_manager
    
    try:
        # Get current conflict info
        old_conflict = await conflict_manager.get_conflict(conflict_id)
        old_phase = old_conflict['phase']
        
        # Update progress
        updated_conflict = await conflict_manager.update_conflict_progress(conflict_id, progress_increment)
        
        return {
            "conflict_id": conflict_id,
            "new_progress": updated_conflict['progress'],
            "new_phase": updated_conflict['phase'],
            "phase_changed": updated_conflict['phase'] != old_phase,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error updating conflict progress: {str(e)}", exc_info=True)
        return {
            "conflict_id": conflict_id,
            "new_progress": 0,
            "new_phase": "unknown",
            "phase_changed": False,
            "success": False,
            "error": str(e)
        }

@function_tool
async def resolve_conflict(ctx, conflict_id: int) -> Dict[str, Any]:
    """
    Resolve a conflict and apply consequences.
    
    Args:
        conflict_id: ID of the conflict to resolve
        
    Returns:
        Information about the conflict resolution
    """
    context = ctx.context
    conflict_manager = context.conflict_manager
    
    try:
        result = await conflict_manager.resolve_conflict(conflict_id)
        
        consequences = []
        for consequence in result.get('consequences', []):
            consequences.append(consequence.get('description', ''))
        
        return {
            "conflict_id": conflict_id,
            "outcome": result.get('outcome', 'unknown'),
            "consequences": consequences,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error resolving conflict: {str(e)}", exc_info=True)
        return {
            "conflict_id": conflict_id,
            "outcome": "error",
            "consequences": [f"Error: {str(e)}"],
            "success": False
        }

@function_tool
async def analyze_narrative_for_conflict(ctx, narrative_text: str) -> Dict[str, Any]:
    """
    Analyze a narrative text to see if it should trigger a conflict.
    
    Args:
        narrative_text: The narrative text to analyze
        
    Returns:
        Analysis results and possibly a new conflict
    """
    context = ctx.context
    conflict_manager = context.conflict_manager
    
    try:
        result = await conflict_manager.add_conflict_to_narrative(narrative_text)
        return result
    except Exception as e:
        logger.error(f"Error analyzing narrative for conflict: {str(e)}", exc_info=True)
        return {
            "analysis": {
                "conflict_intensity": 0,
                "matched_keywords": []
            },
            "conflict_generated": False,
            "error": str(e)
        }

@function_tool
async def set_player_involvement(
    ctx, 
    conflict_id: int, 
    involvement_level: str,
    faction: str = "neutral",
    money_committed: int = 0,
    supplies_committed: int = 0,
    influence_committed: int = 0,
    action: Optional[str] = None
) -> Dict[str, Any]:
    """
    Set the player's involvement in a conflict.
    
    Args:
        conflict_id: ID of the conflict
        involvement_level: Level of involvement (none, observing, participating, leading)
        faction: Which faction to support (a, b, neutral)
        money_committed: Money committed to the conflict
        supplies_committed: Supplies committed to the conflict
        influence_committed: Influence committed to the conflict
        action: Optional specific action taken
        
    Returns:
        Updated conflict information
    """
    context = ctx.context
    conflict_manager = context.conflict_manager
    
    try:
        # First check if player has sufficient resources
        resource_manager = context.resource_manager
        resource_check = await resource_manager.check_resources(
            money_committed, supplies_committed, influence_committed
        )
        
        if not resource_check['has_resources']:
            return {
                "error": "Insufficient resources to commit",
                "missing": resource_check.get('missing', {}),
                "current": resource_check.get('current', {}),
                "success": False
            }
        
        result = await conflict_manager.set_player_involvement(
            conflict_id, involvement_level, faction,
            money_committed, supplies_committed, influence_committed, action
        )
        
        # Add success flag
        if isinstance(result, dict):
            result["success"] = True
        else:
            # If the result is not a dictionary, create a new one
            result = {
                "conflict_id": conflict_id,
                "involvement_level": involvement_level,
                "faction": faction,
                "resources_committed": {
                    "money": money_committed,
                    "supplies": supplies_committed,
                    "influence": influence_committed
                },
                "action": action,
                "success": True
            }
        
        return result
    except Exception as e:
        logger.error(f"Error setting involvement: {str(e)}", exc_info=True)
        return {
            "conflict_id": conflict_id,
            "error": str(e),
            "success": False
        }

@function_tool
async def get_conflict_details(ctx, conflict_id: int) -> Dict[str, Any]:
    """
    Get detailed information about a specific conflict.
    
    Args:
        conflict_id: ID of the conflict
        
    Returns:
        Detailed conflict information
    """
    context = ctx.context
    conflict_manager = context.conflict_manager
    
    try:
        conflict = await conflict_manager.get_conflict(conflict_id)
        
        # Get additional information about involved NPCs
        involved_npcs = await conflict_manager.get_conflict_npcs(conflict_id)
        
        # Get player involvement
        player_involvement = await conflict_manager.get_player_involvement(conflict_id)
        
        # Create a comprehensive response
        result = {
            "conflict_id": conflict_id,
            "conflict_name": conflict.get('conflict_name', ''),
            "conflict_type": conflict.get('conflict_type', ''),
            "description": conflict.get('description', ''),
            "phase": conflict.get('phase', ''),
            "progress": conflict.get('progress', 0),
            "faction_a_name": conflict.get('faction_a_name', ''),
            "faction_b_name": conflict.get('faction_b_name', ''),
            "involved_npcs": involved_npcs,
            "player_involvement": player_involvement,
            "start_day": conflict.get('start_day', 0),
            "estimated_duration": conflict.get('estimated_duration', 0),
            "resources_required": conflict.get('resources_required', {}),
            "success_rate": conflict.get('success_rate', 0)
        }
        
        return result
    except Exception as e:
        logger.error(f"Error getting conflict details: {str(e)}", exc_info=True)
        return {
            "conflict_id": conflict_id,
            "error": f"Failed to get conflict details: {str(e)}",
            "success": False
        }

# ----- Resource Tools -----

@function_tool
async def check_resources(ctx, money: int = 0, supplies: int = 0, influence: int = 0) -> Dict[str, Any]:
    """
    Check if player has sufficient resources.
    
    Args:
        money: Required amount of money
        supplies: Required amount of supplies
        influence: Required amount of influence
        
    Returns:
        Dictionary with resource check results
    """
    context = ctx.context
    resource_manager = context.resource_manager
    
    try:
        result = await resource_manager.check_resources(money, supplies, influence)
        # Add formatted money
        if result.get('current', {}).get('money') is not None:
            formatted_money = await resource_manager.get_formatted_money(result['current']['money'])
            result['current']['formatted_money'] = formatted_money
        
        return result
    except Exception as e:
        logger.error(f"Error checking resources: {str(e)}", exc_info=True)
        return {
            "has_resources": False,
            "error": str(e),
            "current": {}
        }

@function_tool
async def commit_resources_to_conflict(
    ctx, 
    conflict_id: int, 
    money: int = 0,
    supplies: int = 0,
    influence: int = 0
) -> Dict[str, Any]:
    """
    Commit player resources to a conflict.
    
    Args:
        conflict_id: ID of the conflict
        money: Amount of money to commit
        supplies: Amount of supplies to commit
        influence: Amount of influence to commit
        
    Returns:
        Result of committing resources
    """
    context = ctx.context
    resource_manager = context.resource_manager
    
    try:
        result = await resource_manager.commit_resources_to_conflict(
            conflict_id, money, supplies, influence
        )
        
        # Add formatted money if money was committed
        if money > 0 and result.get('money_result'):
            money_result = result['money_result']
            if 'old_value' in money_result and 'new_value' in money_result:
                old_formatted = await resource_manager.get_formatted_money(money_result['old_value'])
                new_formatted = await resource_manager.get_formatted_money(money_result['new_value'])
                money_result['formatted_old_value'] = old_formatted
                money_result['formatted_new_value'] = new_formatted
                money_result['formatted_change'] = await resource_manager.get_formatted_money(money_result['change'])
                result['money_result'] = money_result
        
        return result
    except Exception as e:
        logger.error(f"Error committing resources: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

@function_tool
async def get_player_resources(ctx) -> Dict[str, Any]:
    """
    Get the current player resources and vitals.
    
    Returns:
        Current resource status
    """
    context = ctx.context
    resource_manager = context.resource_manager
    
    try:
        resources = await resource_manager.get_resources()
        vitals = await resource_manager.get_vitals()
        
        # Get formatted money
        formatted_money = await resource_manager.get_formatted_money()
        
        return {
            "money": resources.get('money', 0),
            "supplies": resources.get('supplies', 0),
            "influence": resources.get('influence', 0),
            "energy": vitals.get('energy', 0),
            "hunger": vitals.get('hunger', 0),
            "formatted_money": formatted_money,
            "updated_at": resources.get('updated_at', datetime.now()).isoformat() if isinstance(resources.get('updated_at'), datetime) else str(resources.get('updated_at'))
        }
    except Exception as e:
        logger.error(f"Error getting player resources: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "money": 0,
            "supplies": 0,
            "influence": 0,
            "energy": 0,
            "hunger": 0,
            "formatted_money": "0"
        }

@function_tool
async def analyze_activity_effects(ctx, activity_text: str) -> Dict[str, Any]:
    """
    Analyze an activity to determine its effects on player resources.
    
    Args:
        activity_text: Description of the activity
        
    Returns:
        Activity effects
    """
    context = ctx.context
    activity_analyzer = context.activity_analyzer
    
    try:
        # Don't apply effects, just analyze them
        result = await activity_analyzer.analyze_activity(activity_text, apply_effects=False)
        
        effects = result.get('effects', {})
        
        return {
            "activity_type": result.get('activity_type', 'unknown'),
            "activity_details": result.get('activity_details', ''),
            "hunger_effect": effects.get('hunger'),
            "energy_effect": effects.get('energy'),
            "money_effect": effects.get('money'),
            "supplies_effect": effects.get('supplies'),
            "influence_effect": effects.get('influence'),
            "description": result.get('description', f"Effects of {activity_text}")
        }
    except Exception as e:
        logger.error(f"Error analyzing activity effects: {str(e)}", exc_info=True)
        return {
            "activity_type": "unknown",
            "activity_details": "",
            "description": f"Failed to analyze: {str(e)}",
            "error": str(e)
        }

@function_tool
async def apply_activity_effects(ctx, activity_text: str) -> Dict[str, Any]:
    """
    Analyze and apply the effects of an activity to player resources.
    
    Args:
        activity_text: Description of the activity
        
    Returns:
        Results of applying activity effects
    """
    context = ctx.context
    activity_analyzer = context.activity_analyzer
    
    try:
        # Apply the effects
        result = await activity_analyzer.analyze_activity(activity_text, apply_effects=True)
        
        # Add formatted money if money was affected
        if 'effects' in result and 'money' in result['effects']:
            resource_manager = context.resource_manager
            resources = await resource_manager.get_resources()
            result['formatted_money'] = await resource_manager.get_formatted_money(resources.get('money', 0))
        
        return result
    except Exception as e:
        logger.error(f"Error applying activity effects: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "activity_type": "unknown",
            "activity_details": "",
            "effects": {}
        }

@function_tool
async def get_resource_history(ctx, resource_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get the history of resource changes.
    
    Args:
        resource_type: Optional filter for specific resource type
                      (money, supplies, influence, energy, hunger)
        limit: Maximum number of history entries to return
        
    Returns:
        List of resource change history entries
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if resource_type:
            cursor.execute("""
                SELECT resource_type, old_value, new_value, amount_changed, 
                       source, description, timestamp
                FROM ResourceHistoryLog
                WHERE user_id=%s AND conversation_id=%s AND resource_type=%s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (user_id, conversation_id, resource_type, limit))
        else:
            cursor.execute("""
                SELECT resource_type, old_value, new_value, amount_changed, 
                       source, description, timestamp
                FROM ResourceHistoryLog
                WHERE user_id=%s AND conversation_id=%s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (user_id, conversation_id, limit))
        
        history = []
        for row in cursor.fetchall():
            resource_type, old_value, new_value, amount_changed, source, description, timestamp = row
            
            # Format money values if resource_type is money
            formatted_old = None
            formatted_new = None
            formatted_change = None
            
            if resource_type == "money":
                resource_manager = context.resource_manager
                formatted_old = await resource_manager.get_formatted_money(old_value)
                formatted_new = await resource_manager.get_formatted_money(new_value)
                formatted_change = await resource_manager.get_formatted_money(amount_changed)
            
            history.append({
                "resource_type": resource_type,
                "old_value": old_value,
                "new_value": new_value,
                "amount_changed": amount_changed,
                "formatted_old_value": formatted_old,
                "formatted_new_value": formatted_new,
                "formatted_change": formatted_change,
                "source": source,
                "description": description,
                "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
            })
        
        return history
    except Exception as e:
        logger.error(f"Error getting resource history: {str(e)}", exc_info=True)
        return []
    finally:
        cursor.close()
        conn.close()

# ----- Narrative Tools -----

@function_tool
async def generate_personal_revelation(ctx, npc_name: str, revelation_type: str) -> Dict[str, Any]:
    """
    Generate a personal revelation for the player about their relationship with an NPC.
    
    Args:
        npc_name: Name of the NPC involved in the revelation
        revelation_type: Type of revelation (dependency, obedience, corruption, willpower, confidence)
        
    Returns:
        A personal revelation
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    # Define revelation templates based on type
    templates = {
        "dependency": [
            "I've been checking my phone constantly to see if {npc_name} has messaged me. When did I start needing her approval so much?",
            "I realized today that I haven't made a significant decision without consulting {npc_name} in weeks. Is that normal?",
            "The thought of spending a day without talking to {npc_name} makes me anxious. I should be concerned about that, shouldn't I?"
        ],
        "obedience": [
            "I caught myself automatically rearranging my schedule when {npc_name} hinted she wanted to see me. I didn't even think twice about it.",
            "Today I changed my opinion the moment I realized it differed from {npc_name}'s. That's... not like me. Or is it becoming like me?",
            "{npc_name} gave me that look, and I immediately stopped what I was saying. When did her disapproval start carrying so much weight?"
        ],
        "corruption": [
            "I found myself enjoying the feeling of following {npc_name}'s instructions perfectly. The pride I felt at her approval was... intense.",
            "Last year, I would have been offended if someone treated me the way {npc_name} did today. Now I'm grateful for her attention.",
            "Sometimes I catch glimpses of my old self, like a stranger I used to know. When did I change so fundamentally?"
        ],
        "willpower": [
            "I had every intention of saying no to {npc_name} today. The 'yes' came out before I even realized I was speaking.",
            "I've been trying to remember what it felt like to disagree with {npc_name}. The memory feels distant, like it belongs to someone else.",
            "I made a list of boundaries I wouldn't cross. Looking at it now, I've broken every single one at {npc_name}'s suggestion."
        ],
        "confidence": [
            "I opened my mouth to speak in the meeting, then saw {npc_name} watching me. I suddenly couldn't remember what I was going to say.",
            "I used to trust my judgment. Now I find myself second-guessing every thought that {npc_name} hasn't explicitly approved.",
            "When did I start feeling this small? This uncertain? I can barely remember how it felt to be sure of myself."
        ]
    }
    
    try:
        # Default to dependency if type not found
        revelation_templates = templates.get(revelation_type.lower(), templates["dependency"])
        
        # Select a random template and format it
        inner_monologue = random.choice(revelation_templates).format(npc_name=npc_name)
        
        # Add to PlayerJournal
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, revelation_types, timestamp)
                VALUES (%s, %s, 'personal_revelation', %s, %s, CURRENT_TIMESTAMP)
                RETURNING id
            """, (user_id, conversation_id, inner_monologue, revelation_type))
            
            journal_id = cursor.fetchone()[0]
            conn.commit()
            
            return {
                "type": "personal_revelation",
                "name": f"{revelation_type.capitalize()} Awareness",
                "inner_monologue": inner_monologue,
                "journal_id": journal_id,
                "success": True
            }
        except Exception as db_error:
            conn.rollback()
            logger.error(f"Database error recording personal revelation: {db_error}")
            raise
        finally:
            cursor.close()
            conn.close()
    except Exception as e:
        logger.error(f"Error generating personal revelation: {str(e)}", exc_info=True)
        return {
            "type": "personal_revelation",
            "name": f"{revelation_type.capitalize()} Awareness",
            "inner_monologue": f"Error generating revelation: {str(e)}",
            "success": False
        }

@function_tool
async def generate_dream_sequence(ctx, npc_names: List[str]) -> Dict[str, Any]:
    """
    Generate a symbolic dream sequence based on player's current state.
    
    Args:
        npc_names: List of NPC names to include in the dream
        
    Returns:
        A dream sequence
    """
    # Ensure we have at least 3 NPC names
    while len(npc_names) < 3:
        npc_names.append(f"Unknown Woman {len(npc_names) + 1}")
    
    npc1, npc2, npc3 = npc_names[:3]
    
    # Dream templates
    dream_templates = [
        "You're sitting in a chair as {npc1} circles you slowly. \"Show me your hands,\" she says. "
        "You extend them, surprised to find intricate strings wrapped around each finger, extending upward. "
        "\"Do you see who's holding them?\" she asks. You look up, but the ceiling is mirrored, "
        "showing only your own face looking back down at you, smiling with an expression that isn't yours.",
        
        "You're searching your home frantically, calling {npc1}'s name. The rooms shift and expand, "
        "doorways leading to impossible spaces. Your phone rings. It's {npc1}. \"Where are you?\" you ask desperately. "
        "\"I'm right here,\" she says, her voice coming both from the phone and from behind you. "
        "\"I've always been right here. You're the one who's lost.\"",
        
        "You're trying to walk away from {npc1}, but your feet sink deeper into the floor with each step. "
        "\"I don't understand why you're struggling,\" she says, not moving yet somehow keeping pace beside you. "
        "\"You stopped walking on your own long ago.\" You look down to find your legs have merged with the floor entirely, "
        "indistinguishable from the material beneath.",
        
        "You're giving a presentation to a room full of people, but every time you speak, your voice comes out as {npc1}'s voice, "
        "saying words you didn't intend. The audience nods approvingly. \"Much better,\" whispers {npc2} from beside you. "
        "\"Your ideas were never as good as hers anyway.\"",
        
        "You're walking through an unfamiliar house, opening doors that should lead outside but only reveal more rooms. "
        "In each room, {npc1} is engaged in a different activity, wearing a different expression. In the final room, "
        "all versions of her turn to look at you simultaneously. \"Which one is real?\" they ask in unison. \"The one you needed, or the one who needed you?\"",
        
        "You're swimming in deep water. Below you, {npc1} and {npc2} walk along the bottom, "
        "looking up at you and conversing, their voices perfectly clear despite the water. "
        "\"They still think they're above it all,\" says {npc1}, and they both laugh. You realize you can't remember how to reach the surface."
    ]
    
    try:
        # Select a random dream template
        dream_text = random.choice(dream_templates).format(npc1=npc1, npc2=npc2, npc3=npc3)
        
        # Add to PlayerJournal
        context = ctx.context
        user_id = context.user_id
        conversation_id = context.conversation_id
        
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, timestamp)
                VALUES (%s, %s, 'dream_sequence', %s, CURRENT_TIMESTAMP)
                RETURNING id
            """, (user_id, conversation_id, dream_text))
            
            journal_id = cursor.fetchone()[0]
            conn.commit()
            
            return {
                "type": "dream_sequence",
                "text": dream_text,
                "journal_id": journal_id,
                "success": True
            }
        except Exception as db_error:
            conn.rollback()
            logger.error(f"Database error recording dream sequence: {db_error}")
            raise
        finally:
            cursor.close()
            conn.close()
    except Exception as e:
        logger.error(f"Error generating dream sequence: {str(e)}", exc_info=True)
        return {
            "type": "dream_sequence",
            "text": f"Error generating dream: {str(e)}",
            "success": False
        }

@function_tool
async def check_relationship_events(ctx) -> Dict[str, Any]:
    """
    Check for relationship events like crossroads or rituals.
    
    Returns:
        Dictionary with any triggered relationship events
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        # Check for crossroads
        crossroads = await check_for_relationship_crossroads(user_id, conversation_id)
        
        # Check for rituals
        ritual = await check_for_relationship_ritual(user_id, conversation_id)
        
        return {
            "crossroads": crossroads,
            "ritual": ritual,
            "has_events": crossroads is not None or ritual is not None
        }
    except Exception as e:
        logger.error(f"Error checking relationship events: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "crossroads": None,
            "ritual": None,
            "has_events": False
        }

@function_tool
async def apply_crossroads_choice(
    ctx,
    link_id: int,
    crossroads_name: str,
    choice_index: int
) -> Dict[str, Any]:
    """
    Apply a chosen effect from a triggered relationship crossroads.
    
    Args:
        link_id: ID of the social link
        crossroads_name: Name of the crossroads event
        choice_index: Index of the chosen option
        
    Returns:
        Result of applying the choice
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        result = await apply_crossroads_choice(
            user_id, conversation_id, link_id, crossroads_name, choice_index
        )
        
        return result
    except Exception as e:
        logger.error(f"Error applying crossroads choice: {str(e)}", exc_info=True)
        return {
            "link_id": link_id,
            "crossroads_name": crossroads_name,
            "choice_index": choice_index,
            "success": False,
            "error": str(e)
        }

@function_tool
async def check_npc_relationship(
    ctx, 
    npc_id: int
) -> Dict[str, Any]:
    """
    Get the relationship between the player and an NPC.
    
    Args:
        npc_id: ID of the NPC
        
    Returns:
        Relationship summary
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        relationship = await get_relationship_summary(
            user_id, conversation_id, 
            "player", user_id, "npc", npc_id
        )
        
        if not relationship:
            # If no relationship exists, create a basic one
            try:
                from logic.social_links_agentic import create_social_link
                link_id = await create_social_link(
                    user_id, conversation_id,
                    "player", user_id, "npc", npc_id
                )
                
                # Fetch again
                relationship = await get_relationship_summary(
                    user_id, conversation_id, 
                    "player", user_id, "npc", npc_id
                )
            except Exception as link_error:
                logger.error(f"Error creating social link: {link_error}")
                return {
                    "error": f"Failed to create relationship: {str(link_error)}",
                    "npc_id": npc_id
                }
        
        return relationship or {
            "error": "Could not get or create relationship",
            "npc_id": npc_id
        }
    except Exception as e:
        logger.error(f"Error checking NPC relationship: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "npc_id": npc_id
        }

@function_tool
async def add_moment_of_clarity(ctx, realization_text: str) -> Dict[str, Any]:
    """
    Add a moment of clarity where the player briefly becomes aware of their situation.
    
    Args:
        realization_text: The specific realization the player has
        
    Returns:
        The created moment of clarity
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    try:
        from logic.narrative_progression import add_moment_of_clarity as add_clarity
        
        result = await add_clarity(user_id, conversation_id, realization_text)
        
        return {
            "type": "moment_of_clarity",
            "content": result,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error adding moment of clarity: {str(e)}", exc_info=True)
        return {
            "type": "moment_of_clarity",
            "content": None,
            "success": False,
            "error": str(e)
        }

@function_tool
async def get_player_journal_entries(ctx, entry_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get entries from the player's journal.
    
    Args:
        entry_type: Optional filter for entry type 
                   (personal_revelation, dream_sequence, moment_of_clarity, etc.)
        limit: Maximum number of entries to return
        
    Returns:
        List of journal entries
    """
    context = ctx.context
    user_id = context.user_id
    conversation_id = context.conversation_id
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if entry_type:
            cursor.execute("""
                SELECT id, entry_type, entry_text, revelation_types, 
                       narrative_moment, fantasy_flag, intensity_level, timestamp
                FROM PlayerJournal
                WHERE user_id=%s AND conversation_id=%s AND entry_type=%s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (user_id, conversation_id, entry_type, limit))
        else:
            cursor.execute("""
                SELECT id, entry_type, entry_text, revelation_types, 
                       narrative_moment, fantasy_flag, intensity_level, timestamp
                FROM PlayerJournal
                WHERE user_id=%s AND conversation_id=%s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (user_id, conversation_id, limit))
        
        entries = []
        for row in cursor.fetchall():
            id, entry_type, entry_text, revelation_types, narrative_moment, fantasy_flag, intensity_level, timestamp = row
            
            entries.append({
                "id": id,
                "entry_type": entry_type,
                "entry_text": entry_text,
                "revelation_types": revelation_types,
                "narrative_moment": narrative_moment,
                "fantasy_flag": fantasy_flag,
                "intensity_level": intensity_level,
                "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
            })
        
        return entries
    except Exception as e:
        logger.error(f"Error getting player journal entries: {str(e)}", exc_info=True)
        return []
    finally:
        cursor.close()
        conn.close()

# ----- Exports -----

# Story state and metadata tools
story_tools = [
    get_story_state,
    get_key_npcs,
    get_narrative_stages,
    analyze_narrative_and_activity
]

# Conflict management tools
conflict_tools = [
    generate_conflict,
    update_conflict_progress,
    resolve_conflict,
    analyze_narrative_for_conflict,
    set_player_involvement,
    get_conflict_details
]

# Resource management tools
resource_tools = [
    check_resources,
    commit_resources_to_conflict,
    get_player_resources,
    analyze_activity_effects,
    apply_activity_effects,
    get_resource_history
]

# Narrative element tools
narrative_tools = [
    generate_personal_revelation,
    generate_dream_sequence,
    check_relationship_events,
    apply_crossroads_choice,
    check_npc_relationship,
    add_moment_of_clarity,
    get_player_journal_entries
]
