# logic/conflict_system/hooks.py
"""
Simple integration hooks for the canonical conflict system.
These functions provide easy ways for other parts of the game to interact with conflicts.
"""

import logging
from typing import Dict, List, Any, Optional
from agents import RunContextWrapper
from db.connection import get_db_connection_context

from logic.conflict_system.canonical_conflict_integration import CanonicalConflictSystem
from logic.conflict_system.enhanced_conflict_generation import (
    generate_organic_conflict, analyze_conflict_pressure
)
from logic.conflict_system.dynamic_stakeholder_agents import (
    process_conflict_stakeholder_turns, force_stakeholder_action
)
from logic.conflict_system.conflict_tools import (
    get_active_conflicts, get_conflict_details, track_story_beat
)

logger = logging.getLogger(__name__)

# Global conflict system instances
_conflict_systems: Dict[str, CanonicalConflictSystem] = {}

async def ensure_conflict_system(user_id: int, conversation_id: int) -> CanonicalConflictSystem:
    """Ensure conflict system is initialized for user/conversation"""
    key = f"{user_id}:{conversation_id}"
    
    if key not in _conflict_systems:
        system = CanonicalConflictSystem(user_id, conversation_id)
        await system.initialize()
        _conflict_systems[key] = system
        
    return _conflict_systems[key]

# ===== SIMPLE HOOKS FOR COMMON OPERATIONS =====

async def check_and_generate_conflict(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    Check if a new conflict should be generated and create it if appropriate.
    
    Returns:
        Generated conflict data or None if no conflict was generated
    """
    try:
        system = await ensure_conflict_system(user_id, conversation_id)
        result = await system.monitor_and_generate_conflicts()
        
        return result.get('generated_conflict')
        
    except Exception as e:
        logger.error(f"Error checking/generating conflict: {e}")
        return None

async def get_player_conflicts(user_id: int, conversation_id: int) -> List[Dict[str, Any]]:
    """
    Get all active conflicts the player is involved in.
    
    Returns:
        List of conflicts with player involvement details
    """
    try:
        ctx = RunContextWrapper({"user_id": user_id, "conversation_id": conversation_id})
        conflicts = await get_active_conflicts(ctx)
        
        # Filter to only conflicts with player involvement
        player_conflicts = []
        for conflict in conflicts:
            if conflict.get('player_involvement', {}).get('involvement_level') != 'none':
                player_conflicts.append(conflict)
                
        return player_conflicts
        
    except Exception as e:
        logger.error(f"Error getting player conflicts: {e}")
        return []

async def advance_conflict_story(user_id: int, conversation_id: int,
                               conflict_id: int,
                               event_description: str,
                               involved_npcs: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Advance a conflict's story based on player action or event.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        conflict_id: The conflict to advance
        event_description: What happened
        involved_npcs: NPCs involved in this event
        
    Returns:
        Result of the story advancement
    """
    try:
        system = await ensure_conflict_system(user_id, conversation_id)
        
        # First, evolve the conflict based on the event
        evolution_result = await system.evolve_conflict(
            conflict_id,
            "player_action",
            {"description": event_description, "npcs": involved_npcs or []}
        )
        
        # Then process stakeholder reactions
        ctx = RunContextWrapper({"user_id": user_id, "conversation_id": conversation_id})
        stakeholder_actions = await process_conflict_stakeholder_turns(ctx, conflict_id)
        
        return {
            "evolution": evolution_result,
            "stakeholder_actions": stakeholder_actions,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error advancing conflict story: {e}")
        return {"success": False, "error": str(e)}

async def trigger_conflict_from_event(user_id: int, conversation_id: int,
                                    event_type: str,
                                    event_data: Dict[str, Any],
                                    preferred_scale: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Trigger conflict generation from a specific game event.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        event_type: Type of triggering event
        event_data: Details about the event
        preferred_scale: Preferred conflict scale
        
    Returns:
        Generated conflict or None
    """
    try:
        # Add event to world state consideration
        async with get_db_connection_context() as conn:
            # Log event as canonical for conflict generation to consider
            await conn.execute("""
                INSERT INTO CanonicalEvents
                (user_id, conversation_id, event_text, tags, significance)
                VALUES ($1, $2, $3, $4, $5)
            """,
            user_id, conversation_id,
            f"{event_type}: {event_data.get('description', 'Event occurred')}",
            [event_type, 'trigger'], 7
            )
        
        # Generate conflict based on new world state
        ctx = RunContextWrapper({"user_id": user_id, "conversation_id": conversation_id})
        conflict = await generate_organic_conflict(ctx, preferred_scale=preferred_scale)
        
        return conflict
        
    except Exception as e:
        logger.error(f"Error triggering conflict from event: {e}")
        return None

async def get_conflict_summary(user_id: int, conversation_id: int,
                             conflict_id: int) -> Dict[str, Any]:
    """
    Get a summary of a conflict suitable for display.
    
    Returns:
        Formatted conflict summary
    """
    try:
        ctx = RunContextWrapper({"user_id": user_id, "conversation_id": conversation_id})
        conflict = await get_conflict_details(ctx, conflict_id)
        
        if not conflict or 'error' in conflict:
            return {"error": "Conflict not found"}
        
        # Format for display
        summary = {
            "name": conflict['conflict_name'],
            "description": conflict['description'],
            "type": conflict['conflict_type'],
            "phase": conflict['phase'],
            "progress": conflict['progress'],
            "player_role": conflict['player_involvement']['involvement_level'],
            "player_faction": conflict['player_involvement'].get('faction', 'neutral'),
            "key_players": []
        }
        
        # Add top stakeholders
        for stakeholder in conflict.get('stakeholders', [])[:5]:
            summary['key_players'].append({
                "name": stakeholder['npc_name'],
                "role": stakeholder.get('faction_position', 'Independent'),
                "stance": stakeholder['public_motivation'][:50] + "..."
            })
        
        # Add available actions
        summary['available_actions'] = []
        for path in conflict.get('resolution_paths', []):
            if not path['is_completed']:
                summary['available_actions'].append({
                    "path": path['name'],
                    "approach": path['approach_type'],
                    "progress": path['progress']
                })
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting conflict summary: {e}")
        return {"error": str(e)}

async def resolve_conflict_path(user_id: int, conversation_id: int,
                              conflict_id: int,
                              path_id: str,
                              final_action: str) -> Dict[str, Any]:
    """
    Complete a resolution path for a conflict.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        conflict_id: The conflict ID
        path_id: The resolution path taken
        final_action: Description of final action
        
    Returns:
        Resolution results
    """
    try:
        ctx = RunContextWrapper({"user_id": user_id, "conversation_id": conversation_id})
        
        # Track final story beat with high progress
        result = await track_story_beat(
            ctx, conflict_id, path_id,
            final_action, [], 50.0  # High progress to complete path
        )
        
        # Get conflict outcome
        conflict = await get_conflict_details(ctx, conflict_id)
        
        return {
            "success": True,
            "path_completed": result.get('is_completed', False),
            "conflict_resolved": not conflict.get('is_active', True),
            "final_phase": conflict.get('phase'),
            "outcome": conflict.get('outcome', 'ongoing')
        }
        
    except Exception as e:
        logger.error(f"Error resolving conflict path: {e}")
        return {"success": False, "error": str(e)}

# ===== EVENT HANDLERS FOR GAME SYSTEMS =====

async def on_npc_relationship_change(user_id: int, conversation_id: int,
                                   npc1_id: int, npc2_id: int,
                                   old_level: int, new_level: int):
    """
    Handle NPC relationship changes that might trigger conflicts.
    
    Called by relationship system when relationships change significantly.
    """
    # Major negative shift might trigger conflict
    if old_level > 0 and new_level < -50:
        await trigger_conflict_from_event(
            user_id, conversation_id,
            "relationship_breakdown",
            {
                "npc1": npc1_id,
                "npc2": npc2_id,
                "description": "Relationship deteriorated into hostility"
            },
            preferred_scale="personal"
        )

async def on_faction_power_shift(user_id: int, conversation_id: int,
                               faction_name: str,
                               power_change: int):
    """
    Handle faction power changes that might trigger conflicts.
    
    Called by faction system when power dynamics shift.
    """
    # Major power shifts might trigger conflicts
    if abs(power_change) > 5:
        await trigger_conflict_from_event(
            user_id, conversation_id,
            "faction_power_shift",
            {
                "faction": faction_name,
                "change": power_change,
                "description": f"{faction_name} {'gained' if power_change > 0 else 'lost'} significant power"
            },
            preferred_scale="local"
        )

async def on_resource_crisis(user_id: int, conversation_id: int,
                           resource_type: str,
                           severity: str):
    """
    Handle resource crises that might trigger conflicts.
    
    Called by resource system when scarcity detected.
    """
    if severity in ["critical", "severe"]:
        await trigger_conflict_from_event(
            user_id, conversation_id,
            "resource_crisis",
            {
                "resource": resource_type,
                "severity": severity,
                "description": f"{severity.capitalize()} {resource_type} shortage"
            },
            preferred_scale="regional"
        )

async def on_player_major_action(user_id: int, conversation_id: int,
                               action_type: str,
                               action_data: Dict[str, Any]):
    """
    Handle major player actions that affect conflicts.
    
    Called by main game loop for significant player choices.
    """
    # Get active conflicts
    conflicts = await get_player_conflicts(user_id, conversation_id)
    
    # Update relevant conflicts
    for conflict in conflicts:
        # Determine if action affects this conflict
        if _action_affects_conflict(action_type, action_data, conflict):
            await advance_conflict_story(
                user_id, conversation_id,
                conflict['conflict_id'],
                f"Player {action_type}: {action_data.get('description', 'took action')}",
                action_data.get('involved_npcs', [])
            )

def _action_affects_conflict(action_type: str, action_data: Dict[str, Any],
                           conflict: Dict[str, Any]) -> bool:
    """Determine if a player action affects a conflict"""
    # Check if action involves conflict stakeholders
    action_npcs = set(action_data.get('involved_npcs', []))
    conflict_npcs = {s['npc_id'] for s in conflict.get('stakeholders', [])}
    
    if action_npcs & conflict_npcs:
        return True
        
    # Check if action type is relevant
    relevant_actions = {
        'faction_support': conflict['conflict_type'] == 'faction_rivalry',
        'resource_allocation': 'economic' in conflict['conflict_type'],
        'investigation': conflict.get('resolution_paths', [{}])[0].get('approach_type') == 'investigative'
    }
    
    return relevant_actions.get(action_type, False)

# ===== CONVENIENCE FUNCTIONS =====

async def get_world_tension_level(user_id: int, conversation_id: int) -> str:
    """
    Get a simple description of current world tension.
    
    Returns:
        Tension level: "calm", "tense", "volatile", "critical"
    """
    try:
        ctx = RunContextWrapper({"user_id": user_id, "conversation_id": conversation_id})
        analysis = await analyze_conflict_pressure(ctx)
        
        pressure = analysis['total_pressure']
        
        if pressure < 50:
            return "calm"
        elif pressure < 150:
            return "tense"
        elif pressure < 300:
            return "volatile"
        else:
            return "critical"
            
    except Exception as e:
        logger.error(f"Error getting world tension: {e}")
        return "unknown"

async def suggest_conflict_action(user_id: int, conversation_id: int,
                                conflict_id: int) -> Dict[str, Any]:
    """
    Get AI suggestion for best player action in a conflict.
    
    Returns:
        Suggested action and reasoning
    """
    try:
        system = await ensure_conflict_system(user_id, conversation_id)
        ctx = RunContextWrapper({"user_id": user_id, "conversation_id": conversation_id})
        
        conflict = await get_conflict_details(ctx, conflict_id)
        
        # Use system's agents to analyze
        # This is a simplified version - you could make this more sophisticated
        suggestion = {
            "action": "Investigate key stakeholders",
            "reasoning": "Understanding motivations will reveal optimal path",
            "alternatives": [
                "Support the underdog faction",
                "Remain neutral and profit",
                "Expose everyone's secrets"
            ]
        }
        
        return suggestion
        
    except Exception as e:
        logger.error(f"Error suggesting conflict action: {e}")
        return {"error": str(e)}

