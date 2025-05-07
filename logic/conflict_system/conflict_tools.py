# logic/conflict_system/conflict_tools.py
"""
Conflict System Function Tools

This module defines the function tools used by the conflict system agents.
"""

import logging
import json
import random
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field
from agents import (
    Agent, Runner, trace, function_tool, 
    RunContextWrapper, handoff, ModelSettings,
    InputGuardrail, GuardrailFunctionOutput, 
    Handoff, RunConfig, FunctionTool
)

from db.connection import get_db_connection_context
from logic.stats_logic import apply_stat_change
from logic.resource_management import ResourceManager
from logic.conflict_system.conflict_agents import get_relationship_status, get_manipulation_leverage
from logic.chatgpt_integration import get_chatgpt_response
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from context.context_performance import track_performance

logger = logging.getLogger(__name__)

# Database Access Tools

@function_tool
async def get_resolution_paths(ctx: RunContextWrapper, conflict_id: int) -> List[Dict[str, Any]]:
    """
    Get all resolution paths for a specific conflict.
    
    Args:
        ctx: RunContextWrapper with user context
        conflict_id: ID of the conflict
        
    Returns:
        List of resolution path dictionaries
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Get resolution paths
            paths_rows = await conn.fetch("""
                SELECT path_id, name, description, approach_type, difficulty,
                       requirements, stakeholders_involved, key_challenges,
                       progress, is_completed
                FROM ResolutionPaths
                WHERE conflict_id = $1
            """, conflict_id)
            
            paths = []
            for row in paths_rows:
                path = dict(row)
                
                # Parse JSON fields
                try:
                    path["requirements"] = json.loads(path["requirements"]) if isinstance(path["requirements"], str) else path["requirements"] or {}
                except (json.JSONDecodeError, TypeError):
                    path["requirements"] = {}
                
                try:
                    path["stakeholders_involved"] = json.loads(path["stakeholders_involved"]) if isinstance(path["stakeholders_involved"], str) else path["stakeholders_involved"] or []
                except (json.JSONDecodeError, TypeError):
                    path["stakeholders_involved"] = []
                
                try:
                    path["key_challenges"] = json.loads(path["key_challenges"]) if isinstance(path["key_challenges"], str) else path["key_challenges"] or []
                except (json.JSONDecodeError, TypeError):
                    path["key_challenges"] = []
                
                paths.append(path)
            
            return paths
    except Exception as e:
        logger.error(f"Error getting resolution paths for conflict {conflict_id}: {e}", exc_info=True)
        return []

@function_tool
@track_performance("update_conflict_progress")
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
        
        # NEW: Store this update as a memory if possible
        if hasattr(context, 'add_narrative_memory'):
            memory_importance = 0.5  # Default importance
            
            # Increase importance if phase changed
            if updated_conflict['phase'] != old_phase:
                memory_importance = 0.7
                
            memory_content = (
                f"Updated conflict {updated_conflict['conflict_name']} progress by {progress_increment} points "
                f"to {updated_conflict['progress']}%. "
            )
            
            if updated_conflict['phase'] != old_phase:
                memory_content += f"Phase advanced from {old_phase} to {updated_conflict['phase']}."
                
            await context.add_narrative_memory(
                memory_content,
                "conflict_progression",
                memory_importance
            )
        
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
async def get_active_conflicts(ctx: RunContextWrapper) -> List[Dict[str, Any]]:
    """
    Get all active conflicts for the current user and conversation.
    
    Returns a list of active conflict dictionaries with all related data.
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Get conflicts
            conflicts_rows = await conn.fetch("""
                SELECT c.conflict_id, c.conflict_name, c.conflict_type, 
                       c.description, c.progress, c.phase, c.start_day,
                       c.estimated_duration, c.success_rate, c.outcome, c.is_active
                FROM Conflicts c
                WHERE c.user_id = $1 AND c.conversation_id = $2 AND c.is_active = TRUE
                ORDER BY c.conflict_id DESC
            """, context.user_id, context.conversation_id)
            
            conflicts = []
            for row in conflicts_rows:
                # Build conflict dictionary
                conflict = dict(row)
                conflict_id = conflict["conflict_id"]
                
                # Get stakeholders
                stakeholders_rows = await conn.fetch("""
                    SELECT s.npc_id, n.npc_name, s.faction_id, s.faction_name,
                           s.faction_position, s.public_motivation, s.private_motivation,
                           s.desired_outcome, s.involvement_level, s.alliances, s.rivalries
                    FROM ConflictStakeholders s
                    JOIN NPCStats n ON s.npc_id = n.npc_id
                    WHERE s.conflict_id = $1
                    ORDER BY s.involvement_level DESC
                """, conflict_id)
                
                stakeholders = []
                for s_row in stakeholders_rows:
                    stakeholders.append(dict(s_row))
                
                conflict["stakeholders"] = stakeholders
                
                # Get resolution paths
                paths_rows = await conn.fetch("""
                    SELECT path_id, name, description, approach_type, difficulty,
                           requirements, stakeholders_involved, key_challenges,
                           progress, is_completed
                    FROM ResolutionPaths
                    WHERE conflict_id = $1
                """, conflict_id)
                
                paths = []
                for p_row in paths_rows:
                    paths.append(dict(p_row))
                
                conflict["resolution_paths"] = paths
                
                # Get player involvement
                involvement_row = await conn.fetchrow("""
                    SELECT involvement_level, faction, money_committed, supplies_committed, 
                           influence_committed
                    FROM PlayerConflictInvolvement
                    WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
                """, conflict_id, context.user_id, context.conversation_id)
                
                if involvement_row:
                    involvement = dict(involvement_row)
                    conflict["player_involvement"] = {
                        "involvement_level": involvement["involvement_level"],
                        "faction": involvement["faction"],
                        "resources_committed": {
                            "money": involvement["money_committed"],
                            "supplies": involvement["supplies_committed"],
                            "influence": involvement["influence_committed"]
                        }
                    }
                else:
                    conflict["player_involvement"] = {
                        "involvement_level": "none",
                        "faction": "neutral",
                        "resources_committed": {
                            "money": 0,
                            "supplies": 0,
                            "influence": 0
                        }
                    }
                
                # Get internal faction conflicts
                internal_conflicts_rows = await conn.fetch("""
                    SELECT struggle_id, faction_id, conflict_name, description,
                           primary_npc_id, target_npc_id, prize, approach, 
                           public_knowledge, current_phase, progress
                    FROM InternalFactionConflicts
                    WHERE parent_conflict_id = $1
                    ORDER BY progress DESC
                """, conflict_id)
                
                internal_conflicts = []
                for ic_row in internal_conflicts_rows:
                    internal_conflicts.append(dict(ic_row))
                
                if internal_conflicts:
                    conflict["internal_faction_conflicts"] = internal_conflicts
                
                conflicts.append(conflict)
            
            return conflicts
    except Exception as e:
        logger.error(f"Error getting active conflicts: {e}", exc_info=True)
        return []

@function_tool
async def update_stakeholder_status(
    ctx: RunContextWrapper,
    conflict_id: int,
    npc_id: int,
    status: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update the status of a stakeholder in a conflict.
    
    Args:
        ctx: RunContextWrapper with user context
        conflict_id: ID of the conflict
        npc_id: ID of the NPC stakeholder
        status: Dictionary with updated status fields
        
    Returns:
        Dictionary with update result
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Check if stakeholder exists
            exists = await conn.fetchval("""
                SELECT 1 FROM ConflictStakeholders
                WHERE conflict_id = $1 AND npc_id = $2
            """, conflict_id, npc_id)
            
            if not exists:
                return {
                    "success": False,
                    "error": f"Stakeholder with NPC ID {npc_id} not found in conflict {conflict_id}"
                }
            
            # Prepare update fields
            update_fields = []
            params = [conflict_id, npc_id]
            param_index = 3
            
            # Handle each possible field to update
            if "involvement_level" in status:
                update_fields.append(f"involvement_level = ${param_index}")
                params.append(status["involvement_level"])
                param_index += 1
                
            if "public_motivation" in status:
                update_fields.append(f"public_motivation = ${param_index}")
                params.append(status["public_motivation"])
                param_index += 1
                
            if "private_motivation" in status:
                update_fields.append(f"private_motivation = ${param_index}")
                params.append(status["private_motivation"])
                param_index += 1
                
            if "desired_outcome" in status:
                update_fields.append(f"desired_outcome = ${param_index}")
                params.append(status["desired_outcome"])
                param_index += 1
                
            if "alliances" in status:
                update_fields.append(f"alliances = ${param_index}")
                params.append(json.dumps(status["alliances"]))
                param_index += 1
                
            if "rivalries" in status:
                update_fields.append(f"rivalries = ${param_index}")
                params.append(json.dumps(status["rivalries"]))
                param_index += 1
                
            if "leadership_ambition" in status:
                update_fields.append(f"leadership_ambition = ${param_index}")
                params.append(status["leadership_ambition"])
                param_index += 1
                
            if "faction_standing" in status:
                update_fields.append(f"faction_standing = ${param_index}")
                params.append(status["faction_standing"])
                param_index += 1
                
            if "willing_to_betray_faction" in status:
                update_fields.append(f"willing_to_betray_faction = ${param_index}")
                params.append(status["willing_to_betray_faction"])
                param_index += 1
                
            if "faction_id" in status:
                update_fields.append(f"faction_id = ${param_index}")
                params.append(status["faction_id"])
                param_index += 1
                
            if "faction_name" in status:
                update_fields.append(f"faction_name = ${param_index}")
                params.append(status["faction_name"])
                param_index += 1
                
            if "faction_position" in status:
                update_fields.append(f"faction_position = ${param_index}")
                params.append(status["faction_position"])
                param_index += 1
                
            # If no fields to update, return error
            if not update_fields:
                return {
                    "success": False,
                    "error": "No valid fields provided for update"
                }
                
            # Build and execute update query
            update_query = f"""
                UPDATE ConflictStakeholders
                SET {", ".join(update_fields)}
                WHERE conflict_id = $1 AND npc_id = $2
            """
            
            await conn.execute(update_query, *params)
            
            # Get NPC name for memory
            npc_name = await get_npc_name(ctx, npc_id)
            
            # Create a memory for this stakeholder update
            await create_conflict_memory(
                ctx,
                conflict_id,
                f"Stakeholder {npc_name}'s status has been updated in the conflict.",
                significance=5
            )
            
            return {
                "success": True,
                "npc_id": npc_id,
                "npc_name": npc_name,
                "conflict_id": conflict_id,
                "updated_fields": [field.split(' = ')[0] for field in update_fields]
            }
    except Exception as e:
        logger.error(f"Error updating stakeholder status for NPC {npc_id} in conflict {conflict_id}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

@function_tool
async def get_player_involvement(ctx: RunContextWrapper, conflict_id: int) -> Dict[str, Any]:
    """
    Get player's involvement in a specific conflict.
    
    Args:
        ctx: RunContextWrapper with user context
        conflict_id: ID of the conflict
        
    Returns:
        Dictionary with player involvement details
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Get player involvement
            involvement_row = await conn.fetchrow("""
                SELECT involvement_level, faction, money_committed, supplies_committed, 
                       influence_committed, actions_taken, manipulated_by
                FROM PlayerConflictInvolvement
                WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
            """, conflict_id, context.user_id, context.conversation_id)
            
            if involvement_row:
                involvement = dict(involvement_row)
                
                # Parse JSON fields
                try:
                    actions_list = json.loads(involvement["actions_taken"]) if isinstance(involvement["actions_taken"], str) else involvement["actions_taken"] or []
                except (json.JSONDecodeError, TypeError):
                    actions_list = []
                
                try:
                    manipulated_by_dict = json.loads(involvement["manipulated_by"]) if isinstance(involvement["manipulated_by"], str) else involvement["manipulated_by"] or None
                except (json.JSONDecodeError, TypeError):
                    manipulated_by_dict = None
                
                return {
                    "involvement_level": involvement["involvement_level"],
                    "faction": involvement["faction"],
                    "resources_committed": {
                        "money": involvement["money_committed"],
                        "supplies": involvement["supplies_committed"],
                        "influence": involvement["influence_committed"]
                    },
                    "actions_taken": actions_list,
                    "is_manipulated": manipulated_by_dict is not None,
                    "manipulated_by": manipulated_by_dict
                }
            else:
                return {
                    "involvement_level": "none",
                    "faction": "neutral",
                    "resources_committed": {
                        "money": 0,
                        "supplies": 0,
                        "influence": 0
                    },
                    "actions_taken": [],
                    "is_manipulated": False,
                    "manipulated_by": None
                }
    except Exception as e:
        logger.error(f"Error getting player involvement for conflict {conflict_id}: {e}", exc_info=True)
        return {
            "involvement_level": "none",
            "faction": "neutral",
            "resources_committed": {
                "money": 0,
                "supplies": 0,
                "influence": 0
            },
            "actions_taken": [],
            "is_manipulated": False,
            "manipulated_by": None
        }

@function_tool
async def get_conflict_details(ctx: RunContextWrapper, conflict_id: int) -> Dict[str, Any]:
    """
    Get detailed information about a specific conflict.
    
    Args:
        conflict_id: ID of the conflict to retrieve
        
    Returns:
        Conflict dictionary with all related data
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Get conflict
            conflict_row = await conn.fetchrow("""
                SELECT c.conflict_id, c.conflict_name, c.conflict_type, 
                       c.description, c.progress, c.phase, c.start_day,
                       c.estimated_duration, c.success_rate, c.outcome, c.is_active
                FROM Conflicts c
                WHERE c.conflict_id = $1 AND c.user_id = $2 AND c.conversation_id = $3
            """, conflict_id, context.user_id, context.conversation_id)
            
            if not conflict_row:
                return {"error": "Conflict not found"}
            
            # Build conflict dictionary
            conflict = dict(conflict_row)
            
            # Get stakeholders
            conflict["stakeholders"] = await get_conflict_stakeholders(ctx, conflict_id)
            
            # Get resolution paths
            paths_rows = await conn.fetch("""
                SELECT path_id, name, description, approach_type, difficulty,
                       requirements, stakeholders_involved, key_challenges,
                       progress, is_completed
                FROM ResolutionPaths
                WHERE conflict_id = $1
            """, conflict_id)
            
            paths = []
            for row in paths_rows:
                path = dict(row)
                
                # Parse JSON fields
                try:
                    path["requirements"] = json.loads(path["requirements"]) if isinstance(path["requirements"], str) else path["requirements"] or {}
                except (json.JSONDecodeError, TypeError):
                    path["requirements"] = {}
                
                try:
                    path["stakeholders_involved"] = json.loads(path["stakeholders_involved"]) if isinstance(path["stakeholders_involved"], str) else path["stakeholders_involved"] or []
                except (json.JSONDecodeError, TypeError):
                    path["stakeholders_involved"] = []
                
                try:
                    path["key_challenges"] = json.loads(path["key_challenges"]) if isinstance(path["key_challenges"], str) else path["key_challenges"] or []
                except (json.JSONDecodeError, TypeError):
                    path["key_challenges"] = []
                
                paths.append(path)
            
            conflict["resolution_paths"] = paths
            
            # Get player involvement
            involvement_row = await conn.fetchrow("""
                SELECT involvement_level, faction, money_committed, supplies_committed, 
                       influence_committed, actions_taken, manipulated_by
                FROM PlayerConflictInvolvement
                WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
            """, conflict_id, context.user_id, context.conversation_id)
            
            if involvement_row:
                involvement = dict(involvement_row)
                
                # Parse JSON fields
                try:
                    actions_list = json.loads(involvement["actions_taken"]) if isinstance(involvement["actions_taken"], str) else involvement["actions_taken"] or []
                except (json.JSONDecodeError, TypeError):
                    actions_list = []
                
                try:
                    manipulated_by_dict = json.loads(involvement["manipulated_by"]) if isinstance(involvement["manipulated_by"], str) else involvement["manipulated_by"] or None
                except (json.JSONDecodeError, TypeError):
                    manipulated_by_dict = None
                
                conflict["player_involvement"] = {
                    "involvement_level": involvement["involvement_level"],
                    "faction": involvement["faction"],
                    "resources_committed": {
                        "money": involvement["money_committed"],
                        "supplies": involvement["supplies_committed"],
                        "influence": involvement["influence_committed"]
                    },
                    "actions_taken": actions_list,
                    "is_manipulated": manipulated_by_dict is not None,
                    "manipulated_by": manipulated_by_dict
                }
            else:
                conflict["player_involvement"] = {
                    "involvement_level": "none",
                    "faction": "neutral",
                    "resources_committed": {
                        "money": 0,
                        "supplies": 0,
                        "influence": 0
                    },
                    "actions_taken": [],
                    "is_manipulated": False,
                    "manipulated_by": None
                }
            
            # Get internal faction conflicts
            internal_conflicts_rows = await conn.fetch("""
                SELECT struggle_id, faction_id, conflict_name, description,
                       primary_npc_id, target_npc_id, prize, approach, 
                       public_knowledge, current_phase, progress
                FROM InternalFactionConflicts
                WHERE parent_conflict_id = $1
                ORDER BY progress DESC
            """, conflict_id)
            
            internal_conflicts = []
            for row in internal_conflicts_rows:
                internal_conflict = dict(row)
                
                # Get faction name
                faction_name = await get_faction_name(ctx, internal_conflict["faction_id"])
                
                # Get NPC names
                primary_npc_name = await get_npc_name(ctx, internal_conflict["primary_npc_id"])
                target_npc_name = await get_npc_name(ctx, internal_conflict["target_npc_id"])
                
                internal_conflict["faction_name"] = faction_name
                internal_conflict["primary_npc_name"] = primary_npc_name
                internal_conflict["target_npc_name"] = target_npc_name
                
                internal_conflicts.append(internal_conflict)
            
            if internal_conflicts:
                conflict["internal_faction_conflicts"] = internal_conflicts
            
            # Get manipulation attempts
            manipulation_attempts = await get_player_manipulation_attempts(ctx, conflict_id)
            if manipulation_attempts:
                conflict["manipulation_attempts"] = manipulation_attempts
            
            return conflict
    except Exception as e:
        logger.error(f"Error getting conflict details for ID {conflict_id}: {e}", exc_info=True)
        return {"error": str(e)}

@function_tool
async def get_conflict_stakeholders(ctx: RunContextWrapper, conflict_id: int) -> List[Dict[str, Any]]:
    """
    Get all stakeholders for a specific conflict.
    
    Args:
        conflict_id: ID of the conflict
        
    Returns:
        List of stakeholder dictionaries
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            stakeholders_rows = await conn.fetch("""
                SELECT s.npc_id, n.npc_name, s.faction_id, s.faction_name,
                       s.faction_position, s.public_motivation, s.private_motivation,
                       s.desired_outcome, s.involvement_level, s.alliances, s.rivalries,
                       s.leadership_ambition, s.faction_standing, s.willing_to_betray_faction
                FROM ConflictStakeholders s
                JOIN NPCStats n ON s.npc_id = n.npc_id
                WHERE s.conflict_id = $1
                ORDER BY s.involvement_level DESC
            """, conflict_id)
            
            stakeholders = []
            for row in stakeholders_rows:
                stakeholder = dict(row)
                
                # Parse JSON fields
                try:
                    stakeholder["alliances"] = json.loads(stakeholder["alliances"]) if isinstance(stakeholder["alliances"], str) else stakeholder["alliances"] or {}
                except (json.JSONDecodeError, TypeError):
                    stakeholder["alliances"] = {}
                
                try:
                    stakeholder["rivalries"] = json.loads(stakeholder["rivalries"]) if isinstance(stakeholder["rivalries"], str) else stakeholder["rivalries"] or {}
                except (json.JSONDecodeError, TypeError):
                    stakeholder["rivalries"] = {}
                
                # Get stakeholder secrets
                stakeholder["secrets"] = await get_stakeholder_secrets(ctx, conflict_id, stakeholder["npc_id"])
                
                # Check if stakeholder manipulates player
                stakeholder["manipulates_player"] = await check_stakeholder_manipulates_player(ctx, conflict_id, stakeholder["npc_id"])
                
                # Get relationship with player
                stakeholder["relationship_with_player"] = await get_npc_relationship_with_player(ctx, stakeholder["npc_id"])
                
                stakeholders.append(stakeholder)
            
            return stakeholders
    except Exception as e:
        logger.error(f"Error getting stakeholders for conflict {conflict_id}: {e}", exc_info=True)
        return []

@function_tool
async def get_player_manipulation_attempts(ctx: RunContextWrapper, conflict_id: int) -> List[Dict[str, Any]]:
    """
    Get all manipulation attempts targeted at the player for a specific conflict.
    
    Args:
        conflict_id: ID of the conflict
        
    Returns:
        List of manipulation attempt dictionaries
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            attempts_rows = await conn.fetch("""
                SELECT attempt_id, npc_id, manipulation_type, content, goal,
                       success, player_response, leverage_used, intimacy_level,
                       created_at, resolved_at
                FROM PlayerManipulationAttempts
                WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
                ORDER BY created_at DESC
            """, conflict_id, context.user_id, context.conversation_id)
            
            attempts = []
            for row in attempts_rows:
                attempt = dict(row)
                
                # Get NPC name
                npc_name = await get_npc_name(ctx, attempt["npc_id"])
                attempt["npc_name"] = npc_name
                
                # Parse JSON fields
                try:
                    attempt["goal"] = json.loads(attempt["goal"]) if isinstance(attempt["goal"], str) else attempt["goal"] or {}
                except (json.JSONDecodeError, TypeError):
                    attempt["goal"] = {}
                
                try:
                    attempt["leverage_used"] = json.loads(attempt["leverage_used"]) if isinstance(attempt["leverage_used"], str) else attempt["leverage_used"] or {}
                except (json.JSONDecodeError, TypeError):
                    attempt["leverage_used"] = {}
                
                # Format dates
                attempt["created_at"] = attempt["created_at"].isoformat() if attempt["created_at"] else None
                attempt["resolved_at"] = attempt["resolved_at"].isoformat() if attempt["resolved_at"] else None
                attempt["is_resolved"] = attempt["resolved_at"] is not None
                
                attempts.append(attempt)
            
            return attempts
    except Exception as e:
        logger.error(f"Error getting player manipulation attempts for conflict {conflict_id}: {e}", exc_info=True)
        return []

# Conflict Generation Tools

@function_tool
async def generate_conflict(
    ctx: RunContextWrapper,
    conflict_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a new conflict with stakeholders and resolution paths.
    
    Args:
        conflict_type: Type of conflict (minor, standard, major, catastrophic)
        
    Returns:
        The generated conflict details
    """
    context = ctx.context
    
    # Get current day
    current_day = await get_current_day(ctx)
    
    # Get active conflicts to ensure we don't generate too many
    active_conflicts = await get_active_conflicts(ctx)
    
    # If there are already too many active conflicts (3+), make this a minor one
    if len(active_conflicts) >= 3 and not conflict_type:
        conflict_type = "minor"
    
    # If there are no conflicts at all and no type specified, make this a standard one
    if len(active_conflicts) == 0 and not conflict_type:
        conflict_type = "standard"
    
    # If still no type specified, choose randomly with weighted probabilities
    if not conflict_type:
        weights = {
            "minor": 0.4,
            "standard": 0.4,
            "major": 0.15,
            "catastrophic": 0.05
        }
        
        conflict_type = random.choices(
            list(weights.keys()),
            weights=list(weights.values()),
            k=1
        )[0]
    
    # Get available NPCs to use as potential stakeholders
    npcs = await get_available_npcs(ctx)
    
    if len(npcs) < 3:
        return {"error": "Not enough NPCs available to create a complex conflict"}
    
    # Determine how many stakeholders to involve
    stakeholder_count = {
        "minor": min(3, len(npcs)),
        "standard": min(4, len(npcs)),
        "major": min(5, len(npcs)),
        "catastrophic": min(6, len(npcs))
    }.get(conflict_type, min(4, len(npcs)))
    
    # Select NPCs to involve as stakeholders
    stakeholder_npcs = random.sample(npcs, stakeholder_count)
    
    # Generate conflict details using AI
    conflict_data = await generate_conflict_details(ctx, conflict_type, stakeholder_npcs, current_day)
    
    # Create the conflict in the database
    conflict_id = await create_conflict_record(ctx, conflict_data, current_day)
    
    # Create stakeholders
    await create_stakeholders(ctx, conflict_id, conflict_data, stakeholder_npcs)
    
    # Create resolution paths
    await create_resolution_paths(ctx, conflict_id, conflict_data)
    
    # Create internal faction conflicts if applicable
    if "internal_faction_conflicts" in conflict_data:
        await create_internal_faction_conflicts(ctx, conflict_id, conflict_data)
    
    # Generate player manipulation attempts
    await generate_player_manipulation_attempts(ctx, conflict_id, stakeholder_npcs)
    
    # Create initial memory event for the conflict
    await create_conflict_memory(
        ctx,
        conflict_id,
        f"A new conflict has emerged: {conflict_data['conflict_name']}. It involves multiple stakeholders with their own agendas.",
        significance=6
    )
    
    # Return the created conflict
    return await get_conflict_details(ctx, conflict_id)

@function_tool
async def get_internal_conflicts(ctx: RunContextWrapper, conflict_id: int) -> List[Dict[str, Any]]:
    """
    Get internal faction conflicts for a specific conflict.
    
    Args:
        ctx: RunContextWrapper with user context
        conflict_id: ID of the parent conflict
        
    Returns:
        List of internal faction conflict dictionaries
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Get internal faction conflicts
            internal_conflicts_rows = await conn.fetch("""
                SELECT struggle_id, faction_id, conflict_name, description,
                       primary_npc_id, target_npc_id, prize, approach, 
                       public_knowledge, current_phase, progress
                FROM InternalFactionConflicts
                WHERE parent_conflict_id = $1
                ORDER BY progress DESC
            """, conflict_id)
            
            internal_conflicts = []
            for row in internal_conflicts_rows:
                internal_conflict = dict(row)
                
                # Get faction name
                faction_name = await get_faction_name(ctx, internal_conflict["faction_id"])
                
                # Get NPC names
                primary_npc_name = await get_npc_name(ctx, internal_conflict["primary_npc_id"])
                target_npc_name = await get_npc_name(ctx, internal_conflict["target_npc_id"])
                
                internal_conflict["faction_name"] = faction_name
                internal_conflict["primary_npc_name"] = primary_npc_name
                internal_conflict["target_npc_name"] = target_npc_name
                
                # Get faction members involved in the struggle if available
                try:
                    members_rows = await conn.fetch("""
                        SELECT npc_id, position, side, standing, loyalty_strength, reason
                        FROM FactionStruggleMembers
                        WHERE struggle_id = $1
                    """, internal_conflict["struggle_id"])
                    
                    if members_rows:
                        members = []
                        for member_row in members_rows:
                            member = dict(member_row)
                            member["npc_name"] = await get_npc_name(ctx, member["npc_id"])
                            members.append(member)
                        
                        internal_conflict["faction_members"] = members
                except Exception:
                    # If table doesn't exist or other error, continue without members
                    pass
                
                internal_conflicts.append(internal_conflict)
            
            return internal_conflicts
    except Exception as e:
        logger.error(f"Error getting internal conflicts for conflict {conflict_id}: {e}", exc_info=True)
        return []

@function_tool
async def get_current_day(ctx: RunContextWrapper) -> int:
    """Get the current in-game day."""
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            value = await conn.fetchval("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentDay'
            """, context.user_id, context.conversation_id)
            
            return int(value) if value else 1
    except Exception as e:
        logger.error(f"Error getting current day: {e}", exc_info=True)
        return 1

@function_tool
async def get_available_npcs(ctx: RunContextWrapper) -> List[Dict[str, Any]]:
    """Get available NPCs that could be involved in conflicts."""
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            npc_rows = await conn.fetch("""
                SELECT npc_id, npc_name, dominance, cruelty, closeness, trust,
                       respect, intensity, sex, current_location, faction_affiliations
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND introduced=TRUE
                ORDER BY dominance DESC
            """, context.user_id, context.conversation_id)
            
            npcs = []
            for row in npc_rows:
                npc = dict(row)
                
                # Parse faction affiliations
                try:
                    npc["faction_affiliations"] = json.loads(npc["faction_affiliations"]) if isinstance(npc["faction_affiliations"], str) else npc["faction_affiliations"] or []
                except (json.JSONDecodeError, TypeError):
                    npc["faction_affiliations"] = []
                
                # Get relationships with player
                npc["relationship_with_player"] = await get_npc_relationship_with_player(ctx, npc["npc_id"])
                
                npcs.append(npc)
            
            return npcs
    except Exception as e:
        logger.error(f"Error getting available NPCs: {e}", exc_info=True)
        return []

@function_tool
async def get_npc_relationship_with_player(ctx: RunContextWrapper, npc_id: int) -> Dict[str, Any]:
    """Get an NPC's relationship with the player."""
    context = ctx.context
    try:
        # Use relationship manager to get status
        relationship = await get_relationship_status(
            context.user_id, context.conversation_id, npc_id
        )
        
        # Get potential leverage
        leverage = await get_manipulation_leverage(
            context.user_id, context.conversation_id, npc_id
        )
        
        return {
            "closeness": relationship.get("closeness", 0),
            "trust": relationship.get("trust", 0),
            "respect": relationship.get("respect", 0),
            "intimidation": relationship.get("intimidation", 0),
            "dominance": relationship.get("dominance", 0),
            "has_leverage": len(leverage) > 0,
            "leverage_types": [l.get("type") for l in leverage],
            "manipulation_potential": relationship.get("dominance", 0) > 70 or relationship.get("closeness", 0) > 80
        }
    except Exception as e:
        logger.error(f"Error getting NPC relationship with player: {e}", exc_info=True)
        return {
            "closeness": 0,
            "trust": 0,
            "respect": 0,
            "intimidation": 0,
            "dominance": 0,
            "has_leverage": False,
            "leverage_types": [],
            "manipulation_potential": False
        }

@function_tool
async def generate_conflict_details(
    ctx: RunContextWrapper,
    conflict_type: str,
    stakeholder_npcs: List[Dict[str, Any]],
    current_day: int
) -> Dict[str, Any]:
    """
    Generate conflict details using the AI.
    
    Args:
        conflict_type: Type of conflict
        stakeholder_npcs: List of NPC dictionaries to use as stakeholders
        current_day: Current in-game day
        
    Returns:
        Dictionary with generated conflict details
    """
    context = ctx.context
    
    # Prepare NPC information for the prompt
    npc_info = ""
    for i, npc in enumerate(stakeholder_npcs):
        npc_info += f"{i+1}. {npc['npc_name']} (Dominance: {npc['dominance']}, Cruelty: {npc['cruelty']}, Closeness: {npc['closeness']})\n"
    
    # Get player stats for context
    player_stats = await get_player_stats(ctx)
    
    prompt = f"""
    As an AI game system, generate a femdom-themed conflict with multiple stakeholders and complex motivations.

    Conflict Type: {conflict_type.capitalize()}
    Current Day: {current_day}
    
    Available NPCs to use as stakeholders:
    {npc_info}
    
    Player Stats:
    {json.dumps(player_stats, indent=2)}
    
    Generate the following details:
    1. A compelling conflict name and description
    2. 3-5 stakeholders with their own motivations and goals
    3. At least 3 distinct resolution paths with different approaches
    4. Potential internal faction conflicts that might emerge
    5. Opportunities for NPCs to manipulate the player using femdom themes
    
    Create stakeholders with:
    - Public and private motivations (what they claim vs. what they really want)
    - Relationships with other stakeholders (alliances and rivalries)
    - Faction affiliations and positions where applicable
    - Secrets that could be revealed during the conflict
    - Potential to manipulate the player based on dominance, corruption, etc.
    
    Create resolution paths that:
    - Allow different play styles (social, investigative, direct, etc.)
    - Require engaging with specific stakeholders
    - Have interesting narrative implications
    - Include key challenges to overcome
    
    Include opportunities for femdom-themed manipulation where:
    - Dominant female NPCs could try to control or influence the player
    - NPCs might use blackmail, seduction, or direct commands
    - Different paths could affect player corruption, obedience, etc.
    
    Return your response in JSON format including all these elements.
    """
    
    response = await get_chatgpt_response(
        context.conversation_id,
        conflict_type,
        prompt
    )
    
    if response and "function_args" in response:
        return response["function_args"]
    else:
        # Try to extract JSON from text response
        try:
            response_text = response.get("response", "{}")
            
            # Find JSON in the response
            json_match = re.search(r'{.*}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Fallback to basic structure
        return {
            "conflict_name": f"{conflict_type.capitalize()} Conflict",
            "conflict_type": conflict_type,
            "description": f"A {conflict_type} conflict involving multiple stakeholders with their own agendas.",
            "stakeholders": [
                {
                    "npc_id": npc["npc_id"],
                    "public_motivation": f"{npc['npc_name']} wants to resolve the conflict peacefully.",
                    "private_motivation": f"{npc['npc_name']} actually wants to gain power through the conflict.",
                    "desired_outcome": "Control the outcome to their advantage",
                    "faction_id": npc.get("faction_affiliations", [{}])[0].get("faction_id") if npc.get("faction_affiliations") else None,
                    "faction_name": npc.get("faction_affiliations", [{}])[0].get("faction_name") if npc.get("faction_affiliations") else None
                }
                for npc in stakeholder_npcs
            ],
            "resolution_paths": [
                {
                    "path_id": "diplomatic",
                    "name": "Diplomatic Resolution",
                    "description": "Resolve the conflict through negotiation and compromise.",
                    "approach_type": "social",
                    "difficulty": 5,
                    "stakeholders_involved": [npc["npc_id"] for npc in stakeholder_npcs[:2]],
                    "key_challenges": ["Building trust", "Finding common ground", "Managing expectations"]
                },
                {
                    "path_id": "force",
                    "name": "Forceful Resolution",
                    "description": "Resolve the conflict through direct action and confrontation.",
                    "approach_type": "direct",
                    "difficulty": 7,
                    "stakeholders_involved": [npc["npc_id"] for npc in stakeholder_npcs[1:3]],
                    "key_challenges": ["Overcoming resistance", "Managing collateral damage", "Securing victory"]
                },
                {
                    "path_id": "manipulation",
                    "name": "Manipulative Resolution",
                    "description": "Resolve the conflict by playing stakeholders against each other.",
                    "approach_type": "deception",
                    "difficulty": 8,
                    "stakeholders_involved": [npc["npc_id"] for npc in stakeholder_npcs[:3]],
                    "key_challenges": ["Maintaining deception", "Balancing interests", "Avoiding exposure"]
                }
            ],
            "internal_faction_conflicts": [],
            "player_manipulation_opportunities": [
                {
                    "npc_id": stakeholder_npcs[0]["npc_id"],
                    "manipulation_type": "domination",
                    "content": f"{stakeholder_npcs[0]['npc_name']} demands your help in the conflict, using her position of power over you.",
                    "goal": {
                        "faction": "a",
                        "involvement_level": "participating"
                    }
                }
            ]
        }

# Manipulation Tools

@function_tool
async def create_manipulation_attempt(
    ctx: RunContextWrapper,
    conflict_id: int,
    npc_id: int,
    manipulation_type: str,
    content: str,
    goal: Dict[str, Any],
    leverage_used: Dict[str, Any],
    intimacy_level: int = 0
) -> Dict[str, Any]:
    """
    Create a manipulation attempt by an NPC targeted at the player.
    
    Args:
        conflict_id: ID of the conflict
        npc_id: ID of the NPC doing the manipulation
        manipulation_type: Type of manipulation (domination, blackmail, seduction, etc.)
        content: Content of the manipulation attempt
        goal: What the NPC wants the player to do
        leverage_used: What leverage the NPC is using
        intimacy_level: Level of intimacy in the manipulation (0-10)
        
    Returns:
        The created manipulation attempt
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Check if NPC has relationship with player
            relationship = await get_npc_relationship_with_player(ctx, npc_id)
            
            # Get NPC name
            npc_name = await get_npc_name(ctx, npc_id)
            
            # Begin transaction
            async with conn.transaction():
                # Insert the manipulation attempt
                attempt_id = await conn.fetchval("""
                    INSERT INTO PlayerManipulationAttempts
                    (conflict_id, user_id, conversation_id, npc_id, manipulation_type, 
                     content, goal, success, leverage_used, intimacy_level, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, CURRENT_TIMESTAMP)
                    RETURNING attempt_id
                """, 
                conflict_id, context.user_id, context.conversation_id, npc_id,
                manipulation_type, content, json.dumps(goal), False,
                json.dumps(leverage_used), intimacy_level)
                
                # Create a memory for this manipulation attempt
                await create_conflict_memory(
                    ctx,
                    conflict_id,
                    f"{npc_name} attempted to {manipulation_type} the player regarding the conflict.",
                    significance=7
                )
                
                return {
                    "attempt_id": attempt_id,
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "manipulation_type": manipulation_type,
                    "content": content,
                    "goal": goal,
                    "leverage_used": leverage_used,
                    "intimacy_level": intimacy_level,
                    "success": False,
                    "is_resolved": False
                }
    except Exception as e:
        logger.error(f"Error creating player manipulation attempt: {e}", exc_info=True)
        raise

@function_tool
async def resolve_manipulation_attempt(
    ctx: RunContextWrapper,
    attempt_id: int,
    success: bool,
    player_response: str
) -> Dict[str, Any]:
    """
    Resolve a manipulation attempt by the player.
    
    Args:
        attempt_id: ID of the manipulation attempt
        success: Whether the manipulation was successful
        player_response: The player's response to the manipulation
        
    Returns:
        Updated manipulation attempt
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Get the manipulation attempt
            attempt_row = await conn.fetchrow("""
                SELECT conflict_id, npc_id, manipulation_type, goal
                FROM PlayerManipulationAttempts
                WHERE attempt_id = $1 AND user_id = $2 AND conversation_id = $3
            """, attempt_id, context.user_id, context.conversation_id)
            
            if not attempt_row:
                return {"error": "Manipulation attempt not found"}
            
            conflict_id = attempt_row["conflict_id"]
            npc_id = attempt_row["npc_id"]
            manipulation_type = attempt_row["manipulation_type"]
            goal = attempt_row["goal"]
            
            async with conn.transaction():
                # Update the manipulation attempt
                await conn.execute("""
                    UPDATE PlayerManipulationAttempts
                    SET success = $1, player_response = $2, resolved_at = CURRENT_TIMESTAMP
                    WHERE attempt_id = $3
                """, success, player_response, attempt_id)
                
                # Apply stat changes based on result
                stat_changes = {}
                
                if success:
                    # If player succumbed to manipulation
                    obedience_change = random.randint(2, 5)
                    dependency_change = random.randint(1, 3)
                    
                    await apply_stat_change(context.user_id, context.conversation_id, "obedience", obedience_change)
                    await apply_stat_change(context.user_id, context.conversation_id, "dependency", dependency_change)
                    
                    stat_changes["obedience"] = obedience_change
                    stat_changes["dependency"] = dependency_change
                    
                    # If successful, update player involvement based on goal
                    try:
                        goal_dict = json.loads(goal) if isinstance(goal, str) else goal or {}
                    except (json.JSONDecodeError, TypeError):
                        goal_dict = {}
                    
                    # Get current involvement
                    involvement_row = await conn.fetchrow("""
                        SELECT involvement_level, faction
                        FROM PlayerConflictInvolvement
                        WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
                    """, conflict_id, context.user_id, context.conversation_id)
                    
                    if involvement_row:
                        current_involvement = involvement_row["involvement_level"]
                        current_faction = involvement_row["faction"]
                        
                        # Update with goal values or keep current if not specified
                        faction = goal_dict.get("faction", current_faction)
                        involvement_level = goal_dict.get("involvement_level", current_involvement)
                    else:
                        # If no involvement yet, set defaults
                        faction = goal_dict.get("faction", "neutral")
                        involvement_level = goal_dict.get("involvement_level", "observing")
                    
                    # Record that player was manipulated
                    await conn.execute("""
                        UPDATE PlayerConflictInvolvement
                        SET involvement_level = $1, faction = $2, manipulated_by = $3
                        WHERE conflict_id = $4 AND user_id = $5 AND conversation_id = $6
                    """, 
                    involvement_level, faction, 
                    json.dumps({"npc_id": npc_id, "manipulation_type": manipulation_type, "attempt_id": attempt_id}),
                    conflict_id, context.user_id, context.conversation_id)
                    
                    # If no rows updated, insert new involvement
                    if await conn.fetchval("SELECT 1") is None:  # This is a way to check if any rows were affected
                        await conn.execute("""
                            INSERT INTO PlayerConflictInvolvement
                            (conflict_id, user_id, conversation_id, player_name, involvement_level,
                            faction, money_committed, supplies_committed, influence_committed, 
                            actions_taken, manipulated_by)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        """, 
                        conflict_id, context.user_id, context.conversation_id, "Player",
                        involvement_level, faction, 0, 0, 0, "[]",
                        json.dumps({"npc_id": npc_id, "manipulation_type": manipulation_type, "attempt_id": attempt_id}))
                        
                else:
                    # If player resisted manipulation
                    willpower_change = random.randint(2, 4)
                    confidence_change = random.randint(1, 3)
                    
                    await apply_stat_change(context.user_id, context.conversation_id, "willpower", willpower_change)
                    await apply_stat_change(context.user_id, context.conversation_id, "confidence", confidence_change)
                    
                    stat_changes["willpower"] = willpower_change
                    stat_changes["confidence"] = confidence_change
                
                # Get NPC name
                npc_name = await get_npc_name(ctx, npc_id)
                
                # Create a memory for the resolution
                if success:
                    await create_conflict_memory(
                        ctx,
                        conflict_id,
                        f"Player succumbed to {npc_name}'s {manipulation_type} attempt in the conflict.",
                        significance=8
                    )
                else:
                    await create_conflict_memory(
                        ctx,
                        conflict_id,
                        f"Player resisted {npc_name}'s {manipulation_type} attempt in the conflict.",
                        significance=7
                    )
                
                return {
                    "attempt_id": attempt_id,
                    "success": success,
                    "player_response": player_response,
                    "is_resolved": True,
                    "stat_changes": stat_changes
                }
    except Exception as e:
        logger.error(f"Error resolving manipulation attempt: {e}", exc_info=True)
        raise

@function_tool
async def suggest_manipulation_content(
    ctx: RunContextWrapper,
    npc_id: int,
    conflict_id: int,
    manipulation_type: str,
    goal: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Suggest manipulation content for an NPC based on their relationship with the player
    and the desired outcome.
    
    Args:
        npc_id: ID of the NPC
        conflict_id: ID of the conflict
        manipulation_type: Type of manipulation (domination, blackmail, seduction, etc.)
        goal: What the NPC wants the player to do
        
    Returns:
        Suggested manipulation content
    """
    context = ctx.context
    
    # Get NPC details
    npc = await get_npc_details(ctx, npc_id)
    
    # Get relationship status
    relationship = await get_npc_relationship_with_player(ctx, npc_id)
    
    # Get conflict details
    conflict = await get_conflict_details(ctx, conflict_id)
    
    # Generate content based on manipulation type
    content = ""
    
    if manipulation_type == "domination":
        content = generate_domination_content(npc, relationship, goal, conflict)
    elif manipulation_type == "seduction":
        content = generate_seduction_content(npc, relationship, goal, conflict)
    elif manipulation_type == "blackmail":
        # Get leverage
        leverage = await get_manipulation_leverage(
            context.user_id, context.conversation_id, npc_id
        )
        content = generate_blackmail_content(npc, relationship, goal, conflict, leverage)
    else:
        content = generate_generic_manipulation_content(npc, relationship, goal, conflict)
    
    # Generate appropriate leverage
    leverage_used = generate_leverage(npc, relationship, manipulation_type)
    
    # Determine intimacy level
    intimacy_level = calculate_intimacy_level(npc, relationship, manipulation_type)
    
    return {
        "npc_id": npc_id,
        "npc_name": npc.get("npc_name", "Unknown"),
        "manipulation_type": manipulation_type,
        "content": content,
        "leverage_used": leverage_used,
        "intimacy_level": intimacy_level,
        "goal": goal
    }

@function_tool
async def analyze_manipulation_potential(
    ctx: RunContextWrapper, 
    npc_id: int,
    player_stats: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Analyze an NPC's potential to manipulate the player based on their relationship
    and the player's current stats.
    
    Args:
        npc_id: ID of the NPC
        player_stats: Optional player stats (will be fetched if not provided)
        
    Returns:
        Dictionary with manipulation potential analysis
    """
    context = ctx.context
    
    # Get NPC details
    npc = await get_npc_details(ctx, npc_id)
    
    # Get relationship status
    relationship = await get_relationship_status(
        context.user_id, context.conversation_id, npc_id
    )
    
    # Get potential leverage
    leverage = await get_manipulation_leverage(
        context.user_id, context.conversation_id, npc_id
    )
    
    # Get player stats if not provided
    if not player_stats:
        player_stats = await get_player_stats(ctx)
    
    # Calculate manipulation potential for different types
    domination_potential = min(100, npc.get("dominance", 0) - player_stats.get("willpower", 50) + 50)
    seduction_potential = min(100, relationship.get("closeness", 0) + player_stats.get("lust", 20))
    blackmail_potential = min(100, 50 + (len(leverage) * 15))
    
    # Determine most effective manipulation type
    manipulation_types = [
        {"type": "domination", "potential": domination_potential},
        {"type": "seduction", "potential": seduction_potential},
        {"type": "blackmail", "potential": blackmail_potential}
    ]
    
    most_effective = max(manipulation_types, key=lambda x: x["potential"])
    
    # Determine overall manipulation potential
    overall_potential = most_effective["potential"]
    
    return {
        "npc_id": npc_id,
        "npc_name": npc.get("npc_name", "Unknown"),
        "overall_potential": overall_potential,
        "manipulation_types": manipulation_types,
        "most_effective_type": most_effective["type"],
        "relationship": relationship,
        "available_leverage": leverage,
        "femdom_compatible": npc.get("sex", "female") == "female" and domination_potential > 60
    }

# Resolution Path Tools

@function_tool
async def track_story_beat(
    ctx: RunContextWrapper,
    conflict_id: int,
    path_id: str,
    beat_description: str,
    involved_npcs: List[int],
    progress_value: float
) -> Dict[str, Any]:
    """
    Track a story beat for a resolution path, advancing progress.
    
    Args:
        conflict_id: ID of the conflict
        path_id: ID of the resolution path
        beat_description: Description of what happened
        involved_npcs: List of NPC IDs involved
        progress_value: Progress value (0-100)
        
    Returns:
        Updated path information
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            async with conn.transaction():
                # Create the story beat
                beat_id = await conn.fetchval("""
                    INSERT INTO PathStoryBeats
                    (conflict_id, path_id, description, involved_npcs, progress_value, created_at)
                    VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)
                    RETURNING beat_id
                """, 
                conflict_id, path_id, beat_description, 
                json.dumps(involved_npcs), progress_value)
                
                # Get current path progress
                path_row = await conn.fetchrow("""
                    SELECT progress, is_completed
                    FROM ResolutionPaths
                    WHERE conflict_id = $1 AND path_id = $2
                """, conflict_id, path_id)
                
                if not path_row:
                    return {"error": "Resolution path not found"}
                
                current_progress = path_row["progress"]
                is_completed = path_row["is_completed"]
                
                # Calculate new progress
                new_progress = min(100, current_progress + progress_value)
                is_now_completed = new_progress >= 100
                
                # Update the path progress
                await conn.execute("""
                    UPDATE ResolutionPaths
                    SET progress = $1, is_completed = $2,
                        completion_date = CASE WHEN $2 = TRUE THEN CURRENT_TIMESTAMP ELSE NULL END
                    WHERE conflict_id = $3 AND path_id = $4
                """, 
                new_progress, is_now_completed, conflict_id, path_id)
                
                # If path completed, check if conflict should advance
                if is_now_completed:
                    await check_conflict_advancement(ctx, conflict_id)
                
                # Create a memory for this story beat
                await create_conflict_memory(
                    ctx,
                    conflict_id,
                    f"Progress made on path '{path_id}': {beat_description}",
                    significance=6
                )
                
                return {
                    "beat_id": beat_id,
                    "conflict_id": conflict_id,
                    "path_id": path_id,
                    "description": beat_description,
                    "progress_value": progress_value,
                    "new_progress": new_progress,
                    "is_completed": is_now_completed
                }
    except Exception as e:
        logger.error(f"Error tracking story beat: {e}", exc_info=True)
        raise

@function_tool
async def resolve_conflict(ctx: RunContextWrapper, conflict_id: int) -> Dict[str, Any]:
    """
    Resolve a conflict and apply consequences.
    
    Args:
        conflict_id: ID of the conflict to resolve
        
    Returns:
        Resolution details and consequences
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Get conflict details
            conflict_row = await conn.fetchrow("""
                SELECT conflict_name, conflict_type, phase, progress
                FROM Conflicts
                WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
            """, conflict_id, context.user_id, context.conversation_id)
            
            if not conflict_row:
                return {"error": "Conflict not found"}
            
            conflict_name = conflict_row["conflict_name"]
            conflict_type = conflict_row["conflict_type"]
            phase = conflict_row["phase"]
            progress = conflict_row["progress"]
            
            # Check if conflict is in a resolvable state
            if phase != "resolution" and progress < 90:
                return {"error": "Conflict is not ready to be resolved. Progress must be at least 90%."}
            
            # Get player involvement
            player_row = await conn.fetchrow("""
                SELECT involvement_level, faction
                FROM PlayerConflictInvolvement
                WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
            """, conflict_id, context.user_id, context.conversation_id)
            
            player_involvement = player_row["involvement_level"] if player_row else "none"
            player_faction = player_row["faction"] if player_row else "neutral"
            
            # Get completed resolution paths
            completed_paths = await conn.fetch("""
                SELECT path_id, name
                FROM ResolutionPaths
                WHERE conflict_id = $1 AND is_completed = TRUE
            """, conflict_id)
            
            # Determine outcome based on completed paths and player involvement
            if not completed_paths:
                outcome = "unresolved"
                description = "The conflict ended without a clear resolution."
            else:
                # Determine winning faction based on completed paths
                # (Simplified logic - in real implementation would be more complex)
                outcome = "resolved"
                winning_faction = player_faction if player_involvement != "none" else "neutral"
                description = f"The conflict was resolved with {winning_faction} faction gaining the advantage."
            
            async with conn.transaction():
                # Update conflict status
                await conn.execute("""
                    UPDATE Conflicts
                    SET is_active = FALSE, progress = 100, phase = 'concluded',
                        outcome = $1, resolution_description = $2,
                        resolved_at = CURRENT_TIMESTAMP
                    WHERE conflict_id = $3
                """, outcome, description, conflict_id)
                
                # Generate consequences
                consequences = generate_conflict_consequences(
                    conflict_type, outcome, player_involvement, player_faction, [dict(row) for row in completed_paths]
                )
                
                # Record consequences
                for consequence in consequences:
                    await conn.execute("""
                        INSERT INTO ConflictConsequences
                        (conflict_id, consequence_type, description, affected_entity_type,
                         affected_entity_id, magnitude, is_permanent)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """, 
                    conflict_id,
                    consequence.get("type", "general"),
                    consequence.get("description", ""),
                    consequence.get("affected_entity_type", "player"),
                    consequence.get("affected_entity_id", 0),
                    consequence.get("magnitude", 1),
                    consequence.get("is_permanent", False))
                    
                    # Apply stat changes if applicable
                    if consequence.get("affected_entity_type") == "player" and "stat_changes" in consequence:
                        for stat, value in consequence["stat_changes"].items():
                            await apply_stat_change(context.user_id, context.conversation_id, stat, value)
                
                # Create memory
                await create_conflict_memory(
                    ctx,
                    conflict_id,
                    f"The conflict '{conflict_name}' has been resolved. {description}",
                    significance=9
                )
                
                return {
                    "conflict_id": conflict_id,
                    "outcome": outcome,
                    "description": description,
                    "consequences": consequences,
                    "resolved_at": "now"
                }
    except Exception as e:
        logger.error(f"Error resolving conflict: {e}", exc_info=True)
        raise

# Faction Power Struggle Tools (Added based on review)

@function_tool
async def initiate_faction_power_struggle(
    ctx: RunContextWrapper,
    conflict_id: int,
    faction_id: int,
    challenger_npc_id: int,
    target_npc_id: int,
    prize: str,
    approach: str,
    is_public: bool = False
) -> Dict[str, Any]:
    """
    Initiate a power struggle within a faction.
    
    Args:
        conflict_id: The main conflict ID
        faction_id: The faction where struggle occurs
        challenger_npc_id: NPC initiating the challenge
        target_npc_id: NPC being challenged (usually leader)
        prize: What's at stake (position, policy, etc.)
        approach: How the challenge is made (direct, subtle, etc.)
        is_public: Whether other stakeholders are aware
        
    Returns:
        The created internal faction conflict
    """
    context = ctx.context
    
    # Get faction name
    faction_name = await get_faction_name(ctx, faction_id)
    
    # Get NPC names
    challenger_name = await get_npc_name(ctx, challenger_npc_id)
    target_name = await get_npc_name(ctx, target_npc_id)
    
    # Generate struggle details
    struggle_details = await generate_struggle_details(
        ctx, faction_id, challenger_npc_id, target_npc_id, prize, approach
    )
    
    # Create in database
    try:
        async with get_db_connection_context() as conn:
            async with conn.transaction():
                # Insert the struggle
                struggle_id = await conn.fetchval("""
                    INSERT INTO InternalFactionConflicts
                    (faction_id, conflict_name, description, primary_npc_id, target_npc_id,
                     prize, approach, public_knowledge, current_phase, progress, parent_conflict_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING struggle_id
                """, 
                faction_id, struggle_details["conflict_name"], struggle_details["description"],
                challenger_npc_id, target_npc_id, prize, approach, is_public,
                "brewing", 10, conflict_id)
                
                # Insert faction members positions
                for member in struggle_details.get("faction_members", []):
                    await conn.execute("""
                        INSERT INTO FactionStruggleMembers
                        (struggle_id, npc_id, position, side, standing, 
                         loyalty_strength, reason)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """, 
                    struggle_id, member["npc_id"], member.get("position", "Member"),
                    member.get("side", "neutral"), member.get("standing", 50),
                    member.get("loyalty_strength", 50), member.get("reason", ""))
                
                # Insert ideological differences
                for diff in struggle_details.get("ideological_differences", []):
                    await conn.execute("""
                        INSERT INTO FactionIdeologicalDifferences
                        (struggle_id, issue, incumbent_position, challenger_position)
                        VALUES ($1, $2, $3, $4)
                    """, 
                    struggle_id, diff.get("issue", ""), 
                    diff.get("incumbent_position", ""),
                    diff.get("challenger_position", ""))
                
                # Create a memory for this power struggle
                await create_conflict_memory(
                    ctx,
                    conflict_id,
                    f"Internal power struggle has emerged in {faction_name} between {challenger_name} and {target_name}.",
                    significance=7
                )
                
                # Return the created struggle
                return {
                    "struggle_id": struggle_id,
                    "faction_id": faction_id,
                    "faction_name": faction_name,
                    "conflict_name": struggle_details["conflict_name"],
                    "description": struggle_details["description"],
                    "primary_npc_id": challenger_npc_id,
                    "primary_npc_name": challenger_name,
                    "target_npc_id": target_npc_id,
                    "target_npc_name": target_name,
                    "prize": prize,
                    "approach": approach,
                    "public_knowledge": is_public,
                    "current_phase": "brewing",
                    "progress": 10
                }
    except Exception as e:
        logger.error(f"Error initiating faction power struggle: {e}", exc_info=True)
        raise

@function_tool
async def attempt_faction_coup(
    ctx: RunContextWrapper,
    struggle_id: int,
    approach: str,
    supporting_npcs: List[int],
    resources_committed: Dict[str, int]
) -> Dict[str, Any]:
    """
    Attempt a coup within a faction to forcefully resolve a power struggle.
    
    Args:
        struggle_id: ID of the internal faction struggle
        approach: The approach used (direct, subtle, force, blackmail)
        supporting_npcs: List of NPCs supporting the coup
        resources_committed: Resources committed to the coup
        
    Returns:
        Coup attempt results
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Get struggle details
            struggle_row = await conn.fetchrow("""
                SELECT faction_id, primary_npc_id, target_npc_id, parent_conflict_id
                FROM InternalFactionConflicts
                WHERE struggle_id = $1
            """, struggle_id)
            
            if not struggle_row:
                return {"error": "Struggle not found"}
            
            faction_id = struggle_row["faction_id"]
            primary_npc_id = struggle_row["primary_npc_id"]
            target_npc_id = struggle_row["target_npc_id"]
            parent_conflict_id = struggle_row["parent_conflict_id"]
            
            # Check if player has sufficient resources
            resource_total = sum(resources_committed.values())
            if resource_total > 0:
                resource_manager = ResourceManager(context.user_id, context.conversation_id)
                resource_check = await resource_manager.check_resources(
                    resources_committed.get("money", 0),
                    resources_committed.get("supplies", 0),
                    resources_committed.get("influence", 0)
                )
                
                if not resource_check["has_resources"]:
                    return {
                        "error": "Insufficient resources to commit to coup",
                        "missing": resource_check["missing"],
                        "current": resource_check["current"]
                    }
                
                # Commit resources
                await resource_manager.commit_resources(
                    resources_committed.get("money", 0),
                    resources_committed.get("supplies", 0),
                    resources_committed.get("influence", 0),
                    "Committed to faction coup attempt"
                )
            
            # Calculate coup success chance
            success_chance = await calculate_coup_success_chance(
                ctx, struggle_id, approach, supporting_npcs, resources_committed
            )
            
            # Determine outcome
            success = random.random() * 100 <= success_chance
            
            async with conn.transaction():
                # Record coup attempt
                coup_id = await conn.fetchval("""
                    INSERT INTO FactionCoupAttempts
                    (struggle_id, approach, supporting_npcs, resources_committed,
                     success, success_chance, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
                    RETURNING id
                """, 
                struggle_id, approach, json.dumps(supporting_npcs),
                json.dumps(resources_committed), success, success_chance)
                
                # Get primary and target names
                primary_name = await get_npc_name(ctx, primary_npc_id)
                target_name = await get_npc_name(ctx, target_npc_id)
                faction_name = await get_faction_name(ctx, faction_id)
                
                # Generate result based on success
                if success:
                    # Update the struggle
                    await conn.execute("""
                        UPDATE InternalFactionConflicts
                        SET current_phase = $1, progress = 100,
                            resolved_at = CURRENT_TIMESTAMP
                        WHERE struggle_id = $2
                    """, "resolved", struggle_id)
                    
                    # Create memory for successful coup
                    await create_conflict_memory(
                        ctx,
                        parent_conflict_id,
                        f"{primary_name}'s coup against {target_name} in {faction_name} has succeeded.",
                        significance=8
                    )
                    
                    result = {
                        "outcome": "success",
                        "description": f"{primary_name} has successfully overthrown {target_name} and taken control of {faction_name}.",
                        "consequences": [
                            f"{primary_name} now controls {faction_name}",
                            f"{target_name} has been removed from power",
                            "The balance of power in the conflict has shifted"
                        ]
                    }
                    
                    # Apply stat changes for successful coup
                    await apply_stat_change(context.user_id, context.conversation_id, "corruption", 3)
                    await apply_stat_change(context.user_id, context.conversation_id, "confidence", 5)
                    
                    stat_changes = {
                        "corruption": 3,
                        "confidence": 5
                    }
                else:
                    # Update the struggle
                    await conn.execute("""
                        UPDATE InternalFactionConflicts
                        SET current_phase = $1, primary_npc_id = $2, target_npc_id = $3,
                            description = $4
                        WHERE struggle_id = $5
                    """, 
                    "aftermath",
                    target_npc_id,  # Roles reversed now
                    primary_npc_id, 
                    f"After a failed coup attempt, {target_name} has consolidated power and {primary_name} is now at their mercy.",
                    struggle_id)
                    
                    # Create memory for failed coup
                    await create_conflict_memory(
                        ctx,
                        parent_conflict_id,
                        f"{primary_name}'s coup against {target_name} in {faction_name} has failed.",
                        significance=8
                    )
                    
                    result = {
                        "outcome": "failure",
                        "description": f"{primary_name}'s attempt to overthrow {target_name} has failed, leaving them vulnerable to retaliation.",
                        "consequences": [
                            f"{target_name} has strengthened their position in {faction_name}",
                            f"{primary_name} is now in a dangerous position",
                            "Supporting NPCs may face punishment"
                        ]
                    }
                    
                    # Apply stat changes for failed coup
                    await apply_stat_change(context.user_id, context.conversation_id, "mental_resilience", 4)
                    
                    stat_changes = {
                        "mental_resilience": 4
                    }
                
                # Record result in database
                await conn.execute("""
                    UPDATE FactionCoupAttempts
                    SET result = $1
                    WHERE id = $2
                """, json.dumps(result), coup_id)
                
                # Get updated resources
                resources = await resource_manager.get_resources()
                
                return {
                    "coup_id": coup_id,
                    "struggle_id": struggle_id,
                    "approach": approach,
                    "success": success,
                    "success_chance": success_chance,
                    "result": result,
                    "stat_changes": stat_changes,
                    "resources": resources
                }
    except Exception as e:
        logger.error(f"Error attempting faction coup: {e}", exc_info=True)
        raise

# Narrative Integration Tool (Added based on review)

async def _internal_add_conflict_to_narrative_logic(ctx: RunContextWrapper, narrative_text: str) -> Dict[str, Any]:
    """
    Core logic to analyze narrative text and potentially add a new conflict.
    This function is intended for direct Python calls.
    """
    
    # Get current day
    current_day = await get_current_day(ctx)
    
    # Get active conflicts to avoid overloading
    active_conflicts = await get_active_conflicts(ctx)
    if len(active_conflicts) >= 3:
        return {
            "trigger_conflict": False,
            "reason": "Too many active conflicts already exist",
            "existing_conflicts": len(active_conflicts)
        }
    
    # Analyze narrative for conflict triggers
    # This could use GPT to analyze the text for potential conflicts
    prompt = f"""
    Analyze the following narrative text to determine if it contains potential for a character-driven conflict:
    
    "{narrative_text}"
    
    Consider:
    1. Are there tensions between characters?
    2. Are there opposing factions or interests?
    3. Are there secrets, betrayals, or hidden agendas mentioned?
    4. Is there a power dynamic that could create conflict?
    5. Are there femdom themes that could be developed into a conflict?
    
    If a conflict potential exists, provide:
    - The type of conflict (minor, standard, major)
    - A brief description of the potential conflict
    - Which NPCs might be involved
    - What femdom themes could be incorporated
    
    Return your response in JSON format.
    """
    
    response = await get_chatgpt_response( # Assuming this is a callable async function
        context.conversation_id,
        "conflict_analysis",
        prompt
    )
    
    conflict_analysis = {}
    if response and "function_args" in response:
        conflict_analysis = response["function_args"]
    else:
        try:
            response_text = response.get("response", "{}")
            json_match = re.search(r'{.*}', response_text, re.DOTALL)
            if json_match:
                conflict_analysis = json.loads(json_match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass # Keep conflict_analysis as {}
    
    should_generate = conflict_analysis.get("conflict_potential", False)
    if not should_generate:
        return {
            "trigger_conflict": False,
            "reason": "No significant conflict potential detected in narrative",
            "analysis": conflict_analysis
        }
    
    # Assuming extract_npcs_from_narrative, create_conflict_record, etc. are callable async functions
    mentioned_npcs = await extract_npcs_from_narrative(ctx, narrative_text)
    if not mentioned_npcs or len(mentioned_npcs) < 2:
        return {
            "trigger_conflict": False,
            "reason": "Not enough NPCs involved in the narrative",
            "mentioned_npcs": mentioned_npcs
        }
    
    conflict_type = conflict_analysis.get("conflict_type", "minor")
    
    conflict_data = {
        "conflict_type": conflict_type,
        "conflict_name": conflict_analysis.get("conflict_name", f"Narrative-triggered {conflict_type} conflict"),
        "description": conflict_analysis.get("description", "A conflict arising from recent events"),
        "stakeholders": [],
        "resolution_paths": [],
        "narrative_source": narrative_text[:100] + "..." if len(narrative_text) > 100 else narrative_text
    }
    
    conflict_id = await create_conflict_record(ctx, conflict_data, current_day)
    
    stakeholder_npcs_details = []
    for npc_id_val in mentioned_npcs[:4]:
        npc_details = await get_npc_details(ctx, npc_id_val) # Ensure get_npc_details is callable
        if npc_details:
            stakeholder_npcs_details.append(npc_details)
    
    await create_stakeholders(ctx, conflict_id, conflict_data, stakeholder_npcs_details) # Ensure this is callable
    
    # Generate basic resolution paths
    default_paths = [
        {
            "path_id": "diplomatic",
            "name": "Diplomatic Resolution",
            "description": "Resolve the conflict through negotiation and compromise.",
            "approach_type": "social",
            "difficulty": 5,
            "stakeholders_involved": [npc["npc_id"] for npc in stakeholder_npcs[:2]],
            "key_challenges": ["Building trust", "Finding common ground", "Managing expectations"]
        },
        {
            "path_id": "direct",
            "name": "Direct Resolution",
            "description": "Resolve the conflict through confrontation and decisive action.",
            "approach_type": "direct",
            "difficulty": 7,
            "stakeholders_involved": [npc["npc_id"] for npc in stakeholder_npcs[1:3]],
            "key_challenges": ["Overcoming resistance", "Managing consequences", "Securing victory"]
        }
    ]
    
    conflict_data["resolution_paths"] = default_paths # This seems to be modifying a local dict, ensure it's used correctly if needed.
    await create_resolution_paths(ctx, conflict_id, {"resolution_paths": default_paths}) # Pass the paths correctly
    
    await create_conflict_memory( # Ensure this is callable
        ctx,
        conflict_id,
        f"A new conflict has emerged from the narrative: {conflict_data['conflict_name']}.",
        significance=7
    )
    
    return {
        "trigger_conflict": True,
        "conflict_id": conflict_id,
        "conflict_name": conflict_data["conflict_name"],
        "conflict_type": conflict_type,
        "stakeholders": [npc["npc_name"] for npc in stakeholder_npcs_details], # Use details for names
        "analysis": conflict_analysis
    }

@function_tool
# @track_performance("add_conflict_to_narrative_tool") # Optional: if you want to track the tool call separately
async def add_conflict_to_narrative(ctx: RunContextWrapper, narrative_text: str) -> Dict[str, Any]:
    """
    OpenAI Agent Tool: Analyzes narrative text to identify and add conflicts.
    (This is the tool definition for the agent framework)
    
    Args:
        narrative_text: The narrative text to analyze.

    Returns:
        A dictionary detailing the outcome of the conflict analysis.
    """
    return await _internal_add_conflict_to_narrative_logic(ctx, narrative_text)


# Helper Functions

@function_tool
async def get_npc_details(ctx: RunContextWrapper, npc_id: int) -> Dict[str, Any]:
    """Get details for an NPC."""
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            npc_row = await conn.fetchrow("""
                SELECT npc_name, dominance, cruelty, closeness, trust, respect, intensity, sex
                FROM NPCStats
                WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
            """, npc_id, context.user_id, context.conversation_id)
            
            if not npc_row:
                return {}
            
            npc = dict(npc_row)
            npc["npc_id"] = npc_id
            
            return npc
    except Exception as e:
        logger.error(f"Error getting NPC details: {e}", exc_info=True)
        return {}

@function_tool
async def get_npc_name(ctx: RunContextWrapper, npc_id: int) -> str:
    """Get an NPC's name by ID."""
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            npc_name = await conn.fetchval("""
                SELECT npc_name
                FROM NPCStats
                WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
            """, npc_id, context.user_id, context.conversation_id)
            
            return npc_name if npc_name else f"NPC {npc_id}"
    except Exception as e:
        logger.error(f"Error getting NPC name for ID {npc_id}: {e}", exc_info=True)
        return f"NPC {npc_id}"

@function_tool
async def get_faction_name(ctx: RunContextWrapper, faction_id: int) -> str:
    """Get a faction's name by ID."""
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            faction_name = await conn.fetchval("""
                SELECT faction_name
                FROM Factions
                WHERE faction_id = $1
            """, faction_id)
            
            return faction_name if faction_name else f"Faction {faction_id}"
    except Exception as e:
        logger.error(f"Error getting faction name for ID {faction_id}: {e}", exc_info=True)
        return f"Faction {faction_id}"

@function_tool
async def get_player_stats(ctx: RunContextWrapper) -> Dict[str, Any]:
    """Get player stats."""
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            stats_row = await conn.fetchrow("""
                SELECT corruption, confidence, willpower, obedience, dependency, lust,
                       mental_resilience, physical_endurance
                FROM PlayerStats
                WHERE user_id = $1 AND conversation_id = $2
            """, context.user_id, context.conversation_id)
            
            if not stats_row:
                return {}
            
            return dict(stats_row)
    except Exception as e:
        logger.error(f"Error getting player stats: {e}", exc_info=True)
        return {}

@function_tool
async def get_stakeholder_secrets(ctx: RunContextWrapper, conflict_id: int, npc_id: int) -> List[Dict[str, Any]]:
    """Get secrets for a stakeholder in a conflict."""
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            secrets_rows = await conn.fetch("""
                SELECT secret_id, secret_type, content, target_npc_id,
                       is_revealed, revealed_to, is_public
                FROM StakeholderSecrets
                WHERE conflict_id = $1 AND npc_id = $2
            """, conflict_id, npc_id)
            
            secrets = []
            for row in secrets_rows:
                secret = dict(row)
                
                # Only return details if the secret is revealed
                if secret["is_revealed"]:
                    secrets.append(secret)
                else:
                    # Otherwise just return that a secret exists
                    secrets.append({
                        "secret_id": secret["secret_id"],
                        "secret_type": secret["secret_type"],
                        "is_revealed": False
                    })
            
            return secrets
    except Exception as e:
        logger.error(f"Error getting stakeholder secrets: {e}", exc_info=True)
        return []

@function_tool
async def check_stakeholder_manipulates_player(ctx: RunContextWrapper, conflict_id: int, npc_id: int) -> bool:
    """Check if a stakeholder has manipulation attempts against the player."""
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            count = await conn.fetchval("""
                SELECT COUNT(*)
                FROM PlayerManipulationAttempts
                WHERE conflict_id = $1 AND npc_id = $2 AND user_id = $3 AND conversation_id = $4
            """, conflict_id, npc_id, context.user_id, context.conversation_id)
            
            return count > 0
    except Exception as e:
        logger.error(f"Error checking if stakeholder manipulates player: {e}", exc_info=True)
        return False

@function_tool
async def create_conflict_memory(
    ctx: RunContextWrapper, 
    conflict_id: int, 
    memory_text: str,
    significance: int = 5
) -> int:
    """
    Create a memory event for a conflict.
    
    Args:
        conflict_id: ID of the conflict
        memory_text: Text of the memory
        significance: Significance level (1-10)
        
    Returns:
        ID of the created memory
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            memory_id = await conn.fetchval("""
                INSERT INTO ConflictMemoryEvents 
                (conflict_id, memory_text, significance, entity_type, entity_id, user_id, conversation_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
            """, conflict_id, memory_text, significance, "conflict", conflict_id, context.user_id, context.conversation_id)
            
            return memory_id
    except Exception as e:
        logger.error(f"Error creating conflict memory: {e}", exc_info=True)
        return 0

@function_tool
async def check_conflict_advancement(ctx: RunContextWrapper, conflict_id: int) -> None:
    """
    Check if a conflict should advance to the next phase.
    
    Args:
        conflict_id: ID of the conflict to check
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Get conflict details
            conflict_row = await conn.fetchrow("""
                SELECT progress, phase
                FROM Conflicts
                WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
            """, conflict_id, context.user_id, context.conversation_id)
            
            if not conflict_row:
                return
            
            progress = conflict_row["progress"]
            phase = conflict_row["phase"]
            
            # Phase transition thresholds
            phase_thresholds = {
                "brewing": 30,    # brewing -> active
                "active": 60,     # active -> climax
                "climax": 90      # climax -> resolution
            }
            
            # Check if we should transition to a new phase
            new_phase = phase
            if phase in phase_thresholds and progress >= phase_thresholds[phase]:
                if phase == "brewing":
                    new_phase = "active"
                elif phase == "active":
                    new_phase = "climax"
                elif phase == "climax":
                    new_phase = "resolution"
            
            # If phase changed, update the conflict
            if new_phase != phase:
                async with conn.transaction():
                    await conn.execute("""
                        UPDATE Conflicts
                        SET phase = $1, updated_at = CURRENT_TIMESTAMP
                        WHERE conflict_id = $2 AND user_id = $3 AND conversation_id = $4
                    """, new_phase, conflict_id, context.user_id, context.conversation_id)
                    
                    # Create a memory for the phase transition
                    await create_conflict_memory(
                        ctx,
                        conflict_id,
                        f"The conflict has progressed from {phase} to {new_phase} phase.",
                        significance=7
                    )
    except Exception as e:
        logger.error(f"Error checking conflict advancement: {e}", exc_info=True)

@function_tool
async def generate_struggle_details(
    ctx: RunContextWrapper,
    faction_id: int,
    challenger_npc_id: int,
    target_npc_id: int,
    prize: str,
    approach: str
) -> Dict[str, Any]:
    """
    Generate details for a faction power struggle.
    
    Args:
        faction_id: ID of the faction
        challenger_npc_id: ID of the challenging NPC
        target_npc_id: ID of the target NPC
        prize: What's at stake
        approach: How the challenge is made
        
    Returns:
        Dictionary with struggle details
    """
    # Get faction name
    faction_name = await get_faction_name(ctx, faction_id)
    
    # Get NPC names
    challenger_name = await get_npc_name(ctx, challenger_npc_id)
    target_name = await get_npc_name(ctx, target_npc_id)
    
    # Get faction members
    members = await get_faction_members(ctx, faction_id)
    
    # Generate a conflict name and description
    conflict_name = f"Power struggle in {faction_name}"
    description = f"{challenger_name} challenges {target_name} for {prize} within {faction_name}."
    
    # Divide members between challenger, target, and neutral
    from collections import defaultdict
    sides = defaultdict(list)
    
    for member in members:
        # Skip challenger and target
        if member["npc_id"] == challenger_npc_id or member["npc_id"] == target_npc_id:
            continue
        
        # Assign based on relationships and random chance
        affinity_to_challenger = random.randint(0, 100)
        affinity_to_target = random.randint(0, 100)
        
        if abs(affinity_to_challenger - affinity_to_target) < 20:
            side = "neutral"
        elif affinity_to_challenger > affinity_to_target:
            side = "challenger"
        else:
            side = "incumbent"
        
        sides[side].append(member)
    
    # Create faction members list with positions
    faction_members = [
        {
            "npc_id": challenger_npc_id,
            "position": "Challenger",
            "side": "challenger",
            "standing": 70,
            "loyalty_strength": 100,
            "reason": "Leading the challenge"
        },
        {
            "npc_id": target_npc_id,
            "position": "Incumbent",
            "side": "incumbent",
            "standing": 80,
            "loyalty_strength": 100,
            "reason": "Defending position"
        }
    ]
    
    # Add supporters
    for side, members_list in sides.items():
        for i, member in enumerate(members_list):
            faction_members.append({
                "npc_id": member["npc_id"],
                "position": member.get("position", "Member"),
                "side": side,
                "standing": random.randint(30, 70),
                "loyalty_strength": random.randint(40, 90),
                "reason": f"Supports {side}"
            })
    
    # Generate ideological differences
    ideological_differences = [
        {
            "issue": f"Approach to {prize}",
            "incumbent_position": f"{target_name}'s traditional approach",
            "challenger_position": f"{challenger_name}'s new vision"
        },
        {
            "issue": "Faction methodology",
            "incumbent_position": "Maintain current methods",
            "challenger_position": "Implement reforms"
        }
    ]
    
    # Create the full struggle details
    struggle_details = {
        "conflict_name": conflict_name,
        "description": description,
        "faction_members": faction_members,
        "ideological_differences": ideological_differences
    }
    
    return struggle_details

@function_tool
async def get_faction_members(ctx: RunContextWrapper, faction_id: int) -> List[Dict[str, Any]]:
    """Get members of a faction."""
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # This assumes NPCs store faction affiliations in a JSON array field
            npc_rows = await conn.fetch("""
                SELECT npc_id, npc_name, dominance, cruelty, faction_affiliations
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2
            """, context.user_id, context.conversation_id)
            
            members = []
            for row in npc_rows:
                npc = dict(row)
                
                # Parse faction affiliations
                try:
                    affiliations = json.loads(npc["faction_affiliations"]) if isinstance(npc["faction_affiliations"], str) else npc["faction_affiliations"] or []
                except (json.JSONDecodeError, TypeError):
                    affiliations = []
                
                # Check if NPC is affiliated with this faction
                is_member = False
                position = "Member"
                
                for affiliation in affiliations:
                    if affiliation.get("faction_id") == faction_id:
                        is_member = True
                        position = affiliation.get("position", "Member")
                        break
                
                if is_member:
                    members.append({
                        "npc_id": npc["npc_id"],
                        "npc_name": npc["npc_name"],
                        "dominance": npc["dominance"],
                        "cruelty": npc["cruelty"],
                        "position": position
                    })
            
            return members
    except Exception as e:
        logger.error(f"Error getting faction members: {e}", exc_info=True)
        return []

@function_tool
async def extract_npcs_from_narrative(ctx: RunContextWrapper, narrative_text: str) -> List[int]:
    """
    Extract NPC IDs mentioned in a narrative text.
    
    Args:
        narrative_text: The narrative text to analyze
        
    Returns:
        List of NPC IDs mentioned in the text
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Get all NPCs for this user/conversation
            npc_rows = await conn.fetch("""
                SELECT npc_id, npc_name
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2 AND introduced = TRUE
            """, context.user_id, context.conversation_id)
            
            mentioned_npcs = []
            
            # Check each NPC name in the narrative
            for row in npc_rows:
                npc_id = row["npc_id"]
                npc_name = row["npc_name"]
                
                # Simple check - can be improved with more sophisticated NLP
                if npc_name in narrative_text:
                    mentioned_npcs.append(npc_id)
            
            return mentioned_npcs
    except Exception as e:
        logger.error(f"Error extracting NPCs from narrative: {e}", exc_info=True)
        return []

@function_tool
async def create_conflict_record(ctx: RunContextWrapper, conflict_data: Dict[str, Any], current_day: int) -> int:
    """
    Create a conflict record in the database.
    
    Args:
        conflict_data: Dictionary with conflict details
        current_day: Current in-game day
        
    Returns:
        ID of the created conflict
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Base success rate on conflict type
            success_rate = {
                "minor": 0.75,
                "standard": 0.5,
                "major": 0.25,
                "catastrophic": 0.1
            }.get(conflict_data.get("conflict_type", "standard"), 0.5)
            
            conflict_id = await conn.fetchval("""
                INSERT INTO Conflicts 
                (user_id, conversation_id, conflict_name, conflict_type,
                 description, progress, phase, start_day, estimated_duration,
                 success_rate, outcome, is_active)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING conflict_id
            """, 
            context.user_id, context.conversation_id,
            conflict_data.get("conflict_name", "Unnamed Conflict"),
            conflict_data.get("conflict_type", "standard"),
            conflict_data.get("description", "Default description"),
            0.0,  # Initial progress
            "brewing",  # Initial phase
            current_day,
            conflict_data.get("estimated_duration", 7),
            success_rate,
            "pending",  # Initial outcome
            True  # Is active
            )
            
            return conflict_id
    except Exception as e:
        logger.error(f"Error creating conflict record: {e}", exc_info=True)
        raise

@function_tool
async def create_stakeholders(
    ctx: RunContextWrapper,
    conflict_id: int,
    conflict_data: Dict[str, Any],
    stakeholder_npcs: List[Dict[str, Any]]
) -> None:
    """
    Create stakeholders for a conflict.
    
    Args:
        conflict_id: ID of the conflict
        conflict_data: Dictionary with conflict details
        stakeholder_npcs: List of NPC dictionaries to use as stakeholders
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Get stakeholders from conflict data
            stakeholders = conflict_data.get("stakeholders", [])
            
            # If no stakeholders in data, create from NPCs
            if not stakeholders:
                stakeholders = []
                for npc in stakeholder_npcs:
                    stakeholder = {
                        "npc_id": npc["npc_id"],
                        "public_motivation": f"{npc['npc_name']} wants to resolve the conflict peacefully.",
                        "private_motivation": f"{npc['npc_name']} actually wants to gain power through the conflict.",
                        "desired_outcome": "Control the outcome to their advantage",
                        "faction_id": npc.get("faction_affiliations", [{}])[0].get("faction_id") if npc.get("faction_affiliations") else None,
                        "faction_name": npc.get("faction_affiliations", [{}])[0].get("faction_name") if npc.get("faction_affiliations") else None,
                        "involvement_level": 7 - stakeholder_npcs.index(npc)  # Decreasing involvement
                    }
                    stakeholders.append(stakeholder)
            
            async with conn.transaction():
                # Create stakeholders in database
                for stakeholder in stakeholders:
                    npc_id = stakeholder.get("npc_id")
                    
                    # Get NPC from list
                    npc = next((n for n in stakeholder_npcs if n["npc_id"] == npc_id), None)
                    if not npc:
                        continue
                    
                    # Default faction info
                    faction_id = stakeholder.get("faction_id")
                    faction_name = stakeholder.get("faction_name")
                    
                    # If not specified, try to get from NPC
                    if not faction_id and npc.get("faction_affiliations"):
                        faction_id = npc.get("faction_affiliations", [{}])[0].get("faction_id")
                        faction_name = npc.get("faction_affiliations", [{}])[0].get("faction_name")
                    
                    # Default alliances and rivalries
                    alliances = stakeholder.get("alliances", {})
                    rivalries = stakeholder.get("rivalries", {})
                    
                    # Insert stakeholder
                    await conn.execute("""
                        INSERT INTO ConflictStakeholders
                        (conflict_id, npc_id, faction_id, faction_name, faction_position,
                         public_motivation, private_motivation, desired_outcome,
                         involvement_level, alliances, rivalries, leadership_ambition,
                         faction_standing, willing_to_betray_faction)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    """, 
                    conflict_id, npc_id, faction_id, faction_name, stakeholder.get("faction_position", "Member"),
                    stakeholder.get("public_motivation", "Resolve the conflict favorably"),
                    stakeholder.get("private_motivation", "Gain advantage from the conflict"),
                    stakeholder.get("desired_outcome", "Success for their side"),
                    stakeholder.get("involvement_level", 5),
                    json.dumps(alliances),
                    json.dumps(rivalries),
                    stakeholder.get("leadership_ambition", npc.get("dominance", 50) // 10),
                    stakeholder.get("faction_standing", 50),
                    stakeholder.get("willing_to_betray_faction", npc.get("cruelty", 20) > 60))
                    
                    # Create secrets if specified
                    if "secrets" in stakeholder:
                        for secret in stakeholder["secrets"]:
                            await conn.execute("""
                                INSERT INTO StakeholderSecrets
                                (conflict_id, npc_id, secret_id, secret_type, content,
                                 target_npc_id, is_revealed, revealed_to, is_public)
                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                            """, 
                            conflict_id, npc_id, 
                            secret.get("secret_id", f"secret_{npc_id}_{random.randint(1000, 9999)}"),
                            secret.get("secret_type", "personal"),
                            secret.get("content", "A hidden secret"),
                            secret.get("target_npc_id"),
                            False, None, False)
    except Exception as e:
        logger.error(f"Error creating stakeholders: {e}", exc_info=True)
        raise

@function_tool
async def create_resolution_paths(
    ctx: RunContextWrapper,
    conflict_id: int,
    conflict_data: Dict[str, Any]
) -> None:
    """
    Create resolution paths for a conflict.
    
    Args:
        conflict_id: ID of the conflict
        conflict_data: Dictionary with conflict details
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Get resolution paths from conflict data
            paths = conflict_data.get("resolution_paths", [])
            
            async with conn.transaction():
                # Create paths in database
                for path in paths:
                    await conn.execute("""
                        INSERT INTO ResolutionPaths
                        (conflict_id, path_id, name, description, approach_type,
                         difficulty, requirements, stakeholders_involved, key_challenges,
                         progress, is_completed)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """, 
                    conflict_id, 
                    path.get("path_id", f"path_{random.randint(1000, 9999)}"),
                    path.get("name", "Unnamed Path"),
                    path.get("description", "A path to resolve the conflict"),
                    path.get("approach_type", "standard"),
                    path.get("difficulty", 5),
                    json.dumps(path.get("requirements", {})),
                    json.dumps(path.get("stakeholders_involved", [])),
                    json.dumps(path.get("key_challenges", [])),
                    0.0,  # Initial progress
                    False  # Not completed
                    )
    except Exception as e:
        logger.error(f"Error creating resolution paths: {e}", exc_info=True)
        raise

@function_tool
async def create_internal_faction_conflicts(
    ctx: RunContextWrapper,
    conflict_id: int,
    conflict_data: Dict[str, Any]
) -> None:
    """
    Create internal faction conflicts for a main conflict.
    
    Args:
        conflict_id: ID of the main conflict
        conflict_data: Dictionary with conflict details
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Get internal faction conflicts from conflict data
            internal_conflicts = conflict_data.get("internal_faction_conflicts", [])
            
            async with conn.transaction():
                # Create internal conflicts in database
                for internal in internal_conflicts:
                    struggle_id = await conn.fetchval("""
                        INSERT INTO InternalFactionConflicts
                        (faction_id, conflict_name, description, primary_npc_id, target_npc_id,
                         prize, approach, public_knowledge, current_phase, progress, parent_conflict_id)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        RETURNING struggle_id
                    """, 
                    internal.get("faction_id", 0),
                    internal.get("conflict_name", "Internal Faction Struggle"),
                    internal.get("description", "A power struggle within the faction"),
                    internal.get("primary_npc_id", 0),
                    internal.get("target_npc_id", 0),
                    internal.get("prize", "Leadership"),
                    internal.get("approach", "subtle"),
                    internal.get("public_knowledge", False),
                    "brewing",  # Initial phase
                    10,  # Initial progress
                    conflict_id)
                    
                    # Create faction members if specified
                    if "faction_members" in internal:
                        for member in internal["faction_members"]:
                            await conn.execute("""
                                INSERT INTO FactionStruggleMembers
                                (struggle_id, npc_id, position, side, standing, 
                                 loyalty_strength, reason)
                                VALUES ($1, $2, $3, $4, $5, $6, $7)
                            """, 
                            struggle_id,
                            member.get("npc_id", 0),
                            member.get("position", "Member"),
                            member.get("side", "neutral"),
                            member.get("standing", 50),
                            member.get("loyalty_strength", 50),
                            member.get("reason", ""))
                    
                    # Create ideological differences if specified
                    if "ideological_differences" in internal:
                        for diff in internal["ideological_differences"]:
                            await conn.execute("""
                                INSERT INTO FactionIdeologicalDifferences
                                (struggle_id, issue, incumbent_position, challenger_position)
                                VALUES ($1, $2, $3, $4)
                            """, 
                            struggle_id,
                            diff.get("issue", ""),
                            diff.get("incumbent_position", ""),
                            diff.get("challenger_position", ""))
    except Exception as e:
        logger.error(f"Error creating internal faction conflicts: {e}", exc_info=True)
        raise

@function_tool
async def generate_player_manipulation_attempts(
    ctx: RunContextWrapper,
    conflict_id: int,
    stakeholder_npcs: List[Dict[str, Any]]
) -> None:
    """
    Generate manipulation attempts targeted at the player.
    
    Args:
        conflict_id: ID of the conflict
        stakeholder_npcs: List of NPC dictionaries
    """
    context = ctx.context
    
    # Find eligible NPCs for manipulation attempts
    eligible_npcs = []
    for npc in stakeholder_npcs:
        # Female NPCs with high dominance or close relationship with player
        if (npc.get("sex", "female") == "female" and 
            (npc.get("dominance", 0) > 70 or 
             npc.get("relationship_with_player", {}).get("closeness", 0) > 70)):
            eligible_npcs.append(npc)
    
    if not eligible_npcs:
        return
    
    # Generate manipulation attempts
    manipulation_types = ["domination", "blackmail", "seduction", "coercion", "bribery"]
    involvement_levels = ["observing", "participating", "leading"]
    factions = ["a", "b", "neutral"]
    
    for npc in eligible_npcs[:2]:  # Limit to top 2 eligible NPCs
        # Skip if random check fails (not all NPCs will attempt manipulation)
        if random.random() > 0.7:
            continue
        
        # Select manipulation type based on NPC traits
        if npc.get("dominance", 0) > 80:
            manipulation_type = "domination"
        elif npc.get("cruelty", 0) > 70:
            manipulation_type = "blackmail"
        elif npc.get("relationship_with_player", {}).get("closeness", 0) > 80:
            manipulation_type = "seduction"
        else:
            manipulation_type = random.choice(manipulation_types)
        
        # Select involvement level and faction based on NPC relationship
        involvement_level = random.choice(involvement_levels)
        faction = random.choice(factions)
        
        # Generate content based on manipulation type
        if manipulation_type == "domination":
            content = generate_domination_content(npc, npc.get("relationship_with_player", {}), 
                                                {"faction": faction, "involvement_level": involvement_level}, 
                                                {})
        elif manipulation_type == "seduction":
            content = generate_seduction_content(npc, npc.get("relationship_with_player", {}),
                                              {"faction": faction, "involvement_level": involvement_level},
                                              {})
        elif manipulation_type == "blackmail":
            content = generate_blackmail_content(npc, npc.get("relationship_with_player", {}),
                                              {"faction": faction, "involvement_level": involvement_level},
                                              {}, [])
        else:
            content = generate_generic_manipulation_content(npc, npc.get("relationship_with_player", {}),
                                                        {"faction": faction, "involvement_level": involvement_level},
                                                        {})
        
        # Generate goal
        goal = {
            "faction": faction,
            "involvement_level": involvement_level,
            "specific_actions": random.choice([
                "Spy on rival faction",
                "Convince another NPC to join their side",
                "Sabotage rival faction's plans",
                "Gather information about a specific stakeholder"
            ])
        }
        
        # Generate leverage
        leverage = generate_leverage(npc, npc.get("relationship_with_player", {}), manipulation_type)
        
        # Determine intimacy level
        intimacy_level = calculate_intimacy_level(npc, npc.get("relationship_with_player", {}), manipulation_type)
        
        # Create the manipulation attempt
        await create_manipulation_attempt(
            ctx, conflict_id, npc["npc_id"], manipulation_type,
            content, goal, leverage, intimacy_level
        )

def generate_domination_content(
    npc: Dict[str, Any],
    relationship: Dict[str, Any],
    goal: Dict[str, Any],
    conflict: Dict[str, Any]
) -> str:
    """Generate domination-based manipulation content."""
    npc_name = npc.get("npc_name", "")
    dominance = npc.get("dominance", 50)
    
    faction = goal.get("faction", "neutral")
    involvement = goal.get("involvement_level", "observing")
    
    # Get faction name if available
    faction_a_name = "first faction"
    faction_b_name = "second faction"
    
    if conflict:
        if "faction_a_name" in conflict:
            faction_a_name = conflict["faction_a_name"]
        if "faction_b_name" in conflict:
            faction_b_name = conflict["faction_b_name"]
    
    faction_name = faction_a_name if faction == "a" else faction_b_name if faction == "b" else "neutral party"
    
    # Templates based on dominance level
    if dominance > 80:  # Very dominant
        templates = [
            f"'{goal.get('specific_actions', 'You will do as I say')},' {npc_name} commands, her voice leaving no room for argument. 'Support {faction_name} by {involvement}. I will not tolerate disobedience.'",
            f"{npc_name} steps closer, towering over you despite her actual height. 'This isn't a request. You will help {faction_name} by {involvement}, or face consequences that you cannot imagine.'",
            f"'Look at me,' {npc_name} says firmly, tilting your chin up with one finger. 'You have no choice in this matter. You will support {faction_name} by {involvement}, and you will do it well. Is that understood?'"
        ]
    elif dominance > 60:  # Moderately dominant
        templates = [
            f"{npc_name} fixes you with a stern gaze. 'I expect you to support {faction_name} by {involvement}. It would be... unwise to disappoint me in this matter.'",
            f"'You will help {faction_name} by {involvement},' {npc_name} states with quiet authority. 'I know you understand the consequences of refusing me.'",
            f"{npc_name} smiles, but it doesn't reach her eyes. 'I need your help with {faction_name}, specifically by {involvement}. And I always get what I need from you, don't I?'"
        ]
    else:  # Mild dominance
        templates = [
            f"'I think it would be best if you supported {faction_name} by {involvement},' {npc_name} suggests with a hint of steel in her voice. 'Don't you agree?'",
            f"{npc_name} places a hand on your shoulder, subtly applying pressure. 'I'm counting on you to help {faction_name} by {involvement}. You wouldn't want to let me down.'",
            f"'You've always been so good at following directions,' {npc_name} says with a meaningful look. 'So you'll support {faction_name} by {involvement}, won't you?'"
        ]
    
    return random.choice(templates)

def generate_seduction_content(
    npc: Dict[str, Any],
    relationship: Dict[str, Any],
    goal: Dict[str, Any],
    conflict: Dict[str, Any]
) -> str:
    """Generate seduction-based manipulation content."""
    npc_name = npc.get("npc_name", "")
    closeness = relationship.get("closeness", 30)
    
    faction = goal.get("faction", "neutral")
    involvement = goal.get("involvement_level", "observing")
    
    # Get faction name if available
    faction_a_name = "first faction"
    faction_b_name = "second faction"
    
    if conflict:
        if "faction_a_name" in conflict:
            faction_a_name = conflict["faction_a_name"]
        if "faction_b_name" in conflict:
            faction_b_name = conflict["faction_b_name"]
    
    faction_name = faction_a_name if faction == "a" else faction_b_name if faction == "b" else "neutral party"
    
    # Templates based on closeness
    if closeness > 80:  # Very close
        templates = [
            f"{npc_name} trails her fingers down your cheek, her touch lingering. 'You know how much it would please me if you helped {faction_name} by {involvement},' she whispers. 'And I can be very... grateful when I'm pleased.'",
            f"'We have something special, don't we?' {npc_name} asks, pressing her body against yours. 'So of course you'll support {faction_name} by {involvement}. For me.' Her lips brush your ear as she says it.",
            f"{npc_name} takes your hand, guiding it to rest on her waist. 'Help {faction_name} by {involvement}, and I promise to make it worth every moment of your time,' she purrs, her meaning unmistakable."
        ]
    elif closeness > 60:  # Moderately close
        templates = [
            f"'I've been thinking about us,' {npc_name} says with a suggestive smile. 'About how things could... develop between us if you were to help {faction_name} by {involvement}.'",
            f"{npc_name} moves closer than strictly necessary, her perfume enveloping you. 'Support {faction_name} by {involvement}, and I'll show you just how appreciative I can be.'",
            f"'We could have a very special arrangement,' {npc_name} suggests, touching your arm lightly. 'You help {faction_name} by {involvement}, and I...' She leaves the rest unsaid, but her meaning is clear."
        ]
    else:  # Beginning closeness
        templates = [
            f"{npc_name} catches your eye, holding your gaze a moment longer than necessary. 'I find myself drawn to people who support {faction_name},' she says. 'Especially those who {involvement}.'",
            f"'I've noticed you,' {npc_name} admits with a shy smile that doesn't quite match her calculating eyes. 'And I could notice you even more if you were to help {faction_name} by {involvement}.'",
            f"{npc_name} leans in, her voice dropping to an intimate whisper. 'Between us, I think we could have something special if you were to support {faction_name} by {involvement}. Don't you think so?'"
        ]
    
    return random.choice(templates)

def generate_blackmail_content(
    npc: Dict[str, Any],
    relationship: Dict[str, Any],
    goal: Dict[str, Any],
    conflict: Dict[str, Any],
    leverage: List[Dict[str, Any]]
) -> str:
    """Generate blackmail-based manipulation content."""
    npc_name = npc.get("npc_name", "")
    cruelty = npc.get("cruelty", 30)
    
    faction = goal.get("faction", "neutral")
    involvement = goal.get("involvement_level", "observing")
    
    # Get faction name if available
    faction_a_name = "first faction"
    faction_b_name = "second faction"
    
    if conflict:
        if "faction_a_name" in conflict:
            faction_a_name = conflict["faction_a_name"]
        if "faction_b_name" in conflict:
            faction_b_name = conflict["faction_b_name"]
    
    faction_name = faction_a_name if faction == "a" else faction_b_name if faction == "b" else "neutral party"
    
    # Get leverage detail if available
    leverage_detail = "certain information"
    if leverage and len(leverage) > 0:
        leverage_detail = leverage[0].get("description", "certain information")
    
    # Templates based on cruelty
    if cruelty > 70:  # Very cruel
        templates = [
            f"{npc_name} smiles coldly. 'I know about {leverage_detail}. Help {faction_name} by {involvement}, or everyone else will know too. It's a simple choice, really.'",
            f"'Let me be clear,' {npc_name} says, her voice like ice. 'Either you support {faction_name} by {involvement}, or {leverage_detail} becomes public knowledge. What will it be?'",
            f"{npc_name} slides a folder across the table to you. Inside is proof of {leverage_detail}. 'Support {faction_name} by {involvement}, or this goes out to everyone who matters to you. Your choice.'"
        ]
    elif cruelty > 50:  # Moderately cruel
        templates = [
            f"'It would be a shame if people learned about {leverage_detail},' {npc_name} says with feigned concern. 'Fortunately, you can ensure my silence by helping {faction_name} with {involvement}.'",
            f"{npc_name} raises an eyebrow. 'We all have secrets, don't we? Yours involve {leverage_detail}. Mine... well, mine could involve keeping that quiet if you support {faction_name} by {involvement}.'",
            f"'I consider myself discreet,' {npc_name} says, studying her nails. 'Information about {leverage_detail} would never come from me... as long as you help {faction_name} by {involvement}, of course.'"
        ]
    else:  # Mildly cruel
        templates = [
            f"{npc_name} looks genuinely uncomfortable. 'I don't like doing this, but I need your help. I know about {leverage_detail}, and I'll use it if I have to. Please support {faction_name} by {involvement}.'",
            f"'This isn't how I wanted to ask,' {npc_name} says with a sigh, 'but I'm desperate. Help {faction_name} by {involvement}, or I'll have to tell people about {leverage_detail}.'",
            f"{npc_name} winces slightly. 'I hate to bring this up, but... {leverage_detail}. I need you to support {faction_name} by {involvement}, and we can both forget I ever mentioned it.'"
        ]
    
    return random.choice(templates)

def generate_generic_manipulation_content(
    npc: Dict[str, Any],
    relationship: Dict[str, Any],
    goal: Dict[str, Any],
    conflict: Dict[str, Any]
) -> str:
    """Generate generic manipulation content."""
    npc_name = npc.get("npc_name", "")
    
    faction = goal.get("faction", "neutral")
    involvement = goal.get("involvement_level", "observing")
    
    # Get faction name if available
    faction_a_name = "first faction"
    faction_b_name = "second faction"
    
    if conflict:
        if "faction_a_name" in conflict:
            faction_a_name = conflict["faction_a_name"]
        if "faction_b_name" in conflict:
            faction_b_name = conflict["faction_b_name"]
    
    faction_name = faction_a_name if faction == "a" else faction_b_name if faction == "b" else "neutral party"
    
    # Generic templates
    templates = [
        f"{npc_name} makes a compelling case for why you should support {faction_name} by {involvement}, appealing to your sense of reason.",
        f"'I need your help,' {npc_name} says earnestly. 'Please support {faction_name} by {involvement}. It would mean a lot to me.'",
        f"{npc_name} outlines the benefits you would receive if you were to help {faction_name} by {involvement}. The offer is tempting."
    ]
    
    return random.choice(templates)

def generate_leverage(
    npc: Dict[str, Any],
    relationship: Dict[str, Any],
    manipulation_type: str
) -> Dict[str, Any]:
    """Generate appropriate leverage based on manipulation type."""
    if manipulation_type == "domination":
        return {
            "type": "dominance",
            "description": "Authority and intimidation",
            "strength": npc.get("dominance", 50)
        }
    elif manipulation_type == "seduction":
        return {
            "type": "desire",
            "description": "Romantic or sexual interest",
            "strength": relationship.get("closeness", 30)
        }
    elif manipulation_type == "blackmail":
        return {
            "type": "information",
            "description": "Compromising information",
            "strength": npc.get("cruelty", 30)
        }
    else:
        return {
            "type": "persuasion",
            "description": "Logical argument",
            "strength": 50
        }

def calculate_intimacy_level(
    npc: Dict[str, Any],
    relationship: Dict[str, Any],
    manipulation_type: str
) -> int:
    """Calculate intimacy level (0-10) based on relationship and manipulation type."""
    base_intimacy = relationship.get("closeness", 0) // 10
    
    if manipulation_type == "seduction":
        # Seduction is more intimate
        return min(10, base_intimacy + 3)
    elif manipulation_type == "domination":
        # Domination can be intimate but depends on relationship
        dominance_factor = npc.get("dominance", 0) // 20
        return min(10, base_intimacy + dominance_factor)
    elif manipulation_type == "blackmail":
        # Blackmail is less intimate
        return max(0, base_intimacy - 2)
    else:
        # Generic manipulation is neutral
        return base_intimacy

def generate_conflict_consequences(
    conflict_type: str,
    outcome: str,
    player_involvement: str,
    player_faction: str,
    completed_paths: List
) -> List[Dict[str, Any]]:
    """
    Generate consequences for a resolved conflict.
    
    Args:
        conflict_type: Type of conflict (minor, standard, major, catastrophic)
        outcome: Outcome of the conflict (resolved, unresolved)
        player_involvement: Player's involvement level
        player_faction: Player's chosen faction
        completed_paths: List of completed resolution paths
        
    Returns:
        List of consequence dictionaries
    """
    consequences = []
    
    # Base impact by conflict type
    impact_level = {
        "minor": 1,
        "standard": 2,
        "major": 3,
        "catastrophic": 4
    }.get(conflict_type, 2)
    
    # Impact multiplier based on player involvement
    involvement_multiplier = {
        "none": 0.5,
        "observing": 1.0,
        "participating": 1.5,
        "leading": 2.0
    }.get(player_involvement, 1.0)
    
    # Player stat changes based on outcome and involvement
    if outcome == "resolved":
        # Successful resolution
        if player_involvement in ["participating", "leading"]:
            if player_faction in ["a", "b"]:  # Player took a side
                consequences.append({
                    "type": "player_stat",
                    "description": f"Your influence with {player_faction} faction has increased.",
                    "affected_entity_type": "player",
                    "magnitude": impact_level * involvement_multiplier,
                    "is_permanent": True,
                    "stat_changes": {
                        "confidence": int(3 * involvement_multiplier),
                        "mental_resilience": int(2 * involvement_multiplier)
                    }
                })
            else:  # Player remained neutral
                consequences.append({
                    "type": "player_stat",
                    "description": "Your reputation for neutrality has been reinforced.",
                    "affected_entity_type": "player",
                    "magnitude": impact_level,
                    "is_permanent": True,
                    "stat_changes": {
                        "willpower": 2,
                        "confidence": 1
                    }
                })
            
            # ADD REWARD ITEMS AND PERKS BASED ON RESOLUTION STYLE
            # Determine resolution style from completed paths
            resolution_styles = []
            for path in completed_paths:
                path_name = path.get("name", "").lower()
                if "violence" in path_name or "force" in path_name:
                    resolution_styles.append("forceful")
                elif "negotiation" in path_name or "diplomacy" in path_name:
                    resolution_styles.append("diplomatic")
                elif "manipulation" in path_name or "deception" in path_name:
                    resolution_styles.append("manipulative")
                elif "submission" in path_name or "obedience" in path_name:
                    resolution_styles.append("submissive")
                else:
                    resolution_styles.append("neutral")
            
            # Get primary resolution style
            primary_style = max(set(resolution_styles), key=resolution_styles.count) if resolution_styles else "neutral"
            
            # Generate rewards based on resolution style, conflict type, and player involvement
            rewards = []
            
            # Always give at least one item
            item_reward = generate_item_reward(primary_style, conflict_type, impact_level)
            rewards.append(item_reward)
            
            # For more significant conflicts or higher involvement, add perks
            if impact_level >= 2 or player_involvement == "leading":
                perk_reward = generate_perk_reward(primary_style, conflict_type, impact_level)
                rewards.append(perk_reward)
            
            # For major/catastrophic conflicts, add a special reward
            if impact_level >= 3:
                special_reward = generate_special_reward(primary_style, conflict_type, impact_level)
                rewards.append(special_reward)
            
            # Add rewards to consequences
            for reward in rewards:
                consequences.append(reward)
    else:
        # Unresolved conflict
        if player_involvement in ["participating", "leading"]:
            consequences.append({
                "type": "player_stat",
                "description": "The unresolved conflict has left a mark on you.",
                "affected_entity_type": "player",
                "magnitude": impact_level,
                "is_permanent": True,
                "stat_changes": {
                    "mental_resilience": 2,
                    "confidence": -1
                }
            })
    
    # World/NPC consequences
    world_consequences = [
        {
            "type": "world_change",
            "description": f"The resolution has shifted the balance of power in the region.",
            "affected_entity_type": "world",
            "magnitude": impact_level,
            "is_permanent": True
        },
        {
            "type": "npc_relationship",
            "description": f"NPCs involved in the conflict have changed their opinion of you.",
            "affected_entity_type": "npcs",
            "magnitude": impact_level,
            "is_permanent": False
        }
    ]
    
    consequences.extend(world_consequences)
    
    return consequences

def generate_item_reward(resolution_style: str, conflict_type: str, impact_level: int) -> Dict[str, Any]:
    """Generate an item reward based on the resolution style and conflict type."""
    items = {
        "forceful": [
            {"name": "Leather Restraints", "description": "High-quality restraints that show dominance.", "rarity": "common"},
            {"name": "Force Gauntlets", "description": "Gloves that enhance the wearer's grip strength.", "rarity": "uncommon"},
            {"name": "Intimidation Collar", "description": "A collar that causes unease in those who see it.", "rarity": "rare"},
            {"name": "Dominance Sigil", "description": "A symbol of authority that commands respect.", "rarity": "very rare"}
        ],
        "diplomatic": [
            {"name": "Negotiator's Pin", "description": "A pin that subtly influences conversations.", "rarity": "common"},
            {"name": "Treaty Document", "description": "A blank document that confers legitimacy to agreements.", "rarity": "uncommon"},
            {"name": "Silver Tongue Amulet", "description": "Enhances persuasiveness in formal settings.", "rarity": "rare"},
            {"name": "Oath Binding Ring", "description": "A ring that helps ensure promises are kept.", "rarity": "very rare"}
        ],
        "manipulative": [
            {"name": "Gossip Brooch", "description": "Helps spread rumors more effectively.", "rarity": "common"},
            {"name": "Secret Keeper's Locket", "description": "Stores overheard secrets for later use.", "rarity": "uncommon"},
            {"name": "Mask of Facades", "description": "Helps the wearer present a false persona.", "rarity": "rare"},
            {"name": "Heart's Desire Mirror", "description": "Reveals what others want most.", "rarity": "very rare"}
        ],
        "submissive": [
            {"name": "Obedience Bracelet", "description": "A token that signifies willing submission.", "rarity": "common"},
            {"name": "Loyalty Mark", "description": "A temporary mark that shows allegiance.", "rarity": "uncommon"},
            {"name": "Devoted Servant's Pendant", "description": "Enhances the wearer's ability to please others.", "rarity": "rare"},
            {"name": "Will Binding Circlet", "description": "Strengthens bonds of loyalty between wearer and owner.", "rarity": "very rare"}
        ],
        "neutral": [
            {"name": "Balance Stone", "description": "A stone that brings clarity of purpose.", "rarity": "common"},
            {"name": "Observer's Monocle", "description": "Helps see situations from multiple perspectives.", "rarity": "uncommon"},
            {"name": "Impartial Judge's Scale", "description": "A small scale that helps make fair decisions.", "rarity": "rare"},
            {"name": "Harmony Crystal", "description": "Creates an atmosphere of cooperation.", "rarity": "very rare"}
        ]
    }
    
    # Select item based on impact level (higher impact = better items)
    item_index = min(impact_level - 1, 3)  # 0-3 index for the items
    item_list = items.get(resolution_style, items["neutral"])
    selected_item = item_list[item_index]
    
    return {
        "type": "item_reward",
        "description": f"You received {selected_item['name']} as a reward: {selected_item['description']}",
        "affected_entity_type": "player",
        "magnitude": impact_level,
        "is_permanent": True,
        "item": {
            "name": selected_item["name"],
            "description": selected_item["description"],
            "rarity": selected_item["rarity"],
            "category": "conflict_reward",
            "resolution_style": resolution_style
        }
    }

def generate_perk_reward(resolution_style: str, conflict_type: str, impact_level: int) -> Dict[str, Any]:
    """Generate a perk reward based on the resolution style and conflict type."""
    perks = {
        "forceful": [
            {"name": "Intimidating Presence", "description": "Your forceful approach causes weaker NPCs to back down more easily.", "tier": 1},
            {"name": "Show of Strength", "description": "You can demonstrate your physical dominance to gain advantage in confrontations.", "tier": 2},
            {"name": "Command Respect", "description": "Your history of forceful resolution makes dominant NPCs recognize your authority.", "tier": 3}
        ],
        "diplomatic": [
            {"name": "Diplomatic Immunity", "description": "Minor social faux pas are more likely to be overlooked.", "tier": 1},
            {"name": "Peace Broker", "description": "You have a reputation for fair solutions, giving you more options in negotiations.", "tier": 2},
            {"name": "Alliance Network", "description": "Your diplomatic approach has created a network of allies willing to provide information.", "tier": 3}
        ],
        "manipulative": [
            {"name": "Subtle Influence", "description": "You can plant suggestions more effectively in casual conversation.", "tier": 1},
            {"name": "Blackmail Expert", "description": "You're better at identifying and leveraging others' secrets.", "tier": 2},
            {"name": "Puppet Master", "description": "Your reputation for manipulation precedes you, making manipulative tactics more effective.", "tier": 3}
        ],
        "submissive": [
            {"name": "Willing Servant", "description": "Dominant NPCs are more likely to protect you in exchange for loyalty.", "tier": 1},
            {"name": "Trusted Confidant", "description": "Your submissive nature makes others more likely to share secrets with you.", "tier": 2},
            {"name": "Perfect Obedience", "description": "Your reputation for submission gives you special privileges with powerful NPCs.", "tier": 3}
        ],
        "neutral": [
            {"name": "Respected Neutral", "description": "Factions are more likely to accept your presence in their territory.", "tier": 1},
            {"name": "Balanced Perspective", "description": "You gain additional insight when observing conflicts from the outside.", "tier": 2},
            {"name": "Impartial Arbiter", "description": "You're sought out to mediate disputes, giving you access to valuable information.", "tier": 3}
        ]
    }
    
    # Select perk based on impact level (higher impact = better perks)
    perk_index = min(impact_level - 1, 2)  # 0-2 index for the perks
    perk_list = perks.get(resolution_style, perks["neutral"])
    selected_perk = perk_list[perk_index]
    
    return {
        "type": "perk_reward",
        "description": f"You gained the '{selected_perk['name']}' perk: {selected_perk['description']}",
        "affected_entity_type": "player",
        "magnitude": impact_level,
        "is_permanent": True,
        "perk": {
            "name": selected_perk["name"],
            "description": selected_perk["description"],
            "tier": selected_perk["tier"],
            "category": "conflict_resolution",
            "resolution_style": resolution_style
        }
    }

def generate_special_reward(resolution_style: str, conflict_type: str, impact_level: int) -> Dict[str, Any]:
    """Generate a special reward for major/catastrophic conflicts."""
    specials = {
        "forceful": {
            "name": "Conquering Trophy",
            "description": "A unique trophy from your forceful resolution that grants authority in related matters.",
            "effect": "Can be presented to demonstrate your power in future conflicts of similar type."
        },
        "diplomatic": {
            "name": "Alliance Charter",
            "description": "A formal documentation of your diplomatic achievement that opens doors.",
            "effect": "Grants access to restricted areas and information related to the resolved conflict."
        },
        "manipulative": {
            "name": "Shadow Network",
            "description": "A network of informants gained through your manipulative tactics.",
            "effect": "Provides periodic information and rumors about hidden activities."
        },
        "submissive": {
            "name": "Patron's Favor",
            "description": "A token of appreciation from a powerful figure impressed by your submission.",
            "effect": "Can be exchanged for a significant favor from your patron."
        },
        "neutral": {
            "name": "Balance Keeper's Token",
            "description": "A symbol of your role in maintaining equilibrium in a major conflict.",
            "effect": "Allows you to call for temporary truces in heated situations."
        }
    }
    
    selected_special = specials.get(resolution_style, specials["neutral"])
    
    return {
        "type": "special_reward",
        "description": f"You earned a special reward: {selected_special['name']} - {selected_special['description']}",
        "affected_entity_type": "player",
        "magnitude": impact_level,
        "is_permanent": True,
        "special_reward": {
            "name": selected_special["name"],
            "description": selected_special["description"],
            "effect": selected_special["effect"],
            "category": "unique_conflict_reward",
            "resolution_style": resolution_style
        }
    }

# Governance System Integration (Added based on review)

@function_tool
async def register_with_governance(ctx: RunContextWrapper, user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Register conflict system with governance system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Registration result
    """
    context = ctx.context
    
    try:
        # Get central governance system
        governance = await get_central_governance(user_id, conversation_id)
        
        # Register as a conflict analyst agent
        registration_result = await governance.register_agent(
            agent_type=AgentType.CONFLICT_ANALYST, 
            agent_instance="conflict_manager", 
            agent_id="conflict_manager"
        )
        
        # Issue directive for conflict analysis
        directive_result = await governance.issue_directive(
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id="conflict_manager",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Manage conflicts and their progression in the game world",
                "scope": "game"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24*60  # 24 hours
        )
        
        logger.info("Conflict System registered with Nyx governance")
        
        return {
            "success": True,
            "registration_result": registration_result,
            "directive_result": directive_result,
            "message": "Conflict System successfully registered with governance"
        }
    except Exception as e:
        logger.error(f"Error registering with governance: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to register Conflict System with governance"
        }

@function_tool
async def calculate_coup_success_chance(
    ctx: RunContextWrapper,
    struggle_id: int,
    approach: str,
    supporting_npcs: List[int],
    resources_committed: Dict[str, int]
) -> float:
    """
    Calculate the success chance of a coup attempt.
    
    Args:
        struggle_id: ID of the internal faction struggle
        approach: The approach used
        supporting_npcs: List of supporting NPC IDs
        resources_committed: Resources committed
        
    Returns:
        Success chance (0-100)
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Get struggle details
            struggle_row = await conn.fetchrow("""
                SELECT primary_npc_id, target_npc_id
                FROM InternalFactionConflicts
                WHERE struggle_id = $1
            """, struggle_id)
            
            if not struggle_row:
                return 0
            
            primary_npc_id = struggle_row["primary_npc_id"]
            target_npc_id = struggle_row["target_npc_id"]
            
            # Get challenger and target stats
            challenger = await get_npc_details(ctx, primary_npc_id)
            target = await get_npc_details(ctx, target_npc_id)
            
            # Base success chance based on challenger vs target
            base_chance = 50 + (challenger.get("dominance", 50) - target.get("dominance", 50)) / 5
            
            # Adjust based on approach
            approach_modifiers = {
                "direct": 0,       # Neutral modifier
                "subtle": 10,      # Subtle approaches have advantage
                "force": -5,       # Force is risky
                "blackmail": 15    # Blackmail has high success chance
            }
            base_chance += approach_modifiers.get(approach, 0)
            
            # Adjust for supporting NPCs
            support_power = 0
            for npc_id in supporting_npcs:
                npc = await get_npc_details(ctx, npc_id)
                support_power += npc.get("dominance", 50) / 10
            
            base_chance += min(25, support_power)  # Cap at +25
            
            # Adjust for resources committed
            resource_total = sum(resources_committed.values())
            resource_modifier = min(15, resource_total / 10)  # Cap at +15
            base_chance += resource_modifier
            
            # Get faction members and their loyalty to incumbent
            faction_members = await conn.fetch("""
                SELECT npc_id, loyalty_strength
                FROM FactionStruggleMembers
                WHERE struggle_id = $1 AND side = 'incumbent'
            """, struggle_id)
            
            total_loyalty = 0
            for row in faction_members:
                total_loyalty += row["loyalty_strength"]
            
            # Loyalty to incumbent reduces success chance
            loyalty_modifier = min(30, total_loyalty / 20)  # Cap at -30
            base_chance -= loyalty_modifier
            
            # Ensure chance is between 5 and 95
            return max(5, min(95, base_chance))
    except Exception as e:
        logger.error(f"Error calculating coup success chance: {e}", exc_info=True)
        return 30  # Default moderate chance on error

@function_tool
async def add_resolution_path(
    ctx: RunContextWrapper,
    conflict_id: int,
    path_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Add a new resolution path to an existing conflict.
    
    Args:
        ctx: RunContextWrapper with user context
        conflict_id: ID of the conflict
        path_data: Dictionary with path details
        
    Returns:
        Dictionary with the created resolution path
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Check if conflict exists
            exists = await conn.fetchval("""
                SELECT 1 FROM Conflicts
                WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
            """, conflict_id, context.user_id, context.conversation_id)
            
            if not exists:
                return {
                    "success": False,
                    "error": f"Conflict with ID {conflict_id} not found"
                }
            
            # Generate path_id if not provided
            path_id = path_data.get("path_id", f"path_{random.randint(1000, 9999)}")
            
            # Insert the new path
            await conn.execute("""
                INSERT INTO ResolutionPaths
                (conflict_id, path_id, name, description, approach_type,
                 difficulty, requirements, stakeholders_involved, key_challenges,
                 progress, is_completed)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, 
            conflict_id, 
            path_id,
            path_data.get("name", "Unnamed Path"),
            path_data.get("description", "A path to resolve the conflict"),
            path_data.get("approach_type", "standard"),
            path_data.get("difficulty", 5),
            json.dumps(path_data.get("requirements", {})),
            json.dumps(path_data.get("stakeholders_involved", [])),
            json.dumps(path_data.get("key_challenges", [])),
            0.0,  # Initial progress
            False  # Not completed
            )
            
            # Create a memory for this new path
            await create_conflict_memory(
                ctx,
                conflict_id,
                f"A new resolution path '{path_data.get('name', 'Unnamed Path')}' has been added to the conflict.",
                significance=6
            )
            
            return {
                "success": True,
                "conflict_id": conflict_id,
                "path_id": path_id,
                "name": path_data.get("name", "Unnamed Path"),
                "description": path_data.get("description", "A path to resolve the conflict")
            }
    except Exception as e:
        logger.error(f"Error adding resolution path to conflict {conflict_id}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

@function_tool
async def update_player_involvement(
    ctx: RunContextWrapper,
    conflict_id: int,
    involvement_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update player's involvement in a conflict.
    
    Args:
        ctx: RunContextWrapper with user context
        conflict_id: ID of the conflict
        involvement_data: Dictionary with involvement details
        
    Returns:
        Dictionary with update result
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Check if player involvement already exists
            involvement_exists = await conn.fetchval("""
                SELECT 1 FROM PlayerConflictInvolvement
                WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
            """, conflict_id, context.user_id, context.conversation_id)
            
            # Extract values from involvement_data
            involvement_level = involvement_data.get("involvement_level", "observing")
            faction = involvement_data.get("faction", "neutral")
            money_committed = involvement_data.get("resources_committed", {}).get("money", 0)
            supplies_committed = involvement_data.get("resources_committed", {}).get("supplies", 0)
            influence_committed = involvement_data.get("resources_committed", {}).get("influence", 0)
            actions_taken = json.dumps(involvement_data.get("actions_taken", []))
            manipulated_by = json.dumps(involvement_data.get("manipulated_by")) if involvement_data.get("manipulated_by") else None
            
            if involvement_exists:
                # Update existing involvement
                await conn.execute("""
                    UPDATE PlayerConflictInvolvement
                    SET involvement_level = $1, faction = $2,
                        money_committed = $3, supplies_committed = $4,
                        influence_committed = $5, actions_taken = $6,
                        manipulated_by = $7
                    WHERE conflict_id = $8 AND user_id = $9 AND conversation_id = $10
                """, 
                involvement_level, faction, money_committed, supplies_committed, 
                influence_committed, actions_taken, manipulated_by,
                conflict_id, context.user_id, context.conversation_id)
            else:
                # Insert new involvement
                await conn.execute("""
                    INSERT INTO PlayerConflictInvolvement
                    (conflict_id, user_id, conversation_id, player_name, involvement_level,
                     faction, money_committed, supplies_committed, influence_committed,
                     actions_taken, manipulated_by)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, 
                conflict_id, context.user_id, context.conversation_id, "Player",
                involvement_level, faction, money_committed, supplies_committed,
                influence_committed, actions_taken, manipulated_by)
            
            # Create a memory for this involvement update
            await create_conflict_memory(
                ctx,
                conflict_id,
                f"Player's involvement in the conflict has changed to {involvement_level} with {faction} faction.",
                significance=7
            )
            
            return {
                "success": True,
                "conflict_id": conflict_id,
                "involvement_level": involvement_level,
                "faction": faction,
                "resources_committed": {
                    "money": money_committed,
                    "supplies": supplies_committed,
                    "influence": influence_committed
                }
            }
    except Exception as e:
        logger.error(f"Error updating player involvement in conflict {conflict_id}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

@function_tool
async def add_internal_conflict(
    ctx: RunContextWrapper,
    conflict_id: int,
    internal_conflict_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Add an internal faction conflict to a main conflict.
    
    Args:
        ctx: RunContextWrapper with user context
        conflict_id: ID of the parent conflict
        internal_conflict_data: Dictionary with internal conflict details
        
    Returns:
        Dictionary with the created internal conflict
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Extract values from internal_conflict_data
            faction_id = internal_conflict_data.get("faction_id", 0)
            conflict_name = internal_conflict_data.get("conflict_name", "Internal Faction Struggle")
            description = internal_conflict_data.get("description", "A power struggle within the faction")
            primary_npc_id = internal_conflict_data.get("primary_npc_id", 0)
            target_npc_id = internal_conflict_data.get("target_npc_id", 0)
            prize = internal_conflict_data.get("prize", "Leadership")
            approach = internal_conflict_data.get("approach", "subtle")
            public_knowledge = internal_conflict_data.get("public_knowledge", False)
            current_phase = internal_conflict_data.get("current_phase", "brewing")
            progress = internal_conflict_data.get("progress", 10)
            
            # Create the internal conflict
            struggle_id = await conn.fetchval("""
                INSERT INTO InternalFactionConflicts
                (faction_id, conflict_name, description, primary_npc_id, target_npc_id,
                 prize, approach, public_knowledge, current_phase, progress, parent_conflict_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING struggle_id
            """, 
            faction_id, conflict_name, description, primary_npc_id, target_npc_id,
            prize, approach, public_knowledge, current_phase, progress, conflict_id)
            
            # Get faction name
            faction_name = await get_faction_name(ctx, faction_id)
            
            # Get NPC names
            primary_npc_name = await get_npc_name(ctx, primary_npc_id)
            target_npc_name = await get_npc_name(ctx, target_npc_id)
            
            # Create faction members if provided
            if "faction_members" in internal_conflict_data:
                for member in internal_conflict_data["faction_members"]:
                    await conn.execute("""
                        INSERT INTO FactionStruggleMembers
                        (struggle_id, npc_id, position, side, standing, 
                         loyalty_strength, reason)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """, 
                    struggle_id,
                    member.get("npc_id", 0),
                    member.get("position", "Member"),
                    member.get("side", "neutral"),
                    member.get("standing", 50),
                    member.get("loyalty_strength", 50),
                    member.get("reason", ""))
            
            # Create a memory for this internal conflict
            await create_conflict_memory(
                ctx,
                conflict_id,
                f"An internal power struggle has emerged in {faction_name} between {primary_npc_name} and {target_npc_name}.",
                significance=7
            )
            
            return {
                "success": True,
                "struggle_id": struggle_id,
                "conflict_id": conflict_id,
                "faction_id": faction_id,
                "faction_name": faction_name,
                "conflict_name": conflict_name,
                "primary_npc_name": primary_npc_name,
                "target_npc_name": target_npc_name
            }
    except Exception as e:
        logger.error(f"Error adding internal conflict to conflict {conflict_id}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


@function_tool
async def resolve_internal_conflict(
    ctx: RunContextWrapper,
    struggle_id: int,
    resolution_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Resolve an internal faction conflict.
    
    Args:
        ctx: RunContextWrapper with user context
        struggle_id: ID of the internal faction struggle
        resolution_data: Dictionary with resolution details
        
    Returns:
        Dictionary with resolution result
    """
    context = ctx.context
    
    try:
        async with get_db_connection_context() as conn:
            # Check if struggle exists
            struggle_row = await conn.fetchrow("""
                SELECT faction_id, conflict_name, primary_npc_id, target_npc_id, parent_conflict_id
                FROM InternalFactionConflicts
                WHERE struggle_id = $1
            """, struggle_id)
            
            if not struggle_row:
                return {
                    "success": False,
                    "error": f"Internal faction struggle with ID {struggle_id} not found"
                }
            
            faction_id = struggle_row["faction_id"]
            conflict_name = struggle_row["conflict_name"]
            primary_npc_id = struggle_row["primary_npc_id"]
            target_npc_id = struggle_row["target_npc_id"]
            parent_conflict_id = struggle_row["parent_conflict_id"]
            
            # Get faction name
            faction_name = await get_faction_name(ctx, faction_id)
            
            # Get NPC names
            primary_npc_name = await get_npc_name(ctx, primary_npc_id)
            target_npc_name = await get_npc_name(ctx, target_npc_id)
            
            # Extract resolution details
            winner_npc_id = resolution_data.get("winner_npc_id")
            resolution_type = resolution_data.get("resolution_type", "negotiated")
            resolution_description = resolution_data.get("description", f"The internal conflict in {faction_name} has been resolved.")
            
            # Determine winner and loser
            winner_npc_id = winner_npc_id or primary_npc_id
            loser_npc_id = target_npc_id if winner_npc_id == primary_npc_id else primary_npc_id
            
            winner_name = await get_npc_name(ctx, winner_npc_id)
            loser_name = await get_npc_name(ctx, loser_npc_id)
            
            # Update the struggle
            await conn.execute("""
                UPDATE InternalFactionConflicts
                SET current_phase = 'resolved', progress = 100,
                    resolution_type = $1, resolution_description = $2,
                    winner_npc_id = $3, loser_npc_id = $4,
                    resolved_at = CURRENT_TIMESTAMP
                WHERE struggle_id = $5
            """, 
            resolution_type, resolution_description, 
            winner_npc_id, loser_npc_id, struggle_id)
            
            # Create a memory for this resolution
            memory_text = f"The internal conflict in {faction_name} has been resolved. {winner_name} has emerged victorious over {loser_name}."
            
            await create_conflict_memory(
                ctx,
                parent_conflict_id,
                memory_text,
                significance=8
            )
            
            return {
                "success": True,
                "struggle_id": struggle_id,
                "faction_name": faction_name,
                "winner_name": winner_name,
                "loser_name": loser_name,
                "resolution_type": resolution_type,
                "description": resolution_description
            }
    except Exception as e:
        logger.error(f"Error resolving internal faction struggle {struggle_id}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

    
