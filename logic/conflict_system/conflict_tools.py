# logic/conflict_system/conflict_tools.py
"""
Conflict System Function Tools

This module defines the function tools used by the conflict system agents.
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field
from agents import function_tool, RunContextWrapper

from db.connection import get_db_connection
from logic.stats_logic import apply_stat_change
from logic.resource_management import ResourceManager
from logic.npc_relationship_manager import get_relationship_status, get_manipulation_leverage
from logic.chatgpt_integration import get_chatgpt_response

logger = logging.getLogger(__name__)

# Database Access Tools

@function_tool
async def get_active_conflicts(ctx: RunContextWrapper) -> List[Dict[str, Any]]:
    """
    Get all active conflicts for the current user and conversation.
    
    Returns a list of active conflict dictionaries with all related data.
    """
    context = ctx.context
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT c.conflict_id, c.conflict_name, c.conflict_type, 
                   c.description, c.progress, c.phase, c.start_day,
                   c.estimated_duration, c.success_rate, c.outcome, c.is_active
            FROM Conflicts c
            WHERE c.user_id = %s AND c.conversation_id = %s AND c.is_active = TRUE
            ORDER BY c.conflict_id DESC
        """, (context.user_id, context.conversation_id))
        
        conflicts = []
        for row in cursor.fetchall():
            conflict_id, conflict_name, conflict_type, description, progress, phase, \
            start_day, estimated_duration, success_rate, outcome, is_active = row
            
            # Build conflict dictionary
            conflict = {
                "conflict_id": conflict_id,
                "conflict_name": conflict_name,
                "conflict_type": conflict_type,
                "description": description,
                "progress": progress,
                "phase": phase,
                "start_day": start_day,
                "estimated_duration": estimated_duration,
                "success_rate": success_rate,
                "outcome": outcome,
                "is_active": is_active
            }
            
            # Get stakeholders
            cursor.execute("""
                SELECT s.npc_id, n.npc_name, s.faction_id, s.faction_name,
                       s.faction_position, s.public_motivation, s.private_motivation,
                       s.desired_outcome, s.involvement_level, s.alliances, s.rivalries
                FROM ConflictStakeholders s
                JOIN NPCStats n ON s.npc_id = n.npc_id
                WHERE s.conflict_id = %s
                ORDER BY s.involvement_level DESC
            """, (conflict_id,))
            
            stakeholders = []
            for stakeholder_row in cursor.fetchall():
                npc_id, npc_name, faction_id, faction_name, faction_position, \
                public_motivation, private_motivation, desired_outcome, \
                involvement_level, alliances, rivalries = stakeholder_row
                
                stakeholders.append({
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "faction_id": faction_id,
                    "faction_name": faction_name,
                    "faction_position": faction_position,
                    "public_motivation": public_motivation,
                    "private_motivation": private_motivation,
                    "desired_outcome": desired_outcome,
                    "involvement_level": involvement_level
                })
            
            conflict["stakeholders"] = stakeholders
            
            # Get resolution paths
            cursor.execute("""
                SELECT path_id, name, description, approach_type, difficulty,
                       requirements, stakeholders_involved, key_challenges,
                       progress, is_completed
                FROM ResolutionPaths
                WHERE conflict_id = %s
            """, (conflict_id,))
            
            paths = []
            for path_row in cursor.fetchall():
                path_id, name, description, approach_type, difficulty, \
                requirements, stakeholders_involved, key_challenges, \
                progress, is_completed = path_row
                
                paths.append({
                    "path_id": path_id,
                    "name": name,
                    "description": description,
                    "approach_type": approach_type,
                    "difficulty": difficulty,
                    "progress": progress,
                    "is_completed": is_completed
                })
            
            conflict["resolution_paths"] = paths
            
            # Get player involvement
            cursor.execute("""
                SELECT involvement_level, faction, money_committed, supplies_committed, 
                       influence_committed
                FROM PlayerConflictInvolvement
                WHERE conflict_id = %s AND user_id = %s AND conversation_id = %s
            """, (conflict_id, context.user_id, context.conversation_id))
            
            involvement_row = cursor.fetchone()
            if involvement_row:
                involvement_level, faction, money, supplies, influence = involvement_row
                conflict["player_involvement"] = {
                    "involvement_level": involvement_level,
                    "faction": faction,
                    "resources_committed": {
                        "money": money,
                        "supplies": supplies,
                        "influence": influence
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
            
            conflicts.append(conflict)
        
        return conflicts
    except Exception as e:
        logger.error(f"Error getting active conflicts: {e}", exc_info=True)
        return []
    finally:
        cursor.close()
        conn.close()

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
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT c.conflict_id, c.conflict_name, c.conflict_type, 
                   c.description, c.progress, c.phase, c.start_day,
                   c.estimated_duration, c.success_rate, c.outcome, c.is_active
            FROM Conflicts c
            WHERE c.conflict_id = %s AND c.user_id = %s AND c.conversation_id = %s
        """, (conflict_id, context.user_id, context.conversation_id))
        
        row = cursor.fetchone()
        
        if not row:
            return {"error": "Conflict not found"}
        
        conflict_id, conflict_name, conflict_type, description, progress, phase, \
        start_day, estimated_duration, success_rate, outcome, is_active = row
        
        # Build conflict dictionary
        conflict = {
            "conflict_id": conflict_id,
            "conflict_name": conflict_name,
            "conflict_type": conflict_type,
            "description": description,
            "progress": progress,
            "phase": phase,
            "start_day": start_day,
            "estimated_duration": estimated_duration,
            "success_rate": success_rate,
            "outcome": outcome,
            "is_active": is_active
        }
        
        # Get stakeholders (similar to get_active_conflicts)
        # ... (implement stakeholder retrieval)
        
        # Get resolution paths
        # ... (implement resolution paths retrieval)
        
        # Get player involvement
        # ... (implement player involvement retrieval)
        
        return conflict
    except Exception as e:
        logger.error(f"Error getting conflict details for ID {conflict_id}: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        cursor.close()
        conn.close()

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
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT s.npc_id, n.npc_name, s.faction_id, s.faction_name,
                   s.faction_position, s.public_motivation, s.private_motivation,
                   s.desired_outcome, s.involvement_level, s.alliances, s.rivalries,
                   s.leadership_ambition, s.faction_standing, s.willing_to_betray_faction
            FROM ConflictStakeholders s
            JOIN NPCStats n ON s.npc_id = n.npc_id
            WHERE s.conflict_id = %s
            ORDER BY s.involvement_level DESC
        """, (conflict_id,))
        
        stakeholders = []
        for row in cursor.fetchall():
            npc_id, npc_name, faction_id, faction_name, faction_position, public_motivation, \
            private_motivation, desired_outcome, involvement_level, alliances, rivalries, \
            leadership_ambition, faction_standing, willing_to_betray = row
            
            # Parse JSON fields
            try:
                alliances_dict = json.loads(alliances) if isinstance(alliances, str) else alliances or {}
            except (json.JSONDecodeError, TypeError):
                alliances_dict = {}
            
            try:
                rivalries_dict = json.loads(rivalries) if isinstance(rivalries, str) else rivalries or {}
            except (json.JSONDecodeError, TypeError):
                rivalries_dict = {}
            
            stakeholder = {
                "npc_id": npc_id,
                "npc_name": npc_name,
                "faction_id": faction_id,
                "faction_name": faction_name,
                "faction_position": faction_position,
                "public_motivation": public_motivation,
                "private_motivation": private_motivation,
                "desired_outcome": desired_outcome,
                "involvement_level": involvement_level,
                "alliances": alliances_dict,
                "rivalries": rivalries_dict,
                "leadership_ambition": leadership_ambition,
                "faction_standing": faction_standing,
                "willing_to_betray_faction": willing_to_betray
            }
            
            stakeholders.append(stakeholder)
        
        return stakeholders
    except Exception as e:
        logger.error(f"Error getting stakeholders for conflict {conflict_id}: {e}", exc_info=True)
        return []
    finally:
        cursor.close()
        conn.close()

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
    
    # Return the created conflict
    return await get_conflict_details(ctx, conflict_id)

@function_tool
async def get_current_day(ctx: RunContextWrapper) -> int:
    """Get the current in-game day."""
    context = ctx.context
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=%s AND conversation_id=%s AND key='CurrentDay'
        """, (context.user_id, context.conversation_id))
        
        row = cursor.fetchone()
        
        return int(row[0]) if row else 1
    except Exception as e:
        logger.error(f"Error getting current day: {e}", exc_info=True)
        return 1
    finally:
        cursor.close()
        conn.close()

@function_tool
async def get_available_npcs(ctx: RunContextWrapper) -> List[Dict[str, Any]]:
    """Get available NPCs that could be involved in conflicts."""
    context = ctx.context
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT npc_id, npc_name, dominance, cruelty, closeness, trust,
                   respect, intensity, sex, current_location, faction_affiliations
            FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND introduced=TRUE
            ORDER BY dominance DESC
        """, (context.user_id, context.conversation_id))
        
        npcs = []
        for row in cursor.fetchall():
            npc_id, npc_name, dominance, cruelty, closeness, trust, \
            respect, intensity, sex, current_location, faction_affiliations = row
            
            # Parse faction affiliations
            try:
                affiliations = json.loads(faction_affiliations) if isinstance(faction_affiliations, str) else faction_affiliations or []
            except (json.JSONDecodeError, TypeError):
                affiliations = []
            
            # Get relationships with player
            relationship = await get_npc_relationship_with_player(ctx, npc_id)
            
            npc = {
                "npc_id": npc_id,
                "npc_name": npc_name,
                "dominance": dominance,
                "cruelty": cruelty,
                "closeness": closeness,
                "trust": trust,
                "respect": respect,
                "intensity": intensity,
                "sex": sex,
                "current_location": current_location,
                "faction_affiliations": affiliations,
                "relationship_with_player": relationship
            }
            
            npcs.append(npc)
        
        return npcs
    except Exception as e:
        logger.error(f"Error getting available NPCs: {e}", exc_info=True)
        return []
    finally:
        cursor.close()
        conn.close()

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
        # Extract JSON from text response or create default structure
        # ... (implementation details)
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
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if NPC has relationship with player
        relationship = await get_npc_relationship_with_player(ctx, npc_id)
        
        # Get NPC name
        npc_name = await get_npc_name(ctx, npc_id)
        
        # Insert the manipulation attempt
        cursor.execute("""
            INSERT INTO PlayerManipulationAttempts
            (conflict_id, user_id, conversation_id, npc_id, manipulation_type, 
             content, goal, success, leverage_used, intimacy_level, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING attempt_id
        """, (
            conflict_id, context.user_id, context.conversation_id, npc_id,
            manipulation_type, content, json.dumps(goal), False,
            json.dumps(leverage_used), intimacy_level
        ))
        
        attempt_id = cursor.fetchone()[0]
        
        # Create a memory for this manipulation attempt
        await create_conflict_memory(
            ctx,
            conflict_id,
            f"{npc_name} attempted to {manipulation_type} the player regarding the conflict.",
            significance=7
        )
        
        conn.commit()
        
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
        conn.rollback()
        logger.error(f"Error creating player manipulation attempt: {e}", exc_info=True)
        raise
    finally:
        cursor.close()
        conn.close()

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
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get the manipulation attempt
        cursor.execute("""
            SELECT conflict_id, npc_id, manipulation_type, goal
            FROM PlayerManipulationAttempts
            WHERE attempt_id = %s AND user_id = %s AND conversation_id = %s
        """, (attempt_id, context.user_id, context.conversation_id))
        
        row = cursor.fetchone()
        if not row:
            return {"error": "Manipulation attempt not found"}
        
        conflict_id, npc_id, manipulation_type, goal = row
        
        # Update the manipulation attempt
        cursor.execute("""
            UPDATE PlayerManipulationAttempts
            SET success = %s, player_response = %s, resolved_at = CURRENT_TIMESTAMP
            WHERE attempt_id = %s
            RETURNING attempt_id
        """, (success, player_response, attempt_id))
        
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
            cursor.execute("""
                SELECT involvement_level, faction
                FROM PlayerConflictInvolvement
                WHERE conflict_id = %s AND user_id = %s AND conversation_id = %s
            """, (conflict_id, context.user_id, context.conversation_id))
            
            involvement_row = cursor.fetchone()
            
            if involvement_row:
                current_involvement, current_faction = involvement_row
                
                # Update with goal values or keep current if not specified
                faction = goal_dict.get("faction", current_faction)
                involvement_level = goal_dict.get("involvement_level", current_involvement)
            else:
                # If no involvement yet, set defaults
                faction = goal_dict.get("faction", "neutral")
                involvement_level = goal_dict.get("involvement_level", "observing")
            
            # Record that player was manipulated
            cursor.execute("""
                UPDATE PlayerConflictInvolvement
                SET involvement_level = %s, faction = %s, manipulated_by = %s
                WHERE conflict_id = %s AND user_id = %s AND conversation_id = %s
            """, (
                involvement_level, faction, 
                json.dumps({"npc_id": npc_id, "manipulation_type": manipulation_type, "attempt_id": attempt_id}),
                conflict_id, context.user_id, context.conversation_id
            ))
            
            # If no rows updated, insert new involvement
            if cursor.rowcount == 0:
                cursor.execute("""
                    INSERT INTO PlayerConflictInvolvement
                    (conflict_id, user_id, conversation_id, player_name, involvement_level,
                    faction, money_committed, supplies_committed, influence_committed, 
                    actions_taken, manipulated_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    conflict_id, context.user_id, context.conversation_id, "Player",
                    involvement_level, faction, 0, 0, 0, "[]",
                    json.dumps({"npc_id": npc_id, "manipulation_type": manipulation_type, "attempt_id": attempt_id})
                ))
                
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
        
        conn.commit()
        
        return {
            "attempt_id": attempt_id,
            "success": success,
            "player_response": player_response,
            "is_resolved": True,
            "stat_changes": stat_changes
        }
    except Exception as e:
        conn.rollback()
        logger.error(f"Error resolving manipulation attempt: {e}", exc_info=True)
        raise
    finally:
        cursor.close()
        conn.close()

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
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Create the story beat
        cursor.execute("""
            INSERT INTO PathStoryBeats
            (conflict_id, path_id, description, involved_npcs, progress_value, created_at)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING beat_id
        """, (
            conflict_id, path_id, beat_description, 
            json.dumps(involved_npcs), progress_value
        ))
        
        beat_id = cursor.fetchone()[0]
        
        # Get current path progress
        cursor.execute("""
            SELECT progress, is_completed
            FROM ResolutionPaths
            WHERE conflict_id = %s AND path_id = %s
        """, (conflict_id, path_id))
        
        row = cursor.fetchone()
        if not row:
            return {"error": "Resolution path not found"}
        
        current_progress, is_completed = row
        
        # Calculate new progress
        new_progress = min(100, current_progress + progress_value)
        is_now_completed = new_progress >= 100
        
        # Update the path progress
        cursor.execute("""
            UPDATE ResolutionPaths
            SET progress = %s, is_completed = %s,
                completion_date = %s
            WHERE conflict_id = %s AND path_id = %s
        """, (
            new_progress, is_now_completed,
            "CURRENT_TIMESTAMP" if is_now_completed else None,
            conflict_id, path_id
        ))
        
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
        
        conn.commit()
        
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
        conn.rollback()
        logger.error(f"Error tracking story beat: {e}", exc_info=True)
        raise
    finally:
        cursor.close()
        conn.close()

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
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get conflict details
        cursor.execute("""
            SELECT conflict_name, conflict_type, phase, progress
            FROM Conflicts
            WHERE conflict_id = %s AND user_id = %s AND conversation_id = %s
        """, (conflict_id, context.user_id, context.conversation_id))
        
        row = cursor.fetchone()
        if not row:
            return {"error": "Conflict not found"}
        
        conflict_name, conflict_type, phase, progress = row
        
        # Check if conflict is in a resolvable state
        if phase != "resolution" and progress < 90:
            return {"error": "Conflict is not ready to be resolved. Progress must be at least 90%."}
        
        # Get player involvement
        cursor.execute("""
            SELECT involvement_level, faction
            FROM PlayerConflictInvolvement
            WHERE conflict_id = %s AND user_id = %s AND conversation_id = %s
        """, (conflict_id, context.user_id, context.conversation_id))
        
        player_row = cursor.fetchone()
        player_involvement = player_row[0] if player_row else "none"
        player_faction = player_row[1] if player_row else "neutral"
        
        # Get completed resolution paths
        cursor.execute("""
            SELECT path_id, name
            FROM ResolutionPaths
            WHERE conflict_id = %s AND is_completed = TRUE
        """, (conflict_id, ))
        
        completed_paths = cursor.fetchall()
        
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
        
        # Update conflict status
        cursor.execute("""
            UPDATE Conflicts
            SET is_active = FALSE, progress = 100, phase = 'concluded',
                outcome = %s, resolution_description = %s,
                resolved_at = CURRENT_TIMESTAMP
            WHERE conflict_id = %s
        """, (outcome, description, conflict_id))
        
        # Generate consequences
        consequences = generate_conflict_consequences(
            conflict_type, outcome, player_involvement, player_faction, completed_paths
        )
        
        # Record consequences
        for consequence in consequences:
            cursor.execute("""
                INSERT INTO ConflictConsequences
                (conflict_id, consequence_type, description, affected_entity_type,
                 affected_entity_id, magnitude, is_permanent)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                conflict_id,
                consequence.get("type", "general"),
                consequence.get("description", ""),
                consequence.get("affected_entity_type", "player"),
                consequence.get("affected_entity_id", 0),
                consequence.get("magnitude", 1),
                consequence.get("is_permanent", False)
            ))
            
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
        
        conn.commit()
        
        return {
            "conflict_id": conflict_id,
            "outcome": outcome,
            "description": description,
            "consequences": consequences,
            "resolved_at": "now"
        }
    except Exception as e:
        conn.rollback()
        logger.error(f"Error resolving conflict: {e}", exc_info=True)
        raise
    finally:
        cursor.close()
        conn.close()
