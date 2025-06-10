# logic/conflict_system/conflict_tools.py
"""
Conflict System Function Tools - Refactored for Canon Integration

This module defines the function tools used by the conflict system agents.
All entity creation and state changes now go through the canon system.
"""

import logging
import json
import random
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict
from pydantic import BaseModel, Field
from agents import (
    Agent, Runner, trace, function_tool, 
    RunContextWrapper, handoff, ModelSettings,
    InputGuardrail, GuardrailFunctionOutput, 
    Handoff, RunConfig, FunctionTool
)

from db.connection import get_db_connection_context
from lore.core import canon
from logic.resource_management import ResourceManager
from logic.conflict_system.conflict_agents import get_relationship_status, get_manipulation_leverage
from logic.chatgpt_integration import get_chatgpt_response
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from context.context_performance import track_performance

logger = logging.getLogger(__name__)

class StakeholderStatusUpdate(BaseModel):
    """Model for stakeholder status updates."""
    involvement_level: Optional[int] = None
    faction_standing: Optional[int] = None
    willing_to_betray_faction: Optional[bool] = None
    public_motivation: Optional[str] = None
    private_motivation: Optional[str] = None
    desired_outcome: Optional[str] = None
    faction_position: Optional[str] = None
    leadership_ambition: Optional[int] = None

# REFACTORED: Create stakeholders with canon integration
async def _internal_create_stakeholders_logic(ctx: RunContextWrapper, conflict_id: int, conflict_data: Dict[str, Any], stakeholder_npcs: List[Dict[str, Any]]) -> None:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            stakeholders_to_insert = conflict_data.get("stakeholders", [])
            if not stakeholders_to_insert:
                # Generate default stakeholders from NPCs
                stakeholders_to_insert = []
                for i, npc in enumerate(stakeholder_npcs):
                    stakeholder = {
                        "npc_id": npc["npc_id"],
                        "npc_name": npc.get("npc_name", f"NPC {npc['npc_id']}"),
                        "public_motivation": f"{npc.get('npc_name', 'They')} wants to see the conflict resolved peacefully.",
                        "private_motivation": f"{npc.get('npc_name', 'They')} actually wants to gain power through the conflict.",
                        "desired_outcome": "Control the outcome to their advantage",
                        "involvement_level": 7 - i,
                        "faction_affiliations": npc.get("faction_affiliations", [])
                    }
                    stakeholders_to_insert.append(stakeholder)
            
            async with conn.transaction():
                for sh in stakeholders_to_insert:
                    # REFACTORED: Ensure NPC exists canonically
                    if "npc_name" in sh and not sh.get("npc_id"):
                        npc_id = await canon.find_or_create_npc(
                            ctx, conn,
                            npc_name=sh["npc_name"],
                            role=sh.get("role", "stakeholder"),
                            affiliations=sh.get("faction_affiliations", [])
                        )
                        sh["npc_id"] = npc_id
                    else:
                        npc_id = sh["npc_id"]
                    
                    # Find matching NPC from the provided list
                    npc = next((n for n in stakeholder_npcs if n["npc_id"] == npc_id), None)
                    if not npc:
                        # If not in list, fetch from database
                        npc_row = await conn.fetchrow("""
                            SELECT npc_id, npc_name, dominance, cruelty, faction_affiliations
                            FROM NPCStats
                            WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                        """, npc_id, context.user_id, context.conversation_id)
                        if npc_row:
                            npc = dict(npc_row)
                        else:
                            continue
                    
                    # REFACTORED: Handle faction creation/lookup canonically
                    faction_id = sh.get("faction_id")
                    faction_name = sh.get("faction_name")
                    
                    if faction_name and not faction_id:
                        faction_id = await canon.find_or_create_faction(
                            ctx, conn,
                            faction_name=faction_name,
                            type=sh.get("faction_type", "organization")
                        )
                    elif not faction_id and npc.get("faction_affiliations"):
                        # Use NPC's primary faction
                        affiliations = npc["faction_affiliations"]
                        if isinstance(affiliations, str):
                            try:
                                affiliations = json.loads(affiliations)
                            except:
                                affiliations = []
                        if affiliations and len(affiliations) > 0:
                            faction_id = affiliations[0].get("faction_id")
                            faction_name = affiliations[0].get("faction_name")
                    
                    # Insert stakeholder
                    await conn.execute("""
                        INSERT INTO ConflictStakeholders 
                        (conflict_id, npc_id, faction_id, faction_name, faction_position, 
                         public_motivation, private_motivation, desired_outcome, involvement_level, 
                         alliances, rivalries, leadership_ambition, faction_standing, willing_to_betray_faction)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                        ON CONFLICT (conflict_id, npc_id) DO UPDATE SET
                            faction_id = EXCLUDED.faction_id,
                            faction_name = EXCLUDED.faction_name,
                            public_motivation = EXCLUDED.public_motivation
                    """, 
                    conflict_id, npc_id, faction_id, faction_name, 
                    sh.get("faction_position", "Member"),
                    sh.get("public_motivation", "Wants favorable resolution"),
                    sh.get("private_motivation", "Seeks personal gain"),
                    sh.get("desired_outcome", "Victory"),
                    sh.get("involvement_level", 5),
                    json.dumps(sh.get("alliances", {})),
                    json.dumps(sh.get("rivalries", {})),
                    sh.get("leadership_ambition", npc.get("dominance", 50) // 10),
                    sh.get("faction_standing", 50),
                    sh.get("willing_to_betray_faction", npc.get("cruelty", 20) > 60))
                    
                    # Handle secrets
                    if "secrets" in sh:
                        for secret in sh["secrets"]:
                            # REFACTORED: Ensure target NPC exists if referenced
                            target_npc_id = secret.get("target_npc_id")
                            if secret.get("target_npc_name") and not target_npc_id:
                                target_npc_id = await canon.find_or_create_npc(
                                    ctx, conn,
                                    npc_name=secret["target_npc_name"]
                                )
                            
                            await conn.execute("""
                                INSERT INTO StakeholderSecrets 
                                (conflict_id, npc_id, secret_id, secret_type, content, 
                                 target_npc_id, is_revealed, revealed_to, is_public)
                                VALUES ($1, $2, $3, $4, $5, $6, FALSE, NULL, FALSE)
                            """, 
                            conflict_id, npc_id, 
                            secret.get("secret_id", f"secret_{npc_id}_{random.randint(1000,9999)}"),
                            secret.get("secret_type", "personal"),
                            secret.get("content", "A hidden truth"),
                            target_npc_id)
                
                # REFACTORED: Log canonical event
                await canon.log_canonical_event(
                    ctx, conn,
                    f"Stakeholders established for conflict {conflict_id} with {len(stakeholders_to_insert)} participants",
                    tags=["conflict", "stakeholders", "initialization"],
                    significance=5
                )
                    
    except Exception as e:
        logger.error(f"Error creating stakeholders: {e}", exc_info=True)
        raise

# REFACTORED: Create manipulation attempt already uses canon
async def _internal_create_manipulation_attempt_logic(ctx: RunContextWrapper, conflict_id: int, npc_id: int, manipulation_type: str, content: str, goal: Dict[str, Any], leverage_used: Dict[str, Any], intimacy_level: int = 0) -> Dict[str, Any]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            npc_name = await _internal_get_npc_name_logic(ctx, npc_id)
            async with conn.transaction():
                # This already uses canon function - good!
                attempt_id = await canon.create_player_manipulation_attempt(
                    ctx,
                    conn,
                    conflict_id,
                    npc_id,
                    manipulation_type,
                    content,
                    goal,
                    leverage_used,
                    intimacy_level,
                )
                
                # REFACTORED: Use canon for memory
                await canon.log_canonical_event(
                    ctx, conn,
                    f"{npc_name} attempted to {manipulation_type} the player regarding conflict {conflict_id}",
                    tags=["conflict", "manipulation", manipulation_type],
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
                    "is_resolved": False,
                }
    except Exception as e:
        logger.error(f"Error creating player manipulation attempt: {e}", exc_info=True)
        raise

# REFACTORED: Resolve manipulation with canon stat updates
async def _internal_resolve_manipulation_attempt_logic(ctx: RunContextWrapper, attempt_id: int, success: bool, player_response: str) -> Dict[str, Any]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
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
            goal_str = attempt_row["goal"]
            
            async with conn.transaction():
                # This already uses canon function - good!
                await canon.resolve_player_manipulation_attempt(
                    ctx,
                    conn,
                    attempt_id,
                    success,
                    player_response,
                )
                
                # REFACTORED: Get current player stats to calculate changes
                player_name = "Chase"  # Default player name
                stat_changes = {}
                
                if success:
                    # Player succumbed to manipulation
                    obedience_change = random.randint(2, 5)
                    dependency_change = random.randint(1, 3)
                    
                    # Get current stats
                    stats_row = await conn.fetchrow("""
                        SELECT obedience, dependency FROM PlayerStats
                        WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
                    """, context.user_id, context.conversation_id, player_name)
                    
                    if stats_row:
                        new_obedience = stats_row['obedience'] + obedience_change
                        new_dependency = stats_row['dependency'] + dependency_change
                    else:
                        # Create player stats if they don't exist
                        await canon.find_or_create_player_stats(ctx, conn, player_name)
                        new_obedience = obedience_change
                        new_dependency = dependency_change
                    
                    # REFACTORED: Use canon stat updates
                    await canon.update_player_stat_canonically(
                        ctx, conn, player_name, "obedience", new_obedience, 
                        f"Manipulation success: {manipulation_type} by NPC {npc_id}"
                    )
                    await canon.update_player_stat_canonically(
                        ctx, conn, player_name, "dependency", new_dependency,
                        f"Manipulation success: {manipulation_type} by NPC {npc_id}"
                    )
                    
                    stat_changes = {"obedience": obedience_change, "dependency": dependency_change}
                    
                    # Update player involvement
                    goal_dict = json.loads(goal_str) if isinstance(goal_str, str) else goal_str or {}
                    involvement_row = await conn.fetchrow("""
                        SELECT involvement_level, faction 
                        FROM PlayerConflictInvolvement 
                        WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
                    """, conflict_id, context.user_id, context.conversation_id)
                    
                    faction = goal_dict.get("faction", involvement_row["faction"] if involvement_row else "neutral")
                    involvement_level = goal_dict.get("involvement_level", involvement_row["involvement_level"] if involvement_row else "observing")
                    manipulated_by_json = json.dumps({
                        "npc_id": npc_id, 
                        "manipulation_type": manipulation_type, 
                        "attempt_id": attempt_id
                    })
                    
                    if involvement_row:
                        await conn.execute("""
                            UPDATE PlayerConflictInvolvement 
                            SET involvement_level = $1, faction = $2, manipulated_by = $3 
                            WHERE conflict_id = $4 AND user_id = $5 AND conversation_id = $6
                        """, involvement_level, faction, manipulated_by_json, 
                        conflict_id, context.user_id, context.conversation_id)
                    else:
                        await conn.execute("""
                            INSERT INTO PlayerConflictInvolvement 
                            (conflict_id, user_id, conversation_id, player_name, 
                             involvement_level, faction, money_committed, supplies_committed, 
                             influence_committed, actions_taken, manipulated_by) 
                            VALUES ($1, $2, $3, $4, $5, $6, 0, 0, 0, '[]', $7)
                        """, conflict_id, context.user_id, context.conversation_id, player_name,
                        involvement_level, faction, manipulated_by_json)
                else:
                    # Player resisted manipulation
                    willpower_change = random.randint(2, 4)
                    confidence_change = random.randint(1, 3)
                    
                    # Get current stats
                    stats_row = await conn.fetchrow("""
                        SELECT willpower, confidence FROM PlayerStats
                        WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
                    """, context.user_id, context.conversation_id, player_name)
                    
                    if stats_row:
                        new_willpower = stats_row['willpower'] + willpower_change
                        new_confidence = stats_row['confidence'] + confidence_change
                    else:
                        await canon.find_or_create_player_stats(ctx, conn, player_name)
                        new_willpower = willpower_change
                        new_confidence = confidence_change
                    
                    # REFACTORED: Use canon stat updates
                    await canon.update_player_stat_canonically(
                        ctx, conn, player_name, "willpower", new_willpower,
                        f"Resisted manipulation: {manipulation_type} by NPC {npc_id}"
                    )
                    await canon.update_player_stat_canonically(
                        ctx, conn, player_name, "confidence", new_confidence,
                        f"Resisted manipulation: {manipulation_type} by NPC {npc_id}"
                    )
                    
                    stat_changes = {"willpower": willpower_change, "confidence": confidence_change}
                
                # Get NPC name and log event
                npc_name = await _internal_get_npc_name_logic(ctx, npc_id)
                
                # REFACTORED: Use canon event logging
                await canon.log_canonical_event(
                    ctx, conn,
                    f"Player {'succumbed to' if success else 'resisted'} {npc_name}'s {manipulation_type} attempt in conflict {conflict_id}",
                    tags=["conflict", "manipulation", "resolution", "success" if success else "failure"],
                    significance=8 if success else 7
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

# REFACTORED: Create conflict memory using canon
async def _internal_create_conflict_memory_logic(ctx: RunContextWrapper, conflict_id: int, memory_text: str, significance: int = 5) -> int:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            # Create the memory event
            memory_id = await conn.fetchval("""
                INSERT INTO ConflictMemoryEvents 
                (conflict_id, memory_text, significance, entity_type, entity_id, 
                 user_id, conversation_id) 
                VALUES ($1, $2, $3, $4, $5, $6, $7) 
                RETURNING id
            """, conflict_id, memory_text, significance, "conflict", conflict_id, 
            context.user_id, context.conversation_id)
            
            # REFACTORED: Also log as canonical event
            await canon.log_canonical_event(
                ctx, conn,
                f"Conflict {conflict_id}: {memory_text}",
                tags=["conflict", "memory", f"significance_{significance}"],
                significance=significance
            )
            
            return memory_id if memory_id else 0
    except Exception as e:
        logger.error(f"Error creating conflict memory: {e}", exc_info=True)
        return 0

# REFACTORED: Generate player manipulation attempts with canon NPCs
async def _internal_generate_player_manipulation_attempts_logic(ctx: RunContextWrapper, conflict_id: int, stakeholder_npcs: List[Dict[str, Any]]) -> None:
    context = ctx.context
    eligible_npcs = [
        npc for npc in stakeholder_npcs 
        if npc.get("sex", "female") == "female" and 
        (npc.get("dominance", 0) > 70 or 
         npc.get("relationship_with_player", {}).get("closeness", 0) > 70)
    ]
    
    if not eligible_npcs: 
        return
    
    manipulation_types = ["domination", "blackmail", "seduction", "coercion", "bribery"]
    involvement_levels = ["observing", "participating", "leading"]
    factions = ["a", "b", "neutral"]
    
    for npc in eligible_npcs[:2]:  # Limit to 2 NPCs
        if random.random() > 0.7: 
            continue
            
        # Determine manipulation type based on NPC traits
        if npc.get("dominance", 0) > 80:
            manip_type = "domination"
        elif npc.get("cruelty", 0) > 70:
            manip_type = "blackmail"
        elif npc.get("relationship_with_player", {}).get("closeness", 0) > 80:
            manip_type = "seduction"
        else:
            manip_type = random.choice(manipulation_types)
        
        involvement = random.choice(involvement_levels)
        faction_choice = random.choice(factions)
        
        # Generate content
        goal = {
            "faction": faction_choice, 
            "involvement_level": involvement
        }
        
        # Get manipulation content
        if manip_type == "domination":
            content = generate_domination_content(npc, npc.get("relationship_with_player", {}), goal, {})
        elif manip_type == "seduction":
            content = generate_seduction_content(npc, npc.get("relationship_with_player", {}), goal, {})
        elif manip_type == "blackmail":
            content = generate_blackmail_content(npc, npc.get("relationship_with_player", {}), goal, {}, [])
        else:
            content = generate_generic_manipulation_content(npc, npc.get("relationship_with_player", {}), goal, {})
        
        goal["specific_actions"] = random.choice(["Spy on the other faction", "Convince others to join", "Sabotage opposition efforts", "Gather intelligence"])
        
        leverage = generate_leverage(npc, npc.get("relationship_with_player", {}), manip_type)
        intimacy = calculate_intimacy_level(npc, npc.get("relationship_with_player", {}), manip_type)
        
        # Create the manipulation attempt
        await _internal_create_manipulation_attempt_logic(
            ctx, conflict_id, npc["npc_id"], manip_type, content, goal, leverage, intimacy
        )

# REFACTORED: Faction coup attempt with canon integration
async def _internal_attempt_faction_coup_logic(ctx: RunContextWrapper, struggle_id: int, approach: str, supporting_npcs: List[int], resources_committed: Dict[str, int]) -> Dict[str, Any]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
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
            
            # REFACTORED: Use canon resource management
            if sum(resources_committed.values()) > 0:
                resource_result = await canon.adjust_player_resource(
                    ctx, conn, "Chase", "money", -resources_committed.get("money", 0), 
                    "coup_attempt", f"Committed to coup in struggle {struggle_id}"
                )
                
                if resource_result['new_value'] < 0:
                    return {"error": "Insufficient money", "current": resource_result['old_value']}
                    
                await canon.adjust_player_resource(
                    ctx, conn, "Chase", "supplies", -resources_committed.get("supplies", 0),
                    "coup_attempt", f"Committed to coup in struggle {struggle_id}"
                )
                
                await canon.adjust_player_resource(
                    ctx, conn, "Chase", "influence", -resources_committed.get("influence", 0),
                    "coup_attempt", f"Committed to coup in struggle {struggle_id}"
                )
            
            # Calculate success chance
            success_chance = await _internal_calculate_coup_success_chance_logic(
                ctx, struggle_id, approach, supporting_npcs, resources_committed
            )
            success = random.random() * 100 <= success_chance
            
            async with conn.transaction():
                # Record coup attempt
                coup_id = await conn.fetchval("""
                    INSERT INTO FactionCoupAttempts 
                    (struggle_id, approach, supporting_npcs, resources_committed, 
                     success, success_chance, timestamp) 
                    VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP) 
                    RETURNING id
                """, struggle_id, approach, json.dumps(supporting_npcs), 
                json.dumps(resources_committed), success, success_chance)
                
                # Get names
                primary_name = await _internal_get_npc_name_logic(ctx, primary_npc_id)
                target_name = await _internal_get_npc_name_logic(ctx, target_npc_id)
                faction_name = await _internal_get_faction_name_logic(ctx, faction_id)
                
                player_name = "Chase"
                stat_changes = {}
                
                if success:
                    # Update struggle status
                    await conn.execute("""
                        UPDATE InternalFactionConflicts 
                        SET current_phase = $1, progress = 100, resolved_at = CURRENT_TIMESTAMP 
                        WHERE struggle_id = $2
                    """, "resolved", struggle_id)
                    
                    # REFACTORED: Log canonical event
                    await canon.log_canonical_event(
                        ctx, conn,
                        f"{primary_name}'s coup against {target_name} in {faction_name} succeeded",
                        tags=["conflict", "coup", "success", "faction_politics"],
                        significance=8
                    )
                    
                    result = {
                        "outcome": "success", 
                        "description": f"{primary_name} successfully overthrew {target_name} in {faction_name}."
                    }
                    
                    # REFACTORED: Update player stats canonically
                    stats_row = await conn.fetchrow("""
                        SELECT corruption, confidence FROM PlayerStats
                        WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
                    """, context.user_id, context.conversation_id, player_name)
                    
                    if stats_row:
                        new_corruption = stats_row['corruption'] + 3
                        new_confidence = stats_row['confidence'] + 5
                    else:
                        await canon.find_or_create_player_stats(ctx, conn, player_name)
                        new_corruption = 3
                        new_confidence = 5
                    
                    await canon.update_player_stat_canonically(
                        ctx, conn, player_name, "corruption", new_corruption,
                        f"Successful coup participation in struggle {struggle_id}"
                    )
                    await canon.update_player_stat_canonically(
                        ctx, conn, player_name, "confidence", new_confidence,
                        f"Successful coup participation in struggle {struggle_id}"
                    )
                    
                    stat_changes = {"corruption": 3, "confidence": 5}
                else:
                    # Failed coup - reverse the leadership
                    await conn.execute("""
                        UPDATE InternalFactionConflicts 
                        SET current_phase = $1, primary_npc_id = $2, target_npc_id = $3, 
                            description = $4 
                        WHERE struggle_id = $5
                    """, "aftermath", target_npc_id, primary_npc_id, 
                    f"Failed coup attempt by {primary_name}. {target_name} consolidates power.", 
                    struggle_id)
                    
                    # REFACTORED: Log canonical event
                    await canon.log_canonical_event(
                        ctx, conn,
                        f"{primary_name}'s coup against {target_name} in {faction_name} failed",
                        tags=["conflict", "coup", "failure", "faction_politics"],
                        significance=8
                    )
                    
                    result = {
                        "outcome": "failure", 
                        "description": f"{primary_name}'s coup failed. {target_name} remains in power."
                    }
                    
                    # REFACTORED: Update player stats canonically
                    stats_row = await conn.fetchrow("""
                        SELECT mental_resilience FROM PlayerStats
                        WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
                    """, context.user_id, context.conversation_id, player_name)
                    
                    if stats_row:
                        new_resilience = stats_row['mental_resilience'] + 4
                    else:
                        await canon.find_or_create_player_stats(ctx, conn, player_name)
                        new_resilience = 4
                    
                    await canon.update_player_stat_canonically(
                        ctx, conn, player_name, "mental_resilience", new_resilience,
                        f"Failed coup participation in struggle {struggle_id}"
                    )
                    
                    stat_changes = {"mental_resilience": 4}
                
                # Update coup attempt result
                await conn.execute("""
                    UPDATE FactionCoupAttempts 
                    SET result = $1 
                    WHERE id = $2
                """, json.dumps(result), coup_id)
                
                # Get current resources
                resources = {}
                resource_row = await conn.fetchrow("""
                    SELECT money, supplies, influence FROM PlayerResources
                    WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
                """, context.user_id, context.conversation_id, player_name)
                
                if resource_row:
                    resources = {
                        "money": resource_row['money'],
                        "supplies": resource_row['supplies'],
                        "influence": resource_row['influence']
                    }
                
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

# REFACTORED: Create conflict record with canon
async def _internal_create_conflict_record_logic(ctx: RunContextWrapper, conflict_data: Dict[str, Any], current_day: int) -> int:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            success_rate = {
                "minor": 0.75, 
                "standard": 0.5, 
                "major": 0.25, 
                "catastrophic": 0.1
            }.get(conflict_data.get("conflict_type", "standard"), 0.5)
            
            conflict_id = await conn.fetchval("""
                INSERT INTO Conflicts 
                (user_id, conversation_id, conflict_name, conflict_type, description, 
                 progress, phase, start_day, estimated_duration, success_rate, 
                 outcome, is_active) 
                VALUES ($1, $2, $3, $4, $5, 0.0, 'brewing', $6, $7, $8, 'pending', TRUE) 
                RETURNING conflict_id
            """, context.user_id, context.conversation_id, 
            conflict_data.get("conflict_name", "Unnamed Conflict"), 
            conflict_data.get("conflict_type", "standard"), 
            conflict_data.get("description", "A conflict has emerged"), 
            current_day, 
            conflict_data.get("estimated_duration", 7), 
            success_rate)
            
            # REFACTORED: Log canonical event
            await canon.log_canonical_event(
                ctx, conn,
                f"New conflict created: {conflict_data.get('conflict_name', 'Unnamed')} ({conflict_data.get('conflict_type', 'standard')})",
                tags=["conflict", "creation", conflict_data.get("conflict_type", "standard")],
                significance=7
            )
            
            return conflict_id if conflict_id else 0
    except Exception as e:
        logger.error(f"Error creating conflict record: {e}", exc_info=True)
        raise

# REFACTORED: Resolve conflict with canon rewards
async def _internal_resolve_conflict_logic(ctx: RunContextWrapper, conflict_id: int) -> Dict[str, Any]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
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
            
            if phase != "resolution" and progress < 90: 
                return {"error": "Conflict is not ready to be resolved. Must be in resolution phase or have 90%+ progress."}
            
            # Get player involvement
            player_row = await conn.fetchrow("""
                SELECT involvement_level, faction 
                FROM PlayerConflictInvolvement 
                WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
            """, conflict_id, context.user_id, context.conversation_id)
            
            player_involvement = player_row["involvement_level"] if player_row else "none"
            player_faction = player_row["faction"] if player_row else "neutral"
            
            # Get completed paths
            completed_paths = await conn.fetch("""
                SELECT path_id, name 
                FROM ResolutionPaths 
                WHERE conflict_id = $1 AND is_completed = TRUE
            """, conflict_id)
            
            # Determine outcome
            if not completed_paths:
                outcome = "unresolved"
                description = "The conflict ended without clear resolution."
            else:
                outcome = "resolved"
                description = f"The conflict was resolved with {player_faction if player_involvement != 'none' else 'neutral'} faction gaining advantage."
            
            async with conn.transaction():
                # Update conflict status
                await conn.execute("""
                    UPDATE Conflicts 
                    SET is_active = FALSE, progress = 100, phase = 'concluded', 
                        outcome = $1, resolution_description = $2, resolved_at = CURRENT_TIMESTAMP 
                    WHERE conflict_id = $3
                """, outcome, description, conflict_id)
                
                # Generate consequences
                consequences = generate_conflict_consequences(
                    conflict_type, outcome, player_involvement, 
                    player_faction, [dict(row) for row in completed_paths]
                )
                
                # Apply consequences
                player_name = "Chase"
                for con in consequences:
                    # Store consequence record
                    await conn.execute("""
                        INSERT INTO ConflictConsequences 
                        (conflict_id, consequence_type, description, affected_entity_type, 
                         affected_entity_id, magnitude, is_permanent) 
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """, conflict_id, con.get("type", "general"), 
                    con.get("description", ""), con.get("affected_entity_type", "player"), 
                    con.get("affected_entity_id", 0), con.get("magnitude", 1), 
                    con.get("is_permanent", False))
                    
                    # REFACTORED: Apply player stat changes canonically
                    if con.get("affected_entity_type") == "player" and "stat_changes" in con:
                        for stat, change in con["stat_changes"].items():
                            # Get current value
                            current_val = await conn.fetchval(
                                f"SELECT {stat} FROM PlayerStats WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3",
                                context.user_id, context.conversation_id, player_name
                            )
                            
                            if current_val is None:
                                await canon.find_or_create_player_stats(ctx, conn, player_name)
                                current_val = 0
                            
                            new_val = current_val + change
                            
                            await canon.update_player_stat_canonically(
                                ctx, conn, player_name, stat, new_val,
                                f"Conflict resolution reward: {con.get('description', 'Conflict outcome')}"
                            )
                    
                    # REFACTORED: Add item rewards canonically
                    elif con.get("type") == "item_reward" and "item" in con:
                        item = con["item"]
                        await canon.find_or_create_inventory_item(
                            ctx, conn,
                            item_name=item["name"],
                            player_name=player_name,
                            item_description=item.get("description", ""),
                            item_category=item.get("category", "conflict_reward"),
                            item_properties={
                                "rarity": item.get("rarity", "common"),
                                "resolution_style": item.get("resolution_style", "neutral"),
                                "source": "conflict_resolution",
                                "conflict_id": conflict_id,
                                "conflict_type": conflict_type
                            },
                            quantity=1,
                            equipped=False
                        )
                    
                    # REFACTORED: Handle perk rewards (store as special inventory items)
                    elif con.get("type") == "perk_reward" and "perk" in con:
                        perk = con["perk"]
                        await canon.find_or_create_inventory_item(
                            ctx, conn,
                            item_name=f"Perk: {perk['name']}",
                            player_name=player_name,
                            item_description=perk.get("description", ""),
                            item_category="perk",
                            item_properties={
                                "tier": perk.get("tier", 1),
                                "resolution_style": perk.get("resolution_style", "neutral"),
                                "source": "conflict_resolution",
                                "conflict_id": conflict_id,
                                "perk_type": perk.get("category", "conflict_resolution")
                            },
                            quantity=1,
                            equipped=True  # Perks are automatically "equipped"
                        )
                    
                    # REFACTORED: Handle special rewards
                    elif con.get("type") == "special_reward" and "special_reward" in con:
                        special = con["special_reward"]
                        await canon.find_or_create_inventory_item(
                            ctx, conn,
                            item_name=special["name"],
                            player_name=player_name,
                            item_description=special.get("description", ""),
                            item_category="unique_conflict_reward",
                            item_properties={
                                "effect": special.get("effect", ""),
                                "resolution_style": special.get("resolution_style", "neutral"),
                                "source": "conflict_resolution",
                                "conflict_id": conflict_id,
                                "unique": True
                            },
                            quantity=1,
                            equipped=False
                        )
                
                # REFACTORED: Log canonical event for resolution
                await canon.log_canonical_event(
                    ctx, conn,
                    f"Conflict '{conflict_name}' has been resolved. {description}",
                    tags=["conflict", "resolution", outcome, conflict_type],
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

# REFACTORED: Extract NPCs from narrative with canon consideration
async def _internal_extract_npcs_from_narrative_logic(ctx: RunContextWrapper, narrative_text: str) -> List[int]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            # Get all introduced NPCs
            npc_rows = await conn.fetch("""
                SELECT npc_id, npc_name 
                FROM NPCStats 
                WHERE user_id = $1 AND conversation_id = $2 AND introduced = TRUE
            """, context.user_id, context.conversation_id)
            
            mentioned_npc_ids = []
            
            # Check which NPCs are mentioned in the narrative
            for row in npc_rows:
                if row["npc_name"] in narrative_text:
                    mentioned_npc_ids.append(row["npc_id"])
            
            # REFACTORED: Check for new NPC names that might need creation
            # This is a simple pattern match - in production you'd want more sophisticated NER
            import re
            # Look for capitalized names that might be NPCs
            potential_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', narrative_text)
            
            existing_names = [row["npc_name"] for row in npc_rows]
            new_names = [name for name in potential_names if name not in existing_names and len(name.split()) <= 3]
            
            # For truly new NPCs mentioned, create them canonically
            for new_name in new_names[:3]:  # Limit to 3 new NPCs per narrative
                # Skip common words that might be capitalized
                if new_name.lower() in ['the', 'player', 'you', 'i', 'we', 'they', 'he', 'she']:
                    continue
                    
                # Create the new NPC canonically
                npc_id = await canon.find_or_create_npc(
                    ctx, conn,
                    npc_name=new_name,
                    role="mentioned_in_narrative"
                )
                mentioned_npc_ids.append(npc_id)
            
            return mentioned_npc_ids
    except Exception as e:
        logger.error(f"Error extracting NPCs from narrative: {e}", exc_info=True)
        return []

# REFACTORED: Add conflict to narrative with canon
async def _internal_add_conflict_to_narrative_logic(ctx: RunContextWrapper, narrative_text: str) -> Dict[str, Any]:
    context = ctx.context
    current_day = await _internal_get_current_day_logic(ctx)
    active_conflicts = await _internal_get_active_conflicts_logic(ctx)
    
    if len(active_conflicts) >= 3: 
        return {
            "trigger_conflict": False, 
            "reason": "Too many active conflicts", 
            "existing_conflicts": len(active_conflicts)
        }
    
    # Analyze narrative for conflict potential
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
    
    response = await get_chatgpt_response(context.conversation_id, "conflict_analysis", prompt)
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
            pass
    
    if not conflict_analysis.get("conflict_potential", False): 
        return {
            "trigger_conflict": False, 
            "reason": "No significant conflict potential", 
            "analysis": conflict_analysis
        }
    
    # Extract NPCs from narrative (now with canon creation)
    mentioned_npcs_ids = await _internal_extract_npcs_from_narrative_logic(ctx, narrative_text)
    
    if not mentioned_npcs_ids or len(mentioned_npcs_ids) < 2: 
        return {
            "trigger_conflict": False, 
            "reason": "Not enough NPCs involved", 
            "mentioned_npcs": mentioned_npcs_ids
        }
    
    # Create the conflict
    conflict_type = conflict_analysis.get("conflict_type", "minor")
    conflict_data = {
        "conflict_type": conflict_type, 
        "conflict_name": conflict_analysis.get("conflict_name", f"Narrative-triggered {conflict_type} conflict"), 
        "description": conflict_analysis.get("description", "A conflict arising from recent events"), 
        "narrative_source": narrative_text[:100] + "..."
    }
    
    # Create conflict record
    conflict_id = await _internal_create_conflict_record_logic(ctx, conflict_data, current_day)
    
    # Get NPC details for stakeholders
    stakeholder_npcs_details = []
    for npc_id in mentioned_npcs_ids[:4]:  # Limit to 4 stakeholders
        npc_details = await _internal_get_npc_details_logic(ctx, npc_id)
        if npc_details:
            stakeholder_npcs_details.append(npc_details)
    
    # Create stakeholders with canon integration
    await _internal_create_stakeholders_logic(ctx, conflict_id, conflict_data, stakeholder_npcs_details)

    # Generate basic resolution paths
    default_paths = [
        {
            "path_id": "diplomatic",
            "name": "Diplomatic Resolution",
            "description": "Resolve the conflict through negotiation and compromise.",
            "approach_type": "social",
            "difficulty": 5,
            "stakeholders_involved": [npc["npc_id"] for npc in stakeholder_npcs_details[:2]] if len(stakeholder_npcs_details) >= 2 else [],
            "key_challenges": ["Building trust", "Finding common ground", "Managing expectations"]
        },
        {
            "path_id": "direct",
            "name": "Direct Resolution",
            "description": "Resolve the conflict through confrontation and decisive action.",
            "approach_type": "direct",
            "difficulty": 7,
            "stakeholders_involved": [npc["npc_id"] for npc in stakeholder_npcs_details[1:3]] if len(stakeholder_npcs_details) >= 3 else [],
            "key_challenges": ["Overcoming resistance", "Managing consequences", "Securing victory"]
        }
    ]
    
    conflict_data["resolution_paths"] = default_paths
    await _internal_create_resolution_paths_logic(ctx, conflict_id, {"resolution_paths": default_paths})
    
    # REFACTORED: Log canonical event
    async with get_db_connection_context() as conn:
        await canon.log_canonical_event(
            ctx, conn,
            f"A new conflict emerged from the narrative: {conflict_data['conflict_name']}",
            tags=["conflict", "narrative_triggered", conflict_type],
            significance=7
        )
    
    return {
        "trigger_conflict": True,
        "conflict_id": conflict_id,
        "conflict_name": conflict_data["conflict_name"],
        "conflict_type": conflict_type,
        "stakeholders": [npc["npc_name"] for npc in stakeholder_npcs_details],
        "analysis": conflict_analysis
    }

# Helper Functions

async def _internal_get_npc_details_logic(ctx: RunContextWrapper, npc_id: int) -> Dict[str, Any]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            npc_row = await conn.fetchrow("SELECT npc_name, dominance, cruelty, closeness, trust, respect, intensity, sex FROM NPCStats WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3", npc_id, context.user_id, context.conversation_id)
            if not npc_row: return {}
            npc = dict(npc_row); npc["npc_id"] = npc_id
            return npc
    except Exception as e:
        logger.error(f"Error getting NPC details: {e}", exc_info=True)
        return {}

async def _internal_get_npc_name_logic(ctx: RunContextWrapper, npc_id: int) -> str:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            npc_name = await conn.fetchval("SELECT npc_name FROM NPCStats WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3", npc_id, context.user_id, context.conversation_id)
            return npc_name if npc_name else f"NPC {npc_id}"
    except Exception as e:
        logger.error(f"Error getting NPC name for ID {npc_id}: {e}", exc_info=True)
        return f"NPC {npc_id}"


async def _internal_get_faction_name_logic(ctx: RunContextWrapper, faction_id: int) -> str:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            faction_name = await conn.fetchval("SELECT faction_name FROM Factions WHERE faction_id = $1", faction_id) # Assuming Factions table exists and is not user/conversation specific
            return faction_name if faction_name else f"Faction {faction_id}"
    except Exception as e:
        logger.error(f"Error getting faction name for ID {faction_id}: {e}", exc_info=True)
        return f"Faction {faction_id}"

async def _internal_get_player_stats_logic(ctx: RunContextWrapper) -> Dict[str, Any]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            stats_row = await conn.fetchrow("SELECT corruption, confidence, willpower, obedience, dependency, lust, mental_resilience, physical_endurance FROM PlayerStats WHERE user_id = $1 AND conversation_id = $2", context.user_id, context.conversation_id)
            return dict(stats_row) if stats_row else {}
    except Exception as e:
        logger.error(f"Error getting player stats: {e}", exc_info=True)
        return {}

async def _internal_get_stakeholder_secrets_logic(ctx: RunContextWrapper, conflict_id: int, npc_id: int) -> List[Dict[str, Any]]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            secrets_rows = await conn.fetch("SELECT secret_id, secret_type, content, target_npc_id, is_revealed, revealed_to, is_public FROM StakeholderSecrets WHERE conflict_id = $1 AND npc_id = $2", conflict_id, npc_id)
            secrets = []
            for row in secrets_rows:
                secret = dict(row)
                secrets.append({"secret_id": secret["secret_id"], "secret_type": secret["secret_type"], "is_revealed": False} if not secret["is_revealed"] else secret)
            return secrets
    except Exception as e:
        logger.error(f"Error getting stakeholder secrets: {e}", exc_info=True)
        return []

async def _internal_check_stakeholder_manipulates_player_logic(ctx: RunContextWrapper, conflict_id: int, npc_id: int) -> bool:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM PlayerManipulationAttempts WHERE conflict_id = $1 AND npc_id = $2 AND user_id = $3 AND conversation_id = $4", conflict_id, npc_id, context.user_id, context.conversation_id)
            return count > 0
    except Exception as e:
        logger.error(f"Error checking if stakeholder manipulates player: {e}", exc_info=True)
        return False

async def _internal_check_conflict_advancement_logic(ctx: RunContextWrapper, conflict_id: int) -> None:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            conflict_row = await conn.fetchrow("SELECT progress, phase FROM Conflicts WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3", conflict_id, context.user_id, context.conversation_id)
            if not conflict_row: return
            progress, phase = conflict_row["progress"], conflict_row["phase"]
            thresholds = {"brewing": 30, "active": 60, "climax": 90}
            new_phase = phase
            if phase in thresholds and progress >= thresholds[phase]:
                if phase == "brewing": new_phase = "active"
                elif phase == "active": new_phase = "climax"
                elif phase == "climax": new_phase = "resolution"
            if new_phase != phase:
                async with conn.transaction():
                    await conn.execute("UPDATE Conflicts SET phase = $1, updated_at = CURRENT_TIMESTAMP WHERE conflict_id = $2 AND user_id = $3 AND conversation_id = $4", new_phase, conflict_id, context.user_id, context.conversation_id)
                    await _internal_create_conflict_memory_logic(ctx, conflict_id, f"Conflict progressed from {phase} to {new_phase}.", significance=7)
    except Exception as e:
        logger.error(f"Error checking conflict advancement: {e}", exc_info=True)

async def _internal_generate_struggle_details_logic(ctx: RunContextWrapper, faction_id: int, challenger_npc_id: int, target_npc_id: int, prize: str, approach: str) -> Dict[str, Any]:
    faction_name = await _internal_get_faction_name_logic(ctx, faction_id)
    challenger_name = await _internal_get_npc_name_logic(ctx, challenger_npc_id)
    target_name = await _internal_get_npc_name_logic(ctx, target_npc_id)
    members = await _internal_get_faction_members_logic(ctx, faction_id)
    conflict_name = f"Power struggle in {faction_name}"
    description = f"{challenger_name} challenges {target_name} for {prize} within {faction_name}."
    sides = defaultdict(list) # Ensure defaultdict is imported from collections
    for member in members:
        if member["npc_id"] == challenger_npc_id or member["npc_id"] == target_npc_id: continue
        affinity_challenger, affinity_target = random.randint(0, 100), random.randint(0, 100)
        side = "neutral" if abs(affinity_challenger - affinity_target) < 20 else ("challenger" if affinity_challenger > affinity_target else "incumbent")
        sides[side].append(member)
    faction_members_list = [{"npc_id": challenger_npc_id, "position": "Challenger", "side": "challenger"}, {"npc_id": target_npc_id, "position": "Incumbent", "side": "incumbent"}]
    for side, members_list_val in sides.items():
        for member in members_list_val: faction_members_list.append({"npc_id": member["npc_id"], "position": member.get("position", "Member"), "side": side})
    ideological_differences = [{"issue": f"Approach to {prize}", "incumbent_position": f"{target_name}'s way", "challenger_position": f"{challenger_name}'s way"}]
    return {"conflict_name": conflict_name, "description": description, "faction_members": faction_members_list, "ideological_differences": ideological_differences}

async def _internal_get_faction_members_logic(ctx: RunContextWrapper, faction_id: int) -> List[Dict[str, Any]]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            npc_rows = await conn.fetch("SELECT npc_id, npc_name, dominance, cruelty, faction_affiliations FROM NPCStats WHERE user_id = $1 AND conversation_id = $2", context.user_id, context.conversation_id)
            members = []
            for row in npc_rows:
                npc = dict(row)
                affiliations = json.loads(npc.get("faction_affiliations", "[]")) if isinstance(npc.get("faction_affiliations"), str) else npc.get("faction_affiliations", []) or []
                for aff in affiliations:
                    if aff.get("faction_id") == faction_id:
                        members.append({"npc_id": npc["npc_id"], "npc_name": npc["npc_name"], "dominance": npc["dominance"], "cruelty": npc["cruelty"], "position": aff.get("position", "Member")})
                        break
            return members
    except Exception as e:
        logger.error(f"Error getting faction members: {e}", exc_info=True)
        return []

async def _internal_create_resolution_paths_logic(ctx: RunContextWrapper, conflict_id: int, conflict_data: Dict[str, Any]) -> None:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            paths = conflict_data.get("resolution_paths", [])
            async with conn.transaction():
                for path in paths: await conn.execute("INSERT INTO ResolutionPaths (conflict_id, path_id, name, description, approach_type, difficulty, requirements, stakeholders_involved, key_challenges, progress, is_completed) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 0.0, FALSE)", conflict_id, path.get("path_id", f"path_{random.randint(1000,9999)}"), path.get("name", "Unnamed Path"), path.get("description", "A path"), path.get("approach_type", "standard"), path.get("difficulty", 5), json.dumps(path.get("requirements", {})), json.dumps(path.get("stakeholders_involved", [])), json.dumps(path.get("key_challenges", [])))
    except Exception as e:
        logger.error(f"Error creating resolution paths: {e}", exc_info=True)
        raise

async def _internal_create_internal_faction_conflicts_logic(ctx: RunContextWrapper, conflict_id: int, conflict_data: Dict[str, Any]) -> None:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            internal_conflicts = conflict_data.get("internal_faction_conflicts", [])
            async with conn.transaction():
                for internal in internal_conflicts:
                    struggle_id = await conn.fetchval("INSERT INTO InternalFactionConflicts (faction_id, conflict_name, description, primary_npc_id, target_npc_id, prize, approach, public_knowledge, current_phase, progress, parent_conflict_id) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'brewing', 10, $9) RETURNING struggle_id", internal.get("faction_id", 0), internal.get("conflict_name", "Internal Struggle"), internal.get("description", "Power struggle"), internal.get("primary_npc_id", 0), internal.get("target_npc_id", 0), internal.get("prize", "Leadership"), internal.get("approach", "subtle"), internal.get("public_knowledge", False), conflict_id)
                    if "faction_members" in internal:
                        for member in internal["faction_members"]: await conn.execute("INSERT INTO FactionStruggleMembers (struggle_id, npc_id, position, side, standing, loyalty_strength, reason) VALUES ($1, $2, $3, $4, $5, $6, $7)", struggle_id, member.get("npc_id", 0), member.get("position", "Member"), member.get("side", "neutral"), member.get("standing", 50), member.get("loyalty_strength", 50), member.get("reason", ""))
                    if "ideological_differences" in internal:
                        for diff in internal["ideological_differences"]: await conn.execute("INSERT INTO FactionIdeologicalDifferences (struggle_id, issue, incumbent_position, challenger_position) VALUES ($1, $2, $3, $4)", struggle_id, diff.get("issue", ""), diff.get("incumbent_position", ""), diff.get("challenger_position", ""))
    except Exception as e:
        logger.error(f"Error creating internal faction conflicts: {e}", exc_info=True)
        raise

async def _internal_calculate_coup_success_chance_logic(ctx: RunContextWrapper, struggle_id: int, approach: str, supporting_npcs: List[int], resources_committed: Dict[str, int]) -> float:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            struggle_row = await conn.fetchrow("SELECT primary_npc_id, target_npc_id FROM InternalFactionConflicts WHERE struggle_id = $1", struggle_id)
            if not struggle_row: return 0.0
            primary_npc_id, target_npc_id = struggle_row["primary_npc_id"], struggle_row["target_npc_id"]
            challenger, target = await _internal_get_npc_details_logic(ctx, primary_npc_id), await _internal_get_npc_details_logic(ctx, target_npc_id)
            base_chance = 50 + (challenger.get("dominance", 50) - target.get("dominance", 50)) / 5
            base_chance += {"direct": 0, "subtle": 10, "force": -5, "blackmail": 15}.get(approach, 0)
            support_power = sum( (await _internal_get_npc_details_logic(ctx, npc_id)).get("dominance", 50) / 10 for npc_id in supporting_npcs)
            base_chance += min(25, support_power)
            base_chance += min(15, sum(resources_committed.values()) / 10)
            faction_members = await conn.fetch("SELECT loyalty_strength FROM FactionStruggleMembers WHERE struggle_id = $1 AND side = 'incumbent'", struggle_id)
            total_loyalty = sum(row["loyalty_strength"] for row in faction_members)
            base_chance -= min(30, total_loyalty / 20)
            return max(5.0, min(95.0, base_chance))
    except Exception as e:
        logger.error(f"Error calculating coup success chance: {e}", exc_info=True)
        return 30.0

async def _internal_add_resolution_path_logic(ctx: RunContextWrapper, conflict_id: int, path_data: Dict[str, Any]) -> Dict[str, Any]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            exists = await conn.fetchval("SELECT 1 FROM Conflicts WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3", conflict_id, context.user_id, context.conversation_id)
            if not exists: return {"success": False, "error": f"Conflict {conflict_id} not found"}
            path_id = path_data.get("path_id", f"path_{random.randint(1000,9999)}")
            await conn.execute("INSERT INTO ResolutionPaths (conflict_id, path_id, name, description, approach_type, difficulty, requirements, stakeholders_involved, key_challenges, progress, is_completed) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 0.0, FALSE)", conflict_id, path_id, path_data.get("name", "Unnamed"), path_data.get("description", "A path"), path_data.get("approach_type", "standard"), path_data.get("difficulty", 5), json.dumps(path_data.get("requirements", {})), json.dumps(path_data.get("stakeholders_involved", [])), json.dumps(path_data.get("key_challenges", [])))
            await _internal_create_conflict_memory_logic(ctx, conflict_id, f"New resolution path '{path_data.get('name', 'Unnamed')}' added.", significance=6)
            return {"success": True, "conflict_id": conflict_id, "path_id": path_id, "name": path_data.get("name", "Unnamed")}
    except Exception as e:
        logger.error(f"Error adding resolution path to conflict {conflict_id}: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

async def _internal_update_player_involvement_logic(ctx: RunContextWrapper, conflict_id: int, involvement_data: Dict[str, Any]) -> Dict[str, Any]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            # Check if player involvement already exists
            exists = await conn.fetchval("""
                SELECT 1 FROM PlayerConflictInvolvement 
                WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
            """, conflict_id, context.user_id, context.conversation_id)
            
            level = involvement_data.get("involvement_level", "observing")
            faction = involvement_data.get("faction", "neutral")
            money = involvement_data.get("resources_committed", {}).get("money", 0)
            supplies = involvement_data.get("resources_committed", {}).get("supplies", 0)
            influence = involvement_data.get("resources_committed", {}).get("influence", 0)
            actions = json.dumps(involvement_data.get("actions_taken", []))
            manipulated_by = json.dumps(involvement_data.get("manipulated_by")) if involvement_data.get("manipulated_by") else None
            
            # REFACTORED: Validate resources if committing any
            player_name = "Chase"
            if money > 0 or supplies > 0 or influence > 0:
                # Check current resources
                resource_row = await conn.fetchrow("""
                    SELECT money, supplies, influence FROM PlayerResources
                    WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
                """, context.user_id, context.conversation_id, player_name)
                
                if not resource_row:
                    # Create default resources
                    await canon.create_default_resources(ctx, conn, player_name)
                    resource_row = {"money": 100, "supplies": 20, "influence": 10}
                
                # Validate sufficient resources
                if resource_row['money'] < money:
                    return {"success": False, "error": f"Insufficient money. Have: {resource_row['money']}, Need: {money}"}
                if resource_row['supplies'] < supplies:
                    return {"success": False, "error": f"Insufficient supplies. Have: {resource_row['supplies']}, Need: {supplies}"}
                if resource_row['influence'] < influence:
                    return {"success": False, "error": f"Insufficient influence. Have: {resource_row['influence']}, Need: {influence}"}
            
            if exists:
                await conn.execute("""
                    UPDATE PlayerConflictInvolvement 
                    SET involvement_level = $1, faction = $2, money_committed = $3, 
                        supplies_committed = $4, influence_committed = $5, 
                        actions_taken = $6, manipulated_by = $7 
                    WHERE conflict_id = $8 AND user_id = $9 AND conversation_id = $10
                """, level, faction, money, supplies, influence, actions, manipulated_by, 
                conflict_id, context.user_id, context.conversation_id)
            else:
                await conn.execute("""
                    INSERT INTO PlayerConflictInvolvement 
                    (conflict_id, user_id, conversation_id, player_name, involvement_level, 
                     faction, money_committed, supplies_committed, influence_committed, 
                     actions_taken, manipulated_by) 
                    VALUES ($1, $2, $3, 'Player', $4, $5, $6, $7, $8, $9, $10)
                """, conflict_id, context.user_id, context.conversation_id, level, faction, 
                money, supplies, influence, actions, manipulated_by)
            
            # REFACTORED: Log canonical event
            await canon.log_canonical_event(
                ctx, conn,
                f"Player involvement in conflict {conflict_id} changed to {level} with {faction} faction",
                tags=["conflict", "player_involvement", level, faction],
                significance=7
            )
            
            return {
                "success": True, 
                "conflict_id": conflict_id, 
                "involvement_level": level, 
                "faction": faction, 
                "resources_committed": {"money": money, "supplies": supplies, "influence": influence}
            }
    except Exception as e:
        logger.error(f"Error updating player involvement in conflict {conflict_id}: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

async def _internal_add_internal_conflict_logic(ctx: RunContextWrapper, conflict_id: int, internal_conflict_data: Dict[str, Any]) -> Dict[str, Any]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            faction_id, name, desc = internal_conflict_data.get("faction_id", 0), internal_conflict_data.get("conflict_name", "Internal Struggle"), internal_conflict_data.get("description", "Power struggle")
            primary_npc, target_npc = internal_conflict_data.get("primary_npc_id", 0), internal_conflict_data.get("target_npc_id", 0)
            prize, approach, public, phase, progress = internal_conflict_data.get("prize", "Leadership"), internal_conflict_data.get("approach", "subtle"), internal_conflict_data.get("public_knowledge", False), internal_conflict_data.get("current_phase", "brewing"), internal_conflict_data.get("progress", 10)
            struggle_id = await conn.fetchval("INSERT INTO InternalFactionConflicts (faction_id, conflict_name, description, primary_npc_id, target_npc_id, prize, approach, public_knowledge, current_phase, progress, parent_conflict_id) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) RETURNING struggle_id", faction_id, name, desc, primary_npc, target_npc, prize, approach, public, phase, progress, conflict_id)
            faction_name_val = await _internal_get_faction_name_logic(ctx, faction_id)
            primary_name, target_name = await _internal_get_npc_name_logic(ctx, primary_npc), await _internal_get_npc_name_logic(ctx, target_npc)
            if "faction_members" in internal_conflict_data:
                for member in internal_conflict_data["faction_members"]: await conn.execute("INSERT INTO FactionStruggleMembers (struggle_id, npc_id, position, side, standing, loyalty_strength, reason) VALUES ($1, $2, $3, $4, $5, $6, $7)", struggle_id, member.get("npc_id",0), member.get("position","Member"), member.get("side","neutral"), member.get("standing",50), member.get("loyalty_strength",50), member.get("reason",""))
            await _internal_create_conflict_memory_logic(ctx, conflict_id, f"Internal power struggle in {faction_name_val} between {primary_name} and {target_name}.", significance=7)
            return {"success": True, "struggle_id": struggle_id, "conflict_id": conflict_id, "faction_id": faction_id, "faction_name": faction_name_val, "conflict_name": name, "primary_npc_name": primary_name, "target_npc_name": target_name}
    except Exception as e:
        logger.error(f"Error adding internal conflict to conflict {conflict_id}: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

async def _internal_resolve_internal_conflict_logic(ctx: RunContextWrapper, struggle_id: int, resolution_data: Dict[str, Any]) -> Dict[str, Any]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            struggle_row = await conn.fetchrow("SELECT faction_id, conflict_name, primary_npc_id, target_npc_id, parent_conflict_id FROM InternalFactionConflicts WHERE struggle_id = $1", struggle_id)
            if not struggle_row: return {"success": False, "error": f"Struggle {struggle_id} not found"}
            faction_id, parent_conflict_id = struggle_row["faction_id"], struggle_row["parent_conflict_id"]
            faction_name_val = await _internal_get_faction_name_logic(ctx, faction_id)
            winner_npc_id = resolution_data.get("winner_npc_id", struggle_row["primary_npc_id"])
            loser_npc_id = struggle_row["target_npc_id"] if winner_npc_id == struggle_row["primary_npc_id"] else struggle_row["primary_npc_id"]
            winner_name, loser_name = await _internal_get_npc_name_logic(ctx, winner_npc_id), await _internal_get_npc_name_logic(ctx, loser_npc_id)
            res_type, res_desc = resolution_data.get("resolution_type", "negotiated"), resolution_data.get("description", f"Internal conflict in {faction_name_val} resolved.")
            await conn.execute("UPDATE InternalFactionConflicts SET current_phase = 'resolved', progress = 100, resolution_type = $1, resolution_description = $2, winner_npc_id = $3, loser_npc_id = $4, resolved_at = CURRENT_TIMESTAMP WHERE struggle_id = $5", res_type, res_desc, winner_npc_id, loser_npc_id, struggle_id)
            memory_text = f"Internal conflict in {faction_name_val} resolved. {winner_name} won against {loser_name}."
            await _internal_create_conflict_memory_logic(ctx, parent_conflict_id, memory_text, significance=8)
            return {"success": True, "struggle_id": struggle_id, "faction_name": faction_name_val, "winner_name": winner_name, "loser_name": loser_name, "resolution_type": res_type, "description": res_desc}
    except Exception as e:
        logger.error(f"Error resolving internal faction struggle {struggle_id}: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# Additional internal logic functions for the public functions

async def _internal_get_resolution_paths_logic(ctx: RunContextWrapper, conflict_id: int) -> List[Dict[str, Any]]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
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
                path['requirements'] = json.loads(path['requirements']) if path['requirements'] else {}
                path['stakeholders_involved'] = json.loads(path['stakeholders_involved']) if path['stakeholders_involved'] else []
                path['key_challenges'] = json.loads(path['key_challenges']) if path['key_challenges'] else []
                paths.append(path)
            
            return paths
    except Exception as e:
        logger.error(f"Error getting resolution paths: {e}", exc_info=True)
        return []

async def _internal_update_conflict_progress_logic(ctx: RunContextWrapper, conflict_id: int, progress_increment: float) -> Dict[str, Any]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            async with conn.transaction():
                current_progress = await conn.fetchval("""
                    SELECT progress FROM Conflicts
                    WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
                """, conflict_id, context.user_id, context.conversation_id)
                
                if current_progress is None:
                    return {"success": False, "error": "Conflict not found"}
                
                new_progress = min(100.0, current_progress + progress_increment)
                
                await conn.execute("""
                    UPDATE Conflicts
                    SET progress = $1, updated_at = CURRENT_TIMESTAMP
                    WHERE conflict_id = $2 AND user_id = $3 AND conversation_id = $4
                """, new_progress, conflict_id, context.user_id, context.conversation_id)
                
                # Check for phase advancement
                await _internal_check_conflict_advancement_logic(ctx, conflict_id)
                
                return {
                    "success": True,
                    "conflict_id": conflict_id,
                    "old_progress": current_progress,
                    "new_progress": new_progress
                }
    except Exception as e:
        logger.error(f"Error updating conflict progress: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

async def _internal_get_active_conflicts_logic(ctx: RunContextWrapper) -> List[Dict[str, Any]]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            conflicts_rows = await conn.fetch("""
                SELECT conflict_id, conflict_name, conflict_type, description,
                       progress, phase, start_day, estimated_duration
                FROM Conflicts
                WHERE user_id = $1 AND conversation_id = $2 AND is_active = TRUE
                ORDER BY start_day DESC
            """, context.user_id, context.conversation_id)
            
            return [dict(row) for row in conflicts_rows]
    except Exception as e:
        logger.error(f"Error getting active conflicts: {e}", exc_info=True)
        return []

async def _internal_update_stakeholder_status_logic(ctx: RunContextWrapper, conflict_id: int, npc_id: int, status: StakeholderStatusUpdate) -> Dict[str, Any]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            # Build update query dynamically based on provided fields
            update_fields = []
            values = []
            param_count = 1
            
            for field, value in status.dict(exclude_none=True).items():
                update_fields.append(f"{field} = ${param_count}")
                values.append(value)
                param_count += 1
            
            if not update_fields:
                return {"success": False, "error": "No fields to update"}
            
            values.extend([conflict_id, npc_id])
            
            await conn.execute(f"""
                UPDATE ConflictStakeholders
                SET {', '.join(update_fields)}
                WHERE conflict_id = ${param_count} AND npc_id = ${param_count + 1}
            """, *values)
            
            return {
                "success": True,
                "conflict_id": conflict_id,
                "npc_id": npc_id,
                "updated_fields": list(status.dict(exclude_none=True).keys())
            }
    except Exception as e:
        logger.error(f"Error updating stakeholder status: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

async def _internal_get_player_involvement_logic(ctx: RunContextWrapper, conflict_id: int) -> Dict[str, Any]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            involvement_row = await conn.fetchrow("""
                SELECT involvement_level, faction, money_committed, 
                       supplies_committed, influence_committed, actions_taken, 
                       manipulated_by
                FROM PlayerConflictInvolvement
                WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
            """, conflict_id, context.user_id, context.conversation_id)
            
            if not involvement_row:
                return {
                    "involved": False,
                    "conflict_id": conflict_id
                }
            
            result = dict(involvement_row)
            result['involved'] = True
            result['conflict_id'] = conflict_id
            result['actions_taken'] = json.loads(result['actions_taken']) if result['actions_taken'] else []
            result['manipulated_by'] = json.loads(result['manipulated_by']) if result['manipulated_by'] else None
            
            return result
    except Exception as e:
        logger.error(f"Error getting player involvement: {e}", exc_info=True)
        return {"involved": False, "conflict_id": conflict_id, "error": str(e)}

async def _internal_get_conflict_details_logic(ctx: RunContextWrapper, conflict_id: int) -> Dict[str, Any]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            conflict_row = await conn.fetchrow("""
                SELECT conflict_name, conflict_type, description, progress, phase,
                       start_day, estimated_duration, success_rate, outcome, 
                       resolution_description, is_active
                FROM Conflicts
                WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
            """, conflict_id, context.user_id, context.conversation_id)
            
            if not conflict_row:
                return {"error": "Conflict not found"}
            
            details = dict(conflict_row)
            details['conflict_id'] = conflict_id
            
            # Get stakeholders count
            stakeholder_count = await conn.fetchval("""
                SELECT COUNT(*) FROM ConflictStakeholders
                WHERE conflict_id = $1
            """, conflict_id)
            details['stakeholder_count'] = stakeholder_count
            
            # Get resolution paths count
            paths_count = await conn.fetchval("""
                SELECT COUNT(*) FROM ResolutionPaths
                WHERE conflict_id = $1
            """, conflict_id)
            details['resolution_paths_count'] = paths_count
            
            # Get completed paths count
            completed_paths = await conn.fetchval("""
                SELECT COUNT(*) FROM ResolutionPaths
                WHERE conflict_id = $1 AND is_completed = TRUE
            """, conflict_id)
            details['completed_paths_count'] = completed_paths
            
            return details
    except Exception as e:
        logger.error(f"Error getting conflict details: {e}", exc_info=True)
        return {"error": str(e)}

async def _internal_get_conflict_stakeholders_logic(ctx: RunContextWrapper, conflict_id: int) -> List[Dict[str, Any]]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            stakeholders_rows = await conn.fetch("""
                SELECT cs.npc_id, cs.faction_id, cs.faction_name, cs.faction_position,
                       cs.public_motivation, cs.private_motivation, cs.desired_outcome,
                       cs.involvement_level, cs.alliances, cs.rivalries, 
                       cs.leadership_ambition, cs.faction_standing, 
                       cs.willing_to_betray_faction,
                       n.npc_name, n.dominance, n.cruelty
                FROM ConflictStakeholders cs
                JOIN NPCStats n ON cs.npc_id = n.npc_id 
                    AND n.user_id = $2 AND n.conversation_id = $3
                WHERE cs.conflict_id = $1
                ORDER BY cs.involvement_level DESC
            """, conflict_id, context.user_id, context.conversation_id)
            
            stakeholders = []
            for row in stakeholders_rows:
                stakeholder = dict(row)
                stakeholder['alliances'] = json.loads(stakeholder['alliances']) if stakeholder['alliances'] else {}
                stakeholder['rivalries'] = json.loads(stakeholder['rivalries']) if stakeholder['rivalries'] else {}
                stakeholders.append(stakeholder)
            
            return stakeholders
    except Exception as e:
        logger.error(f"Error getting conflict stakeholders: {e}", exc_info=True)
        return []

async def _internal_get_player_manipulation_attempts_logic(ctx: RunContextWrapper, conflict_id: int) -> List[Dict[str, Any]]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            attempts_rows = await conn.fetch("""
                SELECT pma.attempt_id, pma.npc_id, pma.manipulation_type, 
                       pma.content, pma.goal, pma.leverage_used, pma.intimacy_level,
                       pma.success, pma.player_response, pma.is_resolved,
                       pma.created_at, n.npc_name
                FROM PlayerManipulationAttempts pma
                JOIN NPCStats n ON pma.npc_id = n.npc_id 
                    AND n.user_id = $2 AND n.conversation_id = $3
                WHERE pma.conflict_id = $1 AND pma.user_id = $2 
                    AND pma.conversation_id = $3
                ORDER BY pma.created_at DESC
            """, conflict_id, context.user_id, context.conversation_id)
            
            attempts = []
            for row in attempts_rows:
                attempt = dict(row)
                attempt['goal'] = json.loads(attempt['goal']) if isinstance(attempt['goal'], str) else attempt['goal']
                attempt['leverage_used'] = json.loads(attempt['leverage_used']) if isinstance(attempt['leverage_used'], str) else attempt['leverage_used']
                attempts.append(attempt)
            
            return attempts
    except Exception as e:
        logger.error(f"Error getting player manipulation attempts: {e}", exc_info=True)
        return []

async def _internal_generate_conflict_logic(ctx: RunContextWrapper, conflict_type: Optional[str] = None) -> Dict[str, Any]:
    context = ctx.context
    try:
        # Get current day
        current_day = await _internal_get_current_day_logic(ctx)
        
        # Get available NPCs
        available_npcs = await _internal_get_available_npcs_logic(ctx)
        
        if len(available_npcs) < 3:
            return {
                "success": False,
                "error": "Not enough NPCs available for conflict. Need at least 3 introduced NPCs."
            }
        
        # Select NPCs for the conflict
        selected_npcs = random.sample(available_npcs, min(6, len(available_npcs)))
        
        # Determine conflict type if not specified
        if not conflict_type:
            conflict_types = ["minor", "standard", "major"]
            weights = [0.4, 0.4, 0.2]  # More likely to get minor/standard conflicts
            conflict_type = random.choices(conflict_types, weights=weights)[0]
        
        # Generate conflict details
        conflict_details = await _internal_generate_conflict_details_logic(
            ctx, conflict_type, selected_npcs, current_day
        )
        
        # Create the conflict
        conflict_id = await _internal_create_conflict_record_logic(
            ctx, conflict_details, current_day
        )
        
        if not conflict_id:
            return {"success": False, "error": "Failed to create conflict record"}
        
        # Create stakeholders
        await _internal_create_stakeholders_logic(
            ctx, conflict_id, conflict_details, selected_npcs
        )
        
        # Create resolution paths
        await _internal_create_resolution_paths_logic(
            ctx, conflict_id, conflict_details
        )
        
        # Create internal faction conflicts if applicable
        if "internal_faction_conflicts" in conflict_details:
            await _internal_create_internal_faction_conflicts_logic(
                ctx, conflict_id, conflict_details
            )
        
        # Generate player manipulation attempts
        await _internal_generate_player_manipulation_attempts_logic(
            ctx, conflict_id, selected_npcs
        )
        
        return {
            "success": True,
            "conflict_id": conflict_id,
            "conflict_name": conflict_details.get("conflict_name", "Unnamed Conflict"),
            "conflict_type": conflict_type,
            "description": conflict_details.get("description", ""),
            "stakeholders": len(selected_npcs),
            "resolution_paths": len(conflict_details.get("resolution_paths", []))
        }
    except Exception as e:
        logger.error(f"Error generating conflict: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

async def _internal_get_internal_conflicts_logic(ctx: RunContextWrapper, conflict_id: int) -> List[Dict[str, Any]]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            internal_rows = await conn.fetch("""
                SELECT struggle_id, faction_id, conflict_name, description,
                       primary_npc_id, target_npc_id, prize, approach,
                       public_knowledge, current_phase, progress
                FROM InternalFactionConflicts
                WHERE parent_conflict_id = $1
                ORDER BY struggle_id
            """, conflict_id)
            
            internal_conflicts = []
            for row in internal_rows:
                conflict = dict(row)
                
                # Get faction name
                conflict['faction_name'] = await _internal_get_faction_name_logic(ctx, conflict['faction_id'])
                
                # Get NPC names
                conflict['primary_npc_name'] = await _internal_get_npc_name_logic(ctx, conflict['primary_npc_id'])
                conflict['target_npc_name'] = await _internal_get_npc_name_logic(ctx, conflict['target_npc_id'])
                
                internal_conflicts.append(conflict)
            
            return internal_conflicts
    except Exception as e:
        logger.error(f"Error getting internal conflicts: {e}", exc_info=True)
        return []

async def _internal_get_current_day_logic(ctx: RunContextWrapper) -> int:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            current_day = await conn.fetchval("""
                SELECT current_day FROM GameState
                WHERE user_id = $1 AND conversation_id = $2
            """, context.user_id, context.conversation_id)
            
            return current_day if current_day is not None else 1
    except Exception as e:
        logger.error(f"Error getting current day: {e}", exc_info=True)
        return 1

async def _internal_get_available_npcs_logic(ctx: RunContextWrapper) -> List[Dict[str, Any]]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            npc_rows = await conn.fetch("""
                SELECT npc_id, npc_name, dominance, cruelty, closeness, 
                       trust, respect, intensity, sex, faction_affiliations
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2 AND introduced = TRUE
                ORDER BY dominance DESC
            """, context.user_id, context.conversation_id)
            
            npcs = []
            for row in npc_rows:
                npc = dict(row)
                # Parse faction affiliations if stored as JSON string
                if isinstance(npc['faction_affiliations'], str):
                    try:
                        npc['faction_affiliations'] = json.loads(npc['faction_affiliations'])
                    except:
                        npc['faction_affiliations'] = []
                npcs.append(npc)
            
            return npcs
    except Exception as e:
        logger.error(f"Error getting available NPCs: {e}", exc_info=True)
        return []

async def _internal_get_npc_relationship_with_player_logic(ctx: RunContextWrapper, npc_id: int) -> Dict[str, Any]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            relationship_row = await conn.fetchrow("""
                SELECT closeness, trust, respect, intensity
                FROM NPCStats
                WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
            """, npc_id, context.user_id, context.conversation_id)
            
            if not relationship_row:
                return {
                    "npc_id": npc_id,
                    "closeness": 0,
                    "trust": 0,
                    "respect": 0,
                    "intensity": 0
                }
            
            result = dict(relationship_row)
            result['npc_id'] = npc_id
            
            return result
    except Exception as e:
        logger.error(f"Error getting NPC relationship: {e}", exc_info=True)
        return {
            "npc_id": npc_id,
            "closeness": 0,
            "trust": 0,
            "respect": 0,
            "intensity": 0,
            "error": str(e)
        }

async def _internal_generate_conflict_details_logic(ctx: RunContextWrapper, conflict_type: str, stakeholder_npcs: List[Dict[str, Any]], current_day: int) -> Dict[str, Any]:
    context = ctx.context
    
    # Create a detailed prompt for ChatGPT
    npc_descriptions = []
    for npc in stakeholder_npcs:
        desc = f"- {npc['npc_name']}: Dominance {npc.get('dominance', 50)}, Cruelty {npc.get('cruelty', 20)}"
        if npc.get('faction_affiliations'):
            affiliations = npc['faction_affiliations']
            if affiliations and len(affiliations) > 0:
                desc += f", Affiliated with {affiliations[0].get('faction_name', 'Unknown')}"
        npc_descriptions.append(desc)
    
    prompt = f"""
    Generate a {conflict_type} conflict for a femdom-themed game.
    
    Current game day: {current_day}
    
    Available NPCs:
    {chr(10).join(npc_descriptions)}
    
    Create a conflict that:
    1. Involves power dynamics and femdom themes
    2. Has clear stakes and consequences
    3. Provides multiple resolution paths
    4. Could involve faction politics or personal rivalries
    5. Fits the {conflict_type} severity level
    
    Provide the response in JSON format with:
    - conflict_name: A descriptive name
    - description: 2-3 sentence description
    - resolution_paths: Array of at least 3 different approaches
    - stakeholders: Array mapping NPCs to roles and motivations
    - internal_faction_conflicts: Optional array of faction power struggles
    """
    
    response = await get_chatgpt_response(context.conversation_id, "conflict_generation", prompt)
    
    # Parse the response
    if response and "function_args" in response:
        conflict_data = response["function_args"]
    else:
        # Fallback to default conflict structure
        conflict_data = {
            "conflict_name": f"Power Struggle on Day {current_day}",
            "conflict_type": conflict_type,
            "description": f"A {conflict_type} conflict emerges between powerful women vying for control.",
            "estimated_duration": {"minor": 3, "standard": 5, "major": 7, "catastrophic": 10}.get(conflict_type, 5),
            "resolution_paths": [
                {
                    "name": "Diplomatic Resolution",
                    "description": "Negotiate a peaceful solution",
                    "approach_type": "social",
                    "difficulty": 5
                },
                {
                    "name": "Domination Path",
                    "description": "Submit to the stronger party",
                    "approach_type": "submission",
                    "difficulty": 4
                },
                {
                    "name": "Manipulation Route",
                    "description": "Use cunning and leverage",
                    "approach_type": "intrigue",
                    "difficulty": 6
                }
            ]
        }
    
    # Ensure required fields exist
    conflict_data["conflict_type"] = conflict_type
    conflict_data["estimated_duration"] = conflict_data.get("estimated_duration", 
        {"minor": 3, "standard": 5, "major": 7, "catastrophic": 10}.get(conflict_type, 5))
    
    return conflict_data

async def _internal_suggest_manipulation_content_logic(ctx: RunContextWrapper, npc_id: int, conflict_id: int, manipulation_type: str, goal: Dict[str, Any]) -> Dict[str, Any]:
    context = ctx.context
    try:
        # Get NPC details
        npc = await _internal_get_npc_details_logic(ctx, npc_id)
        if not npc:
            return {"error": "NPC not found"}
        
        # Get relationship with player
        relationship = await _internal_get_npc_relationship_with_player_logic(ctx, npc_id)
        
        # Get conflict details for context
        conflict_details = await _internal_get_conflict_details_logic(ctx, conflict_id)
        
        # Generate appropriate content based on manipulation type
        if manipulation_type == "domination":
            content = generate_domination_content(npc, relationship, goal, conflict_details)
        elif manipulation_type == "seduction":
            content = generate_seduction_content(npc, relationship, goal, conflict_details)
        elif manipulation_type == "blackmail":
            # Get secrets if available
            secrets = await _internal_get_stakeholder_secrets_logic(ctx, conflict_id, npc_id)
            content = generate_blackmail_content(npc, relationship, goal, conflict_details, secrets)
        else:
            content = generate_generic_manipulation_content(npc, relationship, goal, conflict_details)
        
        # Generate leverage
        leverage = generate_leverage(npc, relationship, manipulation_type)
        
        # Calculate intimacy level
        intimacy = calculate_intimacy_level(npc, relationship, manipulation_type)
        
        return {
            "npc_id": npc_id,
            "npc_name": npc["npc_name"],
            "manipulation_type": manipulation_type,
            "suggested_content": content,
            "leverage": leverage,
            "intimacy_level": intimacy,
            "success_factors": {
                "npc_dominance": npc.get("dominance", 50),
                "relationship_closeness": relationship.get("closeness", 0),
                "manipulation_fit": 80 if manipulation_type in ["domination", "seduction"] else 60
            }
        }
    except Exception as e:
        logger.error(f"Error suggesting manipulation content: {e}", exc_info=True)
        return {"error": str(e)}

async def _internal_analyze_manipulation_potential_logic(ctx: RunContextWrapper, npc_id: int, player_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    context = ctx.context
    try:
        # Get NPC details
        npc = await _internal_get_npc_details_logic(ctx, npc_id)
        if not npc:
            return {"error": "NPC not found"}
        
        # Get relationship
        relationship = await _internal_get_npc_relationship_with_player_logic(ctx, npc_id)
        
        # Get player stats if not provided
        if not player_stats:
            player_stats = await _internal_get_player_stats_logic(ctx)
        
        # Analyze potential for different manipulation types
        analysis = {
            "npc_id": npc_id,
            "npc_name": npc["npc_name"],
            "manipulation_types": {}
        }
        
        # Domination potential
        dom_score = npc.get("dominance", 50) - player_stats.get("willpower", 50)
        dom_score += 20 if npc.get("sex", "female") == "female" else -10
        analysis["manipulation_types"]["domination"] = {
            "score": max(0, min(100, 50 + dom_score)),
            "factors": {
                "npc_dominance": npc.get("dominance", 50),
                "player_willpower": player_stats.get("willpower", 50),
                "gender_dynamic": "favorable" if npc.get("sex", "female") == "female" else "unfavorable"
            }
        }
        
        # Seduction potential
        sed_score = relationship.get("closeness", 0) * 0.5 + relationship.get("intensity", 0) * 0.3
        sed_score -= player_stats.get("mental_resilience", 50) * 0.2
        analysis["manipulation_types"]["seduction"] = {
            "score": max(0, min(100, sed_score)),
            "factors": {
                "relationship_closeness": relationship.get("closeness", 0),
                "relationship_intensity": relationship.get("intensity", 0),
                "player_resilience": player_stats.get("mental_resilience", 50)
            }
        }
        
        # Blackmail potential
        black_score = npc.get("cruelty", 20) * 0.7
        black_score += 30 if relationship.get("trust", 0) < 30 else 0
        analysis["manipulation_types"]["blackmail"] = {
            "score": max(0, min(100, black_score)),
            "factors": {
                "npc_cruelty": npc.get("cruelty", 20),
                "low_trust": relationship.get("trust", 0) < 30
            }
        }
        
        # Overall assessment
        best_type = max(analysis["manipulation_types"].items(), key=lambda x: x[1]["score"])
        analysis["recommended_type"] = best_type[0]
        analysis["success_likelihood"] = best_type[1]["score"]
        
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing manipulation potential: {e}", exc_info=True)
        return {"error": str(e)}

async def _internal_track_story_beat_logic(ctx: RunContextWrapper, conflict_id: int, path_id: str, beat_description: str, involved_npcs: List[int], progress_value: float) -> Dict[str, Any]:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            async with conn.transaction():
                # Update path progress
                current_progress = await conn.fetchval("""
                    SELECT progress FROM ResolutionPaths
                    WHERE conflict_id = $1 AND path_id = $2
                """, conflict_id, path_id)
                
                if current_progress is None:
                    return {"error": "Resolution path not found"}
                
                new_progress = min(100.0, current_progress + progress_value)
                is_completed = new_progress >= 100.0
                
                await conn.execute("""
                    UPDATE ResolutionPaths
                    SET progress = $1, is_completed = $2
                    WHERE conflict_id = $3 AND path_id = $4
                """, new_progress, is_completed, conflict_id, path_id)
                
                # Create story beat record
                beat_id = await conn.fetchval("""
                    INSERT INTO StoryBeats
                    (conflict_id, path_id, description, involved_npcs, 
                     progress_contribution, user_id, conversation_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING beat_id
                """, conflict_id, path_id, beat_description, json.dumps(involved_npcs),
                progress_value, context.user_id, context.conversation_id)
                
                # Update overall conflict progress
                avg_progress = await conn.fetchval("""
                    SELECT AVG(progress) FROM ResolutionPaths
                    WHERE conflict_id = $1
                """, conflict_id)
                
                await conn.execute("""
                    UPDATE Conflicts
                    SET progress = $1
                    WHERE conflict_id = $2 AND user_id = $3 AND conversation_id = $4
                """, avg_progress, conflict_id, context.user_id, context.conversation_id)
                
                # Create memory event
                await _internal_create_conflict_memory_logic(
                    ctx, conflict_id, 
                    f"Story development: {beat_description[:100]}...",
                    significance=6
                )
                
                # Check for conflict advancement
                await _internal_check_conflict_advancement_logic(ctx, conflict_id)
                
                return {
                    "beat_id": beat_id,
                    "conflict_id": conflict_id,
                    "path_id": path_id,
                    "path_progress": new_progress,
                    "path_completed": is_completed,
                    "overall_progress": avg_progress
                }
    except Exception as e:
        logger.error(f"Error tracking story beat: {e}", exc_info=True)
        return {"error": str(e)}

async def _internal_initiate_faction_power_struggle_logic(ctx: RunContextWrapper, conflict_id: int, faction_id: int, challenger_npc_id: int, target_npc_id: int, prize: str, approach: str, is_public: bool = False) -> Dict[str, Any]:
    context = ctx.context
    try:
        # Generate struggle details
        struggle_details = await _internal_generate_struggle_details_logic(
            ctx, faction_id, challenger_npc_id, target_npc_id, prize, approach
        )
        
        # Create internal conflict
        internal_conflict_data = {
            "faction_id": faction_id,
            "conflict_name": struggle_details["conflict_name"],
            "description": struggle_details["description"],
            "primary_npc_id": challenger_npc_id,
            "target_npc_id": target_npc_id,
            "prize": prize,
            "approach": approach,
            "public_knowledge": is_public,
            "faction_members": struggle_details.get("faction_members", []),
            "ideological_differences": struggle_details.get("ideological_differences", [])
        }
        
        result = await _internal_add_internal_conflict_logic(
            ctx, conflict_id, internal_conflict_data
        )
        
        return result
    except Exception as e:
        logger.error(f"Error initiating faction power struggle: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# Now add all the function_tool decorated functions with strict_mode=False

@function_tool(strict_mode=False)
async def get_resolution_paths(ctx: RunContextWrapper, conflict_id: int) -> List[Dict[str, Any]]:
    """Get all resolution paths for a specific conflict."""
    return await _internal_get_resolution_paths_logic(ctx, conflict_id)

@function_tool(strict_mode=False)
@track_performance("update_conflict_progress")
async def update_conflict_progress(ctx: RunContextWrapper, conflict_id: int, progress_increment: float) -> Dict[str, Any]:
    """Update the progress of a conflict."""
    return await _internal_update_conflict_progress_logic(ctx, conflict_id, progress_increment)

@function_tool(strict_mode=False)
async def get_active_conflicts(ctx: RunContextWrapper) -> List[Dict[str, Any]]:
    """Get all active conflicts for the current user and conversation."""
    return await _internal_get_active_conflicts_logic(ctx)

@function_tool(strict_mode=False)
async def update_stakeholder_status(ctx: RunContextWrapper, conflict_id: int, npc_id: int, status: StakeholderStatusUpdate) -> Dict[str, Any]:
    """Update the status of a stakeholder in a conflict."""
    return await _internal_update_stakeholder_status_logic(ctx, conflict_id, npc_id, status)

@function_tool(strict_mode=False)
async def get_player_involvement(ctx: RunContextWrapper, conflict_id: int) -> Dict[str, Any]:
    """Get player's involvement in a specific conflict."""
    return await _internal_get_player_involvement_logic(ctx, conflict_id)

@function_tool(strict_mode=False)
async def get_conflict_details(ctx: RunContextWrapper, conflict_id: int) -> Dict[str, Any]:
    """Get detailed information about a specific conflict."""
    return await _internal_get_conflict_details_logic(ctx, conflict_id)

@function_tool(strict_mode=False)
async def get_conflict_stakeholders(ctx: RunContextWrapper, conflict_id: int) -> List[Dict[str, Any]]:
    """Get all stakeholders for a specific conflict."""
    return await _internal_get_conflict_stakeholders_logic(ctx, conflict_id)

@function_tool(strict_mode=False)
async def get_player_manipulation_attempts(ctx: RunContextWrapper, conflict_id: int) -> List[Dict[str, Any]]:
    """Get all manipulation attempts targeted at the player for a specific conflict."""
    return await _internal_get_player_manipulation_attempts_logic(ctx, conflict_id)

@function_tool(strict_mode=False)
async def generate_conflict(ctx: RunContextWrapper, conflict_type: Optional[str] = None) -> Dict[str, Any]:
    """Generate a new conflict with stakeholders and resolution paths."""
    return await _internal_generate_conflict_logic(ctx, conflict_type)

@function_tool(strict_mode=False)
async def get_internal_conflicts(ctx: RunContextWrapper, conflict_id: int) -> List[Dict[str, Any]]:
    """Get internal faction conflicts for a specific conflict."""
    return await _internal_get_internal_conflicts_logic(ctx, conflict_id)

@function_tool
async def get_current_day(ctx: RunContextWrapper) -> int:
    """Get the current in-game day."""
    return await _internal_get_current_day_logic(ctx)

@function_tool(strict_mode=False)
async def get_available_npcs(ctx: RunContextWrapper) -> List[Dict[str, Any]]:
    """Get available NPCs that could be involved in conflicts."""
    return await _internal_get_available_npcs_logic(ctx)

@function_tool(strict_mode=False)
async def get_npc_relationship_with_player(ctx: RunContextWrapper, npc_id: int) -> Dict[str, Any]:
    """Get an NPC's relationship with the player."""
    return await _internal_get_npc_relationship_with_player_logic(ctx, npc_id)

@function_tool(strict_mode=False)
async def generate_conflict_details(ctx: RunContextWrapper, conflict_type: str, stakeholder_npcs: List[Dict[str, Any]], current_day: int) -> Dict[str, Any]:
    """Generate conflict details using the AI."""
    return await _internal_generate_conflict_details_logic(ctx, conflict_type, stakeholder_npcs, current_day)

@function_tool(strict_mode=False)
async def create_manipulation_attempt(ctx: RunContextWrapper, conflict_id: int, npc_id: int, manipulation_type: str, content: str, goal: Dict[str, Any], leverage_used: Dict[str, Any], intimacy_level: int = 0) -> Dict[str, Any]:
    """Create a manipulation attempt by an NPC targeted at the player."""
    return await _internal_create_manipulation_attempt_logic(ctx, conflict_id, npc_id, manipulation_type, content, goal, leverage_used, intimacy_level)

@function_tool(strict_mode=False)
async def resolve_manipulation_attempt(ctx: RunContextWrapper, attempt_id: int, success: bool, player_response: str) -> Dict[str, Any]:
    """Resolve a manipulation attempt by the player."""
    return await _internal_resolve_manipulation_attempt_logic(ctx, attempt_id, success, player_response)

@function_tool(strict_mode=False)
async def suggest_manipulation_content(ctx: RunContextWrapper, npc_id: int, conflict_id: int, manipulation_type: str, goal: Dict[str, Any]) -> Dict[str, Any]:
    """Suggest manipulation content for an NPC."""
    return await _internal_suggest_manipulation_content_logic(ctx, npc_id, conflict_id, manipulation_type, goal)

@function_tool(strict_mode=False)
async def analyze_manipulation_potential(ctx: RunContextWrapper, npc_id: int, player_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Analyze an NPC's potential to manipulate the player."""
    return await _internal_analyze_manipulation_potential_logic(ctx, npc_id, player_stats)

@function_tool(strict_mode=False)
async def track_story_beat(ctx: RunContextWrapper, conflict_id: int, path_id: str, beat_description: str, involved_npcs: List[int], progress_value: float) -> Dict[str, Any]:
    """Track a story beat for a resolution path, advancing progress."""
    return await _internal_track_story_beat_logic(ctx, conflict_id, path_id, beat_description, involved_npcs, progress_value)

@function_tool(strict_mode=False)
async def resolve_conflict(ctx: RunContextWrapper, conflict_id: int) -> Dict[str, Any]:
    """Resolve a conflict and apply consequences."""
    return await _internal_resolve_conflict_logic(ctx, conflict_id)

@function_tool(strict_mode=False)
async def initiate_faction_power_struggle(ctx: RunContextWrapper, conflict_id: int, faction_id: int, challenger_npc_id: int, target_npc_id: int, prize: str, approach: str, is_public: bool = False) -> Dict[str, Any]:
    """Initiate a power struggle within a faction."""
    return await _internal_initiate_faction_power_struggle_logic(ctx, conflict_id, faction_id, challenger_npc_id, target_npc_id, prize, approach, is_public)

@function_tool(strict_mode=False)
async def attempt_faction_coup(ctx: RunContextWrapper, struggle_id: int, approach: str, supporting_npcs: List[int], resources_committed: Dict[str, int]) -> Dict[str, Any]:
    """Attempt a coup within a faction to forcefully resolve a power struggle."""
    return await _internal_attempt_faction_coup_logic(ctx, struggle_id, approach, supporting_npcs, resources_committed)

@function_tool(strict_mode=False)
async def add_conflict_to_narrative(ctx: RunContextWrapper, narrative_text: str) -> Dict[str, Any]:
    """OpenAI Agent Tool: Analyzes narrative text to identify and add conflicts."""
    return await _internal_add_conflict_to_narrative_logic(ctx, narrative_text)

@function_tool(strict_mode=False)
async def get_npc_details(ctx: RunContextWrapper, npc_id: int) -> Dict[str, Any]:
    """Get details for an NPC."""
    return await _internal_get_npc_details_logic(ctx, npc_id)

@function_tool
async def get_npc_name(ctx: RunContextWrapper, npc_id: int) -> str:
    """Get an NPC's name by ID."""
    return await _internal_get_npc_name_logic(ctx, npc_id)

@function_tool
async def get_faction_name(ctx: RunContextWrapper, faction_id: int) -> str:
    """Get a faction's name by ID."""
    return await _internal_get_faction_name_logic(ctx, faction_id)

@function_tool(strict_mode=False)
async def get_player_stats(ctx: RunContextWrapper) -> Dict[str, Any]:
    """Get player stats."""
    return await _internal_get_player_stats_logic(ctx)

@function_tool(strict_mode=False)
async def get_stakeholder_secrets(ctx: RunContextWrapper, conflict_id: int, npc_id: int) -> List[Dict[str, Any]]:
    """Get secrets for a stakeholder in a conflict."""
    return await _internal_get_stakeholder_secrets_logic(ctx, conflict_id, npc_id)

@function_tool
async def check_stakeholder_manipulates_player(ctx: RunContextWrapper, conflict_id: int, npc_id: int) -> bool:
    """Check if a stakeholder has manipulation attempts against the player."""
    return await _internal_check_stakeholder_manipulates_player_logic(ctx, conflict_id, npc_id)

@function_tool
async def create_conflict_memory(ctx: RunContextWrapper, conflict_id: int, memory_text: str, significance: int = 5) -> int:
    """Create a memory event for a conflict."""
    return await _internal_create_conflict_memory_logic(ctx, conflict_id, memory_text, significance)

@function_tool
async def check_conflict_advancement(ctx: RunContextWrapper, conflict_id: int) -> None:
    """Check if a conflict should advance to the next phase."""
    await _internal_check_conflict_advancement_logic(ctx, conflict_id)

@function_tool(strict_mode=False)
async def generate_struggle_details(ctx: RunContextWrapper, faction_id: int, challenger_npc_id: int, target_npc_id: int, prize: str, approach: str) -> Dict[str, Any]:
    """Generate details for a faction power struggle."""
    return await _internal_generate_struggle_details_logic(ctx, faction_id, challenger_npc_id, target_npc_id, prize, approach)

@function_tool(strict_mode=False)
async def get_faction_members(ctx: RunContextWrapper, faction_id: int) -> List[Dict[str, Any]]:
    """Get members of a faction."""
    return await _internal_get_faction_members_logic(ctx, faction_id)

@function_tool
async def extract_npcs_from_narrative(ctx: RunContextWrapper, narrative_text: str) -> List[int]:
    """Extract NPC IDs mentioned in a narrative text."""
    return await _internal_extract_npcs_from_narrative_logic(ctx, narrative_text)

@function_tool(strict_mode=False)
async def create_conflict_record(ctx: RunContextWrapper, conflict_data: Dict[str, Any], current_day: int) -> int:
    """Create a conflict record in the database."""
    return await _internal_create_conflict_record_logic(ctx, conflict_data, current_day)

@function_tool(strict_mode=False)
async def create_stakeholders(ctx: RunContextWrapper, conflict_id: int, conflict_data: Dict[str, Any], stakeholder_npcs: List[Dict[str, Any]]) -> None:
    """Create stakeholders for a conflict."""
    await _internal_create_stakeholders_logic(ctx, conflict_id, conflict_data, stakeholder_npcs)

@function_tool(strict_mode=False)
async def create_resolution_paths(ctx: RunContextWrapper, conflict_id: int, conflict_data: Dict[str, Any]) -> None:
    """Create resolution paths for a conflict."""
    await _internal_create_resolution_paths_logic(ctx, conflict_id, conflict_data)

@function_tool(strict_mode=False)
async def create_internal_faction_conflicts(ctx: RunContextWrapper, conflict_id: int, conflict_data: Dict[str, Any]) -> None:
    """Create internal faction conflicts for a main conflict."""
    await _internal_create_internal_faction_conflicts_logic(ctx, conflict_id, conflict_data)

@function_tool(strict_mode=False)
async def generate_player_manipulation_attempts(ctx: RunContextWrapper, conflict_id: int, stakeholder_npcs: List[Dict[str, Any]]) -> None:
    """Generate manipulation attempts targeted at the player."""
    await _internal_generate_player_manipulation_attempts_logic(ctx, conflict_id, stakeholder_npcs)

@function_tool(strict_mode=False)
async def calculate_coup_success_chance(ctx: RunContextWrapper, struggle_id: int, approach: str, supporting_npcs: List[int], resources_committed: Dict[str, int]) -> float:
    """Calculate the success chance of a coup attempt."""
    return await _internal_calculate_coup_success_chance_logic(ctx, struggle_id, approach, supporting_npcs, resources_committed)

@function_tool(strict_mode=False)
async def add_resolution_path(ctx: RunContextWrapper, conflict_id: int, path_data: Dict[str, Any]) -> Dict[str, Any]:
    """Add a new resolution path to an existing conflict."""
    return await _internal_add_resolution_path_logic(ctx, conflict_id, path_data)

@function_tool(strict_mode=False)
async def update_player_involvement(ctx: RunContextWrapper, conflict_id: int, involvement_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update player's involvement in a conflict."""
    return await _internal_update_player_involvement_logic(ctx, conflict_id, involvement_data)

@function_tool(strict_mode=False)
async def add_internal_conflict(ctx: RunContextWrapper, conflict_id: int, internal_conflict_data: Dict[str, Any]) -> Dict[str, Any]:
    """Add an internal faction conflict to a main conflict."""
    return await _internal_add_internal_conflict_logic(ctx, conflict_id, internal_conflict_data)

@function_tool(strict_mode=False)
async def resolve_internal_conflict(ctx: RunContextWrapper, struggle_id: int, resolution_data: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve an internal faction conflict."""
    return await _internal_resolve_internal_conflict_logic(ctx, struggle_id, resolution_data)

# Governance System Integration
@function_tool(strict_mode=False)
async def register_with_governance(ctx: RunContextWrapper, user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Register conflict system with governance system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Registration result
    """
    from nyx.integrate import get_central_governance
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

# Helper function content generation functions remain unchanged
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
