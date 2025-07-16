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
import asyncio
from functools import partial
from typing import Dict, List, Any, Optional, Union, Tuple
import hashlib
from collections import defaultdict
from pydantic import BaseModel, Field
from agents import (
    Agent, Runner, trace, function_tool, 
    RunContextWrapper, handoff, ModelSettings,
    Handoff, RunConfig, FunctionTool
)
from logic.chatgpt_integration import (
get_chatgpt_response,
get_openai_client
)

from db.connection import get_db_connection_context
from lore.core import canon
from logic.resource_management import ResourceManager
from logic.conflict_system.conflict_agents import get_relationship_status, get_manipulation_leverage
from logic.chatgpt_integration import get_chatgpt_response
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from context.context_performance import track_performance

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Constants for Magic Numbers
# -------------------------------------------------------
# Note: Ensure all JSON columns in database are JSONB type for proper indexing
DEFAULT_PLAYER_NAME = "Chase"  # Move to settings or fetch from PlayerProfile
MANIPULATION_DOM_THRESHOLD = 70
MANIPULATION_CLOSENESS_THRESHOLD = 70
MANIPULATION_CRUELTY_THRESHOLD = 70
MANIPULATION_TRIGGER_CHANCE = 0.7
COUP_BASE_CHANCE = 50
COUP_APPROACH_MODIFIERS = {"direct": 0, "subtle": 10, "force": -5, "blackmail": 15}
COUP_SUPPORT_POWER_CAP = 25
COUP_RESOURCE_CAP = 15
COUP_LOYALTY_CAP = 30
COUP_MIN_CHANCE = 5.0
COUP_MAX_CHANCE = 95.0
MAX_ACTIVE_CONFLICTS = 3  # Consider pulling from campaign_settings
MIN_NPCS_FOR_CONFLICT = 3
DEFAULT_CONFLICT_STAKEHOLDERS = 6
OBEDIENCE_CHANGE_RANGE = (2, 5)
DEPENDENCY_CHANGE_RANGE = (1, 3)
WILLPOWER_CHANGE_RANGE = (2, 4)
CONFIDENCE_CHANGE_RANGE = (1, 3)
CORRUPTION_COUP_SUCCESS = 3
CONFIDENCE_COUP_SUCCESS = 5
RESILIENCE_COUP_FAILURE = 4
RUNNER_TIMEOUT_SECONDS = 30
DAYS_PER_MONTH = 30

# Impact level mapping
CONFLICT_IMPACT_LEVELS = {
    "minor": 1,
    "standard": 2,
    "major": 3,
    "catastrophic": 4
}

CONFLICT_SUCCESS_RATES = {
    "minor": 0.75,
    "standard": 0.5,
    "major": 0.25,
    "catastrophic": 0.1
}

CONFLICT_DURATIONS = {
    "minor": 3,
    "standard": 5,
    "major": 7,
    "catastrophic": 10
}

# -------------------------------------------------------
# Dynamic Content Generation Agents
# -------------------------------------------------------

manipulation_content_generator = Agent(
    name="Manipulation Content Generator",
    instructions="""
You generate manipulation dialogue/content for NPCs in a dark-erotic femdom RPG.
Given manipulation context, create compelling, psychologically realistic content.

Input JSON includes:
- manipulation_type: domination/seduction/blackmail/coercion/bribery
- npc_name, dominance, cruelty
- relationship: closeness, trust, respect
- goal: faction, involvement_level, specific_actions
- leverage: type, description, strength

Return JSON with:
{
  "content": "The actual manipulation dialogue/narration (1-3 sentences)",
  "intensity": 0-10,
  "psychological_approach": "brief description of the approach"
}

Keep content suggestive but not explicit. Focus on power dynamics and psychological manipulation.
""",
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.8)
)

conflict_details_generator = Agent(
    name="Conflict Details Generator",
    instructions="""
Generate detailed conflict scenarios for a femdom-themed narrative game.

Input includes:
- conflict_type: minor/standard/major/catastrophic
- current_day: game day number
- npcs: list of available NPCs with stats
- themes: current narrative themes

Return JSON:
{
  "conflict_name": "Descriptive name",
  "description": "2-3 sentence overview",
  "estimated_duration": days as integer,
  "resolution_paths": [
    {
      "name": "Path name",
      "description": "How to resolve",
      "approach_type": "social/direct/intrigue/submission",
      "difficulty": 1-10,
      "key_challenges": ["challenge1", "challenge2"]
    }
  ],
  "stakeholders": [
    {
      "npc_name": "Name",
      "role": "Their role in conflict",
      "public_motivation": "What they claim to want",
      "private_motivation": "What they really want",
      "faction": "a/b/neutral"
    }
  ],
  "femdom_elements": ["element1", "element2"]
}
""",
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.9)
)

struggle_details_generator = Agent(
    name="Faction Struggle Generator",
    instructions="""
Generate internal faction power struggle details for narrative conflicts.

Input includes:
- faction_name, faction_id
- challenger_name, challenger_stats
- target_name, target_stats
- prize: what they're fighting over
- approach: how the challenge is made
- faction_members: list of members

Return JSON:
{
  "conflict_name": "Power struggle name",
  "description": "2-3 sentence description",
  "ideological_differences": [
    {
      "issue": "What they disagree on",
      "incumbent_position": "Current leader's stance",
      "challenger_position": "Challenger's stance"
    }
  ],
  "faction_dynamics": "Description of how this affects the faction",
  "potential_outcomes": ["outcome1", "outcome2", "outcome3"]
}
""",
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.85)
)

consequence_generator = Agent(
    name="Conflict Consequence Generator",
    instructions="""
Generate consequences and rewards for resolved conflicts in a dark fantasy game.

Input includes:
- conflict_type, outcome, player_involvement, player_faction
- resolution_style: how it was resolved
- impact_level: 1-4

Return JSON:
{
  "narrative_consequences": [
    {
      "description": "What happens as a result",
      "scope": "personal/local/regional/global",
      "duration": "temporary/permanent"
    }
  ],
  "stat_changes": {
    "stat_name": change_value
  },
  "relationship_changes": [
    {
      "entity": "who is affected",
      "change_type": "improved/worsened/complicated",
      "magnitude": 1-5
    }
  ],
  "rewards": [
    {
      "type": "item/perk/special",
      "name": "Reward name",
      "description": "What it does",
      "rarity": "common/uncommon/rare/unique",
      "theme_alignment": "Which themes it reinforces"
    }
  ]
}
""",
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.9)
)

narrative_hook_generator = Agent(
    name="Narrative Conflict Hook Generator",
    instructions="""
Analyze narrative text to identify potential conflicts for a femdom-themed game.

Input is narrative text to analyze.

Return JSON:
{
  "conflict_potential": true/false,
  "conflict_type": "minor/standard/major",
  "conflict_name": "Suggested name",
  "description": "Potential conflict description",
  "key_tensions": ["tension1", "tension2"],
  "involved_characters": ["character names mentioned"],
  "femdom_themes": ["theme1", "theme2"],
  "suggested_factions": {
    "faction_a": "Description",
    "faction_b": "Description"
  }
}
""",
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.85)
)

# NEW AGENTS based on suggestions

smart_entity_extractor = Agent(
    name="Smart Entity Extractor",
    instructions="""
Extract character names, locations, and factions from narrative text.
Given narrative text and current campaign roster, identify entities intelligently.

Input JSON:
{
  "narrative_text": "the text to analyze",
  "known_npcs": [{"npc_id": X, "npc_name": "Name"}],
  "known_locations": ["location1", "location2"],
  "known_factions": [{"faction_id": X, "faction_name": "Name"}]
}

Return JSON:
{
  "npc_ids": [list of matched NPC IDs],
  "new_npcs": [{"name": "Name", "context": "how they appear", "suggested_role": "merchant/noble/etc"}],
  "location_refs": ["mentioned locations"],
  "faction_refs": [faction IDs mentioned],
  "confidence_notes": "any ambiguities"
}

Be smart about titles vs names (e.g., "The Baron" vs "Baron Harwick").
""",
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.3)
)

coup_odds_estimator = Agent(
    name="Coup Odds Estimator",
    instructions="""
Calculate coup success chance with nuanced factors for a political intrigue game.

Input includes full struggle context:
- challenger/target stats and relationships
- supporting NPCs and their loyalties
- resources committed
- approach type
- recent events/blackmail/leverage
- faction morale and rumors

Return JSON:
{
  "base_chance": percentage,
  "modifiers": [
    {"factor": "description", "impact": +/-X}
  ],
  "final_chance": percentage,
  "rationale": "one-line explanation of key factors",
  "wild_cards": ["unpredictable elements"]
}
""",
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.6)
)

resolution_path_generator = Agent(
    name="Resolution Path Generator",
    instructions="""
Generate unique conflict resolution paths based on stakeholder psychology and themes.

Input:
- conflict summary
- stakeholder profiles (personality, goals, relationships)
- active themes/kinks
- faction dynamics

Return JSON with 3-5 paths:
{
  "paths": [
    {
      "name": "Path name",
      "description": "Detailed approach",
      "approach_type": "social/direct/intrigue/submission/seduction",
      "theme_alignment": ["matching themes/kinks"],
      "risk_level": "low/medium/high/extreme",
      "fetish_hooks": ["specific elements"],
      "key_npcs": [npc_ids who feature prominently],
      "requirements": {"stat": value},
      "unique_mechanics": "what makes this path special"
    }
  ]
}

Make each path feel distinct and tied to the specific conflict.
""",
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.9)
)

secret_generator = Agent(
    name="Secret Generator",
    instructions="""
Generate personalized secrets for NPCs in conflicts.

Input:
- npc profile (name, personality, role)
- current conflict stakes
- faction affiliations
- relationship web

Return JSON:
{
  "secret_type": "affair/betrayal/identity/crime/desire/weakness",
  "content": "The specific secret (1-2 sentences)",
  "risk_if_revealed": "consequences if exposed",
  "who_might_know": ["hints about who could discover this"],
  "blackmail_value": 1-10,
  "narrative_hooks": ["story potential"]
}

Secrets should feel personal and create future drama.
""",
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.85)
)

item_skinner = Agent(
    name="Item Skinner",
    instructions="""
Create flavorful names and descriptions for mechanically-defined items.

Input:
- item_type and base stats
- rarity level
- thematic context (conflict, resolution style, setting)
- any fetish/kink alignment

Return JSON:
{
  "name": "Evocative item name",
  "description": "1-2 sentence lore that references the conflict",
  "appearance": "what it looks like",
  "flavor_text": "optional quote or legend"
}

Keep descriptions concise but atmospheric. Reference the conflict origins when possible.
""",
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.9)
)

result_narrator = Agent(
    name="Result Narrator",
    instructions="""
Write dramatic summaries for conflict outcomes and major events.

Input:
- raw outcome data (who won/lost, how)
- key participants and their roles
- resolution method used
- consequences applied

Return JSON:
{
  "summary": "1-2 paragraph dramatic narrative",
  "headline": "short dramatic title",
  "key_moment": "the pivotal scene in 1 sentence",
  "mood": "triumphant/tragic/pyrrhic/bittersweet"
}

Write in active voice with vivid language. Make players feel the weight of events.
""",
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.8)
)

phase_transition_bark_generator = Agent(
    name="Phase Transition Narrator",
    instructions="""
Generate atmospheric scene descriptions when conflicts change phases.

Input:
- conflict name and type
- old phase -> new phase
- key stakeholders
- current tensions

Return JSON:
{
  "scene": "1-2 sentence atmospheric description",
  "rumor": "what people are saying",
  "atmosphere_shift": "how the mood changes",
  "foreshadowing": "hint of what's to come"
}

Examples: "Rumors spread through taverns...", "Tension crackles in the air as..."
Keep it diegetic and immersive.
""",
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.85)
)

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
                # Generate dynamic stakeholders from NPCs
                stakeholders_to_insert = []
                for i, npc in enumerate(stakeholder_npcs):
                    # Generate motivations based on NPC personality
                    dominance = npc.get("dominance", 50)
                    cruelty = npc.get("cruelty", 20)
                    
                    if dominance > MANIPULATION_DOM_THRESHOLD:
                        public_motivation = f"{npc.get('npc_name', 'They')} seeks to control the outcome and assert dominance."
                        private_motivation = f"{npc.get('npc_name', 'They')} wants to crush any opposition and expand their power."
                    elif cruelty > 60:
                        public_motivation = f"{npc.get('npc_name', 'They')} claims to want justice for past wrongs."
                        private_motivation = f"{npc.get('npc_name', 'They')} actually seeks revenge and to cause suffering."
                    elif dominance < 30:
                        public_motivation = f"{npc.get('npc_name', 'They')} hopes for a peaceful resolution."
                        private_motivation = f"{npc.get('npc_name', 'They')} wants to avoid confrontation and find a protector."
                    else:
                        public_motivation = f"{npc.get('npc_name', 'They')} wants to see the conflict resolved fairly."
                        private_motivation = f"{npc.get('npc_name', 'They')} seeks to gain advantage through careful maneuvering."
                    
                    stakeholder = {
                        "npc_id": npc["npc_id"],
                        "npc_name": npc.get("npc_name", f"NPC {npc['npc_id']}"),
                        "public_motivation": public_motivation,
                        "private_motivation": private_motivation,
                        "desired_outcome": "Favorable resolution aligned with their goals",
                        "involvement_level": max(3, 10 - i * 2),  # Decreasing involvement
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
                    
                    # Handle secrets dynamically
                    if "secrets" in sh or (npc.get("cruelty", 20) > 50 and random.random() > 0.7):
                        # Generate secrets for cruel or important NPCs
                        secrets = sh.get("secrets", [])
                        
                        # If no secrets provided, generate some
                        if not secrets and random.random() > 0.5:
                            try:
                                # Get other stakeholders for potential secret targets
                                # FIXED: Define stakeholder_npcs_details properly
                                other_stakeholders = [s for s in stakeholder_npcs if s["npc_id"] != npc_id]
                                
                                secret_payload = json.dumps({
                                    "npc_profile": {
                                        "name": npc.get("npc_name"),
                                        "personality": {
                                            "dominance": npc.get("dominance", 50),
                                            "cruelty": npc.get("cruelty", 20)
                                        },
                                        "role": sh.get("faction_position", "Member")
                                    },
                                    "current_conflict_stakes": {
                                        "public_goal": sh.get("public_motivation"),
                                        "private_goal": sh.get("private_motivation"),
                                        "conflict_type": conflict_data.get("conflict_type", "standard")
                                    },
                                    "faction_affiliations": [faction_name] if faction_name else [],
                                    "relationship_web": [{"name": s["npc_name"], "role": "fellow stakeholder"} for s in other_stakeholders[:3]]
                                }, ensure_ascii=False)
                                
                                result = await Runner.run(
                                    starting_agent=secret_generator,
                                    input=secret_payload,
                                    timeout_seconds=RUNNER_TIMEOUT_SECONDS
                                )
                                
                                # Handle both string and dict outputs from Runner
                                if isinstance(result.output, str):
                                    generated_secret = json.loads(result.output.strip())
                                else:
                                    generated_secret = result.output
                                
                                secrets = [{
                                    "secret_type": generated_secret["secret_type"],
                                    "content": generated_secret["content"],
                                    "risk_level": generated_secret.get("blackmail_value", 5)
                                }]
                                
                            except Exception as e:
                                logger.error(f"Secret generation failed: {e}")
                                secrets = [{
                                    "secret_type": "personal",
                                    "content": "Has hidden ambitions regarding the conflict"
                                }]
                        
                        for secret in secrets:
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
        raise RuntimeError(f"Failed to create stakeholders: {str(e)}")

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
        raise RuntimeError(f"Failed to create manipulation attempt: {str(e)}")

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
                player_name = await _internal_get_player_name_logic(ctx)
                stat_changes = {}
                
                if success:
                    # Player succumbed to manipulation
                    obedience_change = random.randint(*OBEDIENCE_CHANGE_RANGE)
                    dependency_change = random.randint(*DEPENDENCY_CHANGE_RANGE)
                    
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
                    willpower_change = random.randint(*WILLPOWER_CHANGE_RANGE)
                    confidence_change = random.randint(*CONFIDENCE_CHANGE_RANGE)
                    
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
        raise RuntimeError(f"Failed to resolve manipulation attempt: {str(e)}")

# REFACTORED: Create conflict memory using canon
async def _internal_create_conflict_memory_logic(ctx: RunContextWrapper, conflict_id: int, memory_text: str, significance: int = 5) -> int:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            # Get conflict details for richer memory
            conflict_row = await conn.fetchrow("""
                SELECT conflict_name, conflict_type, phase
                FROM Conflicts
                WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
            """, conflict_id, context.user_id, context.conversation_id)
            
            # Try to generate a more narrative memory if it's significant
            if significance >= 6 and conflict_row:
                try:
                    # Use the result narrator for significant memories
                    memory_payload = json.dumps({
                        "raw_outcome_data": {
                            "event": memory_text,
                            "conflict_name": conflict_row["conflict_name"],
                            "conflict_type": conflict_row["conflict_type"],
                            "phase": conflict_row["phase"]
                        },
                        "key_participants": {},
                        "resolution_method": "ongoing",
                        "consequences_applied": "memory recorded"
                    }, ensure_ascii=False)
                    
                    narration_result = await Runner.run(
                        starting_agent=result_narrator,
                        input=memory_payload,
                        timeout_seconds=RUNNER_TIMEOUT_SECONDS
                    )
                    # Handle both string and dict outputs
                    if isinstance(narration_result.output, str):
                        narration = json.loads(narration_result.output.strip())
                    else:
                        narration = narration_result.output
                    
                    # Use the key moment as the memory text
                    enhanced_memory = narration.get("key_moment", memory_text)
                    memory_text = enhanced_memory if len(enhanced_memory) < 200 else memory_text
                    
                except Exception as e:
                    logger.debug(f"Memory enhancement failed, using original: {e}")
            
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
        (npc.get("dominance", 0) > MANIPULATION_DOM_THRESHOLD or 
         npc.get("relationship_with_player", {}).get("closeness", 0) > MANIPULATION_CLOSENESS_THRESHOLD)
    ]
    
    if not eligible_npcs: 
        return
    
    manipulation_types = ["domination", "blackmail", "seduction", "coercion", "bribery"]
    involvement_levels = ["observing", "participating", "leading"]
    factions = ["a", "b", "neutral"]
    
    for npc in eligible_npcs[:2]:  # Limit to 2 NPCs
        if random.random() > MANIPULATION_TRIGGER_CHANCE: 
            continue
            
        # Determine manipulation type based on NPC traits
        if npc.get("dominance", 0) > 80:
            manip_type = "domination"
        elif npc.get("cruelty", 0) > MANIPULATION_CRUELTY_THRESHOLD:
            manip_type = "blackmail"
        elif npc.get("relationship_with_player", {}).get("closeness", 0) > 80:
            manip_type = "seduction"
        else:
            manip_type = random.choice(manipulation_types)
        
        involvement = random.choice(involvement_levels)
        faction_choice = random.choice(factions)
        
        # Generate content using dynamic agent
        goal = {
            "faction": faction_choice, 
            "involvement_level": involvement
        }
        
        # Get manipulation content dynamically - USE DYNAMIC TEMPLATES INSTEAD
        content = await generate_manipulation_content_from_templates(
            ctx, npc, npc.get("relationship_with_player", {}), goal, {}, manip_type
        )
        
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
            # Start transaction for atomicity
            async with conn.transaction():
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
                
                # FIXED: Validate and deduct resources BEFORE calculating success chance
                player_name = await _internal_get_player_name_logic(ctx)
                if sum(resources_committed.values()) > 0:
                    # Check current resources first
                    resource_row = await conn.fetchrow("""
                        SELECT money, supplies, influence FROM PlayerResources
                        WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
                    """, context.user_id, context.conversation_id, player_name)
                    
                    if not resource_row:
                        await canon.create_default_resources(ctx, conn, player_name)
                        resource_row = {"money": 100, "supplies": 20, "influence": 10}
                    
                    # Validate sufficient resources
                    if resource_row['money'] < resources_committed.get("money", 0):
                        return {"error": "Insufficient money", "current": resource_row['money']}
                    if resource_row['supplies'] < resources_committed.get("supplies", 0):
                        return {"error": "Insufficient supplies", "current": resource_row['supplies']}
                    if resource_row['influence'] < resources_committed.get("influence", 0):
                        return {"error": "Insufficient influence", "current": resource_row['influence']}
                    
                    # Deduct resources NOW, before success calculation
                    await canon.adjust_player_resource(
                        ctx, conn, player_name, "money", -resources_committed.get("money", 0), 
                        "coup_attempt", f"Committed to coup in struggle {struggle_id}"
                    )
                    await canon.adjust_player_resource(
                        ctx, conn, player_name, "supplies", -resources_committed.get("supplies", 0),
                        "coup_attempt", f"Committed to coup in struggle {struggle_id}"
                    )
                    await canon.adjust_player_resource(
                        ctx, conn, player_name, "influence", -resources_committed.get("influence", 0),
                        "coup_attempt", f"Committed to coup in struggle {struggle_id}"
                    )
                
                # Calculate success chance AFTER resource deduction
                success_chance = await _internal_calculate_coup_success_chance_logic(
                    ctx, struggle_id, approach, supporting_npcs, resources_committed
                )
                success = random.random() * 100 <= success_chance
                
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
                
                stat_changes = {}
                
                if success:
                    # Update struggle status
                    await conn.execute("""
                        UPDATE InternalFactionConflicts 
                        SET current_phase = $1, progress = 100, resolved_at = CURRENT_TIMESTAMP 
                        WHERE struggle_id = $2
                    """, "resolved", struggle_id)
                    
                    # Generate dramatic result narrative
                    try:
                        narrative_payload = json.dumps({
                            "raw_outcome_data": {
                                "winner": primary_name,
                                "loser": target_name,
                                "method": approach,
                                "supporters": len(supporting_npcs),
                                "faction": faction_name
                            },
                            "key_participants": {
                                "challenger": primary_name,
                                "incumbent": target_name,
                                "faction": faction_name
                            },
                            "resolution_method": f"coup via {approach}",
                            "consequences_applied": "leadership change"
                        }, ensure_ascii=False)
                        
                        result_narration = await Runner.run(
                            starting_agent=result_narrator,
                            input=narrative_payload,
                            timeout_seconds=RUNNER_TIMEOUT_SECONDS
                        )
                        # Handle both string and dict outputs
                        if isinstance(result_narration.output, str):
                            narration = json.loads(result_narration.output.strip())
                        else:
                            narration = result_narration.output
                        
                        result = {
                            "outcome": "success",
                            "description": narration.get("summary", f"{primary_name} successfully overthrew {target_name} in {faction_name}."),
                            "headline": narration.get("headline", "Coup Succeeds"),
                            "mood": narration.get("mood", "triumphant")
                        }
                    except Exception as e:
                        logger.error(f"Result narration failed: {e}")
                        result = {
                            "outcome": "success",
                            "description": f"{primary_name} successfully overthrew {target_name} in {faction_name}."
                        }
                    
                    # REFACTORED: Log canonical event
                    await canon.log_canonical_event(
                        ctx, conn,
                        result.get("headline", f"{primary_name}'s coup succeeds"),
                        tags=["conflict", "coup", "success", "faction_politics"],
                        significance=8
                    )
                    
                    # REFACTORED: Update player stats canonically
                    stats_row = await conn.fetchrow("""
                        SELECT corruption, confidence FROM PlayerStats
                        WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
                    """, context.user_id, context.conversation_id, player_name)
                    
                    if stats_row:
                        new_corruption = stats_row['corruption'] + CORRUPTION_COUP_SUCCESS
                        new_confidence = stats_row['confidence'] + CONFIDENCE_COUP_SUCCESS
                    else:
                        await canon.find_or_create_player_stats(ctx, conn, player_name)
                        new_corruption = CORRUPTION_COUP_SUCCESS
                        new_confidence = CONFIDENCE_COUP_SUCCESS
                    
                    await canon.update_player_stat_canonically(
                        ctx, conn, player_name, "corruption", new_corruption,
                        f"Successful coup participation in struggle {struggle_id}"
                    )
                    await canon.update_player_stat_canonically(
                        ctx, conn, player_name, "confidence", new_confidence,
                        f"Successful coup participation in struggle {struggle_id}"
                    )
                    
                    stat_changes = {"corruption": CORRUPTION_COUP_SUCCESS, "confidence": CONFIDENCE_COUP_SUCCESS}
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
                    
                    # Generate dramatic failure narrative
                    try:
                        narrative_payload = json.dumps({
                            "raw_outcome_data": {
                                "winner": target_name,
                                "loser": primary_name,
                                "method": f"failed {approach} coup",
                                "supporters": len(supporting_npcs),
                                "faction": faction_name
                            },
                            "key_participants": {
                                "challenger": primary_name,
                                "incumbent": target_name,
                                "faction": faction_name
                            },
                            "resolution_method": f"failed coup via {approach}",
                            "consequences_applied": "challenger weakened, incumbent strengthened"
                        }, ensure_ascii=False)
                        
                        result_narration = await Runner.run(
                            starting_agent=result_narrator,
                            input=narrative_payload,
                            timeout_seconds=RUNNER_TIMEOUT_SECONDS
                        )
                        # Handle both string and dict outputs
                        if isinstance(result_narration.output, str):
                            narration = json.loads(result_narration.output.strip())
                        else:
                            narration = result_narration.output
                        
                        result = {
                            "outcome": "failure",
                            "description": narration.get("summary", f"{primary_name}'s coup failed. {target_name} remains in power."),
                            "headline": narration.get("headline", "Coup Attempt Failed"),
                            "mood": narration.get("mood", "tragic")
                        }
                    except Exception as e:
                        logger.error(f"Result narration failed: {e}")
                        result = {
                            "outcome": "failure",
                            "description": f"{primary_name}'s coup failed. {target_name} remains in power."
                        }
                    
                    # REFACTORED: Log canonical event
                    await canon.log_canonical_event(
                        ctx, conn,
                        result.get("headline", f"{primary_name}'s coup fails"),
                        tags=["conflict", "coup", "failure", "faction_politics"],
                        significance=8
                    )
                    
                    # REFACTORED: Update player stats canonically
                    stats_row = await conn.fetchrow("""
                        SELECT mental_resilience FROM PlayerStats
                        WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
                    """, context.user_id, context.conversation_id, player_name)
                    
                    if stats_row:
                        new_resilience = stats_row['mental_resilience'] + RESILIENCE_COUP_FAILURE
                    else:
                        await canon.find_or_create_player_stats(ctx, conn, player_name)
                        new_resilience = RESILIENCE_COUP_FAILURE
                    
                    await canon.update_player_stat_canonically(
                        ctx, conn, player_name, "mental_resilience", new_resilience,
                        f"Failed coup participation in struggle {struggle_id}"
                    )
                    
                    stat_changes = {"mental_resilience": RESILIENCE_COUP_FAILURE}
                
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
                
                # Expose rationale to the player
                rationale_display = f"Intelligence suggests {int(success_chance)}% success odds"
                
                return {
                    "coup_id": coup_id, 
                    "struggle_id": struggle_id, 
                    "approach": approach, 
                    "success": success, 
                    "success_chance": success_chance,
                    "rationale": rationale_display,
                    "result": result, 
                    "stat_changes": stat_changes, 
                    "resources": resources
                }
    except Exception as e:
        logger.error(f"Error attempting faction coup: {e}", exc_info=True)
        raise RuntimeError(f"Failed to attempt faction coup: {str(e)}")

# REFACTORED: Create conflict record with canon
async def _internal_create_conflict_record_logic(ctx: RunContextWrapper, conflict_data: Dict[str, Any], current_day: int) -> int:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            conflict_type = conflict_data.get("conflict_type", "standard")
            success_rate = CONFLICT_SUCCESS_RATES.get(conflict_type, 0.5)
            
            conflict_id = await conn.fetchval("""
                INSERT INTO Conflicts 
                (user_id, conversation_id, conflict_name, conflict_type, description, 
                 progress, phase, start_day, estimated_duration, success_rate, 
                 outcome, is_active) 
                VALUES ($1, $2, $3, $4, $5, 0.0, 'brewing', $6, $7, $8, 'pending', TRUE) 
                RETURNING conflict_id
            """, context.user_id, context.conversation_id, 
            conflict_data.get("conflict_name", "Unnamed Conflict"), 
            conflict_type, 
            conflict_data.get("description", "A conflict has emerged"), 
            current_day, 
            conflict_data.get("estimated_duration", CONFLICT_DURATIONS.get(conflict_type, 7)), 
            success_rate)
            
            # REFACTORED: Log canonical event
            await canon.log_canonical_event(
                ctx, conn,
                f"New conflict created: {conflict_data.get('conflict_name', 'Unnamed')} ({conflict_type})",
                tags=["conflict", "creation", conflict_type],
                significance=7
            )
            
            return conflict_id if conflict_id else 0
    except Exception as e:
        logger.error(f"Error creating conflict record: {e}", exc_info=True)
        raise RuntimeError(f"Failed to create conflict record: {str(e)}")

async def get_conflict_consequences(
    user_id: int,
    conversation_id: int,
    *,
    conflict_type: str,
    outcome: str,
    player_involvement: str,
    player_faction: str,
    resolution_style: str,
    completed_paths: list[str] | None = None,
    environment_desc: str = "",
) -> List[Dict[str, Any]]:
    """
    Return a rich consequence package for `resolve_conflict`.

    • **completed_paths** → list of *path names* the player actually finished.
      This lets GPT tailor rewards (e.g. give a seduction perk if the
      “Velvet Submission” path closed).

    Output schema mirrors the expectations of the internal resolver:
      ▶ narrative_consequences – list[dict]
      ▶ stat_changes          – dict[str, int]
      ▶ relationship_changes  – list[dict]
      ▶ rewards               – list[dict]
    """
    completed_paths = completed_paths or []

    try:
        request = json.dumps(
            {
                "env": environment_desc,
                "conflict_type": conflict_type,
                "outcome": outcome,
                "player_involvement": player_involvement,
                "player_faction": player_faction,
                "resolution_style": resolution_style,
                "completed_paths": completed_paths,
            },
            indent=2,
        )
        data = await _gpt_json(  # ← your existing wrapper
            "Design post‑conflict consequence package for a dark fantasy femdom RPG. Return JSON array.",
            request,
        )
        if isinstance(data, list):
            return data  # type: ignore[return-value]
    except Exception as e:
        logger.error("consequence GPT fallback: %s", e)

    # ── minimal static fallback ──────────────────────────────────────────
    return [
        {
            "narrative_consequences": [{
                "description": "Word of the resolution spreads through the city.",
                "scope": "local",
                "duration": "temporary",
            }],
            "stat_changes": {"confidence": 1},
            "relationship_changes": [],
            "rewards": [],
        }
    ]
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
            
            # FIXED: Use larger transaction chunks or increase timeout for big operations
            async with conn.transaction():
                # Set statement timeout for this transaction to 30 seconds
                await conn.execute("SET LOCAL statement_timeout = '30s'")
                
                # Update conflict status
                await conn.execute("""
                    UPDATE Conflicts 
                    SET is_active = FALSE, progress = 100, phase = 'concluded', 
                        outcome = $1, resolution_description = $2, resolved_at = CURRENT_TIMESTAMP 
                    WHERE conflict_id = $3
                """, outcome, description, conflict_id)
                
                # Build resolution style from completed paths
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
                
                primary_style = max(set(resolution_styles), key=resolution_styles.count) if resolution_styles else "neutral"
                
                # Use Dynamic Templates to get consequences
                consequences = await get_conflict_consequences(
                    context.user_id,
                    context.conversation_id,
                    conflict_type=conflict_type,
                    outcome=outcome,
                    player_involvement=player_involvement,
                    player_faction=player_faction,
                    resolution_style=primary_style,
                    completed_paths=[p["name"] for p in completed_paths],
                )
                
                # Apply consequences
                player_name = await _internal_get_player_name_logic(ctx)
                for con in consequences:
                    # Store consequence record
                    await conn.execute("""
                        INSERT INTO ConflictConsequences 
                        (conflict_id, consequence_type, description, affected_entity_type, 
                         affected_entity_id, magnitude, is_permanent) 
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """, conflict_id, con.get("consequence_type", "general"), 
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
                    elif con.get("reward_type") == "item" and "reward_data" in con:
                        item = con["reward_data"]
                        await canon.find_or_create_inventory_item(
                            ctx, conn,
                            item_name=item["name"],
                            player_name=player_name,
                            item_description=item.get("description", ""),
                            item_category=item.get("category", "conflict_reward"),
                            item_properties={
                                "rarity": item.get("rarity", "common"),
                                "resolution_style": primary_style,
                                "source": "conflict_resolution",
                                "conflict_id": conflict_id,
                                "conflict_type": conflict_type
                            },
                            quantity=1,
                            equipped=False
                        )
                    
                    # REFACTORED: Handle perk rewards (store as special inventory items)
                    elif con.get("reward_type") == "perk" and "reward_data" in con:
                        perk = con["reward_data"]
                        await canon.find_or_create_inventory_item(
                            ctx, conn,
                            item_name=f"Perk: {perk['name']}",
                            player_name=player_name,
                            item_description=perk.get("description", ""),
                            item_category="perk",
                            item_properties={
                                "tier": perk.get("tier", 1),
                                "resolution_style": primary_style,
                                "source": "conflict_resolution",
                                "conflict_id": conflict_id,
                                "perk_type": perk.get("category", "conflict_resolution")
                            },
                            quantity=1,
                            equipped=True  # Perks are automatically "equipped"
                        )
                    
                    # REFACTORED: Handle special rewards
                    elif con.get("reward_type") == "special" and "reward_data" in con:
                        special = con["reward_data"]
                        await canon.find_or_create_inventory_item(
                            ctx, conn,
                            item_name=special["name"],
                            player_name=player_name,
                            item_description=special.get("description", ""),
                            item_category="unique_conflict_reward",
                            item_properties={
                                "effect": special.get("effect", ""),
                                "resolution_style": primary_style,
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
        raise RuntimeError(f"Failed to resolve conflict: {str(e)}")

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
            
            # Get known locations and factions for context
            location_rows = await conn.fetch("""
                SELECT DISTINCT current_location FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2 AND current_location IS NOT NULL
            """, context.user_id, context.conversation_id)
            
            faction_rows = await conn.fetch("""
                SELECT faction_id, faction_name FROM Factions
                WHERE faction_id IN (
                    SELECT DISTINCT faction_id FROM ConflictStakeholders
                    WHERE conflict_id IN (
                        SELECT conflict_id FROM Conflicts
                        WHERE user_id = $1 AND conversation_id = $2
                    )
                )
            """, context.user_id, context.conversation_id)
            
            # Use smart entity extractor
            try:
                payload = json.dumps({
                    "narrative_text": narrative_text,
                    "known_npcs": [{"npc_id": row["npc_id"], "npc_name": row["npc_name"]} for row in npc_rows],
                    "known_locations": [row["current_location"] for row in location_rows if row["current_location"]],
                    "known_factions": [{"faction_id": row["faction_id"], "faction_name": row["faction_name"]} for row in faction_rows]
                }, ensure_ascii=False)
                
                result = await Runner.run(
                    starting_agent=smart_entity_extractor,
                    input=payload,
                    timeout_seconds=RUNNER_TIMEOUT_SECONDS
                )
                # Handle both string and dict outputs
                if isinstance(result.output, str):
                    extracted = json.loads(result.output.strip())
                else:
                    extracted = result.output
                
                mentioned_npc_ids = extracted.get("npc_ids", [])
                
                # Handle new NPCs that need creation
                for new_npc in extracted.get("new_npcs", []):
                    npc_id = await canon.find_or_create_npc(
                        ctx, conn,
                        npc_name=new_npc["name"],
                        role=new_npc.get("suggested_role", "mentioned_in_narrative")
                    )
                    mentioned_npc_ids.append(npc_id)
                
                return mentioned_npc_ids
                
            except Exception as e:
                logger.error(f"Smart entity extraction failed, falling back: {e}")
                # FIXED: Fallback with word boundary matching to avoid duplicates
                mentioned_npc_ids = []
                for row in npc_rows:
                    # Use word boundaries to avoid substring matches
                    pattern = r'\b' + re.escape(row["npc_name"]) + r'\b'
                    if re.search(pattern, narrative_text, re.IGNORECASE):
                        mentioned_npc_ids.append(row["npc_id"])
                return mentioned_npc_ids
                
    except Exception as e:
        logger.error(f"Error extracting NPCs from narrative: {e}", exc_info=True)
        return []

# Helper Functions

async def _internal_get_player_name_logic(ctx: RunContextWrapper) -> str:
    """Get player name from PlayerProfile or use default."""
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            player_name = await conn.fetchval("""
                SELECT player_name FROM PlayerProfile
                WHERE user_id = $1 AND conversation_id = $2
                LIMIT 1
            """, context.user_id, context.conversation_id)
            
            return player_name if player_name else DEFAULT_PLAYER_NAME
    except Exception as e:
        logger.debug(f"Could not fetch player name from profile: {e}")
        return DEFAULT_PLAYER_NAME

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

# REFACTORED: Get calendar context from schema instead of Dynamic Templates
async def _internal_get_calendar_context_logic(ctx: RunContextWrapper) -> Dict[str, Any]:
    """
    Get calendar context directly from the database schema.
    
    Returns:
        Dictionary with calendar months, day names, and current date info
    """
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            current_day = await _internal_get_current_day_logic(ctx)
            
            # Get months from database
            month_rows = await conn.fetch("""
                SELECT month_name, month_order 
                FROM Months 
                WHERE user_id = $1 AND conversation_id = $2
                ORDER BY month_order
            """, context.user_id, context.conversation_id)
            
            # Get day names from database
            day_rows = await conn.fetch("""
                SELECT day_name, day_order 
                FROM WeekDays 
                WHERE user_id = $1 AND conversation_id = $2
                ORDER BY day_order
            """, context.user_id, context.conversation_id)
            
            months = [row['month_name'] for row in month_rows] if month_rows else ["Firstmonth", "Secondmonth", "Thirdmonth", "Fourthmonth", "Fifthmonth", "Sixthmonth", "Seventhmonth", "Eighthmonth", "Ninthmonth", "Tenthmonth", "Eleventhmonth", "Twelfthmonth"]
            day_names = [row['day_name'] for row in day_rows] if day_rows else ["Moonday", "Towerday", "Waterday", "Thunderday", "Fireday", "Starday", "Sunday"]
            
            # Calculate current month and day of week
            current_month_index = (current_day - 1) // DAYS_PER_MONTH
            day_of_month = ((current_day - 1) % DAYS_PER_MONTH) + 1
            day_of_week_index = (current_day - 1) % len(day_names)
            
            return {
                "current_day": current_day,
                "months": months,
                "day_names": day_names,
                "current_month": months[current_month_index % len(months)],
                "day_of_month": day_of_month,
                "day_of_week": day_names[day_of_week_index],
                "formatted_date": f"{day_names[day_of_week_index]}, {day_of_month} of {months[current_month_index % len(months)]}"
            }
            
    except Exception as e:
        logger.error(f"Error getting calendar context: {e}")
        return {
            "current_day": await _internal_get_current_day_logic(ctx),
            "formatted_date": f"Day {await _internal_get_current_day_logic(ctx)}"
        }

async def _gpt_json(system: str, user: str, *, model: str = "gpt-4.1-nano") -> Any:
    if get_openai_client is None:
        raise RuntimeError("OpenAI client unavailable – falling back")

    client = get_openai_client()
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.7,
                max_tokens=640,
                response_format={"type": "json_object"},
            )
            return json.loads(r.choices[0].message.content)
        except Exception as e:  # pragma: no cover
            logger.warning("GPT call failed (%s/3): %s", attempt + 1, e)
            await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError("GPT failed after 3 attempts")

_hash = lambda txt: hashlib.sha1(txt.encode()).hexdigest()[:10]

_MANIP_CACHE: Dict[str, List[str]] = {}

async def get_manipulation_lines(
    user_id: int,
    conversation_id: int,
    *,
    npc_id: int,
    manipulation_type: str,
    n: int = 3,
    environment_desc: str = "",
) -> List[str]:
    """Return up to *n* pre‑baked dialogue snippets for the NPC -> player move.

    This helper first checks an in‑memory cache to avoid duplicate GPT calls
    during the same gameplay session.  The cache key is stable per NPC + method
    (not per player), so cross‑conversation reuse is permissible.
    """

    manipulation_type = manipulation_type.lower()
    if manipulation_type not in {"domination", "seduction", "blackmail", "coercion", "bribery"}:
        manipulation_type = "generic"

    cache_key = _hash(f"{npc_id}:{manipulation_type}:{environment_desc}")
    if cache_key in _MANIP_CACHE:
        return _MANIP_CACHE[cache_key][:n]

    # ── GPT fetch (context‑light) ───────────────────────────────────────────
    try:
        payload = json.dumps(
            {
                "npc_id": npc_id,
                "manipulation_type": manipulation_type,
                "environment": environment_desc,
                "max_lines": n,
            },
            indent=2,
        )
        data = await _gpt_json(
            "Generate short, non‑explicit manipulation lines for a dark‑erotic femdom RPG. Return JSON list.",
            payload,
        )
        if isinstance(data, list):
            _MANIP_CACHE[cache_key] = [str(x) for x in data][:n]
            return _MANIP_CACHE[cache_key]
    except Exception as e:
        logger.error("manipulation GPT fallback: %s", e)

    # ── Static fallback – themed but generic ───────────────────────────────
    defaults = {
        "domination": [
            "Kneel—and maybe I'll consider your plea.",
            "Your will *is* mine; accept it.",
            "Obey first, think later, pet.",
        ],
        "seduction": [
            "Imagine the rewards if you stood by my side.",
            "A whisper from me could melt your resolve.",
            "Let desire guide your next decision, darling.",
        ],
        "blackmail": [
            "One word from me and your secret sings.",
            "Your reputation is a fragile thing—handle with care.",
            "I keep confessions; I trade in them too.",
        ],
        "coercion": [
            "Help me—or I help myself to your assets.",
            "Refusal is an expensive luxury here.",
            "Choices are illusions; compliance is real.",
        ],
        "bribery": [
            "Loyalty has a price; I pay in privileges.",
            "A taste of power for a whisper of aid?",
            "Gold glitters, but influence gleams brighter—take both.",
        ],
        "generic": [
            "Stand with me and profit; stand against me and perish.",
            "Align your interests with mine—wisdom wears many shapes.",
            "Help me tip the scales; I will remember your service.",
        ],
    }
    return defaults[manipulation_type][:n]


# Additional helper for generating manipulation content from Dynamic Templates
async def generate_manipulation_content_from_templates(
    ctx: RunContextWrapper,
    npc: Dict[str, Any],
    relationship: Dict[str, Any],
    goal: Dict[str, Any],
    conflict: Dict[str, Any],
    manipulation_type: str,
) -> str:
    """
    Prefer pre‑generated template lines; fall back to GPT if none found.
    """

    try:
        # The helper now expects the full objects – not just IDs.

        # NOTE: we pass the *same* objects we already have so the template
        #       engine can assess dominance, closeness, etc.
        lines = await get_manipulation_lines(
            npc=npc,
            relationship=relationship,
            goal=goal,
            conflict=conflict,
            method=manipulation_type,
            environment_desc="",   # optional campaign summary if you have one
            lines=3,
        )

        if lines:
            return random.choice(lines)

    except Exception as e:                 # template system unavailable → GPT fallback
        logger.warning("Template lines unavailable: %s – falling back to GPT", e)

    # GPT fallback (same implementation you already had)
    return await generate_manipulation_content(
        ctx, npc, relationship, goal, conflict, manipulation_type
    )
# Continue with more internal logic functions...

async def _internal_get_player_stats_logic(ctx: RunContextWrapper) -> Dict[str, Any]:
    context = ctx.context
    try:
        player_name = await _internal_get_player_name_logic(ctx)
        async with get_db_connection_context() as conn:
            stats_row = await conn.fetchrow("""
                SELECT corruption, confidence, willpower, obedience, dependency, lust, mental_resilience, physical_endurance 
                FROM PlayerStats 
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
            """, context.user_id, context.conversation_id, player_name)
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
            conflict_row = await conn.fetchrow("""
                SELECT conflict_name, conflict_type, progress, phase 
                FROM Conflicts 
                WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
            """, conflict_id, context.user_id, context.conversation_id)
            if not conflict_row: return
            
            conflict_name = conflict_row["conflict_name"]
            progress, phase = conflict_row["progress"], conflict_row["phase"]
            thresholds = {"brewing": 30, "active": 60, "climax": 90}
            new_phase = phase
            
            if phase in thresholds and progress >= thresholds[phase]:
                if phase == "brewing": new_phase = "active"
                elif phase == "active": new_phase = "climax"
                elif phase == "climax": new_phase = "resolution"
                
            if new_phase != phase:
                # Get key stakeholders for context
                stakeholder_rows = await conn.fetch("""
                    SELECT n.npc_name, cs.involvement_level
                    FROM ConflictStakeholders cs
                    JOIN NPCStats n ON cs.npc_id = n.npc_id
                    WHERE cs.conflict_id = $1
                    ORDER BY cs.involvement_level DESC
                    LIMIT 3
                """, conflict_id)
                
                key_stakeholders = [row["npc_name"] for row in stakeholder_rows]
                
                # Generate phase transition narrative
                try:
                    payload = json.dumps({
                        "conflict_name": conflict_name,
                        "conflict_type": conflict_row["conflict_type"],
                        "old_phase": phase,
                        "new_phase": new_phase,
                        "key_stakeholders": key_stakeholders,
                        "current_tensions": f"Progress at {progress}%"
                    }, ensure_ascii=False)
                    
                    result = await Runner.run(
                        starting_agent=phase_transition_bark_generator,
                        input=payload,
                        timeout_seconds=RUNNER_TIMEOUT_SECONDS
                    )
                    transition = json.loads(result.output.strip())
                    
                    # Create atmospheric memory
                    memory_text = transition.get("scene", f"Conflict progressed from {phase} to {new_phase}.")
                    
                except Exception as e:
                    logger.error(f"Phase transition narration failed: {e}")
                    memory_text = f"The {conflict_name} has escalated from {phase} to {new_phase}."
                
                async with conn.transaction():
                    await conn.execute("""
                        UPDATE Conflicts 
                        SET phase = $1, updated_at = CURRENT_TIMESTAMP 
                        WHERE conflict_id = $2 AND user_id = $3 AND conversation_id = $4
                    """, new_phase, conflict_id, context.user_id, context.conversation_id)
                    
                    await _internal_create_conflict_memory_logic(ctx, conflict_id, memory_text, significance=7)
                    
    except Exception as e:
        logger.error(f"Error checking conflict advancement: {e}", exc_info=True)

async def _internal_generate_struggle_details_logic(ctx: RunContextWrapper, faction_id: int, challenger_npc_id: int, target_npc_id: int, prize: str, approach: str) -> Dict[str, Any]:
    """Generate struggle details using GPT agent."""
    try:
        # Get names for context
        faction_name = await _internal_get_faction_name_logic(ctx, faction_id)
        challenger_name = await _internal_get_npc_name_logic(ctx, challenger_npc_id)
        target_name = await _internal_get_npc_name_logic(ctx, target_npc_id)
        
        # Get faction members
        members = await _internal_get_faction_members_logic(ctx, faction_id)
        
        # Get NPC stats for better generation
        challenger_stats = await _internal_get_npc_details_logic(ctx, challenger_npc_id)
        target_stats = await _internal_get_npc_details_logic(ctx, target_npc_id)
        
        payload = json.dumps({
            "faction_name": faction_name,
            "faction_id": faction_id,
            "challenger_name": challenger_name,
            "challenger_stats": challenger_stats,
            "target_name": target_name,
            "target_stats": target_stats,
            "prize": prize,
            "approach": approach,
            "faction_members": members
        }, ensure_ascii=False)
        
        result = await Runner.run(
            starting_agent=struggle_details_generator,
            input=payload,
            timeout_seconds=RUNNER_TIMEOUT_SECONDS
        )
        # Handle both string and dict outputs
        if isinstance(result.output, str):
            generated = json.loads(result.output.strip())
        else:
            generated = result.output
        
        # Build faction members list with sides
        faction_members_list = [
            {"npc_id": challenger_npc_id, "position": "Challenger", "side": "challenger"},
            {"npc_id": target_npc_id, "position": "Incumbent", "side": "incumbent"}
        ]
        
        # Assign other members to sides based on relationships
        for member in members:
            if member["npc_id"] == challenger_npc_id or member["npc_id"] == target_npc_id:
                continue
            
            # Use dominance and cruelty to determine allegiance
            side = "neutral"
            if member.get("dominance", 50) > MANIPULATION_DOM_THRESHOLD:
                side = "incumbent"  # Support existing power
            elif member.get("cruelty", 20) > 60:
                side = "challenger"  # Support disruption
            
            faction_members_list.append({
                "npc_id": member["npc_id"],
                "position": member.get("position", "Member"),
                "side": side
            })
        
        return {
            "conflict_name": generated["conflict_name"],
            "description": generated["description"],
            "faction_members": faction_members_list,
            "ideological_differences": generated["ideological_differences"],
            "faction_dynamics": generated.get("faction_dynamics", ""),
            "potential_outcomes": generated.get("potential_outcomes", [])
        }
        
    except Exception as e:
        logger.error(f"Error generating struggle details: {e}")
        # Fallback to simple generation
        faction_name = await _internal_get_faction_name_logic(ctx, faction_id)
        challenger_name = await _internal_get_npc_name_logic(ctx, challenger_npc_id)
        target_name = await _internal_get_npc_name_logic(ctx, target_npc_id)
        
        return {
            "conflict_name": f"Power struggle in {faction_name}",
            "description": f"{challenger_name} challenges {target_name} for {prize} within {faction_name}.",
            "faction_members": [
                {"npc_id": challenger_npc_id, "position": "Challenger", "side": "challenger"},
                {"npc_id": target_npc_id, "position": "Incumbent", "side": "incumbent"}
            ],
            "ideological_differences": [
                {"issue": f"Approach to {prize}", "incumbent_position": f"{target_name}'s way", "challenger_position": f"{challenger_name}'s way"}
            ]
        }

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

async def _internal_calculate_coup_success_chance_logic(ctx: RunContextWrapper, struggle_id: int, approach: str, supporting_npcs: List[int], resources_committed: Dict[str, int]) -> float:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            struggle_row = await conn.fetchrow("SELECT primary_npc_id, target_npc_id, faction_id FROM InternalFactionConflicts WHERE struggle_id = $1", struggle_id)
            if not struggle_row: return 0.0
            
            primary_npc_id, target_npc_id = struggle_row["primary_npc_id"], struggle_row["target_npc_id"]
            faction_id = struggle_row["faction_id"]
            
            # Get detailed information for coup odds estimation
            challenger = await _internal_get_npc_details_logic(ctx, primary_npc_id)
            target = await _internal_get_npc_details_logic(ctx, target_npc_id)
            
            # Get supporter details
            supporters = []
            for npc_id in supporting_npcs:
                supporter = await _internal_get_npc_details_logic(ctx, npc_id)
                supporters.append(supporter)
            
            # Get faction loyalty data
            faction_members = await conn.fetch("""
                SELECT npc_id, side, loyalty_strength, standing 
                FROM FactionStruggleMembers 
                WHERE struggle_id = $1
            """, struggle_id)
            
            # Get any blackmail or leverage
            leverage_data = await conn.fetch("""
                SELECT secret_type, content, target_npc_id 
                FROM StakeholderSecrets 
                WHERE npc_id = $1 AND target_npc_id = $2
            """, primary_npc_id, target_npc_id)
            
            # Get faction morale/recent events
            recent_events = await conn.fetch("""
                SELECT memory_text, significance 
                FROM ConflictMemoryEvents 
                WHERE conflict_id IN (
                    SELECT parent_conflict_id FROM InternalFactionConflicts WHERE struggle_id = $1
                ) 
                ORDER BY id DESC LIMIT 5
            """, struggle_id)
            
            # Use GPT to calculate nuanced odds
            try:
                payload = json.dumps({
                    "challenger": challenger,
                    "target": target,
                    "supporting_npcs": supporters,
                    "resources_committed": resources_committed,
                    "approach": approach,
                    "faction_members": [dict(row) for row in faction_members],
                    "leverage": [dict(row) for row in leverage_data],
                    "recent_events": [dict(row) for row in recent_events],
                    "faction_id": faction_id
                }, ensure_ascii=False)
                
                result = await Runner.run(
                    starting_agent=coup_odds_estimator,
                    input=payload,
                    timeout_seconds=RUNNER_TIMEOUT_SECONDS
                )
                # Handle both string and dict outputs
                if isinstance(result.output, str):
                    estimation = json.loads(result.output.strip())
                else:
                    estimation = result.output
                
                # Log the rationale for transparency
                logger.info(f"Coup odds for struggle {struggle_id}: {estimation['final_chance']}% - {estimation['rationale']}")
                
                return float(estimation['final_chance'])
                
            except Exception as e:
                logger.error(f"GPT coup estimation failed, using fallback: {e}")
                # Fallback to original calculation
                base_chance = COUP_BASE_CHANCE + (challenger.get("dominance", 50) - target.get("dominance", 50)) / 5
                base_chance += COUP_APPROACH_MODIFIERS.get(approach, 0)
                support_power = sum((s.get("dominance", 50) / 10 for s in supporters), 0)
                base_chance += min(COUP_SUPPORT_POWER_CAP, support_power)
                base_chance += min(COUP_RESOURCE_CAP, sum(resources_committed.values()) / 10)
                total_loyalty = sum(row["loyalty_strength"] for row in faction_members if row["side"] == "incumbent")
                base_chance -= min(COUP_LOYALTY_CAP, total_loyalty / 20)
                return max(COUP_MIN_CHANCE, min(COUP_MAX_CHANCE, base_chance))
                
    except Exception as e:
        logger.error(f"Error calculating coup success chance: {e}", exc_info=True)
        return 30.0

# Continue with remaining internal logic functions...

async def _internal_create_resolution_paths_logic(ctx: RunContextWrapper, conflict_id: int, conflict_data: Dict[str, Any]) -> None:
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            # Check if paths already provided
            paths = conflict_data.get("resolution_paths", [])
            
            if not paths:
                # Generate paths dynamically based on conflict and stakeholders
                # Get stakeholder profiles
                stakeholder_rows = await conn.fetch("""
                    SELECT cs.npc_id, n.npc_name, n.dominance, n.cruelty, 
                           cs.public_motivation, cs.private_motivation,
                           cs.faction_id, cs.faction_name
                    FROM ConflictStakeholders cs
                    JOIN NPCStats n ON cs.npc_id = n.npc_id 
                        AND n.user_id = $2 AND n.conversation_id = $3
                    WHERE cs.conflict_id = $1
                """, conflict_id, context.user_id, context.conversation_id)
                
                # Get active themes/kinks
                kink_rows = await conn.fetch("""
                    SELECT kink_type, level FROM UserKinkProfile
                    WHERE user_id = $1 AND level >= 2
                """, context.user_id)
                
                # Get faction dynamics
                faction_info = {}
                for row in stakeholder_rows:
                    if row["faction_id"] and row["faction_id"] not in faction_info:
                        faction_info[row["faction_id"]] = {
                            "name": row["faction_name"],
                            "members": []
                        }
                    if row["faction_id"]:
                        faction_info[row["faction_id"]]["members"].append(row["npc_name"])
                
                try:
                    payload = json.dumps({
                        "conflict_summary": {
                            "name": conflict_data.get("conflict_name", "Unknown Conflict"),
                            "type": conflict_data.get("conflict_type", "standard"),
                            "description": conflict_data.get("description", "")
                        },
                        "stakeholder_profiles": [{
                            "npc_id": row["npc_id"],
                            "name": row["npc_name"],
                            "dominance": row["dominance"],
                            "cruelty": row["cruelty"],
                            "public_goal": row["public_motivation"],
                            "private_goal": row["private_motivation"]
                        } for row in stakeholder_rows],
                        "active_themes": [row["kink_type"] for row in kink_rows],
                        "faction_dynamics": faction_info
                    }, ensure_ascii=False)
                    
                    result = await Runner.run(
                        starting_agent=resolution_path_generator,
                        input=payload,
                        timeout_seconds=RUNNER_TIMEOUT_SECONDS
                    )
                    # Handle both string and dict outputs
                    if isinstance(result.output, str):
                        generated = json.loads(result.output.strip())
                    else:
                        generated = result.output
                    paths = generated.get("paths", [])
                    
                except Exception as e:
                    logger.error(f"Dynamic path generation failed, using defaults: {e}")
                    # Fallback paths
                    paths = [
                        {
                            "path_id": "diplomatic",
                            "name": "Diplomatic Resolution",
                            "description": "Resolve through negotiation.",
                            "approach_type": "social",
                            "difficulty": 5
                        },
                        {
                            "path_id": "direct",
                            "name": "Direct Confrontation",
                            "description": "Resolve through force.",
                            "approach_type": "direct",
                            "difficulty": 7
                        }
                    ]
            
            # FIXED: Use chunking for large batch inserts
            async with conn.transaction():
                # Set timeout for this transaction
                await conn.execute("SET LOCAL statement_timeout = '30s'")
                
                for i, path in enumerate(paths):
                    await conn.execute("""
                        INSERT INTO ResolutionPaths 
                        (conflict_id, path_id, name, description, approach_type, 
                         difficulty, requirements, stakeholders_involved, 
                         key_challenges, progress, is_completed) 
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 0.0, FALSE)
                    """, 
                    conflict_id, 
                    path.get("path_id", f"path_{i}_{random.randint(1000,9999)}"), 
                    path.get("name", "Unnamed Path"), 
                    path.get("description", "A resolution path"), 
                    path.get("approach_type", "standard"), 
                    path.get("difficulty", 5), 
                    json.dumps(path.get("requirements", {})), 
                    json.dumps(path.get("key_npcs", path.get("stakeholders_involved", []))), 
                    json.dumps(path.get("key_challenges", []))
                    )
                    
    except Exception as e:
        logger.error(f"Error creating resolution paths: {e}", exc_info=True)
        raise RuntimeError(f"Failed to create resolution paths: {str(e)}")

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
        raise RuntimeError(f"Failed to create internal faction conflicts: {str(e)}")

# Continue with the rest of the internal logic functions...

async def _internal_get_max_active_conflicts_logic(ctx: RunContextWrapper) -> int:
    """Get max active conflicts from campaign settings or use default."""
    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            max_conflicts = await conn.fetchval("""
                SELECT settings->>'max_active_conflicts' 
                FROM CampaignSettings
                WHERE user_id = $1 AND conversation_id = $2
            """, context.user_id, context.conversation_id)
            
            return int(max_conflicts) if max_conflicts else MAX_ACTIVE_CONFLICTS
    except Exception as e:
        logger.debug(f"Could not fetch max conflicts from settings: {e}")
        return MAX_ACTIVE_CONFLICTS

async def _internal_add_conflict_to_narrative_logic(ctx: RunContextWrapper, narrative_text: str) -> Dict[str, Any]:
    context = ctx.context
    current_day = await _internal_get_current_day_logic(ctx)
    active_conflicts = await _internal_get_active_conflicts_logic(ctx)
    max_conflicts = await _internal_get_max_active_conflicts_logic(ctx)
    
    if len(active_conflicts) >= max_conflicts: 
        return {
            "trigger_conflict": False, 
            "reason": "Too many active conflicts", 
            "existing_conflicts": len(active_conflicts)
        }
    
    # Analyze narrative for conflict potential using GPT agent
    try:
        result = await Runner.run(
            starting_agent=narrative_hook_generator,
            input=narrative_text,
            timeout_seconds=RUNNER_TIMEOUT_SECONDS
        )
        # Handle both string and dict outputs
        if isinstance(result.output, str):
            conflict_analysis = json.loads(result.output.strip())
        else:
            conflict_analysis = result.output
    except Exception as e:
        logger.error(f"Error analyzing narrative for conflicts: {e}")
        # Fallback analysis
        conflict_analysis = {
            "conflict_potential": False,
            "reason": "Analysis failed"
        }
    
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

    # Let the resolution path generator create paths based on the conflict
    # Don't pass default paths - let _internal_create_resolution_paths_logic generate them
    await _internal_create_resolution_paths_logic(ctx, conflict_id, {})
    
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

# Continue with remaining helper functions...

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

# Continue with all remaining functions...

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
        
        if len(available_npcs) < MIN_NPCS_FOR_CONFLICT:
            return {
                "success": False,
                "error": f"Not enough NPCs available for conflict. Need at least {MIN_NPCS_FOR_CONFLICT} introduced NPCs."
            }
        
        # Select NPCs for the conflict
        selected_npcs = random.sample(available_npcs, min(DEFAULT_CONFLICT_STAKEHOLDERS, len(available_npcs)))
        
        # Determine conflict type if not specified
        if not conflict_type:
            conflict_types = ["minor", "standard", "major"]
            weights = [0.4, 0.4, 0.2]  # More likely to get minor/standard conflicts
            conflict_type = random.choices(conflict_types, weights=weights)[0]
        
        # Generate conflict details dynamically
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
    """Generate conflict details using GPT agent."""
    context = ctx.context
    
    try:
        # Create a detailed prompt for the agent
        npc_descriptions = []
        for npc in stakeholder_npcs:
            desc = {
                "npc_name": npc['npc_name'],
                "dominance": npc.get('dominance', 50),
                "cruelty": npc.get('cruelty', 20),
                "faction_affiliations": npc.get('faction_affiliations', [])
            }
            npc_descriptions.append(desc)
        
        # Get current narrative themes
        themes = []  # Could be fetched from context or lore system
        
        payload = json.dumps({
            "conflict_type": conflict_type,
            "current_day": current_day,
            "npcs": npc_descriptions,
            "themes": themes
        }, ensure_ascii=False)
        
        result = await Runner.run(
            starting_agent=conflict_details_generator,
            input=payload,
            timeout_seconds=RUNNER_TIMEOUT_SECONDS
        )
        # Handle both string and dict outputs
        if isinstance(result.output, str):
            conflict_data = json.loads(result.output.strip())
        else:
            conflict_data = result.output
        
        # Ensure all required fields exist
        conflict_data["conflict_type"] = conflict_type
        conflict_data["estimated_duration"] = conflict_data.get("estimated_duration", 
            CONFLICT_DURATIONS.get(conflict_type, 5))
        
        # Map stakeholder NPCs to the generated stakeholder data
        if "stakeholders" in conflict_data:
            for i, stakeholder in enumerate(conflict_data["stakeholders"]):
                if i < len(stakeholder_npcs):
                    stakeholder["npc_id"] = stakeholder_npcs[i]["npc_id"]
                    stakeholder["npc_name"] = stakeholder_npcs[i]["npc_name"]
        
        return conflict_data
        
    except Exception as e:
        logger.error(f"Error generating conflict details: {e}")
        # Fallback to default conflict structure
        # Even in fallback, we'll let the resolution path generator handle paths
        return {
            "conflict_name": f"Power Struggle on Day {current_day}",
            "conflict_type": conflict_type,
            "description": f"A {conflict_type} conflict emerges between powerful women vying for control.",
            "estimated_duration": CONFLICT_DURATIONS.get(conflict_type, 5),
            # Don't include resolution_paths - let them be generated dynamically
        }

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
        
        # Generate appropriate content dynamically
        content = await generate_manipulation_content_from_templates(
            ctx, npc, relationship, goal, conflict_details, manipulation_type
        )
        
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
        # Generate struggle details dynamically
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
            player_name = await _internal_get_player_name_logic(ctx)
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
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, conflict_id, context.user_id, context.conversation_id, player_name, level, faction, 
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

# Dynamic content generation functions

async def generate_manipulation_content(
    ctx: RunContextWrapper,
    npc: Dict[str, Any],
    relationship: Dict[str, Any],
    goal: Dict[str, Any],
    conflict: Dict[str, Any],
    manipulation_type: str,
) -> str:
    """Generate manipulation content dynamically using GPT – with static fallback."""
    try:
        payload = json.dumps({
            "manipulation_type": manipulation_type,
            "npc_name": npc.get("npc_name", ""),
            "dominance": npc.get("dominance", 50),
            "cruelty": npc.get("cruelty", 20),
            "relationship": {
                "closeness": relationship.get("closeness", 30),
                "trust": relationship.get("trust", 30),
                "respect": relationship.get("respect", 30)
            },
            "goal": goal,
            "leverage": {
                "type": manipulation_type,
                "description": f"Using {manipulation_type} tactics",
                "strength": npc.get("dominance", 50) if manipulation_type == "domination" else relationship.get("closeness", 30)
            }
        }, ensure_ascii=False)
        
        result = await Runner.run(
            starting_agent=manipulation_content_generator,
            input=payload,
            timeout_seconds=RUNNER_TIMEOUT_SECONDS
        )
        
        # Handle both string and dict outputs
        if isinstance(result.output, str):
            parsed = json.loads(result.output.strip())
        else:
            parsed = result.output
        
        return parsed["content"]

    except Exception as e:
        logger.error("Error generating manipulation content: %s", e)
        # ── static fallback avoids “None” artefact when goal lacks 'faction' ──
        return (
            f"{npc.get('npc_name', 'She')} attempts to {manipulation_type} you regarding "
            f"{goal.get('faction', 'this matter')}."
        )

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
async def update_player_involvement(
    ctx: RunContextWrapper,
    conflict_id: int,
    involvement_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create or update the player's stake in a conflict **and** debit resources.
    """

    context = ctx.context
    try:
        async with get_db_connection_context() as conn:
            # ── 1. Gather inputs ───────────────────────────────────────────
            level     = involvement_data.get("involvement_level", "observing")
            faction   = involvement_data.get("faction", "neutral")
            money     = involvement_data.get("resources_committed", {}).get("money", 0)
            supplies  = involvement_data.get("resources_committed", {}).get("supplies", 0)
            influence = involvement_data.get("resources_committed", {}).get("influence", 0)
            actions   = json.dumps(involvement_data.get("actions_taken", []))
            manipulated_by = (
                json.dumps(involvement_data["manipulated_by"])
                if involvement_data.get("manipulated_by")
                else None
            )

            player_name = await _internal_get_player_name_logic(ctx)

            # ── 2. Validate & deduct resources ────────────────────────────
            if any((money, supplies, influence)):
                resource_row = await conn.fetchrow(
                    """
                    SELECT money, supplies, influence
                      FROM PlayerResources
                     WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
                    """,
                    context.user_id,
                    context.conversation_id,
                    player_name,
                )

                if not resource_row:
                    await canon.create_default_resources(ctx, conn, player_name)
                    resource_row = {"money": 100, "supplies": 20, "influence": 10}

                # Guard‑clauses
                if resource_row["money"] < money:
                    return {"success": False, "error": f"Insufficient money ({resource_row['money']} available)"}
                if resource_row["supplies"] < supplies:
                    return {"success": False, "error": f"Insufficient supplies ({resource_row['supplies']} available)"}
                if resource_row["influence"] < influence:
                    return {"success": False, "error": f"Insufficient influence ({resource_row['influence']} available)"}

                # Now **deduct** the resources
                await canon.adjust_player_resource(ctx, conn, player_name, "money",     -money,
                                                   "conflict_involvement", f"Conflict {conflict_id}")
                await canon.adjust_player_resource(ctx, conn, player_name, "supplies",  -supplies,
                                                   "conflict_involvement", f"Conflict {conflict_id}")
                await canon.adjust_player_resource(ctx, conn, player_name, "influence", -influence,
                                                   "conflict_involvement", f"Conflict {conflict_id}")

            # ── 3. Upsert involvement row ────────────────────────────────
            exists = await conn.fetchval(
                """
                SELECT 1 FROM PlayerConflictInvolvement
                 WHERE conflict_id = $1 AND user_id = $2 AND conversation_id = $3
                """,
                conflict_id,
                context.user_id,
                context.conversation_id,
            )

            if exists:
                await conn.execute(
                    """
                    UPDATE PlayerConflictInvolvement
                       SET involvement_level  = $1,
                           faction            = $2,
                           money_committed    = money_committed + $3,
                           supplies_committed = supplies_committed + $4,
                           influence_committed= influence_committed + $5,
                           actions_taken      = $6,
                           manipulated_by     = $7
                     WHERE conflict_id = $8 AND user_id = $9 AND conversation_id = $10
                    """,
                    level,
                    faction,
                    money,
                    supplies,
                    influence,
                    actions,
                    manipulated_by,
                    conflict_id,
                    context.user_id,
                    context.conversation_id,
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO PlayerConflictInvolvement
                           (conflict_id, user_id, conversation_id, player_name,
                            involvement_level, faction,
                            money_committed, supplies_committed, influence_committed,
                            actions_taken, manipulated_by)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                    """,
                    conflict_id,
                    context.user_id,
                    context.conversation_id,
                    player_name,
                    level,
                    faction,
                    money,
                    supplies,
                    influence,
                    actions,
                    manipulated_by,
                )

            # ── 4. Canonical log ─────────────────────────────────────────
            await canon.log_canonical_event(
                ctx,
                conn,
                f"Player involvement in conflict {conflict_id} updated → {level}/{faction}",
                tags=["conflict", "player_involvement", level, faction],
                significance=7,
            )

            return {
                "success": True,
                "conflict_id": conflict_id,
                "involvement_level": level,
                "faction": faction,
                "resources_committed": {"money": money, "supplies": supplies, "influence": influence},
            }

    except Exception as e:
        logger.error("Error updating player involvement: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


@function_tool(strict_mode=False)
async def add_internal_conflict(ctx: RunContextWrapper, conflict_id: int, internal_conflict_data: Dict[str, Any]) -> Dict[str, Any]:
    """Add an internal faction conflict to a main conflict."""
    return await _internal_add_internal_conflict_logic(ctx, conflict_id, internal_conflict_data)

@function_tool(strict_mode=False)
async def resolve_internal_conflict(ctx: RunContextWrapper, struggle_id: int, resolution_data: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve an internal faction conflict."""
    return await _internal_resolve_internal_conflict_logic(ctx, struggle_id, resolution_data)

@function_tool(strict_mode=False)
async def get_calendar_context(ctx: RunContextWrapper) -> Dict[str, Any]:
    """Get calendar context with setting-specific month and day names."""
    return await _internal_get_calendar_context_logic(ctx)

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
