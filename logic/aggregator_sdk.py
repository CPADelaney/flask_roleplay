# logic/aggregator_sdk.py
"""
Data aggregation system using OpenAI's Agents SDK with Nyx Governance integration.

This module is responsible for gathering data from multiple database tables,
aggregating them into a cohesive context for other systems, and ensuring
all data retrieval is governed by Nyx.
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# OpenAI Agents SDK imports
from agents import (
    Agent,
    ModelSettings,
    Runner,
    function_tool,
    RunContextWrapper,
    GuardrailFunctionOutput,
    InputGuardrail,
    trace,
    handoff
)
from pydantic import BaseModel, Field

# DB connection
from db.connection import get_db_connection
import asyncpg

# Nyx governance integration
from nyx.nyx_governance import (
    NyxUnifiedGovernor,
    AgentType,
    DirectiveType,
    DirectivePriority
)
from nyx.integrate import get_central_governance

# Calendar integration
from logic.calendar import update_calendar_names, load_calendar_names

# -------------------------------------------------------------------------------
# Pydantic Models for Structured Outputs
# -------------------------------------------------------------------------------

class AggregatedData(BaseModel):
    """Structure for aggregated game data"""
    player_stats: Dict[str, Any] = Field(default_factory=dict, description="Player statistics")
    introduced_npcs: List[Dict[str, Any]] = Field(default_factory=list, description="NPCs that have been introduced")
    unintroduced_npcs: List[Dict[str, Any]] = Field(default_factory=list, description="NPCs that have not been introduced")
    current_roleplay: Dict[str, Any] = Field(default_factory=dict, description="Current roleplay context")
    calendar: Dict[str, Any] = Field(default_factory=dict, description="Calendar information")
    year: str = Field("1040", description="Current year")
    month: str = Field("6", description="Current month")
    day: str = Field("15", description="Current day")
    time_of_day: str = Field("Morning", description="Current time of day")
    social_links: List[Dict[str, Any]] = Field(default_factory=list, description="Social relationships")
    player_perks: List[Dict[str, Any]] = Field(default_factory=list, description="Player perks")
    inventory: List[Dict[str, Any]] = Field(default_factory=list, description="Player inventory")
    events: List[Dict[str, Any]] = Field(default_factory=list, description="Game events")
    planned_events: List[Dict[str, Any]] = Field(default_factory=list, description="Planned events")
    quests: List[Dict[str, Any]] = Field(default_factory=list, description="Player quests")
    game_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Game rules")
    stat_definitions: List[Dict[str, Any]] = Field(default_factory=list, description="Stat definitions")
    locations: List[Dict[str, Any]] = Field(default_factory=list, description="Game locations")
    npc_agent_states: Dict[str, Any] = Field(default_factory=dict, description="NPC agent states")
    aggregator_text: str = Field("", description="Compiled aggregator text")

class SceneContextData(BaseModel):
    """Structure for scene context data"""
    location: str = Field(..., description="Current location")
    npcs_present: List[str] = Field(default_factory=list, description="NPCs present in the scene")
    time_info: Dict[str, Any] = Field(default_factory=dict, description="Current time information")
    environment_description: str = Field("", description="Description of the environment")
    player_role: str = Field("", description="Player's role in the scene")
    tension_level: int = Field(0, description="Tension level (0-10)")

class GlobalSummary(BaseModel):
    """Structure for global game summary"""
    introduced_npcs_count: int = Field(0, description="Number of introduced NPCs")
    unintroduced_npcs_count: int = Field(0, description="Number of unintroduced NPCs")
    active_quests_count: int = Field(0, description="Number of active quests")
    locations_count: int = Field(0, description="Number of locations")
    current_time: str = Field("", description="Current in-game time")
    main_quest_status: str = Field("", description="Status of the main quest")
    recent_events: List[str] = Field(default_factory=list, description="Recent notable events")

# -------------------------------------------------------------------------------
# Agent Context
# -------------------------------------------------------------------------------

class AggregatorContext:
    """Context object for aggregator agents"""
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = None
        self.cached_data = {}
        
    async def initialize(self):
        """Initialize context with governance integration"""
        self.governor = await get_central_governance(self.user_id, self.conversation_id)

# -------------------------------------------------------------------------------
# Function Tools
# -------------------------------------------------------------------------------

@function_tool
async def get_player_stats(
    ctx: RunContextWrapper[AggregatorContext],
    player_name: str
) -> Dict[str, Any]:
    """
    Retrieve player statistics from the database.
    
    Args:
        player_name: Name of the player to retrieve stats for
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    governor = ctx.context.governor
    
    # Check permission with governance system
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="aggregator",
        action_type="get_player_stats",
        action_details={"player_name": player_name}
    )
    
    if not permission["approved"]:
        return {"error": permission["reasoning"]}
    
    # Connect to database
    try:
        conn = await asyncpg.connect(dsn=get_db_connection())
        
        # Query player stats
        row = await conn.fetchrow("""
            SELECT corruption, confidence, willpower,
                   obedience, dependency, lust,
                   mental_resilience, physical_endurance
            FROM PlayerStats
            WHERE user_id=$1
              AND conversation_id=$2
              AND player_name=$3
        """, user_id, conversation_id, player_name)
        
        if not row:
            return {"error": f"No stats found for player {player_name}"}
        
        player_stats = {
            "name": player_name,
            "corruption": row["corruption"],
            "confidence": row["confidence"],
            "willpower": row["willpower"],
            "obedience": row["obedience"],
            "dependency": row["dependency"],
            "lust": row["lust"],
            "mental_resilience": row["mental_resilience"],
            "physical_endurance": row["physical_endurance"]
        }
        
        # Report action to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="aggregator",
            action={"type": "get_player_stats", "player_name": player_name},
            result={"stats_retrieved": True}
        )
        
        return player_stats
        
    except Exception as e:
        logging.error(f"Error retrieving player stats: {e}")
        return {"error": str(e)}
    finally:
        await conn.close()

@function_tool
async def get_introduced_npcs(
    ctx: RunContextWrapper[AggregatorContext]
) -> Dict[str, Any]:
    """
    Retrieve NPCs that have been introduced to the player.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    governor = ctx.context.governor
    
    # Check permission with governance system
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="aggregator",
        action_type="get_introduced_npcs",
        action_details={}
    )
    
    if not permission["approved"]:
        return {"npcs": [], "error": permission["reasoning"]}
    
    # Connect to database
    try:
        conn = await asyncpg.connect(dsn=get_db_connection())
        
        # Query introduced NPCs
        rows = await conn.fetch("""
            SELECT npc_id, npc_name,
                   dominance, cruelty, closeness,
                   trust, respect, intensity,
                   hobbies, personality_traits, likes, dislikes,
                   schedule, current_location, physical_description, archetype_extras_summary
            FROM NPCStats
            WHERE user_id=$1
              AND conversation_id=$2
              AND introduced=TRUE
            ORDER BY npc_id
        """, user_id, conversation_id)
        
        introduced_npcs = []
        for row in rows:
            # Parse JSON fields
            try:
                schedule = json.loads(row["schedule"]) if row["schedule"] else {}
            except:
                schedule = {}
                
            try:
                hobbies = row["hobbies"] if row["hobbies"] else []
                personality_traits = row["personality_traits"] if row["personality_traits"] else []
                likes = row["likes"] if row["likes"] else []
                dislikes = row["dislikes"] if row["dislikes"] else []
            except:
                hobbies, personality_traits, likes, dislikes = [], [], [], []
            
            introduced_npcs.append({
                "npc_id": row["npc_id"],
                "npc_name": row["npc_name"],
                "dominance": row["dominance"],
                "cruelty": row["cruelty"],
                "closeness": row["closeness"],
                "trust": row["trust"],
                "respect": row["respect"],
                "intensity": row["intensity"],
                "hobbies": hobbies,
                "personality_traits": personality_traits,
                "likes": likes,
                "dislikes": dislikes,
                "schedule": schedule,
                "current_location": row["current_location"] or "Unknown",
                "physical_description": row["physical_description"] or "",
                "archetype_extras_summary": row["archetype_extras_summary"] or ""
            })
        
        # Report action to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="aggregator",
            action={"type": "get_introduced_npcs"},
            result={"npcs_count": len(introduced_npcs)}
        )
        
        return {"npcs": introduced_npcs}
        
    except Exception as e:
        logging.error(f"Error retrieving introduced NPCs: {e}")
        return {"npcs": [], "error": str(e)}
    finally:
        await conn.close()

@function_tool
async def get_unintroduced_npcs(
    ctx: RunContextWrapper[AggregatorContext]
) -> Dict[str, Any]:
    """
    Retrieve NPCs that have not yet been introduced to the player.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    governor = ctx.context.governor
    
    # Check permission with governance system
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="aggregator",
        action_type="get_unintroduced_npcs",
        action_details={}
    )
    
    if not permission["approved"]:
        return {"npcs": [], "error": permission["reasoning"]}
    
    # Connect to database
    try:
        conn = await asyncpg.connect(dsn=get_db_connection())
        
        # Query unintroduced NPCs
        rows = await conn.fetch("""
            SELECT npc_id, npc_name,
                   schedule, current_location
            FROM NPCStats
            WHERE user_id=$1
              AND conversation_id=$2
              AND introduced=FALSE
            ORDER BY npc_id
        """, user_id, conversation_id)
        
        unintroduced_npcs = []
        for row in rows:
            # Parse JSON fields
            try:
                schedule = json.loads(row["schedule"]) if row["schedule"] else {}
            except:
                schedule = {}
            
            unintroduced_npcs.append({
                "npc_id": row["npc_id"],
                "npc_name": row["npc_name"],
                "current_location": row["current_location"] or "Unknown",
                "schedule": schedule
            })
        
        # Report action to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="aggregator",
            action={"type": "get_unintroduced_npcs"},
            result={"npcs_count": len(unintroduced_npcs)}
        )
        
        return {"npcs": unintroduced_npcs}
        
    except Exception as e:
        logging.error(f"Error retrieving unintroduced NPCs: {e}")
        return {"npcs": [], "error": str(e)}
    finally:
        await conn.close()

@function_tool
async def get_time_info(
    ctx: RunContextWrapper[AggregatorContext]
) -> Dict[str, Any]:
    """
    Retrieve current time information (year, month, day, time of day).
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    governor = ctx.context.governor
    
    # Check permission with governance system
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="aggregator",
        action_type="get_time_info",
        action_details={}
    )
    
    if not permission["approved"]:
        return {
            "year": "1040",
            "month": "6",
            "day": "15",
            "time_of_day": "Morning",
            "error": permission["reasoning"]
        }
    
    # Connect to database
    try:
        conn = await asyncpg.connect(dsn=get_db_connection())
        
        time_info = {
            "year": "1040",
            "month": "6",
            "day": "15",
            "time_of_day": "Morning"
        }
        
        # Query time information
        for key in ["CurrentYear", "CurrentMonth", "CurrentDay", "TimeOfDay"]:
            row = await conn.fetchrow("""
                SELECT value
                FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key=$3
            """, user_id, conversation_id, key)
            
            if row:
                if key == "CurrentYear":
                    time_info["year"] = row["value"]
                elif key == "CurrentMonth":
                    time_info["month"] = row["value"]
                elif key == "CurrentDay":
                    time_info["day"] = row["value"]
                elif key == "TimeOfDay":
                    time_info["time_of_day"] = row["value"]
        
        # Get calendar information
        calendar_raw = None
        calendar_row = await conn.fetchrow("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='CalendarNames'
        """, user_id, conversation_id)
        
        if calendar_row and calendar_row["value"]:
            try:
                calendar_raw = json.loads(calendar_row["value"])
            except:
                calendar_raw = None
        
        if calendar_raw:
            time_info["calendar"] = calendar_raw
        else:
            time_info["calendar"] = {"year_name": "Year 1", "months": [], "days": []}
        
        # Report action to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="aggregator",
            action={"type": "get_time_info"},
            result={"time_retrieved": True}
        )
        
        return time_info
        
    except Exception as e:
        logging.error(f"Error retrieving time info: {e}")
        return {
            "year": "1040",
            "month": "6",
            "day": "15",
            "time_of_day": "Morning",
            "calendar": {"year_name": "Year 1", "months": [], "days": []},
            "error": str(e)
        }
    finally:
        await conn.close()

@function_tool
async def get_current_roleplay_data(
    ctx: RunContextWrapper[AggregatorContext]
) -> Dict[str, Any]:
    """
    Retrieve current roleplay data from the CurrentRoleplay table.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    governor = ctx.context.governor
    
    # Check permission with governance system
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="aggregator",
        action_type="get_current_roleplay_data",
        action_details={}
    )
    
    if not permission["approved"]:
        return {"error": permission["reasoning"]}
    
    # Connect to database
    try:
        conn = await asyncpg.connect(dsn=get_db_connection())
        
        # Query all current roleplay data
        rows = await conn.fetch("""
            SELECT key, value
            FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2
        """, user_id, conversation_id)
        
        current_roleplay_data = {}
        
        for row in rows:
            key, value = row["key"], row["value"]
            
            # Try to parse JSON for specific fields
            if key == "ChaseSchedule":
                try:
                    current_roleplay_data[key] = json.loads(value)
                except:
                    current_roleplay_data[key] = value
            else:
                current_roleplay_data[key] = value
        
        # Report action to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="aggregator",
            action={"type": "get_current_roleplay_data"},
            result={"keys_retrieved": len(current_roleplay_data)}
        )
        
        return current_roleplay_data
        
    except Exception as e:
        logging.error(f"Error retrieving current roleplay data: {e}")
        return {"error": str(e)}
    finally:
        await conn.close()

@function_tool
async def get_social_links(
    ctx: RunContextWrapper[AggregatorContext]
) -> Dict[str, Any]:
    """
    Retrieve social links between entities.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    governor = ctx.context.governor
    
    # Check permission with governance system
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="aggregator",
        action_type="get_social_links",
        action_details={}
    )
    
    if not permission["approved"]:
        return {"links": [], "error": permission["reasoning"]}
    
    # Connect to database
    try:
        conn = await asyncpg.connect(dsn=get_db_connection())
        
        # Query social links
        rows = await conn.fetch("""
            SELECT link_id, entity1_type, entity1_id,
                   entity2_type, entity2_id,
                   link_type, link_level, link_history
            FROM SocialLinks
            WHERE user_id=$1 AND conversation_id=$2
            ORDER BY link_id
        """, user_id, conversation_id)
        
        social_links = []
        
        for row in rows:
            social_links.append({
                "link_id": row["link_id"],
                "entity1_type": row["entity1_type"],
                "entity1_id": row["entity1_id"],
                "entity2_type": row["entity2_type"],
                "entity2_id": row["entity2_id"],
                "link_type": row["link_type"],
                "link_level": row["link_level"],
                "link_history": row["link_history"] or []
            })
        
        # Report action to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="aggregator",
            action={"type": "get_social_links"},
            result={"links_count": len(social_links)}
        )
        
        return {"links": social_links}
        
    except Exception as e:
        logging.error(f"Error retrieving social links: {e}")
        return {"links": [], "error": str(e)}
    finally:
        await conn.close()

@function_tool
async def get_additional_game_data(
    ctx: RunContextWrapper[AggregatorContext]
) -> Dict[str, Any]:
    """
    Retrieve additional game data (player perks, inventory, events, quests, etc.)
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    governor = ctx.context.governor
    
    # Check permission with governance system
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="aggregator",
        action_type="get_additional_game_data",
        action_details={}
    )
    
    if not permission["approved"]:
        return {"error": permission["reasoning"]}
    
    # Connect to database
    try:
        conn = await asyncpg.connect(dsn=get_db_connection())
        
        # Get player perks
        perks_rows = await conn.fetch("""
            SELECT perk_name, perk_description, perk_effect
            FROM PlayerPerks
            WHERE user_id=$1 AND conversation_id=$2
        """, user_id, conversation_id)
        
        player_perks = []
        for row in perks_rows:
            player_perks.append({
                "perk_name": row["perk_name"],
                "perk_description": row["perk_description"],
                "perk_effect": row["perk_effect"]
            })
        
        # Get inventory
        inventory_rows = await conn.fetch("""
            SELECT player_name, item_name, item_description, item_effect,
                   quantity, category
            FROM PlayerInventory
            WHERE user_id=$1 AND conversation_id=$2
        """, user_id, conversation_id)
        
        inventory_list = []
        for row in inventory_rows:
            inventory_list.append({
                "player_name": row["player_name"],
                "item_name": row["item_name"],
                "item_description": row["item_description"],
                "item_effect": row["item_effect"],
                "quantity": row["quantity"],
                "category": row["category"]
            })
        
        # Get events
        events_rows = await conn.fetch("""
            SELECT id, event_name, description, start_time, end_time, location,
                   year, month, day, time_of_day
            FROM Events
            WHERE user_id=$1 AND conversation_id=$2
            ORDER BY id
        """, user_id, conversation_id)
        
        events_list = []
        for row in events_rows:
            events_list.append({
                "event_id": row["id"],
                "event_name": row["event_name"],
                "description": row["description"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "location": row["location"],
                "year": row["year"],
                "month": row["month"],
                "day": row["day"],
                "time_of_day": row["time_of_day"]
            })
        
        # Get planned events
        planned_events_rows = await conn.fetch("""
            SELECT event_id, npc_id, year, month, day, time_of_day, override_location
            FROM PlannedEvents
            WHERE user_id=$1 AND conversation_id=$2
            ORDER BY event_id
        """, user_id, conversation_id)
        
        planned_events_list = []
        for row in planned_events_rows:
            planned_events_list.append({
                "event_id": row["event_id"],
                "npc_id": row["npc_id"],
                "year": row["year"],
                "month": row["month"],
                "day": row["day"],
                "time_of_day": row["time_of_day"],
                "override_location": row["override_location"]
            })
        
        # Get quests
        quests_rows = await conn.fetch("""
            SELECT quest_id, quest_name, status, progress_detail,
                   quest_giver, reward
            FROM Quests
            WHERE user_id=$1 AND conversation_id=$2
            ORDER BY quest_id
        """, user_id, conversation_id)
        
        quests_list = []
        for row in quests_rows:
            quests_list.append({
                "quest_id": row["quest_id"],
                "quest_name": row["quest_name"],
                "status": row["status"],
                "progress_detail": row["progress_detail"],
                "quest_giver": row["quest_giver"],
                "reward": row["reward"]
            })
        
        # Report action to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="aggregator",
            action={"type": "get_additional_game_data"},
            result={
                "perks_count": len(player_perks),
                "inventory_count": len(inventory_list),
                "events_count": len(events_list),
                "quests_count": len(quests_list)
            }
        )
        
        return {
            "player_perks": player_perks,
            "inventory": inventory_list,
            "events": events_list,
            "planned_events": planned_events_list,
            "quests": quests_list
        }
        
    except Exception as e:
        logging.error(f"Error retrieving additional game data: {e}")
        return {"error": str(e)}
    finally:
        await conn.close()

@function_tool
async def get_global_tables(
    ctx: RunContextWrapper[AggregatorContext]
) -> Dict[str, Any]:
    """
    Retrieve global tables data (game rules, stat definitions, etc.)
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    governor = ctx.context.governor
    
    # Check permission with governance system
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="aggregator",
        action_type="get_global_tables",
        action_details={}
    )
    
    if not permission["approved"]:
        return {"error": permission["reasoning"]}
    
    # Connect to database
    try:
        conn = await asyncpg.connect(dsn=get_db_connection())
        
        # Get game rules
        rules_rows = await conn.fetch("""
            SELECT rule_name, condition, effect
            FROM GameRules
            ORDER BY rule_name
        """)
        
        game_rules_list = []
        for row in rules_rows:
            game_rules_list.append({
                "rule_name": row["rule_name"],
                "condition": row["condition"],
                "effect": row["effect"]
            })
        
        # Get stat definitions
        stat_rows = await conn.fetch("""
            SELECT id, scope, stat_name, range_min, range_max,
                   definition, effects, progression_triggers
            FROM StatDefinitions
            ORDER BY id
        """)
        
        stat_def_list = []
        for row in stat_rows:
            stat_def_list.append({
                "id": row["id"],
                "scope": row["scope"],
                "stat_name": row["stat_name"],
                "range_min": row["range_min"],
                "range_max": row["range_max"],
                "definition": row["definition"],
                "effects": row["effects"],
                "progression_triggers": row["progression_triggers"]
            })
        
        # Get locations
        location_rows = await conn.fetch("""
            SELECT id, location_name, description
            FROM Locations
            WHERE user_id=$1 AND conversation_id=$2
            ORDER BY id
        """, user_id, conversation_id)
        
        locations_list = []
        for row in location_rows:
            locations_list.append({
                "location_id": row["id"],
                "location_name": row["location_name"],
                "description": row["description"]
            })
        
        # Get NPC agent states
        npc_agent_rows = await conn.fetch("""
            SELECT npc_id, current_state, last_decision
            FROM NPCAgentState
            WHERE user_id=$1 AND conversation_id=$2
        """, user_id, conversation_id)
        
        npc_agent_states = {}
        for row in npc_agent_rows:
            npc_agent_states[row["npc_id"]] = {
                "current_state": row["current_state"],
                "last_decision": row["last_decision"]
            }
        
        # Report action to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="aggregator",
            action={"type": "get_global_tables"},
            result={
                "rules_count": len(game_rules_list),
                "stats_count": len(stat_def_list),
                "locations_count": len(locations_list)
            }
        )
        
        return {
            "game_rules": game_rules_list,
            "stat_definitions": stat_def_list,
            "locations": locations_list,
            "npc_agent_states": npc_agent_states
        }
        
    except Exception as e:
        logging.error(f"Error retrieving global tables: {e}")
        return {"error": str(e)}
    finally:
        await conn.close()

@function_tool
async def generate_global_summary(
    ctx: RunContextWrapper[AggregatorContext],
    aggregated_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a global summary based on aggregated data.
    
    Args:
        aggregated_data: The aggregated game data
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    governor = ctx.context.governor
    
    # Check permission with governance system
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="aggregator",
        action_type="generate_global_summary",
        action_details={}
    )
    
    if not permission["approved"]:
        return {"summary": "", "error": permission["reasoning"]}
    
    # Connect to database
    try:
        conn = await asyncpg.connect(dsn=get_db_connection())
        
        # Get existing summary
        row = await conn.fetchrow("""
            SELECT value
            FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='GlobalSummary'
        """, user_id, conversation_id)
        
        existing_summary = row["value"] if row else ""
        
        # Build changes summary
        introduced_count = len(aggregated_data.get("introduced_npcs", []))
        unintroduced_count = len(aggregated_data.get("unintroduced_npcs", []))
        
        changes = f"Introduced NPCs: {introduced_count}, Unintroduced: {unintroduced_count}"
        
        # Update global summary
        if introduced_count == 0 and unintroduced_count == 0:
            updated_summary = existing_summary
        else:
            updated_summary = update_global_summary(existing_summary, changes)
            
            # Save updated summary
            await conn.execute("""
                INSERT INTO CurrentRoleplay(user_id, conversation_id, key, value)
                VALUES($1, $2, 'GlobalSummary', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value=EXCLUDED.value
            """, user_id, conversation_id, updated_summary)
        
        # Report action to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="aggregator",
            action={"type": "generate_global_summary"},
            result={"summary_updated": updated_summary != existing_summary}
        )
        
        return {"summary": updated_summary}
        
    except Exception as e:
        logging.error(f"Error generating global summary: {e}")
        return {"summary": "", "error": str(e)}
    finally:
        await conn.close()

@function_tool
async def build_aggregator_text(
    ctx: RunContextWrapper[AggregatorContext],
    aggregated_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build the final aggregator text based on aggregated data.
    
    Args:
        aggregated_data: The aggregated game data
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    governor = ctx.context.governor
    
    # Check permission with governance system
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="aggregator",
        action_type="build_aggregator_text",
        action_details={}
    )
    
    if not permission["approved"]:
        return {"aggregator_text": "", "error": permission["reasoning"]}
    
    # Get the global summary
    summary = aggregated_data.get("global_summary", "")
    
    # Get calendar information for immersive date
    calendar_info = aggregated_data.get("calendar", {})
    if not calendar_info or not isinstance(calendar_info, dict):
        calendar_info = {"year_name": "Year 1", "months": [], "days": []}
    
    # Format immersive date
    year = aggregated_data.get("year", "1040")
    month = aggregated_data.get("month", "6")
    day = aggregated_data.get("day", "15")
    time_of_day = aggregated_data.get("time_of_day", "Morning")
    
    immersive_date = f"Year: {calendar_info.get('year_name', year)}"
    months = calendar_info.get("months", [])
    if months and month.isdigit():
        month_index = int(month) - 1
        if 0 <= month_index < len(months):
            immersive_date += f", Month: {months[month_index]}"
    immersive_date += f", Day: {day}, {time_of_day}."
    
    # Create scene snapshot
    introduced_npcs = aggregated_data.get("introduced_npcs", [])
    unintroduced_npcs = aggregated_data.get("unintroduced_npcs", [])
    
    scene_lines = [
        f"- It is {year_str}, {month_str} {day_str}, {time_of_day}.\n"
    ]
    
    # Introduced NPCs snippet
    scene_lines.append("Introduced NPCs in the area:")
    for npc in introduced_npcs[:4]:
        loc = npc.get("current_location", "Unknown")
        scene_lines.append(f"  - {npc['npc_name']} is at {loc}")
    
    # Unintroduced NPCs snippet
    scene_lines.append("Unintroduced NPCs (possible random encounters):")
    for npc in unintroduced_npcs[:2]:
        loc = npc.get("current_location", "Unknown")
        scene_lines.append(f"  - ???: '{npc['npc_name']}' lurking around {loc}")
    
    scene_snapshot = "\n".join(scene_lines)
    
    # Build the aggregator text
    aggregator_text = (
        f"{summary}\n\n"
        f"{immersive_date}\n"
        "Scene Snapshot:\n"
        f"{scene_snapshot}"
    )
    
    # Add additional context from CurrentRoleplay if available
    current_roleplay = aggregated_data.get("current_roleplay", {})
    if "EnvironmentDesc" in current_roleplay:
        aggregator_text += "\n\nEnvironment Description:\n" + current_roleplay["EnvironmentDesc"]
    if "PlayerRole" in current_roleplay:
        aggregator_text += "\n\nPlayer Role:\n" + current_roleplay["PlayerRole"]
    if "MainQuest" in current_roleplay:
        aggregator_text += "\n\nMain Quest (hint):\n" + current_roleplay["MainQuest"]
    if "ChaseSchedule" in current_roleplay:
        aggregator_text += "\n\nChase Schedule:\n" + json.dumps(current_roleplay["ChaseSchedule"], indent=2)
    
    # Add Notable Events and Locations
    events = aggregated_data.get("events", [])
    if events:
        aggregator_text += "\n\nNotable Events:\n"
        for ev in events[:3]:
            aggregator_text += f"- {ev['event_name']}: {ev['description']} (at {ev['location']})\n"
    
    locations = aggregated_data.get("locations", [])
    if locations:
        aggregator_text += "\n\nNotable Locations:\n"
        for loc in locations[:3]:
            aggregator_text += f"- {loc['location_name']}: {loc['description']}\n"
    
    # Add MegaSettingModifiers if available
    modifiers_str = current_roleplay.get("MegaSettingModifiers", "")
    if modifiers_str:
        aggregator_text += "\n\n=== MEGA SETTING MODIFIERS ===\n"
        try:
            mod_dict = json.loads(modifiers_str)
            for k, v in mod_dict.items():
                aggregator_text += f"- {k}: {v}\n"
        except:
            aggregator_text += "(Could not parse MegaSettingModifiers)\n"
    
    # Report action to governance
    await governor.process_agent_action_report(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="aggregator",
        action={"type": "build_aggregator_text"},
        result={"text_length": len(aggregator_text)}
    )
    
    return {"aggregator_text": aggregator_text}

# -------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------

def update_global_summary(old_summary, new_stuff, max_len=3000):
    """
    Update the global summary with new information.
    """
    combined = old_summary.strip() + "\n\n" + new_stuff.strip()
    if len(combined) > max_len:
        combined = combined[-max_len:]
    return combined

# -------------------------------------------------------------------------------
# Agent Definitions
# -------------------------------------------------------------------------------

# Scene Context Agent
scene_context_agent = Agent[AggregatorContext](
    name="Scene Context Agent",
    instructions="""
    You analyze game state data to create concise scene context.
    Your role is to:
    1. Identify the most relevant information for the current scene
    2. Summarize critical environmental details
    3. Track NPCs present in the current scene
    4. Note any important time-related information
    5. Create a focused snapshot of the current game state
    
    Provide scene context that helps other agents understand the current situation.
    """,
    tools=[
        get_time_info,
        get_introduced_npcs,
        get_current_roleplay_data
    ],
    output_type=SceneContextData
)

# Global Summary Agent
global_summary_agent = Agent[AggregatorContext](
    name="Global Summary Agent",
    instructions="""
    You create and maintain the global game summary.
    Your role is to:
    1. Track key changes in the game world
    2. Summarize the current game state
    3. Note important NPCs, quests, and locations
    4. Provide context about the current time period
    5. Create a cohesive summary that other agents can reference
    
    Create summaries that are concise yet comprehensive.
    """,
    tools=[
        generate_global_summary
    ],
    output_type=GlobalSummary
)

# Main Aggregator Agent
aggregator_agent = Agent[AggregatorContext](
    name="Data Aggregator Agent",
    instructions="""
    You are the central data aggregation system for a femdom roleplaying game.
    
    Your role is to:
    1. Gather data from multiple database tables
    2. Merge data into a cohesive context for other systems
    3. Generate global summaries and scene contexts
    4. Provide relevant data to other agents as needed
    5. Track changes in the game world over time
    
    Ensure all data is properly integrated and presented in a format
    that helps other agents understand the current game state.
    """,
    handoffs=[
        handoff(scene_context_agent, tool_name_override="generate_scene_context"),
        handoff(global_summary_agent, tool_name_override="generate_global_summary")
    ],
    tools=[
        get_player_stats,
        get_introduced_npcs,
        get_unintroduced_npcs,
        get_time_info,
        get_current_roleplay_data,
        get_social_links,
        get_additional_game_data,
        get_global_tables,
        build_aggregator_text
    ],
    output_type=AggregatedData
)

# -------------------------------------------------------------------------------
# Main Functions
# -------------------------------------------------------------------------------

async def get_aggregated_roleplay_context(
    user_id: int,
    conversation_id: int,
    player_name: str = "Chase"
) -> Dict[str, Any]:
    """
    Gather and aggregate data from multiple tables with governance oversight.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        player_name: Name of the player (default: "Chase")
        
    Returns:
        Aggregated game data including player stats, NPCs, time, etc.
    """
    # Create aggregator context
    aggregator_context = AggregatorContext(user_id, conversation_id)
    await aggregator_context.initialize()
    
    # Create trace for monitoring
    with trace(
        workflow_name="Data Aggregation",
        trace_id=f"aggregator-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
        # Create prompt
        prompt = f"""
        Aggregate game data for player {player_name}.
        Include:
        - Player stats
        - NPC information
        - Time and calendar data
        - Social links
        - Inventory, perks, events, and quests
        - Global rules and definitions
        - Locations
        
        Create a comprehensive but focused aggregation of game state.
        """
        
        # Run the agent
        result = await Runner.run(
            aggregator_agent,
            prompt,
            context=aggregator_context
        )
    
    # Get structured output
    aggregated_data = result.final_output_as(AggregatedData)
    
    # Convert to dictionary for return
    aggregated_dict = aggregated_data.dict()
    
    return aggregated_dict

async def get_scene_context(
    user_id: int,
    conversation_id: int
) -> Dict[str, Any]:
    """
    Get focused scene context data with governance oversight.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Scene context data including location, NPCs present, etc.
    """
    # Create aggregator context
    aggregator_context = AggregatorContext(user_id, conversation_id)
    await aggregator_context.initialize()
    
    # Create trace for monitoring
    with trace(
        workflow_name="Scene Context",
        trace_id=f"scene-context-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
        # Run the agent
        result = await Runner.run(
            scene_context_agent,
            "Generate the current scene context",
            context=aggregator_context
        )
    
    # Get structured output
    scene_context = result.final_output_as(SceneContextData)
    
    # Convert to dictionary for return
    scene_context_dict = scene_context.dict()
    
    return scene_context_dict

# Register with Nyx governance
async def register_with_governance(user_id: int, conversation_id: int):
    """
    Register aggregator agents with Nyx governance system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
    """
    # Get governor
    governor = await get_central_governance(user_id, conversation_id)
    
    # Register main agent
    await governor.register_agent(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_instance=aggregator_agent,
        agent_id="aggregator"
    )
    
    # Issue directive for data aggregation
    await governor.issue_directive(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="aggregator",
        directive_type=DirectiveType.ACTION,
        directive_data={
            "instruction": "Aggregate game data to provide context for other systems",
            "scope": "game"
        },
        priority=DirectivePriority.MEDIUM,
        duration_minutes=24*60  # 24 hours
    )
    
    logging.info("Aggregator system registered with Nyx governance")
