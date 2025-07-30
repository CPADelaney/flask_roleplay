# aggregator_sdk.py

"""
Optimized replacement for aggregator.py that integrates with the
unified context service for improved performance and efficiency.

Usage:
  1) Import in your main or startup code:
       from logic.aggregator_sdk import init_singletons
  2) Call `await init_singletons()` once in an async startup routine
     to initialize global singletons like context_cache or incremental_context_manager.

Then use:
  from logic.aggregator_sdk import (
      get_aggregated_roleplay_context,
      context_cache,
      incremental_context_manager,
      ...
  )
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from context.context_service import (
    get_context_service,
    get_comprehensive_context,
    cleanup_context_services
)
from context.context_config import get_config
from context.context_performance import PerformanceMonitor, track_performance

# Using new async context manager for DB connections
from db.connection import get_db_connection_context
from story_templates.preset_manager import PresetStoryManager


logger = logging.getLogger(__name__)

###############################################################################
# Global Singletons: Will be set by init_singletons()
###############################################################################
context_cache: Optional["OptimizedContextCache"] = None
incremental_context_manager: Optional["OptimizedIncrementalContextManager"] = None


###############################################################################
# Public Initialization
###############################################################################
async def init_singletons() -> None:
    """
    Initialize global singletons in aggregator_sdk. Must be called once in an async context.
    For example, in main.py after creating your Flask app, do:

        import asyncio
        from logic.aggregator_sdk import init_singletons

        async def startup():
            await init_singletons()

        if __name__ == "__main__":
            asyncio.run(startup())
            # Then run your Flask/Quart server

    or, if you're using an async framework's startup event (FastAPI, etc.):

        @app.on_event("startup")
        async def startup():
            await init_singletons()

    After calling init_singletons(), you can safely access context_cache or
    incremental_context_manager in aggregator_sdk.
    """
    global context_cache, incremental_context_manager
    logger.info("Initializing aggregator_sdk singletons...")

    context_cache = await OptimizedContextCache.create()
    incremental_context_manager = await OptimizedIncrementalContextManager.create()

    logger.info("Aggregator SDK singletons initialized successfully.")


###############################################################################
# Main Context Retrieval
###############################################################################

@track_performance("get_aggregated_roleplay_context")
async def get_aggregated_roleplay_context(user_id: int, conversation_id: int, player_name: str) -> Dict[str, Any]:
    """
    Get aggregated roleplay context with preset story support
    """
    async with get_db_connection_context() as conn:
        # Get current roleplay state
        roleplay_rows = await conn.fetch("""
            SELECT key, value 
            FROM CurrentRoleplay 
            WHERE user_id = $1 AND conversation_id = $2
        """, user_id, conversation_id)
        
        current_roleplay = {}
        for row in roleplay_rows:
            try:
                current_roleplay[row['key']] = json.loads(row['value'])
            except:
                current_roleplay[row['key']] = row['value']
        
        # Get current location
        current_location = current_roleplay.get('CurrentLocation', 'Unknown')
        time_of_day = current_roleplay.get('TimeOfDay', 'Morning')
        
        # Get NPCs present at location
        npc_rows = await conn.fetch("""
            SELECT npc_id, npc_name, physical_description, personality_traits,
                   trust, dominance, cruelty, affection, intensity, introduced
            FROM NPCStats
            WHERE user_id = $1 AND conversation_id = $2 
            AND current_location = $3
        """, user_id, conversation_id, current_location)
        
        npcs_present = []
        for npc in npc_rows:
            try:
                personality_traits = []
                if npc['personality_traits']:
                    try:
                        personality_traits = json.loads(npc['personality_traits'])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse personality_traits for NPC {npc['npc_id']}")
                        personality_traits = []
                
                npcs_present.append({
                    'id': npc['npc_id'],
                    'name': npc['npc_name'],
                    'description': npc['physical_description'],
                    'traits': personality_traits,
                    'stats': {
                        'trust': npc['trust'],
                        'dominance': npc['dominance'],
                        'cruelty': npc['cruelty'],
                        'affection': npc['affection'],
                        'intensity': npc['intensity']
                    },
                    'introduced': npc['introduced']
                })
            except Exception as e:
                logger.error(f"Error processing NPC {npc.get('npc_id', 'unknown')}: {e}")
                continue
        
        # Get player stats
        player_stats_row = await conn.fetchrow("""
            SELECT corruption, confidence, willpower, obedience, dependency,
                   lust, mental_resilience, physical_endurance
            FROM PlayerStats
            WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
        """, user_id, conversation_id, player_name)
        
        player_stats = dict(player_stats_row) if player_stats_row else {}
        
        # Get active events
        event_rows = await conn.fetch("""
            SELECT event_name, description, location, fantasy_level
            FROM Events
            WHERE user_id = $1 AND conversation_id = $2
            AND (
                (day = $3 AND time_of_day = $4) OR
                (
                    start_time IS NOT NULL 
                    AND end_time IS NOT NULL
                    AND start_time != ''
                    AND end_time != ''
                    AND (
                        (LENGTH(start_time) >= 10 AND SUBSTRING(start_time, 1, 10) ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' AND start_time::timestamp <= NOW())
                        OR FALSE
                    )
                    AND (
                        (LENGTH(end_time) >= 10 AND SUBSTRING(end_time, 1, 10) ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' AND end_time::timestamp >= NOW())
                        OR FALSE
                    )
                )
            )
        """, user_id, conversation_id, 
            current_roleplay.get('CurrentDay', 1),
            time_of_day)
        
        active_events = [dict(row) for row in event_rows]
        
        # Get active quests
        quest_rows = await conn.fetch("""
            SELECT quest_id, quest_name, status, progress_detail
            FROM Quests
            WHERE user_id = $1 AND conversation_id = $2
            AND status IN ('active', 'in_progress')
        """, user_id, conversation_id)
        
        active_quests = [dict(row) for row in quest_rows]
        
        # Build base context
        result = {
            'currentRoleplay': current_roleplay,
            'currentLocation': current_location,
            'timeOfDay': time_of_day,
            'playerName': player_name,
            'playerStats': player_stats,
            'npcsPresent': npcs_present,
            'activeEvents': active_events,
            'activeQuests': active_quests,
            'timestamp': datetime.now().isoformat()
        }
        
        # Check for preset story
        preset_info = await PresetStoryManager.check_preset_story(conversation_id)
        
        if preset_info and preset_info.get('story_id') == 'the_moth_and_flame':
            from story_templates.moth.lore import SFBayMothFlamePreset
            from story_templates.moth.lore.consistency_guide import QueenOfThornsConsistencyGuide
            
            # Get location-specific lore
            location_lore = await get_location_specific_lore(
                conversation_id, 
                current_location,
                preset_info['story_id']
            )
            
            # Add preset-specific context
            result['preset_story'] = {
                'active': True,
                'story_id': 'the_moth_and_flame',
                'setting': 'San Francisco Bay Area',
                'year': 2025,
                'current_act': preset_info.get('current_act', 1),
                'current_beat': preset_info.get('current_beat'),
                'story_flags': preset_info.get('story_flags', {}),
                'consistency_rules': QueenOfThornsConsistencyGuide.get_critical_rules(),
                'current_location_lore': location_lore,
                'network_naming': {
                    'internal': ['the network', 'the garden', 'our people'],
                    'external': ['The Rose & Thorn Society', 'The Thorn Garden', 'that secret feminist cult']
                }
            }
            
            # Add critical constraints as a separate field for easy access
            result['preset_constraints'] = """
CRITICAL CONSTRAINTS FOR THIS STORY:
1. The network has NO official name - internally called "the network" or "the garden"
2. The Queen of Thorns identity is ALWAYS ambiguous - never confirm if one person or many
3. The network controls Bay Area ONLY - other cities have allies, not branches
4. Transformation takes months/years, never instant
5. Use the four-layer information model: PUBLIC|SEMI-PRIVATE|HIDDEN|DEEP SECRET
6. Never place the Queen's private locations precisely
7. The network cannot operate openly or control everyone
"""
            
            # Add quick reference for generation
            result['generation_hints'] = {
                'forbidden_phrases': [
                    "The Rose & Thorn Society announced",
                    "The Garden's official",
                    "Queen [Name]",
                    "our Seattle chapter",
                    "instantly transformed"
                ],
                'correct_usage': {
                    'network_reference': 'the network',
                    'queen_reference': 'The Queen, whoever she is',
                    'other_cities': 'allied networks in [city]',
                    'transformation_time': 'months of careful work'
                }
            }
            
            # Check if we're at a special location
            if is_preset_special_location(current_location):
                result['special_location_active'] = True
                result['location_special_rules'] = get_location_special_rules(current_location)
        
        # Generate aggregator text
        aggregator_text = build_aggregator_text(result)
        
        # Enhance aggregator text with preset warnings if active
        if preset_info:
            aggregator_text = f"""{aggregator_text}

ACTIVE PRESET STORY: {preset_info.get('story_id')}
You MUST follow all consistency rules for this preset story.
{result.get('preset_constraints', '')}
"""
        
        result['aggregatorText'] = aggregator_text
        
        return result


async def get_location_specific_lore(
    conversation_id: int, 
    location_name: str,
    story_id: str
) -> Dict[str, Any]:
    """Get location-specific lore for preset stories"""
    
    if story_id != 'the_moth_and_flame':
        return {}
    
    from story_templates.moth.lore import SFBayMothFlamePreset
    
    # Get all locations from preset
    all_locations = SFBayMothFlamePreset.get_specific_locations()
    
    # Find matching location
    location_lower = location_name.lower()
    matching_location = None
    
    for loc in all_locations:
        if loc['name'].lower() in location_lower or location_lower in loc['name'].lower():
            matching_location = loc
            break
    
    if not matching_location:
        # Check districts
        districts = SFBayMothFlamePreset.get_districts()
        for district in districts:
            if district['name'].lower() in location_lower:
                return {
                    'district': district,
                    'type': 'district',
                    'special_rules': district.get('special_rules', [])
                }
        return {}
    
    # Get relevant myths for this location
    all_myths = SFBayMothFlamePreset.get_urban_myths()
    relevant_myths = [
        myth for myth in all_myths
        if any(keyword in location_lower for keyword in ['sanctum', 'garden', 'underground'])
    ]
    
    return {
        'location': matching_location,
        'type': 'specific_location',
        'myths': relevant_myths,
        'access_level': matching_location.get('access_level', 'public'),
        'special_mechanics': matching_location.get('special_mechanics', [])
    }


def is_preset_special_location(location: str) -> bool:
    """Check if location has special preset rules"""
    location_lower = location.lower()
    special_locations = [
        'velvet sanctum',
        'rose garden café',
        'butterfly house',
        'safehouse',
        'inner garden',
        'the underground',
        'financial district - level -5'
    ]
    
    return any(special in location_lower for special in special_locations)


def get_location_special_rules(location: str) -> List[str]:
    """Get special rules for preset locations"""
    location_lower = location.lower()
    
    rules = []
    
    if 'velvet sanctum' in location_lower:
        rules.extend([
            'sanctuary_rules_apply',
            'performances_possible',
            'queen_may_appear',
            'power_dynamics_heightened'
        ])
    elif 'rose garden' in location_lower:
        rules.extend([
            'recruitment_possible',
            'coded_conversations',
            'network_entry_point'
        ])
    elif 'safehouse' in location_lower or 'butterfly house' in location_lower:
        rules.extend([
            'no_violence_allowed',
            'healing_space',
            'moth_queen_protects',
            'new_identities_available'
        ])
    elif 'level -5' in location_lower:
        rules.extend([
            'hidden_power_center',
            'invitation_only',
            'extreme_secrecy'
        ])
    
    return rules


def format_context_for_compatibility(optimized_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format optimized context to match the format expected by existing code.
    """
    compatible = {}

    # Time info
    if "time_info" in optimized_context:
        time_info = optimized_context["time_info"]
        compatible["year"] = time_info.get("year", "1040")
        compatible["month"] = time_info.get("month", "6")
        compatible["day"] = time_info.get("day", "15")
        compatible["time_of_day"] = time_info.get("time_of_day", "Morning")

    # Player stats
    if "player_stats" in optimized_context:
        compatible["player_stats"] = optimized_context["player_stats"]

    # NPC lists
    compatible["introduced_npcs"] = optimized_context.get("npcs", [])
    compatible["unintroduced_npcs"] = []

    # Current roleplay
    if "current_roleplay" in optimized_context:
        compatible["current_roleplay"] = optimized_context["current_roleplay"]

    # Location details
    if "location_details" in optimized_context:
        location_details = optimized_context["location_details"]
        compatible["current_location"] = location_details.get(
            "location_name",
            optimized_context.get("current_location", "Unknown")
        )
        compatible["location_details"] = location_details
    else:
        compatible["current_location"] = optimized_context.get("current_location", "Unknown")

    # Memories
    if "memories" in optimized_context:
        compatible["memories"] = optimized_context["memories"]

    # Quests
    if "quests" in optimized_context:
        compatible["quests"] = optimized_context["quests"]

    # Delta info
    if "is_delta" in optimized_context:
        compatible["is_delta"] = optimized_context["is_delta"]
        if optimized_context.get("is_delta", False) and "delta_changes" in optimized_context:
            compatible["delta_changes"] = optimized_context["delta_changes"]

    # Token usage
    if "token_usage" in optimized_context:
        compatible["token_usage"] = optimized_context["token_usage"]

    # Copy any other fields
    ignored_keys = {
        "time_info", "npcs", "location_details", "is_delta",
        "delta_changes", "token_usage", "timestamp"
    }
    for key, value in optimized_context.items():
        if key not in compatible and key not in ignored_keys:
            compatible[key] = value

    return compatible


async def fallback_get_context(
    user_id: int,
    conversation_id: int,
    player_name: str = "Chase"
) -> Dict[str, Any]:
    """
    Fallback context retrieval directly from the database.
    """
    try:
        minimal_context = {
            "player_stats": {},
            "introduced_npcs": [],
            "unintroduced_npcs": [],
            "current_roleplay": {},
            "year": "1040",
            "month": "6",
            "day": "15",
            "time_of_day": "Morning"
        }

        async with get_db_connection_context() as conn:
            # 1. Player stats
            row = await conn.fetchrow(
                """
                SELECT corruption, confidence, willpower,
                       obedience, dependency, lust,
                       mental_resilience, physical_endurance
                FROM PlayerStats
                WHERE user_id=$1 AND conversation_id=$2
                LIMIT 1
                """,
                user_id, conversation_id
            )
            if row:
                minimal_context["player_stats"] = dict(row)

            # 2. Introduced NPCs
            rows = await conn.fetch(
                """
                SELECT npc_id, npc_name, dominance, cruelty, closeness,
                       trust, respect, intensity, current_location,
                       physical_description
                FROM NPCStats
                WHERE user_id=$1 AND conversation_id=$2 AND introduced=TRUE
                LIMIT 20
                """,
                user_id, conversation_id
            )
            for r in rows:
                minimal_context["introduced_npcs"].append(dict(r))

            # 3. Time info
            time_keys = [
                ("CurrentYear", "year"),
                ("CurrentMonth", "month"),
                ("CurrentDay", "day"),
                ("TimeOfDay", "time_of_day")
            ]
            for key, context_key in time_keys:
                row = await conn.fetchrow(
                    """
                    SELECT value
                    FROM CurrentRoleplay
                    WHERE user_id=$1 AND conversation_id=$2 AND key=$3
                    """,
                    user_id, conversation_id, key
                )
                if row:
                    minimal_context[context_key] = row["value"]

            # 4. Current roleplay data
            rows = await conn.fetch(
                """
                SELECT key, value
                FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2
                """,
                user_id, conversation_id
            )
            for r in rows:
                minimal_context["current_roleplay"][r["key"]] = r["value"]

        return minimal_context

    except Exception as e:
        logger.error(f"Error in fallback_get_context: {e}", exc_info=True)
        return {
            "player_stats": {},
            "introduced_npcs": [],
            "unintroduced_npcs": [],
            "current_roleplay": {},
            "error": str(e)
        }


###############################################################################
# Optimized Context Cache
###############################################################################

class OptimizedContextCache:
    """
    Multi-level context cache wrapper that integrates with the unified cache.
    """

    def __init__(self, config):
        # All the real config usage is here, no `await`.
        self.l1_ttl = config.get("cache", "l1_ttl_seconds", 60)
        self.l2_ttl = config.get("cache", "l2_ttl_seconds", 300)
        self.l3_ttl = config.get("cache", "l3_ttl_seconds", 3600)
        self.enabled = config.get("cache", "enabled", True)

    @classmethod
    async def create(cls) -> "OptimizedContextCache":
        """
        Async factory that awaits get_config() so we don't do it in __init__.
        """
        cfg = await get_config()
        return cls(cfg)

    async def get(self, key: str, fetch_func, cache_level: int = 1) -> Any:
        """
        Get an item from cache or fetch it if not found. Then store it.

        Args:
            key: Cache key
            fetch_func: Async function to fetch data if cache miss
            cache_level: Which level of the cache (1, 2, or 3)
        """
        if not self.enabled:
            return await fetch_func()

        from context.unified_cache import context_cache

        if cache_level == 1:
            ttl = self.l1_ttl
        elif cache_level == 2:
            ttl = self.l2_ttl
        else:
            ttl = self.l3_ttl

        return await context_cache.get(
            key=key,
            fetch_func=fetch_func,
            cache_level=cache_level,
            ttl_override=ttl
        )

    def invalidate(self, key_prefix: str) -> None:
        """
        Invalidate cache entries that match a given prefix.
        """
        from context.unified_cache import context_cache
        context_cache.invalidate(key_prefix)


###############################################################################
# Optimized Incremental Context Manager
###############################################################################

class OptimizedIncrementalContextManager:
    """
    Incremental context manager that leverages the unified context service
    for partial updates.
    """

    def __init__(self, config):
        self.enabled = config.is_enabled("use_incremental_context")
        self.token_budget = config.get("token_budget", "default_budget", 4000)
        self.use_vector = config.is_enabled("use_vector_search")

    @classmethod
    async def create(cls) -> "OptimizedIncrementalContextManager":
        """
        Async factory method to create an instance once we can await get_config().
        """
        cfg = await get_config()
        return cls(cfg)

    async def get_context(
        self,
        user_id: int,
        conversation_id: int,
        user_input: str,
        location: Optional[str] = None,
        include_delta: bool = True
    ) -> Dict[str, Any]:
        """
        Get context with delta tracking if enabled.
        """
        if not self.enabled or not include_delta:
            return await self.get_full_context(user_id, conversation_id, user_input, location)

        context_service = await get_context_service(user_id, conversation_id)
        context_data = await context_service.get_context(
            input_text=user_input,
            location=location,
            context_budget=self.token_budget,
            use_vector_search=self.use_vector,
            use_delta=True
        )

        result = {
            "full_context": format_context_for_compatibility(context_data),
            "is_incremental": context_data.get("is_delta", False)
        }

        if context_data.get("is_delta", False) and "delta_changes" in context_data:
            result["delta_context"] = context_data["delta_changes"]

        return result

    async def get_full_context(
        self,
        user_id: int,
        conversation_id: int,
        user_input: str,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get the full context without delta tracking.
        """
        context_data = await get_aggregated_roleplay_context(
            user_id=user_id,
            conversation_id=conversation_id,
            current_input=user_input,
            location=location
        )
        return {
            "full_context": context_data,
            "is_incremental": False
        }


###############################################################################
# Additional Utilities
###############################################################################

async def get_optimized_context(
    user_id: int,
    conversation_id: int,
    current_input: str,
    location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Similar to get_aggregated_roleplay_context, but specifically for an "optimized" flow.
    """
    config = await get_config()
    context_budget = config.get("token_budget", "default_budget", 4000)

    context_service = await get_context_service(user_id, conversation_id)
    context_data = await context_service.get_context(
        input_text=current_input,
        location=location,
        context_budget=context_budget,
        use_delta=False
    )
    return format_context_for_compatibility(context_data)


def build_aggregator_text(aggregated_data: Dict[str, Any]) -> str:
    """
    Build aggregator text from the provided data for display or logging.
    Includes preset story context when active.
    """
    # If pre-built aggregator text exists, check if we need to enhance it
    if "aggregator_text" in aggregated_data and not aggregated_data.get("preset_story"):
        return aggregated_data["aggregator_text"]
    
    # Extract basic information
    current_location = aggregated_data.get("currentLocation") or aggregated_data.get("current_location", "Unknown")
    
    # Handle date/time - check both naming conventions
    current_roleplay = aggregated_data.get("currentRoleplay") or aggregated_data.get("current_roleplay", {})
    year = current_roleplay.get("CurrentYear") or aggregated_data.get("year", "1040")
    month = current_roleplay.get("CurrentMonth") or aggregated_data.get("month", "6")
    day = current_roleplay.get("CurrentDay") or aggregated_data.get("day", "15")
    time_of_day = aggregated_data.get("timeOfDay") or aggregated_data.get("time_of_day", "Morning")
    
    # Get NPCs - handle both naming conventions
    introduced_npcs = aggregated_data.get("npcsPresent") or aggregated_data.get("introduced_npcs", [])
    
    # Build base text
    date_line = f"- It is {year}, {month} {day}, {time_of_day}.\n"
    location_line = f"- Current location: {current_location}\n"
    
    # Build NPC section
    npc_lines = ["Introduced NPCs in the area:"]
    for npc in introduced_npcs[:5]:
        # Handle different NPC data structures
        if isinstance(npc, dict):
            npc_loc = npc.get("current_location", current_location)
            npc_name = npc.get("name") or npc.get("npc_name", "Unnamed NPC")
            npc_lines.append(f"  - {npc_name} is at {npc_loc}")
    
    if introduced_npcs:
        npc_section = "\n".join(npc_lines)
    else:
        npc_section = "No NPCs currently in the area."
    
    text = f"{date_line}{location_line}\n{npc_section}\n"
    
    # Add environment description
    environment_desc = current_roleplay.get("EnvironmentDesc")
    if environment_desc:
        text += f"\nEnvironment:\n{environment_desc}\n"
    
    # Add player role
    player_role = current_roleplay.get("PlayerRole")
    if player_role:
        text += f"\nPlayer Role:\n{player_role}\n"
    
    # Add active events if present
    if aggregated_data.get("activeEvents"):
        event_names = [event.get("event_name", "Unknown Event") for event in aggregated_data["activeEvents"]]
        text += f"\nActive Events: {', '.join(event_names)}\n"
    
    # Add active quests if present
    if aggregated_data.get("activeQuests"):
        quest_names = [quest.get("quest_name", "Unknown Quest") for quest in aggregated_data["activeQuests"]]
        text += f"\nActive Quests: {', '.join(quest_names)}\n"
    
    # Add player stats summary if available
    if aggregated_data.get("playerStats"):
        stats = aggregated_data["playerStats"]
        if stats:
            dominant_stat = max(stats.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
            text += f"\nPlayer's dominant trait: {dominant_stat[0]} ({dominant_stat[1]})\n"
    
    # ADD PRESET STORY SECTION
    if aggregated_data.get("preset_story"):
        preset = aggregated_data["preset_story"]
        text += f"\n==== PRESET STORY ACTIVE ====\n"
        text += f"Story: {preset.get('story_id', 'Unknown')}\n"
        text += f"Setting: {preset.get('setting', 'Unknown')}, Year: {preset.get('year', 'Modern')}\n"
        text += f"Act {preset.get('current_act', 1)}"
        
        if preset.get('current_beat'):
            text += f", Beat: {preset['current_beat']}"
        text += "\n"
        
        # Add location-specific lore if available
        if preset.get('current_location_lore'):
            lore = preset['current_location_lore']
            if lore.get('type') == 'specific_location':
                text += f"\nLocation Type: {lore['location'].get('location_type', 'Unknown')}\n"
                if lore['location'].get('access_level'):
                    text += f"Access Level: {lore['location']['access_level']}\n"
            elif lore.get('type') == 'district':
                text += f"\nDistrict: {lore['district']['name']}\n"
        
        # Add any special location rules
        if aggregated_data.get('location_special_rules'):
            rules = aggregated_data['location_special_rules']
            text += f"\nSpecial Rules Active: {', '.join(rules)}\n"
    
    # Add preset constraints if active
    if aggregated_data.get("preset_constraints"):
        text += f"\n{aggregated_data['preset_constraints']}\n"
    
    # Add generation hints if available
    if aggregated_data.get("generation_hints"):
        hints = aggregated_data["generation_hints"]
        text += "\n==== GENERATION REMINDERS ====\n"
        if hints.get('correct_usage'):
            text += "Correct usage:\n"
            for key, value in hints['correct_usage'].items():
                text += f"  - {key}: {value}\n"
    
    # Add context optimization note
    text += "\n\n<!-- Context optimized with unified context system -->"
    
    # Add relevance note if NPCs have relevance scores
    has_relevance = any(
        "relevance_score" in npc or "relevance" in npc 
        for npc in introduced_npcs 
        if isinstance(npc, dict)
    )
    if has_relevance:
        text += "\n<!-- NPCs sorted by relevance to current context -->"
    
    # Add preset story warning if active
    if aggregated_data.get("preset_story"):
        text += "\n<!-- PRESET STORY ACTIVE - Consistency rules MUST be followed -->"
    
    return text

###############################################################################
# Maintenance and Migration
###############################################################################

async def run_context_maintenance(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Run maintenance tasks for context optimization.
    """
    context_service = await get_context_service(user_id, conversation_id)
    return await context_service.run_maintenance()


async def migrate_old_context_to_new(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Migrate data from old context system to the new optimized system.
    """
    try:
        from logic.aggregator_sdk import get_aggregated_roleplay_context as old_get_context
        old_context = await old_get_context(user_id, conversation_id)

        context_service = await get_context_service(user_id, conversation_id)
        from context.memory_manager import get_memory_manager
        memory_manager = await get_memory_manager(user_id, conversation_id)

        memory_migrations = 0
        if "memories" in old_context:
            for mem in old_context["memories"]:
                content = mem.get("content") or mem.get("text") or ""
                memory_type = mem.get("type", "observation")
                if content:
                    await memory_manager.add_memory(
                        content=content,
                        memory_type=memory_type,
                        importance=0.7
                    )
                    memory_migrations += 1

        npc_migrations = 0
        config = await get_config()
        if config.is_enabled("use_vector_search"):
            from context.vector_service import get_vector_service
            vector_service = await get_vector_service(user_id, conversation_id)
            if "introduced_npcs" in old_context:
                npc_migrations = len(old_context["introduced_npcs"])

        return {
            "memory_migrations": memory_migrations,
            "npc_migrations": npc_migrations,
            "success": True
        }

    except Exception as e:
        logger.error(f"Error during context migration: {e}", exc_info=True)
        return {
            "error": str(e),
            "success": False
        }


###############################################################################
# Optional Monkey Patching
###############################################################################

def apply_context_optimizations() -> bool:
    """
    Apply context optimizations by monkey patching existing aggregator functions.
    """
    import sys
    aggregator_sdk = sys.modules.get("logic.aggregator_sdk")
    if not aggregator_sdk:
        logger.warning("Could not find aggregator_sdk module for patching")
        return False

    original_get_context = getattr(aggregator_sdk, "get_aggregated_roleplay_context", None)
    original_build_text = getattr(aggregator_sdk, "build_aggregator_text", None)
    if not original_get_context or not original_build_text:
        logger.warning("Required aggregator functions not found in aggregator_sdk")
        return False

    setattr(aggregator_sdk, "get_aggregated_roleplay_context", get_aggregated_roleplay_context)
    setattr(aggregator_sdk, "build_aggregator_text", build_aggregator_text)
    setattr(aggregator_sdk, "ContextCache", OptimizedContextCache)
    setattr(aggregator_sdk, "IncrementalContextManager", OptimizedIncrementalContextManager)
    setattr(aggregator_sdk, "get_optimized_context", get_optimized_context)
    setattr(aggregator_sdk, "run_context_maintenance", run_context_maintenance)

    logger.info("Applied context optimizations via monkey patching.")
    return True


async def update_context_with_universal_updates(
    context: dict,
    universal_updates: dict,
    user_id: str,
    conversation_id: str
) -> dict:
    """
    Update the context with universal updates while maintaining consistency.
    """
    try:
        updated_context = context.copy()

        # NPC Updates
        if "npc_updates" in universal_updates:
            for npc_update in universal_updates["npc_updates"]:
                npc_id = npc_update.get("npc_id")
                if not npc_id:
                    continue
                if "npcs" not in updated_context:
                    updated_context["npcs"] = {}
                if npc_id not in updated_context["npcs"]:
                    updated_context["npcs"][npc_id] = {}

                # Update NPC stats
                if "stats" in npc_update:
                    updated_context["npcs"][npc_id]["stats"] = npc_update["stats"]

                # Update NPC location
                if "location" in npc_update:
                    updated_context["npcs"][npc_id]["current_location"] = npc_update["location"]

                # Update NPC memory
                if "memory" in npc_update:
                    if "memory" not in updated_context["npcs"][npc_id]:
                        updated_context["npcs"][npc_id]["memory"] = []
                    updated_context["npcs"][npc_id]["memory"].extend(npc_update["memory"])

        # Relationship updates
        if "social_links" in universal_updates:
            if "relationships" not in updated_context:
                updated_context["relationships"] = {}

            for link in universal_updates["social_links"]:
                e1_type = link.get("entity1_type")
                e1_id = link.get("entity1_id")
                e2_type = link.get("entity2_type")
                e2_id = link.get("entity2_id")

                if not all([e1_type, e1_id, e2_type, e2_id]):
                    continue

                link_key = f"{e1_type}_{e1_id}_{e2_type}_{e2_id}"
                updated_context["relationships"][link_key] = {
                    "type": link.get("link_type", "neutral"),
                    "level": link.get("link_level", 0),
                    "group_context": link.get("group_context", ""),
                    "events": link.get("events", [])
                }

        # Quest updates
        if "quest_updates" in universal_updates:
            if "quests" not in updated_context:
                updated_context["quests"] = {}
            for quest in universal_updates["quest_updates"]:
                quest_id = quest.get("quest_id")
                if not quest_id:
                    continue
                updated_context["quests"][quest_id] = {
                    "status": quest.get("status", "In Progress"),
                    "progress": quest.get("progress_detail", ""),
                    "giver": quest.get("quest_giver", ""),
                    "reward": quest.get("reward", "")
                }

        # Inventory updates
        if "inventory_updates" in universal_updates:
            if "inventory" not in updated_context:
                updated_context["inventory"] = {"items": {}, "removed_items": []}

            for item in universal_updates["inventory_updates"].get("added_items", []):
                if isinstance(item, str):
                    item_name = item
                    item_data = {"name": item_name}
                else:
                    item_name = item.get("name")
                    item_data = item

                if item_name:
                    updated_context["inventory"]["items"][item_name] = item_data

            for item in universal_updates["inventory_updates"].get("removed_items", []):
                if isinstance(item, str):
                    item_name = item
                else:
                    item_name = item.get("name")
                if item_name:
                    if item_name in updated_context["inventory"]["items"]:
                        del updated_context["inventory"]["items"][item_name]
                    updated_context["inventory"]["removed_items"].append(item_name)

        # Activity updates
        if "activity_updates" in universal_updates:
            if "activities" not in updated_context:
                updated_context["activities"] = []
            for activity in universal_updates["activity_updates"]:
                if "activity_name" in activity:
                    updated_context["activities"].append({
                        "name": activity["activity_name"],
                        "purpose": activity.get("purpose", {}),
                        "stats": activity.get("stat_integration", {}),
                        "intensity": activity.get("intensity_tier", 0),
                        "setting": activity.get("setting_variant", "")
                    })

        updated_context["last_modified"] = datetime.now().isoformat()
        return updated_context

    except Exception as e:
        logger.error(f"Error updating context with universal updates: {e}", exc_info=True)
        return context  # Return original context on error
