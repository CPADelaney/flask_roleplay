# lore/main.py

import logging
from datetime import datetime
import json
from typing import Dict, List, Any, Optional

from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

from lore.core.base_manager import BaseLoreManager
from lore.core.registry import ManagerRegistry
from lore.utils.theming import MatriarchalThemingUtils

class MatriarchalLoreSystem(BaseLoreManager):
    """
    Consolidated master class that integrates all lore systems with a matriarchal theme focus.
    Acts as the primary interface for all lore generation and management.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        
        # Create registry for component managers
        self.registry = ManagerRegistry(user_id, conversation_id)
        self.cache_namespace = "matriarchal_lore"
    
    async def ensure_initialized(self):
        """Ensure system is initialized"""
        if not self.initialized:
            await super().ensure_initialized()
            # Initialize core subsystems - others will be loaded on demand
            await self.registry.get_lore_dynamics()
            await self.registry.get_geopolitical_manager()
            self.initialized = True
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="handle_narrative_event",
        action_description="Handling narrative event impacts",
        id_from_context=lambda ctx: "matriarchal_lore_system"
    )
    async def handle_narrative_event(
        self, 
        ctx,
        event_description: str,
        affected_location_id: int = None,
        player_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Handle impacts of a narrative event on the world
        
        Args:
            event_description: Description of the event that occurred
            affected_location_id: Optional ID of specifically affected location
            player_data: Optional player character data
            
        Returns:
            Dictionary with all updates applied
        """
        # Create run context
        run_ctx = self.create_run_context(ctx)
        
        # First apply matriarchal theming to the event description
        themed_event = MatriarchalThemingUtils.apply_matriarchal_theme("event", event_description, emphasis_level=1)
        
        # Create response structure
        response = {
            "event": themed_event,
            "original_event": event_description,
            "event_impact": await self._calculate_event_impact(themed_event),
            "lore_updates": None,
            "local_updates": None,
            "conflict_results": None
        }
        
        # Evolve general lore
        lore_dynamics = await self.registry.get_lore_dynamics()
        response["lore_updates"] = await lore_dynamics.evolve_lore_with_event(themed_event)
        
        # If a specific location is affected, update local lore
        if affected_location_id:
            local_lore_manager = await self.registry.get_local_lore_manager()
            response["local_updates"] = await local_lore_manager.evolve_location_lore(
                run_ctx, affected_location_id, themed_event
            )
        
        # If event has significant impact, update conflicts
        if response["event_impact"] > 6:
            world_politics = await self.registry.get_world_politics_manager()
            response["conflict_results"] = await world_politics.evolve_all_conflicts(run_ctx, days_passed=7)
        
        return response
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_complete_world",
        action_description="Generating complete matriarchal world lore",
        id_from_context=lambda ctx: "matriarchal_lore_system"
    )
    async def generate_complete_world(self, ctx, environment_desc: str) -> Dict[str, Any]:
        """
        Generate a complete world with matriarchal theming
        
        Args:
            environment_desc: Description of the environment/setting
            
        Returns:
            Dictionary containing the complete world lore
        """
        # Create run context
        run_ctx = self.create_run_context(ctx)
        
        # Track progress
        progress = {}
        
        # 1. Generate foundation lore through DynamicLoreGenerator
        dynamic_lore = DynamicLoreGenerator(self.user_id, self.conversation_id)
        
        # Generate foundation lore with matriarchal theming
        foundation_data = await dynamic_lore.initialize_world_lore(environment_desc)
        for key, content in foundation_data.items():
            foundation_data[key] = MatriarchalThemingUtils.apply_matriarchal_theme(key, content)
        progress["foundation_lore"] = foundation_data
        
        # 2. Generate factions with matriarchal power structures
        factions_data = await dynamic_lore.generate_factions(environment_desc, foundation_data)
        progress["factions"] = factions_data
        
        # 3. Generate cultural elements
        cultural_data = await dynamic_lore.generate_cultural_elements(environment_desc, factions_data)
        progress["cultural_elements"] = cultural_data
        
        # 4. Generate historical events emphasizing matriarchal history
        historical_data = await dynamic_lore.generate_historical_events(
            environment_desc, foundation_data, factions_data
        )
        progress["historical_events"] = historical_data
        
        # 5. Generate locations
        locations_data = await dynamic_lore.generate_locations(environment_desc, factions_data)
        progress["locations"] = locations_data
        
        # 6. Generate quest hooks
        quests_data = await dynamic_lore.generate_quest_hooks(factions_data, locations_data)
        progress["quests"] = quests_data
        
        # 7. Generate world nations
        geopolitical_manager = await self.registry.get_geopolitical_manager()
        nations = await geopolitical_manager.generate_world_nations(run_ctx)
        progress["nations"] = nations
        
        # 8. Generate religion
        religion_manager = await self.registry.get_religion_manager()
        religious_data = await religion_manager.generate_complete_faith_system(run_ctx)
        progress["religions"] = religious_data
        
        # 9. Generate international conflicts
        world_politics = await self.registry.get_world_politics_manager()
        conflicts = await world_politics.generate_initial_conflicts(run_ctx, count=3)
        progress["conflicts"] = conflicts
        
        # 10. Generate regional cultures
        culture_system = await self.registry.get_regional_culture_system()
        languages = await culture_system.generate_languages(run_ctx, count=3)
        progress["languages"] = languages
        
        return progress
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_world",
        action_description="Evolving world by days passed",
        id_from_context=lambda ctx: "matriarchal_lore_system"
    )
    async def evolve_world(self, ctx, days_passed: int = 30) -> Dict[str, Any]:
        """
        Evolve the entire world by simulating the passage of time
        
        Args:
            days_passed: Number of days to simulate
            
        Returns:
            Dictionary with all evolution results
        """
        # Create run context
        run_ctx = self.create_run_context(ctx)
        
        # Use the lore dynamics system to evolve the world
        lore_dynamics = await self.registry.get_lore_dynamics()
        return await lore_dynamics.evolve_world_over_time(ctx, days_passed)
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_world_state",
        action_description="Getting current world state",
        id_from_context=lambda ctx: "matriarchal_lore_system"
    )
    async def get_world_state(self, ctx) -> Dict[str, Any]:
        """
        Get a comprehensive view of the current world state
        
        Returns:
            Current world state
        """
        # Check cache first
        cached = self.get_cache("world_state")
        if cached:
            return cached
        
        # Create run context
        run_ctx = self.create_run_context(ctx)
        
        # Get basic world state
        async with await self._initialize_db_pool() as conn:
            # Query the WorldState table
            world_state = await conn.fetchrow("""
                SELECT * FROM WorldState 
                WHERE user_id = $1 AND conversation_id = $2
                LIMIT 1
            """, self.user_id, self.conversation_id)
            
            world_data = dict(world_state) if world_state else {
                'stability_index': 8,
                'narrative_tone': 'dramatic',
                'power_dynamics': 'strict_hierarchy',
                'power_hierarchy': {}
            }
            
            # Parse JSON fields
            if 'power_hierarchy' in world_data and world_data['power_hierarchy']:
                try:
                    world_data['power_hierarchy'] = json.loads(world_data['power_hierarchy'])
                except:
                    world_data['power_hierarchy'] = {}
        
        # Get nations
        geopolitical_manager = await self.registry.get_geopolitical_manager()
        nations = await geopolitical_manager.get_all_nations(run_ctx)
        
        # Get active conflicts
        world_politics = await self.registry.get_world_politics_manager()
        conflicts = await world_politics.get_active_conflicts(run_ctx)
        
        # Get a sample of locations and factions
        # Combine into single query for efficiency
        async with await self._initialize_db_pool() as conn:
            # Get major factions
            factions = await conn.fetch("""
                SELECT id, name, type, description 
                FROM Factions
                ORDER BY RANDOM()
                LIMIT 5
            """)
            
            faction_data = [dict(faction) for faction in factions]
            
            # Get recent events
            recent_events = await conn.fetch("""
                SELECT id, name, date_description, description
                FROM HistoricalEvents
                ORDER BY id DESC
                LIMIT 3
            """)
            
            event_data = [dict(event) for event in recent_events]
        
        # Compile result
        result = {
            "world_state": world_data,
            "nations": nations,
            "active_conflicts": conflicts,
            "major_factions": faction_data,
            "recent_events": event_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache result
        self.set_cache("world_state", result, ttl=3600)  # 1 hour TTL
        
        return result
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_additional_content",
        action_description="Generating additional content of type: {content_type}",
        id_from_context=lambda ctx: "matriarchal_lore_system"
    )
    async def generate_additional_content(
        self, 
        ctx, 
        content_type: str, 
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate additional content of a specific type using appropriate subsystem
        
        Args:
            content_type: Type of content to generate (faction, location, etc.)
            parameters: Optional parameters specific to the content type
            
        Returns:
            Generated content
        """
        # Create run context and default parameters
        run_ctx = self.create_run_context(ctx)
        parameters = parameters or {}
        
        # Map content types to manager methods
        content_generators = {
            "faction": (await self.registry.get_lore_dynamics()).generate_additional_faction,
            "location": (await self.registry.get_lore_dynamics()).generate_additional_locations,
            "cultural_element": (await self.registry.get_lore_dynamics()).generate_additional_cultural_elements,
            "nation": (await self.registry.get_geopolitical_manager()).generate_additional_nation,
            "conflict": (await self.registry.get_world_politics_manager()).generate_initial_conflicts,
            "language": (await self.registry.get_regional_culture_system()).generate_languages,
        }
        
        # Special case for religion content
        if content_type == "religion":
            religion_manager = await self.registry.get_religion_manager()
            
            if "pantheon_id" in parameters:
                # Generate components for existing pantheon
                if "component_type" in parameters:
                    component_type = parameters["component_type"]
                    pantheon_id = parameters["pantheon_id"]
                    
                    component_generators = {
                        "religious_practices": religion_manager.generate_religious_practices,
                        "holy_sites": religion_manager.generate_holy_sites,
                    }
                    
                    if component_type in component_generators:
                        return await component_generators[component_type](run_ctx, pantheon_id)
                    else:
                        return {"error": f"Unknown religion component type: {component_type}"}
            else:
                # Generate new pantheon
                return await religion_manager.generate_pantheon(run_ctx)
                
        # Special case for local lore
        elif content_type == "local_lore":
            if "location_id" not in parameters:
                return {"error": "location_id parameter required for local_lore generation"}
                
            local_lore_manager = await self.registry.get_local_lore_manager()
            location_id = parameters["location_id"]
            return await local_lore_manager.generate_location_lore(run_ctx, {"id": location_id})
        
        # Handle standard content types
        elif content_type in content_generators:
            generator_func = content_generators[content_type]
            
            # Handle count parameter for bulk generators
            if content_type in ["conflict", "language"] and "count" in parameters:
                return await generator_func(run_ctx, count=parameters["count"])
            else:
                return await generator_func(run_ctx, **parameters)
            
        else:
            return {"error": f"Unknown content type: {content_type}"}
    
    async def _calculate_event_impact(self, event_text: str) -> int:
        """
        Calculate the impact level of an event
        
        Args:
            event_text: The event description
            
        Returns:
            Impact level (1-10)
        """
        # Define impact keywords for different levels
        impact_keywords = {
            'high': [
                "catastrophe", "revolution", "assassination", "coronation", 
                "invasion", "war", "defeat", "victory", "disaster", "miracle"
            ],
            'medium': [
                "conflict", "dispute", "change", "election", "discovery", 
                "alliance", "treaty", "ceremony", "unveiling", "ritual"
            ],
            'low': [
                "minor", "small", "limited", "isolated", "contained",
                "private", "personal", "individual", "trivial"
            ]
        }
        
        # Count keyword occurrences more efficiently
        event_text_lower = event_text.lower()
        
        high_count = sum(1 for word in impact_keywords['high'] if word in event_text_lower)
        medium_count = sum(1 for word in impact_keywords['medium'] if word in event_text_lower)
        low_count = sum(1 for word in impact_keywords['low'] if word in event_text_lower)
        
        # Calculate impact based on keyword counts
        if high_count > 0:
            return min(10, 7 + min(high_count, 3))  # Max 10
        elif medium_count > 0:
            return min(7, 4 + min(medium_count, 3))  # Max 7
        elif low_count > 0:
            return min(4, 2 + min(low_count, 2))  # Max 4
        else:
            return 3  # Default moderate impact
    
    async def register_with_governance(self):
        """Register all lore subsystems with Nyx governance system."""
        # Register the main system
        await super().register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="matriarchal_lore_system",
            directive_text="Create and maintain a cohesive world with matriarchal themes and power dynamics.",
            scope="world_building",
            priority=DirectivePriority.HIGH
        )
        
        logging.info(f"MatriarchalLoreSystem registered with governance for user {self.user_id}, conversation {self.conversation_id}")

# -------------------------------------------------
# MATRIARCHAL THEMING UTILITIES
# -------------------------------------------------

# A dictionary of regex patterns to replacement strings for basic feminization
# \b ensures "god" won't become "Goddess" inside words like "good"
_FEMDOM_WORD_MAP = {
    r"\bgod\b": "Goddess",
    r"\bgods\b": "Goddesses",
    r"\bgodhood\b": "Goddesshood",
    r"\bking\b": "Queen",
    r"\bkings\b": "Queens",
    r"\bkingdom\b": "Queendom",
    r"\bprince\b": "princess",
    r"\bprinces\b": "princesses",
    r"\bfather\b": "mother",
    r"\bfathers\b": "mothers",
    r"\bhim\b": "her",
    r"\bhis\b": "her",
    r"\bhe\b": "she",
    r"\blord\b": "lady",
    r"\blords\b": "ladies",
    r"\bman\b": "woman",
    r"\bmen\b": "women",
}

# Random synonyms for a supreme feminine figure
_GODDESS_SYNONYMS = [
    "Supreme Goddess",
    "High Empress",
    "Great Matriarch",
    "Divine Mother",
    "Infinite Mistress of Creation",
]

# Default emphasis level for matriarchal theming (1=low, 3=very high)
_DEFAULT_EMPHASIS_LEVEL = 2


def _apply_basic_replacements(text: str) -> str:
    """
    Runs a set of regex-based replacements to feminize words/phrases.
    Respects case; if the original word is capitalized, keep it capitalized.
    """
    result = text

    for pattern_str, replacement_str in _FEMDOM_WORD_MAP.items():
        pattern = re.compile(pattern_str, re.IGNORECASE)

        def _replacement_func(match):
            original = match.group(0)
            # If the original word starts with uppercase, we uppercase the replacement's first letter
            if original and original[0].isupper():
                return replacement_str.capitalize()
            return replacement_str

        result = pattern.sub(_replacement_func, result)

    return result


def _ensure_goddess_reference(text: str) -> str:
    """
    If there's no mention of 'Goddess' or a similar figure, insert a default reference
    to a supreme feminine force at the end of the text.
    """
    if not re.search(r"(goddess|divine mother|matriarch|empress of creation)", text, re.IGNORECASE):
        chosen_title = random.choice(_GODDESS_SYNONYMS)
        insertion = (
            f"\n\nAt the cosmic center stands {chosen_title}, "
            "the eternal wellspring of existence. Her dominion weaves reality itself."
        )
        text += insertion

    return text


def _inject_contextual_lore(text: str, insertion: str, label: str = "", append_if_not_found: bool = True) -> str:
    """
    Insert 'insertion' text after a specific label or heading (case-insensitive).
    If label not found and append_if_not_found is True, just append at the end.
    """
    if not label:
        # No label provided, always append at the end
        return text.strip() + "\n\n" + insertion

    pattern = re.compile(rf"({re.escape(label)})", re.IGNORECASE)
    match = pattern.search(text)

    if match:
        # Insert after the matched label
        idx = match.end()
        # Insert a blank line before the snippet
        return text[:idx] + "\n\n" + insertion + text[idx:]
    else:
        if append_if_not_found:
            return text.strip() + "\n\n" + insertion
        else:
            # Return unchanged if not found and we don't want to append
            return text


def _emphasize_feminine_power(text: str, emphasis_level: int = _DEFAULT_EMPHASIS_LEVEL) -> str:
    """
    Add extra lines emphasizing matriarchal/feminine power, depending on emphasis_level.
    1 = minimal emphasis
    2 = moderate emphasis
    3 = strong emphasis
    """
    if emphasis_level >= 2:
        text += (
            "\n\nThroughout every realm, the ascendancy of womanhood underpins all aspects of life, "
            "unquestioned and unassailable."
        )
    if emphasis_level >= 3:
        text += (
            "\n\nNo force—mortal or divine—can transcend the supremacy of the feminine essence, "
            "the primal wellspring of creation itself."
        )
    return text
