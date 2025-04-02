# lore/lore_generator.py

"""
Lore Generator Components - Consolidated

This module provides components for generating and evolving lore content,
including dynamic generation, evolution, and component generation.
"""

import logging
import json
import asyncio
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime
from dataclasses import dataclass

# Nyx governance integration
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting

# Import data access layer
from .data_access import (
    NPCDataAccess,
    LocationDataAccess,
    FactionDataAccess,
    LoreKnowledgeAccess
)

# Generation tools
from agents.run_context import RunContextWrapper
from .lore_tools import (
    generate_foundation_lore,
    generate_factions,
    generate_cultural_elements,
    generate_historical_events,
    generate_locations,
    generate_quest_hooks
)

# Import error handling
from .error_manager import LoreError, ErrorHandler, handle_errors

logger = logging.getLogger(__name__)

#---------------------------
# Component Generator Base Classes
#---------------------------

@dataclass
class ComponentConfig:
    """Configuration for component generation"""
    min_length: int = 100
    max_length: int = 500
    style: str = "descriptive"
    tone: str = "neutral"
    include_metadata: bool = True

class BaseGenerator:
    """Base class for all generator components."""
    
    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None):
        """
        Initialize the base generator component.
        
        Args:
            user_id: Optional user ID for filtering
            conversation_id: Optional conversation ID for filtering
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = None
        self.initialized = False
        self._cache = {}
        
        # Initialize data access components
        self.npc_data = NPCDataAccess(user_id, conversation_id)
        self.location_data = LocationDataAccess(user_id, conversation_id)
        self.faction_data = FactionDataAccess(user_id, conversation_id)
        self.lore_knowledge = LoreKnowledgeAccess(user_id, conversation_id)
    
    async def initialize(self) -> bool:
        """
        Initialize the generator component.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.initialized:
            return True
            
        try:
            # Initialize governance
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
            
            # Initialize data access components
            await self.npc_data.initialize()
            await self.location_data.initialize()
            await self.faction_data.initialize()
            await self.lore_knowledge.initialize()
            
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing {self.__class__.__name__}: {e}")
            return False
    
    async def initialize_governance(self) -> bool:
        """Initialize governance connection."""
        try:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
            return True
        except Exception as e:
            logger.error(f"Error initializing governance for {self.__class__.__name__}: {e}")
            return False
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            # Cleanup data access components
            await self.npc_data.cleanup()
            await self.location_data.cleanup()
            await self.faction_data.cleanup()
            await self.lore_knowledge.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a cached component if available"""
        return self._cache.get(key)
    
    def _cache_component(self, key: str, component: Dict[str, Any]):
        """Cache a generated component"""
        self._cache[key] = component

class ComponentGenerator(BaseGenerator):
    """Base class for all component generators"""
    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None, 
                 config: Optional[ComponentConfig] = None):
        super().__init__(user_id, conversation_id)
        self.config = config or ComponentConfig()
    
    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a component with the given context"""
        raise NotImplementedError

class CharacterGenerator(ComponentGenerator):
    """Generator for character components"""
    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = f"character_{context.get('name', '')}"
        if cached := self._get_cached(cache_key):
            return cached
            
        component = {
            "type": "character",
            "name": context.get("name", "Unknown Character"),
            "description": await self._generate_description(context),
            "traits": await self._generate_traits(context),
            "background": await self._generate_background(context),
            "relationships": await self._generate_relationships(context),
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "version": "1.0"
            } if self.config.include_metadata else {}
        }
        
        self._cache_component(cache_key, component)
        return component
    
    async def _generate_description(self, context: Dict[str, Any]) -> str:
        """Generate a detailed description for a character."""
        try:
            # Extract relevant context
            name = context.get("name", "Unknown Character")
            role = context.get("role", "Unknown Role")
            background = context.get("background", {})
            
            # Build description components
            run_ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
            
            # Sample implementation - would use more sophisticated generation in practice
            description = f"{name} is a {role}. "
            if "origin" in background:
                description += f"They come from {background['origin']}. "
            if "appearance" in context:
                description += context["appearance"]
            
            return description
        except Exception as e:
            logger.error(f"Error generating character description: {str(e)}")
            return f"Description for {context.get('name', 'Unknown Character')}"
    
    async def _generate_traits(self, context: Dict[str, Any]) -> List[str]:
        """Generate character traits."""
        # Sample implementation - would use more sophisticated generation in practice
        try:
            return context.get("predefined_traits", ["Intelligent", "Resourceful", "Cautious"])
        except Exception as e:
            logger.error(f"Error generating character traits: {str(e)}")
            return ["Resourceful", "Adaptable"]
    
    async def _generate_background(self, context: Dict[str, Any]) -> str:
        """Generate character background."""
        # Sample implementation - would use more sophisticated generation in practice
        try:
            return context.get("predefined_background", "A mysterious past shrouded in secrecy.")
        except Exception as e:
            logger.error(f"Error generating character background: {str(e)}")
            return "Background unknown."
    
    async def _generate_relationships(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate character relationships."""
        # Sample implementation - would use more sophisticated generation in practice
        try:
            return context.get("predefined_relationships", [])
        except Exception as e:
            logger.error(f"Error generating character relationships: {str(e)}")
            return []

class LocationGenerator(ComponentGenerator):
    """Generator for location components"""
    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = f"location_{context.get('name', '')}"
        if cached := self._get_cached(cache_key):
            return cached
            
        component = {
            "type": "location",
            "name": context.get("name", "Unknown Location"),
            "description": await self._generate_description(context),
            "climate": await self._generate_climate(context),
            "geography": await self._generate_geography(context),
            "culture": await self._generate_culture(context),
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "version": "1.0"
            } if self.config.include_metadata else {}
        }
        
        self._cache_component(cache_key, component)
        return component
    
    async def _generate_description(self, context: Dict[str, Any]) -> str:
        """Generate a detailed description for a location."""
        # Sample implementation
        try:
            name = context.get("name", "Unknown Location")
            location_type = context.get("type", "area")
            
            description = f"{name} is a {location_type}. "
            if "features" in context:
                description += f"It features {', '.join(context['features'])}. "
            if "atmosphere" in context:
                description += context["atmosphere"]
                
            return description
        except Exception as e:
            logger.error(f"Error generating location description: {str(e)}")
            return f"Description of {context.get('name', 'Unknown Location')}"
    
    async def _generate_climate(self, context: Dict[str, Any]) -> str:
        """Generate climate information."""
        # Sample implementation
        try:
            return context.get("climate", "Temperate")
        except Exception as e:
            logger.error(f"Error generating climate: {str(e)}")
            return "Temperate"
    
    async def _generate_geography(self, context: Dict[str, Any]) -> str:
        """Generate geographical information."""
        # Sample implementation
        try:
            return context.get("geography", "Varied terrain")
        except Exception as e:
            logger.error(f"Error generating geography: {str(e)}")
            return "Varied terrain"
    
    async def _generate_culture(self, context: Dict[str, Any]) -> str:
        """Generate cultural information."""
        # Sample implementation
        try:
            return context.get("culture", "Diverse")
        except Exception as e:
            logger.error(f"Error generating culture: {str(e)}")
            return "Diverse"

class EventGenerator(ComponentGenerator):
    """Generator for event components"""
    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = f"event_{context.get('name', '')}"
        if cached := self._get_cached(cache_key):
            return cached
            
        component = {
            "type": "event",
            "name": context.get("name", "Unknown Event"),
            "description": await self._generate_description(context),
            "date": await self._generate_date(context),
            "participants": await self._generate_participants(context),
            "consequences": await self._generate_consequences(context),
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "version": "1.0"
            } if self.config.include_metadata else {}
        }
        
        self._cache_component(cache_key, component)
        return component
    
    async def _generate_description(self, context: Dict[str, Any]) -> str:
        """Generate a detailed description for an event."""
        # Sample implementation
        try:
            name = context.get("name", "Unknown Event")
            event_type = context.get("type", "occurrence")
            
            description = f"{name} was a significant {event_type}. "
            if "details" in context:
                if "summary" in context["details"]:
                    description += context["details"]["summary"]
                
            return description
        except Exception as e:
            logger.error(f"Error generating event description: {str(e)}")
            return f"Description of {context.get('name', 'Unknown Event')}"
    
    async def _generate_date(self, context: Dict[str, Any]) -> str:
        """Generate date information."""
        # Sample implementation
        try:
            year = context.get("year")
            month = context.get("month")
            day = context.get("day")
            
            if all([year, month, day]):
                return f"{year}-{month:02d}-{day:02d}"
            elif year:
                return f"Year {year}"
            else:
                return "Unknown date"
        except Exception as e:
            logger.error(f"Error generating event date: {str(e)}")
            return "Unknown date"
    
    async def _generate_participants(self, context: Dict[str, Any]) -> List[str]:
        """Generate participant information."""
        # Sample implementation
        try:
            return context.get("participants", [])
        except Exception as e:
            logger.error(f"Error generating event participants: {str(e)}")
            return []
    
    async def _generate_consequences(self, context: Dict[str, Any]) -> List[str]:
        """Generate consequence information."""
        # Sample implementation
        try:
            return context.get("consequences", [])
        except Exception as e:
            logger.error(f"Error generating event consequences: {str(e)}")
            return []

class ComponentGeneratorFactory:
    """Factory for creating component generators"""
    @staticmethod
    def create_generator(component_type: str, user_id: Optional[int] = None, 
                         conversation_id: Optional[int] = None,
                         config: Optional[ComponentConfig] = None) -> ComponentGenerator:
        """Create a component generator of the specified type."""
        generators = {
            "character": CharacterGenerator,
            "location": LocationGenerator,
            "event": EventGenerator
        }
        
        generator_class = generators.get(component_type.lower())
        if not generator_class:
            raise ValueError(f"Unknown component type: {component_type}")
            
        return generator_class(user_id, conversation_id, config)

#---------------------------
# World and Setting Generators
#---------------------------

class WorldBuilder(BaseGenerator):
    """Generates foundation world lore."""
    
    async def initialize_world_lore(self, environment_desc: str) -> Dict[str, Any]:
        """
        Initialize core foundation lore (cosmology, magic system, world history, etc.)
        
        Args:
            environment_desc: Short textual description of the environment
            
        Returns:
            Dict containing the foundation lore fields
        """
        if not self.initialized:
            await self.initialize()
            
        # First, check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="initialize_world_lore",
            action_details={"environment_desc": environment_desc}
        )
        
        if not permission["approved"]:
            logging.warning(f"World lore initialization not approved: {permission.get('reasoning')}")
            return {"error": permission.get("reasoning"), "approved": False}
            
        # Create run context
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })

        # Call the function tool with governance integration
        foundation_data = await generate_foundation_lore(run_ctx, environment_desc)

        # Store in database
        for category, desc in foundation_data.items():
            # e.g. "category" might be "cosmology", "magic_system", ...
            await self._store_world_lore(
                name=f"{category.title()} of {await self.get_setting_name()}",
                category=category,
                description=desc,
                significance=8,
                tags=[category, "foundation", "world_building"]
            )

        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "initialize_world_lore",
                "description": f"Generated foundation lore for {environment_desc[:50]}"
            },
            result={
                "categories": list(foundation_data.keys()),
                "world_name": await self.get_setting_name()
            }
        )

        return foundation_data
    
    async def _store_world_lore(self, name: str, category: str, 
                              description: str, significance: int,
                              tags: List[str]) -> int:
        """
        Store world lore in the database.
        
        Args:
            name: Lore name
            category: Lore category
            description: Lore description
            significance: Lore significance (1-10)
            tags: List of tags
            
        Returns:
            ID of the created lore
        """
        try:
            query = """
                INSERT INTO WorldLore (
                    user_id, conversation_id, name, category,
                    description, significance, tags, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                RETURNING id
            """
            
            # Convert tags to JSON string if necessary
            if not isinstance(tags, str):
                tags_json = json.dumps(tags)
            else:
                tags_json = tags
            
            # Execute query
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    lore_id = await conn.fetchval(
                        query,
                        self.user_id,
                        self.conversation_id,
                        name,
                        category,
                        description,
                        significance,
                        tags_json
                    )
                    
                    return lore_id
                    
        except Exception as e:
            logger.error(f"Error storing world lore: {e}")
            return 0
    
    async def get_setting_name(self) -> str:
        """
        Get the current setting name.
        
        Returns:
            Setting name
        """
        try:
            query = """
                SELECT value FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentSetting'
                LIMIT 1
            """
            
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    setting_name = await conn.fetchval(
                        query,
                        self.user_id,
                        self.conversation_id
                    )
                    
                    if setting_name:
                        return setting_name
                    else:
                        return "The Setting"
                    
        except Exception as e:
            logger.error(f"Error getting setting name: {e}")
            return "The Setting"

    async def get_connection_pool(self):
        """Get database connection pool."""
        # This would need to be implemented based on your database connection implementation
        from db.connection import get_connection_pool
        return await get_connection_pool()

class FactionGenerator(BaseGenerator):
    """Generates faction and related lore content."""
    
    async def generate_factions(self, environment_desc: str, 
                              world_lore: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate 3-5 distinct factions referencing the environment description.
        
        Args:
            environment_desc: Text describing environment or setting
            world_lore: The dictionary from initialize_world_lore
            
        Returns:
            A list of faction dictionaries
        """
        if not self.initialized:
            await self.initialize()
            
        # First, check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="generate_factions",
            action_details={"environment_desc": environment_desc}
        )
        
        if not permission["approved"]:
            logging.warning(f"Faction generation not approved: {permission.get('reasoning')}")
            return []
            
        # Typically we want the 'social_structure' from foundation_data
        social_structure = world_lore.get("social_structure", "")

        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })

        # Produce a list of faction dicts via our sub-agent
        factions_data = await generate_factions(run_ctx, environment_desc, social_structure)

        # Store each in the DB
        for faction in factions_data:
            try:
                await self._store_faction(faction)
            except Exception as e:
                logging.error(f"Error storing faction '{faction['name']}': {e}")

        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "generate_factions",
                "description": f"Generated {len(factions_data)} factions"
            },
            result={
                "faction_count": len(factions_data),
                "faction_names": [f.get("name", "Unknown") for f in factions_data]
            }
        )

        return factions_data
    
    async def generate_cultural_elements(self, environment_desc: str, 
                                      factions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate cultural elements referencing environment + the names of the existing factions.
        
        Args:
            environment_desc: Text describing environment
            factions: List of faction dictionaries
            
        Returns:
            List of cultural element dictionaries
        """
        if not self.initialized:
            await self.initialize()
            
        # First, check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="generate_cultural_elements",
            action_details={"environment_desc": environment_desc}
        )
        
        if not permission["approved"]:
            logging.warning(f"Cultural elements generation not approved: {permission.get('reasoning')}")
            return []
            
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        faction_names = ", ".join([f.get("name", "Unknown") for f in factions])

        cultural_data = await generate_cultural_elements(run_ctx, environment_desc, faction_names)

        # Store them
        for element in cultural_data:
            try:
                await self._store_cultural_element(element)
            except Exception as e:
                logging.error(f"Error storing cultural element '{element.get('name','unknown')}': {e}")

        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "generate_cultural_elements",
                "description": f"Generated {len(cultural_data)} cultural elements"
            },
            result={
                "element_count": len(cultural_data),
                "element_types": list(set([e.get("type", "unknown") for e in cultural_data]))
            }
        )

        return cultural_data
    
    async def generate_historical_events(self, environment_desc: str, 
                                      foundation_data: Dict[str, Any], 
                                      factions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate 5-7 major historical events referencing environment + existing 'world_history' + faction names.
        
        Args:
            environment_desc: Environment description text
            foundation_data: Foundation lore dictionary
            factions: List of faction dictionaries
            
        Returns:
            List of historical event dictionaries
        """
        if not self.initialized:
            await self.initialize()
            
        # First, check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="generate_historical_events",
            action_details={"environment_desc": environment_desc}
        )
        
        if not permission["approved"]:
            logging.warning(f"Historical events generation not approved: {permission.get('reasoning')}")
            return []
            
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        # We can feed in the previously generated 'world_history'
        world_history = foundation_data.get("world_history", "")
        faction_names = ", ".join([f.get("name","Unknown") for f in factions])

        events_data = await generate_historical_events(run_ctx, environment_desc, world_history, faction_names)

        # Then store them
        for event in events_data:
            try:
                await self._store_historical_event(event)
            except Exception as e:
                logging.error(f"Error storing historical event '{event.get('name','Unknown')}': {e}")

        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "generate_historical_events",
                "description": f"Generated {len(events_data)} historical events"
            },
            result={
                "event_count": len(events_data),
                "significant_events": [e.get("name") for e in events_data if e.get("significance", 0) > 7]
            }
        )

        return events_data
    
    async def generate_locations(self, environment_desc: str, 
                              factions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate 5-8 significant locations referencing environment + faction names.
        
        Args:
            environment_desc: Environment description text
            factions: List of faction dictionaries
            
        Returns:
            List of location dictionaries
        """
        if not self.initialized:
            await self.initialize()
            
        # First, check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="generate_locations",
            action_details={"environment_desc": environment_desc}
        )
        
        if not permission["approved"]:
            logging.warning(f"Locations generation not approved: {permission.get('reasoning')}")
            return []
            
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        faction_names = ", ".join([f.get("name","Unknown") for f in factions])

        locations_data = await generate_locations(run_ctx, environment_desc, faction_names)

        # Store each location
        for loc in locations_data:
            try:
                # Create the location record
                location_id = await self._store_location(loc)

                # Add location lore
                controlling_faction = loc.get("controlling_faction")
                hidden_secrets = loc.get("hidden_secrets", [])
                founding_story = f"Founded as a {loc['type']}."

                await self._store_location_lore(
                    location_id=location_id,
                    founding_story=founding_story,
                    hidden_secrets=hidden_secrets,
                    local_legends=[],
                    historical_significance=loc.get("strategic_importance", "")
                )

                # Record controlling_faction if needed
                if controlling_faction:
                    await self._connect_faction_to_location(location_id, controlling_faction)
            except Exception as e:
                logging.error(f"Error storing location '{loc.get('name','Unknown')}': {e}")

        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "generate_locations",
                "description": f"Generated {len(locations_data)} locations"
            },
            result={
                "location_count": len(locations_data),
                "location_types": list(set([l.get("type", "unknown") for l in locations_data]))
            }
        )

        return locations_data
    
    async def generate_quest_hooks(self, factions: List[Dict[str, Any]], 
                                 locations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate 5-7 quest hooks referencing existing factions + location names.
        
        Args:
            factions: List of faction dictionaries
            locations: List of location dictionaries
            
        Returns:
            List of quest hook dictionaries
        """
        if not self.initialized:
            await self.initialize()
            
        # First, check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="generate_quest_hooks",
            action_details={"faction_count": len(factions), "location_count": len(locations)}
        )
        
        if not permission["approved"]:
            logging.warning(f"Quest hooks generation not approved: {permission.get('reasoning')}")
            return []
            
        run_ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        
        faction_names = ", ".join([f.get("name","Unknown") for f in factions])
        location_names = ", ".join([l.get("name","Unknown") for l in locations])

        quests_data = await generate_quest_hooks(run_ctx, faction_names, location_names)

        # Store them
        for quest in quests_data:
            try:
                await self._store_quest(quest)
            except Exception as e:
                logging.error(f"Error storing quest hook '{quest.get('quest_name','Unknown')}': {e}")

        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "generate_quest_hooks",
                "description": f"Generated {len(quests_data)} quest hooks"
            },
            result={
                "quest_count": len(quests_data),
                "quest_difficulties": list(set([q.get("difficulty", 0) for q in quests_data]))
            }
        )

        return quests_data
    
    async def _store_faction(self, faction_data: Dict[str, Any]) -> int:
        """Store a faction in the database."""
        try:
            query = """
                INSERT INTO Factions (
                    user_id, conversation_id, name, type,
                    description, values, goals, headquarters,
                    rivals, allies, hierarchy_type, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
                RETURNING id
            """
            
            # Extract values from faction data
            name = faction_data.get("name", "Unknown Faction")
            faction_type = faction_data.get("type", "organization")
            description = faction_data.get("description", "")
            values = faction_data.get("values", [])
            goals = faction_data.get("goals", [])
            headquarters = faction_data.get("headquarters")
            rivals = faction_data.get("rivals", [])
            allies = faction_data.get("allies", [])
            hierarchy_type = faction_data.get("hierarchy_type")
            
            # Convert lists to JSON if needed
            values_json = json.dumps(values) if not isinstance(values, str) else values
            goals_json = json.dumps(goals) if not isinstance(goals, str) else goals
            rivals_json = json.dumps(rivals) if not isinstance(rivals, str) and rivals is not None else rivals
            allies_json = json.dumps(allies) if not isinstance(allies, str) and allies is not None else allies
            
            # Execute query
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    faction_id = await conn.fetchval(
                        query,
                        self.user_id,
                        self.conversation_id,
                        name,
                        faction_type,
                        description,
                        values_json,
                        goals_json,
                        headquarters,
                        rivals_json,
                        allies_json,
                        hierarchy_type
                    )
                    
                    return faction_id
                    
        except Exception as e:
            logger.error(f"Error storing faction: {e}")
            return 0
    
    async def _store_cultural_element(self, element_data: Dict[str, Any]) -> int:
        """Store a cultural element in the database."""
        try:
            query = """
                INSERT INTO CulturalElements (
                    user_id, conversation_id, name, type,
                    description, practiced_by, significance,
                    historical_origin, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                RETURNING id
            """
            
            # Extract values from element data
            name = element_data.get("name", "Unknown Element")
            element_type = element_data.get("type", "tradition")
            description = element_data.get("description", "")
            practiced_by = element_data.get("practiced_by", [])
            significance = element_data.get("significance", 5)
            historical_origin = element_data.get("historical_origin")
            
            # Convert lists to JSON if needed
            practiced_by_json = json.dumps(practiced_by) if not isinstance(practiced_by, str) else practiced_by
            
            # Execute query
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    element_id = await conn.fetchval(
                        query,
                        self.user_id,
                        self.conversation_id,
                        name,
                        element_type,
                        description,
                        practiced_by_json,
                        significance,
                        historical_origin
                    )
                    
                    return element_id
                    
        except Exception as e:
            logger.error(f"Error storing cultural element: {e}")
            return 0
    
    async def _store_historical_event(self, event_data: Dict[str, Any]) -> int:
        """Store a historical event in the database."""
        try:
            query = """
                INSERT INTO HistoricalEvents (
                    user_id, conversation_id, name, description,
                    date_description, significance, participating_factions,
                    consequences, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                RETURNING id
            """
            
            # Extract values from event data
            name = event_data.get("name", "Unknown Event")
            description = event_data.get("description", "")
            date_description = event_data.get("date_description", "Unknown date")
            significance = event_data.get("significance", 5)
            participating_factions = event_data.get("participating_factions", [])
            consequences = event_data.get("consequences", [])
            
            # Convert lists to JSON if needed
            factions_json = json.dumps(participating_factions) if not isinstance(participating_factions, str) else participating_factions
            consequences_json = json.dumps(consequences) if not isinstance(consequences, str) else consequences
            
            # Execute query
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    event_id = await conn.fetchval(
                        query,
                        self.user_id,
                        self.conversation_id,
                        name,
                        description,
                        date_description,
                        significance,
                        factions_json,
                        consequences_json
                    )
                    
                    return event_id
                    
        except Exception as e:
            logger.error(f"Error storing historical event: {e}")
            return 0
    
    async def _store_location(self, location_data: Dict[str, Any]) -> int:
        """Store a location in the database."""
        try:
            query = """
                INSERT INTO Locations (
                    user_id, conversation_id, location_name,
                    description, location_type, created_at
                ) VALUES ($1, $2, $3, $4, $5, NOW())
                RETURNING id
            """
            
            # Extract values from location data
            name = location_data.get("name", "Unknown Location")
            description = location_data.get("description", "")
            location_type = location_data.get("type", "area")
            
            # Execute query
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    location_id = await conn.fetchval(
                        query,
                        self.user_id,
                        self.conversation_id,
                        name,
                        description,
                        location_type
                    )
                    
                    return location_id
                    
        except Exception as e:
            logger.error(f"Error storing location: {e}")
            return 0
    
    async def _store_location_lore(self, location_id: int, founding_story: str,
                                hidden_secrets: List[str], local_legends: List[str],
                                historical_significance: str) -> int:
        """Store location lore in the database."""
        try:
            query = """
                INSERT INTO LocationLore (
                    user_id, conversation_id, location_id,
                    founding_story, hidden_secrets, local_legends,
                    historical_significance, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                RETURNING id
            """
            
            # Convert lists to JSON if needed
            secrets_json = json.dumps(hidden_secrets) if not isinstance(hidden_secrets, str) else hidden_secrets
            legends_json = json.dumps(local_legends) if not isinstance(local_legends, str) else local_legends
            
            # Execute query
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    lore_id = await conn.fetchval(
                        query,
                        self.user_id,
                        self.conversation_id,
                        location_id,
                        founding_story,
                        secrets_json,
                        legends_json,
                        historical_significance
                    )
                    
                    return lore_id
                    
        except Exception as e:
            logger.error(f"Error storing location lore: {e}")
            return 0
    
    async def _connect_faction_to_location(self, location_id: int, faction_name: str) -> bool:
        """Connect a faction to a location in the database."""
        try:
            # First, get the faction ID
            faction_query = """
                SELECT id FROM Factions
                WHERE name = $1 AND user_id = $2 AND conversation_id = $3
                LIMIT 1
            """
            
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    faction_id = await conn.fetchval(
                        faction_query,
                        faction_name,
                        self.user_id,
                        self.conversation_id
                    )
                    
                    if not faction_id:
                        return False
                    
                    # Create connection
                    connection_query = """
                        INSERT INTO LoreConnections (
                            user_id, conversation_id, source_type,
                            source_id, target_type, target_id,
                            connection_type, strength, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                        RETURNING id
                    """
                    
                    await conn.fetchval(
                        connection_query,
                        self.user_id,
                        self.conversation_id,
                        "Factions",
                        faction_id,
                        "Locations",
                        location_id,
                        "controls",
                        8  # Strong control
                    )
                    
                    return True
                    
        except Exception as e:
            logger.error(f"Error connecting faction to location: {e}")
            return False
    
    async def _store_quest(self, quest_data: Dict[str, Any]) -> int:
        """Store a quest in the database."""
        try:
            query = """
                INSERT INTO Quests (
                    user_id, conversation_id, quest_name,
                    quest_giver, location, description,
                    difficulty, objectives, rewards,
                    lore_significance, status, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
                RETURNING id
            """
            
            # Extract values from quest data
            quest_name = quest_data.get("quest_name", "Unknown Quest")
            quest_giver = quest_data.get("quest_giver", "")
            location = quest_data.get("location", "")
            description = quest_data.get("description", "")
            difficulty = quest_data.get("difficulty", 5)
            objectives = quest_data.get("objectives", [])
            rewards = quest_data.get("rewards", [])
            lore_significance = quest_data.get("lore_significance", 5)
            
            # Convert lists to JSON if needed
            objectives_json = json.dumps(objectives) if not isinstance(objectives, str) else objectives
            rewards_json = json.dumps(rewards) if not isinstance(rewards, str) else rewards
            
            # Execute query
            async with self.get_connection_pool() as pool:
                async with pool.acquire() as conn:
                    quest_id = await conn.fetchval(
                        query,
                        self.user_id,
                        self.conversation_id,
                        quest_name,
                        quest_giver,
                        location,
                        description,
                        difficulty,
                        objectives_json,
                        rewards_json,
                        lore_significance,
                        "available"  # Default status
                    )
                    
                    return quest_id
                    
        except Exception as e:
            logger.error(f"Error storing quest: {e}")
            return 0
    
    async def get_connection_pool(self):
        """Get database connection pool."""
        # This would need to be implemented based on your database connection implementation
        from db.connection import get_connection_pool
        return await get_connection_pool()

class LoreEvolution(BaseGenerator):
    """Handles lore evolution over time."""
    
    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None):
        """Initialize the lore evolution component."""
        super().__init__(user_id, conversation_id)
        self.active_triggers = set()
        self.evolution_history = []
    
    async def evolve_lore_with_event(self, event_description: str) -> Dict[str, Any]:
        """
        Update world lore based on a significant narrative event.
        
        Args:
            event_description: Description of the narrative event
            
        Returns:
            Dictionary with lore updates
        """
        if not self.initialized:
            await self.initialize()
            
        # First, check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="evolve_lore_with_event",
            action_details={"event_description": event_description}
        )
        
        if not permission["approved"]:
            logging.warning(f"Lore evolution not approved: {permission.get('reasoning')}")
            return {"error": permission.get("reasoning"), "approved": False}
        
        # Process any existing directives before proceeding
        await self._check_and_process_directives()
        
        # Evolve lore using the implemented evolution logic
        evolved_lore = await self._evolve_lore(
            {"event": event_description},
            {"type": "event_evolution"}
        )
        
        result = {
            "updated": True,
            "event": event_description,
            "affected_elements": evolved_lore.get("affected_elements", []),
            "evolution_history": evolved_lore.get("evolution_history", {}),
            "future_implications": evolved_lore.get("future_implications", {})
        }
        
        # Report action to governance
        await self.governor.process_agent_action_report(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action={
                "type": "evolve_lore_with_event",
                "description": f"Evolved lore with event: {event_description[:50]}"
            },
            result=result
        )
        
        return result
    
    async def _check_and_process_directives(self):
        """Check for and process any pending directives from Nyx."""
        # Get directives for lore generator
        directives = await self.governor.get_agent_directives(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator"
        )
        
        for directive in directives:
            directive_type = directive.get("type")
            
            # Process prohibition directives
            if directive_type == DirectiveType.PROHIBITION:
                prohibited_actions = directive.get("prohibited_actions", [])
                logging.info(f"Found prohibition directive: {prohibited_actions}")
                
                # Store prohibited actions (will be checked during permission checks)
                if not hasattr(self, 'prohibited_lore_actions'):
                    self.prohibited_lore_actions = []
                
                self.prohibited_lore_actions.extend(prohibited_actions)
            
            # Process action directives
            elif directive_type == DirectiveType.ACTION:
                instruction = directive.get("instruction", "")
                logging.info(f"Processing action directive: {instruction}")
                
                # Implement action directive processing as needed
                # We'll report back that we've processed it
                await self.governor.process_agent_action_report(
                    agent_type=AgentType.NARRATIVE_CRAFTER,
                    agent_id="lore_generator",
                    action={
                        "type": "process_directive",
                        "description": f"Processed directive: {instruction[:50]}"
                    },
                    result={
                        "directive_id": directive.get("id"),
                        "processed": True
                    }
                )
    
    async def _evolve_lore(
        self,
        lore: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evolve lore based on context and world state.
        
        Args:
            lore: Current lore state
            context: Evolution context
            
        Returns:
            Evolved lore
        """
        try:
            # Analyze potential evolution triggers
            triggers = await self._analyze_evolution_triggers(lore, context)
            
            if not triggers:
                logger.info("No evolution triggers found")
                return lore
                
            # Generate evolution plan
            evolution_plan = await self._generate_evolution_plan(
                lore,
                triggers,
                context
            )
            
            # Apply evolution
            evolved_lore = await self._apply_evolution(
                lore,
                evolution_plan,
                context
            )
            
            # Validate evolved lore
            validated_lore = await self._validate_evolution(
                evolved_lore,
                lore,
                context
            )
            
            # Enhance evolved lore
            enhanced_lore = await self._enhance_evolution(
                validated_lore,
                lore,
                context
            )
            
            return enhanced_lore
            
        except Exception as e:
            logger.error(f"Failed to evolve lore: {str(e)}")
            raise
    
    async def _analyze_evolution_triggers(
        self,
        lore: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Analyze potential evolution triggers.
        
        Args:
            lore: Current lore state
            context: Evolution context
            
        Returns:
            List of identified triggers
        """
        try:
            triggers = []
            
            # Check event-based triggers
            event_triggers = await self._check_event_triggers(lore, context)
            triggers.extend(event_triggers)
            
            return triggers
            
        except Exception as e:
            logger.error(f"Failed to analyze evolution triggers: {str(e)}")
            raise
    
    async def _check_event_triggers(
        self,
        lore: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Check for event-based evolution triggers.
        
        Args:
            lore: Current lore state
            context: Evolution context
            
        Returns:
            List of event-based triggers
        """
        triggers = []
        
        # If we have an event context, create a trigger
        if "event" in context:
            triggers.append({
                "id": f"event_{datetime.now().timestamp()}",
                "type": "event",
                "description": context["event"],
                "priority": "high",
                "impact": random.uniform(0.5, 0.9)
            })
        
        return triggers
    
    async def _generate_evolution_plan(
        self,
        lore: Dict[str, Any],
        triggers: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate evolution plan.
        
        Args:
            lore: Current lore state
            triggers: Identified triggers
            context: Evolution context
            
        Returns:
            Evolution plan
        """
        # Mock implementation - would be replaced with actual logic
        return {
            "steps": [
                {
                    "id": "step_1",
                    "description": "Update affected lore elements",
                    "dependencies": []
                },
                {
                    "id": "step_2",
                    "description": "Create new elements if needed",
                    "dependencies": ["step_1"]
                },
                {
                    "id": "step_3",
                    "description": "Update relationships",
                    "dependencies": ["step_1", "step_2"]
                }
            ],
            "timeline": ["step_1", "step_2", "step_3"],
            "impact_analysis": {}
        }
    
    async def _apply_evolution(
        self,
        lore: Dict[str, Any],
        plan: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply evolution plan to lore.
        
        Args:
            lore: Current lore state
            plan: Evolution plan
            context: Evolution context
            
        Returns:
            Evolved lore
        """
        # Mock implementation - would be replaced with actual logic
        # For now, just return the original lore with an added history entry
        evolved_lore = lore.copy()
        
        evolved_lore["affected_elements"] = ["Element 1", "Element 2"]
        evolved_lore["evolution_history"] = {
            "timestamp": datetime.now().isoformat(),
            "event": context.get("event", "Unknown event"),
            "impact": "moderate"
        }
        evolved_lore["future_implications"] = {
            "short_term": "Some immediate effects on local population",
            "long_term": "Potential shift in faction dynamics"
        }
        
        return evolved_lore
    
    async def _validate_evolution(
        self,
        evolved_lore: Dict[str, Any],
        original_lore: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate evolved lore.
        
        Args:
            evolved_lore: Evolved lore state
            original_lore: Original lore state
            context: Evolution context
            
        Returns:
            Validated lore
        """
        # Mock implementation - would be replaced with actual logic
        return evolved_lore
    
    async def _enhance_evolution(
        self,
        evolved_lore: Dict[str, Any],
        original_lore: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance evolved lore with additional information.
        
        Args:
            evolved_lore: Evolved lore state
            original_lore: Original lore state
            context: Evolution context
            
        Returns:
            Enhanced lore
        """
        # Mock implementation - would be replaced with actual logic
        return evolved_lore

class DynamicLoreGenerator(BaseGenerator):
    """
    Main lore generation coordinator.
    
    This class orchestrates the complete lore generation process by delegating
    to specialized components for different aspects of lore generation.
    """
    
    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None):
        """Initialize the dynamic lore generator."""
        super().__init__(user_id, conversation_id)
        self.world_builder = None
        self.faction_generator = None
        self.lore_evolution = None
        self.error_handler = None
    
    async def initialize(self) -> bool:
        """Initialize the dynamic lore generator."""
        if not await super().initialize():
            return False
            
        try:
            # Initialize specialized components
            self.world_builder = WorldBuilder(self.user_id, self.conversation_id)
            await self.world_builder.initialize()
            
            self.faction_generator = FactionGenerator(self.user_id, self.conversation_id)
            await self.faction_generator.initialize()
            
            self.lore_evolution = LoreEvolution(self.user_id, self.conversation_id)
            await self.lore_evolution.initialize()
            
            # Initialize error handler
            self.error_handler = ErrorHandler(self.user_id, self.conversation_id)
            
            return True
        except Exception as e:
            logger.error(f"Error initializing DynamicLoreGenerator: {e}")
            return False
    
    async def initialize_world_lore(self, environment_desc: str) -> Dict[str, Any]:
        """
        Initialize core foundation lore for a world.
        
        Args:
            environment_desc: Description of the environment
            
        Returns:
            Dictionary containing foundation lore
        """
        return await self.world_builder.initialize_world_lore(environment_desc)
    
    async def generate_complete_lore(self, environment_desc: str) -> Dict[str, Any]:
        """
        Generate a complete set of lore for a game world.
        
        Args:
            environment_desc: Description of the environment
            
        Returns:
            Complete lore package
        """
        if not self.initialized:
            await self.initialize()
            
        # First, check permission with governance system
        permission = await self.governor.check_action_permission(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            action_type="generate_complete_lore",
            action_details={"environment_desc": environment_desc}
        )
        
        if not permission["approved"]:
            logging.warning(f"Complete lore generation not approved: {permission.get('reasoning')}")
            return {"error": permission.get("reasoning"), "approved": False}
        
        try:
            # 1) Foundation lore
            foundation_data = await self.world_builder.initialize_world_lore(environment_desc)
            if isinstance(foundation_data, dict) and "error" in foundation_data:
                return foundation_data
    
            # 2) Factions referencing 'social_structure' from foundation_data
            factions_data = await self.faction_generator.generate_factions(environment_desc, foundation_data)
    
            # 3) Cultural elements referencing environment + factions
            cultural_data = await self.faction_generator.generate_cultural_elements(environment_desc, factions_data)
    
            # 4) Historical events referencing environment + foundation_data + factions
            historical_data = await self.faction_generator.generate_historical_events(environment_desc, foundation_data, factions_data)
    
            # 5) Locations referencing environment + factions
            locations_data = await self.faction_generator.generate_locations(environment_desc, factions_data)
    
            # 6) Quest hooks referencing factions + locations
            quests_data = await self.faction_generator.generate_quest_hooks(factions_data, locations_data)
    
            # Report complete action to governance
            await self.governor.process_agent_action_report(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="lore_generator",
                action={
                    "type": "generate_complete_lore",
                    "description": f"Generated complete lore for environment: {environment_desc[:50]}"
                },
                result={
                    "world_lore_count": len(foundation_data) if isinstance(foundation_data, dict) else 0,
                    "factions_count": len(factions_data),
                    "cultural_elements_count": len(cultural_data),
                    "historical_events_count": len(historical_data),
                    "locations_count": len(locations_data),
                    "quests_count": len(quests_data),
                    "setting_name": await self.world_builder.get_setting_name()
                }
            )
    
            return {
                "world_lore": foundation_data,
                "factions": factions_data,
                "cultural_elements": cultural_data,
                "historical_events": historical_data,
                "locations": locations_data,
                "quests": quests_data
            }
        except Exception as e:
            error_msg = f"Error generating complete lore: {str(e)}"
            logger.error(error_msg)
            
            # Use error handler if available
            if self.error_handler:
                from .error_manager import LoreError, ErrorType
                error = LoreError(error_msg, ErrorType.UNKNOWN)
                await self.error_handler.handle_error(error)
            
            return {"error": error_msg}
    
    async def evolve_lore_with_event(self, event_description: str) -> Dict[str, Any]:
        """
        Update world lore based on a significant narrative event.
        
        Args:
            event_description: Description of the narrative event
            
        Returns:
            Dictionary with lore updates
        """
        return await self.lore_evolution.evolve_lore_with_event(event_description)
    
    async def cleanup(self):
        """Clean up resources."""
        await super().cleanup()
        
        if self.world_builder:
            await self.world_builder.cleanup()
            
        if self.faction_generator:
            await self.faction_generator.cleanup()
            
        if self.lore_evolution:
            await self.lore_evolution.cleanup()
