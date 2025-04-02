# lore/generator.py

"""
Lore Generator Components

This module provides components for generating and evolving lore content.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime
import random

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
from lore.lore_tools import (
    generate_foundation_lore,
    generate_factions,
    generate_cultural_elements,
    generate_historical_events,
    generate_locations,
    generate_quest_hooks
)

logger = logging.getLogger(__name__)

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
                await self._store_faction(
                    name=faction["name"],
                    faction_type=faction["type"],
                    description=faction["description"],
                    values=faction["values"],
                    goals=faction["goals"],
                    headquarters=faction.get("headquarters"),
                    rivals=faction.get("rivals", []),
                    allies=faction.get("allies", []),
                    hierarchy_type=faction.get("hierarchy_type")
                )
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
                await self._store_cultural_element(
                    name=element["name"],
                    element_type=element["type"],
                    description=element["description"],
                    practiced_by=element["practiced_by"],
                    significance=element["significance"],
                    historical_origin=element.get("historical_origin", "")
                )
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
                await self._store_historical_event(
                    name=event["name"],
                    description=event["description"],
                    date_description=event["date_description"],
                    significance=event["significance"],
                    participating_factions=event["participating_factions"],
                    consequences=event["consequences"]
                )
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
                location_id = await self._store_location(
                    name=loc["name"],
                    description=loc["description"],
                    location_type=loc["type"]
                )

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
                await self._store_quest(
                    quest_name=quest["quest_name"],
                    quest_giver=quest["quest_giver"],
                    location=quest["location"],
                    description=quest["description"],
                    difficulty=quest["difficulty"],
                    objectives=quest["objectives"],
                    rewards=quest["rewards"],
                    lore_significance=quest["lore_significance"]
                )
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
    
    async def _store_faction(self, name: str, faction_type: str, description: str,
                           values: List[str], goals: List[str], headquarters: Optional[str] = None,
                           rivals: Optional[List[str]] = None, allies: Optional[List[str]] = None,
                           hierarchy_type: Optional[str] = None) -> int:
        """
        Store a faction in the database.
        
        Args:
            name: Faction name
            faction_type: Type of faction
            description: Faction description
            values: List of faction values
            goals: List of faction goals
            headquarters: Optional headquarters location
            rivals: Optional list of rival factions
            allies: Optional list of allied factions
            hierarchy_type: Optional hierarchy type
            
        Returns:
            ID of the created faction
        """
        try:
            query = """
                INSERT INTO Factions (
                    user_id, conversation_id, name, type,
                    description, values, goals, headquarters,
                    rivals, allies, hierarchy_type, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
                RETURNING id
            """
            
            # Convert arrays to JSON strings if necessary
            if not isinstance(values, str):
                values_json = json.dumps(values)
            else:
                values_json = values
                
            if not isinstance(goals, str):
                goals_json = json.dumps(goals)
            else:
                goals_json = goals
                
            if not isinstance(rivals, str) and rivals is not None:
                rivals_json = json.dumps(rivals)
            else:
                rivals_json = rivals
                
            if not isinstance(allies, str) and allies is not None:
                allies_json = json.dumps(allies)
            else:
                allies_json = allies
            
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
    
    async def _store_cultural_element(self, name: str, element_type: str, 
                                    description: str, practiced_by: List[str],
                                    significance: int,
                                    historical_origin: Optional[str] = None) -> int:
        """
        Store a cultural element in the database.
        
        Args:
            name: Element name
            element_type: Type of cultural element
            description: Element description
            practiced_by: List of practitioners
            significance: Element significance (1-10)
            historical_origin: Optional historical origin
            
        Returns:
            ID of the created cultural element
        """
        try:
            query = """
                INSERT INTO CulturalElements (
                    user_id, conversation_id, name, type,
                    description, practiced_by, significance,
                    historical_origin, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                RETURNING id
            """
            
            # Convert arrays to JSON strings if necessary
            if not isinstance(practiced_by, str):
                practiced_by_json = json.dumps(practiced_by)
            else:
                practiced_by_json = practiced_by
            
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
    
    async def _store_historical_event(self, name: str, description: str,
                                    date_description: str, significance: int,
                                    participating_factions: List[str],
                                    consequences: List[str]) -> int:
        """
        Store a historical event in the database.
        
        Args:
            name: Event name
            description: Event description
            date_description: Description of when it occurred
            significance: Event significance (1-10)
            participating_factions: List of participating factions
            consequences: List of consequences
            
        Returns:
            ID of the created historical event
        """
        try:
            query = """
                INSERT INTO HistoricalEvents (
                    user_id, conversation_id, name, description,
                    date_description, significance, participating_factions,
                    consequences, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                RETURNING id
            """
            
            # Convert arrays to JSON strings if necessary
            if not isinstance(participating_factions, str):
                factions_json = json.dumps(participating_factions)
            else:
                factions_json = participating_factions
                
            if not isinstance(consequences, str):
                consequences_json = json.dumps(consequences)
            else:
                consequences_json = consequences
            
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
    
    async def _store_location(self, name: str, description: str,
                            location_type: str) -> int:
        """
        Store a location in the database.
        
        Args:
            name: Location name
            description: Location description
            location_type: Type of location
            
        Returns:
            ID of the created location
        """
        try:
            query = """
                INSERT INTO Locations (
                    user_id, conversation_id, location_name,
                    description, location_type, created_at
                ) VALUES ($1, $2, $3, $4, $5, NOW())
                RETURNING id
            """
            
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
        """
        Store location lore in the database.
        
        Args:
            location_id: ID of the location
            founding_story: Story of the location's founding
            hidden_secrets: List of hidden secrets
            local_legends: List of local legends
            historical_significance: Historical significance
            
        Returns:
            ID of the created location lore
        """
        try:
            query = """
                INSERT INTO LocationLore (
                    user_id, conversation_id, location_id,
                    founding_story, hidden_secrets, local_legends,
                    historical_significance, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                RETURNING id
            """
            
            # Convert arrays to JSON strings if necessary
            if not isinstance(hidden_secrets, str):
                secrets_json = json.dumps(hidden_secrets)
            else:
                secrets_json = hidden_secrets
                
            if not isinstance(local_legends, str):
                legends_json = json.dumps(local_legends)
            else:
                legends_json = local_legends
            
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
        """
        Connect a faction to a location.
        
        Args:
            location_id: ID of the location
            faction_name: Name of the faction
            
        Returns:
            True if successful, False otherwise
        """
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
    
    async def _store_quest(self, quest_name: str, quest_giver: str, location: str,
                         description: str, difficulty: int, objectives: List[str],
                         rewards: List[str], lore_significance: int) -> int:
        """
        Store a quest in the database.
        
        Args:
            quest_name: Name of the quest
            quest_giver: Name of the quest giver
            location: Location of the quest
            description: Quest description
            difficulty: Quest difficulty (1-10)
            objectives: List of objectives
            rewards: List of rewards
            lore_significance: Significance to lore (1-10)
            
        Returns:
            ID of the created quest
        """
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
            
            # Convert arrays to JSON strings if necessary
            if not isinstance(objectives, str):
                objectives_json = json.dumps(objectives)
            else:
                objectives_json = objectives
                
            if not isinstance(rewards, str):
                rewards_json = json.dumps(rewards)
            else:
                rewards_json = rewards
            
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
            
            # Check time-based triggers
            time_triggers = await self._check_time_triggers(lore, context)
            triggers.extend(time_triggers)
            
            # Check event-based triggers
            event_triggers = await self._check_event_triggers(lore, context)
            triggers.extend(event_triggers)
            
            # Check state-based triggers
            state_triggers = await self._check_state_triggers(lore, context)
            triggers.extend(state_triggers)
            
            # Check relationship-based triggers
            relationship_triggers = await self._check_relationship_triggers(
                lore,
                context
            )
            triggers.extend(relationship_triggers)
            
            return triggers
            
        except Exception as e:
            logger.error(f"Failed to analyze evolution triggers: {str(e)}")
            raise
    
    async def _check_time_triggers(
        self,
        lore: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Check for time-based evolution triggers.
        
        Args:
            lore: Current lore state
            context: Evolution context
            
        Returns:
            List of time-based triggers
        """
        # Mock implementation - would be replaced with actual logic
        return []
    
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
    
    async def _check_state_triggers(
        self,
        lore: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Check for state-based evolution triggers.
        
        Args:
            lore: Current lore state
            context: Evolution context
            
        Returns:
            List of state-based triggers
        """
        # Mock implementation - would be replaced with actual logic
        return []
    
    async def _check_relationship_triggers(
        self,
        lore: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Check for relationship-based evolution triggers.
        
        Args:
            lore: Current lore state
            context: Evolution context
            
        Returns:
            List of relationship-based triggers
        """
        # Mock implementation - would be replaced with actual logic
        return []
    
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
