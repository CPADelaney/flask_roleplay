# lore/lore_system.py

"""
Lore System - Main Entry Point

This module serves as the primary interface for all lore-related functionality.
It delegates to specialized components and provides a clean, unified API.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# Import data access layer
from .data_access import (
    NPCDataAccess,
    LocationDataAccess,
    FactionDataAccess,
    LoreKnowledgeAccess
)

# Import integration components
from .integration import (
    NPCLoreIntegration,
    ConflictIntegration,
    ContextEnhancer
)

# Import generator components
from .generator import DynamicLoreGenerator

# Nyx governance integration
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance

logger = logging.getLogger(__name__)

# Cache for LoreSystem instances
LORE_SYSTEM_INSTANCES = {}

class LoreSystem:
    """
    Unified interface for all lore-related functionality.
    
    This class serves as the main entry point for all lore operations,
    delegating to specialized components for implementation details.
    """
    
    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None):
        """
        Initialize the LoreSystem with optional user and conversation context.
        
        Args:
            user_id: Optional user ID for filtering
            conversation_id: Optional conversation ID for filtering
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.initialized = False
        self.governor = None
        
        # Initialize data access components
        self.npc_data = NPCDataAccess(user_id, conversation_id)
        self.location_data = LocationDataAccess(user_id, conversation_id)
        self.faction_data = FactionDataAccess(user_id, conversation_id)
        self.lore_knowledge = LoreKnowledgeAccess(user_id, conversation_id)
        
        # Initialize integration components
        self.npc_integration = NPCLoreIntegration(user_id, conversation_id)
        self.conflict_integration = ConflictIntegration(user_id, conversation_id)
        self.context_enhancer = ContextEnhancer(user_id, conversation_id)
        
        # Initialize generator component
        self.generator = DynamicLoreGenerator(user_id, conversation_id)
        
    @classmethod
    def get_instance(cls, user_id: Optional[int] = None, conversation_id: Optional[int] = None) -> 'LoreSystem':
        """
        Get a singleton instance of LoreSystem for the given user and conversation.
        
        Args:
            user_id: Optional user ID for filtering
            conversation_id: Optional conversation ID for filtering
            
        Returns:
            LoreSystem instance
        """
        key = f"{user_id or 'global'}:{conversation_id or 'global'}"
        
        if key not in LORE_SYSTEM_INSTANCES:
            LORE_SYSTEM_INSTANCES[key] = cls(user_id, conversation_id)
            
        return LORE_SYSTEM_INSTANCES[key]
    
    async def initialize(self) -> bool:
        """
        Initialize the LoreSystem and all its components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.initialized:
            return True
            
        try:
            # Initialize governance
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
            
            # Initialize all components
            await self.npc_data.initialize()
            await self.location_data.initialize()
            await self.faction_data.initialize()
            await self.lore_knowledge.initialize()
            
            await self.npc_integration.initialize()
            await self.conflict_integration.initialize()
            await self.context_enhancer.initialize()
            
            await self.generator.initialize()
            
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing LoreSystem: {e}")
            return False
    
    #---------------------------
    # NPC Lore Methods
    #---------------------------
    
    async def get_npc_lore_knowledge(self, npc_id: int) -> List[Dict[str, Any]]:
        """
        Get all lore known by an NPC.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            List of lore items known by the NPC
        """
        return await self.lore_knowledge.get_entity_knowledge("npc", npc_id)
    
    async def initialize_npc_lore_knowledge(self, npc_id: int, 
                                           cultural_background: str,
                                           faction_affiliations: List[str]) -> Dict[str, Any]:
        """
        Initialize an NPC's knowledge of lore based on their background.
        
        Args:
            npc_id: ID of the NPC
            cultural_background: Cultural background of the NPC
            faction_affiliations: List of faction names the NPC is affiliated with
            
        Returns:
            Dictionary of knowledge granted
        """
        return await self.npc_integration.initialize_npc_lore_knowledge(
            npc_id, cultural_background, faction_affiliations
        )
    
    async def process_npc_lore_interaction(self, npc_id: int, player_input: str) -> Dict[str, Any]:
        """
        Process a potential lore interaction between the player and an NPC.
        
        Args:
            npc_id: ID of the NPC
            player_input: The player's input text
            
        Returns:
            Lore response information if relevant
        """
        return await self.npc_integration.process_npc_lore_interaction(npc_id, player_input)
    
    async def apply_dialect_to_text(self, text: str, dialect_id: int, 
                                   intensity: str = 'medium',
                                   npc_id: Optional[int] = None) -> str:
        """
        Apply dialect features to a text.
        
        Args:
            text: Original text
            dialect_id: ID of the dialect to apply
            intensity: Intensity of dialect application ('light', 'medium', 'strong')
            npc_id: Optional NPC ID for personalized dialect features
            
        Returns:
            Modified text with dialect features applied
        """
        return await self.npc_integration.apply_dialect_to_text(text, dialect_id, intensity, npc_id)
    
    #---------------------------
    # Location Lore Methods
    #---------------------------
    
    async def get_location_lore(self, location_id: int) -> Dict[str, Any]:
        """
        Get lore specific to a location.
        
        Args:
            location_id: ID of the location
            
        Returns:
            Dictionary with location lore
        """
        return await self.location_data.get_location_with_lore(location_id)
    
    async def get_location_by_name(self, location_name: str) -> Dict[str, Any]:
        """
        Get location details by name.
        
        Args:
            location_name: Name of the location
            
        Returns:
            Location details
        """
        return await self.location_data.get_location_by_name(location_name)
    
    async def get_comprehensive_location_context(self, location_name: str) -> Dict[str, Any]:
        """
        Get comprehensive lore context for a specific location.
        
        Args:
            location_name: Name of the location
            
        Returns:
            Dictionary with comprehensive location lore
        """
        location = await self.location_data.get_location_by_name(location_name)
        if not location:
            return {}
            
        location_id = location.get("id")
        
        # Get basic location lore
        location_lore = await self.location_data.get_location_with_lore(location_id)
        
        # Get cultural context
        cultural_info = await self.location_data.get_cultural_context_for_location(location_id)
        
        # Get political context
        political_info = await self.location_data.get_political_context_for_location(location_id)
        
        # Get environmental context
        environment_info = await self.location_data.get_environmental_conditions(location_id)
        
        return {
            "location": location,
            "lore": location_lore,
            "cultural_context": cultural_info,
            "political_context": political_info,
            "environment": environment_info
        }
    
    async def generate_scene_description_with_lore(self, location: str) -> Dict[str, Any]:
        """
        Generate a scene description enhanced with relevant lore.
        
        Args:
            location: Current location name
            
        Returns:
            Enhanced scene description
        """
        return await self.context_enhancer.generate_scene_description(location)
    
    #---------------------------
    # Faction Lore Methods
    #---------------------------
    
    async def get_faction_details(self, faction_id: int) -> Dict[str, Any]:
        """
        Get details about a faction.
        
        Args:
            faction_id: ID of the faction
            
        Returns:
            Faction details
        """
        return await self.faction_data.get_faction_details(faction_id)
    
    async def get_faction_relationships(self, faction_id: int) -> List[Dict[str, Any]]:
        """
        Get relationships for a faction.
        
        Args:
            faction_id: ID of the faction
            
        Returns:
            List of faction relationships
        """
        return await self.faction_data.get_faction_relationships(faction_id)
    
    #---------------------------
    # Context Enhancement Methods
    #---------------------------
    
    async def enhance_context_with_lore(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance a context object with relevant lore.
        
        Args:
            context: Current context
            
        Returns:
            Enhanced context with relevant lore
        """
        return await self.context_enhancer.enhance_context(context)
    
    #---------------------------
    # Lore Generation Methods
    #---------------------------
    
    async def initialize_world_lore(self, environment_desc: str) -> Dict[str, Any]:
        """
        Initialize core foundation lore for a world.
        
        Args:
            environment_desc: Description of the environment
            
        Returns:
            Dictionary containing foundation lore
        """
        return await self.generator.initialize_world_lore(environment_desc)
    
    async def generate_complete_lore(self, environment_desc: str) -> Dict[str, Any]:
        """
        Generate a complete set of lore for a game world.
        
        Args:
            environment_desc: Description of the environment
            
        Returns:
            Complete lore package
        """
        return await self.generator.generate_complete_lore(environment_desc)
    
    async def evolve_lore_with_event(self, event_description: str) -> Dict[str, Any]:
        """
        Update world lore based on a significant narrative event.
        
        Args:
            event_description: Description of the narrative event
            
        Returns:
            Dictionary with lore updates
        """
        return await self.generator.evolve_lore_with_event(event_description)
    
    #---------------------------
    # Knowledge Management Methods
    #---------------------------
    
    async def discover_lore(self, lore_type: str, lore_id: int, entity_type: str, 
                          entity_id: int, knowledge_level: int) -> bool:
        """
        Record that an entity has discovered a piece of lore.
        
        Args:
            lore_type: Type of lore (e.g., "Factions", "WorldLore")
            lore_id: ID of the lore
            entity_type: Type of entity discovering the lore (e.g., "npc", "player")
            entity_id: ID of the entity
            knowledge_level: Level of knowledge (1-10)
            
        Returns:
            True if successful, False otherwise
        """
        return await self.lore_knowledge.add_lore_knowledge(
            entity_type, entity_id, lore_type, lore_id, knowledge_level
        )
    
    async def get_relevant_lore(self, query: str, min_relevance: float = 0.6, 
                              limit: int = 5, lore_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get lore relevant to a search query.
        
        Args:
            query: Search query text
            min_relevance: Minimum relevance score (0-1)
            limit: Maximum number of results
            lore_types: Optional list of lore types to include
            
        Returns:
            List of relevant lore items
        """
        return await self.lore_knowledge.get_relevant_lore(
            query, min_relevance, limit, lore_types
        )
    
    async def generate_available_lore_for_context(self, query_text: str, entity_type: str, 
                                               entity_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get lore known by an entity that is relevant to the current context.
        
        Args:
            query_text: The context query
            entity_type: Type of entity (e.g., "npc", "player")
            entity_id: ID of the entity
            limit: Maximum number of results
            
        Returns:
            List of relevant known lore
        """
        return await self.lore_knowledge.generate_available_lore_for_context(
            query_text, entity_type, entity_id, limit
        )
    
    #---------------------------
    # Utility Methods
    #---------------------------
    
    async def cleanup(self):
        """Clean up resources used by the LoreSystem."""
        try:
            # Cleanup all components
            await self.npc_data.cleanup()
            await self.location_data.cleanup()
            await self.faction_data.cleanup()
            await self.lore_knowledge.cleanup()
            
            await self.npc_integration.cleanup()
            await self.conflict_integration.cleanup()
            await self.context_enhancer.cleanup()
            
            await self.generator.cleanup()
            
            # Remove from instances cache
            key = f"{self.user_id or 'global'}:{self.conversation_id or 'global'}"
            if key in LORE_SYSTEM_INSTANCES:
                del LORE_SYSTEM_INSTANCES[key]
                
        except Exception as e:
            logger.error(f"Error during LoreSystem cleanup: {e}")

# Create a singleton instance for easy access
lore_system = LoreSystem()
