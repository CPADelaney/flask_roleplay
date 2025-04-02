# lore/lore_system.py

"""
Lore System - Main Entry Point

This module serves as the primary interface for all lore-related functionality,
with enhanced capabilities and proper governance integration.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# Import data access layer
from .data_access import NPCDataAccess, LocationDataAccess, FactionDataAccess, LoreKnowledgeAccess

# Import integration components
from .integration import NPCLoreIntegration, ConflictIntegration, ContextEnhancer

# Import generator component
from .lore_generator import DynamicLoreGenerator

# Import Nyx governance
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance

# Import for caching
from datetime import timedelta

logger = logging.getLogger(__name__)

# Cache for LoreSystem instances
LORE_SYSTEM_INSTANCES = {}

# Global cache for all lore subsystems
class LoreCache:
    """Unified cache system for all lore types with improved organization"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls, max_size=1000, ttl=7200):
        """Get singleton instance of the cache"""
        if cls._instance is None:
            cls._instance = LoreCache(max_size, ttl)
        return cls._instance
    
    def __init__(self, max_size=1000, ttl=7200):
        self.cache = {}
        self.max_size = max_size
        self.default_ttl = ttl
        self.access_times = {}
        self._lock = None  # Will be initialized as asyncio.Lock() when needed
    
    async def get(self, namespace, key, user_id=None, conversation_id=None):
        """Get an item from the cache with async support"""
        import asyncio
        
        if self._lock is None:
            self._lock = asyncio.Lock()
            
        full_key = self._create_key(namespace, key, user_id, conversation_id)
        
        async with self._lock:
            if full_key in self.cache:
                value, expiry = self.cache[full_key]
                if expiry > datetime.now().timestamp():
                    # Update access time for LRU
                    self.access_times[full_key] = datetime.now().timestamp()
                    return value
                # Remove expired item
                self._remove_key(full_key)
        return None
    
    async def set(self, namespace, key, value, ttl=None, user_id=None, conversation_id=None):
        """Set an item in the cache with async support"""
        import asyncio
        
        if self._lock is None:
            self._lock = asyncio.Lock()
            
        full_key = self._create_key(namespace, key, user_id, conversation_id)
        expiry = datetime.now().timestamp() + (ttl or self.default_ttl)
        
        async with self._lock:
            # Manage cache size - use LRU strategy
            if len(self.cache) >= self.max_size:
                # Find oldest accessed item
                oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                self._remove_key(oldest_key)
                
            self.cache[full_key] = (value, expiry)
            self.access_times[full_key] = datetime.now().timestamp()
    
    async def invalidate(self, namespace, key, user_id=None, conversation_id=None):
        """Invalidate a specific key with async support"""
        import asyncio
        
        if self._lock is None:
            self._lock = asyncio.Lock()
            
        full_key = self._create_key(namespace, key, user_id, conversation_id)
        async with self._lock:
            self._remove_key(full_key)
    
    async def invalidate_pattern(self, namespace, pattern, user_id=None, conversation_id=None):
        """Invalidate keys matching a pattern with async support"""
        import asyncio
        
        if self._lock is None:
            self._lock = asyncio.Lock()
            
        namespace_pattern = f"{namespace}:"
        async with self._lock:
            keys_to_remove = []
            
            for key in self.cache.keys():
                if key.startswith(namespace_pattern):
                    # Extract the key part after the namespace
                    key_part = key.split(':', 1)[1]
                    if pattern in key_part:
                        keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_key(key)
    
    async def clear_namespace(self, namespace):
        """Clear all items in a namespace with async support"""
        import asyncio
        
        if self._lock is None:
            self._lock = asyncio.Lock()
            
        namespace_pattern = f"{namespace}:"
        async with self._lock:
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(namespace_pattern)]
            for key in keys_to_remove:
                self._remove_key(key)
    
    def _create_key(self, namespace, key, user_id=None, conversation_id=None):
        """Create a unique cache key with proper namespacing"""
        if user_id and conversation_id:
            return f"{namespace}:{user_id}:{conversation_id}:{key}"
        elif user_id:
            return f"{namespace}:{user_id}:global:{key}"
        else:
            return f"{namespace}:global:{key}"
    
    def _remove_key(self, full_key):
        """Remove a key from both cache and access times"""
        if full_key in self.cache:
            del self.cache[full_key]
        if full_key in self.access_times:
            del self.access_times[full_key]

class LoreSystem:
    """
    Unified interface for all lore-related functionality with enhanced capabilities.
    
    This class serves as the main entry point for all lore operations,
    delegating to specialized components and ensuring proper governance integration.
    """
    
    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None):
        """Initialize the LoreSystem with optional user and conversation context."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.initialized = False
        self.governor = None
        self.cache = LoreCache.get_instance()
        
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
        
        # Store prohibited actions from directives
        self.prohibited_actions = []
        
        # Store action modifications from directives
        self.action_modifications = {}
    
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
            
            # Register with governance
            await self.register_with_governance()
            
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing LoreSystem: {e}")
            return False
    
    async def register_with_governance(self):
        """Register the lore system with Nyx governance."""
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
        
        # Register this system with governance
        await self.governor.register_agent(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_system",
            agent_instance=self
        )
        
        # Issue standard directives
        await self._issue_standard_directives()
    
    async def _issue_standard_directives(self):
        """Issue standard directives for the lore system."""
        if not self.governor:
            return
            
        # Directive for lore generation
        await self.governor.issue_directive(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Maintain world lore consistency and generate new lore as needed.",
                "scope": "narrative"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24*60  # 24 hours
        )
        
        # Directive for NPC lore integration
        await self.governor.issue_directive(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="npc_lore_integration",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Ensure NPCs have appropriate lore knowledge based on their backgrounds.",
                "scope": "narrative"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24*60  # 24 hours
        )
    
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
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="initialize_npc_lore_knowledge",
        action_description="Initializing lore knowledge for NPC {npc_id}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def initialize_npc_lore_knowledge(self, ctx, npc_id: int, 
                                         cultural_background: str,
                                         faction_affiliations: List[str]) -> Dict[str, Any]:
        """
        Initialize an NPC's knowledge of lore based on their background.
        
        Args:
            ctx: Governance context
            npc_id: ID of the NPC
            cultural_background: Cultural background of the NPC
            faction_affiliations: List of faction names the NPC is affiliated with
            
        Returns:
            Dictionary of knowledge granted
        """
        return await self.npc_integration.initialize_npc_lore_knowledge(
            ctx, npc_id, cultural_background, faction_affiliations
        )
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="process_npc_lore_interaction",
        action_description="Processing lore interaction for NPC {npc_id}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def process_npc_lore_interaction(self, ctx, npc_id: int, player_input: str) -> Dict[str, Any]:
        """
        Process a potential lore interaction between the player and an NPC.
        
        Args:
            ctx: Governance context
            npc_id: ID of the NPC
            player_input: The player's input text
            
        Returns:
            Lore response information if relevant
        """
        return await self.npc_integration.process_npc_lore_interaction(ctx, npc_id, player_input)
    
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
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_comprehensive_location_context",
        action_description="Getting comprehensive lore for location: {location_name}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def get_comprehensive_location_context(self, ctx, location_name: str) -> Dict[str, Any]:
        """
        Get comprehensive lore context for a specific location.
        
        Args:
            ctx: Governance context
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
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_scene_description",
        action_description="Generating scene description for location: {location}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def generate_scene_description_with_lore(self, ctx, location: str) -> Dict[str, Any]:
        """
        Generate a scene description enhanced with relevant lore.
        
        Args:
            ctx: Governance context
            location: Current location name
            
        Returns:
            Enhanced scene description
        """
        return await self.context_enhancer.generate_scene_description(location)
    
    #---------------------------
    # Lore Generation Methods
    #---------------------------
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_complete_lore",
        action_description="Generating complete lore for environment: {environment_desc}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def generate_complete_lore(self, ctx, environment_desc: str) -> Dict[str, Any]:
        """
        Generate a complete set of lore for a game world.
        
        Args:
            ctx: Governance context
            environment_desc: Description of the environment
            
        Returns:
            Complete lore package
        """
        return await self.generator.generate_complete_lore(environment_desc)
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_lore_with_event",
        action_description="Evolving lore based on event: {event_description}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def evolve_lore_with_event(self, ctx, event_description: str) -> Dict[str, Any]:
        """
        Update world lore based on a significant narrative event.
        
        Args:
            ctx: Governance context
            event_description: Description of the narrative event
            
        Returns:
            Dictionary with lore updates
        """
        return await self.generator.evolve_lore_with_event(event_description)
    
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
    # Cache Methods
    #---------------------------
    
    async def get_cached_lore(self, lore_type: str, lore_id: int) -> Optional[Dict[str, Any]]:
        """
        Get lore from cache if available, otherwise fetch from database.
        
        Args:
            lore_type: Type of lore (WorldLore, Factions, etc.)
            lore_id: ID of the lore element
            
        Returns:
            Lore data or None if not found
        """
        # Try cache first
        cache_key = f"{lore_type}_{lore_id}"
        cached = await self.cache.get("lore", cache_key, self.user_id, self.conversation_id)
        if cached:
            return cached
            
        # Fetch from database
        lore = None
        if lore_type == "WorldLore":
            # Get world lore
            pass
        elif lore_type == "Factions":
            lore = await self.faction_data.get_faction_details(lore_id)
        elif lore_type == "Locations":
            lore = await self.location_data.get_location_with_lore(lore_id)
        
        # Cache for future use
        if lore:
            await self.cache.set("lore", cache_key, lore, None, self.user_id, self.conversation_id)
            
        return lore
    
    async def invalidate_lore_cache(self, lore_type: str, lore_id: Optional[int] = None):
        """
        Invalidate lore cache entries.
        
        Args:
            lore_type: Type of lore (WorldLore, Factions, etc.)
            lore_id: Optional ID of specific lore element or None to invalidate all of type
        """
        if lore_id is not None:
            cache_key = f"{lore_type}_{lore_id}"
            await self.cache.invalidate("lore", cache_key, self.user_id, self.conversation_id)
        else:
            pattern = lore_type
            await self.cache.invalidate_pattern("lore", pattern, self.user_id, self.conversation_id)
    
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
