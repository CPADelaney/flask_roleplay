# lore/enhanced_lore_consolidated.py

"""
Consolidated Enhanced Lore System

This module provides a unified interface to all lore subsystems,
with proper Nyx governance integration, optimized caching,
and clean architecture.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta

# Agents SDK imports
from agents import Agent, ModelSettings, function_tool, Runner, trace
from agents.run_context import RunContextWrapper

# Nyx governance integration
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from nyx.directive_handler import DirectiveHandler

# Database and embedding functionality
from db.connection import get_db_connection
from embedding.vector_store import generate_embedding, vector_similarity

# Import core systems
from lore.lore_manager import LoreManager
from lore.dynamic_lore_generator import DynamicLoreGenerator
from lore.npc_lore_integration import NPCLoreIntegration
from lore.lore_integration import LoreIntegrationSystem

# Import specialized systems as needed
from lore.setting_analyzer import SettingAnalyzer

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
        self._lock = asyncio.Lock()  # Thread-safety for async operations
    
    async def get(self, namespace, key, user_id=None, conversation_id=None):
        """Get an item from the cache with async support"""
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
        full_key = self._create_key(namespace, key, user_id, conversation_id)
        async with self._lock:
            self._remove_key(full_key)
    
    async def invalidate_pattern(self, namespace, pattern, user_id=None, conversation_id=None):
        """Invalidate keys matching a pattern with async support"""
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

class EnhancedLoreSystem:
    """
    Unified entry point for all lore systems with streamlined API.
    
    This class serves as the primary interface for accessing different
    lore subsystems, coordinating their actions, and ensuring proper
    governance integration.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the enhanced lore system."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = None
        self.cache = LoreCache.get_instance()
        
        # Core systems
        self.lore_manager = LoreManager(user_id, conversation_id)
        self.generator = DynamicLoreGenerator(user_id, conversation_id)
        self.npc_integration = NPCLoreIntegration(user_id, conversation_id)
        self.integration = LoreIntegrationSystem(user_id, conversation_id)
        
        # Specialized systems - initialize on demand
        self._setting_analyzer = None
        self._registered = False
        
    async def initialize(self):
        """Initialize all lore subsystems and ensure proper governance registration"""
        await self._initialize_governance()
        await self.lore_manager.initialize_tables()
        
        if not self._registered:
            await self.register_with_governance()
            self._registered = True
        
        return self
    
    async def _initialize_governance(self):
        """Initialize connection to Nyx governance"""
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
        return self.governor
    
    async def register_with_governance(self):
        """Register all lore subsystems with Nyx governance"""
        from lore.governance_registration import register_all_lore_modules_with_governance
        return await register_all_lore_modules_with_governance(self.user_id, self.conversation_id)
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_complete_lore",
        action_description="Generating complete lore for environment: {environment_desc}",
        id_from_context=lambda ctx: "enhanced_lore_system"
    )
    async def generate_complete_lore(self, environment_desc: str) -> Dict[str, Any]:
        """
        Generate complete lore for an environment, integrating all subsystems.
        
        This method orchestrates the generation of all lore components:
        - Foundation lore (cosmology, magic system, etc.)
        - Factions and political structures
        - Cultural elements and traditions
        - Historical events
        - Locations and geographic regions
        - Quest hooks and narrative opportunities
        
        Args:
            environment_desc: Description of the environment/setting
            
        Returns:
            Complete lore data dictionary
        """
        # Delegate to the generator with governance oversight
        return await self.generator.generate_complete_lore(environment_desc)
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_lore_for_location",
        action_description="Getting comprehensive lore for location: {location_name}",
        id_from_context=lambda ctx: "enhanced_lore_system"
    )
    async def get_lore_for_location(self, location_name: str) -> Dict[str, Any]:
        """
        Get comprehensive lore for a specific location.
        
        This aggregates lore relevant to the location including:
        - Basic location details
        - Cultural elements associated with the location
        - Historical events that occurred at the location
        - Factions with presence at the location
        - Local customs, traditions, and taboos
        
        Args:
            location_name: Name of the location
            
        Returns:
            Dictionary with comprehensive location lore
        """
        return await self.integration.get_comprehensive_location_context(location_name)
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="integrate_lore_with_npcs",
        action_description="Integrating lore with NPCs",
        id_from_context=lambda ctx: "enhanced_lore_system"
    )
    async def integrate_lore_with_npcs(self, npc_ids: List[int]) -> Dict[str, Any]:
        """
        Integrate lore with NPCs, distributing knowledge appropriately.
        
        This method:
        - Assigns lore knowledge to NPCs based on their background
        - Creates memory entries for NPCs about lore they know
        - Sets up knowledge distribution based on NPC factions and roles
        
        Args:
            npc_ids: List of NPC IDs to integrate with lore
            
        Returns:
            Dictionary with integration results
        """
        return await self.integration.integrate_lore_with_npcs(npc_ids)
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_lore_with_event",
        action_description="Evolving lore based on event: {event_description}",
        id_from_context=lambda ctx: "enhanced_lore_system"
    )
    async def evolve_lore_with_event(self, event_description: str) -> Dict[str, Any]:
        """
        Evolve lore based on a narrative event.
        
        This method identifies lore elements affected by the event and updates them,
        ensuring consistency and narrative progression.
        
        Args:
            event_description: Description of the event
            
        Returns:
            Dictionary with evolved lore
        """
        return await self.integration.update_lore_after_narrative_event(event_description)
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="analyze_setting",
        action_description="Analyzing setting demographics and structure",
        id_from_context=lambda ctx: "enhanced_lore_system"
    )
    async def analyze_setting(self) -> Dict[str, Any]:
        """
        Analyze the current setting to understand demographics and structure.
        
        This creates a profile of the setting based on NPCs, locations, and other
        elements already defined, which helps ensure consistency in future lore
        generation.
        
        Returns:
            Dictionary with setting analysis
        """
        if not self._setting_analyzer:
            self._setting_analyzer = SettingAnalyzer(self.user_id, self.conversation_id)
            
        ctx = RunContextWrapper(agent_context={"user_id": self.user_id, "conversation_id": self.conversation_id})
        return await self._setting_analyzer.analyze_setting_demographics(ctx)
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_npc_lore_response",
        action_description="Generating lore-based response for NPC {npc_id}",
        id_from_context=lambda ctx: "enhanced_lore_system"
    )
    async def generate_npc_lore_response(self, npc_id: int, player_input: str) -> Dict[str, Any]:
        """
        Generate a lore-based response for an NPC based on player input.
        
        This considers:
        - The NPC's knowledge of lore (what they know and don't know)
        - The NPC's personality and relationship with the player
        - The context of the conversation
        
        Args:
            npc_id: ID of the NPC
            player_input: Player's input/question
            
        Returns:
            Response data including the NPC's response text
        """
        return await self.integration.generate_npc_lore_response(npc_id, player_input)
    
    async def add_lore_to_agent_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add relevant lore to an agent's context data.
        
        This enriches the agent context with lore elements that may be relevant,
        including location lore, faction information, and cultural data.
        
        Args:
            context: The existing agent context
            
        Returns:
            Enhanced context with lore data
        """
        # No governance needed as this is an internal helper method
        enhanced_context = await self.integration.enhance_gpt_context_with_lore(context)
        return enhanced_context
    
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
            
        # Fetch from database using LoreManager
        lore = await self.lore_manager.get_lore_element(lore_type, lore_id)
        
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


