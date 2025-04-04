# lore/core/registry.py

from typing import Dict, Any, Type, List, Optional
import logging
import asyncio
from datetime import datetime

from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

from lore.core.base_manager import BaseLoreManager
from lore.utils.theming import MatriarchalThemingUtils

class ManagerRegistry:
    """
    Enhanced registry for all manager classes with centralized coordination
    of the matriarchal lore system functionality.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._managers = {}
        self._class_map = {}  # This will be populated in __init__.py
        self.governor = None
        self.initialized = False
    
    async def initialize_governance(self):
        """Initialize Nyx governance connection"""
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
        return self.governor
    
    async def ensure_initialized(self):
        """Ensure registry and all core systems are initialized"""
        if not self.initialized:
            await self.initialize_governance()
            await self.initialize_all_systems()
            self.initialized = True
            logging.info(f"Manager registry initialized for user {self.user_id}, conversation {self.conversation_id}")
    
    async def initialize_all_systems(self):
        """Initialize all subsystems and their tables"""
        # Initialize core systems
        await self.get_geopolitical_manager()
        await self.get_religion_manager()
        await self.get_local_lore_manager()
        await self.get_regional_culture_system()
        await self.get_educational_system_manager()
        await self.get_world_politics_manager()
        await self.get_lore_update_manager()
        
        # Register systems with governance
        await self._register_all_systems()
    
    async def _register_all_systems(self):
        """Register all systems with governance"""
        # Get each manager and register it
        for key in self._class_map.keys():
            manager = await self.get_manager(key)
            if hasattr(manager, 'register_with_governance'):
                await manager.register_with_governance()
    
    async def get_manager(self, manager_key: str) -> 'BaseLoreManager':
        """
        Get a manager instance by key, creating it if not already created.
        Will ensure the manager is initialized.
        
        Args:
            manager_key: Key of the manager to get
            
        Returns:
            Instance of the manager
        """
        if manager_key not in self._managers:
            if manager_key not in self._class_map:
                raise ValueError(f"Unknown manager key: {manager_key}")
                
            manager_class = self._class_map[manager_key]
            self._managers[manager_key] = manager_class(self.user_id, self.conversation_id)
        
        # Ensure manager is initialized
        manager = self._managers[manager_key]
        if hasattr(manager, 'ensure_initialized'):
            await manager.ensure_initialized()
        return manager
    
    async def get_lore_dynamics(self):
        """Get the LoreDynamicsSystem instance"""
        return await self.get_manager('lore_dynamics')
    
    async def get_geopolitical_manager(self):
        """Get the GeopoliticalSystemManager instance"""
        return await self.get_manager('geopolitical')
    
    async def get_local_lore_manager(self):
        """Get the LocalLoreManager instance"""
        return await self.get_manager('local_lore')
    
    async def get_religion_manager(self):
        """Get the ReligionManager instance"""
        return await self.get_manager('religion')
    
    async def get_world_politics_manager(self):
        """Get the WorldPoliticsManager instance"""
        return await self.get_manager('world_politics')
    
    async def get_regional_culture_system(self):
        """Get the RegionalCultureSystem instance"""
        return await self.get_manager('regional_culture')
    
    async def get_educational_system_manager(self):
        """Get the EducationalSystemManager instance"""
        return await self.get_manager('educational')
    
    async def get_lore_update_manager(self):
        """Get the LoreUpdateManager instance"""
        return await self.get_manager('lore_update')
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_complete_world",
        action_description="Generating complete matriarchal world lore",
        id_from_context=lambda ctx: "lore_registry"
    )
    async def generate_complete_world(self, ctx, environment_desc: str) -> Dict[str, Any]:
        """
        Generate a complete world with matriarchal theming
        
        Args:
            environment_desc: Description of the environment/setting
            
        Returns:
            Dictionary containing the complete world lore
        """
        # Get all necessary managers
        geo_manager = await self.get_geopolitical_manager()
        religion_manager = await self.get_religion_manager()
        culture_system = await self.get_regional_culture_system()
        conflict_system = await self.get_world_politics_manager()
        
        # 1. Generate foundation lore
        foundation_data = await self._generate_foundation_lore(ctx, environment_desc)
        
        # 2. Generate nations and geographic regions
        nations_data = await geo_manager.generate_world_nations(ctx, 5)
        
        # 3. Generate religious pantheons and distribute them
        religious_data = await religion_manager.generate_complete_faith_system(ctx)
        
        # 4. Generate languages through regional culture system
        languages = await culture_system.generate_languages(ctx, count=5)
        
        # 5. Generate conflicts and domestic issues
        conflicts = await conflict_system.generate_initial_conflicts(ctx, count=3)
        
        # 6. For each nation, generate cultural norms and etiquette
        nation_cultures = []
        for nation in nations_data:
            # Generate cultural norms
            norms = await culture_system.generate_cultural_norms(ctx, nation["id"])
            
            # Generate etiquette
            etiquette = await culture_system.generate_etiquette(ctx, nation["id"])
            
            # Generate domestic issues
            issues = await conflict_system.generate_domestic_issues(ctx, nation["id"])
            
            nation_cultures.append({
                "nation_id": nation["id"],
                "name": nation["name"],
                "cultural_norms": norms,
                "etiquette": etiquette,
                "domestic_issues": issues
            })
        
        # Combine all results
        complete_lore = {
            "world_lore": foundation_data,
            "nations": nations_data,
            "religions": religious_data,
            "languages": languages,
            "conflicts": conflicts,
            "nation_cultures": nation_cultures
        }
        
        return complete_lore
    
    async def _generate_foundation_lore(self, ctx, environment_desc: str) -> Dict[str, Any]:
        """Generate foundation lore with matriarchal theming"""
        # Create base foundation data
        foundation_data = {
            "cosmology": await self._generate_cosmology(ctx, environment_desc),
            "magic_system": await self._generate_magic_system(ctx, environment_desc),
            "social_structure": await self._generate_social_structure(ctx, environment_desc),
            "world_history": await self._generate_world_history(ctx, environment_desc),
            "calendar_system": await self._generate_calendar_system(ctx, environment_desc)
        }
        
        # Apply matriarchal theming to foundation lore
        for key, content in foundation_data.items():
            foundation_data[key] = MatriarchalThemingUtils.apply_matriarchal_theme(key, content)
        
        return foundation_data
    
    # Foundation lore generation methods
    async def _generate_cosmology(self, ctx, environment_desc: str) -> str:
        """Generate cosmology description"""
        # Implementation would use LLM to generate cosmology
        # Placeholder for demonstration
        return f"The cosmos of {environment_desc} is structured around the eternal cycle of creation and renewal."
    
    async def _generate_magic_system(self, ctx, environment_desc: str) -> str:
        """Generate magic system description"""
        # Placeholder for demonstration
        return f"Magic in {environment_desc} flows from the fundamental forces of nature and spirit."
    
    async def _generate_social_structure(self, ctx, environment_desc: str) -> str:
        """Generate social structure description"""
        # Placeholder for demonstration
        return f"Society in {environment_desc} is organized into clans and communities."
    
    async def _generate_world_history(self, ctx, environment_desc: str) -> str:
        """Generate world history description"""
        # Placeholder for demonstration
        return f"The history of {environment_desc} spans many ages of glory and conflict."
    
    async def _generate_calendar_system(self, ctx, environment_desc: str) -> str:
        """Generate calendar system description"""
        # Placeholder for demonstration
        return f"Time in {environment_desc} is measured through cycles of the moons and stars."
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="handle_narrative_event",
        action_description="Handling narrative event impacts",
        id_from_context=lambda ctx: "lore_registry"
    )
    async def handle_narrative_event(
        self, 
        ctx,
        event_description: str,
        affected_lore_ids: List[str] = None,
        player_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Handle impacts of a narrative event on the world
        
        Args:
            event_description: Description of the event that occurred
            affected_lore_ids: Optional list of specifically affected lore IDs
            player_data: Optional player character data
            
        Returns:
            Dictionary with all updates applied
        """
        # Get the lore update manager
        update_manager = await self.get_lore_update_manager()
        
        # Let the update manager handle it
        return await update_manager.handle_narrative_event(
            ctx, 
            event_description, 
            affected_lore_ids, 
            player_data
        )
