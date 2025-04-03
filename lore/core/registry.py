# lore/core/registry.py

from typing import Dict, Any

class ManagerRegistry:
    """
    Enhanced registry for all manager classes with lazy loading and dependency injection.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._managers = {}
        self._class_map = {}  # This will be populated in __init__.py
    
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
    
    async def get_master_lore_system(self):
        """Get the MatriarchalLoreSystem instance"""
        return await self.get_manager('master')
    
    async def initialize_all(self):
        """Initialize all manager instances."""
        for key in self._class_map.keys():
            await self.get_manager(key)
