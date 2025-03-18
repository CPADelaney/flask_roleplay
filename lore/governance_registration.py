# lore/governance_registration.py

import logging
import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from .base_manager import BaseManager
from .resource_manager import resource_manager

from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from .lore_system import LoreSystem
from .lore_validation import LoreValidator
from .error_handler import ErrorHandler
from .dynamic_lore_generator import DynamicLoreGenerator
from .unified_validation import ValidationManager

logger = logging.getLogger(__name__)

# Initialize components
lore_system = DynamicLoreGenerator()
lore_validator = ValidationManager()
error_handler = ErrorHandler()

class GovernanceRegistration(BaseManager):
    """Manager for governance registration with resource management support."""
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        max_size_mb: float = 100,
        redis_url: Optional[str] = None
    ):
        super().__init__(user_id, conversation_id, max_size_mb, redis_url)
        self.registration_data = {}
        self.resource_manager = resource_manager
    
    async def start(self):
        """Start the governance registration manager and resource management."""
        await super().start()
        await self.resource_manager.start()
    
    async def stop(self):
        """Stop the governance registration manager and cleanup resources."""
        await super().stop()
        await self.resource_manager.stop()
    
    async def get_registration_data(
        self,
        registration_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get registration data from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('registration', registration_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting registration data: {e}")
            return None
    
    async def set_registration_data(
        self,
        registration_id: str,
        data: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set registration data in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('registration', registration_id, data, tags)
        except Exception as e:
            logger.error(f"Error setting registration data: {e}")
            return False
    
    async def invalidate_registration_data(
        self,
        registration_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate registration data cache."""
        try:
            await self.invalidate_cached_data('registration', registration_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating registration data: {e}")
    
    async def get_registration_history(
        self,
        registration_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get registration history from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('registration_history', registration_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting registration history: {e}")
            return None
    
    async def set_registration_history(
        self,
        registration_id: str,
        history: List[Dict[str, Any]],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set registration history in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('registration_history', registration_id, history, tags)
        except Exception as e:
            logger.error(f"Error setting registration history: {e}")
            return False
    
    async def invalidate_registration_history(
        self,
        registration_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate registration history cache."""
        try:
            await self.invalidate_cached_data('registration_history', registration_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating registration history: {e}")
    
    async def get_registration_metadata(
        self,
        registration_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get registration metadata from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('registration_metadata', registration_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting registration metadata: {e}")
            return None
    
    async def set_registration_metadata(
        self,
        registration_id: str,
        metadata: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set registration metadata in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('registration_metadata', registration_id, metadata, tags)
        except Exception as e:
            logger.error(f"Error setting registration metadata: {e}")
            return False
    
    async def invalidate_registration_metadata(
        self,
        registration_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate registration metadata cache."""
        try:
            await self.invalidate_cached_data('registration_metadata', registration_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating registration metadata: {e}")
    
    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        try:
            return await self.resource_manager.get_resource_stats()
        except Exception as e:
            logger.error(f"Error getting resource stats: {e}")
            return {}
    
    async def optimize_resources(self):
        """Optimize resource usage."""
        try:
            await self.resource_manager._optimize_resource_usage('memory')
        except Exception as e:
            logger.error(f"Error optimizing resources: {e}")
    
    async def cleanup_resources(self):
        """Clean up unused resources."""
        try:
            await self.resource_manager._cleanup_all_resources()
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")

# Create a singleton instance for easy access
governance_registration = GovernanceRegistration()

async def register_all_lore_modules_with_governance(user_id: int, conversation_id: int) -> Dict[str, bool]:
    """
    Register all lore modules with Nyx governance.
    
    This function ensures that all lore-related modules are properly
    registered with the central Nyx governance system. It also issues
    standard directives for proper operation.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Dictionary of registration results by module name
    """
    # Get the Nyx governance system
    governance = await get_central_governance(user_id, conversation_id)
    
    # Track registration results
    registration_results = {}
    
    # Register lore agents
    try:
        from lore.lore_agents import register_with_governance as register_lore_agents
        await register_lore_agents(user_id, conversation_id)
        registration_results["lore_agents"] = True
        logger.info("Lore agents registered with Nyx governance")
    except Exception as e:
        logger.error(f"Error registering lore agents: {e}")
        registration_results["lore_agents"] = False
    
    # Register lore manager
    try:
        from lore.lore_manager import LoreManager
        lore_manager = LoreManager(user_id, conversation_id)
        await lore_manager.register_with_governance()
        registration_results["lore_manager"] = True
        logger.info("Lore manager registered with Nyx governance")
    except Exception as e:
        logger.error(f"Error registering lore manager: {e}")
        registration_results["lore_manager"] = False
    
    # Register dynamic lore generator
    try:
        from lore.dynamic_lore_generator import DynamicLoreGenerator
        lore_generator = DynamicLoreGenerator(user_id, conversation_id)
        await lore_generator.initialize_governance()
        registration_results["lore_generator"] = True
        logger.info("Dynamic lore generator registered with Nyx governance")
    except Exception as e:
        logger.error(f"Error registering dynamic lore generator: {e}")
        registration_results["lore_generator"] = False
    
    # Register setting analyzer
    try:
        from lore.setting_analyzer import SettingAnalyzer
        analyzer = SettingAnalyzer(user_id, conversation_id)
        await analyzer.register_with_governance()
        registration_results["setting_analyzer"] = True
        logger.info("Setting analyzer registered with Nyx governance")
    except Exception as e:
        logger.error(f"Error registering setting analyzer: {e}")
        registration_results["setting_analyzer"] = False
    
    # Register lore integration system
    try:
        from lore.lore_integration import LoreIntegrationSystem
        integration_system = LoreIntegrationSystem(user_id, conversation_id)
        await integration_system.initialize_governance()
        registration_results["lore_integration"] = True
        logger.info("Lore integration system registered with Nyx governance")
    except Exception as e:
        logger.error(f"Error registering lore integration system: {e}")
        registration_results["lore_integration"] = False
    
    # Register NPC lore integration
    try:
        from lore.npc_lore_integration import NPCLoreIntegration
        npc_lore = NPCLoreIntegration(user_id, conversation_id)
        await npc_lore.register_with_nyx_governance()
        registration_results["npc_lore_integration"] = True
        logger.info("NPC lore integration registered with Nyx governance")
    except Exception as e:
        logger.error(f"Error registering NPC lore integration: {e}")
        registration_results["npc_lore_integration"] = False
    
    # Issue directives for lore system
    await issue_standard_lore_directives(governance)
    
    return registration_results

async def issue_standard_lore_directives(governance) -> List[int]:
    """
    Issue standard directives for the lore system.
    
    Args:
        governance: The Nyx governance instance
        
    Returns:
        List of issued directive IDs
    """
    directive_ids = []
    
    # Directive for lore generation
    try:
        directive_id = await governance.issue_directive(
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
        directive_ids.append(directive_id)
    except Exception as e:
        logger.error(f"Error issuing lore generation directive: {e}")
    
    # Directive for NPC lore integration
    try:
        directive_id = await governance.issue_directive(
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
        directive_ids.append(directive_id)
    except Exception as e:
        logger.error(f"Error issuing NPC lore directive: {e}")
    
    # Directive for lore manager
    try:
        directive_id = await governance.issue_directive(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_manager",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Maintain lore knowledge system and ensure proper discovery opportunities.",
                "scope": "narrative"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24*60  # 24 hours
        )
        directive_ids.append(directive_id)
    except Exception as e:
        logger.error(f"Error issuing lore manager directive: {e}")
    
    # Directive for setting analyzer
    try:
        directive_id = await governance.issue_directive(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="setting_analyzer",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Analyze setting data to generate coherent organizations.",
                "scope": "setting"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24*60  # 24 hours
        )
        directive_ids.append(directive_id)
    except Exception as e:
        logger.error(f"Error issuing setting analyzer directive: {e}")
    
    return directive_ids

async def register_lore_module_with_governance(
    module_name: str,
    user_id: int,
    conversation_id: int,
    agent_id: str = None
) -> bool:
    """
    Register a specific lore module with Nyx governance.
    
    Args:
        module_name: Name of the module to register
        user_id: User ID
        conversation_id: Conversation ID
        agent_id: Optional agent ID (defaults to module_name)
        
    Returns:
        Registration success status
    """
    # Get the Nyx governance system
    governance = await get_central_governance(user_id, conversation_id)
    agent_id = agent_id or module_name
    
    # Map module names to registration functions
    module_map = {
        "lore_agents": register_lore_agents,
        "lore_manager": register_lore_manager,
        "lore_generator": register_lore_generator,
        "setting_analyzer": register_setting_analyzer,
        "lore_integration": register_lore_integration,
        "npc_lore_integration": register_npc_lore_integration
    }
    
    # Check if module exists
    if module_name not in module_map:
        logger.error(f"Unknown lore module: {module_name}")
        return False
    
    # Register the module
    try:
        await module_map[module_name](user_id, conversation_id, governance, agent_id)
        logger.info(f"Lore module {module_name} registered with Nyx governance")
        return True
    except Exception as e:
        logger.error(f"Error registering lore module {module_name}: {e}")
        return False

# Individual registration functions

async def register_lore_agents(
    user_id: int,
    conversation_id: int,
    governance: Any,
    agent_id: str = "lore_agents"
) -> bool:
    """Register lore agents with governance"""
    try:
        await governance.register_agent(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id=agent_id,
            agent_instance=None  # Will be instantiated when needed
        )
        return True
    except Exception as e:
        logger.error(f"Error registering lore agents: {e}")
        return False

async def register_lore_manager(
    user_id: int,
    conversation_id: int,
    governance: Any,
    agent_id: str = "lore_manager"
) -> bool:
    """Register lore manager with governance"""
    try:
        await governance.register_agent(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id=agent_id,
            agent_instance=None  # Will be instantiated when needed
        )
        return True
    except Exception as e:
        logger.error(f"Error registering lore manager: {e}")
        return False

async def register_lore_generator(
    user_id: int,
    conversation_id: int,
    governance: Any,
    agent_id: str = "lore_generator"
) -> bool:
    """Register lore generator with governance"""
    try:
        await governance.register_agent(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id=agent_id,
            agent_instance=None  # Will be instantiated when needed
        )
        return True
    except Exception as e:
        logger.error(f"Error registering lore generator: {e}")
        return False

async def register_setting_analyzer(
    user_id: int,
    conversation_id: int,
    governance: Any,
    agent_id: str = "setting_analyzer"
) -> bool:
    """Register setting analyzer with governance"""
    try:
        await governance.register_agent(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id=agent_id,
            agent_instance=None  # Will be instantiated when needed
        )
        return True
    except Exception as e:
        logger.error(f"Error registering setting analyzer: {e}")
        return False

async def register_lore_integration(
    user_id: int,
    conversation_id: int,
    governance: Any,
    agent_id: str = "lore_integration"
) -> bool:
    """Register lore integration with governance"""
    try:
        await governance.register_agent(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id=agent_id,
            agent_instance=None  # Will be instantiated when needed
        )
        return True
    except Exception as e:
        logger.error(f"Error registering lore integration: {e}")
        return False

async def register_npc_lore_integration(
    user_id: int,
    conversation_id: int,
    governance: Any,
    agent_id: str = "npc_lore_integration"
) -> bool:
    """Register NPC lore integration with governance"""
    try:
        await governance.register_agent(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id=agent_id,
            agent_instance=None  # Will be instantiated when needed
        )
        return True
    except Exception as e:
        logger.error(f"Error registering NPC lore integration: {e}")
        return False
