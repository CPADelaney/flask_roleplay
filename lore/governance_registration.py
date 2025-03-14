# lore/governance_registration.py

import logging
import asyncio
from typing import Dict, Any, List, Optional

from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority

logger = logging.getLogger(__name__)

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

async def register_lore_agents(user_id, conversation_id, governance, agent_id):
    """Register lore agents with governance."""
    from lore.lore_agents import register_with_governance
    await register_with_governance(user_id, conversation_id)

async def register_lore_manager(user_id, conversation_id, governance, agent_id):
    """Register lore manager with governance."""
    from lore.lore_manager import LoreManager
    lore_manager = LoreManager(user_id, conversation_id)
    await lore_manager.register_with_governance()

async def register_lore_generator(user_id, conversation_id, governance, agent_id):
    """Register dynamic lore generator with governance."""
    from lore.dynamic_lore_generator import DynamicLoreGenerator
    lore_generator = DynamicLoreGenerator(user_id, conversation_id)
    await lore_generator.initialize_governance()

async def register_setting_analyzer(user_id, conversation_id, governance, agent_id):
    """Register setting analyzer with governance."""
    from lore.setting_analyzer import SettingAnalyzer
    analyzer = SettingAnalyzer(user_id, conversation_id)
    await analyzer.register_with_governance()

async def register_lore_integration(user_id, conversation_id, governance, agent_id):
    """Register lore integration system with governance."""
    from lore.lore_integration import LoreIntegrationSystem
    integration_system = LoreIntegrationSystem(user_id, conversation_id)
    await integration_system.initialize_governance()

async def register_npc_lore_integration(user_id, conversation_id, governance, agent_id):
    """Register NPC lore integration with governance."""
    from lore.npc_lore_integration import NPCLoreIntegration
    npc_lore = NPCLoreIntegration(user_id, conversation_id)
    await npc_lore.register_with_nyx_governance()
