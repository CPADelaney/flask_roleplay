# lore/unified_lore_system.py
"""
Unified lore system with governance integration.
This module consolidates functionality from:
- governance_registration.py
- lore_agents.py
- lore_directive_handler.py

Provides a complete lore system with governance integration, 
agent functionality, and directive handling in a single module.
"""

import logging
import asyncio
import json
import time
import psutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable

# Agents SDK imports
from agents import Agent, ModelSettings, function_tool, Runner, trace
from agents.models.openai_responses import OpenAIResponsesModel

# Nyx governance integration
from nyx.integrate import get_central_governance
from nyx.nyx_governance import (
    AgentType, 
    DirectiveType, 
    DirectivePriority
)
from nyx.governance_helpers import (
    with_governance_permission,
    with_action_reporting, 
    with_governance
)
from nyx.directive_handler import DirectiveHandler

# Import local modules
from .base_manager import BaseManager
from .resource_manager import resource_manager
from .lore_system import LoreSystem
from .lore_validation import LoreValidator
from .error_handler import ErrorHandler
from .lore_cache_manager import LoreCacheManager
from .dynamic_lore_generator import DynamicLoreGenerator
from .unified_validation import ValidationManager

# Pydantic schemas for outputs
from .unified_schemas import (
    FoundationLoreOutput,
    FactionsOutput,
    CulturalElementsOutput,
    HistoricalEventsOutput,
    LocationsOutput,
    QuestsOutput,
    IntegrationOutput,
    ConflictResolutionOutput,
    ValidationOutput,
    FixOutput
)

# Set up logging
logger = logging.getLogger(__name__)

# Initialize components (only once)
lore_system = DynamicLoreGenerator()
validation_manager = ValidationManager()
error_handler = ErrorHandler()

# -------------------------------------------------------------------------------
# Core Governance & Resource Management
# -------------------------------------------------------------------------------

class LoreGovernance(BaseManager):
    """
    Unified governance manager for lore system.
    Handles registration, directive processing, permissions, and resource management.
    """
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        agent_type: str = AgentType.NARRATIVE_CRAFTER,
        agent_id: str = "lore_generator",
        max_size_mb: float = 100,
        redis_url: Optional[str] = None
    ):
        super().__init__(user_id, conversation_id, max_size_mb, redis_url)
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.resource_manager = resource_manager
        self.governor = None
        self.directive_handler = None
        
        # Store directive-related data
        self.prohibited_actions = []
        self.action_modifications = {}
        self.registration_data = {}
        self.agent_data = {}
    
    async def start(self):
        """Start the governance manager and resource management."""
        await super().start()
        await self.resource_manager.start()
        
        # Initialize governance connection
        self.governor = await get_central_governance(self.user_id, self.conversation_id)
        
        # Initialize directive handler
        self.directive_handler = DirectiveHandler(
            self.user_id, 
            self.conversation_id, 
            self.agent_type,
            self.agent_id
        )
        
        # Register handlers for different directive types
        self.directive_handler.register_handler(DirectiveType.ACTION, self._handle_action_directive)
        self.directive_handler.register_handler(DirectiveType.PROHIBITION, self._handle_prohibition_directive)
        
        # Start background processing of directives
        self.directive_task = await self.directive_handler.start_background_processing(interval=60.0)
    
    async def stop(self):
        """Stop the governance manager and cleanup resources."""
        await super().stop()
        await self.resource_manager.stop()
        
        # Stop directive handler
        if self.directive_task:
            self.directive_task.cancel()
            try:
                await self.directive_task
            except asyncio.CancelledError:
                pass
    
    # Registration Methods
    
    async def register_all_lore_modules(self) -> Dict[str, bool]:
        """
        Register all lore modules with Nyx governance.
        
        Returns:
            Dictionary of registration results by module name
        """
        # Track registration results
        registration_results = {}
        
        # Register lore agents
        try:
            await self.register_lore_module("lore_agents")
            registration_results["lore_agents"] = True
            logger.info("Lore agents registered with Nyx governance")
        except Exception as e:
            logger.error(f"Error registering lore agents: {e}")
            registration_results["lore_agents"] = False
        
        # Register other modules
        modules = [
            "lore_manager", "lore_generator", "setting_analyzer", 
            "lore_integration", "npc_lore_integration"
        ]
        
        for module in modules:
            try:
                await self.register_lore_module(module)
                registration_results[module] = True
                logger.info(f"{module} registered with Nyx governance")
            except Exception as e:
                logger.error(f"Error registering {module}: {e}")
                registration_results[module] = False
        
        # Issue directives for lore system
        await self.issue_standard_directives()
        
        return registration_results
    
    async def register_lore_module(
        self,
        module_name: str,
        agent_id: str = None
    ) -> bool:
        """
        Register a specific lore module with Nyx governance.
        
        Args:
            module_name: Name of the module to register
            agent_id: Optional agent ID (defaults to module_name)
            
        Returns:
            Registration success status
        """
        agent_id = agent_id or module_name
        
        # Register the module with appropriate agent type
        try:
            await self.governor.register_agent(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id=agent_id,
                agent_instance=None  # Will be instantiated when needed
            )
            return True
        except Exception as e:
            logger.error(f"Error registering module {module_name}: {e}")
            return False
    
    async def issue_standard_directives(self) -> List[int]:
        """
        Issue standard directives for the lore system.
        
        Returns:
            List of issued directive IDs
        """
        directive_ids = []
        
        # Define standard directives
        standard_directives = [
            {
                "agent_type": AgentType.NARRATIVE_CRAFTER,
                "agent_id": "lore_generator",
                "directive_type": DirectiveType.ACTION,
                "directive_data": {
                    "instruction": "Maintain world lore consistency and generate new lore as needed.",
                    "scope": "narrative"
                },
                "priority": DirectivePriority.MEDIUM
            },
            {
                "agent_type": AgentType.NARRATIVE_CRAFTER,
                "agent_id": "npc_lore_integration",
                "directive_type": DirectiveType.ACTION,
                "directive_data": {
                    "instruction": "Ensure NPCs have appropriate lore knowledge based on their backgrounds.",
                    "scope": "narrative"
                },
                "priority": DirectivePriority.MEDIUM
            },
            {
                "agent_type": AgentType.NARRATIVE_CRAFTER,
                "agent_id": "lore_manager",
                "directive_type": DirectiveType.ACTION,
                "directive_data": {
                    "instruction": "Maintain lore knowledge system and ensure proper discovery opportunities.",
                    "scope": "narrative"
                },
                "priority": DirectivePriority.MEDIUM
            },
            {
                "agent_type": AgentType.NARRATIVE_CRAFTER,
                "agent_id": "setting_analyzer",
                "directive_type": DirectiveType.ACTION,
                "directive_data": {
                    "instruction": "Analyze setting data to generate coherent organizations.",
                    "scope": "setting"
                },
                "priority": DirectivePriority.MEDIUM
            }
        ]
        
        # Issue each directive
        for directive in standard_directives:
            try:
                directive_id = await self.governor.issue_directive(
                    agent_type=directive["agent_type"],
                    agent_id=directive["agent_id"],
                    directive_type=directive["directive_type"],
                    directive_data=directive["directive_data"],
                    priority=directive["priority"],
                    duration_minutes=24*60  # 24 hours
                )
                directive_ids.append(directive_id)
            except Exception as e:
                logger.error(f"Error issuing directive: {e}")
        
        return directive_ids
    
    # Directive Handling Methods
    
    async def _handle_action_directive(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle action directives.
        
        Args:
            directive: The directive data
            
        Returns:
            Result of processing
        """
        directive_id = directive.get("id")
        instruction = directive.get("instruction", "")
        
        logger.info(f"Processing action directive {directive_id}: {instruction}")
        
        if "generate_lore" in instruction.lower():
            # Handle lore generation directive
            environment_desc = directive.get("environment_desc", "")
            if environment_desc:
                result = await lore_system.generate_complete_lore(environment_desc)
                return {
                    "status": "completed",
                    "directive_id": directive_id,
                    "lore_generated": True,
                    "environment": environment_desc
                }
        
        elif "integrate_lore" in instruction.lower():
            # Handle lore integration directive
            npc_ids = directive.get("npc_ids", [])
            if npc_ids:
                from .lore_integration import LoreIntegrationSystem
                integration_system = LoreIntegrationSystem(self.user_id, self.conversation_id)
                result = await integration_system.integrate_lore_with_npcs(npc_ids)
                return {
                    "status": "completed",
                    "directive_id": directive_id,
                    "npcs_integrated": len(npc_ids)
                }
        
        elif "analyze_setting" in instruction.lower():
            # Handle setting analysis directive
            from .setting_analyzer import SettingAnalyzer
            analyzer = SettingAnalyzer(self.user_id, self.conversation_id)
            await analyzer.initialize_governance()
            result = await analyzer.aggregate_npc_data(None)
            return {
                "status": "completed",
                "directive_id": directive_id,
                "setting_analyzed": True,
                "npc_count": len(result.get("npcs", []))
            }
        
        elif "modify_action" in instruction.lower():
            # Store action modifications for future use
            action_type = directive.get("action_type")
            modifications = directive.get("modifications", {})
            
            if action_type:
                self.action_modifications[action_type] = modifications
                return {
                    "status": "completed",
                    "directive_id": directive_id,
                    "action_type": action_type,
                    "modifications_stored": True
                }
        
        # Default unknown directive case
        return {
            "status": "unknown_directive",
            "directive_id": directive_id,
            "instruction": instruction
        }
    
    async def _handle_prohibition_directive(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle prohibition directives.
        
        Args:
            directive: The directive data
            
        Returns:
            Result of processing
        """
        directive_id = directive.get("id")
        prohibited_actions = directive.get("prohibited_actions", [])
        reason = directive.get("reason", "No reason provided")
        
        logger.info(f"Processing prohibition directive {directive_id}: {prohibited_actions}")
        
        # Store prohibited actions
        self.prohibited_actions.extend(prohibited_actions)
        
        # Remove duplicates
        self.prohibited_actions = list(set(self.prohibited_actions))
        
        return {
            "status": "prohibition_registered",
            "directive_id": directive_id,
            "prohibited_actions": self.prohibited_actions,
            "reason": reason
        }
    
    async def check_permission(self, action_type: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check if an action is permitted based on directives.
        
        Args:
            action_type: Type of action to check
            details: Optional action details
            
        Returns:
            Permission status dictionary
        """
        # Check if action is prohibited
        if action_type in self.prohibited_actions or "*" in self.prohibited_actions:
            return {
                "approved": False,
                "reasoning": f"Action {action_type} is prohibited by Nyx directive",
                "directive_applied": True
            }
        
        # Check if action has modifications
        if action_type in self.action_modifications:
            modifications = self.action_modifications[action_type]
            return {
                "approved": True,
                "reasoning": f"Action {action_type} is modified by Nyx directive",
                "directive_applied": True,
                "modifications": modifications
            }
        
        # Default approval
        return {
            "approved": True,
            "reasoning": "No directives prohibit this action",
            "directive_applied": False
        }
    
    async def process_directives(self, force_check: bool = False) -> Dict[str, Any]:
        """
        Process all active directives for this lore agent.
        
        Args:
            force_check: Whether to force checking directives
            
        Returns:
            Processing results
        """
        return await self.directive_handler.process_directives(force_check)
    
    async def apply_directive_to_response(self, response: Any, action_type: str) -> Any:
        """
        Apply any directive modifications to a response.
        
        Args:
            response: Original response
            action_type: Type of action
            
        Returns:
            Modified response if applicable, otherwise original
        """
        # If action is prohibited, return error response
        if action_type in self.prohibited_actions or "*" in self.prohibited_actions:
            if isinstance(response, dict):
                return {
                    "error": f"Action {action_type} is prohibited by Nyx directive",
                    "approved": False
                }
            return response
        
        # Apply modifications if any
        modifications = self.action_modifications.get(action_type, {})
        if modifications and isinstance(response, dict):
            # Apply each modification
            for key, value in modifications.items():
                if key in response:
                    response[key] = value
        
        return response
    
    # Resource Management Methods
    
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
    
    async def get_agent_data(
        self,
        agent_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get agent data from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('agent', agent_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting agent data: {e}")
            return None
    
    async def set_agent_data(
        self,
        agent_id: str,
        data: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set agent data in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('agent', agent_id, data, tags)
        except Exception as e:
            logger.error(f"Error setting agent data: {e}")
            return False
    
    async def invalidate_agent_data(
        self,
        agent_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate agent data cache."""
        try:
            await self.invalidate_cached_data('agent', agent_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating agent data: {e}")
    
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
lore_governance = LoreGovernance(0, 0)  # Will be properly initialized when needed

def get_lore_governance(user_id: int, conversation_id: int) -> LoreGovernance:
    """Get or initialize the lore governance instance."""
    global lore_governance
    
    # Initialize if needed or if different user/conversation
    if lore_governance.user_id != user_id or lore_governance.conversation_id != conversation_id:
        lore_governance = LoreGovernance(user_id, conversation_id)
    
    return lore_governance

# -------------------------------------------------------------------------------
# Agent Definitions
# -------------------------------------------------------------------------------

# Foundation lore agent
foundation_lore_agent = Agent(
    name="FoundationLoreAgent",
    instructions=(
        "You produce foundational world lore for a fantasy environment. "
        "Return valid JSON that matches FoundationLoreOutput, which has keys: "
        "[cosmology, magic_system, world_history, calendar_system, social_structure]. "
        "Do NOT include any extra text outside the JSON.\n\n"
        "Always respect directives from the Nyx governance system and check permissions "
        "before performing any actions."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.4),
    output_type=FoundationLoreOutput,
)

# Factions agent
factions_agent = Agent(
    name="FactionsAgent",
    instructions=(
        "You generate 3-5 distinct factions for a given setting. "
        "Return valid JSON as an array of objects, matching FactionsOutput. "
        "Each faction object has: name, type, description, values, goals, "
        "headquarters, rivals, allies, hierarchy_type, etc. "
        "No extra text outside the JSON.\n\n"
        "Always respect directives from the Nyx governance system and check permissions "
        "before performing any actions."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.7),
    output_type=FactionsOutput,
)

# Cultural elements agent
cultural_agent = Agent(
    name="CulturalAgent",
    instructions=(
        "You create cultural elements like traditions, customs, rituals. "
        "Return JSON matching CulturalElementsOutput: an array of objects. "
        "Fields include: name, type, description, practiced_by, significance, "
        "historical_origin. No extra text outside the JSON.\n\n"
        "Always respect directives from the Nyx governance system and check permissions "
        "before performing any actions."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.5),
    output_type=CulturalElementsOutput,
)

# Historical events agent
history_agent = Agent(
    name="HistoryAgent",
    instructions=(
        "You create major historical events. Return JSON matching "
        "HistoricalEventsOutput: an array with fields name, date_description, "
        "description, participating_factions, consequences, significance. "
        "No extra text outside the JSON.\n\n"
        "Always respect directives from the Nyx governance system and check permissions "
        "before performing any actions."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.6),
    output_type=HistoricalEventsOutput,
)

# Locations agent
locations_agent = Agent(
    name="LocationsAgent",
    instructions=(
        "You generate 5-8 significant locations. Return JSON matching "
        "LocationsOutput: an array of objects with fields name, description, "
        "type, controlling_faction, notable_features, hidden_secrets, "
        "strategic_importance. No extra text outside the JSON.\n\n"
        "Always respect directives from the Nyx governance system and check permissions "
        "before performing any actions."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.7),
    output_type=LocationsOutput,
)

# Quests agent
quests_agent = Agent(
    name="QuestsAgent",
    instructions=(
        "You create 5-7 quest hooks. Return JSON matching QuestsOutput: an "
        "array of objects with quest_name, quest_giver, location, description, "
        "objectives, rewards, difficulty, lore_significance. "
        "No extra text outside the JSON.\n\n"
        "Always respect directives from the Nyx governance system and check permissions "
        "before performing any actions."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.7),
    output_type=QuestsOutput,
)

# Setting Analysis Agent
setting_analysis_agent = Agent(
    name="SettingAnalysisAgent",
    instructions=(
        "You analyze the game setting and NPC data to propose relevant "
        "organizations or factions. Return JSON with categories like "
        "academic, athletic, social, professional, cultural, political, other. "
        "Each category is an array of objects with fields like name, type, "
        "description, membership_basis, hierarchy, gathering_location, etc. "
        "No extra text outside the JSON.\n\n"
        "Always respect directives from the Nyx governance system and check permissions "
        "before performing any actions."
    ),
    model=OpenAIResponsesModel(model="o3-mini"),
    model_settings=ModelSettings(temperature=0.7),
)

# -------------------------------------------------------------------------------
# Function Tools with Governance Integration
# -------------------------------------------------------------------------------

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_foundation_lore",
    action_description="Generating foundation lore for environment: {environment_desc}",
    id_from_context=lambda ctx: "foundation_lore"
)
async def generate_foundation_lore(ctx, environment_desc: str) -> Dict[str, Any]:
    """
    Generate foundation lore (cosmology, magic system, etc.) for a given environment
    with Nyx governance oversight.
    
    Args:
        environment_desc: Environment description
    """
    # Get governance for permission checking
    user_id = ctx.context.get("user_id", 0)
    conversation_id = ctx.context.get("conversation_id", 0)
    governance = get_lore_governance(user_id, conversation_id)
    
    # Check permission
    permission = await governance.check_permission("generate_foundation_lore")
    if not permission.get("approved", False):
        return {"error": permission.get("reasoning", "Action not approved")}
    
    user_prompt = f"""
    Generate cohesive foundational world lore for this environment:
    {environment_desc}

    Return as JSON with keys:
    cosmology, magic_system, world_history, calendar_system, social_structure
    """
    
    result = await Runner.run(foundation_lore_agent, user_prompt, context=ctx.context)
    final_output = result.final_output_as(FoundationLoreOutput)
    
    # Apply any directive modifications
    return await governance.apply_directive_to_response(final_output.dict(), "generate_foundation_lore")

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_factions",
    action_description="Generating factions for environment: {environment_desc}",
    id_from_context=lambda ctx: "factions"
)
async def generate_factions(ctx, environment_desc: str, social_structure: str) -> List[Dict[str, Any]]:
    """
    Generate 3-5 distinct factions referencing environment_desc + social_structure
    with Nyx governance oversight.
    
    Args:
        environment_desc: Environment description
        social_structure: Social structure description
    """
    # Get governance for permission checking
    user_id = ctx.context.get("user_id", 0)
    conversation_id = ctx.context.get("conversation_id", 0)
    governance = get_lore_governance(user_id, conversation_id)
    
    # Check permission
    permission = await governance.check_permission("generate_factions")
    if not permission.get("approved", False):
        return [{"error": permission.get("reasoning", "Action not approved")}]
    
    user_prompt = f"""
    Generate 3-5 distinct factions for this environment:
    Environment: {environment_desc}
    Social Structure: {social_structure}
    
    Return JSON as an array of objects (matching FactionsOutput).
    """
    
    result = await Runner.run(factions_agent, user_prompt, context=ctx.context)
    final_output = result.final_output_as(FactionsOutput)
    
    # Convert to list of dicts
    factions = [f.dict() for f in final_output.__root__]
    
    # Apply any directive
# Apply any directive modifications
    return await governance.apply_directive_to_response(factions, "generate_factions")

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_cultural_elements",
    action_description="Generating cultural elements for environment: {environment_desc}",
    id_from_context=lambda ctx: "cultural"
)
async def generate_cultural_elements(ctx, environment_desc: str, faction_names: str) -> List[Dict[str, Any]]:
    """
    Generate cultural elements (traditions, taboos, etc.) referencing environment + faction names
    with Nyx governance oversight.
    
    Args:
        environment_desc: Environment description
        faction_names: Comma-separated faction names
    """
    # Get governance for permission checking
    user_id = ctx.context.get("user_id", 0)
    conversation_id = ctx.context.get("conversation_id", 0)
    governance = get_lore_governance(user_id, conversation_id)
    
    # Check permission
    permission = await governance.check_permission("generate_cultural_elements")
    if not permission.get("approved", False):
        return [{"error": permission.get("reasoning", "Action not approved")}]
    
    user_prompt = f"""
    Generate 4-7 unique cultural elements for:
    Environment: {environment_desc}
    Factions: {faction_names}

    Return JSON array matching CulturalElementsOutput.
    """
    
    result = await Runner.run(cultural_agent, user_prompt, context=ctx.context)
    final_output = result.final_output_as(CulturalElementsOutput)
    
    # Convert to list of dicts
    cultural_elements = [c.dict() for c in final_output.__root__]
    
    # Apply any directive modifications
    return await governance.apply_directive_to_response(cultural_elements, "generate_cultural_elements")

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_historical_events",
    action_description="Generating historical events for environment: {environment_desc}",
    id_from_context=lambda ctx: "history"
)
async def generate_historical_events(ctx, environment_desc: str, world_history: str, faction_names: str) -> List[Dict[str, Any]]:
    """
    Generate historical events referencing environment, existing world_history, faction_names
    with Nyx governance oversight.
    
    Args:
        environment_desc: Environment description
        world_history: Existing world history
        faction_names: Comma-separated faction names
    """
    # Get governance for permission checking
    user_id = ctx.context.get("user_id", 0)
    conversation_id = ctx.context.get("conversation_id", 0)
    governance = get_lore_governance(user_id, conversation_id)
    
    # Check permission
    permission = await governance.check_permission("generate_historical_events")
    if not permission.get("approved", False):
        return [{"error": permission.get("reasoning", "Action not approved")}]
    
    user_prompt = f"""
    Generate 5-7 significant historical events:
    Environment: {environment_desc}
    Existing World History: {world_history}
    Factions: {faction_names}

    Return JSON array matching HistoricalEventsOutput.
    """
    
    result = await Runner.run(history_agent, user_prompt, context=ctx.context)
    final_output = result.final_output_as(HistoricalEventsOutput)
    
    # Convert to list of dicts
    events = [h.dict() for h in final_output.__root__]
    
    # Apply any directive modifications
    return await governance.apply_directive_to_response(events, "generate_historical_events")

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_locations",
    action_description="Generating locations for environment: {environment_desc}",
    id_from_context=lambda ctx: "locations"
)
async def generate_locations(ctx, environment_desc: str, faction_names: str) -> List[Dict[str, Any]]:
    """
    Generate 5-8 significant locations referencing environment_desc + faction names
    with Nyx governance oversight.
    
    Args:
        environment_desc: Environment description
        faction_names: Comma-separated faction names
    """
    # Get governance for permission checking
    user_id = ctx.context.get("user_id", 0)
    conversation_id = ctx.context.get("conversation_id", 0)
    governance = get_lore_governance(user_id, conversation_id)
    
    # Check permission
    permission = await governance.check_permission("generate_locations")
    if not permission.get("approved", False):
        return [{"error": permission.get("reasoning", "Action not approved")}]
    
    user_prompt = f"""
    Generate 5-8 significant locations for:
    Environment: {environment_desc}
    Factions: {faction_names}

    Return JSON array matching LocationsOutput.
    """
    
    result = await Runner.run(locations_agent, user_prompt, context=ctx.context)
    final_output = result.final_output_as(LocationsOutput)
    
    # Convert to list of dicts
    locations = [l.dict() for l in final_output.__root__]
    
    # Apply any directive modifications
    return await governance.apply_directive_to_response(locations, "generate_locations")

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="generate_quest_hooks",
    action_description="Generating quest hooks for environment: {environment_desc}",
    id_from_context=lambda ctx: "quests"
)
async def generate_quest_hooks(ctx, environment_desc: str, faction_names: str, locations: str) -> List[Dict[str, Any]]:
    """
    Generate 3-5 quest hooks referencing environment, factions, and locations
    with Nyx governance oversight.
    
    Args:
        environment_desc: Environment description
        faction_names: Comma-separated faction names
        locations: Comma-separated location names
    """
    # Get governance for permission checking
    user_id = ctx.context.get("user_id", 0)
    conversation_id = ctx.context.get("conversation_id", 0)
    governance = get_lore_governance(user_id, conversation_id)
    
    # Check permission
    permission = await governance.check_permission("generate_quest_hooks")
    if not permission.get("approved", False):
        return [{"error": permission.get("reasoning", "Action not approved")}]
    
    user_prompt = f"""
    Generate 3-5 quest hooks for:
    Environment: {environment_desc}
    Factions: {faction_names}
    Locations: {locations}

    Return JSON array matching QuestsOutput.
    """
    
    result = await Runner.run(quests_agent, user_prompt, context=ctx.context)
    final_output = result.final_output_as(QuestsOutput)
    
    # Convert to list of dicts
    quests = [q.dict() for q in final_output.__root__]
    
    # Apply any directive modifications
    return await governance.apply_directive_to_response(quests, "generate_quest_hooks")

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="analyze_setting",
    action_description="Analyzing setting data for environment: {environment_desc}",
    id_from_context=lambda ctx: "setting"
)
async def analyze_setting(ctx, environment_desc: str) -> Dict[str, Any]:
    """
    Analyze setting data to generate coherent organizations and relationships
    with Nyx governance oversight.
    
    Args:
        environment_desc: Environment description
    """
    # Get governance for permission checking
    user_id = ctx.context.get("user_id", 0)
    conversation_id = ctx.context.get("conversation_id", 0)
    governance = get_lore_governance(user_id, conversation_id)
    
    # Check permission
    permission = await governance.check_permission("analyze_setting")
    if not permission.get("approved", False):
        return {"error": permission.get("reasoning", "Action not approved")}
    
    user_prompt = f"""
    Analyze setting data for:
    Environment: {environment_desc}

    Return JSON object with:
    - organizations
    - relationships
    - power_structures
    - cultural_norms
    - economic_systems
    """
    
    result = await Runner.run(setting_analysis_agent, user_prompt, context=ctx.context)
    
    # Apply any directive modifications
    return await governance.apply_directive_to_response(result.final_output, "analyze_setting")

# -------------------------------------------------------------------------------
# Main Functions for Lore Creation
# -------------------------------------------------------------------------------

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="create_complete_lore",
    action_description="Creating complete lore for environment: {environment_desc}",
    id_from_context=lambda ctx: "complete_lore"
)
async def create_complete_lore(ctx, environment_desc: str) -> Dict[str, Any]:
    """
    Create complete lore for an environment with Nyx governance oversight.
    
    Args:
        environment_desc: Environment description
    """
    try:
        # Generate foundation lore
        foundation = await generate_foundation_lore(ctx, environment_desc)
        
        # Generate factions
        factions = await generate_factions(ctx, environment_desc, foundation.get("social_structure", ""))
        faction_names = ", ".join(f["name"] for f in factions)
        
        # Generate cultural elements
        cultural = await generate_cultural_elements(ctx, environment_desc, faction_names)
        
        # Generate historical events
        history = await generate_historical_events(
            ctx, 
            environment_desc,
            foundation.get("world_history", ""),
            faction_names
        )
        
        # Generate locations
        locations = await generate_locations(ctx, environment_desc, faction_names)
        location_names = ", ".join(l["name"] for l in locations)
        
        # Generate quest hooks
        quests = await generate_quest_hooks(ctx, environment_desc, faction_names, location_names)
        
        # Analyze setting
        setting = await analyze_setting(ctx, environment_desc)
        
        # Combine all lore
        complete_lore = {
            "foundation": foundation,
            "factions": factions,
            "cultural_elements": cultural,
            "history": history,
            "locations": locations,
            "quests": quests,
            "setting_analysis": setting,
            "generated_at": datetime.now().isoformat()
        }
        
        return complete_lore
        
    except Exception as e:
        logger.error(f"Error creating complete lore: {e}")
        return {"error": str(e)}

@function_tool
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="integrate_lore_with_npcs",
    action_description="Integrating lore with NPCs: {npc_ids}",
    id_from_context=lambda ctx: "npc_lore"
)
async def integrate_lore_with_npcs(ctx, npc_ids: List[int], lore_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Integrate lore with NPCs with Nyx governance oversight.
    
    Args:
        npc_ids: List of NPC IDs
        lore_context: Lore context to integrate
    """
    try:
        # Get governance for permission checking
        user_id = ctx.context.get("user_id", 0)
        conversation_id = ctx.context.get("conversation_id", 0)
        governance = get_lore_governance(user_id, conversation_id)
        
        # Check permission
        permission = await governance.check_permission("integrate_lore_with_npcs")
        if not permission.get("approved", False):
            return {"error": permission.get("reasoning", "Action not approved")}
        
        # Get NPC data
        npc_data = await get_npc_data(npc_ids)
        
        # Integrate lore for each NPC
        integration_results = {}
        for npc_id, npc in npc_data.items():
            # Determine relevant lore
            relevant_lore = await determine_relevant_lore(npc_id, lore_context)
            
            # Integrate lore
            integration_result = await integrate_npc_lore(npc_id, relevant_lore)
            
            integration_results[npc_id] = integration_result
        
        result = {
            "success": True,
            "npcs_processed": len(npc_ids),
            "results": integration_results
        }
        
        # Apply any directive modifications
        return await governance.apply_directive_to_response(result, "integrate_lore_with_npcs")
        
    except Exception as e:
        logger.error(f"Error integrating lore with NPCs: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# -------------------------------------------------------------------------------
# Specialized Agent Classes
# -------------------------------------------------------------------------------

class QuestAgent:
    """Agent responsible for managing quest-related lore and progression."""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_system = lore_system
        self.governance = get_lore_governance(user_id, conversation_id)
        self.state = {
            'active_quests': {},
            'quest_progress': {},
            'quest_relationships': {}
        }
    
    async def initialize(self):
        """Initialize the quest agent."""
        try:
            # Load active quests
            active_quests = await self.lore_system.get_all_components('quest')
            self.state['active_quests'] = {
                quest['id']: quest for quest in active_quests
                if quest.get('status') == 'active'
            }
            
            # Load quest progress
            for quest_id in self.state['active_quests']:
                progress = await self.lore_system.get_quest_progression(quest_id)
                if progress:
                    self.state['quest_progress'][quest_id] = progress
            
            return True
        except Exception as e:
            logger.error(f"Error initializing QuestAgent: {e}")
            return False
    
    async def get_quest_context(self, quest_id: int) -> Dict[str, Any]:
        """Get comprehensive quest context including related lore and NPCs."""
        try:
            # Check permission
            permission = await self.governance.check_permission("get_quest_context")
            if not permission.get("approved", False):
                return {"error": permission.get("reasoning", "Action not approved")}
            
            # Get quest data
            quest = self.state['active_quests'].get(quest_id)
            if not quest:
                return {}
            
            # Get quest lore
            quest_lore = await self.lore_system.get_quest_lore(quest_id)
            
            # Get quest progression
            progression = self.state['quest_progress'].get(quest_id, {})
            
            # Get related NPCs
            related_npcs = []
            for lore in quest_lore:
                if lore.get('metadata', {}).get('npc_id'):
                    npc_id = lore['metadata']['npc_id']
                    npc_data = await self.lore_system.get_component(f"npc_{npc_id}")
                    if npc_data:
                        related_npcs.append(npc_data)
            
            result = {
                'quest': quest,
                'lore': quest_lore,
                'progression': progression,
                'related_npcs': related_npcs
            }
            
            # Apply any directive modifications
            return await self.governance.apply_directive_to_response(result, "get_quest_context")
            
        except Exception as e:
            logger.error(f"Error getting quest context: {e}")
            return {}
    
    async def update_quest_stage(self, quest_id: int, stage: str, data: Dict[str, Any]) -> bool:
        """Update quest stage with new data and handle related updates."""
        try:
            # Check permission
            permission = await self.governance.check_permission("update_quest_stage")
            if not permission.get("approved", False):
                return False
            
            # Update quest progression
            success = await self.lore_system.update_quest_progression(quest_id, stage, data)
            if not success:
                return False
            
            # Update quest state
            if quest_id in self.state['active_quests']:
                quest = self.state['active_quests'][quest_id]
                quest['current_stage'] = stage
                quest['last_updated'] = datetime.now().isoformat()
            
            # Update quest progress in state
            self.state['quest_progress'][quest_id] = {
                'stage': stage,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating quest stage: {e}")
            return False

class NarrativeAgent:
    """Agent responsible for managing narrative progression and story elements."""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_system = lore_system
        self.governance = get_lore_governance(user_id, conversation_id)
        self.state = {
            'active_narratives': {},
            'narrative_stages': {},
            'story_elements': {}
        }
    
    async def initialize(self):
        """Initialize the narrative agent."""
        try:
            # Load active narratives
            narratives = await self.lore_system.get_all_components('narrative')
            self.state['active_narratives'] = {
                narrative['id']: narrative for narrative in narratives
                if narrative.get('status') == 'active'
            }
            
            # Load narrative stages
            for narrative_id in self.state['active_narratives']:
                data = await self.lore_system.get_narrative_data(narrative_id)
                if data:
                    self.state['narrative_stages'][narrative_id] = data
            
            return True
        except Exception as e:
            logger.error(f"Error initializing NarrativeAgent: {e}")
            return False
    
    # Additional methods would follow here...

class EnvironmentAgent:
    """Agent responsible for managing environmental conditions and state."""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_system = lore_system
        self.governance = get_lore_governance(user_id, conversation_id)
        self._environmental_states = {}
        self._resource_levels = {}
        self._current_time = None
    
    async def initialize(self):
        """Initialize the environment agent."""
        try:
            # Load environmental states
            locations = await self.lore_system.get_all_components('location')
            for location in locations:
                location_id = location['id']
                state = await self.lore_system.get_environment_state(location_id)
                conditions = await self.lore_system.get_environmental_conditions(location_id)
                resources = await self.lore_system.get_location_resources(location_id)
                
                self._environmental_states[location_id] = {
                    'state': state,
                    'conditions': conditions,
                    'resources': resources
                }
            
            return True
        except Exception as e:
            logger.error(f"Error initializing EnvironmentAgent: {e}")
            return False
    
    # Additional methods would follow here...

# -------------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------------

async def get_npc_data(npc_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Get NPC data for a list of NPC IDs.
    
    Args:
        npc_ids: List of NPC IDs
        
    Returns:
        Dictionary mapping NPC IDs to their data
    """
    try:
        npc_data = {}
        for npc_id in npc_ids:
            # Get basic NPC info
            query = """
                SELECT id, name, description, faction_id, location_id, traits, 
                       relationships, knowledge, status
                FROM NPCs
                WHERE id = $1
            """
            result = await lore_system.execute_query(query, npc_id)
            if result:
                npc_data[npc_id] = result[0]
                
                # Get NPC stats
                stats_query = """
                    SELECT health, energy, influence
                    FROM NPCStats
                    WHERE npc_id = $1
                """
                stats_result = await lore_system.execute_query(stats_query, npc_id)
                if stats_result:
                    npc_data[npc_id]["stats"] = stats_result[0]
                
                # Get relationships
                rel_query = """
                    SELECT related_npc_id, relationship_type, strength
                    FROM NPCRelationships
                    WHERE npc_id = $1
                """
                rel_result = await lore_system.execute_query(rel_query, npc_id)
                if rel_result:
                    npc_data[npc_id]["relationships"] = rel_result
                
                # Get schedule
                schedule_query = """
                    SELECT schedule_data
                    FROM NPCSchedules
                    WHERE npc_id = $1
                """
                schedule_result = await lore_system.execute_query(schedule_query, npc_id)
                if schedule_result:
                    npc_data[npc_id]["schedule"] = schedule_result[0]["schedule_data"]
        
        return npc_data
        
    except Exception as e:
        logger.error(f"Error getting NPC data: {e}")
        return {}

async def determine_relevant_lore(npc_id: int, lore_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Determine which lore elements are relevant to a specific NPC.
    
    Args:
        npc_id: NPC ID to check
        lore_context: Optional lore context
        
    Returns:
        Dict of relevant lore elements
    """
    try:
        # Implementation would go here
        # This is a placeholder implementation
        return {
            'world_lore': [],
            'location_lore': [],
            'faction_lore': [],
            'historical_events': [],
            'cultural_elements': []
        }
    except Exception as e:
        logger.error(f"Error determining relevant lore: {e}")
        return {}

async def integrate_npc_lore(npc_id: int, relevant_lore: Dict[str, Any]) -> Dict[str, Any]:
    """
    Integrate relevant lore with an NPC.
    
    Args:
        npc_id: NPC ID to update
        relevant_lore: Relevant lore to integrate
        
    Returns:
        Result of integration
    """
    try:
        # Implementation would go here
        # This is a placeholder implementation
        return {
            'updated_knowledge': [],
            'updated_beliefs': [],
            'updated_relationships': [],
            'new_memories': []
        }
    except Exception as e:
        logger.error(f"Error integrating NPC lore: {e}")
        return {}

# -------------------------------------------------------------------------------
# Module Registration
# -------------------------------------------------------------------------------

async def register_with_governance(user_id: int, conversation_id: int) -> bool:
    """
    Register lore system with Nyx governance.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
    """
    try:
        # Get lore governance
        governance = get_lore_governance(user_id, conversation_id)
        
        # Start governance
        await governance.start()
        
        # Register all modules
        results = await governance.register_all_lore_modules()
        
        # Return overall success
        return all(results.values())
        
    except Exception as e:
        logger.error(f"Error registering with governance: {e}")
        return False

# -------------------------------------------------------------------------------
# Agent Creation Factory
# -------------------------------------------------------------------------------

def create_agent(agent_type: str, user_id: int, conversation_id: int) -> Any:
    """
    Create and return an agent of the specified type.
    
    Args:
        agent_type: Type of agent to create
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Agent instance
    """
    agent_map = {
        "quest": QuestAgent,
        "narrative": NarrativeAgent,
        "environment": EnvironmentAgent,
        # Add more agent types as needed
    }
    
    if agent_type in agent_map:
        return agent_map[agent_type](user_id, conversation_id)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
