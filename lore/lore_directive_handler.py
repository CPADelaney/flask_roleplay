# lore/lore_directive_handler.py

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union

from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType
from nyx.directive_handler import DirectiveHandler

logger = logging.getLogger(__name__)

class LoreDirectiveHandler:
    """
    Standardized handler for processing lore-related directives from Nyx governance.
    
    This class provides a unified way for all lore agents to handle directives
    from the central Nyx governance system.
    """
    
    def __init__(self, user_id: int, conversation_id: int, agent_type: str = AgentType.NARRATIVE_CRAFTER, agent_id: str = "lore_generator"):
        """
        Initialize the lore directive handler.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            agent_type: Agent type (default: NARRATIVE_CRAFTER)
            agent_id: Agent ID (default: lore_generator)
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.governor = None
        self.directive_handler = None
        
        # Store prohibited actions from directives
        self.prohibited_actions = []
        
        # Store action modifications from directives
        self.action_modifications = {}
    
    async def initialize(self):
        """Initialize the handler with Nyx governance."""
        # Get governance system
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
                # Import here to avoid circular imports
                from lore.dynamic_lore_generator import DynamicLoreGenerator
                lore_generator = DynamicLoreGenerator(self.user_id, self.conversation_id)
                result = await lore_generator.generate_complete_lore(environment_desc)
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
                # Import here to avoid circular imports
                from lore.lore_integration import LoreIntegrationSystem
                integration_system = LoreIntegrationSystem(self.user_id, self.conversation_id)
                result = await integration_system.integrate_lore_with_npcs(npc_ids)
                return {
                    "status": "completed",
                    "directive_id": directive_id,
                    "npcs_integrated": len(npc_ids)
                }
        
        elif "update_lore" in instruction.lower():
            # Handle lore update directive
            event_description = directive.get("event_description", "")
            if event_description:
                # Import here to avoid circular imports
                from lore.lore_integration import LoreIntegrationSystem
                integration_system = LoreIntegrationSystem(self.user_id, self.conversation_id)
                result = await integration_system.update_lore_after_narrative_event(event_description)
                return {
                    "status": "completed",
                    "directive_id": directive_id,
                    "lore_updated": True,
                    "event_description": event_description
                }
        
        elif "analyze_setting" in instruction.lower():
            # Handle setting analysis directive
            # Import here to avoid circular imports
            from lore.setting_analyzer import SettingAnalyzer
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
    
    async def get_action_modifications(self, action_type: str) -> Dict[str, Any]:
        """
        Get any modifications for a specific action type.
        
        Args:
            action_type: Type of action
            
        Returns:
            Modifications dictionary (empty if none)
        """
        return self.action_modifications.get(action_type, {})
    
    async def is_action_prohibited(self, action_type: str) -> bool:
        """
        Check if an action is prohibited.
        
        Args:
            action_type: Type of action
            
        Returns:
            True if prohibited, False otherwise
        """
        return action_type in self.prohibited_actions or "*" in self.prohibited_actions
    
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
        if await self.is_action_prohibited(action_type):
            if isinstance(response, dict):
                return {
                    "error": f"Action {action_type} is prohibited by Nyx directive",
                    "approved": False
                }
            return response
        
        # Apply modifications if any
        modifications = await self.get_action_modifications(action_type)
        if modifications and isinstance(response, dict):
            # Apply each modification
            for key, value in modifications.items():
                if key in response:
                    response[key] = value
        
        return response
