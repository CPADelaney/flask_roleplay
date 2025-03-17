# logic/conflict_system/conflict_integration.py
"""
Integration module for Conflict System with Nyx governance.

This module provides classes and functions to properly integrate
the conflict system with Nyx central governance.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple

from agents import function_tool, RunContextWrapper
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance

from logic.conflict_system.conflict_agents import (
    triage_agent, conflict_generation_agent, stakeholder_agent,
    manipulation_agent, resolution_agent, initialize_agents
)

logger = logging.getLogger(__name__)

class ConflictSystemIntegration:
    """
    Integration class for conflict system with Nyx governance.
    
    This class wraps the conflict system agents and provides 
    governance-compliant methods for permission checking, action reporting,
    and directive handling.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the conflict system integration."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.agents = None
        self.agent_id = "conflict_manager"  # Consistent ID for governance
        
    async def initialize(self):
        """Initialize the conflict system agents."""
        self.agents = await initialize_agents()
        return self
    
    async def check_permission(self, action_type: str, action_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if an action is permitted by Nyx governance.
        
        Args:
            action_type: Type of action being performed
            action_details: Details of the action
            
        Returns:
            Permission check result
        """
        governance = await get_central_governance(self.user_id, self.conversation_id)
        
        # Check permission with governance
        permission = await governance.check_action_permission(
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=self.agent_id,
            action_type=action_type,
            action_details=action_details
        )
        
        return permission
    
    async def report_action(self, action: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Report an action and its results to Nyx governance.
        
        Args:
            action: Information about the action performed
            result: Result of the action
            
        Returns:
            Action reporting result
        """
        governance = await get_central_governance(self.user_id, self.conversation_id)
        
        # Report action to governance
        report_result = await governance.process_agent_action_report(
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=self.agent_id,
            action=action,
            result=result
        )
        
        return report_result
    
    async def handle_directive(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a directive from Nyx governance.
        
        Args:
            directive: The directive to handle
            
        Returns:
            Directive handling result
        """
        if not self.agents:
            await self.initialize()
        
        directive_type = directive.get("type")
        directive_data = directive.get("data", {})
        
        # Route the directive to the appropriate agent based on type
        if directive_type == "generate_conflict":
            return await self.generate_conflict(directive_data)
        elif directive_type == "resolve_conflict":
            return await self.resolve_conflict(directive_data)
        elif directive_type == "update_stakeholders":
            return await self.update_stakeholders(directive_data)
        elif directive_type == "manage_manipulation":
            return await self.manage_manipulation(directive_data)
        else:
            return {
                "success": False,
                "error": f"Unknown directive type: {directive_type}"
            }
    
    @with_governance(
        agent_type=AgentType.CONFLICT_ANALYST,
        action_type="generate_conflict",
        action_description="Generate a new conflict with stakeholders"
    )
    async def generate_conflict(self, conflict_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a new conflict.
        
        Args:
            conflict_data: Data for conflict generation
            
        Returns:
            Generated conflict details
        """
        if not self.agents:
            await self.initialize()
        
        # Create a mock context
        from logic.conflict_system.conflict_agents import ConflictContext
        context = ConflictContext(self.user_id, self.conversation_id)
        
        # Use the conflict generation agent
        conflict_gen_agent = self.agents["conflict_generation_agent"]
        
        # Create a runner message structure
        from agents import Runner, RunConfig
        result = await Runner.run(
            conflict_gen_agent,
            json.dumps(conflict_data),
            context=context,
            run_config=RunConfig(
                workflow_name="Conflict Generation",
                trace_id=f"conflict-gen-{self.conversation_id}"
            )
        )
        
        # Process result
        conflict = result.final_output
        
        # Update game state via Nyx
        governance = await get_central_governance(self.user_id, self.conversation_id)
        await governance.update_game_state(
            path=f"conflicts.active.{conflict.get('conflict_id')}",
            value={
                "type": conflict.get("conflict_type"),
                "name": conflict.get("conflict_name"),
                "progress": conflict.get("progress", 0),
                "creation_time": conflict.get("creation_time")
            },
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=self.agent_id
        )
        
        return conflict
    
    @with_governance(
        agent_type=AgentType.CONFLICT_ANALYST,
        action_type="resolve_conflict",
        action_description="Resolve an existing conflict"
    )
    async def resolve_conflict(self, resolution_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve a conflict.
        
        Args:
            resolution_data: Data for conflict resolution
            
        Returns:
            Resolution results
        """
        if not self.agents:
            await self.initialize()
        
        # Create a mock context
        from logic.conflict_system.conflict_agents import ConflictContext
        context = ConflictContext(self.user_id, self.conversation_id)
        
        # Use the resolution agent
        resolution_agent = self.agents["resolution_agent"]
        
        # Create a runner message structure
        from agents import Runner, RunConfig
        result = await Runner.run(
            resolution_agent,
            json.dumps(resolution_data),
            context=context,
            run_config=RunConfig(
                workflow_name="Conflict Resolution",
                trace_id=f"conflict-res-{self.conversation_id}"
            )
        )
        
        # Process result
        resolution = result.final_output
        
        # Update game state via Nyx
        governance = await get_central_governance(self.user_id, self.conversation_id)
        
        conflict_id = resolution_data.get("conflict_id")
        await governance.update_game_state(
            path=f"conflicts.active.{conflict_id}.resolved",
            value=True,
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=self.agent_id
        )
        
        await governance.update_game_state(
            path=f"conflicts.resolved.{conflict_id}",
            value={
                "outcome": resolution.get("outcome"),
                "resolution_time": resolution.get("resolved_at"),
                "consequences": resolution.get("consequences")
            },
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=self.agent_id
        )
        
        return resolution
    
    @with_governance(
        agent_type=AgentType.CONFLICT_ANALYST,
        action_type="update_stakeholders",
        action_description="Update conflict stakeholders"
    )
    async def update_stakeholders(self, stakeholder_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update stakeholders in a conflict.
        
        Args:
            stakeholder_data: Data for stakeholder updates
            
        Returns:
            Update results
        """
        if not self.agents:
            await self.initialize()
        
        # Create a mock context
        from logic.conflict_system.conflict_agents import ConflictContext
        context = ConflictContext(self.user_id, self.conversation_id)
        
        # Use the stakeholder agent
        stakeholder_agent = self.agents["stakeholder_agent"]
        
        # Create a runner message structure
        from agents import Runner, RunConfig
        result = await Runner.run(
            stakeholder_agent,
            json.dumps(stakeholder_data),
            context=context,
            run_config=RunConfig(
                workflow_name="Stakeholder Management",
                trace_id=f"stakeholder-{self.conversation_id}"
            )
        )
        
        # Process result
        update_result = result.final_output
        
        return update_result
    
    @with_governance(
        agent_type=AgentType.CONFLICT_ANALYST,
        action_type="manage_manipulation",
        action_description="Manage NPC manipulation attempts"
    )
    async def manage_manipulation(self, manipulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage character manipulation.
        
        Args:
            manipulation_data: Data for manipulation management
            
        Returns:
            Manipulation results
        """
        if not self.agents:
            await self.initialize()
        
        # Create a mock context
        from logic.conflict_system.conflict_agents import ConflictContext
        context = ConflictContext(self.user_id, self.conversation_id)
        
        # Use the manipulation agent
        manipulation_agent = self.agents["manipulation_agent"]
        
        # Create a runner message structure
        from agents import Runner, RunConfig
        result = await Runner.run(
            manipulation_agent,
            json.dumps(manipulation_data),
            context=context,
            run_config=RunConfig(
                workflow_name="Manipulation Management",
                trace_id=f"manipulation-{self.conversation_id}"
            )
        )
        
        # Process result
        manipulation_result = result.final_output
        
        return manipulation_result
    
    async def process_request(self, request_type: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request through the conflict system with governance integration.
        
        Args:
            request_type: Type of request
            request_data: Request data
            
        Returns:
            Processing result with governance oversight
        """
        # Check permission first
        permission = await self.check_permission(request_type, request_data)
        if not permission["approved"]:
            return {
                "success": False,
                "error": permission["reasoning"],
                "governance_blocked": True
            }
        
        # Process the request through the appropriate method
        action_mapping = {
            "generate_conflict": self.generate_conflict,
            "resolve_conflict": self.resolve_conflict,
            "update_stakeholders": self.update_stakeholders,
            "manage_manipulation": self.manage_manipulation
        }
        
        if request_type in action_mapping:
            result = await action_mapping[request_type](request_data)
        else:
            # Use the triage agent for other request types
            if not self.agents:
                await self.initialize()
                
            # Create a mock context
            from logic.conflict_system.conflict_agents import ConflictContext
            context = ConflictContext(self.user_id, self.conversation_id)
            
            # Use triage agent
            from logic.conflict_system.initialize_agents import process_conflict_request
            result = await process_conflict_request(
                request_type=request_type,
                request_data=request_data,
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
        
        # Report action
        await self.report_action(
            action={
                "type": request_type,
                "data": request_data
            },
            result=result
        )
        
        return result

# Register the conflict system with governance
async def register_with_governance(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Register the conflict system with Nyx governance.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Registration result
    """
    try:
        # Get central governance
        governance = await get_central_governance(user_id, conversation_id)
        
        # Create and initialize conflict system integration
        conflict_system = ConflictSystemIntegration(user_id, conversation_id)
        await conflict_system.initialize()
        
        # Register with governance
        registration_result = await governance.register_agent(
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_instance=conflict_system,
            agent_id="conflict_manager"
        )
        
        # Issue directive for conflict analysis
        directive_result = await governance.issue_directive(
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id="conflict_manager",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Manage conflicts and their progression in the game world",
                "scope": "game"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24*60  # 24 hours
        )
        
        logger.info("Conflict System registered with Nyx governance")
        
        return {
            "success": True,
            "registration_result": registration_result,
            "directive_result": directive_result,
            "message": "Conflict System successfully registered with governance"
        }
    except Exception as e:
        logger.error(f"Error registering with governance: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to register Conflict System with governance"
        }

# Updated function tool to use the new integration
@function_tool
async def register_with_governance_tool(ctx: RunContextWrapper) -> Dict[str, Any]:
    """Register the conflict system with governance."""
    context = ctx.context
    return await register_with_governance(context.user_id, context.conversation_id)

# Enhanced integration for conflict_integration.py

import logging
from typing import Dict, List, Any, Optional

from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance

class EnhancedConflictSystemIntegration:
    """
    Enhanced integration class for conflict system with Nyx governance.
    
    This class extends the existing ConflictSystemIntegration with improved
    memory integration, temporal consistency, and user preference adaptation.
    """
    
    async def handle_directive(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a directive from Nyx governance with enhanced capabilities.
        
        Args:
            directive: The directive to handle
            
        Returns:
            Directive handling result
        """
        if not self.agents:
            await self.initialize()
        
        directive_type = directive.get("type")
        directive_data = directive.get("data", {})
        
        # Handle known directive types
        if directive_type == "generate_conflict":
            return await self.generate_conflict(directive_data)
        elif directive_type == "resolve_conflict":
            return await self.resolve_conflict(directive_data)
        elif directive_type == "update_stakeholders":
            return await self.update_stakeholders(directive_data)
        elif directive_type == "manage_manipulation":
            return await self.manage_manipulation(directive_data)
        elif directive_type == DirectiveType.ACTION:
            # Handle generic action directives from Nyx
            return await self._handle_action_directive(directive_data)
        elif directive_type == DirectiveType.SCENE:
            # Handle scene directives from Nyx
            return await self._handle_scene_directive(directive_data)
        elif directive_type == DirectiveType.PROHIBITION:
            # Handle prohibition directives from Nyx
            return await self._handle_prohibition_directive(directive_data)
        else:
            # Unknown directive type
            return {
                "success": False,
                "error": f"Unknown directive type: {directive_type}",
                "message": "Conflict system doesn't know how to handle this directive type"
            }
    
    async def _handle_action_directive(self, directive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a generic action directive from Nyx."""
        instruction = directive_data.get("instruction", "")
        scope = directive_data.get("scope", "")
        
        # Determine what action to take based on instruction
        if "analyze conflicts" in instruction.lower():
            # Find all active conflicts
            from logic.conflict_system.conflict_tools import get_active_conflicts
            conflicts = await get_active_conflicts(self.context)
            
            return {
                "success": True,
                "action": "analyze_conflicts",
                "conflicts_found": len(conflicts),
                "conflict_details": conflicts
            }
        elif "create conflict" in instruction.lower():
            # Generate a new conflict
            return await self.generate_conflict({})
        else:
            return {
                "success": False,
                "error": "Unclear action directive",
                "instruction": instruction
            }
    
    async def _handle_scene_directive(self, directive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a scene directive from Nyx."""
        # Extract location and context
        location = directive_data.get("location", "unknown")
        context = directive_data.get("context", {})
        
        # Check if we should generate a location-appropriate conflict
        if "generate_conflict" in directive_data.get("instructions", "").lower():
            # Generate a conflict appropriate for this location
            conflict_data = {
                "location": location,
                "context": context,
                "conflict_type": directive_data.get("conflict_type", "minor")
            }
            
            return await self.generate_conflict(conflict_data)
        
        # Otherwise just acknowledge the scene
        return {
            "success": True,
            "acknowledged_scene": location,
            "message": "Conflict system is aware of scene change"
        }
    
    async def _handle_prohibition_directive(self, directive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a prohibition directive from Nyx."""
        prohibited_actions = directive_data.get("prohibited_actions", [])
        reason = directive_data.get("reason", "")
        
        # Store prohibition locally
        self._prohibitions = self._prohibitions if hasattr(self, "_prohibitions") else {}
        
        for action in prohibited_actions:
            self._prohibitions[action] = reason
        
        return {
            "success": True,
            "prohibitions_added": len(prohibited_actions),
            "current_prohibitions": list(self._prohibitions.keys())
        }
    
    @with_governance(
        agent_type=AgentType.CONFLICT_ANALYST,
        action_type="generate_conflict",
        action_description="Generate a new conflict with stakeholders"
    )
    async def generate_conflict(self, conflict_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a new conflict with enhanced governance integration.
        
        Args:
            conflict_data: Data for conflict generation
            
        Returns:
            Generated conflict details
        """
        # Check temporal consistency with Nyx
        governance = await get_central_governance(self.user_id, self.conversation_id)
        temporal_check = await governance.ensure_temporal_consistency(
            proposed_action={"type": "generate_conflict", "data": conflict_data},
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=self.agent_id
        )
        
        if not temporal_check["is_consistent"]:
            return {
                "success": False,
                "error": "Temporal inconsistency detected",
                "issues": temporal_check.get("time_issues", []) + 
                         temporal_check.get("location_issues", []) + 
                         temporal_check.get("causal_issues", [])
            }
        
        # Enhance with memory context
        enhanced_data = await governance.enhance_decision_with_memories(
            "directive_issuance",
            {
                "agent_type": AgentType.CONFLICT_ANALYST,
                "agent_id": self.agent_id,
                "directive_type": "generate_conflict",
                "directive_data": conflict_data
            }
        )
        
        # Extract enhanced directive data if available
        if "directive_data" in enhanced_data:
            conflict_data = enhanced_data["directive_data"]
        
        # Apply user preferences
        conflict_data = await governance.apply_user_preferences(conflict_data)
        
        # Original implementation
        if not self.agents:
            await self.initialize()
        
        # Create context and run conflict generation
        from logic.conflict_system.conflict_agents import ConflictContext
        context = ConflictContext(self.user_id, self.conversation_id)
        
        conflict_gen_agent = self.agents["conflict_generation_agent"]
        
        from agents import Runner, RunConfig
        result = await Runner.run(
            conflict_gen_agent,
            json.dumps(conflict_data),
            context=context,
            run_config=RunConfig(
                workflow_name="Conflict Generation",
                trace_id=f"conflict-gen-{self.conversation_id}"
            )
        )
        
        conflict = result.final_output
        
        # Update game state via Nyx
        await governance.update_game_state(
            path=f"conflicts.active.{conflict.get('conflict_id')}",
            value={
                "type": conflict.get("conflict_type"),
                "name": conflict.get("conflict_name"),
                "progress": conflict.get("progress", 0),
                "creation_time": conflict.get("creation_time")
            },
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id=self.agent_id
        )
        
        # Add to memory
        memory_integration = await governance.memory_integration.initialize()
        await memory_integration.remember(
            entity_type="conflict",
            entity_id=conflict.get("conflict_id"),
            memory_text=f"Generated conflict: {conflict.get('conflict_name')} of type {conflict.get('conflict_type')}",
            importance="medium",
            tags=["conflict", "generation", "system"]
        )
        
        return conflict
    
    async def process_request(self, request_type: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request through the conflict system with enhanced governance integration.
        
        Args:
            request_type: Type of request
            request_data: Request data
            
        Returns:
            Processing result with governance oversight
        """
        # Get governance system
        governance = await get_central_governance(self.user_id, self.conversation_id)
        
        # Check permission first
        permission = await self.check_permission(request_type, request_data)
        if not permission["approved"]:
            return {
                "success": False,
                "error": permission["reasoning"],
                "governance_blocked": True
            }
        
        # Enhance with memory context
        enhanced_data = await governance.enhance_decision_with_memories(
            "action_processing",
            {
                "agent_type": AgentType.CONFLICT_ANALYST,
                "agent_id": self.agent_id,
                "request_type": request_type,
                "request_data": request_data
            }
        )
        
        # Extract enhanced request data if available
        if "request_data" in enhanced_data:
            request_data = enhanced_data["request_data"]
        
        # Apply user preferences
        request_data = await governance.apply_user_preferences(request_data)
        
        # Process the request through the appropriate method
        action_mapping = {
            "generate_conflict": self.generate_conflict,
            "resolve_conflict": self.resolve_conflict,
            "update_stakeholders": self.update_stakeholders,
            "manage_manipulation": self.manage_manipulation
        }
        
        if request_type in action_mapping:
            result = await action_mapping[request_type](request_data)
        else:
            # Use the triage agent for other request types
            if not self.agents:
                await self.initialize()
                
            # Create a context
            from logic.conflict_system.conflict_agents import ConflictContext
            context = ConflictContext(self.user_id, self.conversation_id)
            
            # Use triage agent
            from logic.conflict_system.initialize_agents import process_conflict_request
            result = await process_conflict_request(
                request_type=request_type,
                request_data=request_data,
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
        
        # Report action
        action_report = await self.report_action(
            action={
                "type": request_type,
                "data": request_data
            },
            result=result
        )
        
        # If we got feedback from governance, include it
        if action_report and "feedback" in action_report:
            result["governance_feedback"] = action_report["feedback"]
        
        return result

# Use this function to register the enhanced integration
async def register_enhanced_integration(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Register the enhanced conflict system with Nyx governance.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Registration result
    """
    try:
        # Get central governance
        governance = await get_central_governance(user_id, conversation_id)
        
        # Import the original integration class and extend it with our enhancements
        from logic.conflict_system.conflict_integration import ConflictSystemIntegration
        
        # Create a new class that combines both
        class NyxEnhancedConflictSystem(ConflictSystemIntegration, EnhancedConflictSystemIntegration):
            """Combined class with original and enhanced functionality."""
            pass
        
        # Create and initialize conflict system integration
        conflict_system = NyxEnhancedConflictSystem(user_id, conversation_id)
        await conflict_system.initialize()
        
        # Register with governance
        registration_result = await governance.register_agent(
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_instance=conflict_system,
            agent_id="conflict_manager"
        )
        
        # Issue directive for conflict analysis
        directive_result = await governance.issue_directive(
            agent_type=AgentType.CONFLICT_ANALYST,
            agent_id="conflict_manager",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Manage conflicts and their progression in the game world",
                "scope": "game"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24*60  # 24 hours
        )
        
        # Subscribe to game state changes
        if hasattr(governance, "game_state"):
            # Monitor location changes to potentially trigger location-specific conflicts
            await governance.game_state.register_change_listener(
                "environment.current_location",
                conflict_system._handle_location_change
            )
            
            # Monitor time changes to potentially advance conflicts
            await governance.game_state.register_change_listener(
                "environment.time_of_day",
                conflict_system._handle_time_change
            )
        
        logging.info("Enhanced Conflict System registered with Nyx governance")
        
        return {
            "success": True,
            "registration_result": registration_result,
            "directive_result": directive_result,
            "message": "Enhanced Conflict System successfully registered with governance",
            "enhanced_features": [
                "Memory integration",
                "Temporal consistency",
                "User preference adaptation",
                "Game state monitoring",
                "Expanded directive handling"
            ]
        }
    except Exception as e:
        logging.error(f"Error registering with governance: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to register Enhanced Conflict System with governance"
        }
