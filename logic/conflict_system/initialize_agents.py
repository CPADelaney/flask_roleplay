# logic/conflict_system/initialize_agents.py
"""
Conflict System Agent Initialization

This module initializes and configures the conflict system agents using the OpenAI Agents SDK.
It serves as the primary entry point for integrating the agent-based architecture with the
existing conflict system.
"""

import logging
import asyncio
from typing import Dict, Any, Optional

from agents import Agent, Runner, trace
from logic.conflict_system.conflict_agents import (
    triage_agent, conflict_generation_agent, stakeholder_agent,
    manipulation_agent, resolution_agent, ConflictContext
)
from logic.conflict_system.conflict_guardrails import apply_guardrails
from logic.conflict_system.conflict_tools import register_with_governance

logger = logging.getLogger(__name__)

async def initialize_conflict_system(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Initialize the entire conflict system with agents and register with governance.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Dictionary containing initialized agents and status
    """
    try:
        # Create context
        context = ConflictContext(user_id, conversation_id)
        
        # Set up handoffs for triage agent
        triage_agent.handoffs = [
            conflict_generation_agent,
            stakeholder_agent,
            manipulation_agent,
            resolution_agent
        ]
        
        # Apply guardrails to all agents
        agents_dict = {
            "triage_agent": triage_agent,
            "conflict_generation_agent": conflict_generation_agent,
            "stakeholder_agent": stakeholder_agent,
            "manipulation_agent": manipulation_agent,
            "resolution_agent": resolution_agent
        }
        agents_dict = apply_guardrails(agents_dict)
        
        # Register with governance system
        governance_result = await register_with_governance(context, user_id, conversation_id)
        
        logger.info(f"Conflict system initialized for user {user_id}, conversation {conversation_id}")
        
        return {
            "agents": agents_dict,
            "context": context,
            "governance_registration": governance_result,
            "status": "initialized"
        }
    except Exception as e:
        logger.error(f"Error initializing conflict system: {e}", exc_info=True)
        return {
            "error": str(e),
            "status": "failed"
        }

async def process_conflict_request(
    request_type: str,
    request_data: Dict[str, Any],
    user_id: int,
    conversation_id: int
) -> Dict[str, Any]:
    """
    Process a conflict system request using the agent architecture.
    
    Args:
        request_type: Type of request (generate, resolve, etc.)
        request_data: Data for the request
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Result of the request processing
    """
    # Create context for this request
    context = ConflictContext(user_id, conversation_id)
    
    # Create a trace for this workflow
    with trace(workflow_name=f"Conflict_{request_type}", group_id=str(conversation_id)):
        try:
            # Format the request for the triage agent
            formatted_request = f"Request type: {request_type}\nRequest data: {request_data}"
            
            # Run the request through the triage agent
            result = await Runner.run(triage_agent, formatted_request, context=context)
            
            return {
                "success": True,
                "result": result.final_output,
                "agent": result.last_agent.name,
                "status": "completed"
            }
        except Exception as e:
            logger.error(f"Error processing conflict request: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "status": "failed"
            }

async def direct_agent_request(
    agent_name: str,
    input_message: str,
    user_id: int,
    conversation_id: int
) -> Dict[str, Any]:
    """
    Make a direct request to a specific agent, bypassing triage.
    
    Args:
        agent_name: Name of the agent to call
        input_message: Input for the agent
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Result from the agent
    """
    # Create context for this request
    context = ConflictContext(user_id, conversation_id)
    
    # Get the agent dictionary
    agents_dict = {
        "triage_agent": triage_agent,
        "conflict_generation_agent": conflict_generation_agent,
        "stakeholder_agent": stakeholder_agent,
        "manipulation_agent": manipulation_agent,
        "resolution_agent": resolution_agent
    }
    
    # Get the requested agent
    if agent_name not in agents_dict:
        return {
            "success": False,
            "error": f"Agent '{agent_name}' not found",
            "status": "failed"
        }
    
    agent = agents_dict[agent_name]
    
    # Create a trace for this direct request
    with trace(workflow_name=f"Direct_{agent_name}", group_id=str(conversation_id)):
        try:
            # Run the request on the specific agent
            result = await Runner.run(agent, input_message, context=context)
            
            return {
                "success": True,
                "result": result.final_output,
                "agent": agent_name,
                "status": "completed"
            }
        except Exception as e:
            logger.error(f"Error processing direct agent request: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "status": "failed"
            }
