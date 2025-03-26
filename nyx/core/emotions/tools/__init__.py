# nyx/core/emotions/tools/__init__.py

"""
Tools for the Nyx Emotional Core system with enhanced OpenAI Agents SDK integration.

This package contains function tools that implement the various capabilities
of the emotional system, organized by their domain:
- neurochemical_tools: Tools for managing neurochemical state
- emotion_tools: Tools for emotion derivation and analysis
- reflection_tools: Tools for reflection and learning
- learning_tools: Tools for adaptive learning
"""

from typing import Callable, Dict, List, Optional, Any
from agents import function_tool, Tool

from nyx.core.emotions.tools.neurochemical_tools import NeurochemicalTools
from nyx.core.emotions.tools.emotion_tools import EmotionTools
from nyx.core.emotions.tools.reflection_tools import ReflectionTools
from nyx.core.emotions.tools.learning_tools import LearningTools

# Registry to hold all available tools
_TOOL_REGISTRY: Dict[str, Tool] = {}

def register_tool(name: str, tool_func: Callable, 
                 name_override: Optional[str] = None, 
                 description_override: Optional[str] = None) -> Tool:
    """
    Register a tool in the central registry for better SDK integration
    
    Args:
        name: Tool identifier
        tool_func: Function to wrap as a tool
        name_override: Optional override for the tool name
        description_override: Optional override for the tool description
        
    Returns:
        The created tool
    """
    # Create the tool
    tool = function_tool(
        tool_func, 
        name_override=name_override,
        description_override=description_override
    )
    
    # Register the tool
    _TOOL_REGISTRY[name] = tool
    
    return tool

def get_tools_for_agent(agent_type: str) -> List[Tool]:
    """
    Get tools for a specific agent type
    
    Args:
        agent_type: The type of agent to get tools for
        
    Returns:
        List of tools for the specified agent
    """
    agent_tool_map = {
        "neurochemical": [
            "update_neurochemical", 
            "apply_chemical_decay",
            "process_chemical_interactions",
            "get_neurochemical_state"
        ],
        "emotion_derivation": [
            "get_neurochemical_state",
            "derive_emotional_state",
            "get_emotional_state_matrix"
        ],
        "reflection": [
            "get_emotional_state_matrix",
            "generate_internal_thought",
            "analyze_emotional_patterns"
        ],
        "learning": [
            "record_interaction_outcome",
            "update_learning_rules",
            "apply_learned_adaptations"
        ],
        "orchestrator": [
            "analyze_text_sentiment"
        ]
    }
    
    # Get tool list for agent type, defaulting to empty list if not found
    tool_names = agent_tool_map.get(agent_type, [])
    
    # Return only registered tools
    return [_TOOL_REGISTRY[name] for name in tool_names if name in _TOOL_REGISTRY]

# Update exports
__all__ = [
    'NeurochemicalTools', 
    'EmotionTools', 
    'ReflectionTools', 
    'LearningTools',
    'register_tool',
    'get_tools_for_agent'
]
