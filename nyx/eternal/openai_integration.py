# nyx/eternal/openai_integration.py

from typing import Dict, Any, Optional, Callable, Awaitable
import os
import logging

from .openai_api import OpenAIAgentsAPI

logger = logging.getLogger(__name__)

# Global API instance
_api_instance = None

def initialize(api_key: Optional[str] = None,
              original_processor: Optional[Callable] = None):
    """
    Initialize the OpenAI Agents integration.
    
    Args:
        api_key: OpenAI API key
        original_processor: Function that processes with original Nyx system
    """
    global _api_instance
    
    if _api_instance is None:
        _api_instance = OpenAIAgentsAPI(
            api_key=api_key,
            original_processor=original_processor
        )
        
    logger.info("OpenAI Agents integration initialized")

def get_api() -> OpenAIAgentsAPI:
    """Get API instance"""
    global _api_instance
    
    if _api_instance is None:
        raise RuntimeError("OpenAI Agents integration not initialized")
    
    return _api_instance

async def process_with_enhancement(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process with enhancement from OpenAI Agents.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        user_input: User's input message
        context: Context information
        
    Returns:
        Processing result
    """
    api = get_api()
    return await api.process_with_enhancement(
        user_id, conversation_id, user_input, context
    )

async def process_standalone(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process with standalone OpenAI Agents implementation.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        user_input: User's input message
        context: Context information
        
    Returns:
        Processing result
    """
    api = get_api()
    return await api.process_standalone(
        user_id, conversation_id, user_input, context
    )
