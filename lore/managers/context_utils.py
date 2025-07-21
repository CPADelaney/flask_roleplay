# lore/managers/context_utils.py

from agents import RunContextWrapper
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

def make_run_context(
    user_id: int, 
    conversation_id: int, 
    manager: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None
) -> RunContextWrapper:
    """
    Factory function to create a properly initialized RunContextWrapper.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        manager: Optional manager instance
        extra: Optional extra context fields
        
    Returns:
        RunContextWrapper with required fields populated
    """
    base = {
        "user_id": user_id, 
        "conversation_id": conversation_id
    }
    
    if manager is not None:
        base["manager"] = manager
        
    if extra:
        base.update(extra)
        
    return RunContextWrapper(context=base)

def ensure_context_fields(ctx: RunContextWrapper, required_fields: list) -> bool:
    """
    Validate that a context has all required fields.
    
    Args:
        ctx: RunContextWrapper to validate
        required_fields: List of required field names
        
    Returns:
        True if all fields present, False otherwise
    """
    if not hasattr(ctx, 'context') or not isinstance(ctx.context, dict):
        logger.error("Invalid RunContextWrapper: missing or invalid context")
        return False
        
    missing = [field for field in required_fields if field not in ctx.context]
    if missing:
        logger.error(f"Context missing required fields: {missing}")
        return False
        
    return True
