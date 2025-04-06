# nyx/governance_helpers.py
# New file with standardized governance integration helpers

import logging
import traceback
from typing import Dict, Any, Optional, Callable, Union, Awaitable
from functools import wraps

from nyx.nyx_governance import AgentType

logger = logging.getLogger(__name__)

async def check_permission(
    user_id: int,
    conversation_id: int,
    agent_type: str,
    agent_id: Union[int, str],
    action_type: str,
    action_details: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Standard function to check permissions with the governance system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        agent_type: Type of agent (use AgentType constants)
        agent_id: ID of agent instance
        action_type: Type of action being performed
        action_details: Details of the action
        context: Optional additional context
        
    Returns:
        Dictionary with permission check results
    """
    from nyx.integrate import get_central_governance
    try:
        # Get governance system
        governance = await get_central_governance(user_id, conversation_id)
        
        # Check permission
        result = await governance.check_action_permission(
            agent_type=agent_type,
            agent_id=agent_id,
            action_type=action_type,
            action_details=action_details,
            context=context
        )
        
        return result
    except Exception as e:
        logger.error(f"Error checking permission: {e}")
        logger.error(traceback.format_exc())
        
        # Default to approved in case of error to prevent system lockup
        return {
            "approved": True,
            "directive_applied": False,
            "override_action": None,
            "reasoning": f"Error checking permission: {str(e)}",
            "tracking_id": -1
        }

def with_governance_permission(
    agent_type: str,
    action_type: str,
    id_from_context: Optional[Callable] = None
):
    """
    Decorator to ensure an action has permission from governance.
    
    Args:
        agent_type: Type of agent (use AgentType constants)
        action_type: Type of action being performed
        id_from_context: Optional function to extract agent_id from context
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, ctx, *args, **kwargs):
            # Extract user_id and conversation_id from context
            user_id = ctx.context.get("user_id") if hasattr(ctx, "context") else getattr(ctx, "user_id", None)
            conversation_id = ctx.context.get("conversation_id") if hasattr(ctx, "context") else getattr(ctx, "conversation_id", None)
            
            if not user_id or not conversation_id:
                logger.warning(f"Missing user_id or conversation_id in context for {func.__name__}")
                return await func(self, ctx, *args, **kwargs)
            
            # Determine agent_id
            if id_from_context:
                agent_id = id_from_context(ctx)
            else:
                agent_id = getattr(self, "agent_id", f"{agent_type}_{conversation_id}")
            
            # Create action details from args and kwargs
            action_details = {
                "function": func.__name__,
                "args": [str(arg)[:100] for arg in args],  # Truncate for logging
                "kwargs": {k: str(v)[:100] for k, v in kwargs.items()}  # Truncate for logging
            }
            
            # Check permission
            permission = await check_permission(
                user_id,
                conversation_id,
                agent_type,
                agent_id,
                action_type,
                action_details
            )
            
            if not permission["approved"]:
                logger.warning(f"Permission denied for {func.__name__}: {permission.get('reasoning')}")
                return {
                    "error": permission.get("reasoning", "Not approved by governance"),
                    "approved": False,
                    "governance_blocked": True
                }
            
            # Apply override if present
            if permission.get("override_action"):
                override = permission["override_action"]
                # Update args and kwargs based on override
                # This is simplified; in practice would need custom logic per function
                if "args" in override and len(override["args"]) == len(args):
                    args = override["args"]
                if "kwargs" in override:
                    kwargs.update(override["kwargs"])
            
            # Call the function
            result = await func(self, ctx, *args, **kwargs)
            
            # Check for governance tracking ID
            if "tracking_id" in permission:
                # If result is a dict, attach tracking ID
                if isinstance(result, dict):
                    result["governance_tracking_id"] = permission["tracking_id"]
            
            return result
        
        return wrapper
    
    return decorator

async def report_action(
    user_id: int,
    conversation_id: int,
    agent_type: str,
    agent_id: Union[int, str],
    action: Dict[str, Any],
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Standard function to report actions to the governance system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        agent_type: Type of agent (use AgentType constants)
        agent_id: ID of agent instance
        action: Information about the action performed
        result: Result of the action
        
    Returns:
        Dictionary with reporting results
    """
    from nyx.integrate import get_central_governance
    try:
        # Get governance system
        governance = await get_central_governance(user_id, conversation_id)
        
        # Report action
        report_result = await governance.process_agent_action_report(
            agent_type=agent_type,
            agent_id=agent_id,
            action=action,
            result=result
        )
        
        return report_result
    except Exception as e:
        logger.error(f"Error reporting action: {e}")
        logger.error(traceback.format_exc())
        
        # Return basic result in case of error
        return {
            "reported": False,
            "error": str(e)
        }

def with_action_reporting(
    agent_type: str,
    action_type: str,
    action_description: str,
    id_from_context: Optional[Callable] = None,
    extract_result: Optional[Callable] = None
):
    """
    Decorator to ensure an action is reported to governance.
    
    Args:
        agent_type: Type of agent (use AgentType constants)
        action_type: Additional label or classification for the reported action
        action_description: Description of the action (may contain placeholders 
                            that are formatted by kwargs)
        id_from_context: Optional function to extract agent_id from context
        extract_result: Optional function to extract result details for reporting
        
    Returns:
        A decorator function.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, ctx, *args, **kwargs):
            from nyx.governance_helpers import report_action  # Ensure available here
            
            # Extract user_id and conversation_id from the context (depends on your usage)
            user_id = ctx.context.get("user_id") if hasattr(ctx, "context") else getattr(ctx, "user_id", None)
            conversation_id = ctx.context.get("conversation_id") if hasattr(ctx, "context") else getattr(ctx, "conversation_id", None)
            
            if not user_id or not conversation_id:
                logger.warning(f"Missing user_id or conversation_id in context for {func.__name__}")
                return await func(self, ctx, *args, **kwargs)
            
            # Determine agent_id (if not provided by a helper function)
            if id_from_context:
                agent_id = id_from_context(ctx)
            else:
                agent_id = getattr(self, "agent_id", f"{agent_type}_{conversation_id}")
            
            # Call the original function
            result = await func(self, ctx, *args, **kwargs)
            
            # Build the action dictionary for reporting
            # We merge keyword arguments (truncated to 100 chars each) into the 
            # top-level dictionary so governance sees them.
            truncated_kwargs = {k: str(v)[:100] for k, v in kwargs.items()}
            action = {
                "type": action_type,
                "description": action_description,
                **truncated_kwargs
            }
            
            # Optionally extract extra details from the function result
            result_details = {}
            if extract_result:
                result_details = extract_result(result)
            elif isinstance(result, dict):
                # Common keys to extract
                for key in ["success", "message", "count", "length"]:
                    if key in result:
                        result_details[key] = result[key]
            
            # Report this action to governance
            report_result = await report_action(
                user_id=user_id,
                conversation_id=conversation_id,
                agent_type=agent_type,
                agent_id=agent_id,
                action=action,
                result=result_details
            )
            
            # If the function returned a dict, optionally attach the reporting info
            if isinstance(result, dict):
                result["governance_reported"] = report_result.get("reported", False)
            
            return result
        
        return wrapper
    
    return decorator


def with_governance(
    agent_type: str,
    action_type: str,
    action_description: str,
    id_from_context: Optional[Callable] = None,
    extract_result: Optional[Callable] = None
):
    """
    Combined decorator for both permission checks and action reporting.
    
    Args:
        agent_type: Type of agent (use AgentType constants)
        action_type: Type of action being performed
        action_description: Description of the action
        id_from_context: Optional function to extract agent_id from context
        extract_result: Optional function to extract result details
        
    Returns:
        Decorator function
    """
    def decorator(func):
        # Apply both decorators
        permission_check = with_governance_permission(agent_type, action_type, id_from_context)
        action_report = with_action_reporting(agent_type, action_description, id_from_context, extract_result)
        
        # Apply in correct order - permission check first, then action report
        return action_report(permission_check(func))
    
    return decorator
