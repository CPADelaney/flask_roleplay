# nyx/governance_helpers.py
# Updated governance integration helpers with canon system integration

import logging
import traceback
from typing import Dict, Any, Optional, Callable, Union, Awaitable
from functools import wraps
import inspect
from datetime import datetime
import json

from nyx.governance import AgentType
# Remove circular import - will import canon lazily when needed

logger = logging.getLogger(__name__)

def ensure_canonical_context(ctx):
    """
    Ensure we have a proper CanonicalContext for canon operations.
    
    Args:
        ctx: Any context-like object
        
    Returns:
        CanonicalContext instance
    """
    # Lazy import to avoid circular dependency
    from lore.core.context import CanonicalContext
    
    if isinstance(ctx, CanonicalContext):
        return ctx
    elif isinstance(ctx, dict):
        return CanonicalContext.from_dict(ctx)
    else:
        return CanonicalContext.from_object(ctx)

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
    
    Returns a standardized response that includes fields expected by decorators.
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
        
        # Standardize the response for decorators
        standardized = {
            "approved": result.get("approved", True),
            "reasoning": result.get("reasoning", ""),
            "prohibition_id": result.get("prohibition_id"),
            # Add expected fields with defaults
            "directive_applied": False,
            "override_action": None,
            "tracking_id": result.get("prohibition_id", -1)  # Use prohibition_id as tracking
        }
        
        # Check if there are active directives that might override
        if governance and hasattr(governance, 'get_agent_directives'):
            directives = await governance.get_agent_directives(agent_type, agent_id)
            for directive in directives:
                if directive.get("type") == "override" and directive.get("status") == "active":
                    standardized["directive_applied"] = True
                    standardized["override_action"] = directive.get("data", {})
                    standardized["tracking_id"] = directive.get("id", -1)
                    break
        
        return standardized
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
    
    Returns a standardized response for consistency.
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
        
        # Standardize response - map 'success' to 'reported'
        return {
            "reported": report_result.get("success", False),
            "report_id": report_result.get("report_id"),
            "error": report_result.get("error")
        }
    except Exception as e:
        logger.error(f"Error reporting action: {e}")
        logger.error(traceback.format_exc())
        
        # Return basic result in case of error
        return {
            "reported": False,
            "error": str(e),
            "report_id": None
        }

def with_governance_permission(
    agent_type: str,
    action_type: str,
    id_from_context: Optional[Callable] = None
):
    """
    Decorator to ensure an action has permission from governance.
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
                if "args" in override and len(override["args"]) == len(args):
                    args = override["args"]
                if "kwargs" in override:
                    kwargs.update(override["kwargs"])
            
            # Call the function
            result = await func(self, ctx, *args, **kwargs)
            
            # Check for governance tracking ID
            if "tracking_id" in permission and permission["tracking_id"] != -1:
                # If result is a dict, attach tracking ID
                if isinstance(result, dict):
                    result["governance_tracking_id"] = permission["tracking_id"]
            
            return result
        
        return wrapper
    
    return decorator

# Canon-specific helpers

def with_canon_tracking(
    entity_type: str,
    significance: int = 5,
    extract_entity_name: Optional[Callable] = None
):
    """
    Decorator that ensures canon operations are properly tracked.
    
    Args:
        entity_type: Type of entity being modified
        significance: Significance level for canonical events (1-10)
        extract_entity_name: Optional function to extract entity name from args/kwargs
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract context
            ctx = None
            if len(args) > 0 and hasattr(args[0], 'context'):
                ctx = args[0]
            elif 'ctx' in kwargs:
                ctx = kwargs['ctx']
            
            if ctx:
                ctx = ensure_canonical_context(ctx)
            
            # Call the original function
            result = await func(*args, **kwargs)
            
            # Log canonical event if successful
            if isinstance(result, dict) and result.get('status') in ['success', 'committed']:
                if ctx:
                    from db.connection import get_db_connection_context
                    
                    # Extract entity name
                    entity_name = 'unknown'
                    if extract_entity_name:
                        entity_name = extract_entity_name(*args, **kwargs)
                    elif 'entity_name' in kwargs:
                        entity_name = kwargs['entity_name']
                    elif 'name' in kwargs:
                        entity_name = kwargs['name']
                    
                    # Create event description
                    event_text = f"{entity_type} '{entity_name}' modified via {func.__name__}"
                    if 'reason' in kwargs:
                        event_text += f": {kwargs['reason']}"
                    
                    # Log the event
                    try:
                        async with get_db_connection_context() as conn:
                            # Lazy import to avoid circular dependency
                            from lore.core import canon
                            await canon.log_canonical_event(
                                ctx, conn,
                                event_text,
                                tags=[entity_type.lower(), 'governance', func.__name__],
                                significance=significance
                            )
                    except Exception as e:
                        logger.error(f"Failed to log canonical event: {e}")
            
            return result
        
        return wrapper
    
    return decorator

def with_conflict_resolution(
    resolve_strategy: str = "lore_evolution"
):
    """
    Decorator that handles conflicts detected by the canon system.
    
    Args:
        resolve_strategy: How to resolve conflicts ('lore_evolution', 'override', 'abort')
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Check if we got a conflict
            if isinstance(result, dict) and result.get('status') == 'conflict_generated':
                logger.info(f"Conflict detected in {func.__name__}: {result.get('details', [])}")
                
                if resolve_strategy == "lore_evolution":
                    # The canon system already handled this via lore evolution
                    result['conflict_resolved'] = True
                    result['resolution_method'] = 'lore_evolution'
                elif resolve_strategy == "override":
                    # Try again with force flag if supported
                    if 'force' in inspect.signature(func).parameters:
                        kwargs['force'] = True
                        result = await func(*args, **kwargs)
                        result['conflict_resolved'] = True
                        result['resolution_method'] = 'override'
                else:
                    # Abort - just return the conflict
                    result['conflict_resolved'] = False
                    result['resolution_method'] = 'aborted'
            
            return result
        
        return wrapper
    
    return decorator

def with_canon_governance(
    agent_type: str,
    action_type: str,
    action_description: str,
    entity_type: str,
    significance: int = 5,
    id_from_context: Optional[Callable] = None,
    extract_result: Optional[Callable] = None,
    extract_entity_name: Optional[Callable] = None
):
    """
    Combined decorator for governance + canon tracking + conflict resolution.
    This is the recommended decorator for canon operations.
    """
    def decorator(func):
        # Apply decorators in order
        func = with_conflict_resolution()(func)
        func = with_canon_tracking(
            entity_type=entity_type,
            significance=significance,
            extract_entity_name=extract_entity_name
        )(func)
        func = with_governance(
            agent_type=agent_type,
            action_type=action_type,
            action_description=action_description,
            id_from_context=id_from_context,
            extract_result=extract_result
        )(func)
        
        return func
    
    return decorator

async def propose_canonical_change(
    ctx,
    entity_type: str,
    entity_identifier: Dict[str, Any],
    updates: Dict[str, Any],
    reason: str,
    agent_type: str = AgentType.NARRATIVE_CRAFTER,
    agent_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Helper function to propose changes through both governance and canon systems.
    
    This integrates with LoreSystem's propose_and_enact_change method.
    
    Args:
        ctx: Context (will be converted to CanonicalContext)
        entity_type: Type of entity to update
        entity_identifier: How to find the entity
        updates: Changes to make
        reason: Reason for the change
        agent_type: Agent type making the change
        agent_id: Agent ID (optional)
        
    Returns:
        Result dictionary with status
    """
    # Ensure canonical context
    ctx = ensure_canonical_context(ctx)
    
    # Get governance permission first
    permission = await check_permission(
        user_id=ctx.user_id,
        conversation_id=ctx.conversation_id,
        agent_type=agent_type,
        agent_id=agent_id or f"{agent_type}_canon",
        action_type="propose_canonical_change",
        action_details={
            "entity_type": entity_type,
            "entity_identifier": entity_identifier,
            "updates": updates,
            "reason": reason
        }
    )
    
    if not permission["approved"]:
        return {
            "status": "error",
            "message": f"Governance denied: {permission.get('reasoning', 'Unknown reason')}",
            "governance_blocked": True
        }
    
    # Get the LoreSystem and propose the change
    from lore.core.lore_system import LoreSystem
    lore_system = await LoreSystem.get_instance(ctx.user_id, ctx.conversation_id)
    
    # Apply any governance overrides
    if permission.get("override_action"):
        override = permission["override_action"]
        if "updates" in override:
            updates.update(override["updates"])
        if "reason" in override:
            reason = override["reason"]
    
    # Propose the change through the canon system
    result = await lore_system.propose_and_enact_change(
        ctx=ctx,
        entity_type=entity_type,
        entity_identifier=entity_identifier,
        updates=updates,
        reason=reason
    )
    
    # Report the action
    await report_action(
        user_id=ctx.user_id,
        conversation_id=ctx.conversation_id,
        agent_type=agent_type,
        agent_id=agent_id or f"{agent_type}_canon",
        action={
            "type": "canonical_change",
            "entity_type": entity_type,
            "updates": updates,
            "reason": reason
        },
        result=result
    )
    
    return result

async def create_canonical_entity(
    ctx,
    entity_type: str,
    entity_name: str,
    entity_data: Dict[str, Any],
    agent_type: str = AgentType.NARRATIVE_CRAFTER,
    agent_id: Optional[str] = None,
    **kwargs
) -> Union[int, str, Dict[str, Any]]:
    """
    Helper to create entities through the canon system with governance.
    
    This uses the appropriate find_or_create_* function from canon.
    
    Args:
        ctx: Context
        entity_type: Type of entity ('npc', 'location', 'faction', etc.)
        entity_name: Name of the entity
        entity_data: Additional data for the entity
        agent_type: Agent type creating the entity
        agent_id: Agent ID
        **kwargs: Additional arguments for the canon function
        
    Returns:
        Entity ID or result dictionary
    """
    from db.connection import get_db_connection_context
    
    # Ensure canonical context
    ctx = ensure_canonical_context(ctx)
    
    # Check governance permission
    permission = await check_permission(
        user_id=ctx.user_id,
        conversation_id=ctx.conversation_id,
        agent_type=agent_type,
        agent_id=agent_id or f"{agent_type}_canon",
        action_type=f"create_{entity_type}",
        action_details={
            "entity_type": entity_type,
            "entity_name": entity_name,
            "entity_data": entity_data
        }
    )
    
    if not permission["approved"]:
        return {
            "status": "error",
            "message": f"Governance denied: {permission.get('reasoning', 'Unknown reason')}",
            "governance_blocked": True
        }
    
    # Apply any governance overrides
    if permission.get("override_action"):
        override = permission["override_action"]
        if "entity_data" in override:
            entity_data.update(override["entity_data"])
        if "entity_name" in override:
            entity_name = override["entity_name"]
    
    # Route to appropriate canon function
    async with get_db_connection_context() as conn:
        # Lazy import to avoid circular dependency
        from lore.core import canon
        
        result = None
        
        if entity_type == "npc":
            result = await canon.find_or_create_npc(
                ctx, conn, entity_name, **entity_data, **kwargs
            )
        elif entity_type == "location":
            result = await canon.find_or_create_location(
                ctx, conn, entity_name, **entity_data, **kwargs
            )
        elif entity_type == "faction":
            result = await canon.find_or_create_faction(
                ctx, conn, entity_name, **entity_data, **kwargs
            )
        elif entity_type == "event":
            result = await canon.find_or_create_event(
                ctx, conn, entity_name, **entity_data, **kwargs
            )
        elif entity_type == "quest":
            result = await canon.find_or_create_quest(
                ctx, conn, entity_name, **entity_data, **kwargs
            )
        elif entity_type == "historical_event":
            result = await canon.find_or_create_historical_event(
                ctx, conn, entity_name, **entity_data, **kwargs
            )
        elif entity_type == "notable_figure":
            result = await canon.find_or_create_notable_figure(
                ctx, conn, entity_name, **entity_data, **kwargs
            )
        elif entity_type == "nation":
            result = await canon.find_or_create_nation(
                ctx, conn, entity_name, **entity_data, **kwargs
            )
        elif entity_type == "conflict":
            result = await canon.find_or_create_conflict(
                ctx, conn, entity_data.get('conflict_name', entity_name),
                entity_data.get('involved_nations', []),
                entity_data.get('conflict_type', 'political'),
                **kwargs
            )
        else:
            # Generic entity creation
            result = await canon.find_or_create_entity(
                ctx, conn,
                entity_type=entity_type,
                entity_name=entity_name,
                search_fields={"name": entity_name},
                create_data=entity_data,
                table_name=entity_type.title(),  # Assumes table name matches entity type
                embedding_text=f"{entity_name} {json.dumps(entity_data)}",
                **kwargs
            )
    
    # Report the action
    await report_action(
        user_id=ctx.user_id,
        conversation_id=ctx.conversation_id,
        agent_type=agent_type,
        agent_id=agent_id or f"{agent_type}_canon",
        action={
            "type": f"create_{entity_type}",
            "entity_name": entity_name,
            "entity_data": entity_data,
            "result": result
        },
        result={"entity_id": result, "created": True}
    )
    
    return result

# Example usage patterns
"""
# For a function that modifies canon entities:
@with_canon_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="update_npc_relationship",
    action_description="Updating relationship between {npc1_name} and {npc2_name}",
    entity_type="relationship",
    significance=6,
    extract_entity_name=lambda *args, **kwargs: f"{kwargs.get('npc1_name', 'NPC1')}-{kwargs.get('npc2_name', 'NPC2')}"
)
async def update_npc_relationship(ctx, npc1_name: str, npc2_name: str, relationship_type: str, reason: str):
    # Your implementation
    pass

# For creating new canonical entities:
async def create_new_faction(ctx, faction_name: str, faction_type: str, description: str):
    return await create_canonical_entity(
        ctx=ctx,
        entity_type="faction",
        entity_name=faction_name,
        entity_data={
            "type": faction_type,
            "description": description,
            "power_level": 5,
            "influence_scope": "local"
        },
        agent_type=AgentType.NARRATIVE_CRAFTER,
        agent_id="faction_creator"
    )

# For proposing complex changes:
async def execute_political_coup(ctx, nation_id: int, new_leader_id: int):
    return await propose_canonical_change(
        ctx=ctx,
        entity_type="Nations",
        entity_identifier={"id": nation_id},
        updates={"leader_npc_id": new_leader_id},
        reason="Political coup orchestrated by opposition forces",
        agent_type=AgentType.CONFLICT_ANALYST,
        agent_id="political_system"
    )
"""

def with_action_reporting(
    agent_type: str,
    action_type: str,
    action_description: str = "", 
    id_from_context: Optional[Callable] = None,
    extract_result: Optional[Callable] = None
):
    """
    Decorator to ensure an action is reported to governance.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, ctx, *args, **kwargs):
            # Extract user_id and conversation_id from the context
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
            
            # Call the original function
            result = await func(self, ctx, *args, **kwargs)
            
            # Build the action dictionary for reporting
            truncated_kwargs = {k: str(v)[:100] for k, v in kwargs.items()}
            action = {
                "type": action_type,
                "description": action_description,
                **truncated_kwargs
            }
            
            # Extract result details
            result_details = {}
            if extract_result:
                result_details = extract_result(result)
            elif isinstance(result, dict):
                # Common keys to extract
                for key in ["success", "message", "count", "length", "status", "committed"]:
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
            
            # If the function returned a dict, attach the reporting info
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
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get function signature to properly handle arguments
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Determine if this is a method or function by checking first param name
            is_method = len(params) > 0 and params[0] == 'self'
            
            # Extract arguments based on function type
            if is_method:
                # Method call: (self, ctx, ...) or (self, ..., ctx=ctx)
                if len(args) >= 2:
                    # ctx passed as positional
                    self_arg = args[0]
                    ctx = args[1]
                    remaining_args = args[2:]
                elif len(args) >= 1 and 'ctx' in kwargs:
                    # ctx passed as keyword
                    self_arg = args[0]
                    ctx = kwargs.pop('ctx')  # Remove ctx from kwargs to avoid duplicate
                    remaining_args = args[1:]
                else:
                    raise ValueError(f"Method {func.__name__} requires self and ctx arguments")
            else:
                # Function call: (ctx, ...) or (..., ctx=ctx)
                if len(args) >= 1:
                    # ctx passed as positional
                    self_arg = None
                    ctx = args[0]
                    remaining_args = args[1:]
                elif 'ctx' in kwargs:
                    # ctx passed as keyword
                    self_arg = None
                    ctx = kwargs.pop('ctx')  # Remove ctx from kwargs to avoid duplicate
                    remaining_args = args
                else:
                    raise ValueError(f"Function {func.__name__} requires ctx argument")
            
            # Extract user_id and conversation_id with multiple fallback strategies
            user_id = None
            conversation_id = None
            
            # Strategy 1: Check if ctx has a context dict
            if hasattr(ctx, "context") and isinstance(ctx.context, dict):
                user_id = ctx.context.get("user_id")
                conversation_id = ctx.context.get("conversation_id")
            
            # Strategy 2: Check if ctx has direct attributes
            if user_id is None:
                user_id = getattr(ctx, "user_id", None)
            if conversation_id is None:
                conversation_id = getattr(ctx, "conversation_id", None)
            
            # Strategy 3: Check if ctx is dict-like
            if user_id is None and hasattr(ctx, "get"):
                user_id = ctx.get("user_id")
            if conversation_id is None and hasattr(ctx, "get"):
                conversation_id = ctx.get("conversation_id")
            
            # Strategy 4: Check if ctx is dict-like with __getitem__
            if user_id is None and hasattr(ctx, "__getitem__"):
                try:
                    user_id = ctx["user_id"]
                except (KeyError, TypeError):
                    pass
            if conversation_id is None and hasattr(ctx, "__getitem__"):
                try:
                    conversation_id = ctx["conversation_id"]
                except (KeyError, TypeError):
                    pass
            
            # Final validation
            if user_id is None or conversation_id is None:
                logger.warning(
                    f"Missing user_id or conversation_id in context for {func.__name__}. "
                    f"user_id={user_id}, conversation_id={conversation_id}, "
                    f"ctx type={type(ctx)}, ctx attrs={dir(ctx) if hasattr(ctx, '__dir__') else 'unknown'}"
                )
                # For system-level operations, use defaults
                if user_id is None:
                    user_id = 0
                if conversation_id is None:
                    conversation_id = 0
            
            # Determine agent_id
            if id_from_context:
                agent_id = id_from_context(ctx)
            elif is_method and self_arg:
                agent_id = getattr(self_arg, "agent_id", f"{agent_type}_{conversation_id}")
            else:
                agent_id = f"{agent_type}_{conversation_id}"
            
            # ===== PERMISSION CHECK =====
            # Create action details from args and kwargs
            action_details = {
                "function": func.__name__,
                "args": [str(arg)[:100] for arg in remaining_args],  # Truncate for logging
                "kwargs": {k: str(v)[:100] for k, v in kwargs.items()},  # Truncate for logging
                "description": action_description.format(**kwargs) if kwargs else action_description
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
                if "args" in override and len(override["args"]) == len(remaining_args):
                    remaining_args = override["args"]
                if "kwargs" in override:
                    kwargs.update(override["kwargs"])
            
            # ===== EXECUTE FUNCTION =====
            start_time = datetime.now()
            error_occurred = None
            
            try:
                # Call the original function with ctx in the correct position
                if is_method:
                    result = await func(self_arg, ctx, *remaining_args, **kwargs)
                else:
                    result = await func(ctx, *remaining_args, **kwargs)
            except Exception as e:
                error_occurred = e
                result = {
                    "error": str(e),
                    "success": False,
                    "exception_type": type(e).__name__
                }
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # ===== ACTION REPORTING =====
            # Build the action dictionary for reporting
            truncated_kwargs = {k: str(v)[:100] for k, v in kwargs.items()}
            action = {
                "type": action_type,
                "description": action_description.format(**kwargs) if kwargs else action_description,
                "function": func.__name__,
                "duration_seconds": duration,
                "permission_tracking_id": permission.get("tracking_id", -1),
                **truncated_kwargs
            }
            
            # Extract result details
            result_details = {
                "duration_seconds": duration,
                "error_occurred": error_occurred is not None
            }
            
            if extract_result and not error_occurred:
                try:
                    extracted = extract_result(result)
                    if isinstance(extracted, dict):
                        result_details.update(extracted)
                except Exception as e:
                    logger.warning(f"Error extracting result details: {e}")
            elif isinstance(result, dict):
                # Common keys to extract
                for key in ["success", "message", "count", "length", "status", "committed"]:
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
            
            # Attach governance metadata to result if it's a dict
            if isinstance(result, dict):
                result["governance_metadata"] = {
                    "permission_approved": True,
                    "permission_tracking_id": permission.get("tracking_id", -1),
                    "directive_applied": permission.get("directive_applied", False),
                    "action_reported": report_result.get("reported", False),
                    "report_id": report_result.get("report_id", None)
                }
            
            # Re-raise the error if one occurred
            if error_occurred:
                raise error_occurred
            
            return result
        
        return wrapper
    
    return decorator
