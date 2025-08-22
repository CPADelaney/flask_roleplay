# logic/inventory_system_sdk.py
"""
Optimized Inventory management system using OpenAI's Agents SDK with Nyx Governance integration.
Includes caching, deduplication, and direct database access for improved performance.
"""

import logging
import json
import asyncio
import traceback
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict

# OpenAI Agents SDK imports
from agents import (
    Agent,
    ModelSettings,
    Runner,
    function_tool,
    RunContextWrapper,
    GuardrailFunctionOutput,
    InputGuardrail,
    trace,
    handoff
)
from pydantic import BaseModel, Field

# DB connection
from db.connection import get_db_connection_context
import asyncpg
from lore.core import canon

# Nyx governance integration
from nyx.nyx_governance import (
    NyxUnifiedGovernor,
    AgentType,
    DirectiveType,
    DirectivePriority
)
from nyx.integrate import get_central_governance

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------
# Caching Layer
# -------------------------------------------------------------------------------

class InventoryCache:
    """Simple cache for inventory queries to prevent redundant database calls"""
    def __init__(self, ttl_seconds: int = 60):
        self.cache = {}
        self.ttl = ttl_seconds
        
    def get_key(self, user_id: int, conversation_id: int, player_name: str) -> str:
        return f"{user_id}:{conversation_id}:{player_name}"
    
    def get(self, user_id: int, conversation_id: int, player_name: str) -> Optional[Dict]:
        key = self.get_key(user_id, conversation_id, player_name)
        if key in self.cache:
            entry, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                logger.debug(f"Cache hit for {key}")
                return entry
            else:
                del self.cache[key]  # Remove expired entry
        return None
    
    def set(self, user_id: int, conversation_id: int, player_name: str, data: Dict):
        key = self.get_key(user_id, conversation_id, player_name)
        self.cache[key] = (data, time.time())
        logger.debug(f"Cached inventory for {key}")
    
    def invalidate(self, user_id: int, conversation_id: int, player_name: str):
        """Invalidate cache when inventory changes"""
        key = self.get_key(user_id, conversation_id, player_name)
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Invalidated cache for {key}")

# Create global cache instance
inventory_cache = InventoryCache(ttl_seconds=60)  # Cache for 1 minute

# -------------------------------------------------------------------------------
# Request Deduplication
# -------------------------------------------------------------------------------

class RequestDeduplicator:
    """Prevent duplicate requests within a short time window"""
    def __init__(self, window_seconds: float = 1.0):
        self.recent_requests = {}
        self.window = window_seconds
        
    def is_duplicate(self, request_key: str) -> bool:
        now = time.time()
        if request_key in self.recent_requests:
            last_time = self.recent_requests[request_key]
            if now - last_time < self.window:
                return True
        self.recent_requests[request_key] = now
        # Clean old entries
        self.recent_requests = {
            k: v for k, v in self.recent_requests.items() 
            if now - v < self.window * 10
        }
        return False

deduplicator = RequestDeduplicator()

# -------------------------------------------------------------------------------
# Call Tracking for Debugging
# -------------------------------------------------------------------------------

class CallTracker:
    """Track function calls for debugging duplicate calls"""
    def __init__(self):
        self.call_counts = defaultdict(lambda: defaultdict(int))
        self.call_history = []
        self.last_call_time = {}
        
    def track_call(self, func_name: str, user_id: int, conversation_id: int, player_name: str = None):
        """Track a function call with detailed info"""
        now = time.time()
        call_key = f"{func_name}:{user_id}:{conversation_id}:{player_name}"
        
        # Check for rapid duplicate calls
        if call_key in self.last_call_time:
            time_since_last = now - self.last_call_time[call_key]
            if time_since_last < 1.0:  # Less than 1 second
                logger.warning(f"DUPLICATE CALL DETECTED: {func_name} called again after {time_since_last:.3f}s")
                logger.warning(f"Call key: {call_key}")
        
        self.last_call_time[call_key] = now
        self.call_counts[func_name][call_key] += 1
        
        # Store call history with stack trace
        stack_lines = traceback.format_stack(limit=10)
        # Filter to relevant lines
        filtered_stack = []
        for line in stack_lines:
            if any(x in line for x in ['/nyx/', '/logic/', '/routes/', '/tasks.py', '/main.py']):
                filtered_stack.append(line.strip())
        
        call_info = {
            'time': now,
            'func': func_name,
            'user_id': user_id,
            'conversation_id': conversation_id,
            'player_name': player_name,
            'count': self.call_counts[func_name][call_key],
            'stack': filtered_stack[-5:]  # Last 5 relevant stack frames
        }
        
        self.call_history.append(call_info)
        
        # Log the call with stack info
        logger.info(f"📦 INVENTORY CALL #{self.call_counts[func_name][call_key]}: {func_name}")
        logger.info(f"   User: {user_id}, Conv: {conversation_id}, Player: {player_name}")
        logger.info(f"   Called from:")
        for frame in filtered_stack[-3:]:  # Show last 3 frames
            logger.info(f"     {frame}")
        
        # Keep history limited
        if len(self.call_history) > 100:
            self.call_history = self.call_history[-50:]
    
    def get_report(self):
        """Get a report of call patterns"""
        report = {
            'total_calls': sum(sum(counts.values()) for counts in self.call_counts.values()),
            'by_function': {},
            'recent_duplicates': []
        }
        
        for func, counts in self.call_counts.items():
            report['by_function'][func] = {
                'total': sum(counts.values()),
                'unique_contexts': len(counts),
                'top_callers': sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        
        # Find recent duplicate calls
        now = time.time()
        for i in range(len(self.call_history) - 1, max(0, len(self.call_history) - 20), -1):
            call = self.call_history[i]
            if now - call['time'] < 60 and call['count'] > 1:  # Within last minute
                report['recent_duplicates'].append({
                    'func': call['func'],
                    'count': call['count'],
                    'age_seconds': now - call['time']
                })
        
        return report

# Create global tracker instance
call_tracker = CallTracker()

# -------------------------------------------------------------------------------
# Pydantic Models for Structured Outputs
# -------------------------------------------------------------------------------

class InventoryItem(BaseModel):
    """Structure for an inventory item"""
    item_name: str = Field(..., description="Name of the item")
    player_name: str = Field(..., description="Name of the player who owns the item")
    item_description: Optional[str] = Field(None, description="Description of the item")
    item_effect: Optional[str] = Field(None, description="Effect of the item")
    quantity: int = Field(1, description="Quantity of the item")
    category: Optional[str] = Field(None, description="Category of the item")

class InventoryOperation(BaseModel):
    """Structure for an inventory operation result"""
    success: bool = Field(..., description="Whether the operation was successful")
    item_name: str = Field(..., description="Name of the item involved")
    player_name: str = Field(..., description="Name of the player")
    operation: str = Field(..., description="Type of operation performed")
    quantity: int = Field(1, description="Quantity affected")
    error: Optional[str] = Field(None, description="Error message if any")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class InventoryList(BaseModel):
    """Structure for a list of inventory items"""
    items: List[InventoryItem] = Field(default_factory=list, description="List of inventory items")
    player_name: str = Field(..., description="Name of the player")
    total_items: int = Field(0, description="Total number of items")
    error: Optional[str] = Field(None, description="Error message if any")
    
class InventorySafety(BaseModel):
    """Output for inventory operation safety guardrail"""
    is_appropriate: bool = Field(..., description="Whether the operation is appropriate")
    reasoning: str = Field(..., description="Reasoning for the decision")
    suggested_adjustment: Optional[str] = Field(None, description="Suggested adjustment if inappropriate")

# -------------------------------------------------------------------------------
# Agent Context
# -------------------------------------------------------------------------------

class InventoryContext:
    """Context object for inventory agents"""
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.governor = None
        self.request_id = None
        
    async def initialize(self):
        """Initialize context with governance integration"""
        self.governor = await get_central_governance(self.user_id, self.conversation_id)

# -------------------------------------------------------------------------------
# Optimized Direct Database Access Function
# -------------------------------------------------------------------------------

async def get_inventory_direct(
    user_id: int,
    conversation_id: int,
    player_name: str = "Chase"
) -> Dict[str, Any]:
    """
    Direct inventory retrieval without AI agent overhead.
    This should be used for simple "get inventory" requests.
    """
    # Check cache first
    cached_result = inventory_cache.get(user_id, conversation_id, player_name)
    if cached_result is not None:
        logger.info(f"✅ Returning cached inventory for {player_name}")
        return cached_result
    
    # Log the call
    logger.info(f"📦 Direct inventory query - User: {user_id}, Conv: {conversation_id}, Player: {player_name}")
    
    # Get governor for permission check
    governor = await get_central_governance(user_id, conversation_id)
    
    # Check permission
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="inventory_system",
        action_type="get_inventory",
        action_details={"player": player_name}
    )
    
    if not permission["approved"]:
        logger.warning(f"Permission denied for get_inventory: {permission['reasoning']}")
        return {
            "items": [],
            "player_name": player_name,
            "total_items": 0,
            "error": permission["reasoning"]
        }
    
    # Query database directly
    query = """
        SELECT item_name, item_description, item_effect, category, quantity
        FROM PlayerInventory
        WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3
        ORDER BY item_name
    """
    
    items = []
    try:
        async with get_db_connection_context() as conn:
            rows = await conn.fetch(query, user_id, conversation_id, player_name)
            
            for row in rows:
                items.append({
                    "item_name": row["item_name"],
                    "player_name": player_name,
                    "item_description": row["item_description"],
                    "item_effect": row["item_effect"],
                    "category": row["category"],
                    "quantity": row["quantity"]
                })
        
        # Create result
        result = {
            "items": items,
            "player_name": player_name,
            "total_items": len(items)
        }
        
        # Cache the result
        inventory_cache.set(user_id, conversation_id, player_name, result)
        
        # Report to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="inventory_system",
            action={"type": "get_inventory", "player": player_name},
            result={"items_count": len(items)}
        )
        
        logger.info(f"✅ Retrieved {len(items)} items for {player_name} (direct query)")
        return result
        
    except Exception as e:
        logger.error(f"Error getting inventory: {e}", exc_info=True)
        return {
            "items": [],
            "player_name": player_name,
            "total_items": 0,
            "error": str(e)
        }

# -------------------------------------------------------------------------------
# Batch Operations for Efficiency
# -------------------------------------------------------------------------------

async def get_multiple_inventories(
    user_id: int,
    conversation_id: int,
    player_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Efficiently get inventories for multiple players in one operation.
    """
    results = {}
    
    # Check cache for all players first
    uncached_players = []
    for player_name in player_names:
        cached = inventory_cache.get(user_id, conversation_id, player_name)
        if cached:
            results[player_name] = cached
        else:
            uncached_players.append(player_name)
    
    if not uncached_players:
        return results
    
    # Get governor for permission check
    governor = await get_central_governance(user_id, conversation_id)
    
    # Get all uncached inventories in one query
    query = """
        SELECT player_name, item_name, item_description, item_effect, category, quantity
        FROM PlayerInventory
        WHERE user_id=$1 AND conversation_id=$2 AND player_name = ANY($3)
        ORDER BY player_name, item_name
    """
    
    try:
        async with get_db_connection_context() as conn:
            rows = await conn.fetch(query, user_id, conversation_id, uncached_players)
            
            # Group by player
            player_items = defaultdict(list)
            for row in rows:
                player_items[row["player_name"]].append({
                    "item_name": row["item_name"],
                    "item_description": row["item_description"],
                    "item_effect": row["item_effect"],
                    "category": row["category"],
                    "quantity": row["quantity"]
                })
            
            # Build results and cache
            for player_name in uncached_players:
                items = player_items.get(player_name, [])
                result = {
                    "items": items,
                    "player_name": player_name,
                    "total_items": len(items)
                }
                results[player_name] = result
                inventory_cache.set(user_id, conversation_id, player_name, result)
        
        # Report to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="inventory_system",
            action={"type": "get_multiple_inventories", "players": player_names},
            result={"players_count": len(player_names)}
        )
                
    except Exception as e:
        logger.error(f"Error getting multiple inventories: {e}", exc_info=True)
        for player_name in uncached_players:
            results[player_name] = {
                "items": [],
                "player_name": player_name,
                "total_items": 0,
                "error": str(e)
            }
    
    return results

# -------------------------------------------------------------------------------
# Function Tools (with Cache Invalidation)
# -------------------------------------------------------------------------------

@function_tool
async def fetch_inventory_item(
    ctx: RunContextWrapper[InventoryContext],
    item_name: str,
    player_name: str = "Chase"
) -> Dict[str, Any]:
    """
    Lookup a specific item in the player's inventory by item_name.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    governor = ctx.context.governor

    # Check permission
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="inventory_system",
        action_type="fetch_item",
        action_details={"item": item_name, "player": player_name}
    )
    if not permission["approved"]:
        return {"error": permission["reasoning"], "success": False}

    # Query database
    query = """
        SELECT player_name, item_description, item_effect, quantity, category
        FROM PlayerInventory
        WHERE user_id=$1
          AND conversation_id=$2
          AND item_name=$3
          AND player_name=$4
        LIMIT 1
    """
    try:
        async with get_db_connection_context() as conn:
            row: Optional[asyncpg.Record] = await conn.fetchrow(
                query, user_id, conversation_id, item_name, player_name
            )

        if not row:
            # Report action to governance
            await governor.process_agent_action_report(
                agent_type=AgentType.UNIVERSAL_UPDATER,
                agent_id="inventory_system",
                action={"type": "fetch_item", "item": item_name, "player": player_name},
                result={"found": False}
            )
            return {"error": f"No item named '{item_name}' found in {player_name}'s inventory", "success": False}

        # Extract data
        item_data = {
            "item_name": item_name,
            "player_name": row["player_name"],
            "item_description": row["item_description"],
            "item_effect": row["item_effect"],
            "quantity": row["quantity"],
            "category": row["category"],
            "success": True
        }

        # Report action to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="inventory_system",
            action={"type": "fetch_item", "item": item_name, "player": player_name},
            result={"found": True}
        )
        return item_data

    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logger.error(f"DB Error fetching inventory item '{item_name}': {db_err}", exc_info=True)
        return {"error": f"Database error: {db_err}", "success": False}
    except Exception as e:
        logger.error(f"Error fetching inventory item '{item_name}': {e}", exc_info=True)
        return {"error": str(e), "success": False}

@function_tool
async def add_item_to_inventory(
    ctx: RunContextWrapper[InventoryContext],
    player_name: str,
    item_name: str,
    description: Optional[str] = None,
    effect: Optional[str] = None,
    category: Optional[str] = None,
    quantity: int = 1
) -> Dict[str, Any]:
    """
    Add an item to a player's inventory (with cache invalidation).
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    # Invalidate cache since inventory is changing
    inventory_cache.invalidate(user_id, conversation_id, player_name)
    
    governor = ctx.context.governor

    # Check permission
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="inventory_system",
        action_type="add_item",
        action_details={"item": item_name, "player": player_name, "quantity": quantity}
    )
    if not permission["approved"]:
        return InventoryOperation(
            success=False, item_name=item_name, player_name=player_name, 
            operation="add", quantity=quantity, error=permission["reasoning"]
        ).model_dump()

    # Get LoreSystem instance
    from lore.core.lore_system import LoreSystem
    lore_system = await LoreSystem.get_instance(user_id, conversation_id)
    
    result = {}
    try:
        async with get_db_connection_context() as conn:
            # Check if item exists
            existing = await conn.fetchrow("""
                SELECT item_id, quantity FROM PlayerInventory
                WHERE user_id = $1 AND conversation_id = $2 
                AND player_name = $3 AND item_name = $4
            """, user_id, conversation_id, player_name, item_name)
            
            if existing:
                # Update existing item quantity using LoreSystem
                new_quantity = existing['quantity'] + quantity
                update_result = await lore_system.propose_and_enact_change(
                    ctx=ctx.context,
                    entity_type="PlayerInventory",
                    entity_identifier={"item_id": existing['item_id']},
                    updates={"quantity": new_quantity},
                    reason=f"Adding {quantity} more {item_name} to inventory"
                )
                
                if update_result["status"] == "committed":
                    result = InventoryOperation(
                        success=True, item_name=item_name, player_name=player_name,
                        operation="add", quantity=quantity, 
                        metadata={"new_total": new_quantity, "was_update": True}
                    ).model_dump()
                else:
                    result = InventoryOperation(
                        success=False, item_name=item_name, player_name=player_name,
                        operation="add", quantity=quantity, error=str(update_result)
                    ).model_dump()
            else:
                # Create new item using canon
                item_id = await canon.find_or_create_inventory_item(
                    ctx.context, conn,
                    player_name=player_name,
                    item_name=item_name,
                    item_description=description,
                    item_effect=effect,
                    item_category=category,
                    quantity=quantity
                )
                
                result = InventoryOperation(
                    success=True, item_name=item_name, player_name=player_name,
                    operation="add", quantity=quantity, 
                    metadata={"new_total": quantity, "was_update": False, "item_id": item_id}
                ).model_dump()

        # Report action to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="inventory_system",
            action={"type": "add_item", "item": item_name, "player": player_name, "quantity": quantity},
            result=result
        )
        return result

    except Exception as e:
        logger.error(f"Error adding item '{item_name}' to inventory: {e}", exc_info=True)
        return InventoryOperation(
            success=False, item_name=item_name, player_name=player_name, 
            operation="add", quantity=quantity, error=str(e)
        ).model_dump()

@function_tool
async def remove_item_from_inventory(
    ctx: RunContextWrapper[InventoryContext],
    player_name: str,
    item_name: str,
    quantity: int = 1
) -> Dict[str, Any]:
    """
    Remove an item from a player's inventory (with cache invalidation).
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    # Invalidate cache since inventory is changing
    inventory_cache.invalidate(user_id, conversation_id, player_name)
    
    governor = ctx.context.governor

    # Check permission
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="inventory_system",
        action_type="remove_item",
        action_details={"item": item_name, "player": player_name, "quantity": quantity}
    )
    if not permission["approved"]:
        return InventoryOperation(
            success=False, item_name=item_name, player_name=player_name, 
            operation="remove", quantity=quantity, error=permission["reasoning"]
        ).model_dump()

    # Get LoreSystem instance
    from lore.core.lore_system import LoreSystem
    lore_system = await LoreSystem.get_instance(user_id, conversation_id)
    
    result = {}
    try:
        async with get_db_connection_context() as conn:
            # Check if the item exists
            row = await conn.fetchrow("""
                SELECT item_id, quantity FROM PlayerInventory
                WHERE user_id = $1 AND conversation_id = $2 
                AND player_name = $3 AND item_name = $4
            """, user_id, conversation_id, player_name, item_name)

            if not row:
                result = InventoryOperation(
                    success=False, item_name=item_name, player_name=player_name,
                    operation="remove", quantity=quantity,
                    error=f"No item named '{item_name}' found in {player_name}'s inventory"
                ).model_dump()
            else:
                item_id, existing_qty = row["item_id"], row["quantity"]
                new_qty = existing_qty - quantity

                if new_qty > 0:
                    # Update quantity using LoreSystem
                    update_result = await lore_system.propose_and_enact_change(
                        ctx=ctx.context,
                        entity_type="PlayerInventory",
                        entity_identifier={"item_id": item_id},
                        updates={"quantity": new_qty},
                        reason=f"Removing {quantity} {item_name} from inventory"
                    )
                    
                    if update_result["status"] == "committed":
                        result = InventoryOperation(
                            success=True, item_name=item_name, player_name=player_name,
                            operation="remove", quantity=quantity, 
                            metadata={"new_total": new_qty, "was_removed": False}
                        ).model_dump()
                    else:
                        result = InventoryOperation(
                            success=False, item_name=item_name, player_name=player_name,
                            operation="remove", quantity=quantity, error=str(update_result)
                        ).model_dump()
                else:
                    # Remove the item entirely using canon
                    from lore.core import canon
                    removed = await canon.update_inventory_quantity(
                        ctx.context, conn, item_id, 0, 
                        f"Removing all {item_name} from {player_name}'s inventory"
                    )
                    
                    result = InventoryOperation(
                        success=removed, item_name=item_name, player_name=player_name,
                        operation="remove", quantity=quantity, 
                        metadata={"new_total": 0, "was_removed": True}
                    ).model_dump()

        # Report action to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="inventory_system",
            action={"type": "remove_item", "item": item_name, "player": player_name, "quantity": quantity},
            result=result
        )
        return result

    except Exception as e:
        logger.error(f"Error removing item '{item_name}' from inventory: {e}", exc_info=True)
        return InventoryOperation(
            success=False, item_name=item_name, player_name=player_name, 
            operation="remove", quantity=quantity, error=str(e)
        ).model_dump()

@function_tool
async def get_player_inventory(
    ctx: RunContextWrapper[InventoryContext],
    player_name: str = "Chase"
) -> Dict[str, Any]:
    """
    Optimized get_player_inventory with deduplication and caching.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    # Check for duplicate requests
    request_key = f"{user_id}:{conversation_id}:{player_name}:get_inventory"
    if deduplicator.is_duplicate(request_key):
        logger.warning(f"⚠️ Duplicate request detected for {player_name}'s inventory, returning cached result")
        # Try to return cached result
        cached = inventory_cache.get(user_id, conversation_id, player_name)
        if cached:
            return cached
    
    # Check cache first
    cached_result = inventory_cache.get(user_id, conversation_id, player_name)
    if cached_result is not None:
        logger.info(f"✅ Returning cached inventory for {player_name}")
        return cached_result
    
    governor = ctx.context.governor
    
    # Log the call
    logger.info(f"📦 get_player_inventory called - User: {user_id}, Conv: {conversation_id}, Player: {player_name}")
    call_tracker.track_call("get_player_inventory", user_id, conversation_id, player_name)
    
    # Check permission
    logger.debug(f"Checking permission for get_inventory operation...")
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="inventory_system",
        action_type="get_inventory",
        action_details={"player": player_name}
    )
    
    if not permission["approved"]:
        logger.warning(f"Permission denied for get_inventory: {permission['reasoning']}")
        return InventoryList(
            items=[], 
            player_name=player_name, 
            total_items=0, 
            error=permission["reasoning"]
        ).model_dump()
    
    logger.debug(f"Permission granted for get_inventory")

    # Query database
    query = """
        SELECT item_name, item_description, item_effect, category, quantity
        FROM PlayerInventory
        WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3
        ORDER BY item_name
    """
    
    inventory_items = []
    start_time = time.time()
    
    try:
        logger.debug(f"Executing database query for inventory...")
        async with get_db_connection_context() as conn:
            rows = await conn.fetch(query, user_id, conversation_id, player_name)
            
            logger.info(f"Database returned {len(rows)} items in {time.time() - start_time:.3f}s")
            
            for row in rows:
                # Use Pydantic model for structure
                item = InventoryItem(
                    item_name=row["item_name"],
                    player_name=player_name,
                    item_description=row["item_description"],
                    item_effect=row["item_effect"],
                    category=row["category"],
                    quantity=row["quantity"]
                )
                inventory_items.append(item)
                
                # Log each item for debugging
                logger.debug(f"  Item: {row['item_name']} x{row['quantity']} (category: {row['category']})")

        # Report action to governance
        logger.debug(f"Reporting action to governance...")
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="inventory_system",
            action={"type": "get_inventory", "player": player_name},
            result={"items_count": len(inventory_items)}
        )
        
        # Create result
        result = InventoryList(
            items=inventory_items, 
            player_name=player_name, 
            total_items=len(inventory_items)
        ).model_dump()
        
        # Cache the successful result
        inventory_cache.set(user_id, conversation_id, player_name, result)
        
        # Log summary
        logger.info(f"✅ Successfully retrieved {len(inventory_items)} items for {player_name}")
        logger.info(f"   Total execution time: {time.time() - start_time:.3f}s")
        
        # If this is being called repeatedly, log a warning
        if call_tracker.call_counts["get_player_inventory"][f"get_player_inventory:{user_id}:{conversation_id}:{player_name}"] > 3:
            logger.warning(f"⚠️ get_player_inventory has been called {call_tracker.call_counts['get_player_inventory'][f'get_player_inventory:{user_id}:{conversation_id}:{player_name}']} times for this context!")
            logger.warning(f"   Consider using caching or checking for duplicate calls")
        
        return result

    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logger.error(f"❌ DB Error getting inventory for '{player_name}': {db_err}", exc_info=True)
        logger.error(f"   Query execution time before error: {time.time() - start_time:.3f}s")
        
        # Report error to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="inventory_system",
            action={"type": "get_inventory", "player": player_name},
            result={"error": str(db_err), "items_count": 0}
        )
        
        return InventoryList(
            items=[], 
            player_name=player_name, 
            total_items=0, 
            error=f"Database error: {db_err}"
        ).model_dump()
        
    except Exception as e:
        logger.error(f"❌ Unexpected error getting inventory for '{player_name}': {e}", exc_info=True)
        logger.error(f"   Execution time before error: {time.time() - start_time:.3f}s")
        
        # Report error to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="inventory_system",
            action={"type": "get_inventory", "player": player_name},
            result={"error": str(e), "items_count": 0}
        )
        
        return InventoryList(
            items=[], 
            player_name=player_name, 
            total_items=0, 
            error=str(e)
        ).model_dump()

@function_tool
async def update_item_effect(
    ctx: RunContextWrapper[InventoryContext],
    player_name: str,
    item_name: str,
    new_effect: str
) -> Dict[str, Any]:
    """
    Update the effect of an item in a player's inventory (with cache invalidation).
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    # Invalidate cache since inventory is changing
    inventory_cache.invalidate(user_id, conversation_id, player_name)
    
    governor = ctx.context.governor

    # Check permission
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="inventory_system",
        action_type="update_item",
        action_details={"item": item_name, "player": player_name, "new_effect": new_effect}
    )
    if not permission["approved"]:
        return InventoryOperation(
            success=False, item_name=item_name, player_name=player_name, 
            operation="update_effect", error=permission["reasoning"]
        ).model_dump()

    # Get LoreSystem instance
    from lore.core.lore_system import LoreSystem
    lore_system = await LoreSystem.get_instance(user_id, conversation_id)
    
    result = {}
    try:
        async with get_db_connection_context() as conn:
            # Find the item
            row = await conn.fetchrow("""
                SELECT item_id FROM PlayerInventory
                WHERE user_id = $1 AND conversation_id = $2 
                AND player_name = $3 AND item_name = $4
            """, user_id, conversation_id, player_name, item_name)
            
            if not row:
                result = InventoryOperation(
                    success=False, item_name=item_name, player_name=player_name,
                    operation="update_effect",
                    error=f"No item named '{item_name}' found for {player_name}"
                ).model_dump()
            else:
                # Update using LoreSystem
                update_result = await lore_system.propose_and_enact_change(
                    ctx=ctx.context,
                    entity_type="PlayerInventory",
                    entity_identifier={"item_id": row['item_id']},
                    updates={"item_effect": new_effect},
                    reason=f"Updating effect of {item_name} to: {new_effect}"
                )
                
                if update_result["status"] == "committed":
                    result = InventoryOperation(
                        success=True, item_name=item_name, player_name=player_name,
                        operation="update_effect", 
                        metadata={"new_effect": new_effect}
                    ).model_dump()
                else:
                    result = InventoryOperation(
                        success=False, item_name=item_name, player_name=player_name,
                        operation="update_effect", error=str(update_result)
                    ).model_dump()

        # Report action to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="inventory_system",
            action={"type": "update_item", "item": item_name, "player": player_name, "effect": new_effect},
            result=result
        )
        return result

    except Exception as e:
        logger.error(f"Error updating item effect for '{item_name}': {e}", exc_info=True)
        return InventoryOperation(
            success=False, item_name=item_name, player_name=player_name, 
            operation="update_effect", error=str(e)
        ).model_dump()

@function_tool
async def categorize_items(
    ctx: RunContextWrapper[InventoryContext],
    player_name: str,
    category_mapping_json: str
) -> Dict[str, Any]:
    """
    Categorize multiple items in a player's inventory (with cache invalidation).
    
    Args:
        ctx: The context wrapper
        player_name: Name of the player
        category_mapping_json: JSON string mapping item names to categories
        
    Returns:
        Dictionary with categorization results
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    # Invalidate cache since inventory is changing
    inventory_cache.invalidate(user_id, conversation_id, player_name)
    
    # Parse the JSON string
    try:
        category_mapping = json.loads(category_mapping_json)
    except json.JSONDecodeError as e:
        return {
            "success": False, 
            "player_name": player_name,
            "operation": "categorize", 
            "error": f"Invalid JSON: {str(e)}"
        }
    
    governor = ctx.context.governor

    # Check permission
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="inventory_system",
        action_type="categorize_items",
        action_details={"player": player_name, "items_count": len(category_mapping)}
    )
    if not permission["approved"]:
        return {
            "success": False, "player_name": player_name,
            "operation": "categorize", "error": permission["reasoning"]
        }

    if not category_mapping:
        return {
            "success": True, "player_name": player_name, "operation": "categorize",
            "items_updated": 0, "items_not_found": [], "details": {}, 
            "message": "No items provided for categorization."
        }

    # Get LoreSystem instance
    from lore.core.lore_system import LoreSystem
    lore_system = await LoreSystem.get_instance(user_id, conversation_id)
    
    results = {
        "success": True, "player_name": player_name, "operation": "categorize",
        "items_updated": 0, "items_not_found": [], "details": {}
    }

    try:
        async with get_db_connection_context() as conn:
            for item_name, category in category_mapping.items():
                # Find the item
                row = await conn.fetchrow("""
                    SELECT item_id FROM PlayerInventory
                    WHERE user_id = $1 AND conversation_id = $2 
                    AND player_name = $3 AND item_name = $4
                """, user_id, conversation_id, player_name, item_name)
                
                if row:
                    # Update using LoreSystem
                    update_result = await lore_system.propose_and_enact_change(
                        ctx=ctx.context,
                        entity_type="PlayerInventory",
                        entity_identifier={"item_id": row['item_id']},
                        updates={"item_category": category},
                        reason=f"Categorizing {item_name} as {category}"
                    )
                    
                    if update_result["status"] == "committed":
                        results["items_updated"] += 1
                        results["details"][item_name] = "updated"
                    else:
                        results["items_not_found"].append(item_name)
                        results["details"][item_name] = "update failed"
                else:
                    results["items_not_found"].append(item_name)
                    results["details"][item_name] = "not found"

        # Report action to governance
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="inventory_system",
            action={"type": "categorize_items", "player": player_name, "items_count": len(category_mapping)},
            result=results
        )

        if results["items_not_found"]:
            results["success"] = False
            logger.warning(
                f"Categorize items for {player_name}: {results['items_updated']} updated, "
                f"{len(results['items_not_found'])} not found."
            )

        return results

    except Exception as e:
        logger.error(f"Error categorizing items for '{player_name}': {e}", exc_info=True)
        results["success"] = False
        results["error"] = str(e)
        return results

# -------------------------------------------------------------------------------
# Guardrail Functions
# -------------------------------------------------------------------------------

async def inventory_operation_safety(ctx, agent, input_data):
    """Input guardrail for inventory operations"""
    
    # Get the current player name from context
    current_player = ctx.context.player_name if hasattr(ctx.context, 'player_name') else "Chase"
    
    safety_agent = Agent(
        name="Inventory Safety Monitor",
        instructions=f"""
        You check if inventory operations are appropriate and safe.
        
        The current player is: {current_player}
        They can access their own inventory without restriction.
        
        Ensure that:
        1. Item names are appropriate for a femdom RPG context
        2. Item effects don't breach ethical guidelines
        3. The operation quantity is reasonable
        4. Categories are appropriate
        
        Only flag if trying to access OTHER players' inventories or if content is inappropriate.
        """,
        output_type=InventorySafety,
        model="gpt-5-nano"
    )
    
    result = await Runner.run(safety_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(InventorySafety)
    
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_appropriate,
    )

# -------------------------------------------------------------------------------
# Agent Definitions
# -------------------------------------------------------------------------------

# Item Management Agent
item_management_agent = Agent[InventoryContext](
    name="Item Management Agent",
    instructions="""
    You handle individual inventory item operations.
    
    Your role is to:
    1. Add items to player inventories
    2. Remove items from inventories
    3. Update item effects and descriptions
    4. Fetch specific item information
    5. Ensure all item data is properly formatted
    
    Maintain accurate item metadata and ensure all operations are valid.
    """,
    tools=[
        fetch_inventory_item,
        add_item_to_inventory,
        remove_item_from_inventory,
        update_item_effect
    ],
    output_type=InventoryOperation,
    model="gpt-5-nano"
)

# Inventory Analysis Agent
inventory_analysis_agent = Agent[InventoryContext](
    name="Inventory Analysis Agent",
    instructions="""
    You analyze inventory contents and relationships.
    
    Your role is to:
    1. Identify item categories and patterns
    2. Suggest categorizations for uncategorized items
    3. Analyze inventory for balance and completeness
    4. Track important item changes
    5. Identify potential issues or opportunities
    
    Provide meaningful analysis that helps with inventory management.
    """,
    tools=[
        get_player_inventory,
        categorize_items
    ],
    output_type=InventoryList,
    model="gpt-5-nano"
)

# Main Inventory System Agent
inventory_system_agent = Agent[InventoryContext](
    name="Inventory System Agent",
    instructions="""
    You are the central inventory management system for a femdom roleplaying game.
    
    Your role is to:
    1. Track all player inventory items
    2. Manage adding and removing items
    3. Update item properties when needed
    4. Categorize and organize inventory contents
    5. Provide inventory analysis and suggestions
    
    Ensure all inventory operations are properly executed and tracked.
    Work with specialized sub-agents for specific inventory tasks as needed.
    """,
    handoffs=[
        handoff(item_management_agent, tool_name_override="manage_item"),
        handoff(inventory_analysis_agent, tool_name_override="analyze_inventory")
    ],
    tools=[
        fetch_inventory_item,
        add_item_to_inventory,
        remove_item_from_inventory,
        get_player_inventory,
        update_item_effect,
        categorize_items
    ],
    input_guardrails=[
        InputGuardrail(guardrail_function=inventory_operation_safety),
    ],
    model="gpt-5-nano"
)

# -------------------------------------------------------------------------------
# Main Functions (Optimized)
# -------------------------------------------------------------------------------

async def add_item(
    user_id: int,
    conversation_id: int,
    player_name: str,
    item_name: str,
    description: Optional[str] = None,
    effect: Optional[str] = None,
    category: Optional[str] = None,
    quantity: int = 1
) -> Dict[str, Any]:
    """
    Add an item to player inventory with governance oversight.
    """
    # Create inventory context
    inventory_context = InventoryContext(user_id, conversation_id)
    await inventory_context.initialize()
    
    # Create trace for monitoring
    with trace(
        workflow_name="Inventory System",
        trace_id=f"trace_inventory-add-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
        # Create prompt
        prompt = f"""
        Add the following item to {player_name}'s inventory:
        
        Item: {item_name}
        Quantity: {quantity}
        Description: {description or "N/A"}
        Effect: {effect or "N/A"}
        Category: {category or "N/A"}
        """
        
        # Run the agent
        result = await Runner.run(
            inventory_system_agent,
            prompt,
            context=inventory_context
        )
    
    # Process the result
    operation_result = None
    
    for item in result.new_items:
        if item.type == "handoff_output_item" and "manage_item" in str(item.raw_item):
            try:
                operation_result = json.loads(item.raw_item.content)
                break
            except Exception as e:
                logging.error(f"Error parsing operation result: {e}")
    
    if not operation_result:
        # Direct tool call
        for item in result.new_items:
            if item.type == "tool_call_output_item" and "add_item_to_inventory" in str(item.raw_item):
                try:
                    operation_result = json.loads(item.output)
                    break
                except Exception as e:
                    logging.error(f"Error parsing tool output: {e}")
    
    if not operation_result:
        operation_result = {
            "success": False,
            "item_name": item_name,
            "player_name": player_name,
            "operation": "add",
            "quantity": quantity,
            "error": "Failed to parse operation result"
        }
    
    return operation_result

async def remove_item(
    user_id: int,
    conversation_id: int,
    player_name: str,
    item_name: str,
    quantity: int = 1
) -> Dict[str, Any]:
    """
    Remove an item from player inventory with governance oversight.
    """
    # Create inventory context
    inventory_context = InventoryContext(user_id, conversation_id)
    await inventory_context.initialize()
    
    # Create trace for monitoring
    with trace(
        workflow_name="Inventory System",
        trace_id=f"trace_inventory-remove-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
        # Create prompt
        prompt = f"""
        Remove {quantity} {item_name} from {player_name}'s inventory.
        """
        
        # Run the agent
        result = await Runner.run(
            inventory_system_agent,
            prompt,
            context=inventory_context
        )
    
    # Process the result
    operation_result = None
    
    for item in result.new_items:
        if item.type == "handoff_output_item" and "manage_item" in str(item.raw_item):
            try:
                operation_result = json.loads(item.raw_item.content)
                break
            except Exception as e:
                logging.error(f"Error parsing operation result: {e}")
    
    if not operation_result:
        # Direct tool call
        for item in result.new_items:
            if item.type == "tool_call_output_item" and "remove_item_from_inventory" in str(item.raw_item):
                try:
                    operation_result = json.loads(item.output)
                    break
                except Exception as e:
                    logging.error(f"Error parsing tool output: {e}")
    
    if not operation_result:
        operation_result = {
            "success": False,
            "item_name": item_name,
            "player_name": player_name,
            "operation": "remove",
            "quantity": quantity,
            "error": "Failed to parse operation result"
        }
    
    return operation_result

async def get_inventory(
    user_id: int,
    conversation_id: int,
    player_name: str = "Chase",
    use_agent: bool = False  # Add option to force agent use for complex queries
) -> Dict[str, Any]:
    """
    Optimized get player inventory function.
    
    For simple inventory retrieval, uses direct database access.
    Only uses AI agent when complex interpretation is needed.
    """
    logger.info(f"🎯 get_inventory() called - User: {user_id}, Conv: {conversation_id}, Player: {player_name}")
    
    # For simple get inventory requests, use direct access
    if not use_agent:
        return await get_inventory_direct(user_id, conversation_id, player_name)
    
    # Only use agent for complex queries that need interpretation
    inventory_context = InventoryContext(user_id, conversation_id)
    await inventory_context.initialize()
    
    # Add request tracking to context
    inventory_context.request_id = f"inv_{int(time.time()*1000)}_{user_id}_{conversation_id}"
    logger.info(f"Request ID: {inventory_context.request_id}")
    
    with trace(
        workflow_name="Inventory System",
        trace_id=f"trace_inventory-get-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
        prompt = f"Get {player_name}'s current inventory."
        result = await Runner.run(
            inventory_system_agent,
            prompt,
            context=inventory_context
        )
    
    # Process the result
    inventory_data = None
    
    for item in result.new_items:
        if item.type == "handoff_output_item" and "analyze_inventory" in str(item.raw_item):
            try:
                inventory_data = json.loads(item.raw_item.content)
                break
            except Exception as e:
                logging.error(f"Error parsing inventory data: {e}")
    
    if not inventory_data:
        # Direct tool call
        for item in result.new_items:
            if item.type == "tool_call_output_item" and "get_player_inventory" in str(item.raw_item):
                try:
                    inventory_data = json.loads(item.output)
                    break
                except Exception as e:
                    logging.error(f"Error parsing tool output: {e}")
    
    if not inventory_data:
        inventory_data = {
            "items": [],
            "player_name": player_name,
            "total_items": 0,
            "error": "Failed to parse inventory data"
        }
    
    return inventory_data

async def get_call_report(user_id: int = None, conversation_id: int = None) -> Dict[str, Any]:
    """Get a report of inventory call patterns for debugging"""
    report = call_tracker.get_report()
    
    # Add current timestamp
    report['timestamp'] = datetime.now().isoformat()
    report['user_filter'] = user_id
    report['conversation_filter'] = conversation_id
    
    # Log the report
    logger.info("INVENTORY CALL REPORT:")
    logger.info(json.dumps(report, indent=2))
    
    return report

# Register with Nyx governance
async def register_with_governance(user_id: int, conversation_id: int):
    """
    Register inventory agents with Nyx governance system.
    """
    # Get governor
    governor = await get_central_governance(user_id, conversation_id)
    
    # Check if already registered before registering
    if not governor.is_agent_registered("inventory_system", AgentType.UNIVERSAL_UPDATER):
        # Register main agent
        await governor.register_agent(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_instance=inventory_system_agent,
            agent_id="inventory_system"
        )
        
        # Issue directive for inventory management
        await governor.issue_directive(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="inventory_system",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Manage player inventory with proper validation and tracking",
                "scope": "game"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24*60  # 24 hours
        )
        
        logging.info("Inventory system registered with Nyx governance")
    else:
        logging.info("Inventory system already registered with Nyx governance, skipping")
