# logic/inventory_system_sdk.py
"""
Inventory management system using OpenAI's Agents SDK with Nyx Governance integration.

This module provides agent-based inventory management functionality, replacing the
traditional function-based approach in inventory_logic.py with a more agentic system
that integrates with Nyx governance.
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

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

class InventoryList(BaseModel):
    """Structure for a list of inventory items"""
    items: List[InventoryItem] = Field(default_factory=list, description="List of inventory items")
    player_name: str = Field(..., description="Name of the player")
    total_items: int = Field(0, description="Total number of items")
    
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
        
    async def initialize(self):
        """Initialize context with governance integration"""
        self.governor = await get_central_governance(self.user_id, self.conversation_id)

# -------------------------------------------------------------------------------
# Function Tools
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

    # Check permission (Keep as is)
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="inventory_system",
        action_type="fetch_item",
        action_details={"item": item_name, "player": player_name}
    )
    if not permission["approved"]:
        return {"error": permission["reasoning"], "success": False}

    # --- Updated DB Access ---
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
        async with get_db_connection_context() as conn: # Use context manager
            row: Optional[asyncpg.Record] = await conn.fetchrow(
                query, user_id, conversation_id, item_name, player_name
            )

        if not row:
            # Report action to governance (Keep as is)
            await governor.process_agent_action_report(
                agent_type=AgentType.UNIVERSAL_UPDATER,
                agent_id="inventory_system",
                action={"type": "fetch_item", "item": item_name, "player": player_name},
                result={"found": True}
            )
            return {"error": f"No item named '{item_name}' found in {player_name}'s inventory", "success": False}

        # Extract data (Keep as is)
        item_data = {
            "item_name": item_name,
            "player_name": row["player_name"],
            "item_description": row["item_description"],
            "item_effect": row["item_effect"],
            "quantity": row["quantity"],
            "category": row["category"],
            "success": True
        }

        # Report action to governance (Keep as is)
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="inventory_system",
            action={"type": "add_item", "item": item_name, "player": player_name, "quantity": quantity},
            result=result
        )
        return item_data

    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logger.error(f"DB Error fetching inventory item '{item_name}': {db_err}", exc_info=True)
        return {"error": f"Database error: {db_err}", "success": False}
    except Exception as e:
        logger.error(f"Error fetching inventory item '{item_name}': {e}", exc_info=True)
        return {"error": str(e), "success": False}
    # No finally block needed

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
    Add an item to a player's inventory.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
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
    
    # Use canon to find or create item
    from lore.core import canon
    
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
            action_type="add_item",
            result=result,
            context={"item": item_name, "player": player_name}
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
    Remove an item from a player's inventory.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
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
    Get all items in a player's inventory.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    governor = ctx.context.governor

    # Check permission (Keep as is)
    permission = await governor.check_action_permission(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        agent_id="inventory_system",
        action_type="get_inventory",
        action_details={"player": player_name}
    )
    if not permission["approved"]:
        return InventoryList(items=[], player_name=player_name, total_items=0, error=permission["reasoning"]).model_dump()


    # --- Updated DB Access ---
    query = """
        SELECT item_name, item_description, item_effect, category, quantity
        FROM PlayerInventory
        WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3
        ORDER BY item_name
    """
    inventory_items = []
    try:
        async with get_db_connection_context() as conn: # Use context manager
            rows: List[asyncpg.Record] = await conn.fetch(
                query, user_id, conversation_id, player_name
            )

            for row in rows:
                 # Use Pydantic model for structure
                 inventory_items.append(InventoryItem(
                     item_name=row["item_name"],
                     player_name=player_name, # Add player name here
                     item_description=row["item_description"],
                     item_effect=row["item_effect"],
                     category=row["category"],
                     quantity=row["quantity"]
                 ))

        # Report action to governance (Keep as is)
        await governor.process_agent_action_report(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="inventory_system",
            action={"type": "get_inventory", "player": player_name},
            result={"items_count": len(inventory_items)}
        )

        # Return using Pydantic model
        return InventoryList(items=inventory_items, player_name=player_name, total_items=len(inventory_items)).model_dump()

    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logger.error(f"DB Error getting inventory for '{player_name}': {db_err}", exc_info=True)
        return InventoryList(items=[], player_name=player_name, total_items=0, error=f"Database error: {db_err}").model_dump()
    except Exception as e:
        logger.error(f"Error getting inventory for '{player_name}': {e}", exc_info=True)
        return InventoryList(items=[], player_name=player_name, total_items=0, error=str(e)).model_dump()
    # No finally block needed

@function_tool
async def update_item_effect(
    ctx: RunContextWrapper[InventoryContext],
    player_name: str,
    item_name: str,
    new_effect: str
) -> Dict[str, Any]:
    """
    Update the effect of an item in a player's inventory.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
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
    Categorize multiple items in a player's inventory.
    
    Args:
        ctx: The context wrapper
        player_name: Name of the player
        category_mapping_json: JSON string mapping item names to categories
        
    Returns:
        Dictionary with categorization results
    """
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
    
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
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
    # No finally block needed

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
# Main Functions
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
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        player_name: Name of the player
        item_name: Name of the item to add
        description: Description of the item
        effect: Effect of the item
        category: Category of the item
        quantity: Quantity to add (default: 1)
        
    Returns:
        Operation result
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
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        player_name: Name of the player
        item_name: Name of the item to remove
        quantity: Quantity to remove (default: 1)
        
    Returns:
        Operation result
    """
    # Create inventory context
    inventory_context = InventoryContext(user_id, conversation_id)
    await inventory_context.initialize()
    
    # Create trace for monitoring
    with trace(
        workflow_name="Inventory System",
        trace_id=f"trace_trace_inventory-remove-{conversation_id}-{int(datetime.now().timestamp())}",
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
    player_name: str = "Chase"
) -> Dict[str, Any]:
    """
    Get player inventory with governance oversight.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        player_name: Name of the player (default: "Chase")
        
    Returns:
        Inventory data
    """
    # Create inventory context
    inventory_context = InventoryContext(user_id, conversation_id)
    await inventory_context.initialize()
    
    # Create trace for monitoring
    with trace(
        workflow_name="Inventory System",
        trace_id=f"trace_inventory-get-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
        # Create prompt
        prompt = f"""
        Get {player_name}'s current inventory.
        """
        
        # Run the agent
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

# Register with Nyx governance
async def register_with_governance(user_id: int, conversation_id: int):
    """
    Register inventory agents with Nyx governance system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
    """
    # Get governor
    governor = await get_central_governance(user_id, conversation_id)
    
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
