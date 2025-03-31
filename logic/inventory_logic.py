# logic/inventory_logic.py
import logging
import asyncio
import asyncpg
from db.connection import get_db_connection_context

async def fetch_inventory_item(user_id, conversation_id, item_name):
    """
    Lookup a specific item in the player's inventory by item_name.
    """
    try:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT player_name, item_description, item_effect, quantity, category
                FROM PlayerInventory
                WHERE user_id=$1
                  AND conversation_id=$2
                  AND item_name=$3
                LIMIT 1
            """, user_id, conversation_id, item_name)
            
            if not row:
                return {"error": f"No item named '{item_name}' found in inventory"}
            
            item_data = {
                "item_name": item_name,
                "player_name": row['player_name'],
                "item_description": row['item_description'],
                "item_effect": row['item_effect'],
                "quantity": row['quantity'],
                "category": row['category']
            }
            
            return item_data
            
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logging.error(f"DB Error fetching inventory item '{item_name}': {db_err}", exc_info=True)
        return {"error": f"Database error: {db_err}"}
    except Exception as e:
        logging.error(f"Error fetching inventory item '{item_name}': {e}", exc_info=True)
        return {"error": str(e)}

async def add_item_to_inventory(user_id, conversation_id, player_name,
                              item_name, description=None, effect=None,
                              category=None, quantity=1):
    """
    Adds an item to the player's inventory for a specific user & conversation.
    If the item already exists (for that user_id, conversation_id, player_name, item_name),
    increments the quantity. Otherwise, inserts a new row.
    """
    try:
        async with get_db_connection_context() as conn:
            # Check if this item already exists
            row = await conn.fetchrow("""
                SELECT id, quantity
                FROM PlayerInventory
                WHERE user_id=$1 AND conversation_id=$2
                  AND player_name=$3
                  AND item_name=$4
            """, user_id, conversation_id, player_name, item_name)

            if row:
                # Update the quantity
                existing_id, existing_qty = row['id'], row['quantity']
                new_qty = existing_qty + quantity
                await conn.execute("""
                    UPDATE PlayerInventory
                    SET quantity=$1
                    WHERE id=$2
                """, new_qty, existing_id)
            else:
                # Insert brand new row
                await conn.execute("""
                    INSERT INTO PlayerInventory (
                        user_id, conversation_id,
                        player_name, item_name,
                        item_description, item_effect,
                        category, quantity
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, 
                    user_id, conversation_id,
                    player_name, item_name,
                    description, effect,
                    category, quantity
                )
            
            return {"success": True, "item_name": item_name, "quantity": quantity}
            
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logging.error(f"DB Error adding item '{item_name}': {db_err}", exc_info=True)
        return {"success": False, "error": f"Database error: {db_err}"}
    except Exception as e:
        logging.error(f"Error adding item '{item_name}': {e}", exc_info=True)
        return {"success": False, "error": str(e)}

async def remove_item_from_inventory(user_id, conversation_id,
                                   player_name, item_name, quantity=1):
    """
    Removes a certain quantity of the given item from the player's inventory
    (scoped to user_id + conversation_id).
    If the resulting quantity <= 0, the row is deleted entirely.
    Returns True if something changed, or False if the item wasn't found.
    """
    try:
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT id, quantity
                FROM PlayerInventory
                WHERE user_id=$1 AND conversation_id=$2
                  AND player_name=$3
                  AND item_name=$4
            """, user_id, conversation_id, player_name, item_name)

            if not row:
                # The user doesn't have this item in this conversation
                return False

            item_id, existing_qty = row['id'], row['quantity']
            new_qty = existing_qty - quantity

            if new_qty > 0:
                # Just update the quantity
                await conn.execute("""
                    UPDATE PlayerInventory
                    SET quantity=$1
                    WHERE id=$2
                """, new_qty, item_id)
            else:
                # Remove the row entirely
                await conn.execute("""
                    DELETE FROM PlayerInventory
                    WHERE id=$1
                """, item_id)

            return True
            
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logging.error(f"DB Error removing item '{item_name}': {db_err}", exc_info=True)
        return False
    except Exception as e:
        logging.error(f"Error removing item '{item_name}': {e}", exc_info=True)
        return False

async def get_player_inventory(user_id, conversation_id, player_name):
    """
    Returns a list of dicts for all items the player currently holds,
    scoped to user_id + conversation_id.
    Each dict has keys: item_name, description, effect, category, quantity.
    """
    try:
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT item_name, item_description, item_effect, category, quantity
                FROM PlayerInventory
                WHERE user_id=$1 AND conversation_id=$2
                  AND player_name=$3
                ORDER BY item_name
            """, user_id, conversation_id, player_name)

            inventory = []
            for row in rows:
                inventory.append({
                    "item_name": row['item_name'],
                    "description": row['item_description'],
                    "effect": row['item_effect'],
                    "category": row['category'],
                    "quantity": row['quantity']
                })
            return inventory
            
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logging.error(f"DB Error getting player inventory: {db_err}", exc_info=True)
        return []
    except Exception as e:
        logging.error(f"Error getting player inventory: {e}", exc_info=True)
        return []
        
async def update_item_effect(user_id, conversation_id,
                           player_name, item_name, new_effect):
    """
    Updates the 'item_effect' field of an existing item for a specific user+conversation,
    in case we want to store new or changed effects over time.
    """
    try:
        async with get_db_connection_context() as conn:
            result = await conn.execute("""
                UPDATE PlayerInventory
                SET item_effect=$1
                WHERE user_id=$2 AND conversation_id=$3
                  AND player_name=$4
                  AND item_name=$5
            """, new_effect, user_id, conversation_id, player_name, item_name)
            
            # Parse the result string (e.g., "UPDATE 1")
            affected = 0
            if result and result.startswith("UPDATE"):
                try:
                    affected = int(result.split()[1])
                except (IndexError, ValueError):
                    pass
                    
            return {"success": affected > 0, "affected_rows": affected}
            
    except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as db_err:
        logging.error(f"DB Error updating item effect: {db_err}", exc_info=True)
        return {"success": False, "error": f"Database error: {db_err}"}
    except Exception as e:
        logging.error(f"Error updating item effect: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
