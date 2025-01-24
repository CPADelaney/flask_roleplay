# logic/inventory_logic.py

from db.connection import get_db_connection

def add_item_to_inventory(player_name, item_name, description=None, effect=None, category=None, quantity=1):
    """
    Adds an item to the player's inventory. If the item already exists,
    increments the quantity. Otherwise, inserts a new row.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Check if an item with this name already exists for the player
        cursor.execute("""
            SELECT id, quantity
            FROM PlayerInventory
            WHERE player_name = %s AND item_name = %s
        """, (player_name, item_name))
        row = cursor.fetchone()

        if row:
            # Already exists, so just update the quantity
            existing_id, existing_qty = row
            new_qty = existing_qty + quantity
            cursor.execute("""
                UPDATE PlayerInventory
                SET quantity = %s
                WHERE id = %s
            """, (new_qty, existing_id))
        else:
            # Insert a brand new row
            cursor.execute("""
                INSERT INTO PlayerInventory
                  (player_name, item_name, item_description, item_effect, category, quantity)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (player_name, item_name, description, effect, category, quantity))

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def remove_item_from_inventory(player_name, item_name, quantity=1):
    """
    Removes a certain quantity of the given item from the player's inventory.
    If the resulting quantity <= 0, the row is deleted entirely.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id, quantity
            FROM PlayerInventory
            WHERE player_name = %s AND item_name = %s
        """, (player_name, item_name))
        row = cursor.fetchone()

        if not row:
            # The player doesn't have this item
            return False  # or raise an exception

        item_id, existing_qty = row
        new_qty = existing_qty - quantity

        if new_qty > 0:
            # Just update the quantity
            cursor.execute("""
                UPDATE PlayerInventory
                SET quantity = %s
                WHERE id = %s
            """, (new_qty, item_id))
        else:
            # Remove the row altogether
            cursor.execute("""
                DELETE FROM PlayerInventory
                WHERE id = %s
            """, (item_id,))
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_player_inventory(player_name):
    """
    Returns a list of dicts representing all items the player currently holds.
    Each dict has keys: item_name, item_description, item_effect, category, quantity.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT item_name, item_description, item_effect, category, quantity
            FROM PlayerInventory
            WHERE player_name = %s
            ORDER BY item_name
        """, (player_name,))
        rows = cursor.fetchall()

        inventory = []
        for (iname, idesc, ieffect, cat, qty) in rows:
            inventory.append({
                "item_name": iname,
                "description": idesc,
                "effect": ieffect,
                "category": cat,
                "quantity": qty
            })
        return inventory
    finally:
        conn.close()

def update_item_effect(player_name, item_name, new_effect):
    """
    Example function to update the 'effect' field of an existing item,
    in case we want to store new or changed effects over time.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE PlayerInventory
            SET item_effect = %s
            WHERE player_name = %s AND item_name = %s
        """, (new_effect, player_name, item_name))
        conn.commit()
    except:
        conn.rollback()
        raise
    finally:
        conn.close()
