from db.connection import get_db_connection

def add_item_to_inventory(user_id, conversation_id, player_name,
                          item_name, description=None, effect=None,
                          category=None, quantity=1):
    """
    Adds an item to the player's inventory for a specific user & conversation.
    If the item already exists (for that user_id, conversation_id, player_name, item_name),
    increments the quantity. Otherwise, inserts a new row.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Check if this item already exists for the same (user_id, conv_id, player_name, item_name)
        cursor.execute("""
            SELECT id, quantity
            FROM PlayerInventory
            WHERE user_id=%s AND conversation_id=%s
              AND player_name=%s
              AND item_name=%s
        """, (user_id, conversation_id, player_name, item_name))
        row = cursor.fetchone()

        if row:
            # Update the quantity
            existing_id, existing_qty = row
            new_qty = existing_qty + quantity
            cursor.execute("""
                UPDATE PlayerInventory
                SET quantity=%s
                WHERE id=%s
            """, (new_qty, existing_id))
        else:
            # Insert brand new row
            cursor.execute("""
                INSERT INTO PlayerInventory (
                    user_id, conversation_id,
                    player_name, item_name,
                    item_description, item_effect,
                    category, quantity
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                user_id, conversation_id,
                player_name, item_name,
                description, effect,
                category, quantity
            ))

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def remove_item_from_inventory(user_id, conversation_id,
                               player_name, item_name, quantity=1):
    """
    Removes a certain quantity of the given item from the player's inventory
    (scoped to user_id + conversation_id).
    If the resulting quantity <= 0, the row is deleted entirely.
    Returns True if something changed, or False if the item wasn't found.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id, quantity
            FROM PlayerInventory
            WHERE user_id=%s AND conversation_id=%s
              AND player_name=%s
              AND item_name=%s
        """, (user_id, conversation_id, player_name, item_name))
        row = cursor.fetchone()

        if not row:
            # The user doesn't have this item in this conversation
            return False

        item_id, existing_qty = row
        new_qty = existing_qty - quantity

        if new_qty > 0:
            # Just update the quantity
            cursor.execute("""
                UPDATE PlayerInventory
                SET quantity=%s
                WHERE id=%s
            """, (new_qty, item_id))
        else:
            # Remove the row entirely
            cursor.execute("""
                DELETE FROM PlayerInventory
                WHERE id=%s
            """, (item_id,))

        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_player_inventory(user_id, conversation_id, player_name):
    """
    Returns a list of dicts for all items the player currently holds,
    scoped to user_id + conversation_id.
    Each dict has keys: item_name, description, effect, category, quantity.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT item_name, item_description, item_effect, category, quantity
            FROM PlayerInventory
            WHERE user_id=%s AND conversation_id=%s
              AND player_name=%s
            ORDER BY item_name
        """, (user_id, conversation_id, player_name))
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

def update_item_effect(user_id, conversation_id,
                       player_name, item_name, new_effect):
    """
    Updates the 'item_effect' field of an existing item for a specific user+conversation,
    in case we want to store new or changed effects over time.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE PlayerInventory
            SET item_effect=%s
            WHERE user_id=%s AND conversation_id=%s
              AND player_name=%s
              AND item_name=%s
        """, (new_effect, user_id, conversation_id, player_name, item_name))
        conn.commit()
    except:
        conn.rollback()
        raise
    finally:
        conn.close()
