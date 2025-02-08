from db.connection import get_db_connection
import json

def get_social_link(user_id, conversation_id,
                    entity1_type, entity1_id,
                    entity2_type, entity2_id):
    """
    Fetch an existing social link row if it exists for (user_id, conversation_id, e1, e2).
    Return a dict with link_id, link_type, link_level, link_history, else None.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT link_id, link_type, link_level, link_history
            FROM SocialLinks
            WHERE user_id=%s AND conversation_id=%s
              AND entity1_type=%s AND entity1_id=%s
              AND entity2_type=%s AND entity2_id=%s
        """, (user_id, conversation_id, entity1_type, entity1_id,
              entity2_type, entity2_id))
        row = cursor.fetchone()
        if row:
            (link_id, link_type, link_level, link_hist) = row
            return {
                "link_id": link_id,
                "link_type": link_type,
                "link_level": link_level,
                "link_history": link_hist
            }
        else:
            return None
    finally:
        conn.close()

def create_social_link(user_id, conversation_id,
                       entity1_type, entity1_id,
                       entity2_type, entity2_id,
                       link_type="neutral", link_level=0):
    """
    Create a new SocialLinks row for (user_id, conversation_id, e1, e2).
    Initialize link_history as an empty array.
    If a matching row already exists, return its link_id.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO SocialLinks (
                user_id, conversation_id,
                entity1_type, entity1_id,
                entity2_type, entity2_id,
                link_type, link_level, link_history
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, '[]')
            ON CONFLICT (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id)
            DO NOTHING
            RETURNING link_id
            """,
            (
                user_id, conversation_id,
                entity1_type, entity1_id,
                entity2_type, entity2_id,
                link_type, link_level
            )
        )
        result = cursor.fetchone()
        if result is None:
            # If the insert did nothing because the row already exists,
            # retrieve the existing link_id.
            cursor.execute(
                """
                SELECT link_id FROM SocialLinks
                WHERE user_id=%s AND conversation_id=%s
                  AND entity1_type=%s AND entity1_id=%s
                  AND entity2_type=%s AND entity2_id=%s
                """,
                (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id)
            )
            result = cursor.fetchone()
        link_id = result[0]
        conn.commit()
        return link_id
    except:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_link_type_and_level(user_id, conversation_id,
                               link_id, new_type=None, level_change=0):
    """
    Adjust an existing link's type or level, scoping to user_id + conversation_id + link_id.
    We fetch the old link_type & link_level, then set new values.
    Return dict with new_type, new_level if found.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # We'll match link_id, user_id, conversation_id so we don't update other users' links
        cursor.execute("""
            SELECT link_type, link_level
            FROM SocialLinks
            WHERE link_id=%s AND user_id=%s AND conversation_id=%s
        """, (link_id, user_id, conversation_id))
        row = cursor.fetchone()
        if not row:
            return None  # Not found

        (old_type, old_level) = row
        final_type = new_type if new_type else old_type
        final_level = old_level + level_change

        cursor.execute("""
            UPDATE SocialLinks
            SET link_type=%s, link_level=%s
            WHERE link_id=%s AND user_id=%s AND conversation_id=%s
        """, (final_type, final_level, link_id, user_id, conversation_id))
        conn.commit()
        return {
            "link_id": link_id,
            "new_type": final_type,
            "new_level": final_level
        }
    except:
        conn.rollback()
        raise
    finally:
        conn.close()

def add_link_event(user_id, conversation_id,
                   link_id, event_text):
    """
    Append a string to link_history array for link_id (scoped to user_id + conversation_id).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE SocialLinks
            SET link_history = COALESCE(link_history, '[]'::jsonb) || to_jsonb(%s)
            WHERE link_id=%s AND user_id=%s AND conversation_id=%s
            RETURNING link_history
        """, (event_text, link_id, user_id, conversation_id))
        updated = cursor.fetchone()
        if not updated:
            print(f"No link found for link_id={link_id}, user_id={user_id}, conv_id={conversation_id}")
        else:
            print(f"Appended event to link_history => {updated[0]}")
        conn.commit()
    except:
        conn.rollback()
        raise
    finally:
        conn.close()
