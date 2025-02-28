# init_db_script.py

from db.schema_and_seed import initialize_all_data
from db.connection import get_db_connection

def get_or_create_seed_user_and_conversation():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1) Try to find the first user in the DB
    cursor.execute("SELECT id FROM users LIMIT 1;")
    row = cursor.fetchone()
    
    if not row:
        # If no users exist, you can either fail here or create a dummy user
        # to seed data against.
        print("No users found! Please create at least one user first.")
        raise SystemExit
    
    user_id = row[0]

    # 2) Try to find an existing conversation for that user
    cursor.execute("SELECT id FROM conversations WHERE user_id = %s LIMIT 1;", (user_id,))
    convo_row = cursor.fetchone()

    if convo_row:
        conversation_id = convo_row[0]
        print(f"Using existing conversation_id={conversation_id} for user_id={user_id}.")
    else:
        # 3) If there's no conversation, create one
        cursor.execute(
            "INSERT INTO conversations (user_id, conversation_name) "
            "VALUES (%s, 'SeedingConversation') RETURNING id;",
            (user_id,)
        )
        conversation_id = cursor.fetchone()[0]
        conn.commit()
        print(f"Created new conversation_id={conversation_id} for user_id={user_id}.")

    cursor.close()
    conn.close()

    return user_id, conversation_id

if __name__ == "__main__":
    user_id, conversation_id = get_or_create_seed_user_and_conversation()
    initialize_all_data(user_id, conversation_id)
    print("Database created and default data seeded.")
