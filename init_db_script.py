# init_db_script.py
from db.schema_and_seed import initialize_all_data

if __name__ == "__main__":
    initialize_all_data(user_id, conversation_id)
    print("Database created and default data seeded.")
