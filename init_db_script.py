# init_db_script.py
from main import initialize_database, insert_missing_settings

if __name__ == "__main__":
    initialize_database()
    insert_missing_settings()
    print("Database and settings inserted (or already exist).")
