# init_db_script.py
from db.initialization import initialize_database
from routes.settings_routes import insert_missing_settings

if __name__ == "__main__":
    initialize_database()
    insert_missing_settings()
    print("Database and settings inserted (or already exist).")
