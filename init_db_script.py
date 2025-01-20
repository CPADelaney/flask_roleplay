# init_db_script.py
from main import initialize_database  # <--- or whatever your main file is named, e.g. 'from main import initialize_database'

if __name__ == "__main__":
    initialize_database()
    print("Database initialized (or already exists).")
