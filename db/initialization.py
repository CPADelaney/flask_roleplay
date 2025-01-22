# db/initialization.py

from db.connection import get_db_connection

def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    # CREATE TABLE statements...
    conn.commit()
    conn.close()

