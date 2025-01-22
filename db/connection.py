# db/connection.py

import os
import psycopg2

def get_db_connection():
    """
    A simple function to open a new psycopg2 connection each time.
    You can call this in any route or function that needs database access.
    """
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise EnvironmentError("DATABASE_URL is not set.")
    return psycopg2.connect(DATABASE_URL, sslmode='require')
