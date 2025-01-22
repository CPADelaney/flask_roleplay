# db/connection.py

import os
import psycopg2

def get_db_connection():
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise EnvironmentError("DATABASE_URL is not set.")
    return psycopg2.connect(DATABASE_URL, sslmode='require')

