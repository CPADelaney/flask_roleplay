# db/connection.py

import os
import psycopg2
import asyncio

def get_db_connection():
    """
    A simple function to open a new psycopg2 connection each time.
    You can call this in any route or function that needs database access.
    """
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise EnvironmentError("DATABASE_URL is not set.")
    return psycopg2.connect(DATABASE_URL, sslmode='require')

def get_async_db_connection_string():
    """
    Returns the connection string for asyncpg.
    
    This function is used by the Data Access Layer to establish
    async database connections with asyncpg.
    
    Returns:
        str: PostgreSQL connection string for asyncpg
    """
    # Get the connection string from environment
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    if not DATABASE_URL:
        raise EnvironmentError("DATABASE_URL is not set.")
    
    # Transform psycopg2 connection string to asyncpg format if needed
    # Most connection strings will work directly with asyncpg
    return DATABASE_URL
