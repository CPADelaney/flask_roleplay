# db/connection.py

import os
import time
import logging
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

# Connection pool instance
_connection_pool = None

# Default max connections - can be overridden by environment
DEFAULT_MIN_CONNECTIONS = 5
DEFAULT_MAX_CONNECTIONS = 20

def initialize_connection_pool():
    """
    Initialize the database connection pool.
    Should be called during application startup.
    
    Returns:
        bool: True if successful, False otherwise
    """
    global _connection_pool
    
    # Skip if already initialized
    if _connection_pool is not None:
        return True
    
    try:
        # Get database configuration from environment
        dsn = os.getenv("DB_DSN")
        if not dsn:
            logger.error("DB_DSN environment variable not set")
            return False
            
        min_connections = int(os.getenv("DB_MIN_CONNECTIONS", DEFAULT_MIN_CONNECTIONS))
        max_connections = int(os.getenv("DB_MAX_CONNECTIONS", DEFAULT_MAX_CONNECTIONS))
        
        # Create the connection pool
        _connection_pool = pool.ThreadedConnectionPool(
            minconn=min_connections,
            maxconn=max_connections,
            dsn=dsn
        )
        
        logger.info(f"Database connection pool initialized with {min_connections}-{max_connections} connections")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize connection pool: {str(e)}", exc_info=True)
        return False

def get_db_connection():
    """
    Legacy function that gets a database connection.
    For backward compatibility with existing code.
    
    Returns:
        connection: A PostgreSQL database connection
    
    Raises:
        RuntimeError: If the connection pool is not initialized
    """
    global _connection_pool
    
    # Ensure pool is initialized
    if _connection_pool is None:
        if not initialize_connection_pool():
            raise RuntimeError("Database connection pool is not initialized")
    
    try:
        # Get connection from pool
        connection = _connection_pool.getconn()
        
        # Set up error handling
        connection.autocommit = False
        
        return connection
    except Exception as e:
        logger.error(f"Error getting database connection: {str(e)}", exc_info=True)
        raise

def return_db_connection(connection):
    """
    Return a connection to the pool.
    
    Args:
        connection: The connection to return
    """
    global _connection_pool
    
    if _connection_pool is not None and connection is not None:
        try:
            _connection_pool.putconn(connection)
        except Exception as e:
            logger.error(f"Error returning connection to pool: {str(e)}", exc_info=True)

@contextmanager
def get_db_connection_context():
    """
    Context manager for database connections to ensure proper handling.
    Automatically commits or rolls back transactions and returns connection to pool.
    
    Yields:
        connection: A PostgreSQL database connection
        
    Example:
        with get_db_connection_context() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM table")
                results = cursor.fetchall()
    """
    connection = None
    try:
        connection = get_db_connection()
        yield connection
        connection.commit()
    except Exception:
        if connection is not None:
            connection.rollback()
        raise
    finally:
        if connection is not None:
            return_db_connection(connection)

def execute_with_retry(func, max_retries=3, retry_delay=0.5):
    """
    Execute a database function with retry logic.
    
    Args:
        func: Function to execute (should take a connection as argument)
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    
    Returns:
        The result of the function call
    
    Example:
        def my_db_action(conn):
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM table")
                return cursor.fetchall()
                
        results = execute_with_retry(my_db_action)
    """
    retries = 0
    last_error = None
    
    while retries <= max_retries:
        try:
            with get_db_connection_context() as conn:
                return func(conn)
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            last_error = e
            retries += 1
            if retries <= max_retries:
                logger.warning(f"Database operation failed, retry {retries}/{max_retries}: {str(e)}")
                time.sleep(retry_delay * retries)  # Exponential backoff
            else:
                break
        except Exception as e:
            # Don't retry other types of exceptions
            last_error = e
            break
    
    # If we get here, all retries failed
    logger.error(f"Database operation failed after {retries} retries: {str(last_error)}", exc_info=True)
    raise last_error

def close_connection_pool():
    """
    Close the connection pool.
    Should be called during application shutdown.
    """
    global _connection_pool
    
    if _connection_pool is not None:
        try:
            _connection_pool.closeall()
            logger.info("Database connection pool closed")
        except Exception as e:
            logger.error(f"Error closing connection pool: {str(e)}", exc_info=True)
        finally:
            _connection_pool = None

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
