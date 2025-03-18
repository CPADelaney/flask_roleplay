"""
Database utility functions for the Lore System.
"""

import logging
from typing import Optional
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

from ..config.settings import config

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Custom exception for database-related errors."""
    pass

def create_db_engine():
    """Create a SQLAlchemy engine with connection pooling."""
    try:
        engine = create_engine(
            'postgresql://user:password@localhost:5432/lore_db',
            poolclass=QueuePool,
            pool_size=config.DB_POOL_SIZE,
            max_overflow=config.DB_MAX_OVERFLOW,
            pool_timeout=config.DB_POOL_TIMEOUT
        )
        return engine
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        raise DatabaseError(config.ERROR_MESSAGES["db_connection"]) from e

# Create engine and session factory
engine = create_db_engine()
SessionFactory = sessionmaker(bind=engine)

@contextmanager
def get_db_session() -> Session:
    """
    Context manager for database sessions.
    
    Yields:
        Session: SQLAlchemy session
        
    Raises:
        DatabaseError: If session creation or commit fails
    """
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error: {e}")
        raise DatabaseError(f"Database operation failed: {str(e)}")
    finally:
        session.close()

def execute_query(query: str, params: Optional[dict] = None) -> list:
    """
    Execute a database query with parameters.
    
    Args:
        query: SQL query string
        params: Optional dictionary of query parameters
        
    Returns:
        List of query results
        
    Raises:
        DatabaseError: If query execution fails
    """
    with get_db_session() as session:
        try:
            result = session.execute(query, params or {})
            return result.fetchall()
        except SQLAlchemyError as e:
            logger.error(f"Query execution failed: {e}")
            raise DatabaseError(f"Query execution failed: {str(e)}")

def execute_many(query: str, params_list: list) -> None:
    """
    Execute a database query multiple times with different parameters.
    
    Args:
        query: SQL query string
        params_list: List of parameter dictionaries
        
    Raises:
        DatabaseError: If query execution fails
    """
    with get_db_session() as session:
        try:
            session.execute(query, params_list)
        except SQLAlchemyError as e:
            logger.error(f"Batch query execution failed: {e}")
            raise DatabaseError(f"Batch query execution failed: {str(e)}") 