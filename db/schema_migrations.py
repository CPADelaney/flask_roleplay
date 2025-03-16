"""
Schema migration tracking for Nyx dynamic schema management.

Tracks all schema changes to ensure consistency across environments and deployments.
"""

import logging
import json
import asyncpg
from datetime import datetime
from typing import Dict, Any, List

# Database connection helper
from db.connection import get_db_connection

logger = logging.getLogger(__name__)

async def initialize_migration_tracking():
    """Set up the migration tracking tables if they don't exist."""
    try:
        async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
            async with pool.acquire() as conn:
                # Check if the migrations table exists
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'schema_migrations'
                    )
                """)
                
                if not exists:
                    # Create the migrations table
                    await conn.execute("""
                        CREATE TABLE schema_migrations (
                            id SERIAL PRIMARY KEY,
                            user_id INTEGER NOT NULL,
                            conversation_id INTEGER,
                            migration_type VARCHAR(50) NOT NULL,
                            changes JSONB NOT NULL,
                            applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            environment VARCHAR(50) NOT NULL,
                            
                            CONSTRAINT schema_migrations_user_fk
                                FOREIGN KEY (user_id)
                                REFERENCES users(id)
                                ON DELETE CASCADE
                        )
                    """)
                    
                    # Create indexes
                    await conn.execute("""
                        CREATE INDEX idx_schema_migrations_user
                        ON schema_migrations(user_id)
                    """)
                    
                    logger.info("Created schema_migrations tracking table")
                    
                # Check if the schema registry exists
                registry_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'dynamic_schema_registry'
                    )
                """)
                
                if not registry_exists:
                    # Create the registry table
                    await conn.execute("""
                        CREATE TABLE dynamic_schema_registry (
                            id SERIAL PRIMARY KEY,
                            table_name VARCHAR(100) NOT NULL,
                            is_active BOOLEAN DEFAULT TRUE,
                            columns JSONB NOT NULL,
                            creator_user_id INTEGER,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            metadata JSONB,
                            
                            CONSTRAINT dynamic_schema_registry_table_key
                                UNIQUE (table_name)
                        )
                    """)
                    
                    logger.info("Created dynamic_schema_registry table")
    
    except Exception as e:
        logger.error(f"Error initializing migration tracking: {e}")
        raise

async def register_schema_change(change_data: Dict[str, Any]) -> int:
    """
    Register a schema change for migration tracking.
    
    Args:
        change_data: Dictionary with schema change information
        
    Returns:
        Migration ID
    """
    try:
        # Get environment information
        import os
        environment = os.environ.get("NYX_ENVIRONMENT", "development")
        
        async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
            async with pool.acquire() as conn:
                # Insert migration record
                migration_id = await conn.fetchval("""
                    INSERT INTO schema_migrations (
                        user_id, conversation_id, migration_type, 
                        changes, applied_at, environment
                    )
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                """,
                change_data["user_id"],
                change_data.get("conversation_id"),
                "dynamic_schema",
                json.dumps(change_data),
                datetime.now(),
                environment
                )
                
                # Update registry for tables
                for table_name in change_data.get("changes", {}).get("tables_created", []):
                    if isinstance(table_name, dict) and "table_name" in table_name:
                        table_info = table_name
                        table_name = table_info["table_name"]
                        
                        # Check if table already in registry
                        exists = await conn.fetchval("""
                            SELECT EXISTS (
                                SELECT FROM dynamic_schema_registry
                                WHERE table_name = $1
                            )
                        """, table_name)
                        
                        if not exists:
                            # Add to registry
                            await conn.execute("""
                                INSERT INTO dynamic_schema_registry (
                                    table_name, columns, creator_user_id, metadata
                                )
                                VALUES ($1, $2, $3, $4)
                            """,
                            table_name,
                            json.dumps(table_info.get("columns", [])),
                            change_data["user_id"],
                            json.dumps({
                                "description": table_info.get("description", ""),
                                "created_from_migration": migration_id
                            })
                            )
                        
                # Update registry for extended tables
                for table_info in change_data.get("changes", {}).get("tables_extended", []):
                    if isinstance(table_info, dict) and "table_name" in table_info:
                        table_name = table_info["table_name"]
                        
                        # Check if table in registry
                        registry_record = await conn.fetchrow("""
                            SELECT id, columns 
                            FROM dynamic_schema_registry
                            WHERE table_name = $1
                        """, table_name)
                        
                        if registry_record:
                            # Update existing record
                            registry_id = registry_record["id"]
                            current_columns = json.loads(registry_record["columns"])
                            
                            # Append new columns
                            for col in table_info.get("columns_added", []):
                                if col not in current_columns:
                                    current_columns.append(col)
                            
                            # Update registry
                            await conn.execute("""
                                UPDATE dynamic_schema_registry
                                SET columns = $2,
                                    last_updated = CURRENT_TIMESTAMP
                                WHERE id = $1
                            """,
                            registry_id,
                            json.dumps(current_columns)
                            )
                        else:
                            # Add to registry
                            await conn.execute("""
                                INSERT INTO dynamic_schema_registry (
                                    table_name, columns, creator_user_id, metadata
                                )
                                VALUES ($1, $2, $3, $4)
                            """,
                            table_name,
                            json.dumps(table_info.get("columns_added", [])),
                            change_data["user_id"],
                            json.dumps({
                                "extended_from_migration": migration_id
                            })
                            )
                
                return migration_id
                
    except Exception as e:
        logger.error(f"Error registering schema change: {e}")
        return -1

async def get_schema_migration_history(user_id: int = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get schema migration history.
    
    Args:
        user_id: Optional user ID to filter by
        limit: Maximum number of records to return
        
    Returns:
        List of migration records
    """
    try:
        async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
            async with pool.acquire() as conn:
                if user_id:
                    rows = await conn.fetch("""
                        SELECT id, user_id, conversation_id, migration_type, 
                               changes, applied_at, environment
                        FROM schema_migrations
                        WHERE user_id = $1
                        ORDER BY applied_at DESC
                        LIMIT $2
                    """, user_id, limit)
                else:
                    rows = await conn.fetch("""
                        SELECT id, user_id, conversation_id, migration_type, 
                               changes, applied_at, environment
                        FROM schema_migrations
                        ORDER BY applied_at DESC
                        LIMIT $1
                    """, limit)
                
                return [dict(row) for row in rows]
                
    except Exception as e:
        logger.error(f"Error getting migration history: {e}")
        return []

async def get_dynamic_tables_registry() -> List[Dict[str, Any]]:
    """
    Get the registry of dynamically created tables.
    
    Returns:
        List of registered tables
    """
    try:
        async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
            async with pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, table_name, is_active, columns, 
                           creator_user_id, created_at, last_updated, metadata
                    FROM dynamic_schema_registry
                    ORDER BY created_at DESC
                """)
                
                return [dict(row) for row in rows]
                
    except Exception as e:
        logger.error(f"Error getting dynamic tables registry: {e}")
        return []
