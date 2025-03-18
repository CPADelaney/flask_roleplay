"""
Database Schema Migration System - Manages database schema updates and versioning.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from .connection import get_db_connection
from .utils.migration_utils import MigrationError, MigrationStatus

logger = logging.getLogger(__name__)

class SchemaMigrationSystem:
    """Manages database schema migrations and versioning."""
    
    def __init__(self):
        self.migrations_table = 'schema_migrations'
        self.current_version = 0
        self.migrations: Dict[int, Dict[str, Any]] = {}
        self._load_migrations()
    
    def _load_migrations(self):
        """Load available migrations from the migrations directory."""
        try:
            import os
            from importlib import import_module
            
            migrations_dir = os.path.join(os.path.dirname(__file__), 'migrations')
            for filename in os.listdir(migrations_dir):
                if filename.endswith('.py') and not filename.startswith('__'):
                    version = int(filename.split('_')[0])
                    module = import_module(f'.migrations.{filename[:-3]}', package='db')
                    self.migrations[version] = {
                        'module': module,
                        'filename': filename,
                        'description': getattr(module, 'description', ''),
                        'timestamp': datetime.fromtimestamp(int(filename.split('_')[1]))
                    }
        except Exception as e:
            logger.error(f"Error loading migrations: {e}")
            raise
    
    async def initialize(self):
        """Initialize the migration system."""
        try:
            async with get_db_connection() as conn:
                async with conn.cursor() as cur:
                    # Create migrations table if it doesn't exist
                    await cur.execute("""
                        CREATE TABLE IF NOT EXISTS schema_migrations (
                            version INTEGER PRIMARY KEY,
                            applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            description TEXT,
                            status TEXT DEFAULT 'success',
                            error_message TEXT
                        )
                    """)
                    
                    # Get current version
                    await cur.execute("""
                        SELECT MAX(version) FROM schema_migrations
                    """)
                    result = await cur.fetchone()
                    self.current_version = result[0] if result[0] is not None else 0
        except Exception as e:
            logger.error(f"Error initializing migration system: {e}")
            raise
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        try:
            async with get_db_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("""
                        SELECT version, applied_at, description, status, error_message
                        FROM schema_migrations
                        ORDER BY version DESC
                    """)
                    rows = await cur.fetchall()
                    
                    return {
                        'current_version': self.current_version,
                        'latest_available': max(self.migrations.keys()) if self.migrations else 0,
                        'migrations': [
                            {
                                'version': row[0],
                                'applied_at': row[1],
                                'description': row[2],
                                'status': row[3],
                                'error_message': row[4]
                            }
                            for row in rows
                        ]
                    }
        except Exception as e:
            logger.error(f"Error getting migration status: {e}")
            return {
                'error': str(e)
            }
    
    async def migrate(self, target_version: Optional[int] = None) -> Dict[str, Any]:
        """Run migrations up to the target version."""
        if target_version is None:
            target_version = max(self.migrations.keys())
        
        if target_version <= self.current_version:
            return {
                'status': 'success',
                'message': 'Already at target version',
                'current_version': self.current_version
            }
        
        results = []
        try:
            async with get_db_connection() as conn:
                async with conn.cursor() as cur:
                    for version in range(self.current_version + 1, target_version + 1):
                        if version not in self.migrations:
                            continue
                        
                        migration = self.migrations[version]
                        try:
                            # Run migration
                            await migration['module'].upgrade(cur)
                            
                            # Record success
                            await cur.execute("""
                                INSERT INTO schema_migrations (version, description, status)
                                VALUES (%s, %s, 'success')
                            """, (version, migration['description']))
                            
                            results.append({
                                'version': version,
                                'status': 'success',
                                'description': migration['description']
                            })
                        except Exception as e:
                            # Record failure
                            await cur.execute("""
                                INSERT INTO schema_migrations (version, description, status, error_message)
                                VALUES (%s, %s, 'failed', %s)
                            """, (version, migration['description'], str(e)))
                            
                            results.append({
                                'version': version,
                                'status': 'failed',
                                'description': migration['description'],
                                'error': str(e)
                            })
                            
                            # Rollback failed migration
                            try:
                                await migration['module'].downgrade(cur)
                            except Exception as rollback_error:
                                logger.error(f"Error rolling back migration {version}: {rollback_error}")
                            
                            raise MigrationError(f"Migration {version} failed: {str(e)}")
            
            self.current_version = target_version
            return {
                'status': 'success',
                'message': 'Migrations completed successfully',
                'results': results
            }
        except Exception as e:
            logger.error(f"Error during migration: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'results': results
            }
    
    async def rollback(self, target_version: int) -> Dict[str, Any]:
        """Rollback migrations to the target version."""
        if target_version >= self.current_version:
            return {
                'status': 'error',
                'message': 'Cannot rollback to a version higher than current'
            }
        
        results = []
        try:
            async with get_db_connection() as conn:
                async with conn.cursor() as cur:
                    for version in range(self.current_version, target_version, -1):
                        if version not in self.migrations:
                            continue
                        
                        migration = self.migrations[version]
                        try:
                            # Run rollback
                            await migration['module'].downgrade(cur)
                            
                            # Remove migration record
                            await cur.execute("""
                                DELETE FROM schema_migrations WHERE version = %s
                            """, (version,))
                            
                            results.append({
                                'version': version,
                                'status': 'success',
                                'description': f"Rolled back {migration['description']}"
                            })
                        except Exception as e:
                            results.append({
                                'version': version,
                                'status': 'failed',
                                'description': f"Failed to rollback {migration['description']}",
                                'error': str(e)
                            })
                            raise MigrationError(f"Rollback of migration {version} failed: {str(e)}")
            
            self.current_version = target_version
            return {
                'status': 'success',
                'message': 'Rollback completed successfully',
                'results': results
            }
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'results': results
            }
    
    async def validate_schema(self) -> Dict[str, Any]:
        """Validate current database schema against expected state."""
        try:
            async with get_db_connection() as conn:
                async with conn.cursor() as cur:
                    # Get all tables
                    await cur.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public'
                    """)
                    tables = [row[0] for row in await cur.fetchall()]
                    
                    # Check for missing tables
                    missing_tables = []
                    for version in range(1, self.current_version + 1):
                        if version in self.migrations:
                            migration = self.migrations[version]
                            if hasattr(migration['module'], 'required_tables'):
                                for table in migration['module'].required_tables:
                                    if table not in tables:
                                        missing_tables.append(table)
                    
                    # Check for missing columns
                    missing_columns = []
                    for table in tables:
                        await cur.execute("""
                            SELECT column_name, data_type
                            FROM information_schema.columns
                            WHERE table_name = %s
                        """, (table,))
                        columns = {row[0]: row[1] for row in await cur.fetchall()}
                        
                        # Check against expected schema
                        for version in range(1, self.current_version + 1):
                            if version in self.migrations:
                                migration = self.migrations[version]
                                if hasattr(migration['module'], 'required_columns'):
                                    for col_name, col_type in migration['module'].required_columns.get(table, {}).items():
                                        if col_name not in columns or columns[col_name] != col_type:
                                            missing_columns.append({
                                                'table': table,
                                                'column': col_name,
                                                'expected_type': col_type,
                                                'actual_type': columns.get(col_name)
                                            })
                    
                    return {
                        'status': 'success',
                        'valid': not (missing_tables or missing_columns),
                        'missing_tables': missing_tables,
                        'missing_columns': missing_columns
                    }
        except Exception as e:
            logger.error(f"Error validating schema: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

# Create global instance
schema_migration_system = SchemaMigrationSystem()
