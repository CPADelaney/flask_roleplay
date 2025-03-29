# db/schema_migrations.py

import logging
import asyncio
import os
from importlib import import_module
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncpg # Use asyncpg types if needed

# Import the async context manager
from .connection import get_db_connection_context
# from .utils.migration_utils import MigrationError, MigrationStatus # Keep if these are simple classes

# Define custom error for clarity
class MigrationError(Exception):
    pass

logger = logging.getLogger(__name__)

class SchemaMigrationSystem:
    """Manages database schema migrations using asyncpg."""

    def __init__(self):
        self.migrations_table = 'schema_migrations'
        self.current_version = 0
        self.migrations: Dict[int, Dict[str, Any]] = {}
        self._migration_dir = os.path.join(os.path.dirname(__file__), 'migrations')
        self._load_migrations()

    def _load_migrations(self):
        """Load available migrations from the migrations directory."""
        if not os.path.isdir(self._migration_dir):
            logger.warning(f"Migrations directory not found: {self._migration_dir}")
            return
        try:
            for filename in os.listdir(self._migration_dir):
                if filename.endswith('.py') and not filename.startswith('__'):
                    try:
                        # Expecting format like 001_some_description.py or 001_1678886400_description.py
                        parts = filename.split('_', 1)
                        version = int(parts[0])
                        module_name = filename[:-3]
                        full_module_path = f'.migrations.{module_name}'

                        # Determine timestamp if included in filename
                        timestamp_unix = None
                        description_part = parts[1].replace('.py', '')
                        try:
                             # Check if the second part is a timestamp
                             maybe_ts_parts = description_part.split('_', 1)
                             if len(maybe_ts_parts[0]) == 10 and maybe_ts_parts[0].isdigit():
                                 timestamp_unix = int(maybe_ts_parts[0])
                                 description_part = maybe_ts_parts[1] if len(maybe_ts_parts) > 1 else ""
                        except ValueError:
                             pass # Not a timestamp format

                        timestamp = datetime.fromtimestamp(timestamp_unix) if timestamp_unix else None

                        module = import_module(full_module_path, package=__package__)

                        # Check for required upgrade/downgrade functions (ensure they are async)
                        if not hasattr(module, 'upgrade') or not asyncio.iscoroutinefunction(module.upgrade):
                             logger.error(f"Migration {version} ({filename}) missing async upgrade function.")
                             continue
                        if not hasattr(module, 'downgrade') or not asyncio.iscoroutinefunction(module.downgrade):
                             logger.error(f"Migration {version} ({filename}) missing async downgrade function.")
                             continue

                        self.migrations[version] = {
                            'module': module,
                            'filename': filename,
                            'description': getattr(module, 'description', description_part.replace('_', ' ')),
                            'timestamp': timestamp # Store parsed timestamp or None
                        }
                        logger.debug(f"Loaded migration {version}: {filename}")
                    except (ValueError, ImportError, AttributeError, IndexError) as e:
                        logger.error(f"Error loading migration file {filename}: {e}", exc_info=True)

            if self.migrations:
                 logger.info(f"Loaded {len(self.migrations)} migrations up to version {max(self.migrations.keys())}.")
            else:
                 logger.info("No migrations found or loaded.")

        except Exception as e:
            logger.critical(f"Critical error loading migrations: {e}", exc_info=True)
            raise # Propagate critical error

    async def initialize(self):
        """Initialize the migration system (create table, get current version)."""
        try:
            async with get_db_connection_context() as conn:
                # Create migrations table if it doesn't exist
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version BIGINT PRIMARY KEY, -- Use BIGINT for version
                        applied_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        description TEXT,
                        status TEXT DEFAULT 'success',
                        error_message TEXT
                    )
                """)

                # Get current version (highest successfully applied)
                # Use MAX(version) instead of SELECT...ORDER BY...LIMIT 1 for potentially better performance
                max_version = await conn.fetchval("""
                    SELECT MAX(version) FROM schema_migrations WHERE status = 'success'
                """)
                self.current_version = max_version if max_version is not None else 0
                logger.info(f"Migration system initialized. Current DB schema version: {self.current_version}")

        except Exception as e:
            logger.critical(f"Error initializing migration system: {e}", exc_info=True)
            raise # Propagate critical error

    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        applied_migrations_map: Dict[int, Dict] = {}
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT version, applied_at, description, status, error_message
                    FROM schema_migrations
                    ORDER BY version ASC
                """)
                for row in rows:
                    applied_migrations_map[row['version']] = dict(row)

                # Recalculate current version just in case
                self.current_version = max([v for v, m in applied_migrations_map.items() if m['status'] == 'success'] + [0])

        except Exception as e:
            logger.error(f"Error fetching migration status from DB: {e}", exc_info=True)
            return {'error': f"DB Error fetching status: {e}"}

        # Combine applied status with available migrations
        all_statuses = []
        latest_available = max(self.migrations.keys()) if self.migrations else 0
        for v in sorted(list(self.migrations.keys()) + list(applied_migrations_map.keys())):
            status_info = applied_migrations_map.get(v)
            migration_info = self.migrations.get(v)

            if status_info: # Migration was applied (or attempted)
                 all_statuses.append({
                     'version': v,
                     'status': status_info['status'],
                     'applied_at': status_info['applied_at'],
                     'description': status_info['description'] or (migration_info['description'] if migration_info else 'N/A'),
                     'filename': migration_info['filename'] if migration_info else 'N/A',
                     'error': status_info['error_message']
                 })
            elif migration_info: # Migration is available but not applied
                  all_statuses.append({
                     'version': v,
                     'status': 'pending',
                     'applied_at': None,
                     'description': migration_info['description'],
                     'filename': migration_info['filename'],
                     'error': None
                 })
            # else: # Version exists in DB but not in code (orphaned record?) - log warning?

        return {
            'current_version': self.current_version,
            'latest_available': latest_available,
            'migrations': all_statuses
        }


    async def migrate(self, target_version: Optional[int] = None) -> Dict[str, Any]:
        """Run migrations up to the target version using asyncpg."""
        if not self.migrations:
            logger.info("No migrations available to run.")
            return {'status': 'success', 'message': 'No migrations defined.', 'current_version': self.current_version}

        latest_available = max(self.migrations.keys())
        target_version = target_version if target_version is not None else latest_available

        if target_version > latest_available:
             logger.warning(f"Target version {target_version} is higher than latest available {latest_available}. Migrating to latest available.")
             target_version = latest_available

        # Refresh current version from DB before starting
        await self.initialize()

        if target_version <= self.current_version:
            logger.info(f"Database is already at or beyond target version {target_version} (Current: {self.current_version}). No migration needed.")
            return {'status': 'success', 'message': 'Already at target version', 'current_version': self.current_version}

        logger.info(f"Starting migration from version {self.current_version} to {target_version}...")
        results = []
        applied_version = self.current_version

        try:
            async with get_db_connection_context() as conn:
                # Iterate through versions to apply
                for version in sorted([v for v in self.migrations if self.current_version < v <= target_version]):
                    migration = self.migrations[version]
                    logger.info(f"Applying migration {version}: {migration['description']} ({migration['filename']})...")

                    # Run migration within a transaction for atomicity per migration file
                    async with conn.transaction():
                        try:
                            # Run the async upgrade function, passing the connection
                            await migration['module'].upgrade(conn)

                            # Record success in the migrations table
                            await conn.execute("""
                                INSERT INTO schema_migrations (version, description, status, applied_at)
                                VALUES ($1, $2, 'success', CURRENT_TIMESTAMP)
                                ON CONFLICT (version) DO UPDATE SET
                                    description = EXCLUDED.description,
                                    status = EXCLUDED.status,
                                    applied_at = CURRENT_TIMESTAMP,
                                    error_message = NULL
                            """, version, migration['description'])

                            results.append({'version': version, 'status': 'success', 'description': migration['description']})
                            applied_version = version # Update successfully applied version
                            logger.info(f"Successfully applied migration {version}.")

                        except Exception as migration_err:
                            # Rollback is handled automatically by `conn.transaction()` failing
                            error_msg = f"Migration {version} failed: {migration_err}"
                            logger.error(error_msg, exc_info=True)

                            # Record failure state
                            await conn.execute("""
                                INSERT INTO schema_migrations (version, description, status, error_message, applied_at)
                                VALUES ($1, $2, 'failed', $3, CURRENT_TIMESTAMP)
                                ON CONFLICT (version) DO UPDATE SET
                                    description = EXCLUDED.description,
                                    status = EXCLUDED.status,
                                    error_message = EXCLUDED.error_message,
                                    applied_at = CURRENT_TIMESTAMP
                            """, version, migration['description'], str(migration_err))

                            results.append({'version': version, 'status': 'failed', 'description': migration['description'], 'error': str(migration_err)})

                            # Stop further migrations on failure
                            raise MigrationError(error_msg) from migration_err

            self.current_version = applied_version
            logger.info(f"Migrations completed. Current DB version: {self.current_version}")
            return {'status': 'success', 'message': 'Migrations applied successfully.', 'results': results, 'current_version': self.current_version}

        except MigrationError as migration_stop_err: # Catch the re-raised error
             self.current_version = applied_version # Record the last successful version
             logger.error(f"Migration process stopped due to failure at version {results[-1]['version'] if results else 'unknown'}.")
             return {'status': 'failed', 'message': str(migration_stop_err), 'results': results, 'current_version': self.current_version}
        except Exception as e:
            logger.critical(f"Unexpected error during migration process: {e}", exc_info=True)
            self.current_version = applied_version # Record the last successful version
            return {'status': 'error', 'message': f"Unexpected error: {e}", 'results': results, 'current_version': self.current_version}

    async def rollback(self, target_version: int) -> Dict[str, Any]:
        """Rollback migrations to the target version using asyncpg."""
        # Refresh current version from DB before starting
        await self.initialize()

        if target_version >= self.current_version:
            logger.error(f"Cannot rollback to target version {target_version} (not lower than current {self.current_version}).")
            return {'status': 'error', 'message': 'Target version must be lower than current version'}
        if target_version < 0:
             logger.error("Cannot rollback to a negative version.")
             return {'status': 'error', 'message': 'Target version cannot be negative.'}


        logger.info(f"Starting rollback from version {self.current_version} down to {target_version}...")
        results = []
        rolled_back_to_version = self.current_version

        try:
            async with get_db_connection_context() as conn:
                # Iterate downwards from current to target+1
                for version in sorted([v for v in self.migrations if target_version < v <= self.current_version], reverse=True):
                    migration = self.migrations[version]
                    logger.info(f"Rolling back migration {version}: {migration['description']} ({migration['filename']})...")

                    # Check if this version was successfully applied before rolling back
                    status_rec = await conn.fetchrow("SELECT status FROM schema_migrations WHERE version = $1", version)
                    if not status_rec or status_rec['status'] != 'success':
                         logger.warning(f"Skipping rollback for version {version} as it was not successfully applied (status: {status_rec['status'] if status_rec else 'Not Found'}).")
                         # We might still need to delete the record if it exists in a failed state
                         await conn.execute("DELETE FROM schema_migrations WHERE version = $1", version)
                         rolled_back_to_version = version - 1 # Update effective version
                         continue # Skip the actual downgrade logic


                    # Run downgrade within a transaction
                    async with conn.transaction():
                        try:
                            # Run the async downgrade function
                            await migration['module'].downgrade(conn)

                            # Remove migration record from table
                            await conn.execute("DELETE FROM schema_migrations WHERE version = $1", version)

                            results.append({'version': version, 'status': 'success', 'description': f"Rolled back {migration['description']}"})
                            rolled_back_to_version = version - 1 # Update successfully rolled back version
                            logger.info(f"Successfully rolled back migration {version}.")

                        except Exception as rollback_err:
                            # Rollback of transaction is automatic on exception
                            error_msg = f"Rollback of migration {version} failed: {rollback_err}"
                            logger.error(error_msg, exc_info=True)

                            # Update status to 'rollback_failed'? Or leave as 'success'? Difficult choice.
                            # Let's mark it as failed rollback attempt.
                            await conn.execute("""
                                UPDATE schema_migrations SET status = 'rollback_failed', error_message = $2
                                WHERE version = $1
                            """, version, str(rollback_err))

                            results.append({'version': version, 'status': 'failed', 'description': f"Failed to rollback {migration['description']}", 'error': str(rollback_err)})

                            # Stop further rollbacks on failure
                            raise MigrationError(error_msg) from rollback_err

            self.current_version = rolled_back_to_version
            logger.info(f"Rollback completed. Current DB version is now effectively {self.current_version}")
            return {'status': 'success', 'message': 'Rollback completed successfully.', 'results': results, 'current_version': self.current_version}

        except MigrationError as migration_stop_err: # Catch the re-raised error
             self.current_version = rolled_back_to_version # Record the last successful state
             logger.error(f"Rollback process stopped due to failure at version {results[-1]['version'] if results else 'unknown'}.")
             return {'status': 'failed', 'message': str(migration_stop_err), 'results': results, 'current_version': self.current_version}
        except Exception as e:
            logger.critical(f"Unexpected error during rollback process: {e}", exc_info=True)
            self.current_version = rolled_back_to_version # Record the last successful state
            return {'status': 'error', 'message': f"Unexpected error: {e}", 'results': results, 'current_version': self.current_version}

    async def validate_schema(self) -> Dict[str, Any]:
        """Validate current database schema (basic checks). Needs refinement."""
        # This is a complex task. A full validation would involve checking
        # all tables, columns, types, constraints, indexes defined by applied migrations.
        # This basic version just checks if the migrations table exists.
        logger.info("Performing basic schema validation (checking migrations table existence)...")
        try:
            async with get_db_connection_context() as conn:
                # Check if migrations table exists
                table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = $1
                    )
                """, self.migrations_table)

                if not table_exists:
                    logger.error("Validation failed: Schema migrations table does not exist.")
                    return {'status': 'failed', 'valid': False, 'message': 'Migrations table missing'}

                # TODO: Implement more thorough checks if needed (compare expected vs actual schema)
                logger.info("Basic schema validation passed (migrations table exists).")
                return {'status': 'success', 'valid': True, 'message': 'Migrations table exists'}

        except Exception as e:
            logger.error(f"Error during schema validation: {e}", exc_info=True)
            return {'status': 'error', 'valid': False, 'message': str(e)}

# Create global instance (can be imported and used)
schema_migration_system = SchemaMigrationSystem()

# --- Example Migration File Structure (db/migrations/001_create_users.py) ---
# """
# Creates the initial users table.
# """
# description = "Create initial users table"
#
# # Optional: Define tables/columns this migration affects for validation
# # required_tables = ["users"]
# # required_columns = {
# #     "users": {
# #         "id": "integer", # Or bigint, serial etc. based on actual type
# #         "username": "character varying",
# #         "password_hash": "text",
# #         "created_at": "timestamp with time zone"
# #     }
# # }
#
# async def upgrade(conn):
#     """Applies the migration."""
#     await conn.execute("""
#         CREATE TABLE users (
#             id SERIAL PRIMARY KEY,
#             username VARCHAR(50) NOT NULL UNIQUE,
#             password_hash TEXT NOT NULL,
#             email VARCHAR(100) UNIQUE, -- Added email
#             created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
#         );
#     """)
#     await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")
#     # Add other index creation or alterations
#
# async def downgrade(conn):
#     """Reverts the migration."""
#     # Be careful with destructive downgrades!
#     await conn.execute("DROP TABLE IF EXISTS users;")
# """
