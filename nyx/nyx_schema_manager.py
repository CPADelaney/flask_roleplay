# nyx/nyx_schema_manager.py
"""
Schema management for Nyx - handles dynamic detection and creation of database structures.

This module:
1. Analyzes roleplay context to identify data tracking needs
2. Discovers existing similar tables and columns
3. Optimizes schema changes to minimize redundancy
4. Manages schema versioning and migrations
5. Implements rate limiting for schema changes
"""

import logging
import json
import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Database connection helper
from db.connection import get_db_connection_context

# Vector embedding utility
from utils.embedding_service import get_embedding

# Schema migration tracking
from db.schema_migrations import register_schema_change

logger = logging.getLogger(__name__)

class SchemaChangeType:
    """Constants for schema change types."""
    CREATE_TABLE = "create_table"
    ADD_COLUMN = "add_column"
    EXTEND_TABLE = "extend_table"
    RENAME_COLUMN = "rename_column"

class NyxSchemaManager:
    """
    Manages dynamic schema analysis and creation for Nyx.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the schema manager.
        
        Args:
            user_id: The user ID
            conversation_id: The conversation ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lock = asyncio.Lock()
        
        # Rate limiting - max changes per day
        self.max_daily_changes = 5
        self.changes_today = 0
        self.last_change_reset = datetime.now()
        
        # Schema cache
        self.table_cache = {}
        self.column_cache = {}
        self.cache_timestamp = None
        self.cache_ttl = timedelta(minutes=30)
        
        # Required prefixes for dynamically created tables
        self.dynamic_table_prefix = "dynamic_"
        
        # Important similarity thresholds
        self.high_table_similarity = 0.85  # Threshold for considering tables semantically equivalent
        self.high_column_similarity = 0.80  # Threshold for considering columns semantically equivalent
        self.medium_similarity = 0.70      # Threshold for suggesting alternatives
        
        # Reserved system table names that can't be modified
        self.reserved_tables = {
            "conversations", "user_settings", "npcstats", "npcdialogues", 
            "systemmessages", "memorysystem", "nyxnpcdirectives", "nyxagentdirectives",
            "nyxactiontracking", "nyxdirectiveresponses", "nyxdecisionlog", 
            "nyxagentfeedback", "nyxgamestate"
        }
        
        # Standard columns that every table should have
        self.standard_columns = {
            "id": {"type": "SERIAL", "nullable": False, "is_primary": True},
            "user_id": {"type": "INTEGER", "nullable": False, "index": True},
            "conversation_id": {"type": "INTEGER", "nullable": False, "index": True},
            "created_at": {"type": "TIMESTAMP WITH TIME ZONE", "nullable": False, "default": "CURRENT_TIMESTAMP"},
            "updated_at": {"type": "TIMESTAMP WITH TIME ZONE", "nullable": False, "default": "CURRENT_TIMESTAMP"}
        }
        
    async def refresh_schema_cache(self) -> None:
        """Refresh the cached schema information."""
        if (not self.cache_timestamp or 
            datetime.now() - self.cache_timestamp > self.cache_ttl):
            
            async with self.lock:
                # Only refresh if still needed after acquiring lock
                if (not self.cache_timestamp or 
                    datetime.now() - self.cache_timestamp > self.cache_ttl):
                    
                    await self._load_schema_information()
                    self.cache_timestamp = datetime.now()
    
    async def _load_schema_information(self) -> None:
        """Load table and column information from the database."""
        try:
            async with get_db_connection_context() as conn:
                # Get all tables and their comments
                tables = await conn.fetch("""
                    SELECT 
                        t.table_name,
                        pg_catalog.obj_description(
                            pg_catalog.pg_class.oid, 'pg_class'
                        ) as description,
                        tc.table_type,
                        to_char(tc.creation_time, 'YYYY-MM-DD HH24:MI:SS') as created,
                        tc.is_dynamic
                    FROM information_schema.tables t
                    JOIN (
                        SELECT c.relname as table_name, 
                               c.relkind as table_type,
                               c.relcreated as creation_time,
                               CASE 
                                 WHEN c.relname LIKE 'dynamic\\_%' THEN TRUE 
                                 ELSE FALSE 
                               END as is_dynamic
                        FROM pg_catalog.pg_class c
                        WHERE c.relkind = 'r'
                          AND c.relnamespace = (
                              SELECT oid FROM pg_catalog.pg_namespace 
                              WHERE nspname = 'public'
                          )
                    ) tc ON t.table_name = tc.table_name
                    WHERE t.table_schema = 'public'
                """)
                
                # Process table information
                self.table_cache = {}
                for table in tables:
                    table_name = table['table_name']
                    self.table_cache[table_name] = {
                        'description': table['description'] or '',
                        'is_dynamic': table['is_dynamic'],
                        'created': table['created'],
                        'columns': {}
                    }
                
                # Get column information for all tables
                for table_name in self.table_cache:
                    columns = await conn.fetch("""
                        SELECT 
                            column_name,
                            data_type,
                            is_nullable,
                            column_default,
                            pg_catalog.col_description(
                                (table_schema || '.' || table_name)::regclass::oid,
                                ordinal_position
                            ) as description
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = $1
                        ORDER BY ordinal_position
                    """, table_name)
                    
                    for column in columns:
                        col_name = column['column_name']
                        self.table_cache[table_name]['columns'][col_name] = {
                            'data_type': column['data_type'],
                            'nullable': column['is_nullable'] == 'YES',
                            'default': column['column_default'],
                            'description': column['description'] or ''
                        }
                
                # Build flat column cache for similarity comparisons
                self.column_cache = {}
                for table_name, table_info in self.table_cache.items():
                    for col_name, col_info in table_info['columns'].items():
                        key = f"{table_name}.{col_name}"
                        self.column_cache[key] = {
                            **col_info,
                            'table': table_name
                        }
        
        except Exception as e:
            logger.error(f"Error loading schema information: {e}")
            # Initialize empty caches on error
            self.table_cache = {}
            self.column_cache = {}
    
    async def analyze_schema_needs(self, context_text: str) -> Dict[str, Any]:
        """
        Analyze roleplay context to identify database schema needs.
        
        Args:
            context_text: Text describing the roleplay context/scenario
            
        Returns:
            Analysis result with detected needs and proposed schema
        """
        # Make sure we have fresh schema information
        await self.refresh_schema_cache()
        
        # Check rate limiting
        if not self._check_rate_limit():
            return {
                "error": "Rate limit exceeded for schema changes",
                "status": "rate_limited",
                "next_available": self._get_rate_limit_reset_time()
            }
        
        # Generate schema analysis using LLM
        from nyx.llm_integration import generate_text_completion
        
        schema_analysis = await generate_text_completion(
            system_prompt="""You are analyzing roleplay context to identify database schema needs.
            You will identify only truly necessary database schema elements, being conservative in 
            your recommendations. Only suggest new tables if the data genuinely cannot be stored in
            existing structures. Be specific and precise in your analysis.""",
            user_prompt=f"""
            Analyze this roleplay context and identify any database schema needs:
            
            {context_text}
            
            RULES:
            1. Be extremely conservative. Only suggest new tables or columns if absolutely necessary.
            2. Focus on persistent data that needs to be tracked long-term.
            3. Do not suggest tables for temporary or transient data.
            4. Do not suggest tables that would duplicate existing functionality.
            5. All tables you propose MUST start with "dynamic_" prefix.
            6. Provide clear, specific column definitions with appropriate data types.
            
            Generate a JSON response with these fields:
            1. "detected_needs": Array of detected data tracking needs (max 2-3)
            2. "proposed_schema": Object with tables and columns that should be created
               (format: {{{{ "table_name": {{{{ "column_name": {{{{ "type": "TEXT", "nullable": true, "description": "..." }}}}} }}}}}}})
            3. "reasoning": Explanation of why these schema changes are needed
            
            Format as valid JSON.
            """,
            temperature=0.2,
            max_tokens=800
        )
        
        # Parse and validate the proposal
        try:
            schema_proposal = json.loads(schema_analysis)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse schema analysis: {schema_analysis}")
            return {"error": "Failed to parse schema analysis", "status": "failed"}
        
        # Validate structure of response
        if not isinstance(schema_proposal.get("proposed_schema"), dict):
            return {"error": "Invalid schema proposal format", "status": "invalid_format"}
        
        # Find similar existing structures
        similarity_analysis = await self._find_similar_existing_structures(schema_proposal["proposed_schema"])
        
        # Optimize based on similarity analysis
        optimized_schema = await self._optimize_schema_proposal(
            schema_proposal["proposed_schema"], 
            similarity_analysis
        )
        
        # Update the proposal with optimized schema
        schema_proposal["original_proposed_schema"] = schema_proposal["proposed_schema"]
        schema_proposal["proposed_schema"] = optimized_schema
        schema_proposal["similarity_analysis"] = similarity_analysis
        
        # Check if we actually need to make any changes
        if not optimized_schema and not similarity_analysis.get("tables_to_extend"):
            schema_proposal["status"] = "no_changes_needed"
            schema_proposal["message"] = "Existing database structures are sufficient"
        else:
            schema_proposal["status"] = "changes_proposed"
        
        return schema_proposal
    
    async def implement_schema_changes(self, schema_proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement the proposed schema changes safely.
        
        Args:
            schema_proposal: Validated and optimized schema proposal
            
        Returns:
            Results of the schema changes
        """
        # Make sure we have an optimized proposal
        if "original_proposed_schema" not in schema_proposal:
            # Needs to go through optimization first
            return {"error": "Schema proposal has not been analyzed and optimized", "status": "invalid_proposal"}
        
        optimized_schema = schema_proposal["proposed_schema"]
        tables_to_extend = schema_proposal.get("similarity_analysis", {}).get("tables_to_extend", {})
        
        # Validate the optimized schema once more
        validation_result = await self._validate_schema_proposal(optimized_schema)
        if not validation_result["is_valid"]:
            return {"error": validation_result["reasons"], "status": "invalid_schema"}
        
        # Start implementing changes
        changes_made = {
            "tables_created": [],
            "columns_added": [],
            "tables_extended": [],
            "errors": []
        }
        
        try:
            async with get_db_connection_context() as conn:
                # Start a transaction
                async with conn.transaction():
                    # 1. Create new tables
                    for table_name, columns in optimized_schema.items():
                        result = await self._create_table(conn, table_name, columns)
                        if "error" in result:
                            changes_made["errors"].append(result["error"])
                        else:
                            changes_made["tables_created"].append(result)
                    
                    # 2. Extend existing tables
                    for table_name, columns in tables_to_extend.items():
                        result = await self._extend_table(conn, table_name, columns)
                        if "error" in result:
                            changes_made["errors"].append(result["error"])
                        else:
                            changes_made["tables_extended"].append(result)
                            changes_made["columns_added"].extend(
                                [f"{table_name}.{col}" for col in result["columns_added"]]
                            )
                    
                    # 3. Update the schema cache
                    await self._load_schema_information()
                    
                    # 4. Register schema changes for migrations
                    await register_schema_change({
                        "user_id": self.user_id,
                        "conversation_id": self.conversation_id,
                        "changes": changes_made,
                        "original_proposal": schema_proposal["original_proposed_schema"],
                        "optimized_proposal": optimized_schema,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # 5. Increment the change counter for rate limiting
                    self._increment_change_counter()
            
            if not changes_made["errors"]:
                changes_made["status"] = "success"
            else:
                changes_made["status"] = "partial_success" if (
                    changes_made["tables_created"] or 
                    changes_made["tables_extended"]
                ) else "failed"
            
            return changes_made
                
        except Exception as e:
            logger.error(f"Error implementing schema changes: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "tables_created": [],
                "columns_added": [],
                "tables_extended": []
            }
    
    async def _create_table(self, conn, table_name: str, columns: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new table in the database."""
        try:
            # Ensure table name starts with the dynamic prefix
            if not table_name.startswith(self.dynamic_table_prefix):
                return {"error": f"Table name must start with '{self.dynamic_table_prefix}'"}
            
            # Get table description
            table_description = columns.pop("table_description", f"Dynamically created table for {table_name}")
            
            # Build column definitions
            column_defs = []
            
            # Add standard columns first
            for col_name, col_def in self.standard_columns.items():
                col_type = col_def.get("type", "TEXT")
                is_nullable = "" if col_def.get("nullable", True) else "NOT NULL"
                default = f"DEFAULT {col_def.get('default')}" if "default" in col_def else ""
                column_defs.append(f"{col_name} {col_type} {is_nullable} {default}")
            
            # Add custom columns
            for col_name, col_def in columns.items():
                # Skip if it's a standard column or a metadata field
                if col_name in self.standard_columns or col_name == "table_description":
                    continue
                
                col_type = col_def.get("type", "TEXT")
                is_nullable = "" if col_def.get("nullable", True) else "NOT NULL"
                default = f"DEFAULT {col_def.get('default')}" if "default" in col_def else ""
                column_defs.append(f"{col_name} {col_type} {is_nullable} {default}")
            
            # Create the table
            create_table_sql = f"""
                CREATE TABLE {table_name} (
                    {', '.join(column_defs)},
                    PRIMARY KEY (id),
                    CONSTRAINT {table_name}_user_conversation_fk
                        FOREIGN KEY (user_id, conversation_id)
                        REFERENCES conversations(user_id, id)
                        ON DELETE CASCADE
                )
            """
            await conn.execute(create_table_sql)
            
            # Create index on user_id, conversation_id
            await conn.execute(f"""
                CREATE INDEX idx_{table_name}_user_conv
                ON {table_name}(user_id, conversation_id)
            """)
            
            # Add table comment/description
            await conn.execute(f"""
                COMMENT ON TABLE {table_name} IS $1
            """, table_description)
            
            # Add column comments
            for col_name, col_def in columns.items():
                if col_name == "table_description":
                    continue
                    
                if "description" in col_def:
                    await conn.execute(f"""
                        COMMENT ON COLUMN {table_name}.{col_name} IS $1
                    """, col_def["description"])
            
            return {
                "table_name": table_name,
                "columns": list(columns.keys()),
                "description": table_description
            }
            
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {e}")
            return {"error": f"Failed to create table {table_name}: {str(e)}"}
    
    async def _extend_table(self, conn, table_name: str, columns: Dict[str, Any]) -> Dict[str, Any]:
        """Add new columns to an existing table."""
        try:
            # Don't allow extending reserved tables
            if table_name.lower() in self.reserved_tables:
                return {"error": f"Cannot extend reserved table {table_name}"}
            
            # Get current columns to avoid duplicates
            current_columns = await conn.fetch("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'public' AND table_name = $1
            """, table_name)
            
            existing_columns = {row['column_name'].lower() for row in current_columns}
            
            # Track added columns
            columns_added = []
            
            # Add each column that doesn't exist
            for col_name, col_def in columns.items():
                if col_name.lower() in existing_columns:
                    continue  # Skip existing columns
                
                col_type = col_def.get("type", "TEXT")
                is_nullable = "" if col_def.get("nullable", True) else "NOT NULL"
                default = f"DEFAULT {col_def.get('default')}" if "default" in col_def else ""
                
                # Add the column
                await conn.execute(f"""
                    ALTER TABLE {table_name}
                    ADD COLUMN {col_name} {col_type} {is_nullable} {default}
                """)
                
                # Add column comment if provided
                if "description" in col_def:
                    await conn.execute(f"""
                        COMMENT ON COLUMN {table_name}.{col_name} IS $1
                    """, col_def["description"])
                
                columns_added.append(col_name)
            
            return {
                "table_name": table_name,
                "columns_added": columns_added
            }
            
        except Exception as e:
            logger.error(f"Error extending table {table_name}: {e}")
            return {"error": f"Failed to extend table {table_name}: {str(e)}"}
    
    async def _validate_schema_proposal(self, schema_proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a schema proposal for safety and feasibility.
        
        Args:
            schema_proposal: The proposed schema changes
            
        Returns:
            Validation result
        """
        issues = []
        
        # Check for required fields
        if not isinstance(schema_proposal, dict):
            issues.append("Schema proposal must be a dictionary")
            return {"is_valid": False, "reasons": issues}
        
        # Validate table names
        for table_name in schema_proposal.keys():
            # Check naming convention
            if not table_name.startswith(self.dynamic_table_prefix):
                issues.append(f"Table {table_name} must start with '{self.dynamic_table_prefix}'")
            
            # Check for SQL injection
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', table_name):
                issues.append(f"Invalid table name: {table_name}")
            
            # Check for reserved tables
            if table_name.lower() in self.reserved_tables:
                issues.append(f"Cannot create reserved table: {table_name}")
            
            # Check column definitions
            columns = schema_proposal[table_name]
            if not isinstance(columns, dict):
                issues.append(f"Columns for {table_name} must be a dictionary")
                continue
                
            for col_name, col_def in columns.items():
                # Skip metadata fields
                if col_name == "table_description":
                    continue
                    
                # Check for standard columns
                if col_name in self.standard_columns:
                    issues.append(f"Column {col_name} is a standard column and cannot be redefined")
                    continue
                
                # Check naming convention
                if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', col_name):
                    issues.append(f"Invalid column name: {col_name}")
                
                # Check column definition
                if not isinstance(col_def, dict):
                    issues.append(f"Column definition for {col_name} must be a dictionary")
                    continue
                
                # Validate column type
                valid_types = ["TEXT", "VARCHAR", "INTEGER", "BOOLEAN", "TIMESTAMP", 
                              "JSONB", "FLOAT", "NUMERIC", "DATE", "TIMESTAMPTZ"]
                col_type = col_def.get("type", "").upper()
                
                if col_type and col_type not in valid_types:
                    issues.append(f"Invalid column type: {col_type} for {col_name}")
        
        return {
            "is_valid": len(issues) == 0,
            "reasons": issues
        }
    
    async def _find_similar_existing_structures(
        self, 
        proposed_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Find existing tables or columns that might serve the same purpose as proposed structures.
        
        Args:
            proposed_schema: The proposed schema changes
            
        Returns:
            Dictionary mapping proposed tables/columns to existing alternatives
        """
        # Ensure schema cache is loaded
        if not self.table_cache:
            await self.refresh_schema_cache()
        
        results = {
            "tables": {},
            "columns": {},
            "tables_to_extend": {},
            "recommendations": []
        }
        
        # Generate embeddings for all existing tables
        existing_table_embeddings = {}
        for table_name, table_info in self.table_cache.items():
            # Skip system tables
            if (table_name.startswith('pg_') or 
                table_name.startswith('information_schema') or
                table_name in self.reserved_tables):
                continue
            
            # Create text representation of table
            table_text = f"{table_name} {table_info['description']}"
            column_texts = []
            
            for col_name, col_info in table_info['columns'].items():
                # Skip standard columns
                if col_name in self.standard_columns:
                    continue
                column_texts.append(f"{col_name} {col_info['description']}")
            
            # Combine table and column information
            full_text = table_text + " " + " ".join(column_texts)
            
            # Get embedding
            embedding = await get_embedding(full_text)
            existing_table_embeddings[table_name] = embedding
        
        # Compare each proposed table with existing tables
        for table_name, columns in proposed_schema.items():
            # Extract table description
            table_description = columns.get("table_description", "")
            
            # Create text representation of proposed table
            table_text = f"{table_name} {table_description}"
            column_texts = []
            
            for col_name, col_def in columns.items():
                if col_name == "table_description":
                    continue
                column_desc = col_def.get("description", "")
                column_texts.append(f"{col_name} {column_desc}")
            
            # Combine table and column information
            full_text = table_text + " " + " ".join(column_texts)
            
            # Get embedding for proposed table
            proposed_embedding = await get_embedding(full_text)
            
            # Find similar existing tables
            similar_tables = []
            for exist_table, exist_embedding in existing_table_embeddings.items():
                # Calculate cosine similarity
                similarity = self._calculate_similarity(proposed_embedding, exist_embedding)
                
                if similarity > self.medium_similarity:
                    # This is a similar table
                    similar_tables.append({
                        "table_name": exist_table,
                        "similarity": similarity,
                        "description": self.table_cache[exist_table]['description'],
                        "columns": list(self.table_cache[exist_table]['columns'].keys())
                    })
            
            # Sort by similarity
            similar_tables.sort(key=lambda x: x["similarity"], reverse=True)
            
            if similar_tables:
                results["tables"][table_name] = similar_tables
                
                # Check if best match is very similar
                best_match = similar_tables[0]
                if best_match["similarity"] > self.high_table_similarity:
                    # Check for column similarity
                    columns_to_add = {}
                    
                    for col_name, col_def in columns.items():
                        if col_name == "table_description":
                            continue
                            
                        # Skip if column already exists in the table
                        if col_name in self.table_cache[best_match["table_name"]]["columns"]:
                            continue
                        
                        # Check if there's a similar column already
                        col_desc = col_def.get("description", "")
                        col_text = f"{col_name} {col_desc}"
                        col_embedding = await get_embedding(col_text)
                        
                        has_similar_column = False
                        for exist_col in self.table_cache[best_match["table_name"]]["columns"]:
                            # Skip standard columns
                            if exist_col in self.standard_columns:
                                continue
                                
                            exist_col_info = self.table_cache[best_match["table_name"]]["columns"][exist_col]
                            exist_col_text = f"{exist_col} {exist_col_info['description']}"
                            exist_col_embedding = await get_embedding(exist_col_text)
                            
                            col_similarity = self._calculate_similarity(col_embedding, exist_col_embedding)
                            
                            if col_similarity > self.high_column_similarity:
                                has_similar_column = True
                                results["columns"][f"{table_name}.{col_name}"] = {
                                    "similar_column": f"{best_match['table_name']}.{exist_col}",
                                    "similarity": col_similarity
                                }
                                results["recommendations"].append(
                                    f"Use existing column '{best_match['table_name']}.{exist_col}' instead of creating '{table_name}.{col_name}'"
                                )
                                break
                        
                        if not has_similar_column:
                            # This is a new column to add to the existing table
                            columns_to_add[col_name] = col_def
                    
                    if columns_to_add:
                        results["tables_to_extend"][best_match["table_name"]] = columns_to_add
                        results["recommendations"].append(
                            f"Extend existing table '{best_match['table_name']}' with new columns instead of creating '{table_name}'"
                        )
                    else:
                        results["recommendations"].append(
                            f"Use existing table '{best_match['table_name']}' instead of creating '{table_name}' (all needed columns already exist)"
                        )
        
        return results
    
    async def _optimize_schema_proposal(
        self, 
        proposed_schema: Dict[str, Any], 
        similarity_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize a schema proposal based on similarity analysis.
        
        Args:
            proposed_schema: Original proposed schema
            similarity_analysis: Results of similarity analysis
            
        Returns:
            Optimized schema proposal
        """
        optimized_schema = {}
        
        # Process each proposed table
        for table_name, columns in proposed_schema.items():
            # Check if there's a very similar existing table
            if table_name in similarity_analysis["tables"]:
                similar_tables = similarity_analysis["tables"][table_name]
                
                if similar_tables and similar_tables[0]["similarity"] > self.high_table_similarity:
                    # Skip this table, we're extending an existing one instead
                    continue
            
            # No high similarity match, keep the proposed table
            optimized_schema[table_name] = columns
        
        return optimized_schema
    
    def _calculate_similarity(self, embedding1, embedding2) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not isinstance(embedding1, np.ndarray):
            embedding1 = np.array(embedding1)
        if not isinstance(embedding2, np.ndarray):
            embedding2 = np.array(embedding2)
            
        embedding1 = embedding1.reshape(1, -1)
        embedding2 = embedding2.reshape(1, -1)
        
        return cosine_similarity(embedding1, embedding2)[0][0]
    
    def _check_rate_limit(self) -> bool:
        """Check if schema changes are rate limited."""
        # Reset counter if it's a new day
        current_day = datetime.now().day
        last_day = self.last_change_reset.day
        
        if current_day != last_day:
            self.changes_today = 0
            self.last_change_reset = datetime.now()
        
        # Check if we've exceeded the limit
        return self.changes_today < self.max_daily_changes
    
    def _increment_change_counter(self) -> None:
        """Increment the schema change counter."""
        self.changes_today += 1
    
    def _get_rate_limit_reset_time(self) -> str:
        """Get the time when rate limit will reset."""
        # Reset at the start of the next day
        tomorrow = datetime.now() + timedelta(days=1)
        reset_time = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 0)
        return reset_time.isoformat()
