# memory/schemas.py

import logging
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
import asyncio
import openai
from logic.chatgpt_integration import get_openai_client, build_message_history, safe_json_loads, get_chatgpt_response

from .connection import TransactionContext, with_transaction
from .core import Memory, MemoryType, MemorySignificance, UnifiedMemoryManager

logger = logging.getLogger("memory_schemas")

class MemorySchemaManager:
    """
    Manages memory schemas - organized knowledge structures that influence
    memory encoding, storage, recall, and interpretation.
    
    Features:
    - Schema formation from experiences
    - Schema-based interpretation
    - Schema-guided recall
    - Schema evolution over time
    - Schema conflicts and resolution
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
    
    @with_transaction
    async def create_schema(self,
                          entity_type: str,
                          entity_id: int,
                          schema_name: str,
                          description: str,
                          category: str = "general",
                          attributes: Dict[str, Any] = None,
                          example_memory_ids: List[int] = None,
                          conn = None) -> Dict[str, Any]:
        """
        Create a memory schema.
        
        Args:
            entity_type: Type of entity that owns the schema
            entity_id: ID of the entity
            schema_name: Name of the schema
            description: Description of the schema
            category: Category for organizing schemas
            attributes: Key attributes of this schema
            example_memory_ids: IDs of memories that exemplify this schema
            
        Returns:
            Created schema information
        """
        # Format schema data
        schema_data = {
            "name": schema_name,
            "description": description,
            "category": category,
            "attributes": attributes or {},
            "example_memory_ids": example_memory_ids or [],
            "confidence": 0.7,  # Initial confidence
            "creation_date": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "usage_count": 0,
            "conflicts": [],
            "evolution": []
        }
        
        # Create the schema
        schema_id = await conn.fetchval("""
            INSERT INTO MemorySchemas (
                user_id, conversation_id, entity_type, entity_id,
                schema_name, category, schema_data, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP)
            RETURNING id
        """, self.user_id, self.conversation_id, entity_type, entity_id,
            schema_name, category, json.dumps(schema_data))
        
        # Update example memories to reference this schema
        if example_memory_ids:
            for memory_id in example_memory_ids:
                row = await conn.fetchrow("""
                    SELECT metadata
                    FROM unified_memories
                    WHERE id = $1
                      AND entity_type = $2
                      AND entity_id = $3
                      AND user_id = $4
                      AND conversation_id = $5
                """, memory_id, entity_type, entity_id, self.user_id, self.conversation_id)
                
                if row:
                    metadata = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
                    
                    if "schemas" not in metadata:
                        metadata["schemas"] = []
                        
                    metadata["schemas"].append({
                        "schema_id": schema_id,
                        "schema_name": schema_name,
                        "relevance": 1.0  # Full relevance for examples
                    })
                    
                    await conn.execute("""
                        UPDATE unified_memories
                        SET metadata = $1
                        WHERE id = $2
                    """, json.dumps(metadata), memory_id)
        
        return {
            "schema_id": schema_id,
            "schema_name": schema_name,
            "category": category,
            "description": description
        }
    
    @with_transaction
    async def detect_schema_from_memories(self,
                                       entity_type: str,
                                       entity_id: int,
                                       memory_ids: List[int] = None,
                                       tags: List[str] = None,
                                       min_memories: int = 3,
                                       conn = None) -> Dict[str, Any]:
        """
        Detect a potential schema from a set of memories.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            memory_ids: Specific memory IDs to analyze
            tags: Tags to filter memories by (if memory_ids not provided)
            min_memories: Minimum number of memories needed to form a schema
            
        Returns:
            Detected schema information or empty if no pattern found
        """
        # Get memories to analyze
        memories = []
        
        if memory_ids:
            # Get specific memories
            for memory_id in memory_ids:
                row = await conn.fetchrow("""
                    SELECT id, memory_text, tags
                    FROM unified_memories
                    WHERE id = $1
                      AND entity_type = $2
                      AND entity_id = $3
                      AND user_id = $4
                      AND conversation_id = $5
                """, memory_id, entity_type, entity_id, self.user_id, self.conversation_id)
                
                if row:
                    memories.append({
                        "id": row["id"],
                        "text": row["memory_text"],
                        "tags": row["tags"] or []
                    })
        else:
            # Get memories by tags
            tag_filter = tags or ["observation"]
            
            rows = await conn.fetch("""
                SELECT id, memory_text, tags
                FROM unified_memories
                WHERE entity_type = $1
                  AND entity_id = $2
                  AND user_id = $3
                  AND conversation_id = $4
                  AND tags && $5
                ORDER BY timestamp DESC
                LIMIT 20
            """, entity_type, entity_id, self.user_id, self.conversation_id, tag_filter)
            
            for row in rows:
                memories.append({
                    "id": row["id"],
                    "text": row["memory_text"],
                    "tags": row["tags"] or []
                })
        
        if len(memories) < min_memories:
            return {
                "schema_detected": False,
                "reason": f"Not enough memories to detect a schema (need {min_memories}, found {len(memories)})"
            }
            
        # Analyze memories to find a pattern
        schema = await self._detect_pattern(memories)
        
        if not schema:
            return {
                "schema_detected": False,
                "reason": "No clear pattern detected in the memories"
            }
            
        # Check if a similar schema already exists
        existing = await self.find_similar_schema(
            entity_type=entity_type,
            entity_id=entity_id,
            schema_name=schema["name"],
            description=schema["description"],
            conn=conn
        )
        
        if existing:
            # Update the existing schema instead of creating a new one
            updated = await self.update_schema(
                schema_id=existing["schema_id"],
                entity_type=entity_type,
                entity_id=entity_id,
                updates={
                    "confidence_change": 0.1,  # Increase confidence
                    "add_example_ids": [m["id"] for m in memories],
                    "attributes_update": schema["attributes"]
                },
                conn=conn
            )
            
            return {
                "schema_detected": True,
                "schema_already_exists": True,
                "schema_id": existing["schema_id"],
                "schema_name": existing["schema_name"],
                "confidence": updated["new_confidence"]
            }
            
        # Create a new schema
        schema_result = await self.create_schema(
            entity_type=entity_type,
            entity_id=entity_id,
            schema_name=schema["name"],
            description=schema["description"],
            category=schema["category"],
            attributes=schema["attributes"],
            example_memory_ids=[m["id"] for m in memories],
            conn=conn
        )
        
        return {
            "schema_detected": True,
            "schema_already_exists": False,
            "schema_id": schema_result["schema_id"],
            "schema_name": schema_result["schema_name"],
            "description": schema_result["description"],
            "category": schema_result["category"]
        }
    
    async def _detect_pattern(self, memories: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Detect a recurring pattern/schema in *memories* via Responses API.
        """
        client = get_openai_client()
    
        memories_text = "\n".join(
            f"Memory {i+1}: {m['text']}" for i, m in enumerate(memories)
        )
    
        prompt = f"""
        Analyze these memories to detect a recurring pattern or schema:
    
        {memories_text}
    
        Look for:
          1. Common situations / events
          2. Recurring elements or themes
          3. Consistent cause-effect relationships
          4. Patterns in interactions or outcomes
    
        If a clear pattern is found, output JSON:
        {{
          "pattern_found": true,
          "name": "Schema Name",
          "description": "Description",
          "category": "Category",
          "attributes": {{"key":"value"}}
        }}
    
        Otherwise output: {{"pattern_found": false}}
        """
    
        try:
            resp = await client.responses.create(
                model="gpt-4.1-nano",
                instructions="You detect patterns in memories to form schemas.",
                input=prompt,
                temperature=0.4,
                text={"format": {"type": "json_object"}},
            )
            result = json.loads(resp.output_text)
            if not result.get("pattern_found"):
                return None
            return {
                "name": result.get("name", "Unnamed Pattern"),
                "description": result.get("description", ""),
                "category": result.get("category", "general"),
                "attributes": result.get("attributes", {}),
            }
    
        except Exception as e:
            logger.error("Pattern detection failed: %s", e, exc_info=True)
            return None

    
    @with_transaction
    async def find_similar_schema(self,
                               entity_type: str,
                               entity_id: int,
                               schema_name: str,
                               description: str,
                               conn = None) -> Optional[Dict[str, Any]]:
        """
        Find a schema similar to the one described.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            schema_name: Name to compare
            description: Description to compare
            
        Returns:
            Similar schema if found, None otherwise
        """
        # Get all schemas for this entity
        rows = await conn.fetch("""
            SELECT id, schema_name, schema_data
            FROM MemorySchemas
            WHERE entity_type = $1
              AND entity_id = $2
              AND user_id = $3
              AND conversation_id = $4
        """, entity_type, entity_id, self.user_id, self.conversation_id)
        
        if not rows:
            return None
            
        # Check for name similarity
        for row in rows:
            if self._text_similarity(schema_name, row["schema_name"]) > 0.7:
                return {
                    "schema_id": row["id"],
                    "schema_name": row["schema_name"],
                    "schema_data": row["schema_data"] if isinstance(row["schema_data"], dict) else json.loads(row["schema_data"])
                }
                
        # Check for description similarity
        for row in rows:
            schema_data = row["schema_data"] if isinstance(row["schema_data"], dict) else json.loads(row["schema_data"])
            existing_description = schema_data.get("description", "")
            
            if self._text_similarity(description, existing_description) > 0.7:
                return {
                    "schema_id": row["id"],
                    "schema_name": row["schema_name"],
                    "schema_data": schema_data
                }
                
        return None
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity.
        
        Returns:
            Similarity score from 0.0 to 1.0
        """
        # Simple word overlap for now
        # In a production system, you'd use embeddings or more sophisticated methods
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return overlap / union
    
    @with_transaction
    async def update_schema(self,
                         schema_id: int,
                         entity_type: str,
                         entity_id: int,
                         updates: Dict[str, Any],
                         conn = None) -> Dict[str, Any]:
        """
        Update an existing schema.
        
        Args:
            schema_id: ID of the schema to update
            entity_type: Type of entity
            entity_id: ID of the entity
            updates: Dict of updates to apply
            
        Returns:
            Updated schema information
        """
        # Get current schema data
        row = await conn.fetchrow("""
            SELECT schema_name, schema_data
            FROM MemorySchemas
            WHERE id = $1
              AND entity_type = $2
              AND entity_id = $3
              AND user_id = $4
              AND conversation_id = $5
        """, schema_id, entity_type, entity_id, self.user_id, self.conversation_id)
        
        if not row:
            return {"error": f"Schema {schema_id} not found"}
            
        schema_name = row["schema_name"]
        schema_data = row["schema_data"] if isinstance(row["schema_data"], dict) else json.loads(row["schema_data"])
        
        # Track evolution
        if "evolution" not in schema_data:
            schema_data["evolution"] = []
            
        evolution_entry = {
            "date": datetime.now().isoformat(),
            "changes": []
        }
        
        # Apply name update
        if "name" in updates:
            old_name = schema_data["name"]
            schema_data["name"] = updates["name"]
            evolution_entry["changes"].append({
                "field": "name",
                "old_value": old_name,
                "new_value": updates["name"]
            })
            
            # Also update the schema_name field
            schema_name = updates["name"]
            await conn.execute("""
                UPDATE MemorySchemas
                SET schema_name = $1
                WHERE id = $2
            """, schema_name, schema_id)
            
        # Apply description update
        if "description" in updates:
            old_desc = schema_data["description"]
            schema_data["description"] = updates["description"]
            evolution_entry["changes"].append({
                "field": "description",
                "old_value": old_desc,
                "new_value": updates["description"]
            })
            
        # Apply category update
        if "category" in updates:
            old_category = schema_data["category"]
            schema_data["category"] = updates["category"]
            evolution_entry["changes"].append({
                "field": "category",
                "old_value": old_category,
                "new_value": updates["category"]
            })
            
            # Also update the category field
            await conn.execute("""
                UPDATE MemorySchemas
                SET category = $1
                WHERE id = $2
            """, updates["category"], schema_id)
            
        # Apply confidence change
        old_confidence = schema_data.get("confidence", 0.5)
        if "confidence_change" in updates:
            # Adjust confidence (bounded 0.1 to 1.0)
            new_confidence = max(0.1, min(1.0, old_confidence + updates["confidence_change"]))
            schema_data["confidence"] = new_confidence
            evolution_entry["changes"].append({
                "field": "confidence",
                "old_value": old_confidence,
                "new_value": new_confidence
            })
        else:
            new_confidence = old_confidence
            
        # Update attributes
        if "attributes_update" in updates:
            old_attrs = schema_data.get("attributes", {}).copy()
            
            # Apply updates to attributes
            if not "attributes" in schema_data:
                schema_data["attributes"] = {}
                
            for key, value in updates["attributes_update"].items():
                old_value = schema_data["attributes"].get(key)
                schema_data["attributes"][key] = value
                
                evolution_entry["changes"].append({
                    "field": f"attribute.{key}",
                    "old_value": old_value,
                    "new_value": value
                })
                
        # Update example memories
        if "add_example_ids" in updates:
            if "example_memory_ids" not in schema_data:
                schema_data["example_memory_ids"] = []
                
            # Add new examples, avoiding duplicates
            for memory_id in updates["add_example_ids"]:
                if memory_id not in schema_data["example_memory_ids"]:
                    schema_data["example_memory_ids"].append(memory_id)
                    
                    # Also update the memory to reference this schema
                    row = await conn.fetchrow("""
                        SELECT metadata
                        FROM unified_memories
                        WHERE id = $1
                          AND entity_type = $2
                          AND entity_id = $3
                          AND user_id = $4
                          AND conversation_id = $5
                    """, memory_id, entity_type, entity_id, self.user_id, self.conversation_id)
                    
                    if row:
                        metadata = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
                        
                        if "schemas" not in metadata:
                            metadata["schemas"] = []
                            
                        # Check if this schema is already referenced
                        schema_refs = [s for s in metadata.get("schemas", []) if s.get("schema_id") == schema_id]
                        
                        if not schema_refs:
                            metadata["schemas"].append({
                                "schema_id": schema_id,
                                "schema_name": schema_name,
                                "relevance": 1.0  # Full relevance for examples
                            })
                            
                            await conn.execute("""
                                UPDATE unified_memories
                                SET metadata = $1
                                WHERE id = $2
                            """, json.dumps(metadata), memory_id)
            
            evolution_entry["changes"].append({
                "field": "example_memory_ids",
                "action": "added",
                "added_ids": updates["add_example_ids"]
            })
                    
        if "remove_example_ids" in updates:
            if "example_memory_ids" in schema_data:
                for memory_id in updates["remove_example_ids"]:
                    if memory_id in schema_data["example_memory_ids"]:
                        schema_data["example_memory_ids"].remove(memory_id)
                        
                evolution_entry["changes"].append({
                    "field": "example_memory_ids",
                    "action": "removed",
                    "removed_ids": updates["remove_example_ids"]
                })
        
        # Add conflict if specified
        if "add_conflict" in updates:
            if "conflicts" not in schema_data:
                schema_data["conflicts"] = []
                
            schema_data["conflicts"].append({
                "description": updates["add_conflict"]["description"],
                "conflicting_memory_id": updates["add_conflict"].get("memory_id"),
                "date": datetime.now().isoformat(),
                "resolution": updates["add_conflict"].get("resolution")
            })
            
            evolution_entry["changes"].append({
                "field": "conflicts",
                "action": "added",
                "conflict_description": updates["add_conflict"]["description"]
            })
        
        # Increment usage count
        schema_data["usage_count"] = schema_data.get("usage_count", 0) + 1
        
        # Update last_updated timestamp
        schema_data["last_updated"] = datetime.now().isoformat()
        
        # Add evolution entry if there were changes
        if evolution_entry["changes"]:
            schema_data["evolution"].append(evolution_entry)
        
        # Save the updated schema
        await conn.execute("""
            UPDATE MemorySchemas
            SET schema_name = $1,
                schema_data = $2,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $3
        """, schema_name, json.dumps(schema_data), schema_id)
        
        return {
            "schema_id": schema_id,
            "schema_name": schema_name,
            "new_confidence": new_confidence,
            "changes_made": len(evolution_entry["changes"]),
            "updates": evolution_entry["changes"]
        }
    
    @with_transaction
    async def apply_schema_to_memory(self,
                                  memory_id: int,
                                  entity_type: str,
                                  entity_id: int,
                                  schema_id: int = None,
                                  auto_detect: bool = False,
                                  conn = None) -> Dict[str, Any]:
        """
        Apply a schema to interpret a memory.
        
        Args:
            memory_id: ID of the memory
            entity_type: Type of entity
            entity_id: ID of the entity
            schema_id: ID of the schema to apply (if specific)
            auto_detect: Whether to automatically detect applicable schemas
            
        Returns:
            Schema application results
        """
        # Get the memory
        row = await conn.fetchrow("""
            SELECT memory_text, metadata
            FROM unified_memories
            WHERE id = $1
              AND entity_type = $2
              AND entity_id = $3
              AND user_id = $4
              AND conversation_id = $5
        """, memory_id, entity_type, entity_id, self.user_id, self.conversation_id)
        
        if not row:
            return {"error": f"Memory {memory_id} not found"}
            
        memory_text = row["memory_text"]
        metadata = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
        
        # Initialize schemas field if needed
        if "schemas" not in metadata:
            metadata["schemas"] = []
            
        applied_schemas = []
        
        # If a specific schema ID is provided
        if schema_id:
            schema_row = await conn.fetchrow("""
                SELECT schema_name, schema_data
                FROM MemorySchemas
                WHERE id = $1
                  AND entity_type = $2
                  AND entity_id = $3
                  AND user_id = $4
                  AND conversation_id = $5
            """, schema_id, entity_type, entity_id, self.user_id, self.conversation_id)
            
            if not schema_row:
                return {"error": f"Schema {schema_id} not found"}
                
            schema_name = schema_row["schema_name"]
            schema_data = schema_row["schema_data"] if isinstance(schema_row["schema_data"], dict) else json.loads(schema_row["schema_data"])
            
            # Check if this schema is already applied
            existing_schemas = [s for s in metadata["schemas"] if s.get("schema_id") == schema_id]
            
            if existing_schemas:
                # Just update relevance
                existing_schemas[0]["relevance"] = 1.0
            else:
                # Apply the schema
                metadata["schemas"].append({
                    "schema_id": schema_id,
                    "schema_name": schema_name,
                    "relevance": 1.0,
                    "applied_date": datetime.now().isoformat()
                })
                
            # Update schema's example memories if this is a good example
            if "example_memory_ids" not in schema_data:
                schema_data["example_memory_ids"] = []
                
            if memory_id not in schema_data["example_memory_ids"]:
                schema_data["example_memory_ids"].append(memory_id)
                
                await conn.execute("""
                    UPDATE MemorySchemas
                    SET schema_data = $1
                    WHERE id = $2
                """, json.dumps(schema_data), schema_id)
                
            applied_schemas.append({
                "schema_id": schema_id,
                "schema_name": schema_name,
                "relevance": 1.0
            })
            
        # Auto-detect applicable schemas
        if auto_detect:
            # Get all available schemas
            schema_rows = await conn.fetch("""
                SELECT id, schema_name, schema_data
                FROM MemorySchemas
                WHERE entity_type = $1
                  AND entity_id = $2
                  AND user_id = $3
                  AND conversation_id = $4
            """, entity_type, entity_id, self.user_id, self.conversation_id)
            
            # Evaluate each schema
            for row in schema_rows:
                schema_id = row["id"]
                schema_name = row["schema_name"]
                schema_data = row["schema_data"] if isinstance(row["schema_data"], dict) else json.loads(row["schema_data"])
                
                # Skip if this schema was already explicitly applied
                if any(s.get("schema_id") == schema_id for s in applied_schemas):
                    continue
                    
                # Calculate relevance
                relevance = await self._calculate_schema_relevance(memory_text, schema_data)
                
                if relevance > 0.5:  # Threshold for applying schema
                    # Check if this schema is already applied
                    existing_schemas = [s for s in metadata["schemas"] if s.get("schema_id") == schema_id]
                    
                    if existing_schemas:
                        # Update relevance if higher
                        if relevance > existing_schemas[0].get("relevance", 0):
                            existing_schemas[0]["relevance"] = relevance
                    else:
                        # Apply the schema
                        metadata["schemas"].append({
                            "schema_id": schema_id,
                            "schema_name": schema_name,
                            "relevance": relevance,
                            "applied_date": datetime.now().isoformat(),
                            "auto_detected": True
                        })
                        
                    applied_schemas.append({
                        "schema_id": schema_id,
                        "schema_name": schema_name,
                        "relevance": relevance,
                        "auto_detected": True
                    })
                    
                    # Only consider as an example if highly relevant
                    if relevance > 0.8:
                        if "example_memory_ids" not in schema_data:
                            schema_data["example_memory_ids"] = []
                            
                        if memory_id not in schema_data["example_memory_ids"]:
                            schema_data["example_memory_ids"].append(memory_id)
                            
                            await conn.execute("""
                                UPDATE MemorySchemas
                                SET schema_data = $1
                                WHERE id = $2
                            """, json.dumps(schema_data), schema_id)
        
        # Update the memory metadata
        await conn.execute("""
            UPDATE unified_memories
            SET metadata = $1
            WHERE id = $2
        """, json.dumps(metadata), memory_id)
        
        return {
            "memory_id": memory_id,
            "applied_schemas": applied_schemas,
            "total_schemas_applied": len(applied_schemas)
        }
    
    async def _calculate_schema_relevance(
        self,
        memory_text: str,
        schema_data: Dict[str, Any],
    ) -> float:
        """
        Relevance score (0.0-1.0) of *schema_data* to *memory_text*.
        """
        client = get_openai_client()
    
        schema_name = schema_data.get("name", "")
        schema_description = schema_data.get("description", "")
        attributes = schema_data.get("attributes", {})
        attributes_text = "".join(f"- {k}: {v}\n" for k, v in attributes.items())
    
        prompt = f"""
        MEMORY:
        {memory_text}
    
        SCHEMA:
        Name: {schema_name}
        Description: {schema_description}
        Attributes:
        {attributes_text}
    
        Rate relevance 0.0-1.0 (0 = irrelevant, 1 = perfect).
        Return ONLY the number.
        """
    
        try:
            resp = await client.responses.create(
                model="gpt-4.1-nano",
                instructions="You rate memory/schema relevance.",
                input=prompt,
                temperature=0.1
            )
            score = float(resp.output_text.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.error("Relevance calc failed: %s", e)
            # simple word-overlap fallback
            schema_words = set((schema_name + " " + schema_description).lower().split())
            memory_words = set(memory_text.lower().split())
            if not schema_words:
                return 0.0
            return len(schema_words & memory_words) / len(schema_words)
    
    @with_transaction
    async def interpret_memory_with_schemas(self,
                                         memory_id: int,
                                         entity_type: str,
                                         entity_id: int,
                                         conn = None) -> Dict[str, Any]:
        """
        Generate a rich interpretation of a memory using applicable schemas.
        
        Args:
            memory_id: ID of the memory to interpret
            entity_type: Type of entity
            entity_id: ID of the entity
            
        Returns:
            Schema-based memory interpretation
        """
        # Get the memory
        row = await conn.fetchrow("""
            SELECT memory_text, metadata
            FROM unified_memories
            WHERE id = $1
              AND entity_type = $2
              AND entity_id = $3
              AND user_id = $4
              AND conversation_id = $5
        """, memory_id, entity_type, entity_id, self.user_id, self.conversation_id)
        
        if not row:
            return {"error": f"Memory {memory_id} not found"}
            
        memory_text = row["memory_text"]
        metadata = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
        
        # Check if schemas have been applied
        applied_schemas = metadata.get("schemas", [])
        
        if not applied_schemas:
            # Auto-detect schemas if none applied
            result = await self.apply_schema_to_memory(
                memory_id=memory_id,
                entity_type=entity_type,
                entity_id=entity_id,
                auto_detect=True,
                conn=conn
            )
            
            # Refresh memory data after schema application
            row = await conn.fetchrow("""
                SELECT memory_text, metadata
                FROM unified_memories
                WHERE id = $1
            """, memory_id)
            
            if not row:
                return {"error": "Memory not found after schema application"}
                
            metadata = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
            applied_schemas = metadata.get("schemas", [])
        
        if not applied_schemas:
            return {
                "memory_id": memory_id,
                "interpretation": "This memory doesn't fit any known patterns or schemas."
            }
            
        # Get schema details for interpretation
        schema_details = []
        for schema_ref in applied_schemas:
            schema_id = schema_ref.get("schema_id")
            relevance = schema_ref.get("relevance", 0.0)
            
            if schema_id and relevance > 0.3:  # Only use moderately relevant schemas
                schema_row = await conn.fetchrow("""
                    SELECT schema_name, schema_data
                    FROM MemorySchemas
                    WHERE id = $1
                """, schema_id)
                
                if schema_row:
                    schema_name = schema_row["schema_name"]
                    schema_data = schema_row["schema_data"] if isinstance(schema_row["schema_data"], dict) else json.loads(schema_row["schema_data"])
                    
                    schema_details.append({
                        "id": schema_id,
                        "name": schema_name,
                        "description": schema_data.get("description", ""),
                        "relevance": relevance,
                        "attributes": schema_data.get("attributes", {}),
                        "confidence": schema_data.get("confidence", 0.5)
                    })
        
        if not schema_details:
            return {
                "memory_id": memory_id,
                "interpretation": "This memory doesn't strongly match any known patterns or schemas."
            }
            
        # Generate an interpretation based on the schemas
        interpretation = await self._generate_schema_interpretation(memory_text, schema_details)
        
        # Store the interpretation if successful
        if interpretation:
            if "interpretations" not in metadata:
                metadata["interpretations"] = []
                
            metadata["interpretations"].append({
                "text": interpretation,
                "based_on_schemas": [s["id"] for s in schema_details],
                "generated_at": datetime.now().isoformat()
            })
            
            await conn.execute("""
                UPDATE unified_memories
                SET metadata = $1
                WHERE id = $2
            """, json.dumps(metadata), memory_id)
        
        return {
            "memory_id": memory_id,
            "memory_text": memory_text,
            "interpretation": interpretation,
            "applied_schemas": [
                {
                    "id": s["id"],
                    "name": s["name"],
                    "relevance": s["relevance"]
                }
                for s in schema_details
            ]
        }
    
    async def _generate_schema_interpretation(
        self,
        memory_text: str,
        schema_details: List[Dict[str, Any]],
    ) -> str:
        """
        Produce a 100-150 word interpretation using applicable schemas.
        """
        client = get_openai_client()
    
        schemas_text = ""
        for i, s in enumerate(schema_details):
            schemas_text += (
                f"Schema {i+1}: {s['name']} (relevance {s['relevance']:.2f})\n"
                f"{s['description']}\n\n"
            )
    
        prompt = f"""
        MEMORY:
        {memory_text}
    
        APPLICABLE SCHEMAS:
        {schemas_text}
    
        Write a concise (100-150 words) interpretation that:
          • Explains fit with schemas
          • Weighs each by relevance
          • Notes any deviations
        Return ONLY the interpretation text.
        """
    
        try:
            resp = await client.responses.create(
                model="gpt-4.1-nano",
                instructions="You interpret memories through cognitive schemas.",
                input=prompt,
                temperature=0.4
            )
            return resp.output_text.strip()
    
        except Exception as e:
            logger.error("Interpretation failed: %s", e)
            highest = max(schema_details, key=lambda x: x["relevance"])
            return (
                f"This memory seems to illustrate the '{highest['name']}' pattern. "
                f"{highest['description']}"
            )
        
    @with_transaction
    async def find_schema_conflicting_memories(self,
                                            schema_id: int,
                                            entity_type: str,
                                            entity_id: int,
                                            conn = None) -> Dict[str, Any]:
        """
        Find memories that conflict with a schema.
        This is useful for detecting when a schema needs to evolve.
        
        Args:
            schema_id: ID of the schema
            entity_type: Type of entity
            entity_id: ID of the entity
            
        Returns:
            Conflicting memories information
        """
        # Get the schema
        row = await conn.fetchrow("""
            SELECT schema_name, schema_data
            FROM MemorySchemas
            WHERE id = $1
              AND entity_type = $2
              AND entity_id = $3
              AND user_id = $4
              AND conversation_id = $5
        """, schema_id, entity_type, entity_id, self.user_id, self.conversation_id)
        
        if not row:
            return {"error": f"Schema {schema_id} not found"}
            
        schema_name = row["schema_name"]
        schema_data = row["schema_data"] if isinstance(row["schema_data"], dict) else json.loads(row["schema_data"])
        
        # Get example memories
        example_ids = schema_data.get("example_memory_ids", [])
        if not example_ids:
            return {
                "schema_id": schema_id,
                "schema_name": schema_name,
                "conflicts_found": False,
                "reason": "No example memories associated with this schema"
            }
            
        # Get all memories with this schema applied
        rows = await conn.fetch("""
            SELECT id, memory_text, metadata
            FROM unified_memories
            WHERE entity_type = $1
              AND entity_id = $2
              AND user_id = $3
              AND conversation_id = $4
              AND jsonb_path_exists(metadata, '$.schemas[*].schema_id ? (@ == $5)')
        """, entity_type, entity_id, self.user_id, self.conversation_id, schema_id)
        
        if not rows:
            return {
                "schema_id": schema_id,
                "schema_name": schema_name,
                "conflicts_found": False,
                "reason": "No memories associated with this schema"
            }
            
        schema_memories = []
        for row in rows:
            memory_id = row["id"]
            memory_text = row["memory_text"]
            metadata = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
            
            # Extract schema relevance
            relevance = 0.0
            for schema_ref in metadata.get("schemas", []):
                if schema_ref.get("schema_id") == schema_id:
                    relevance = schema_ref.get("relevance", 0.0)
                    break
                    
            schema_memories.append({
                "id": memory_id,
                "text": memory_text,
                "relevance": relevance
            })
            
        # Look for potential conflicts
        conflicting_memories = []
        for memory in schema_memories:
            # Memories with low relevance but still classified might be conflicts
            if memory["relevance"] < 0.4:
                conflict_score = await self._calculate_conflict_score(memory["text"], schema_data)
                
                if conflict_score > 0.6:  # High conflict
                    conflicting_memories.append({
                        "id": memory["id"],
                        "text": memory["text"],
                        "conflict_score": conflict_score,
                        "relevance": memory["relevance"]
                    })
        
        if not conflicting_memories:
            return {
                "schema_id": schema_id,
                "schema_name": schema_name,
                "conflicts_found": False,
                "reason": "No conflicting memories found"
            }
            
        # Generate conflict explanations
        for memory in conflicting_memories:
            memory["conflict_explanation"] = await self._explain_schema_conflict(
                memory["text"], schema_data
            )
        
        return {
            "schema_id": schema_id,
            "schema_name": schema_name,
            "conflicts_found": True,
            "conflicting_memories": conflicting_memories,
            "schema_description": schema_data.get("description", ""),
            "schema_confidence": schema_data.get("confidence", 0.5)
        }
    
    async def _calculate_conflict_score(
        self,
        memory_text: str,
        schema_data: Dict[str, Any],
    ) -> float:
        """
        Conflict score 0.0-1.0 (1 = total contradiction).
        """
        client = get_openai_client()
    
        schema_name = schema_data.get("name", "")
        schema_description = schema_data.get("description", "")
        attributes = schema_data.get("attributes", {})
        attributes_text = "".join(f"- {k}: {v}\n" for k, v in attributes.items())
    
        prompt = f"""
        MEMORY:
        {memory_text}
    
        SCHEMA:
        Name: {schema_name}
        Description: {schema_description}
        Attributes:
        {attributes_text}
    
        Rate conflict (0.0-1.0). Return ONLY the number.
        """
    
        try:
            resp = await client.responses.create(
                model="gpt-4.1-nano",
                instructions="You calculate schema/memory conflict.",
                input=prompt,
                temperature=0.1
            )
            score = float(resp.output_text.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.error("Conflict calc failed: %s", e)
            return 0.0
    
    async def _explain_schema_conflict(
        self,
        memory_text: str,
        schema_data: Dict[str, Any],
    ) -> str:
        """
        Brief (1-2 sentence) explanation of conflict.
        """
        client = get_openai_client()
    
        prompt = f"""
        MEMORY:
        {memory_text}
    
        SCHEMA:
        {schema_data.get('name', '')}: {schema_data.get('description', '')}
    
        Explain in 1-2 sentences how the memory conflicts with the schema.
        Return ONLY the explanation.
        """
    
        try:
            resp = await client.responses.create(
                model="gpt-4.1-nano",
                instructions="You explain memory/schema conflicts.",
                input=prompt,
                temperature=0.4
            )
            return resp.output_text.strip()
        except Exception as e:
            logger.error("Conflict explanation failed: %s", e)
            return (
                f"This memory contradicts aspects of the '{schema_data.get('name','')}' schema."
            )
    
    @with_transaction
    async def evolve_schema_from_conflicts(self,
                                        schema_id: int,
                                        entity_type: str,
                                        entity_id: int,
                                        conflicts: List[Dict[str, Any]],
                                        conn = None) -> Dict[str, Any]:
        """
        Evolve a schema based on conflicting memories.
        
        Args:
            schema_id: ID of the schema to evolve
            entity_type: Type of entity
            entity_id: ID of the entity
            conflicts: List of conflicting memories with explanations
            
        Returns:
            Results of schema evolution
        """
        # Get the schema
        row = await conn.fetchrow("""
            SELECT schema_name, schema_data
            FROM MemorySchemas
            WHERE id = $1
              AND entity_type = $2
              AND entity_id = $3
              AND user_id = $4
              AND conversation_id = $5
        """, schema_id, entity_type, entity_id, self.user_id, self.conversation_id)
        
        if not row:
            return {"error": f"Schema {schema_id} not found"}
            
        schema_name = row["schema_name"]
        schema_data = row["schema_data"] if isinstance(row["schema_data"], dict) else json.loads(row["schema_data"])
        
        if not conflicts:
            return {
                "schema_id": schema_id,
                "schema_name": schema_name,
                "evolved": False,
                "reason": "No conflicts provided"
            }
            
        # Generate evolved schema based on conflicts
        evolved_schema = await self._generate_evolved_schema(schema_data, conflicts)
        
        if not evolved_schema:
            return {
                "schema_id": schema_id,
                "schema_name": schema_name,
                "evolved": False,
                "reason": "Failed to generate evolved schema"
            }
            
        # Apply the evolution
        updates = {
            "description": evolved_schema.get("description", schema_data.get("description", "")),
            "attributes_update": evolved_schema.get("attributes", {}),
            "confidence_change": -0.1  # Reduce confidence as schema has evolved
        }
        
        # Record conflicts
        for conflict in conflicts:
            updates["add_conflict"] = {
                "description": conflict.get("conflict_explanation", "Unexplained conflict"),
                "memory_id": conflict.get("id"),
                "resolution": evolved_schema.get("resolution", "Schema evolved to accommodate this exception")
            }
            
            # Only add one conflict per update
            break
            
        # Apply updates
        result = await self.update_schema(
            schema_id=schema_id,
            entity_type=entity_type,
            entity_id=entity_id,
            updates=updates,
            conn=conn
        )
        
        return {
            "schema_id": schema_id,
            "schema_name": schema_name,
            "evolved": True,
            "old_description": schema_data.get("description", ""),
            "new_description": evolved_schema.get("description", ""),
            "changes": result.get("updates", [])
        }
    
    async def _generate_evolved_schema(
        self,
        schema_data: Dict[str, Any],
        conflicts: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Produce an evolved schema JSON that resolves *conflicts*.
        """
        client = get_openai_client()
    
        # build prompt pieces
        attrs = "\n".join(f"- {k}: {v}" for k, v in schema_data.get("attributes", {}).items())
        conflicts_text = ""
        for i, c in enumerate(conflicts):
            conflicts_text += (
                f"Conflict {i+1}:\nMemory: {c.get('text','')}\n"
                f"Explanation: {c.get('conflict_explanation','')}\n\n"
            )
    
        prompt = f"""
        CURRENT SCHEMA:
        Name: {schema_data.get('name','')}
        Description: {schema_data.get('description','')}
        Attributes:
        {attrs}
    
        CONFLICTING MEMORIES:
        {conflicts_text}
    
        Produce evolved schema JSON:
        {{
          "description": "...",
          "attributes": {{}},
          "resolution": "..."
        }}
        Return ONLY JSON.
        """
    
        try:
            resp = await client.responses.create(
                model="gpt-4.1-nano",
                instructions="You evolve schemas to handle conflicts.",
                input=prompt,
                temperature=0.4,
                text={"format": {"type": "json_object"}},
            )
            return json.loads(resp.output_text)
        except Exception as e:
            logger.error("Evolved schema failed: %s", e)
            return None
        
    @with_transaction
    async def merge_schemas(self,
                         entity_type: str,
                         entity_id: int,
                         schema_ids: List[int],
                         conn = None) -> Dict[str, Any]:
        """
        Merge multiple schemas into a single more comprehensive schema.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            schema_ids: IDs of schemas to merge
            
        Returns:
            Merged schema information
        """
        if len(schema_ids) < 2:
            return {"error": "At least two schemas are required for merging"}
            
        # Get the schemas
        schemas = []
        for schema_id in schema_ids:
            row = await conn.fetchrow("""
                SELECT id, schema_name, schema_data
                FROM MemorySchemas
                WHERE id = $1
                  AND entity_type = $2
                  AND entity_id = $3
                  AND user_id = $4
                  AND conversation_id = $5
            """, schema_id, entity_type, entity_id, self.user_id, self.conversation_id)
            
            if row:
                schema_data = row["schema_data"] if isinstance(row["schema_data"], dict) else json.loads(row["schema_data"])
                
                schemas.append({
                    "id": row["id"],
                    "name": row["schema_name"],
                    "data": schema_data
                })
                
        if len(schemas) < 2:
            return {"error": "At least two valid schemas are required for merging"}
            
        # Generate merged schema
        merged_schema = await self._generate_merged_schema(schemas)
        
        if not merged_schema:
            return {"error": "Failed to generate merged schema"}
            
        # Create the new schema
        result = await self.create_schema(
            entity_type=entity_type,
            entity_id=entity_id,
            schema_name=merged_schema.get("name", "Merged Schema"),
            description=merged_schema.get("description", ""),
            category=merged_schema.get("category", "general"),
            attributes=merged_schema.get("attributes", {}),
            conn=conn
        )
        
        merged_schema_id = result["schema_id"]
        
        # Collect all example memory IDs from source schemas
        all_example_ids = []
        for schema in schemas:
            example_ids = schema["data"].get("example_memory_ids", [])
            all_example_ids.extend(example_ids)
            
        # Update the merged schema with example memories
        if all_example_ids:
            await self.update_schema(
                schema_id=merged_schema_id,
                entity_type=entity_type,
                entity_id=entity_id,
                updates={
                    "add_example_ids": all_example_ids
                },
                conn=conn
            )
            
        # Add relationship to source schemas
        for schema in schemas:
            source_id = schema["id"]
            
            # Update metadata to indicate this schema was merged
            schema_data = schema["data"]
            
            if "merges" not in schema_data:
                schema_data["merges"] = []
                
            schema_data["merges"].append({
                "merged_into": merged_schema_id,
                "merged_schema_name": merged_schema.get("name", "Merged Schema"),
                "date": datetime.now().isoformat()
            })
            
            # Save the updated schema data
            await conn.execute("""
                UPDATE MemorySchemas
                SET schema_data = $1
                WHERE id = $2
            """, json.dumps(schema_data), source_id)
        
        return {
            "merged_schema_id": merged_schema_id,
            "merged_schema_name": merged_schema.get("name", "Merged Schema"),
            "source_schema_ids": [s["id"] for s in schemas],
            "source_schema_names": [s["name"] for s in schemas]
        }
    
    async def _generate_merged_schema(
        self,
        schemas: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Merge multiple schemas into one comprehensive schema.
        """
        client = get_openai_client()
    
        schemas_text = ""
        for i, s in enumerate(schemas):
            data = s["data"]
            schemas_text += (
                f"Schema {i+1}: {s['name']}\n"
                f"Description: {data.get('description','')}\n"
                "Attributes:\n"
                + "".join(f"- {k}: {v}\n" for k, v in data.get("attributes", {}).items())
                + "\n"
            )
    
        prompt = f"""
        Merge these schemas into one comprehensive schema:
    
        {schemas_text}
    
        Output JSON:
        {{
          "name": "...",
          "description": "...",
          "category": "...",
          "attributes": {{}},
          "integration_notes": "..."
        }}
        """
    
        try:
            resp = await client.responses.create(
                model="gpt-4.1-nano",
                instructions="You merge schemas.",
                input=prompt,
                temperature=0.4,
                text={"format": {"type": "json_object"}},
            )
            return json.loads(resp.output_text)
        except Exception as e:
            logger.error("Merged schema failed: %s", e)
            return None
    
    @with_transaction
    async def run_schema_maintenance(self,
                                  entity_type: str,
                                  entity_id: int,
                                  conn = None) -> Dict[str, Any]:
        """
        Run maintenance tasks for schemas.
        This includes conflict detection, schema merging suggestions, etc.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            
        Returns:
            Results of maintenance operations
        """
        # Get all schemas for this entity
        rows = await conn.fetch("""
            SELECT id, schema_name, category, schema_data, created_at
            FROM MemorySchemas
            WHERE entity_type = $1
              AND entity_id = $2
              AND user_id = $3
              AND conversation_id = $4
        """, entity_type, entity_id, self.user_id, self.conversation_id)
        
        if not rows:
            return {"message": "No schemas found for this entity"}
            
        schemas = []
        for row in rows:
            schema_data = row["schema_data"] if isinstance(row["schema_data"], dict) else json.loads(row["schema_data"])
            
            schemas.append({
                "id": row["id"],
                "name": row["schema_name"],
                "category": row["category"],
                "data": schema_data,
                "created_at": row["created_at"]
            })
            
        # Potential tasks to run:
        # 1. Find schemas with low confidence that might need revision
        # 2. Find closely related schemas that might be merged
        # 3. Look for schemas with conflicting memories
        # 4. Generate new schemas from recent memories
        
        results = {
            "schemas_updated": 0,
            "conflicts_found": 0,
            "merge_opportunities": [],
            "low_confidence_schemas": []
        }
        
        # 1. Find low confidence schemas
        low_confidence = [s for s in schemas if s["data"].get("confidence", 0.5) < 0.5]
        for schema in low_confidence:
            results["low_confidence_schemas"].append({
                "id": schema["id"],
                "name": schema["name"],
                "confidence": schema["data"].get("confidence", 0.5)
            })
            
        # 2. Find schemas that might benefit from merging
        for i in range(len(schemas)):
            for j in range(i+1, len(schemas)):
                schema1 = schemas[i]
                schema2 = schemas[j]
                
                # Don't suggest merging across very different categories
                if schema1["category"] != schema2["category"]:
                    continue
                    
                # Calculate similarity between schemas
                similarity = await self._calculate_schema_similarity(schema1["data"], schema2["data"])
                
                if similarity > 0.7:  # High similarity threshold
                    results["merge_opportunities"].append({
                        "schema1_id": schema1["id"],
                        "schema1_name": schema1["name"],
                        "schema2_id": schema2["id"],
                        "schema2_name": schema2["name"],
                        "similarity": similarity
                    })
                    
        # 3. Check for conflicts
        conflict_checks = 0
        for schema in schemas:
            # Limit checks to maintain performance
            if conflict_checks >= 3:
                break
                
            schema_id = schema["id"]
            
            try:
                conflicts = await self.find_schema_conflicting_memories(
                    schema_id=schema_id,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    conn=conn
                )
                
                if conflicts.get("conflicts_found", False):
                    results["conflicts_found"] += len(conflicts.get("conflicting_memories", []))
                    
                    # Optionally evolve the schema based on conflicts
                    if random.random() < 0.5:  # 50% chance to auto-evolve
                        await self.evolve_schema_from_conflicts(
                            schema_id=schema_id,
                            entity_type=entity_type,
                            entity_id=entity_id,
                            conflicts=conflicts.get("conflicting_memories", []),
                            conn=conn
                        )
                        
                        results["schemas_updated"] += 1
                        
                conflict_checks += 1
                        
            except Exception as e:
                logger.error(f"Error checking conflicts for schema {schema_id}: {e}")
                
        # 4. Generate new schemas from recent memories
        # Get memories from the last week
        one_week_ago = datetime.now() - timedelta(days=7)
        
        recent_rows = await conn.fetch("""
            SELECT id, memory_text, tags
            FROM unified_memories
            WHERE entity_type = $1
              AND entity_id = $2
              AND user_id = $3
              AND conversation_id = $4
              AND timestamp > $5
              AND memory_type = 'observation'
            ORDER BY timestamp DESC
            LIMIT 50
        """, entity_type, entity_id, self.user_id, self.conversation_id, one_week_ago)
        
        if len(recent_rows) >= 10:
            # Randomly sample some memories
            sample_size = min(20, len(recent_rows))
            sampled_indices = random.sample(range(len(recent_rows)), sample_size)
            
            sampled_ids = [recent_rows[i]["id"] for i in sampled_indices]
            
            # Try to detect new schemas
            try:
                schema_result = await self.detect_schema_from_memories(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    memory_ids=sampled_ids,
                    min_memories=3,
                    conn=conn
                )
                
                if schema_result.get("schema_detected", False) and not schema_result.get("schema_already_exists", False):
                    results["new_schema_created"] = {
                        "schema_id": schema_result["schema_id"],
                        "schema_name": schema_result["schema_name"]
                    }
            except Exception as e:
                logger.error(f"Error detecting new schemas: {e}")
        
        return results
    
    async def _calculate_schema_similarity(
        self,
        schema1_data: Dict[str, Any],
        schema2_data: Dict[str, Any],
    ) -> float:
        """
        Similarity 0.0-1.0 between two schema descriptions.
        """
        client = get_openai_client()
    
        desc1 = schema1_data.get("description") or schema1_data.get("name", "")
        desc2 = schema2_data.get("description") or schema2_data.get("name", "")
    
        prompt = f"""
        SCHEMA 1:
        {desc1}
    
        SCHEMA 2:
        {desc2}
    
        Rate similarity 0.0-1.0. Return ONLY the number.
        """
    
        try:
            resp = await client.responses.create(
                model="gpt-4.1-nano",
                instructions="You rate schema similarity.",
                input=prompt,
                temperature=0.1,
            )
            score = float(resp.output_text.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.error("Similarity calc failed: %s", e)
            # quick word-overlap fallback
            w1 = set(desc1.lower().split())
            w2 = set(desc2.lower().split())
            return len(w1 & w2) / len(w1 | w2) if w1 and w2 else 0.0
            
# Create the necessary tables if they don't exist
async def create_schema_tables():
    """Create the necessary tables for the schema memory system if they don't exist."""
    async with TransactionContext() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS MemorySchemas (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                conversation_id INTEGER NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                schema_name TEXT NOT NULL,
                pattern TEXT NOT NULL,
                influence_strength FLOAT NOT NULL DEFAULT 0.5,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_memory_schemas_entity 
            ON MemorySchemas(user_id, conversation_id, entity_type, entity_id);
                        
            CREATE INDEX IF NOT EXISTS idx_memory_schemas_entity 
            ON MemorySchemas(user_id, conversation_id, entity_type, entity_id);
            
            CREATE INDEX IF NOT EXISTS idx_memory_schemas_category 
            ON MemorySchemas(category);
            
            CREATE INDEX IF NOT EXISTS idx_memory_schemas_name 
            ON MemorySchemas(schema_name);
        """)
