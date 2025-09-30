# lore/core/canon.py (Upgraded with Semantic Search)

import asyncio
import json
import logging

from db.connection import get_db_connection_context
from nyx.nyx_governance import AgentType
from nyx.governance_helpers import with_governance
from embedding.vector_store import generate_embedding # Assuming you have this
from lore.core.context import CanonicalContext

from .existence_gate import ExistenceGate, GateDecision
from .validation import CanonValidationAgent

from typing import List, Dict, Any, Union, Optional
from datetime import datetime, timedelta
from agents import Runner

from memory.memory_orchestrator import get_memory_orchestrator, EntityType

# Import the new validation agent
from lore.core.validation import CanonValidationAgent

logger = logging.getLogger(__name__)

_memory_orchestrators = {}

def violates_physics(ctx, action: dict) -> bool:
    """Check if an action violates physics caps."""
    # This would be called by the narrative system
    caps = ctx.physics_caps  # Loaded from world profile
    return (
        action.get("jump_height_m", 0) > caps["max_jump_m"] or
        action.get("projectile_speed_ms", 0) > caps["max_throw_ms"] or
        action.get("fall_survivable_m", 0) > caps["max_safe_fall_m"]
    )

def convert_importance_to_significance(importance):
    """
    Convert importance string or integer to significance float.
    
    Args:
        importance: String ("high", "medium", "low"), integer (1-10), or float (0.0-1.0)
        
    Returns:
        Float between 0.0 and 1.0
    """
    if isinstance(importance, (int, float)):
        # If it's a number 1-10, scale to 0.1-1.0
        if importance > 1:
            return min(1.0, max(0.1, importance / 10.0))
        # If already 0-1, return as is
        return min(1.0, max(0.0, importance))
    
    # String importance to float mapping
    importance_map = {
        "critical": 1.0,
        "high": 0.8,
        "medium": 0.5,
        "low": 0.3,
        "trivial": 0.1
    }
    
    if isinstance(importance, str):
        return importance_map.get(importance.lower(), 0.5)
    
    return 0.5  # Default

def ensure_canonical_context(ctx) -> CanonicalContext:
    """Convert any context to a CanonicalContext."""
    if isinstance(ctx, CanonicalContext):
        return ctx
    elif isinstance(ctx, dict):
        return CanonicalContext.from_dict(ctx)
    else:
        return CanonicalContext.from_object(ctx)


# --- Helper for Semantic Search ---

async def _find_semantically_similar_npc(conn, ctx, name: str, role: Optional[str], threshold: float = 0.90) -> Optional[int]:
    """Uses memory orchestrator's vector embeddings to find a semantically similar NPC."""
    memory_orchestrator = await get_canon_memory_orchestrator(ctx.user_id, ctx.conversation_id)
    
    search_text = f"NPC: {name}"
    if role:
        search_text += f", role: {role}"
    
    search_results = await memory_orchestrator.search_vector_store(
        query=search_text,
        entity_type="npc",
        top_k=1,
        filter_dict={"user_id": ctx.user_id, "conversation_id": ctx.conversation_id}
    )
    
    if search_results:
        top_result = search_results[0]
        similarity = top_result.get("similarity", 0)
        
        if similarity > threshold:
            npc_id = top_result.get("metadata", {}).get("entity_id")
            if npc_id:
                logger.info(
                    f"Found a semantically similar NPC via memory search: ID {npc_id} "
                    f"with similarity {similarity:.2f} to the proposal for '{name}'."
                )
                return npc_id
    
    # Fallback to database search
    search_vector = await memory_orchestrator.generate_embedding(search_text)
    
    query = """
        SELECT npc_id, npc_name, 1 - (embedding <=> $1) AS similarity
        FROM NPCStats
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> $1
        LIMIT 1
    """
    most_similar = await conn.fetchrow(query, search_vector)
    
    if most_similar and most_similar['similarity'] is not None and most_similar['similarity'] > threshold:
        logger.info(
            f"Found a semantically similar NPC in DB: '{most_similar['npc_name']}' (ID: {most_similar['npc_id']}) "
            f"with similarity {most_similar['similarity']:.2f} to the proposal for '{name}'."
        )
        return most_similar['npc_id']
    
    return None

async def get_canon_memory_orchestrator(user_id: int, conversation_id: int):
    """Get or create a memory orchestrator for canon operations."""
    from memory.memory_orchestrator import get_memory_orchestrator
    orchestrator = await get_memory_orchestrator(user_id, conversation_id)
    
    # Ensure canon is synced on first use
    if not hasattr(orchestrator, '_canon_synced') or not orchestrator._canon_synced:
        try:
            async with get_db_connection_context() as conn:
                tables_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_name = 'npcstats'
                    )
                """)
                
                if tables_exist:
                    await orchestrator.ensure_canon_synced()
        except Exception as e:
            logger.warning(f"Canon sync check failed: {e}")
    
    return orchestrator

async def create_canonical_entity_transactional(
    ctx, conn,
    entity_type: str,
    entity_data: Dict[str, Any],
    table_name: str,
    primary_key: str = "id"
) -> Dict[str, Any]:
    """
    Create an entity in both canon and memory systems transactionally.
    
    Args:
        ctx: Context
        conn: Database connection
        entity_type: Type of entity (npc, location, etc.)
        entity_data: Data for creating the entity
        table_name: Database table name
        primary_key: Primary key column name
        
    Returns:
        Dict with entity_id and success status
    """
    ctx = ensure_canonical_context(ctx)
    memory_orchestrator = await get_canon_memory_orchestrator(ctx.user_id, ctx.conversation_id)
    
    # Prepare embedding
    entity_name = entity_data.get('name', entity_data.get('npc_name', 'Unknown'))
    embedding_text = f"{entity_type}: {entity_name}"
    for key, value in entity_data.items():
        if key not in ['embedding', 'user_id', 'conversation_id'] and value:
            embedding_text += f" {value}"
    
    embedding = await memory_orchestrator.generate_embedding(embedding_text[:1000])
    entity_data['embedding'] = embedding
    
    # Ensure we're in a transaction
    if conn.is_in_transaction():
        # Already in transaction, just execute
        return await _create_entity_internal(
            ctx, conn, entity_type, entity_data, table_name, 
            primary_key, memory_orchestrator, entity_name
        )
    else:
        # Start new transaction
        async with conn.transaction():
            return await _create_entity_internal(
                ctx, conn, entity_type, entity_data, table_name,
                primary_key, memory_orchestrator, entity_name
            )

async def _create_entity_internal(
    ctx, conn,
    entity_type: str,
    entity_data: Dict[str, Any],
    table_name: str,
    primary_key: str,
    memory_orchestrator,
    entity_name: str
) -> Dict[str, Any]:
    """Internal helper for transactional entity creation."""
    # Build insert query
    columns = list(entity_data.keys())
    placeholders = [f"${i+1}" for i in range(len(columns))]
    
    insert_query = f"""
        INSERT INTO {table_name} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        RETURNING {primary_key}
    """
    
    try:
        # Create in database
        entity_id = await conn.fetchval(insert_query, *entity_data.values())
        
        # Store in memory system
        await memory_orchestrator.store_canonical_entity(
            entity_type=entity_type,
            entity_id=entity_id,
            entity_name=entity_name,
            entity_data={k: v for k, v in entity_data.items() 
                        if k not in ['embedding', 'user_id', 'conversation_id']},
            significance=7
        )
        
        # Add to vector store for searchability
        await memory_orchestrator.add_to_vector_store(
            text=f"{entity_type}: {entity_name}",
            metadata={
                "entity_type": entity_type.lower(),
                "entity_id": entity_id,
                "entity_name": entity_name,
                "canonical": True,
                "user_id": ctx.user_id,
                "conversation_id": ctx.conversation_id
            },
            entity_type=entity_type.lower()
        )
        
        # Log canonical event
        await log_canonical_event(
            ctx, conn,
            f"Created {entity_type}: {entity_name}",
            tags=[entity_type.lower(), 'creation', 'canon'],
            significance=7
        )
        
        return {
            "success": True,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "entity_name": entity_name
        }
        
    except Exception as e:
        logger.error(f"Failed to create {entity_type}: {e}")
        # Transaction will rollback automatically
        raise

async def get_entity_with_memories(
    ctx, conn,
    entity_type: str,
    entity_id: int,
    include_memories: bool = True,
    include_beliefs: bool = True,
    include_emotional_state: bool = True,
    include_relationships: bool = True,
    memory_limit: int = 5
) -> Dict[str, Any]:
    """
    Get a canonical entity enriched with memory system data.
    
    Args:
        ctx: Context
        conn: Database connection
        entity_type: Type of entity (NPCStats, Locations, etc.)
        entity_id: Entity ID
        include_memories: Include recent memories
        include_beliefs: Include entity beliefs
        include_emotional_state: Include emotional state
        include_relationships: Include relationships
        memory_limit: Max memories to retrieve
        
    Returns:
        Enriched entity data or None if not found
    """
    ctx = ensure_canonical_context(ctx)
    orchestrator = await get_canon_memory_orchestrator(ctx.user_id, ctx.conversation_id)
    
    # Get canonical data
    entity = await get_entity_by_id(ctx, conn, entity_type, entity_id)
    
    if not entity:
        return None
    
    # Map table name to entity type for memory system
    type_mapping = {
        'NPCStats': 'npc',
        'Locations': 'location',
        'Nations': 'nation',
        'Events': 'event',
        'Factions': 'faction'
    }
    
    memory_entity_type = type_mapping.get(entity_type, entity_type.lower())
    
    # Enrich with memory data
    if include_memories:
        memory_result = await orchestrator.retrieve_memories(
            entity_type=memory_entity_type,
            entity_id=entity_id,
            limit=memory_limit,
            include_analysis=True
        )
        entity["memories"] = memory_result.get("memories", [])
        entity["memory_analysis"] = memory_result.get("analysis", {})
    
    if include_beliefs and memory_entity_type in ['npc', 'player', 'nyx']:
        entity["beliefs"] = await orchestrator.get_beliefs(
            entity_type=memory_entity_type,
            entity_id=entity_id,
            min_confidence=0.3
        )
    
    if include_emotional_state and memory_entity_type in ['npc', 'player']:
        entity["emotional_state"] = await orchestrator.get_emotional_state(
            entity_type=memory_entity_type,
            entity_id=entity_id
        )
    
    if include_relationships:
        # Get relationships from SocialLinks
        relationships = await conn.fetch("""
            SELECT entity2_type, entity2_id, link_type, link_level, dynamics
            FROM SocialLinks
            WHERE user_id = $1 AND conversation_id = $2
            AND entity1_type = $3 AND entity1_id = $4
            
            UNION
            
            SELECT entity1_type as entity2_type, entity1_id as entity2_id, 
                   link_type, link_level, dynamics
            FROM SocialLinks
            WHERE user_id = $1 AND conversation_id = $2
            AND entity2_type = $3 AND entity2_id = $4
        """, ctx.user_id, ctx.conversation_id, memory_entity_type, entity_id)
        
        entity["relationships"] = [
            {
                "target_type": r["entity2_type"],
                "target_id": r["entity2_id"],
                "link_type": r["link_type"],
                "strength": r["link_level"],
                "dynamics": r["dynamics"]
            }
            for r in relationships
        ]
    
    # Add narrative context
    entity["narrative_context"] = await orchestrator.get_narrative_context(
        focus_entities=[(memory_entity_type, entity_id)],
        include_predictions=False
    )
    
    return entity

async def check_canon_consistency(
    ctx, conn,
    operation_type: str,
    entity_type: str,
    entity_data: Dict[str, Any],
    enforce: bool = False
) -> Dict[str, Any]:
    """
    Check if an operation would violate canon consistency.
    
    Args:
        ctx: Context
        conn: Database connection
        operation_type: Type of operation (create, update, delete)
        entity_type: Type of entity
        entity_data: Data for the operation
        enforce: If True, raise exception on conflicts
        
    Returns:
        Consistency check results
    """
    ctx = ensure_canonical_context(ctx)
    orchestrator = await get_canon_memory_orchestrator(ctx.user_id, ctx.conversation_id)
    
    # Check for conflicts
    consistency = await orchestrator.validate_canonical_consistency(
        entity_type=entity_type,
        entity_data=entity_data
    )
    
    if not consistency["is_consistent"] and enforce:
        raise ValueError(
            f"Operation '{operation_type}' violates canon: {consistency['conflicts']}"
        )
    
    # Add operation context
    consistency["operation_type"] = operation_type
    consistency["entity_type"] = entity_type
    consistency["checked_at"] = datetime.utcnow().isoformat()
    
    # Log warning if conflicts found
    if not consistency["is_consistent"]:
        logger.warning(
            f"Canon consistency issues for {operation_type} on {entity_type}: "
            f"{consistency['conflicts']}"
        )
        
        # Store as a canonical event for tracking
        await log_canonical_event(
            ctx, conn,
            f"Consistency conflict detected: {operation_type} on {entity_type}",
            tags=['consistency', 'conflict', entity_type.lower()],
            significance=5
        )
    
    return consistency


# ADD this function to sync a specific entity to memory:
async def sync_entity_to_memory(
    ctx, conn,
    entity_type: str,
    entity_id: int,
    force: bool = False
) -> Dict[str, Any]:
    """
    Sync a specific canonical entity to the memory system.
    
    Args:
        ctx: Context
        conn: Database connection
        entity_type: Type of entity (NPCStats, Locations, etc.)
        entity_id: Entity ID
        force: Force resync even if already synced
        
    Returns:
        Sync results
    """
    ctx = ensure_canonical_context(ctx)
    orchestrator = await get_canon_memory_orchestrator(ctx.user_id, ctx.conversation_id)
    
    # Check if already synced (unless forced)
    if not force:
        existing = await orchestrator.search_canonical_entities(
            query=f"{entity_type} id:{entity_id}",
            entity_types=[entity_type.lower()]
        )
        if existing:
            return {"status": "already_synced", "entity_id": entity_id}
    
    # Get entity data
    entity = await get_entity_by_id(ctx, conn, entity_type, entity_id)
    if not entity:
        return {"status": "not_found", "entity_id": entity_id}
    
    # Map to memory entity type
    type_mapping = {
        'NPCStats': 'npc',
        'Locations': 'location',
        'Nations': 'nation',
        'Events': 'event',
        'Factions': 'faction'
    }
    memory_entity_type = type_mapping.get(entity_type, entity_type.lower())
    
    # Extract name
    entity_name = (
        entity.get('npc_name') or 
        entity.get('location_name') or 
        entity.get('name') or 
        entity.get('event_name') or 
        f"{entity_type}_{entity_id}"
    )
    
    # Store in memory system
    await orchestrator.store_canonical_entity(
        entity_type=memory_entity_type,
        entity_id=entity_id,
        entity_name=entity_name,
        entity_data={k: v for k, v in entity.items() 
                    if k not in ['embedding', 'user_id', 'conversation_id']},
        significance=5
    )
    
    # Add to vector store
    description = (
        entity.get('description') or 
        entity.get('role') or 
        entity.get('event_text') or 
        ""
    )
    
    await orchestrator.add_to_vector_store(
        text=f"{memory_entity_type}: {entity_name} - {description}",
        metadata={
            "entity_type": memory_entity_type,
            "entity_id": entity_id,
            "entity_name": entity_name,
            "canonical": True,
            "user_id": ctx.user_id,
            "conversation_id": ctx.conversation_id,
            "synced_at": datetime.utcnow().isoformat()
        },
        entity_type=memory_entity_type
    )
    
    return {
        "status": "synced",
        "entity_type": memory_entity_type,
        "entity_id": entity_id,
        "entity_name": entity_name
    }


# --- Upgraded NPC Canon Function ---
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="find_or_create_npc",
    action_description="Finding or creating NPC: {npc_name}",
    id_from_context=lambda ctx: "canon"
)
async def find_or_create_npc(ctx, conn, npc_name: str, **kwargs) -> int:
    """Find or create NPC with existence gate validation."""
    # Check exact match
    existing_npc = await conn.fetchrow(
        "SELECT npc_id FROM NPCStats WHERE npc_name = $1 AND user_id = $2 AND conversation_id = $3",
        npc_name, ctx.user_id, ctx.conversation_id
    )
    if existing_npc:
        logger.warning(f"NPC '{npc_name}' found via exact match with ID {existing_npc['npc_id']}.")
        return existing_npc['npc_id']
    
    # Run existence gate check
    gate = ExistenceGate(ctx)
    npc_data = {
        'npc_name': npc_name,
        'role': kwargs.get('role'),
        'affiliations': kwargs.get('affiliations', []),
        **kwargs
    }
    
    # Get scene context for institution checks
    scene_context = await conn.fetchrow("""
        SELECT value FROM CurrentRoleplay
        WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentScene'
    """, ctx.user_id, ctx.conversation_id)
    
    if scene_context and scene_context['value']:
        scene_data = json.loads(scene_context['value'])
    else:
        scene_data = None
    
    decision, details = await gate.assess_entity('npc', npc_data, scene_data)
    
    if decision == GateDecision.DENY:
        # NPC role cannot exist
        raise ValueError(f"NPC '{npc_name}' with role '{kwargs.get('role')}' cannot exist: {details['reason']}")
    
    elif decision == GateDecision.DEFER:
        # Return as lead
        logger.warning(f"NPC '{npc_name}' deferred: {details['reason']}")
        return -1  # Signal deferred NPC
    
    elif decision == GateDecision.ANALOG:
        # Use analog role
        analog_role = details['analog']
        logger.info(f"Substituting role '{kwargs.get('role')}' with analog '{analog_role}'")
        kwargs['role'] = analog_role
        npc_data = details.get('analog_data', npc_data)
    
    # Semantic similarity check
    memory_orchestrator = await get_canon_memory_orchestrator(ctx.user_id, ctx.conversation_id)
    role = kwargs.get("role")
    similar_npc_id = await _find_semantically_similar_npc(conn, ctx, npc_name, role)
    
    if similar_npc_id:
        validation_agent = CanonValidationAgent()
        is_duplicate = await validation_agent.confirm_is_duplicate_npc(
            conn,
            proposal={"name": npc_name, "role": role},
            existing_npc_id=similar_npc_id
        )
        
        if is_duplicate:
            logger.warning(f"LLM confirmed that proposal '{npc_name}' is a duplicate of existing NPC ID {similar_npc_id}.")
            return similar_npc_id
        else:
            logger.info(f"LLM determined that proposal '{npc_name}' is NOT a duplicate. Proceeding with creation.")
    
    # Create the new NPC (already passed existence gate)
    embedding_text = f"NPC: {npc_name}"
    if role:
        embedding_text += f", role: {role}"
    
    new_embedding = await memory_orchestrator.generate_embedding(embedding_text)
    
    affiliations = kwargs.get("affiliations", [])
    if isinstance(affiliations, list):
        affiliations_json = json.dumps(affiliations)
    else:
        affiliations_json = affiliations
    
    insert_query = """
        INSERT INTO NPCStats (user_id, conversation_id, npc_name, role, affiliations, embedding)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6) RETURNING npc_id
    """
    
    npc_id = await conn.fetchval(
        insert_query,
        ctx.user_id,
        ctx.conversation_id,
        npc_name,
        role,
        affiliations_json,
        new_embedding
    )
    
    # Store in memory system
    from memory.memory_orchestrator import EntityType
    await memory_orchestrator.store_memory(
        entity_type=EntityType.NYX,
        entity_id=0,
        memory_text=f"Created new NPC '{npc_name}' with role '{role or 'unspecified'}'",
        significance=0.8,
        tags=["npc_creation", "canon", npc_name.lower()],
        metadata={
            "npc_id": npc_id,
            "npc_name": npc_name,
            "role": role,
            "affiliations": affiliations
        }
    )
    
    # Add to vector store
    await memory_orchestrator.add_to_vector_store(
        text=f"NPC: {npc_name}, role: {role or 'unspecified'}, affiliations: {affiliations}",
        metadata={
            "entity_type": "npc",
            "entity_id": npc_id,
            "npc_name": npc_name,
            "user_id": ctx.user_id,
            "conversation_id": ctx.conversation_id
        },
        entity_type="npc"
    )
    
    logger.info(f"Canonically created new, unique NPC '{npc_name}' with ID {npc_id}.")
    return npc_id

    
async def find_or_create_entity(
    ctx,
    conn,
    entity_type: str,
    entity_name: str,
    search_fields: Dict[str, Any],
    create_data: Dict[str, Any],
    table_name: str,
    embedding_text: str,
    similarity_threshold: float = 0.85
) -> int:
    """
    Generic function to find or create any entity with semantic similarity check.
    Now uses memory orchestrator for all embedding and similarity operations.
    """
    ctx = ensure_canonical_context(ctx)
    memory_orchestrator = await get_canon_memory_orchestrator(ctx.user_id, ctx.conversation_id)
    
    # Step 1: Exact match check
    where_clauses = []
    values = []
    for i, (field, value) in enumerate(search_fields.items()):
        where_clauses.append(f"{field} = ${i+1}")
        values.append(value)
    
    if values:
        query = f"SELECT id FROM {table_name} WHERE {' AND '.join(where_clauses)}"
        existing = await conn.fetchrow(query, *values)
        if existing:
            logger.info(f"{entity_type} '{entity_name}' found via exact match with ID {existing['id']}.")
            return existing['id']
    
    # Step 2: Semantic similarity check using memory orchestrator
    search_results = await memory_orchestrator.search_vector_store(
        query=embedding_text,
        entity_type=entity_type.lower(),
        top_k=1,
        filter_dict={"user_id": ctx.user_id, "conversation_id": ctx.conversation_id}
    )
    
    if search_results and search_results[0].get("similarity", 0) > similarity_threshold:
        most_similar = search_results[0]
        entity_id = most_similar.get("metadata", {}).get("entity_id")
        
        if entity_id:
            logger.info(f"Found semantically similar {entity_type} via memory search with ID {entity_id}")
            
            # Use validation agent to confirm
            validation_agent = CanonValidationAgent()
            prompt = f"""
            I am considering creating a new {entity_type}, but found a semantically similar existing one.
            Please determine if they are the same entity described differently.

            Proposed New {entity_type}:
            - Name: "{entity_name}"
            - Details: {json.dumps(create_data, indent=2)}

            Most Similar Existing {entity_type}:
            - ID: {entity_id}
            - From search result

            Are these the same {entity_type}? Answer with only 'true' or 'false'.
            """
            
            result = await Runner.run(validation_agent.agent, prompt)
            is_duplicate = result.final_output.strip().lower() == 'true'
            
            if is_duplicate:
                logger.info(f"LLM confirmed '{entity_name}' is a duplicate of existing {entity_type} ID {entity_id}")
                return entity_id
    
    # Step 3: Create new entity
    columns = list(create_data.keys())
    placeholders = [f"${i+1}" for i in range(len(columns))]
    
    insert_query = f"""
        INSERT INTO {table_name} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        RETURNING id
    """
    
    entity_id = await conn.fetchval(insert_query, *create_data.values())
    
    # Generate and store embedding
    search_vector = await memory_orchestrator.generate_embedding(embedding_text)
    await conn.execute(
        f"UPDATE {table_name} SET embedding = $1 WHERE id = $2",
        search_vector, entity_id
    )
    
    # Store creation as a memory
    await memory_orchestrator.store_memory(
        entity_type=EntityType.NYX,
        entity_id=0,
        memory_text=f"Created new {entity_type}: {entity_name}",
        significance=convert_importance_to_significance("high"),
        tags=[entity_type.lower(), "creation", "canon"],
        metadata={
            "entity_id": entity_id,
            "entity_name": entity_name,
            "entity_type": entity_type,
            "table_name": table_name
        }
    )
    
    # Add to vector store for future searches
    await memory_orchestrator.add_to_vector_store(
        text=embedding_text,
        metadata={
            "entity_type": entity_type.lower(),
            "entity_id": entity_id,
            "entity_name": entity_name,
            "user_id": ctx.user_id,
            "conversation_id": ctx.conversation_id
        },
        entity_type=entity_type.lower()
    )
    
    logger.info(f"Canonically created new {entity_type} '{entity_name}' with ID {entity_id}")
    
    await log_canonical_event(
        ctx, conn, 
        f"Created new {entity_type}: {entity_name}",
        tags=[entity_type.lower(), 'creation'],
        significance=7
    )
    
    return entity_id


async def log_canonical_event(
    ctx,
    conn,
    event_text: str,
    tags: List[str] = None,
    significance: int = 5,
    *,
    persist_memory: bool = True,
):
    """Log a canonical event with causality tracking.

    Args:
        ctx: Canonical context carrying user/conversation identifiers.
        conn: Database connection used for persistence.
        event_text: Description of the canonical event.
        tags: Optional tags to associate with the event.
        significance: Integer significance score (1-10).
        persist_memory: When ``True`` the event is forwarded to the memory system.
    """
    tags = tags or []
    tags_json = json.dumps(tags)
    
    # Get parent events for causality chain
    parent_events = await conn.fetch("""
        SELECT id FROM CanonicalEvents
        WHERE user_id=$1 AND conversation_id=$2
        ORDER BY timestamp DESC
        LIMIT 3
    """, ctx.user_id, ctx.conversation_id)
    
    parent_ids = [p['id'] for p in parent_events]
    
    # Store in database with parent links
    event_timestamp = datetime.utcnow()

    event_id = await conn.fetchval("""
        INSERT INTO CanonicalEvents (
            user_id, conversation_id, event_text, tags, significance,
            timestamp, parent_event_ids
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING id
    """, ctx.user_id, ctx.conversation_id, event_text, tags_json,
        significance, event_timestamp, json.dumps(parent_ids))
    
    # Check for wide blast radius events that need propagation
    if significance >= 8:
        await _propagate_event_consequences(ctx, conn, event_id, event_text, tags, significance)
    
    if persist_memory:
        _schedule_canonical_memory_persist(
            ctx,
            event_id,
            event_text,
            list(tags),
            significance,
            parent_ids,
            event_timestamp,
        )

    return event_id


def _schedule_canonical_memory_persist(
    ctx,
    event_id: int,
    event_text: str,
    tags: List[str],
    significance: int,
    parent_ids: List[int],
    event_timestamp: datetime,
) -> None:
    """Kick off memory persistence without blocking the DB connection."""

    asyncio.create_task(
        _persist_canonical_event_memory(
            ctx,
            event_id,
            event_text,
            tags,
            significance,
            parent_ids,
            event_timestamp,
        )
    )


async def _persist_canonical_event_memory(
    ctx,
    event_id: int,
    event_text: str,
    tags: List[str],
    significance: int,
    parent_ids: List[int],
    event_timestamp: datetime,
) -> None:
    """Persist the canonical event to the memory system with error logging."""

    try:
        memory_orchestrator = await get_canon_memory_orchestrator(
            ctx.user_id, ctx.conversation_id
        )

        significance_float = min(1.0, max(0.1, significance / 10.0))

        await memory_orchestrator.store_memory(
            entity_type=EntityType.LORE,
            entity_id=0,
            memory_text=event_text,
            significance=significance_float,
            tags=[*tags, "canonical_event"],
            metadata={
                "event_id": event_id,
                "significance": significance,
                "parent_events": parent_ids,
                "timestamp": event_timestamp.isoformat(),
            },
        )
    except Exception:
        logger.exception(
            "Failed to persist canonical event %s to memory", event_id
        )

async def _propagate_event_consequences(
    ctx,
    conn,
    event_id: int,
    event_text: str,
    tags: List[str],
    significance: int,
):
    """Propagate consequences of high-impact events."""
    # Identify event type and affected entities
    affected_entities = []
    
    # Parse event for entity mentions
    if 'death' in event_text.lower() or 'killed' in event_text.lower():
        # Find NPCs mentioned
        npcs = await conn.fetch("""
            SELECT npc_id, npc_name FROM NPCStats
            WHERE user_id=$1 AND conversation_id=$2
            AND position(LOWER(npc_name) in LOWER($3)) > 0
        """, ctx.user_id, ctx.conversation_id, event_text)
        
        for npc in npcs:
            # Mark NPC as dead
            await conn.execute("""
                UPDATE NPCStats 
                SET status='dead', death_event_id=$1
                WHERE npc_id=$2
            """, event_id, npc['npc_id'])
            
            affected_entities.append(('npc', npc['npc_id']))
    
    if 'destroyed' in event_text.lower() or 'burned' in event_text.lower():
        # Find locations mentioned
        locations = await conn.fetch("""
            SELECT id, location_name FROM Locations
            WHERE user_id=$1 AND conversation_id=$2
            AND position(LOWER(location_name) in LOWER($3)) > 0
        """, ctx.user_id, ctx.conversation_id, event_text)
        
        for loc in locations:
            # Mark location as destroyed
            await conn.execute("""
                UPDATE Locations
                SET status='destroyed', destruction_event_id=$1
                WHERE id=$2
            """, event_id, loc['id'])
            
            affected_entities.append(('location', loc['id']))
    
    # Update relationships and faction power based on affected entities
    for entity_type, entity_id in affected_entities:
        # Remove social links for dead NPCs
        if entity_type == 'npc':
            await conn.execute("""
                DELETE FROM SocialLinks
                WHERE (entity1_type='npc' AND entity1_id=$1)
                OR (entity2_type='npc' AND entity2_id=$1)
            """, entity_id)
        
        # Update faction power if leader dies
        faction = await conn.fetchrow("""
            SELECT id, power_level FROM Factions
            WHERE user_id=$1 AND conversation_id=$2
            AND leader_npc_id=$3
        """, ctx.user_id, ctx.conversation_id, entity_id)
        
        if faction:
            new_power = max(1, faction['power_level'] - 2)
            await conn.execute("""
                UPDATE Factions
                SET power_level=$1, leader_npc_id=NULL
                WHERE id=$2
            """, new_power, faction['id'])
    
    # Create follow-up task for narrative consequences
    await conn.execute("""
        INSERT INTO CanonicalEvents (
            user_id, conversation_id, event_text, tags, significance,
            timestamp, parent_event_ids
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
    """, ctx.user_id, ctx.conversation_id,
        f"Consequences of: {event_text[:100]}",
        json.dumps(['consequence'] + tags),
        max(1, significance - 1),
        datetime.utcnow() + timedelta(seconds=1),
        json.dumps([event_id])
    )



async def ensure_npc_exists(ctx, conn, npc_reference: Union[int, str, Dict[str, Any]]) -> int:
    """
    Ensure an NPC exists, creating if necessary.
    
    Args:
        ctx: Context
        conn: Database connection
        npc_reference: Either an ID, name string, or dict with NPC details
        
    Returns:
        NPC ID
    """
    from lore.core import canon
    
    if isinstance(npc_reference, int):
        # Check if ID exists
        existing = await conn.fetchrow("SELECT id FROM NPCStats WHERE id = $1", npc_reference)
        if existing:
            return npc_reference
        else:
            # Create with placeholder name
            return await canon.find_or_create_npc(ctx, conn, f"Character {npc_reference}")
    
    elif isinstance(npc_reference, str):
        # Create/find by name
        return await canon.find_or_create_npc(ctx, conn, npc_reference)
    
    elif isinstance(npc_reference, dict):
        # Create/find with full details
        return await canon.find_or_create_npc(
            ctx, conn,
            npc_name=npc_reference.get('name', 'Unknown'),
            role=npc_reference.get('role'),
            affiliations=npc_reference.get('affiliations', [])
        )
    
    else:
        raise ValueError(f"Invalid NPC reference type: {type(npc_reference)}")


async def find_or_create_nation(
    ctx,
    conn,
    nation_name: str,
    government_type: str = None,
    matriarchy_level: int = None,
    **kwargs
) -> int:
    """Find or create a nation with sophisticated matching logic using memory orchestrator."""
    ctx = ensure_canonical_context(ctx)
    memory_orchestrator = await get_canon_memory_orchestrator(ctx.user_id, ctx.conversation_id)
    
    # Step 1: Exact name match
    existing = await conn.fetchrow("""
        SELECT id, name, government_type, matriarchy_level, description
        FROM Nations 
        WHERE LOWER(name) = LOWER($1)
    """, nation_name)
    
    if existing:
        logger.info(f"Nation '{nation_name}' found via exact match with ID {existing['id']}")
        return existing['id']
    
    # Step 2: Memory-based semantic search
    description = kwargs.get('description', '')
    search_text = f"Nation: {nation_name}, Government: {government_type or 'unknown'}, {description}"
    
    search_results = await memory_orchestrator.search_canonical_entities(
        query=search_text,
        entity_types=["nation"],
        similarity_threshold=0.8
    )
    
    if search_results:
        for result in search_results:
            existing_id = result.get("metadata", {}).get("entity_id")
            if existing_id:
                # Validate with LLM
                validation_agent = CanonValidationAgent()
                prompt = f"""
                Are these the same nation?
                
                Proposed: {nation_name} ({government_type or 'unknown government'})
                Existing match found with similarity: {result.get('similarity', 0):.2f}
                
                Consider name variations, abbreviations, and translations.
                Answer only 'true' or 'false'.
                """
                
                agent_result = await Runner.run(validation_agent.agent, prompt)
                if agent_result.final_output.strip().lower() == 'true':
                    logger.info(f"Nation '{nation_name}' matched to existing (ID: {existing_id})")
                    return existing_id
    
    # Step 3: Create new nation
    embedding = await memory_orchestrator.generate_embedding(search_text)
    
    nation_id = await conn.fetchval("""
        INSERT INTO Nations (
            name, government_type, description, relative_power,
            matriarchy_level, population_scale, major_resources,
            major_cities, cultural_traits, notable_features,
            neighboring_nations, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        RETURNING id
    """,
        nation_name,
        government_type or kwargs.get('government_type', 'Unknown'),
        description or kwargs.get('description', f"The nation of {nation_name}"),
        kwargs.get('relative_power', 5),
        matriarchy_level or kwargs.get('matriarchy_level', 5),
        kwargs.get('population_scale'),
        kwargs.get('major_resources', []),
        kwargs.get('major_cities', []),
        kwargs.get('cultural_traits', []),
        kwargs.get('notable_features'),
        kwargs.get('neighboring_nations', []),
        embedding
    )
    
    # Store in memory system
    await memory_orchestrator.store_canonical_entity(
        entity_type="nation",
        entity_id=nation_id,
        entity_name=nation_name,
        entity_data={
            "government_type": government_type,
            "matriarchy_level": matriarchy_level,
            "relative_power": kwargs.get('relative_power', 5),
            "description": description
        },
        significance=8
    )
    
    await log_canonical_event(
        ctx, conn,
        f"Established new nation: {nation_name} with {government_type or 'unknown'} government",
        tags=['nation', 'creation', 'political'],
        significance=8
    )
    
    return nation_id

async def find_or_create_conflict(
    ctx,
    conn,
    conflict_name: str,
    involved_nations: List[int],
    conflict_type: str,
    **kwargs
) -> int:
    """Find or create a conflict with memory-enhanced duplicate detection."""
    ctx = ensure_canonical_context(ctx)
    memory_orchestrator = await get_canon_memory_orchestrator(ctx.user_id, ctx.conversation_id)
    
    # Check for existing conflicts using memory search
    search_text = f"Conflict: {conflict_name}, Type: {conflict_type}, Description: {kwargs.get('description', '')}"
    
    search_results = await memory_orchestrator.search_canonical_entities(
        query=search_text,
        entity_types=["conflict"],
        similarity_threshold=0.8
    )
    
    if search_results:
        for result in search_results:
            conflict_id = result.get("metadata", {}).get("entity_id")
            if conflict_id:
                # Check if nations match
                existing_conflict = await conn.fetchrow("""
                    SELECT * FROM NationalConflicts WHERE id = $1
                """, conflict_id)
                
                if existing_conflict:
                    existing_nations = set(existing_conflict['involved_nations'])
                    proposed_nations = set(involved_nations)
                    overlap = existing_nations & proposed_nations
                    
                    if len(overlap) >= min(len(existing_nations), len(proposed_nations)) * 0.5:
                        # Significant overlap - verify with agent
                        validation_agent = CanonValidationAgent()
                        prompt = f"""
                        Found a similar conflict. Are these the same?
                        
                        Existing: {existing_conflict['name']} ({existing_conflict['conflict_type']})
                        Proposed: {conflict_name} ({conflict_type})
                        Similarity: {result.get('similarity', 0):.2f}
                        
                        Answer only 'true' or 'false'.
                        """
                        
                        agent_result = await Runner.run(validation_agent.agent, prompt)
                        if agent_result.final_output.strip().lower() == 'true':
                            return conflict_id
    
    # Create new conflict
    embedding = await memory_orchestrator.generate_embedding(search_text)
    
    conflict_id = await conn.fetchval("""
        INSERT INTO NationalConflicts (
            name, conflict_type, description, severity, status,
            start_date, involved_nations, primary_aggressor, primary_defender,
            current_casualties, economic_impact, diplomatic_consequences,
            public_opinion, recent_developments, potential_resolution, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
        RETURNING id
    """,
        conflict_name,
        conflict_type,
        kwargs.get('description', ''),
        kwargs.get('severity', 5),
        kwargs.get('status', 'active'),
        kwargs.get('start_date', 'Recently'),
        involved_nations,
        kwargs.get('primary_aggressor'),
        kwargs.get('primary_defender'),
        kwargs.get('current_casualties', 'Unknown'),
        kwargs.get('economic_impact', 'Being assessed'),
        kwargs.get('diplomatic_consequences', 'Developing'),
        json.dumps(kwargs.get('public_opinion', {})),
        kwargs.get('recent_developments', []),
        kwargs.get('potential_resolution', 'Uncertain'),
        embedding
    )
    
    # Get nation names for logging
    nation_names = await conn.fetch("""
        SELECT name FROM Nations WHERE id = ANY($1)
    """, involved_nations)
    nation_list = [n['name'] for n in nation_names]
    
    # Store in memory system
    await memory_orchestrator.store_canonical_entity(
        entity_type="conflict",
        entity_id=conflict_id,
        entity_name=conflict_name,
        entity_data={
            "conflict_type": conflict_type,
            "involved_nations": nation_list,
            "severity": kwargs.get('severity', 5),
            "status": kwargs.get('status', 'active')
        },
        significance=9
    )
    
    # Store conflict as a significant memory for involved nations
    for nation_id in involved_nations:
        await memory_orchestrator.store_memory(
            entity_type="nation",
            entity_id=nation_id,
            memory_text=f"Became involved in conflict: {conflict_name}",
            significance=convert_importance_to_significance("critical"),
            tags=["conflict", conflict_type, "political"],
            metadata={"conflict_id": conflict_id}
        )
    
    await log_canonical_event(
        ctx, conn,
        f"New conflict erupted: {conflict_name} between {', '.join(nation_list)}",
        tags=['conflict', 'political', conflict_type],
        significance=9
    )
    
    return conflict_id

async def find_or_create_cultural_element(
    ctx, conn, 
    name: str, 
    element_type: str, 
    description: str,
    practiced_by: List[str],
    significance: int,
    historical_origin: str
) -> int:
    """Find or create a cultural element with semantic matching."""
    embed_text = f"{name} {element_type} {description}"
    
    create_data = {
        'name': name,
        'element_type': element_type,
        'description': description,
        'practiced_by': practiced_by,
        'significance': significance,
        'historical_origin': historical_origin
    }
    
    search_fields = {
        'name': name,
        'element_type': element_type,
        'name_field': 'name'
    }
    
    return await find_or_create_entity(
        ctx=ctx,
        conn=conn,
        entity_type="cultural_element",
        entity_name=name,
        search_fields=search_fields,
        create_data=create_data,
        table_name="CulturalElements",
        embedding_text=embed_text,
        similarity_threshold=0.85
    )

async def find_or_create_culinary_tradition(
    ctx, conn,
    name: str,
    nation_origin: int,
    description: str,
    **kwargs
) -> int:
    """Find or create a culinary tradition."""
    embed_text = f"{name} {description} cuisine"
    
    create_data = {
        'name': name,
        'nation_origin': nation_origin,
        'description': description,
        'ingredients': kwargs.get('ingredients', []),
        'preparation': kwargs.get('preparation', ''),
        'cultural_significance': kwargs.get('cultural_significance', ''),
        'adopted_by': kwargs.get('adopted_by', [])
    }
    
    search_fields = {
        'name': name,
        'name_field': 'name'
    }
    
    return await find_or_create_entity(
        ctx=ctx,
        conn=conn,
        entity_type="culinary_tradition",
        entity_name=name,
        search_fields=search_fields,
        create_data=create_data,
        table_name="CulinaryTraditions",
        embedding_text=embed_text,
        similarity_threshold=0.90
    )

async def find_or_create_social_custom(
    ctx, conn,
    name: str,
    nation_origin: int,
    description: str,
    **kwargs
) -> int:
    """Find or create a social custom."""
    embed_text = f"{name} {description} custom"
    
    create_data = {
        'name': name,
        'nation_origin': nation_origin,
        'description': description,
        'context': kwargs.get('context', 'social'),
        'formality_level': kwargs.get('formality_level', 'medium'),
        'adopted_by': kwargs.get('adopted_by', []),
        'adoption_date': kwargs.get('adoption_date', datetime.utcnow())
    }
    
    search_fields = {
        'name': name,
        'name_field': 'name'
    }
    
    return await find_or_create_entity(
        ctx=ctx,
        conn=conn,
        entity_type="social_custom",
        entity_name=name,
        search_fields=search_fields,
        create_data=create_data,
        table_name="SocialCustoms",
        embedding_text=embed_text,
        similarity_threshold=0.85
    )

# Helper functions to add nations to existing elements

async def add_nation_to_cultural_element(ctx, conn, element_id: int, nation_ref: str) -> None:
    """Add a nation to the practiced_by array of a cultural element."""
    await conn.execute("""
        UPDATE CulturalElements
        SET practiced_by = array_append(practiced_by, $1)
        WHERE id = $2 AND NOT ($1 = ANY(practiced_by))
    """, nation_ref, element_id)

async def add_nation_to_culinary_tradition(ctx, conn, tradition_id: int, nation_id: int) -> None:
    """Add a nation to the adopted_by array of a culinary tradition."""
    await conn.execute("""
        UPDATE CulinaryTraditions
        SET adopted_by = array_append(adopted_by, $1)
        WHERE id = $2 AND NOT ($1 = ANY(adopted_by))
    """, nation_id, tradition_id)

async def add_nation_to_social_custom(ctx, conn, custom_id: int, nation_id: int) -> None:
    """Add a nation to the adopted_by array of a social custom."""
    await conn.execute("""
        UPDATE SocialCustoms
        SET adopted_by = array_append(adopted_by, $1)
        WHERE id = $2 AND NOT ($1 = ANY(adopted_by))
    """, nation_id, custom_id)

async def create_cultural_exchange(ctx, conn, **kwargs) -> int:
    """Create a cultural exchange record."""
    return await conn.fetchval("""
        INSERT INTO CulturalExchanges (
            nation1_id, nation2_id, exchange_type, 
            exchange_details, timestamp
        )
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
    """, 
        kwargs['nation1_id'],
        kwargs['nation2_id'],
        kwargs['exchange_type'],
        kwargs['exchange_details'],
        kwargs['timestamp']
    )

async def find_or_create_geographic_region(
    ctx, conn,
    name: str,
    region_type: str,
    description: str,
    **kwargs
) -> int:
    """Find or create a geographic region with semantic matching."""
    # Check exact match
    existing = await conn.fetchrow("""
        SELECT id FROM GeographicRegions
        WHERE LOWER(name) = LOWER($1) AND region_type = $2
    """, name, region_type)
    
    if existing:
        return existing['id']
    
    # Semantic similarity check
    climate = kwargs.get('climate', '')
    embedding_text = f"{name} {region_type} {description} {climate}"
    search_vector = await generate_embedding(embedding_text)
    
    similar = await conn.fetchrow("""
        SELECT id, name, 1 - (embedding <=> $1) AS similarity
        FROM GeographicRegions
        WHERE 1 - (embedding <=> $1) > 0.85
        ORDER BY embedding <=> $1
        LIMIT 1
    """, search_vector)
    
    if similar:
        validation_agent = CanonValidationAgent()
        is_duplicate = await validation_agent.confirm_is_duplicate_region(
            conn,
            proposal={"name": name, "region_type": region_type, "description": description},
            existing_region_id=similar['id']
        )
        
        if is_duplicate:
            logger.info(f"Region '{name}' matched to existing ID {similar['id']}")
            return similar['id']
    
    # Create new region
    region_id = await conn.fetchval("""
        INSERT INTO GeographicRegions (
            name, region_type, description, climate, resources,
            governing_faction, population_density, major_settlements,
            cultural_traits, dangers, terrain_features,
            defensive_characteristics, strategic_value,
            matriarchal_influence, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        RETURNING id
    """,
    name, region_type, description, kwargs.get('climate'),
    kwargs.get('resources', []), kwargs.get('governing_faction'),
    kwargs.get('population_density'), kwargs.get('major_settlements', []),
    kwargs.get('cultural_traits', []), kwargs.get('dangers', []),
    kwargs.get('terrain_features', []), kwargs.get('defensive_characteristics'),
    kwargs.get('strategic_value', 5), kwargs.get('matriarchal_influence', 5),
    search_vector)
    
    await log_canonical_event(
        ctx, conn,
        f"Geographic region '{name}' established as {region_type} with strategic value {kwargs.get('strategic_value', 5)}",
        tags=["geography", "region", "canon"],
        significance=8
    )
    
    return region_id

async def create_political_entity(ctx, conn, **kwargs) -> int:
    """Create a political entity."""
    embedding_text = f"{kwargs['name']} {kwargs['entity_type']} {kwargs['description']}"
    embedding = await generate_embedding(embedding_text)
    
    entity_id = await conn.fetchval("""
        INSERT INTO PoliticalEntities (
            name, entity_type, description, region_id,
            governance_style, leadership_structure, population_scale,
            cultural_identity, economic_focus, political_values,
            matriarchy_level, relations, military_strength,
            diplomatic_stance, internal_conflicts, power_centers, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
        RETURNING id
    """,
    kwargs['name'], kwargs['entity_type'], kwargs['description'],
    kwargs.get('region_id'), kwargs['governance_style'],
    kwargs['leadership_structure'], kwargs['population_scale'],
    kwargs['cultural_identity'], kwargs['economic_focus'],
    kwargs['political_values'], kwargs['matriarchy_level'],
    json.dumps(kwargs.get('relations', {})), kwargs.get('military_strength', 5),
    kwargs['diplomatic_stance'], kwargs.get('internal_conflicts', []),
    json.dumps(kwargs.get('power_centers', [])), embedding)
    
    return entity_id

async def create_conflict_simulation(ctx, conn, **kwargs) -> int:
    """Create a conflict simulation record."""
    # Create embedding from primary actors
    actor_names = [a.get('name', '') for a in kwargs.get('primary_actors', [])]
    embed_text = f"{kwargs['conflict_type']} involving {', '.join(actor_names)}"
    embedding = await generate_embedding(embed_text)
    
    sim_id = await conn.fetchval("""
        INSERT INTO ConflictSimulations (
            conflict_type, primary_actors, timeline, intensity_progression,
            diplomatic_events, military_events, civilian_impact,
            resolution_scenarios, most_likely_outcome, duration_months,
            confidence_level, simulation_basis, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        RETURNING id
    """,
    kwargs['conflict_type'], json.dumps(kwargs['primary_actors']),
    json.dumps(kwargs['timeline']), kwargs['intensity_progression'],
    json.dumps(kwargs.get('diplomatic_events', [])),
    json.dumps(kwargs.get('military_events', [])),
    json.dumps(kwargs.get('civilian_impact', {})),
    json.dumps(kwargs.get('resolution_scenarios', [])),
    json.dumps(kwargs['most_likely_outcome']),
    kwargs['duration_months'], kwargs['confidence_level'],
    kwargs['simulation_basis'], embedding)
    
    return sim_id

async def create_border_dispute(ctx, conn, **kwargs) -> int:
    """Create a border dispute record."""
    embed_text = f"{kwargs['dispute_type']} {kwargs['description']} {kwargs['strategic_implications']}"
    embedding = await generate_embedding(embed_text)
    
    dispute_id = await conn.fetchval("""
        INSERT INTO BorderDisputes (
            region1_id, region2_id, dispute_type, description,
            severity, duration, causal_factors, status,
            resolution_attempts, strategic_implications,
            female_leaders_involved, gender_dynamics, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        RETURNING id
    """,
    kwargs['region1_id'], kwargs['region2_id'], kwargs['dispute_type'],
    kwargs['description'], kwargs['severity'], kwargs['duration'],
    kwargs['causal_factors'], kwargs['status'],
    json.dumps(kwargs.get('resolution_attempts', [])),
    kwargs['strategic_implications'], kwargs.get('female_leaders_involved', []),
    kwargs['gender_dynamics'], embedding)
    
    return dispute_id

async def update_border_dispute_resolution(ctx, conn, dispute_id: int, **kwargs) -> None:
    """Update a border dispute with resolution information."""
    await conn.execute("""
        UPDATE BorderDisputes
        SET status = $1,
            resolution_attempts = $2,
            strategic_implications = COALESCE($3, strategic_implications)
        WHERE id = $4
    """, kwargs['status'], json.dumps(kwargs['resolution_attempts']),
    kwargs.get('strategic_implications'), dispute_id)

async def find_or_create_urban_myth(
    ctx, conn,
    name: str,
    description: str,
    **kwargs
) -> int:
    """Find or create an urban myth with memory-enhanced semantic matching."""
    ctx = ensure_canonical_context(ctx)
    memory_orchestrator = await get_canon_memory_orchestrator(ctx.user_id, ctx.conversation_id)
    
    # Check exact match
    existing = await conn.fetchrow("""
        SELECT id FROM UrbanMyths WHERE name = $1
    """, name)
    
    if existing:
        return existing['id']
    
    # Memory-based semantic search
    narrative_style = kwargs.get('narrative_style', 'folklore')
    themes = kwargs.get('themes', [])
    search_text = f"Urban Myth: {name} - {description}. Style: {narrative_style}. Themes: {', '.join(themes)}"
    
    search_results = await memory_orchestrator.search_canonical_entities(
        query=search_text,
        entity_types=["myth", "urban_myth"],
        similarity_threshold=0.80
    )
    
    if search_results:
        for result in search_results:
            myth_id = result.get("metadata", {}).get("entity_id")
            if myth_id:
                validation_agent = CanonValidationAgent()
                is_duplicate = await validation_agent.confirm_is_duplicate_myth(
                    conn,
                    proposal={"name": name, "description": description, "themes": themes},
                    existing_myth_id=myth_id
                )
                
                if is_duplicate:
                    logger.info(f"Urban myth '{name}' matched to existing ID {myth_id}")
                    
                    # Update regions if new ones provided
                    regions_known = kwargs.get('regions_known', [])
                    if regions_known:
                        existing_regions = await conn.fetchval(
                            "SELECT regions_known FROM UrbanMyths WHERE id = $1",
                            myth_id
                        )
                        new_regions = list(set(existing_regions + regions_known))
                        
                        await conn.execute("""
                            UPDATE UrbanMyths
                            SET regions_known = $1, spread_rate = GREATEST(spread_rate, $2)
                            WHERE id = $3
                        """, new_regions, kwargs.get('spread_rate', 5), myth_id)
                    
                    return myth_id
    
    # Create new myth
    embedding = await memory_orchestrator.generate_embedding(search_text)
    
    myth_id = await conn.fetchval("""
        INSERT INTO UrbanMyths (
            name, description, origin_location, origin_event,
            believability, spread_rate, regions_known, narrative_style,
            themes, matriarchal_elements, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        RETURNING id
    """,
        name, description, kwargs.get('origin_location'),
        kwargs.get('origin_event'), kwargs.get('believability', 6),
        kwargs.get('spread_rate', 5), kwargs.get('regions_known', []),
        narrative_style, themes, kwargs.get('matriarchal_elements', []),
        embedding
    )
    
    # Store in memory system
    await memory_orchestrator.store_canonical_entity(
        entity_type="urban_myth",
        entity_id=myth_id,
        entity_name=name,
        entity_data={
            "description": description,
            "origin_location": kwargs.get('origin_location'),
            "believability": kwargs.get('believability', 6),
            "themes": themes,
            "narrative_style": narrative_style
        },
        significance=5
    )
    
    # Create semantic memories for the myth
    await memory_orchestrator.create_belief(
        entity_type="lore",
        entity_id=0,
        belief_text=f"The myth of {name}: {description}",
        confidence=kwargs.get('believability', 6) / 10.0
    )
    
    await log_canonical_event(
        ctx, conn,
        f"Urban myth '{name}' emerges in {kwargs.get('origin_location', 'the local area')}",
        tags=["myth", "folklore", "canon"],
        significance=5
    )
    
    return myth_id



async def create_local_history(ctx, conn, **kwargs) -> int:
    """Create a local historical event."""
    embedding_text = f"{kwargs['event_name']} {kwargs['description']} {kwargs['date_description']} {kwargs['narrative_category']}"
    embedding = await generate_embedding(embedding_text)
    
    event_id = await conn.fetchval("""
        INSERT INTO LocalHistories (
            location_id, event_name, description, date_description,
            significance, impact_type, notable_figures,
            current_relevance, commemoration, connected_myths,
            related_landmarks, narrative_category, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        RETURNING id
    """,
    kwargs['location_id'], kwargs['event_name'], kwargs['description'],
    kwargs['date_description'], kwargs['significance'], kwargs['impact_type'],
    kwargs.get('notable_figures', []), kwargs.get('current_relevance'),
    kwargs.get('commemoration'), kwargs.get('connected_myths', []),
    kwargs.get('related_landmarks', []), kwargs['narrative_category'], embedding)
    
    return event_id

async def find_or_create_landmark(ctx, conn, **kwargs) -> int:
    """Find or create a landmark with semantic matching."""
    location_id = kwargs['location_id']
    name = kwargs['name']
    landmark_type = kwargs['landmark_type']
    
    # Get location name for embedding
    location = await conn.fetchrow("SELECT location_name FROM Locations WHERE id = $1", location_id)
    if not location:
        raise ValueError(f"Location {location_id} not found")
    
    # Check exact match at location
    existing = await conn.fetchrow("""
        SELECT id FROM Landmarks
        WHERE name = $1 AND location_id = $2
    """, name, location_id)
    
    if existing:
        return existing['id']
    
    # Semantic similarity check
    embedding_text = f"{name} {landmark_type} {kwargs['description']} at {location['location_name']}"
    search_vector = await generate_embedding(embedding_text)
    
    similar = await conn.fetchrow("""
        SELECT id, name, landmark_type, 1 - (embedding <=> $1) AS similarity
        FROM Landmarks
        WHERE location_id = $2 AND 1 - (embedding <=> $1) > 0.85
        ORDER BY embedding <=> $1
        LIMIT 1
    """, search_vector, location_id)
    
    if similar:
        logger.info(f"Landmark '{name}' appears similar to existing '{similar['name']}' (ID: {similar['id']})")
        return similar['id']
    
    # Create new landmark
    landmark_id = await conn.fetchval("""
        INSERT INTO Landmarks (
            name, location_id, landmark_type, description,
            historical_significance, current_use, controlled_by,
            legends, connected_histories, architectural_style,
            symbolic_meaning, matriarchal_significance, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        RETURNING id
    """,
    name, location_id, landmark_type, kwargs['description'],
    kwargs.get('historical_significance'), kwargs.get('current_use'),
    kwargs.get('controlled_by'), kwargs.get('legends', []),
    kwargs.get('connected_histories', []), kwargs.get('architectural_style'),
    kwargs.get('symbolic_meaning'), kwargs.get('matriarchal_significance', 'moderate'),
    search_vector)
    
    await log_canonical_event(
        ctx, conn,
        f"Landmark '{name}' established at {location['location_name']} as {landmark_type}",
        tags=["landmark", "location", "canon"],
        significance=6
    )
    
    return landmark_id

async def update_urban_myth(ctx, conn, myth_id: int, updates: Dict[str, Any]) -> None:
    """Update an urban myth."""
    # Build update query dynamically
    set_clauses = []
    values = []
    for i, (key, value) in enumerate(updates.items()):
        set_clauses.append(f"{key} = ${i+1}")
        values.append(value)
    
    if not set_clauses:
        return
    
    values.append(myth_id)
    query = f"UPDATE UrbanMyths SET {', '.join(set_clauses)} WHERE id = ${len(values)}"
    
    await conn.execute(query, *values)
    
    # Update embedding if description changed
    if 'description' in updates:
        myth = await conn.fetchrow("SELECT name FROM UrbanMyths WHERE id = $1", myth_id)
        if myth:
            embedding_text = f"{myth['name']} {updates['description']}"
            embedding = await generate_embedding(embedding_text)
            await conn.execute("UPDATE UrbanMyths SET embedding = $1 WHERE id = $2", embedding, myth_id)

async def update_landmark(ctx, conn, landmark_id: int, updates: Dict[str, Any]) -> None:
    """Update a landmark."""
    set_clauses = []
    values = []
    for i, (key, value) in enumerate(updates.items()):
        set_clauses.append(f"{key} = ${i+1}")
        values.append(value)
    
    if not set_clauses:
        return
    
    values.append(landmark_id)
    query = f"UPDATE Landmarks SET {', '.join(set_clauses)} WHERE id = ${len(values)}"
    
    await conn.execute(query, *values)

# Add these functions to canon.py

async def find_or_create_location(ctx, conn, location_name: str, **kwargs) -> str:
    """Find or create a location with existence gate validation."""
    # First check if location exists
    existing = await conn.fetchrow("""
        SELECT id, location_name, description
        FROM Locations
        WHERE LOWER(location_name) = LOWER($1)
        AND user_id = $2 AND conversation_id = $3
    """, location_name, ctx.user_id, ctx.conversation_id)
    
    if existing:
        logger.info(f"Location '{location_name}' found via exact match")
        return existing['location_name']
    
    # Run existence gate check BEFORE semantic similarity
    gate = ExistenceGate(ctx)
    location_data = {
        'location_name': location_name,
        'location_type': kwargs.get('location_type', 'settlement'),
        'district_type': kwargs.get('district_type'),
        'description': kwargs.get('description', f"The area known as {location_name}"),
        **kwargs
    }
    
    # Get current scene context for topology checks
    scene_context = await conn.fetchrow("""
        SELECT value FROM CurrentRoleplay
        WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentScene'
    """, ctx.user_id, ctx.conversation_id)
    
    if scene_context and scene_context['value']:
        scene_data = json.loads(scene_context['value'])
    else:
        scene_data = None
    
    decision, details = await gate.assess_entity('location', location_data, scene_data)
    
    if decision == GateDecision.DENY:
        # Location cannot exist - throw exception
        raise ValueError(f"Location '{location_name}' cannot exist: {details['reason']}")
    
    elif decision == GateDecision.DEFER:
        # Return as a lead/rumor, not canon
        logger.warning(f"Location '{location_name}' deferred: {details['reason']}")
        return f"LEAD::{details.get('lead', 'Location rumored to exist elsewhere')}"
    
    elif decision == GateDecision.ANALOG:
        # Use the analog instead
        analog_name = details['analog']
        analog_data = details.get('analog_data', {})
        logger.info(f"Substituting '{location_name}' with analog '{analog_name}'")
        
        # Update the creation data with analog
        location_name = analog_name
        location_data = analog_data
        kwargs.update(analog_data)
    
    # Now proceed with semantic similarity check
    memory_orchestrator = await get_canon_memory_orchestrator(ctx.user_id, ctx.conversation_id)
    description = location_data.get('description', f"The area known as {location_name}")
    embedding_text = f"Location: {location_name} - {description}"
    
    search_results = await memory_orchestrator.search_vector_store(
        query=embedding_text,
        entity_type="location",
        top_k=1,
        filter_dict={"user_id": ctx.user_id, "conversation_id": ctx.conversation_id}
    )
    
    if search_results and search_results[0].get("similarity", 0) > 0.85:
        similar = search_results[0]
        similar_name = similar.get("metadata", {}).get("location_name")
        
        if similar_name:
            validation_agent = CanonValidationAgent()
            prompt = f"""
            Are these the same location?
            
            Proposed: {location_name}
            Description: {description}
            
            Existing: {similar_name}
            Similarity: {similar.get("similarity", 0):.2f}
            
            Answer only 'true' or 'false'.
            """
            
            from agents import Runner
            result = await Runner.run(validation_agent.agent, prompt)
            if result.final_output.strip().lower() == 'true':
                logger.info(f"Location '{location_name}' matched to existing '{similar_name}'")
                return similar_name
    
    # Create new location (already passed existence gate)
    search_vector = await memory_orchestrator.generate_embedding(embedding_text)
    
    location_id = await conn.fetchval("""
        INSERT INTO Locations (
            user_id, conversation_id, location_name, description,
            location_type, parent_location, cultural_significance,
            economic_importance, strategic_value, population_density,
            notable_features, hidden_aspects, access_restrictions,
            local_customs, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        RETURNING id
    """,
        ctx.user_id, ctx.conversation_id, location_name, description,
        kwargs.get('location_type', 'settlement'),
        kwargs.get('parent_location'),
        kwargs.get('cultural_significance', 'moderate'),
        kwargs.get('economic_importance', 'moderate'),
        kwargs.get('strategic_value', 5),
        kwargs.get('population_density', 'moderate'),
        json.dumps(kwargs.get('notable_features', [])),
        json.dumps(kwargs.get('hidden_aspects', [])),
        json.dumps(kwargs.get('access_restrictions', [])),
        json.dumps(kwargs.get('local_customs', [])),
        search_vector
    )
    
    # Store in memory system
    from memory.memory_orchestrator import EntityType
    await memory_orchestrator.store_memory(
        entity_type=EntityType.LORE,
        entity_id=0,
        memory_text=f"Location '{location_name}' established: {description}",
        significance=0.8,
        tags=['location', 'creation', 'canon'],
        metadata={
            "location_id": location_id,
            "location_name": location_name,
            "location_type": kwargs.get('location_type', 'settlement')
        }
    )
    
    # Add to vector store
    await memory_orchestrator.add_to_vector_store(
        text=embedding_text,
        metadata={
            "entity_type": "location",
            "location_id": location_id,
            "location_name": location_name,
            "user_id": ctx.user_id,
            "conversation_id": ctx.conversation_id
        },
        entity_type="location"
    )
    
    await log_canonical_event(
        ctx, conn,
        f"Location '{location_name}' established in the world",
        tags=['location', 'creation', 'canon'],
        significance=6
    )
    
    return location_name
    
async def find_or_create_faction(ctx, conn, faction_name: str, **kwargs) -> int:
    """
    Find or create a faction with semantic matching.
    """
    ctx = ensure_canonical_context(ctx)
    
    # Check exact match
    existing = await conn.fetchrow("""
        SELECT id FROM Factions
        WHERE LOWER(name) = LOWER($1)
        AND user_id = $2 AND conversation_id = $3
    """, faction_name, ctx.user_id, ctx.conversation_id)
    
    if existing:
        return existing['id']
    
    # Semantic similarity check
    faction_type = kwargs.get('type', 'organization')
    description = kwargs.get('description', '')
    embedding_text = f"{faction_name} {faction_type} {description}"
    search_vector = await generate_embedding(embedding_text)
    
    similar = await conn.fetchrow("""
        SELECT id, name, type, description, 1 - (embedding <=> $1) AS similarity
        FROM Factions
        WHERE user_id = $2 AND conversation_id = $3
        AND embedding IS NOT NULL
        AND 1 - (embedding <=> $1) > 0.85
        ORDER BY embedding <=> $1
        LIMIT 1
    """, search_vector, ctx.user_id, ctx.conversation_id)
    
    if similar:
        validation_agent = CanonValidationAgent()
        prompt = f"""
        Are these the same faction?
        
        Proposed: {faction_name} ({faction_type})
        Description: {description[:200]}
        
        Existing: {similar['name']} ({similar['type']})
        Description: {similar['description'][:200]}
        
        Answer only 'true' or 'false'.
        """
        
        result = await Runner.run(validation_agent.agent, prompt)
        if result.final_output.strip().lower() == 'true':
            return similar['id']
    
    # Create new faction
    faction_id = await conn.fetchval("""
        INSERT INTO Factions (
            user_id, conversation_id, name, type, description,
            values, goals, hierarchy, resources, territory,
            rivals, allies, public_reputation, secret_activities,
            power_level, influence_scope, recruitment_methods,
            leadership_structure, founding_story, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
        RETURNING id
    """,
        ctx.user_id, ctx.conversation_id, faction_name, faction_type, description,
        kwargs.get('values', ['power', 'control']),
        kwargs.get('goals', ['expansion', 'influence']),
        kwargs.get('hierarchy', 'strict'),
        kwargs.get('resources', []),
        kwargs.get('territory', []),
        kwargs.get('rivals', []),
        kwargs.get('allies', []),
        kwargs.get('public_reputation', 'neutral'),
        kwargs.get('secret_activities', []),
        kwargs.get('power_level', 5),
        kwargs.get('influence_scope', 'local'),
        kwargs.get('recruitment_methods', []),
        kwargs.get('leadership_structure', {}),
        kwargs.get('founding_story', ''),
        search_vector
    )
    
    await log_canonical_event(
        ctx, conn,
        f"Faction '{faction_name}' established as {faction_type}",
        tags=['faction', 'creation', 'canon'],
        significance=7
    )
    
    return faction_id

async def find_or_create_historical_event(ctx, conn, event_name: str, **kwargs) -> int:
    """
    Find or create a historical event with semantic matching.
    """
    ctx = ensure_canonical_context(ctx)
    
    # Check exact match
    existing = await conn.fetchrow("""
        SELECT id FROM HistoricalEvents
        WHERE LOWER(name) = LOWER($1)
        AND user_id = $2 AND conversation_id = $3
    """, event_name, ctx.user_id, ctx.conversation_id)
    
    if existing:
        return existing['id']
    
    # Semantic similarity check  
    description = kwargs.get('description', '')
    date_description = kwargs.get('date_description', 'Unknown time')
    embedding_text = f"{event_name} {description} {date_description}"
    search_vector = await generate_embedding(embedding_text)
    
    similar = await conn.fetchrow("""
        SELECT id, name, description, date_description,
               1 - (embedding <=> $1) AS similarity
        FROM HistoricalEvents
        WHERE user_id = $2 AND conversation_id = $3
        AND embedding IS NOT NULL
        AND 1 - (embedding <=> $1) > 0.85
        ORDER BY embedding <=> $1
        LIMIT 1
    """, search_vector, ctx.user_id, ctx.conversation_id)
    
    if similar:
        validation_agent = CanonValidationAgent()
        prompt = f"""
        Are these the same historical event?
        
        Proposed: {event_name}
        When: {date_description}
        Description: {description[:200]}
        
        Existing: {similar['name']}
        When: {similar['date_description']}
        Description: {similar['description'][:200]}
        
        Answer only 'true' or 'false'.
        """
        
        result = await Runner.run(validation_agent.agent, prompt)
        if result.final_output.strip().lower() == 'true':
            return similar['id']
    
    # Create new event
    event_id = await conn.fetchval("""
        INSERT INTO HistoricalEvents (
            user_id, conversation_id, name, description,
            date_description, event_type, significance,
            involved_entities, location, consequences,
            cultural_impact, disputed_facts, commemorations,
            primary_sources, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        RETURNING id
    """,
        ctx.user_id, ctx.conversation_id, event_name, description,
        date_description,
        kwargs.get('event_type', 'political'),
        kwargs.get('significance', 5),
        kwargs.get('involved_entities', []),
        kwargs.get('location'),
        kwargs.get('consequences', []),
        kwargs.get('cultural_impact', 'moderate'),
        kwargs.get('disputed_facts', []),
        kwargs.get('commemorations', []),
        kwargs.get('primary_sources', []),
        search_vector
    )
    
    await log_canonical_event(
        ctx, conn,
        f"Historical event '{event_name}' recorded for {date_description}",
        tags=['history', 'event', 'canon'],
        significance=kwargs.get('significance', 5)
    )
    
    return event_id

async def find_or_create_notable_figure(ctx, conn, figure_name: str, **kwargs) -> int:
    """
    Find or create a notable figure with semantic matching.
    """
    ctx = ensure_canonical_context(ctx)
    
    # Check exact match
    existing = await conn.fetchrow("""
        SELECT id FROM NotableFigures
        WHERE LOWER(name) = LOWER($1)
        AND user_id = $2 AND conversation_id = $3
    """, figure_name, ctx.user_id, ctx.conversation_id)
    
    if existing:
        return existing['id']
    
    # Semantic similarity check
    description = kwargs.get('description', '')
    title = kwargs.get('title', '')
    embedding_text = f"{figure_name} {title} {description}"
    search_vector = await generate_embedding(embedding_text)
    
    similar = await conn.fetchrow("""
        SELECT id, name, title, description,
               1 - (embedding <=> $1) AS similarity
        FROM NotableFigures
        WHERE user_id = $2 AND conversation_id = $3
        AND embedding IS NOT NULL
        AND 1 - (embedding <=> $1) > 0.85
        ORDER BY embedding <=> $1
        LIMIT 1
    """, search_vector, ctx.user_id, ctx.conversation_id)
    
    if similar:
        validation_agent = CanonValidationAgent()
        prompt = f"""
        Are these the same person?
        
        Proposed: {figure_name} ({title})
        Description: {description[:200]}
        
        Existing: {similar['name']} ({similar['title'] or 'no title'})
        Description: {similar['description'][:200]}
        
        Answer only 'true' or 'false'.
        """
        
        result = await Runner.run(validation_agent.agent, prompt)
        if result.final_output.strip().lower() == 'true':
            return similar['id']
    
    # Create new figure - CONVERT ALL LISTS/DICTS TO JSON STRINGS
    figure_id = await conn.fetchval("""
        INSERT INTO NotableFigures (
            user_id, conversation_id, name, title, description,
            birth_date, death_date, faction_affiliations,
            achievements, failures, personality_traits,
            public_image, hidden_aspects, influence_areas,
            legacy, controversial_actions, relationships,
            current_status, reputation, significance, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
        RETURNING id
    """,
        ctx.user_id, ctx.conversation_id, figure_name, title, description,
        kwargs.get('birth_date'),
        kwargs.get('death_date'),
        json.dumps(kwargs.get('faction_affiliations', [])),  # Convert to JSON string
        json.dumps(kwargs.get('achievements', [])),          # Convert to JSON string
        json.dumps(kwargs.get('failures', [])),              # Convert to JSON string
        json.dumps(kwargs.get('personality_traits', [])),    # Convert to JSON string
        kwargs.get('public_image', 'neutral'),
        json.dumps(kwargs.get('hidden_aspects', [])),        # Convert to JSON string
        json.dumps(kwargs.get('influence_areas', [])),       # Convert to JSON string
        kwargs.get('legacy', ''),
        json.dumps(kwargs.get('controversial_actions', [])), # Convert to JSON string
        json.dumps(kwargs.get('relationships', [])),         # Convert to JSON string
        kwargs.get('current_status', 'active'),
        kwargs.get('reputation', 50),
        kwargs.get('significance', 5),
        search_vector
    )
    
    await log_canonical_event(
        ctx, conn,
        f"Notable figure '{figure_name}' ({title}) entered historical record",
        tags=['figure', 'notable', 'canon'],
        significance=kwargs.get('significance', 5)
    )
    
    return figure_id

async def update_entity_canonically(ctx, conn, entity_type: str, entity_id: int, updates: Dict[str, Any], reason: str):
    """
    Update any entity through the lore system for canonical consistency.
    """
    from lore.core.lore_system import LoreSystem
    
    lore_system = await LoreSystem.get_instance(ctx.user_id, ctx.conversation_id)
    
    # Map entity types to their primary key columns
    primary_key_map = {
        "NPCStats": "npc_id",
        "Locations": "id",
        "Events": "id",
        "Factions": "id",
        # Add other mappings as needed
    }
    
    primary_key = primary_key_map.get(entity_type, "id")
    
    result = await lore_system.propose_and_enact_change(
        ctx,
        entity_type=entity_type,
        entity_identifier={primary_key: entity_id},  # Use correct primary key
        updates=updates,
        reason=reason
    )
    
    return result

async def update_entity_with_governance(
    ctx,
    conn,
    entity_type: str,
    entity_id: int,
    updates: Dict[str, Any],
    reason: str,
    significance: int = 7
) -> Dict[str, Any]:
    """
    Update any entity with governance tracking, validation, and memory integration.
    """
    ctx = ensure_canonical_context(ctx)
    memory_orchestrator = await get_canon_memory_orchestrator(ctx.user_id, ctx.conversation_id)
    
    try:
        # Build the update query
        set_clauses = []
        values = []
        for i, (field, value) in enumerate(updates.items()):
            set_clauses.append(f"{field} = ${i+1}")
            values.append(value)
        
        if not set_clauses:
            return {"status": "error", "message": "No updates provided"}
        
        values.append(entity_id)
        
        query = f"""
            UPDATE {entity_type}
            SET {', '.join(set_clauses)}
            WHERE id = ${len(values)}
            RETURNING *
        """
        
        result = await conn.fetchrow(query, *values)
        
        if not result:
            return {"status": "error", "message": f"Entity not found: {entity_type} with id {entity_id}"}
        
        # Log the canonical event
        event_text = f"{entity_type} (ID: {entity_id}) updated: {reason}. Changes: {json.dumps(updates)}"
        await log_canonical_event(
            ctx, conn,
            event_text,
            tags=[entity_type.lower(), 'update', 'governance'],
            significance=significance
        )
        
        # Store update as a memory
        await memory_orchestrator.store_memory(
            entity_type=EntityType.LORE,
            entity_id=0,
            memory_text=event_text,
            significance=convert_importance_to_significance("medium" if significance < 7 else "high"),
            tags=[entity_type.lower(), 'update', 'governance'],
            metadata={
                "entity_type": entity_type,
                "entity_id": entity_id,
                "updates": updates,
                "reason": reason
            }
        )
        
        # Update embedding if text fields changed
        text_fields = ['name', 'description', 'title']
        if any(field in updates for field in text_fields):
            embedding_parts = []
            for field in text_fields:
                if field in result:
                    embedding_parts.append(str(result[field]))
            
            if embedding_parts:
                embedding_text = ' '.join(embedding_parts)
                embedding = await memory_orchestrator.generate_embedding(embedding_text)
                
                await conn.execute(
                    f"UPDATE {entity_type} SET embedding = $1 WHERE id = $2",
                    embedding, entity_id
                )
                
                # Update vector store
                await memory_orchestrator.add_to_vector_store(
                    text=embedding_text,
                    metadata={
                        "entity_type": entity_type.lower(),
                        "entity_id": entity_id,
                        "user_id": ctx.user_id,
                        "conversation_id": ctx.conversation_id
                    },
                    entity_type=entity_type.lower()
                )
        
        return {
            "status": "success",
            "entity_type": entity_type,
            "entity_id": entity_id,
            "updates": updates,
            "reason": reason
        }
        
    except Exception as e:
        logger.error(f"Error updating {entity_type} (ID: {entity_id}): {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "entity_type": entity_type,
            "entity_id": entity_id
        }

async def get_entity_by_id(ctx, conn, entity_type: str, entity_id: int) -> Optional[Dict[str, Any]]:
    """
    Get an entity by its ID from any table.
    
    Args:
        ctx: Context
        conn: Database connection  
        entity_type: Table name
        entity_id: Entity ID
        
    Returns:
        Entity data as dictionary or None
    """
    try:
        query = f"SELECT * FROM {entity_type} WHERE id = $1"
        row = await conn.fetchrow(query, entity_id)
        return dict(row) if row else None
    except Exception as e:
        logger.error(f"Error fetching {entity_type} with ID {entity_id}: {str(e)}")
        return None

async def find_entity_by_name(
    ctx, 
    conn, 
    entity_type: str, 
    name: str,
    name_field: str = "name"
) -> Optional[Dict[str, Any]]:
    """
    Find an entity by name with fuzzy matching.
    
    Args:
        ctx: Context
        conn: Database connection
        entity_type: Table name
        name: Name to search for
        name_field: Name of the field containing the name
        
    Returns:
        Best matching entity or None
    """
    try:
        # First try exact match
        query = f"SELECT * FROM {entity_type} WHERE LOWER({name_field}) = LOWER($1)"
        row = await conn.fetchrow(query, name)
        if row:
            return dict(row)
        
        # Try fuzzy match
        query = f"""
            SELECT *, similarity({name_field}, $1) as sim
            FROM {entity_type}
            WHERE similarity({name_field}, $1) > 0.6
            ORDER BY sim DESC
            LIMIT 1
        """
        row = await conn.fetchrow(query, name)
        return dict(row) if row else None
        
    except Exception as e:
        logger.error(f"Error finding {entity_type} by name '{name}': {str(e)}")
        return None

async def create_message(ctx, conn, conversation_id: int, sender: str, content: str) -> int:
    """
    Create a message canonically.
    """
    message_id = await conn.fetchval("""
        INSERT INTO messages (conversation_id, sender, content, created_at)
        VALUES ($1, $2, $3, $4)
        RETURNING id
    """, conversation_id, sender, content, datetime.utcnow())
    
    await log_canonical_event(
        ctx, conn,
        f"Message created from {sender} in conversation {conversation_id}",
        tags=["message", "creation"],
        significance=2
    )
    
    return message_id

async def update_current_roleplay(ctx, conn, key: str, value: str) -> None:
    """
    Update a CurrentRoleplay value canonically.
    """
    # Ensure we have a proper context
    ctx = ensure_canonical_context(ctx)
    
    await conn.execute("""
        INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (user_id, conversation_id, key)
        DO UPDATE SET value = EXCLUDED.value
    """, ctx.user_id, ctx.conversation_id, key, value)
    
    await log_canonical_event(
        ctx, conn,
        f"CurrentRoleplay updated: {key} = {value}",
        tags=["roleplay", "update", key.lower()],
        significance=3
    )

    
async def find_or_create_social_link(ctx, conn, **kwargs) -> int:
    """Create relationship using memory-enhanced system."""
    ctx = ensure_canonical_context(ctx)
    memory_orchestrator = await get_canon_memory_orchestrator(ctx.user_id, ctx.conversation_id)
    
    import inspect
    
    # Check if being called from dynamic_relationships to avoid recursion
    for frame_info in inspect.stack():
        if 'dynamic_relationships' in frame_info.filename:
            logger.warning("Detected recursive call from dynamic_relationships - handling directly")
            
            # Handle creation directly (existing code)
            e1 = (kwargs['entity1_type'], kwargs['entity1_id'])
            e2 = (kwargs['entity2_type'], kwargs['entity2_id'])
            if e1 <= e2:
                canonical_key = f"{e1[0]}_{e1[1]}_{e2[0]}_{e2[1]}"
            else:
                canonical_key = f"{e2[0]}_{e2[1]}_{e1[0]}_{e1[1]}"
            
            existing = await conn.fetchrow("""
                SELECT link_id FROM SocialLinks
                WHERE user_id = $1 AND conversation_id = $2 AND canonical_key = $3
            """, ctx.user_id, ctx.conversation_id, canonical_key)
            
            if existing:
                return existing['link_id']
            
            link_id = await conn.fetchval("""
                INSERT INTO SocialLinks (
                    user_id, conversation_id,
                    entity1_type, entity1_id, entity2_type, entity2_id,
                    canonical_key, dynamics, momentum, contexts,
                    patterns, archetypes, version,
                    last_interaction, created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, 
                        $10::jsonb, $11::jsonb, $12::jsonb, $13, $14, $15)
                RETURNING link_id
            """,
                ctx.user_id, ctx.conversation_id,
                kwargs['entity1_type'], kwargs['entity1_id'],
                kwargs['entity2_type'], kwargs['entity2_id'],
                canonical_key,
                json.dumps({}),
                json.dumps({'velocities': {}, 'inertia': 50.0}),
                json.dumps({'base_dimensions': {}, 'context_deltas': {}}),
                json.dumps([]),
                json.dumps([]),
                0,
                datetime.now(),
                datetime.now()
            )
            
            return link_id
    
    # Normal path - use memory system for relationship tracking
    from logic.dynamic_relationships import OptimizedRelationshipManager
    
    user_id = kwargs.get('user_id', ctx.user_id)
    conversation_id = kwargs.get('conversation_id', ctx.conversation_id)
    
    manager = OptimizedRelationshipManager(user_id, conversation_id)
    
    state = await manager.get_relationship_state(
        entity1_type=kwargs['entity1_type'],
        entity1_id=kwargs['entity1_id'],
        entity2_type=kwargs['entity2_type'],
        entity2_id=kwargs['entity2_id']
    )
    
    # Store relationship in memory system
    relationship_text = f"{kwargs['entity1_type']} {kwargs['entity1_id']} forms relationship with {kwargs['entity2_type']} {kwargs['entity2_id']}"
    
    if kwargs.get('link_type'):
        relationship_text += f" of type: {kwargs['link_type']}"
        
        # Map old link types to dimension changes
        link_type_mappings = {
            'friendly': {'affection': 30, 'trust': 20},
            'hostile': {'affection': -30, 'trust': -20, 'respect': -10},
            'romantic': {'affection': 50, 'fascination': 40},
            'mentor': {'respect': 40, 'influence': -30},
            'rival': {'respect': 30, 'affection': -20, 'volatility': 30}
        }
        
        if kwargs['link_type'] in link_type_mappings:
            for dim, value in link_type_mappings[kwargs['link_type']].items():
                if hasattr(state.dimensions, dim):
                    setattr(state.dimensions, dim, value)
    
    # Store as memory for both entities
    for entity_type, entity_id in [
        (kwargs['entity1_type'], kwargs['entity1_id']),
        (kwargs['entity2_type'], kwargs['entity2_id'])
    ]:
        await memory_orchestrator.store_memory(
            entity_type=entity_type,
            entity_id=entity_id,
            memory_text=relationship_text,
            significance=convert_importance_to_significance("medium"),
            tags=["relationship", "social_link", kwargs.get('link_type', 'neutral')],
            metadata={
                "link_id": state.link_id,
                "other_entity": f"{kwargs['entity2_type']}_{kwargs['entity2_id']}" 
                               if entity_type == kwargs['entity1_type'] 
                               else f"{kwargs['entity1_type']}_{kwargs['entity1_id']}"
            }
        )
    
    if kwargs.get('link_level'):
        level = kwargs['link_level']
        state.dimensions.affection = level * 0.4
        state.dimensions.trust = level * 0.3
        state.dimensions.intimacy = level * 0.3
    
    state.dimensions.clamp()
    await manager._queue_update(state)
    await manager._flush_updates()
    
    return state.link_id or 0


    
async def find_or_create_npc_group(ctx, conn, group_data: Dict[str, Any]) -> int:
    """
    Find or create an NPC group.
    """
    ctx = ensure_canonical_context(ctx)
    group_name = group_data['name']
    
    # Check if group exists
    existing = await conn.fetchrow("""
        SELECT group_id FROM NPCGroups
        WHERE user_id = $1 AND conversation_id = $2 AND group_name = $3
    """, ctx.user_id, ctx.conversation_id, group_name)
    
    if existing:
        return existing['group_id']
    
    # Create new group
    group_id = await conn.fetchval("""
        INSERT INTO NPCGroups (
            user_id, conversation_id, group_name, group_data, updated_at
        )
        VALUES ($1, $2, $3, $4::jsonb, NOW())
        RETURNING group_id
    """, ctx.user_id, ctx.conversation_id, group_name, json.dumps(group_data))
    
    # Log canonical event
    await log_canonical_event(
        ctx, conn,
        f"NPC group '{group_name}' created with {len(group_data.get('members', []))} members",
        tags=['npc_group', 'creation'],
        significance=6
    )
    
    return group_id

async def create_journal_entry(ctx, conn, entry_type: str, entry_text: str, **kwargs) -> int:
    """
    Create a journal entry - now integrated with memory system.
    """
    ctx = ensure_canonical_context(ctx)
    memory_orchestrator = await get_canon_memory_orchestrator(ctx.user_id, ctx.conversation_id)
    
    # Handle legacy parameters
    narrative_moment = kwargs.get('narrative_moment')
    if isinstance(narrative_moment, bool):
        narrative_moment = "true" if narrative_moment else None
    
    fantasy_flag = kwargs.get('fantasy_flag', False)
    
    # Determine significance
    significance = kwargs.get('importance', 0.5)
    if kwargs.get('intensity_level'):
        significance = max(significance, kwargs.get('intensity_level', 0) / 10.0)
    if narrative_moment == "true":
        significance = max(significance, 0.7)
    
    # Build tags
    tags = kwargs.get('tags', [])
    if entry_type:
        tags.append(entry_type)
    if kwargs.get('revelation_types'):
        tags.append(kwargs.get('revelation_types'))
    if fantasy_flag:
        tags.append('fantasy')
    
    # Collect metadata
    metadata = kwargs.get('entry_metadata', {})
    metadata.update({
        'entry_type': entry_type,
        'revelation_types': kwargs.get('revelation_types'),
        'narrative_moment': narrative_moment,
        'fantasy_flag': fantasy_flag,
        'intensity_level': kwargs.get('intensity_level'),
        **{k: v for k, v in kwargs.items() 
           if k not in ['tags', 'importance', 'entry_metadata']}
    })
    
    metadata = {k: v for k, v in metadata.items() if v is not None}
    
    # Generate embedding using memory orchestrator
    embedding = await memory_orchestrator.generate_embedding(entry_text)
    
    # Store in database
    entry_id = await conn.fetchval("""
        INSERT INTO PlayerJournal (
            user_id, conversation_id, entry_type, entry_text,
            importance, tags, entry_metadata, embedding,
            created_at
        )
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8, CURRENT_TIMESTAMP)
        RETURNING id
    """,
        ctx.user_id, 
        ctx.conversation_id, 
        entry_type,
        entry_text,
        significance,
        json.dumps(tags),
        json.dumps(metadata),
        embedding
    )
    
    # Get player name from context or default
    player_name = kwargs.get('player_name', 'Chase')
    
    # Store as a memory in the memory system
    importance = "high" if significance > 0.7 else "medium" if significance > 0.3 else "low"
    
    await memory_orchestrator.store_memory(
        entity_type=EntityType.PLAYER,
        entity_id=ctx.user_id,
        memory_text=entry_text,
        importance=importance,
        emotional=True,
        tags=tags + ["journal_entry"],
        metadata={
            "journal_id": entry_id,
            "entry_type": entry_type,
            "player_name": player_name,
            **metadata
        }
    )
    
    # If it's a significant journal entry, also trigger memory analysis
    if significance > 0.7:
        await memory_orchestrator.add_journal_entry(
            player_name=player_name,
            entry_text=entry_text,
            entry_type=entry_type,
            fantasy_flag=fantasy_flag,
            intensity_level=kwargs.get('intensity_level', 0)
        )
    
    return entry_id

# use this for new code


async def add_journal_entry(ctx, conn, entry_text: str, significance: float = None) -> int:
    """
    Simplified journal entry creation integrated with memory system.
    """
    ctx = ensure_canonical_context(ctx)
    memory_orchestrator = await get_canon_memory_orchestrator(ctx.user_id, ctx.conversation_id)
    
    # Let memory orchestrator determine significance if not provided
    if significance is None:
        prompt = await memory_orchestrator.generate_memory_prompt(
            entity_type="player",
            entity_id=ctx.user_id,
            context={"entry_text": entry_text[:500]},
            prompt_type="analysis"
        )
        
        # Use the prompt to analyze significance
        significance = 0.5  # Default if analysis fails
    
    # Generate smart tags using memory orchestrator
    questions = await memory_orchestrator.generate_memory_questions(
        entity_type="player",
        entity_id=ctx.user_id,
        purpose="exploration"
    )
    
    # Extract tags from the questions (simplified approach)
    tags = []
    for q in questions[:3]:
        if "memory" in q.lower():
            tags.append("reflection")
        if "feel" in q.lower() or "emotion" in q.lower():
            tags.append("emotional")
        if "change" in q.lower():
            tags.append("transformation")
    
    # Generate embedding
    embedding = await memory_orchestrator.generate_embedding(entry_text)
    
    entry_id = await conn.fetchval("""
        INSERT INTO PlayerJournal (
            user_id, conversation_id, entry_text,
            importance, tags, embedding, created_at
        )
        VALUES ($1, $2, $3, $4, $5::jsonb, $6, CURRENT_TIMESTAMP)
        RETURNING id
    """,
        ctx.user_id, ctx.conversation_id, entry_text,
        significance, json.dumps(tags), embedding
    )
    
    # Store in memory system for better retrieval
    importance = "high" if significance > 0.7 else "medium" if significance > 0.3 else "low"
    
    await memory_orchestrator.store_memory(
        entity_type=EntityType.PLAYER,
        entity_id=ctx.user_id,
        memory_text=entry_text,
        importance=importance,
        emotional=True,
        tags=tags,
        metadata={"journal_id": entry_id}
    )
    
    return entry_id


async def ensure_addiction_table_exists(ctx, conn):
    """Ensure the PlayerAddictions table exists."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS PlayerAddictions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            conversation_id INTEGER NOT NULL,
            player_name VARCHAR(255) NOT NULL,
            addiction_type VARCHAR(50) NOT NULL,
            level INTEGER NOT NULL DEFAULT 0,
            target_npc_id INTEGER NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, conversation_id, player_name, addiction_type, target_npc_id),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
    """)
    
    # Create index for faster lookups
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_player_addictions_lookup
        ON PlayerAddictions(user_id, conversation_id, player_name)
    """)

async def find_or_create_addiction(
    ctx, conn, 
    player_name: str, 
    addiction_type: str, 
    level: int, 
    target_npc_id: Optional[int] = None
) -> int:
    """
    Find or create an addiction entry.
    Updates the level if the addiction already exists.
    """
    ctx = ensure_canonical_context(ctx)
    # Ensure table exists
    await ensure_addiction_table_exists(ctx, conn)
    
    # Use UPSERT pattern
    addiction_id = await conn.fetchval("""
        INSERT INTO PlayerAddictions
        (user_id, conversation_id, player_name, addiction_type, level, target_npc_id, last_updated)
        VALUES ($1, $2, $3, $4, $5, $6, NOW())
        ON CONFLICT (user_id, conversation_id, player_name, addiction_type, target_npc_id)
        DO UPDATE SET 
            level = EXCLUDED.level, 
            last_updated = NOW()
        RETURNING id
    """,
        ctx.user_id, ctx.conversation_id, player_name, addiction_type,
        level, target_npc_id
    )
    
    # Log canonical event if it's a new addiction or significant change
    if level >= 3:
        npc_clause = f" to NPC {target_npc_id}" if target_npc_id else ""
        await log_canonical_event(
            ctx, conn,
            f"Player {player_name} developed {ADDICTION_LEVELS[level]} addiction to {addiction_type}{npc_clause}",
            tags=['addiction', 'player_state', addiction_type],
            significance=6 if level == 3 else 8 if level == 4 else 4
        )
    
    return addiction_id

async def find_or_create_player_stats(ctx, conn, player_name: str, **initial_stats) -> None:
    """Find or create player stats with memory integration."""
    ctx = ensure_canonical_context(ctx)
    memory_orchestrator = await get_canon_memory_orchestrator(ctx.user_id, ctx.conversation_id)
    
    # Check if player stats exist
    exists = await conn.fetchval("""
        SELECT COUNT(*) FROM PlayerStats
        WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
    """, ctx.user_id, ctx.conversation_id, player_name)
    
    if not exists:
        # Set defaults
        stats = {
            'corruption': 0,
            'confidence': 0,
            'willpower': 0,
            'obedience': 0,
            'dependency': 0,
            'lust': 0,
            'mental_resilience': 0,
            'physical_endurance': 0
        }
        stats.update(initial_stats)
        
        # Create player stats
        await conn.execute("""
            INSERT INTO PlayerStats (
                user_id, conversation_id, player_name,
                corruption, confidence, willpower, obedience,
                dependency, lust, mental_resilience, physical_endurance
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """, 
            ctx.user_id, ctx.conversation_id, player_name,
            stats['corruption'], stats['confidence'], stats['willpower'], 
            stats['obedience'], stats['dependency'], stats['lust'],
            stats['mental_resilience'], stats['physical_endurance']
        )
        
        # Initialize player in memory system
        await memory_orchestrator.setup_entity(
            entity_type="player",
            entity_data={
                "player_name": player_name,
                "initial_stats": stats
            }
        )
        
        # Store initialization as a memory
        await memory_orchestrator.store_memory(
            entity_type=EntityType.PLAYER,
            entity_id=ctx.user_id,
            memory_text=f"Character {player_name} begins their journey",
            significance=convert_importance_to_significance("high"),
            tags=['player', 'initialization', 'stats'],
            metadata={"player_name": player_name, "initial_stats": stats}
        )
        
        await log_canonical_event(
            ctx, conn,
            f"Player character '{player_name}' initialized with stats",
            tags=['player', 'initialization', 'stats'],
            significance=5
        )

async def log_stat_change(
    ctx, conn,
    player_name: str,
    stat_name: str,
    old_value: int,
    new_value: int,
    cause: str
) -> None:
    """
    Log a player stat change to the history table.
    """
    # Insert into stats history
    await conn.execute("""
        INSERT INTO StatsHistory (
            user_id, conversation_id, player_name, stat_name,
            old_value, new_value, cause, timestamp
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP)
    """, 
        ctx.user_id, ctx.conversation_id, player_name, stat_name,
        old_value, new_value, cause
    )
    
    # Log significant changes as canonical events
    change = new_value - old_value
    if abs(change) >= 10 or new_value in [0, 100]:
        event_text = f"Player {player_name}'s {stat_name} "
        if change > 0:
            event_text += f"increased by {change} to {new_value}"
        else:
            event_text += f"decreased by {abs(change)} to {new_value}"
        event_text += f" ({cause})"
        
        significance = 5
        if new_value == 0 or new_value == 100:
            significance = 7  # Maxed out or depleted stats are more significant
        elif abs(change) >= 20:
            significance = 6
            
        await log_canonical_event(
            ctx, conn,
            event_text,
            tags=['player_stats', stat_name, 'change'],
            significance=significance
        )

async def update_player_stat_canonically(
    ctx, conn,
    player_name: str,
    stat_name: str,
    new_value: int,
    reason: str
) -> None:
    """Update a player's stat with memory integration."""
    ctx = ensure_canonical_context(ctx)
    memory_orchestrator = await get_canon_memory_orchestrator(ctx.user_id, ctx.conversation_id)
    
    current_value = await conn.fetchval(
        f"SELECT {stat_name} FROM PlayerStats "
        "WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3",
        ctx.user_id, ctx.conversation_id, player_name
    )

    if current_value is None:
        await find_or_create_player_stats(ctx, conn, player_name, **{stat_name: new_value})
        return

    await conn.execute(
        f"UPDATE PlayerStats SET {stat_name} = $1 "
        "WHERE user_id = $2 AND conversation_id = $3 AND player_name = $4",
        new_value, ctx.user_id, ctx.conversation_id, player_name
    )

    # Log the change
    await log_stat_change(
        ctx, conn,
        player_name,
        stat_name,
        current_value,
        new_value,
        reason
    )
    
    # Store as a memory with emotional analysis
    change = new_value - current_value
    emotion_map = {
        'corruption': 'concern' if change > 0 else 'relief',
        'confidence': 'pride' if change > 0 else 'doubt',
        'willpower': 'determination' if change > 0 else 'weakness',
        'lust': 'desire' if change > 0 else 'restraint'
    }
    
    memory_text = f"{player_name}'s {stat_name} {'increased' if change > 0 else 'decreased'} from {current_value} to {new_value} ({reason})"
    
    # Store memory with emotional context
    await memory_orchestrator.store_memory(
        entity_type=EntityType.PLAYER,
        entity_id=ctx.user_id,
        memory_text=memory_text,
        significance=0.8 if abs(change) >= 20 else 0.5,
        emotional=True,
        tags=['stat_change', stat_name, emotion_map.get(stat_name, 'neutral')],
        metadata={
            "player_name": player_name,
            "stat_name": stat_name,
            "old_value": current_value,
            "new_value": new_value,
            "change": change,
            "reason": reason
        }
    )
    
    # Update emotional state if significant change
    if abs(change) >= 10:
        emotion = emotion_map.get(stat_name, 'neutral')
        intensity = min(abs(change) / 50.0, 1.0)
        
        await memory_orchestrator.update_emotional_state(
            entity_type="player",
            entity_id=ctx.user_id,
            emotion=emotion,
            intensity=intensity
        )


async def find_or_create_currency(
    ctx, conn,
    currency_name: str,
    **currency_data
) -> int:
    """
    Find or create a currency system.
    """
    ctx = ensure_canonical_context(ctx)
    # Check if currency exists
    existing = await conn.fetchrow("""
        SELECT id FROM CurrencySystem
        WHERE user_id = $1 AND conversation_id = $2
    """, ctx.user_id, ctx.conversation_id)
    
    if existing:
        return existing['id']
    
    # Create new currency
    currency_id = await conn.fetchval("""
        INSERT INTO CurrencySystem (
            user_id, conversation_id, currency_name, currency_plural,
            minor_currency_name, minor_currency_plural, exchange_rate,
            currency_symbol, format_template, description, setting_context,
            created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, CURRENT_TIMESTAMP)
        RETURNING id
    """,
        ctx.user_id, ctx.conversation_id, currency_name,
        currency_data.get('currency_plural', currency_name + 's'),
        currency_data.get('minor_currency_name'),
        currency_data.get('minor_currency_plural'),
        currency_data.get('exchange_rate', 100),
        currency_data.get('currency_symbol', '$'),
        currency_data.get('format_template', '{{amount}} {{currency}}'),
        currency_data.get('description', ''),
        currency_data.get('setting_context', '')
    )
    
    await log_canonical_event(
        ctx, conn,
        f"Currency system established: {currency_name}",
        tags=['currency', 'economic_system', 'creation'],
        significance=6
    )
    
    return currency_id

async def log_event_creation(
    ctx, conn,
    event_name: str,
    event_type: str = "general",
    **event_data
) -> int:
    """
    Create and log an event in the Events table.
    """
    event_id = await conn.fetchval("""
        INSERT INTO Events (
            user_id, conversation_id, event_name, description,
            start_time, end_time, location, year, month, day, time_of_day
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        RETURNING id
    """,
        ctx.user_id, ctx.conversation_id, event_name,
        event_data.get('description', ''),
        event_data.get('start_time', 'TBD'),
        event_data.get('end_time', 'TBD'),
        event_data.get('location', 'TBD'),
        event_data.get('year', 1),
        event_data.get('month', 1),
        event_data.get('day', 1),
        event_data.get('time_of_day', 'Morning')
    )
    
    await log_canonical_event(
        ctx, conn,
        f"Event scheduled: {event_name} at {event_data.get('location', 'TBD')}",
        tags=['event', event_type, 'scheduled'],
        significance=5
    )
    
    return event_id

# Constants needed by the addiction functions
ADDICTION_LEVELS = {
    0: "None",
    1: "Mild",
    2: "Moderate",
    3: "Heavy",
    4: "Extreme"
}

# ---------------------------------------------------------------------------
# Additional canonical helpers for core world state updates
# ---------------------------------------------------------------------------

async def update_npc_current_location(ctx, conn, npc_id: int, new_location: str) -> None:
    """Update an NPC's current location canonically."""
    await conn.execute(
        """
        UPDATE NPCStats
        SET current_location = $1
        WHERE user_id = $2 AND conversation_id = $3 AND npc_id = $4
        """,
        new_location, ctx.user_id, ctx.conversation_id, npc_id,
    )

    await log_canonical_event(
        ctx,
        conn,
        f"NPC {npc_id} moved to {new_location}",
        tags=["npc", "location", "update"],
        significance=3,
    )


async def remove_inventory_item(
    ctx,
    conn,
    item_name: str,
    player_name: str = "Chase",
    quantity: int = 1,
) -> bool:
    """Remove quantity of an item from a player's inventory canonically."""
    row = await conn.fetchrow(
        """
        SELECT item_id, quantity FROM PlayerInventory
        WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3 AND item_name=$4
        """,
        ctx.user_id,
        ctx.conversation_id,
        player_name,
        item_name,
    )

    if not row:
        return False

    item_id, current_qty = row["item_id"], row["quantity"]
    new_qty = current_qty - quantity

    if new_qty > 0:
        await conn.execute(
            "UPDATE PlayerInventory SET quantity=$1 WHERE item_id=$2",
            new_qty,
            item_id,
        )
    else:
        await conn.execute("DELETE FROM PlayerInventory WHERE item_id=$1", item_id)

    await log_canonical_event(
        ctx,
        conn,
        f"Removed {quantity} of {item_name} from {player_name}",
        tags=["inventory", "item_removal"],
        significance=4,
    )

    return True


async def update_inventory_item_effect(
    ctx,
    conn,
    item_name: str,
    player_name: str,
    new_effect: str,
) -> bool:
    """Update an inventory item's effect canonically."""
    status = await conn.execute(
        """
        UPDATE PlayerInventory
        SET item_effect=$1
        WHERE user_id=$2 AND conversation_id=$3 AND player_name=$4 AND item_name=$5
        """,
        new_effect,
        ctx.user_id,
        ctx.conversation_id,
        player_name,
        item_name,
    )

    if not status.startswith("UPDATE") or status.endswith("0"):
        return False

    await log_canonical_event(
        ctx,
        conn,
        f"Updated effect of {item_name} for {player_name}",
        tags=["inventory", "item_update"],
        significance=3,
    )

    return True


async def categorize_inventory_items(
    ctx,
    conn,
    player_name: str,
    category_mapping: Dict[str, str],
) -> Dict[str, Any]:
    """Categorize multiple inventory items canonically."""
    results = {"items_updated": 0, "items_not_found": []}
    for item, category in category_mapping.items():
        status = await conn.execute(
            """
            UPDATE PlayerInventory
            SET category=$1
            WHERE user_id=$2 AND conversation_id=$3 AND player_name=$4 AND item_name=$5
            """,
            category,
            ctx.user_id,
            ctx.conversation_id,
            player_name,
            item,
        )
        if status.startswith("UPDATE") and not status.endswith("0"):
            results["items_updated"] += 1
        else:
            results["items_not_found"].append(item)

    await log_canonical_event(
        ctx,
        conn,
        f"Categorized items for {player_name}",
        tags=["inventory", "categorize"],
        significance=3,
    )

    return results


async def create_player_manipulation_attempt(
    ctx,
    conn,
    conflict_id: int,
    npc_id: int,
    manipulation_type: str,
    content: str,
    goal: Dict[str, Any],
    leverage_used: Dict[str, Any],
    intimacy_level: int = 0,
) -> int:
    """Create a player manipulation attempt canonically."""
    attempt_id = await conn.fetchval(
        """
        INSERT INTO PlayerManipulationAttempts (
            conflict_id, user_id, conversation_id, npc_id,
            manipulation_type, content, goal, success,
            leverage_used, intimacy_level, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, CURRENT_TIMESTAMP)
        RETURNING attempt_id
        """,
        conflict_id,
        ctx.user_id,
        ctx.conversation_id,
        npc_id,
        manipulation_type,
        content,
        json.dumps(goal),
        False,
        json.dumps(leverage_used),
        intimacy_level,
    )

    await log_canonical_event(
        ctx,
        conn,
        f"NPC {npc_id} attempted {manipulation_type} in conflict {conflict_id}",
        tags=["conflict", "manipulation", manipulation_type],
        significance=7,
    )

    return attempt_id


async def resolve_player_manipulation_attempt(
    ctx,
    conn,
    attempt_id: int,
    success: bool,
    player_response: str,
) -> None:
    """Resolve a player manipulation attempt canonically."""
    await conn.execute(
        """
        UPDATE PlayerManipulationAttempts
        SET success=$1, player_response=$2, resolved_at=CURRENT_TIMESTAMP
        WHERE attempt_id=$3
        """,
        success,
        player_response,
        attempt_id,
    )

    await log_canonical_event(
        ctx,
        conn,
        f"Manipulation attempt {attempt_id} resolved as {'success' if success else 'failure'}",
        tags=["conflict", "manipulation", "resolution"],
        significance=6,
    )

# ---------------------------------------------------------------------------
# Resource and Vital Management Helpers
# ---------------------------------------------------------------------------

async def create_default_resources(ctx, conn, player_name: str = "Chase") -> None:
    """Ensure a PlayerResources row exists for the given player."""
    await conn.execute(
        """
        INSERT INTO PlayerResources (user_id, conversation_id, player_name, money, supplies, influence)
        VALUES ($1, $2, $3, 100, 20, 10)
        ON CONFLICT (user_id, conversation_id, player_name) DO NOTHING
        """,
        ctx.user_id,
        ctx.conversation_id,
        player_name,
    )

    await log_canonical_event(
        ctx,
        conn,
        f"Default resources initialized for {player_name}",
        tags=["resources", "init"],
        significance=4,
    )


async def create_default_vitals(ctx, conn, player_name: str = "Chase") -> None:
    """Ensure a PlayerVitals row exists for the given player."""
    await conn.execute(
        """
        INSERT INTO PlayerVitals (user_id, conversation_id, player_name, energy, hunger)
        VALUES ($1, $2, $3, 100, 100)
        ON CONFLICT (user_id, conversation_id, player_name) DO NOTHING
        """,
        ctx.user_id,
        ctx.conversation_id,
        player_name,
    )

    await log_canonical_event(
        ctx,
        conn,
        f"Default vitals initialized for {player_name}",
        tags=["vitals", "init"],
        significance=4,
    )


async def adjust_player_resource(
    ctx,
    conn,
    player_name: str,
    resource_type: str,
    amount: int,
    source: str,
    description: str = None,
) -> dict:
    """Adjust a player's resource canonically and log the change."""
    if resource_type not in {"money", "supplies", "influence"}:
        raise ValueError("Invalid resource type")

    row = await conn.fetchrow(
        f"SELECT {resource_type} FROM PlayerResources "
        "WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3 FOR UPDATE",
        ctx.user_id,
        ctx.conversation_id,
        player_name,
    )

    if not row:
        await create_default_resources(ctx, conn, player_name)
        defaults = {"money": 100, "supplies": 20, "influence": 10}
        old_value = defaults[resource_type]
    else:
        old_value = row[resource_type]

    new_value = max(0, old_value + amount)

    await conn.execute(
        f"UPDATE PlayerResources SET {resource_type}=$1, updated_at=CURRENT_TIMESTAMP "
        "WHERE user_id=$2 AND conversation_id=$3 AND player_name=$4",
        new_value,
        ctx.user_id,
        ctx.conversation_id,
        player_name,
    )

    await conn.execute(
        """
        INSERT INTO ResourceHistoryLog
            (user_id, conversation_id, player_name, resource_type,
             old_value, new_value, amount_changed, source, description)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
        ctx.user_id,
        ctx.conversation_id,
        player_name,
        resource_type,
        old_value,
        new_value,
        amount,
        source,
        description,
    )

    await log_canonical_event(
        ctx,
        conn,
        f"{player_name}'s {resource_type} changed by {amount}",
        tags=["resources", resource_type],
        significance=3,
    )

    return {"old_value": old_value, "new_value": new_value, "change": amount}


async def adjust_player_vital(
    ctx,
    conn,
    player_name: str,
    vital_type: str,
    amount: int,
    source: str = None,
    description: str = None,
) -> dict:
    """Adjust a player's vitals canonically and log the change."""
    if vital_type not in {"hunger", "energy"}:
        raise ValueError("Invalid vital type")

    row = await conn.fetchrow(
        f"SELECT {vital_type} FROM PlayerVitals "
        "WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3 FOR UPDATE",
        ctx.user_id,
        ctx.conversation_id,
        player_name,
    )

    if not row:
        await create_default_vitals(ctx, conn, player_name)
        old_value = 100
    else:
        old_value = row[vital_type]

    new_value = max(0, min(100, old_value + amount))

    await conn.execute(
        f"UPDATE PlayerVitals SET {vital_type}=$1, last_update=CURRENT_TIMESTAMP "
        "WHERE user_id=$2 AND conversation_id=$3 AND player_name=$4",
        new_value,
        ctx.user_id,
        ctx.conversation_id,
        player_name,
    )

    await conn.execute(
        """
        INSERT INTO ResourceHistoryLog
            (user_id, conversation_id, player_name, resource_type,
             old_value, new_value, amount_changed, source, description)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
        ctx.user_id,
        ctx.conversation_id,
        player_name,
        vital_type,
        old_value,
        new_value,
        amount,
        source or "activity",
        description,
    )

    await log_canonical_event(
        ctx,
        conn,
        f"{player_name}'s {vital_type} changed by {amount}",
        tags=["vitals", vital_type],
        significance=3,
    )

    return {"old_value": old_value, "new_value": new_value, "change": amount}

async def find_or_create_inventory_item(
    ctx, conn,
    player_name: str,
    item_name: str,
    **kwargs
) -> int:
    """Find or create inventory item with existence gate validation."""
    # Check if item already exists
    existing = await conn.fetchrow("""
        SELECT item_id, quantity FROM PlayerInventory
        WHERE user_id = $1 AND conversation_id = $2 
        AND player_name = $3 AND item_name = $4
    """, ctx.user_id, ctx.conversation_id, player_name, item_name)
    
    if existing:
        return existing['item_id']
    
    # Run existence gate check
    gate = ExistenceGate(ctx)
    item_data = {
        'item_name': item_name,
        'item_type': kwargs.get('item_category', 'misc'),
        'tech_band': kwargs.get('tech_band'),
        'required_materials': kwargs.get('required_materials', []),
        'is_crafting': kwargs.get('is_crafting', False),
        'resource_cost': kwargs.get('resource_cost', 0),
        **kwargs
    }
    
    decision, details = await gate.assess_entity('item', item_data)
    
    if decision == GateDecision.DENY:
        raise ValueError(f"Item '{item_name}' cannot exist: {details['reason']}")
    
    elif decision == GateDecision.DEFER:
        logger.warning(f"Item '{item_name}' deferred: {details['reason']}")
        # Could create a "quest item" marker instead
        return -1
    
    elif decision == GateDecision.ANALOG:
        # Use analog item
        analog_name = details['analog']
        logger.info(f"Substituting '{item_name}' with analog '{analog_name}'")
        item_name = analog_name
        # Update properties for analog
        kwargs['item_description'] = f"A {analog_name} (substituted for {item_data['item_name']})"
    
    # Create new item (passed gate)
    item_id = await conn.fetchval("""
        INSERT INTO PlayerInventory (
            user_id, conversation_id, player_name, item_name,
            item_description, item_category, item_properties,
            quantity, equipped, date_acquired
        ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, CURRENT_TIMESTAMP)
        RETURNING item_id
    """,
        ctx.user_id, ctx.conversation_id, player_name, item_name,
        kwargs.get('item_description', ''),
        kwargs.get('item_category', 'misc'),
        json.dumps(kwargs.get('item_properties', {})),
        kwargs.get('quantity', 1),
        kwargs.get('equipped', False)
    )
    
    await log_canonical_event(
        ctx, conn,
        f"Player {player_name} acquired new item: {item_name}",
        tags=['inventory', 'item_acquisition', kwargs.get('item_category', 'misc')],
        significance=4
    )
    
    return item_id

async def update_inventory_quantity(
    ctx, conn,
    item_id: int,
    new_quantity: int,
    reason: str
) -> bool:
    """
    Update the quantity of an inventory item.
    """
    if new_quantity <= 0:
        # Delete the item
        await conn.execute("""
            DELETE FROM PlayerInventory WHERE item_id = $1
        """, item_id)
        
        await log_canonical_event(
            ctx, conn,
            f"Item removed from inventory: {reason}",
            tags=['inventory', 'item_removal'],
            significance=3
        )
        return True
    else:
        # Update quantity
        result = await conn.execute("""
            UPDATE PlayerInventory 
            SET quantity = $1, date_acquired = CURRENT_TIMESTAMP
            WHERE item_id = $2
        """, new_quantity, item_id)
        
        await log_canonical_event(
            ctx, conn,
            f"Inventory quantity updated: {reason}",
            tags=['inventory', 'quantity_change'],
            significance=2
        )
        return result != "UPDATE 0"

async def find_or_create_currency_system(
    ctx, conn,
    **currency_data
) -> int:
    """
    Find or create a currency system.
    """
    ctx = ensure_canonical_context(ctx)
    # Check if currency exists
    existing = await conn.fetchrow("""
        SELECT id FROM CurrencySystem
        WHERE user_id = $1 AND conversation_id = $2
    """, ctx.user_id, ctx.conversation_id)
    
    if existing:
        return existing['id']
    
    # Create new currency
    currency_id = await conn.fetchval("""
        INSERT INTO CurrencySystem (
            user_id, conversation_id, currency_name, currency_plural,
            minor_currency_name, minor_currency_plural, exchange_rate,
            currency_symbol, format_template, description, setting_context,
            created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, CURRENT_TIMESTAMP)
        RETURNING id
    """,
        ctx.user_id, ctx.conversation_id, 
        currency_data['currency_name'],
        currency_data.get('currency_plural', currency_data['currency_name'] + 's'),
        currency_data.get('minor_currency_name'),
        currency_data.get('minor_currency_plural'),
        currency_data.get('exchange_rate', 100),
        currency_data.get('currency_symbol', '$'),
        currency_data.get('format_template', '{{amount}} {{currency}}'),
        currency_data.get('description', ''),
        currency_data.get('setting_context', '')
    )
    
    await log_canonical_event(
        ctx, conn,
        f"Currency system established: {currency_data['currency_name']}",
        tags=['currency', 'economic_system', 'creation'],
        significance=6
    )
    
    return currency_id

# Add these functions to lore/core/canon.py

async def find_or_create_event(ctx, conn, event_name: str, **kwargs) -> int:
    """
    Find or create an event canonically with semantic matching.
    """
    # Ensure we have a proper context
    ctx = ensure_canonical_context(ctx)
    
    # Check exact match
    existing = await conn.fetchrow("""
        SELECT id FROM Events
        WHERE LOWER(event_name) = LOWER($1)
        AND user_id = $2 AND conversation_id = $3
        AND year = $4 AND month = $5 AND day = $6
    """, event_name, ctx.user_id, ctx.conversation_id,
        kwargs.get('year', 1), kwargs.get('month', 1), kwargs.get('day', 1))
    
    if existing:
        return existing['id']
    
    # Semantic similarity check
    description = kwargs.get('description', '')
    location = kwargs.get('location', 'Unknown')
    embedding_text = f"{event_name} {description} at {location}"
    search_vector = await generate_embedding(embedding_text)
    
    similar = await conn.fetchrow("""
        SELECT id, event_name, description, location,
               1 - (embedding <=> $1) AS similarity
        FROM Events
        WHERE user_id = $2 AND conversation_id = $3
        AND embedding IS NOT NULL
        AND 1 - (embedding <=> $1) > 0.85
        ORDER BY embedding <=> $1
        LIMIT 1
    """, search_vector, ctx.user_id, ctx.conversation_id)
    
    if similar:
        validation_agent = CanonValidationAgent()
        prompt = f"""
        Are these the same event?
        
        Proposed: {event_name} at {location}
        Description: {description[:200]}
        Date: Year {kwargs.get('year', 1)}, Month {kwargs.get('month', 1)}, Day {kwargs.get('day', 1)}
        
        Existing: {similar['event_name']} at {similar['location']}
        Description: {similar['description'][:200]}
        Similarity: {similar['similarity']:.2f}
        
        Answer only 'true' or 'false'.
        """
        
        result = await Runner.run(validation_agent.agent, prompt)
        if result.final_output.strip().lower() == 'true':
            return similar['id']
    
    # Create new event
    event_id = await conn.fetchval("""
        INSERT INTO Events (
            user_id, conversation_id, event_name, description,
            start_time, end_time, location, year, month, day,
            time_of_day, embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        RETURNING id
    """,
        ctx.user_id, ctx.conversation_id, event_name, description,
        kwargs.get('start_time', 'TBD'),
        kwargs.get('end_time', 'TBD'),
        location,
        kwargs.get('year', 1),
        kwargs.get('month', 1),
        kwargs.get('day', 1),
        kwargs.get('time_of_day', 'Morning'),
        search_vector
    )
    
    await log_canonical_event(
        ctx, conn,
        f"Event '{event_name}' scheduled at {location}",
        tags=['event', 'creation', 'canon'],
        significance=5
    )
    
    return event_id

async def find_or_create_quest(ctx, conn, quest_name: str, **kwargs) -> int:
    """
    Find or create a quest canonically.
    """
    ctx = ensure_canonical_context(ctx)
    # Check exact match
    existing = await conn.fetchrow("""
        SELECT quest_id FROM Quests
        WHERE LOWER(quest_name) = LOWER($1)
        AND user_id = $2 AND conversation_id = $3
        AND status = 'In Progress'
    """, quest_name, ctx.user_id, ctx.conversation_id)
    
    if existing:
        return existing['quest_id']
    
    # Create new quest
    quest_id = await conn.fetchval("""
        INSERT INTO Quests (
            user_id, conversation_id, quest_name, status,
            progress_detail, quest_giver, reward
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING quest_id
    """,
        ctx.user_id, ctx.conversation_id, quest_name,
        kwargs.get('status', 'In Progress'),
        kwargs.get('progress_detail', ''),
        kwargs.get('quest_giver', ''),
        kwargs.get('reward', '')
    )
    
    await log_canonical_event(
        ctx, conn,
        f"Quest '{quest_name}' initiated",
        tags=['quest', 'creation', 'canon'],
        significance=6
    )
    
    return quest_id

async def create_game_setting(ctx, conn, setting_name: str, **kwargs) -> None:
    """
    Create or update the game setting canonically.
    """
    # Ensure we have a proper context - this is what's missing!
    ctx = ensure_canonical_context(ctx)
    
    # Store environment description
    env_desc = kwargs.get('environment_desc', '')
    env_history = kwargs.get('environment_history', '')
    full_desc = f"{env_desc}\n\nHistory: {env_history}" if env_history else env_desc
    
    await update_current_roleplay(ctx, conn,
                                 'EnvironmentDesc', full_desc)
    
    # Store setting name
    await update_current_roleplay(ctx, conn,
                                 'CurrentSetting', setting_name)
    
    # Store calendar data if provided
    if 'calendar_data' in kwargs:
        await update_current_roleplay(ctx, conn,
                                     'CalendarNames', json.dumps(kwargs['calendar_data']))
    
    # Update conversation name if scenario name provided
    if 'scenario_name' in kwargs:
        await conn.execute("""
            UPDATE conversations
            SET conversation_name = $1
            WHERE id = $2 AND user_id = $3
        """, kwargs['scenario_name'], ctx.conversation_id, ctx.user_id)
    
    await log_canonical_event(
        ctx, conn,
        f"Game setting '{setting_name}' established",
        tags=['setting', 'world_creation', 'canon'],
        significance=9
    )
    
async def store_player_schedule(ctx, conn, player_name: str, schedule: Dict[str, Any]) -> None:
    """
    Store a player's schedule canonically.
    """
    schedule_json = json.dumps(schedule)
    
    await update_current_roleplay(ctx, conn, 
                                 f'{player_name}Schedule', schedule_json)
    
    # Also create a journal entry for the schedule
    schedule_summary = f"{player_name}'s typical schedule: "
    for day, activities in schedule.items():
        day_summary = f"\n{day}: "
        if isinstance(activities, dict):
            for period, activity in activities.items():
                day_summary += f"{period}: {activity}; "
        else:
            day_summary += str(activities)
        schedule_summary += day_summary
    
    await create_journal_entry(
        ctx, conn,
        entry_type="schedule",
        entry_text=schedule_summary,
        tags=["schedule", player_name.lower()],
        importance=0.6
    )
    
    await log_canonical_event(
        ctx, conn,
        f"Schedule created for {player_name}",
        tags=['schedule', 'player', 'canon'],
        significance=4
    )

async def sync_embeddings_to_memory_system(ctx, conn):
    """
    One-time sync of existing database embeddings to memory orchestrator's vector store.
    This ensures all existing canonical data is searchable through the memory system.
    """
    ctx = ensure_canonical_context(ctx)
    memory_orchestrator = await get_canon_memory_orchestrator(ctx.user_id, ctx.conversation_id)
    
    # Sync NPCs
    npcs = await conn.fetch("""
        SELECT npc_id, npc_name, role, affiliations
        FROM NPCStats
        WHERE user_id = $1 AND conversation_id = $2
    """, ctx.user_id, ctx.conversation_id)
    
    for npc in npcs:
        text = f"NPC: {npc['npc_name']}, role: {npc.get('role', 'unspecified')}"
        await memory_orchestrator.add_to_vector_store(
            text=text,
            metadata={
                "entity_type": "npc",
                "entity_id": npc['npc_id'],
                "npc_name": npc['npc_name'],
                "user_id": ctx.user_id,
                "conversation_id": ctx.conversation_id
            },
            entity_type="npc"
        )
    
    # Sync Locations
    locations = await conn.fetch("""
        SELECT id, location_name, description, location_type
        FROM Locations
        WHERE user_id = $1 AND conversation_id = $2
    """, ctx.user_id, ctx.conversation_id)
    
    for loc in locations:
        text = f"Location: {loc['location_name']} - {loc.get('description', '')}"
        await memory_orchestrator.add_to_vector_store(
            text=text,
            metadata={
                "entity_type": "location",
                "location_id": loc['id'],
                "location_name": loc['location_name'],
                "user_id": ctx.user_id,
                "conversation_id": ctx.conversation_id
            },
            entity_type="location"
        )
    
    logger.info(f"Synced {len(npcs)} NPCs and {len(locations)} locations to memory system")

async def create_opening_message(ctx, conn, sender: str, content: str) -> int:
    """
    Create the opening message canonically.
    """
    # Ensure we have a proper context
    ctx = ensure_canonical_context(ctx)
    
    # Now we can access conversation_id directly
    message_id = await create_message(ctx, conn, ctx.conversation_id, sender, content)
    
    await log_canonical_event(
        ctx, conn,
        f"Opening narrative created by {sender}",
        tags=['narrative', 'opening', 'canon'],
        significance=8
    )
    
    return message_id

# Add to the existing functions - update the embedding columns
async def ensure_embedding_columns(conn):
    """
    Ensure all relevant tables have embedding columns.
    """
    tables_to_update = [
        ('Events', 'event_name, description, location'),
        ('Quests', 'quest_name, progress_detail'),
        ('Locations', 'location_name, description')
    ]
    
    for table, text_fields in tables_to_update:
        # Check if embedding column exists
        exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = $1 AND column_name = 'embedding'
            )
        """, table.lower())
        
        if not exists:
            await conn.execute(f"""
                ALTER TABLE {table} 
                ADD COLUMN embedding VECTOR(1536)
            """)
            
            # Create index
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table.lower()}_embedding_hnsw
                ON {table}
                USING hnsw (embedding vector_cosine_ops)
            """)


async def initialize_canon_memory_integration(user_id: int, conversation_id: int):
    """Initialize the canon-memory integration for a conversation."""
    orchestrator = await get_canon_memory_orchestrator(user_id, conversation_id)
    
    # Run maintenance to ensure consistency
    await orchestrator.run_maintenance(operations=["consolidation"])
    
    # Sync any existing embeddings
    async with get_db_connection_context() as conn:
        ctx = CanonicalContext(user_id=user_id, conversation_id=conversation_id)
        await sync_embeddings_to_memory_system(ctx, conn)
    
    logger.info(f"Canon-Memory integration initialized for user {user_id}, conversation {conversation_id}")
    return orchestrator

async def process_canonical_update(
    ctx, conn,
    update_type: str,
    entity_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Central processor for all canonical updates.
    Routes through memory orchestrator for consistency.
    """
    ctx = ensure_canonical_context(ctx)
    memory_orchestrator = await get_canon_memory_orchestrator(ctx.user_id, ctx.conversation_id)
    
    # Validate consistency before update
    validation_result = await memory_orchestrator.validate_canonical_consistency(
        entity_type=update_type,
        entity_data=entity_data
    )
    
    if not validation_result["is_consistent"]:
        logger.warning(f"Canonical consistency issues detected: {validation_result['conflicts']}")
        # Could throw exception or handle conflicts here
    
    # Process the update based on type
    result = {}
    
    if update_type == "npc":
        result["id"] = await find_or_create_npc(ctx, conn, **entity_data)
    elif update_type == "location":
        result["name"] = await find_or_create_location(ctx, conn, **entity_data)
    elif update_type == "nation":
        result["id"] = await find_or_create_nation(ctx, conn, **entity_data)
    elif update_type == "conflict":
        result["id"] = await find_or_create_conflict(ctx, conn, **entity_data)
    elif update_type == "myth":
        result["id"] = await find_or_create_urban_myth(ctx, conn, **entity_data)
    else:
        result = await find_or_create_entity(
            ctx, conn,
            entity_type=update_type,
            entity_name=entity_data.get("name", "Unknown"),
            search_fields={"name": entity_data.get("name")},
            create_data=entity_data,
            table_name=update_type.title() + "s",  # Simple pluralization
            embedding_text=f"{update_type}: {entity_data}",
            similarity_threshold=0.85
        )
    
    # Get narrative context after update
    result["narrative_context"] = await memory_orchestrator.get_canonical_context(
        entity_types=[update_type]
    )
    
    return result
