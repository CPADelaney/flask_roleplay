# lore/core/canon.py (Upgraded with Semantic Search)

import logging
import json
from typing import Dict, Any, Optional

from db.connection import get_db_connection_context
from nyx.nyx_governance import AgentType
from nyx.governance_helpers import with_governance
from embedding.vector_store import generate_embedding # Assuming you have this

# Import the new validation agent
from lore.core.validation import CanonValidationAgent

logger = logging.getLogger(__name__)

# --- Helper for Semantic Search ---

async def _find_semantically_similar_npc(conn, name: str, role: Optional[str], threshold: float = 0.90) -> Optional[int]:
    """
    Uses vector embeddings to find a semantically similar NPC.
    Returns the ID of the most similar NPC if above the threshold, otherwise None.
    """
    # Create a detailed embedding string for the new NPC proposal
    search_text = f"{name}"
    if role:
        search_text += f", {role}"
    
    search_vector = await generate_embedding(search_text)

    # Perform a cosine similarity search on the database
    # This requires a vector index on the `embedding` column for performance
    query = """
        SELECT npc_id, npc_name, 1 - (embedding <=> $1) AS similarity
        FROM NPCStats
        ORDER BY embedding <=> $1
        LIMIT 1
    """
    most_similar = await conn.fetchrow(query, search_vector)

    if most_similar and most_similar['similarity'] > threshold:
        logger.info(
            f"Found a semantically similar NPC: '{most_similar['npc_name']}' (ID: {most_similar['npc_id']}) "
            f"with similarity {most_similar['similarity']:.2f} to the proposal for '{name}'."
        )
        return most_similar['npc_id']

    return None

# --- Upgraded NPC Canon Function ---
@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="find_or_create_npc",
    action_description="Finding or creating NPC: {npc_name}",
    id_from_context=lambda ctx: "canon"
)
async def find_or_create_npc(ctx, conn, npc_name: str, **kwargs) -> int:
    """
    Finds an NPC by exact name or semantic similarity, or creates them if they don't exist.
    Returns the NPC's ID.
    `conn` must be an active asyncpg connection or transaction.
    """
    # --- Step 1: Exact Match Check (Fastest) ---
    existing_npc = await conn.fetchrow(
        "SELECT npc_id FROM NPCStats WHERE npc_name = $1 AND user_id = $2 AND conversation_id = $3",
        npc_name, ctx.user_id, ctx.conversation_id
    )
    if existing_npc:
        logger.warning(f"NPC '{npc_name}' found via exact match with ID {existing_npc['npc_id']}.")
        return existing_npc['npc_id']

    # --- Step 2: Semantic Similarity Check (The New Logic) ---
    role = kwargs.get("role")
    similar_npc_id = await _find_semantically_similar_npc(conn, npc_name, role)

    if similar_npc_id:
        # We found a very similar NPC. Now, we must use an LLM to make the final judgment.
        validation_agent = CanonValidationAgent()
        is_duplicate = await validation_agent.confirm_is_duplicate_npc(
            conn,
            proposal={"name": npc_name, "role": role},
            existing_npc_id=similar_npc_id
        )

        if is_duplicate:
            logger.warning(f"LLM confirmed that proposal '{npc_name}' is a duplicate of existing NPC ID {similar_npc_id}. Merging/returning existing.")
            # Optional: Add logic here to merge any new kwargs into the existing NPC record.
            return similar_npc_id
        else:
            logger.info(f"LLM determined that proposal '{npc_name}' is NOT a duplicate. Proceeding with creation.")


    # --- Step 3: Commit - Create the new NPC if no duplicate was found ---
    # This part remains the same. It only runs if no duplicate is found.
    insert_query = """
        INSERT INTO NPCStats (user_id, conversation_id, npc_name, role, affiliations, embedding)
        VALUES ($1, $2, $3, $4, $5, $6) RETURNING npc_id
    """
    # Create the embedding for the new entry
    embedding_text = f"{npc_name}"
    if role:
        embedding_text += f", {role}"
    new_embedding = await generate_embedding(embedding_text)

    npc_id = await conn.fetchval(
        insert_query,
        ctx.user_id,
        ctx.conversation_id,
        npc_name,
        role,
        kwargs.get("affiliations", []),
        new_embedding
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
    
    Args:
        ctx: Context with user_id and conversation_id
        conn: Database connection
        entity_type: Type of entity (e.g., "nation", "deity", "language")
        entity_name: Name of the entity
        search_fields: Fields to search for exact match
        create_data: Data to use when creating the entity
        table_name: Database table name
        embedding_text: Text to use for embedding generation
        similarity_threshold: Threshold for semantic similarity
        
    Returns:
        Entity ID (existing or newly created)
    """
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
    
    # Step 2: Semantic similarity check
    search_vector = await generate_embedding(embedding_text)
    
    similarity_query = f"""
        SELECT id, {search_fields.get('name_field', 'name')} as entity_name, 
               1 - (embedding <=> $1) AS similarity
        FROM {table_name}
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> $1
        LIMIT 1
    """
    
    most_similar = await conn.fetchrow(similarity_query, search_vector)
    
    if most_similar and most_similar['similarity'] > similarity_threshold:
        logger.info(
            f"Found semantically similar {entity_type}: '{most_similar['entity_name']}' "
            f"(ID: {most_similar['id']}) with similarity {most_similar['similarity']:.2f}"
        )
        
        # Use validation agent to confirm
        validation_agent = CanonValidationAgent()
        prompt = f"""
        I am considering creating a new {entity_type}, but found a semantically similar existing one.
        Please determine if they are the same entity described differently.

        Proposed New {entity_type}:
        - Name: "{entity_name}"
        - Details: {json.dumps(create_data, indent=2)}

        Most Similar Existing {entity_type}:
        - ID: {most_similar['id']}
        - Name: "{most_similar['entity_name']}"

        Are these the same {entity_type}? Answer with only 'true' or 'false'.
        """
        
        result = await Runner.run(validation_agent.agent, prompt)
        is_duplicate = result.final_output.strip().lower() == 'true'
        
        if is_duplicate:
            logger.info(f"LLM confirmed '{entity_name}' is a duplicate of existing {entity_type} ID {most_similar['id']}")
            return most_similar['id']
    
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
    await conn.execute(
        f"UPDATE {table_name} SET embedding = $1 WHERE id = $2",
        search_vector, entity_id
    )
    
    logger.info(f"Canonically created new {entity_type} '{entity_name}' with ID {entity_id}")
    
    # Log canonical event
    await log_canonical_event(
        ctx, conn, 
        f"Created new {entity_type}: {entity_name}",
        tags=[entity_type.lower(), 'creation'],
        significance=7
    )
    
    return entity_id

@with_governance(
    agent_type=AgentType.NARRATIVE_CRAFTER,
    action_type="log_canonical_event",
    action_description="Logging canonical event: {event_text}",
    id_from_context=lambda ctx: "canon"
)
async def log_canonical_event(ctx, conn, event_text: str, tags: List[str] = None, significance: int = 5):
    """Log a canonical event to establish world history."""
    tags = tags or []
    
    await conn.execute("""
        INSERT INTO CanonicalEvents (
            user_id, conversation_id, event_text, tags, significance, timestamp
        )
        VALUES ($1, $2, $3, $4, $5, $6)
    """, ctx.user_id, ctx.conversation_id, event_text, tags, significance, datetime.now())

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
