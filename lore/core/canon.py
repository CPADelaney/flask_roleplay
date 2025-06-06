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


# lore/core/canon.py - More sophisticated entity handling

async def find_or_create_nation(
    ctx,
    conn,
    nation_name: str,
    government_type: str = None,
    matriarchy_level: int = None,
    **kwargs
) -> int:
    """
    Find or create a nation with sophisticated matching logic.
    """
    # Step 1: Exact name match
    existing = await conn.fetchrow("""
        SELECT id, name, government_type, matriarchy_level, description
        FROM Nations 
        WHERE LOWER(name) = LOWER($1)
    """, nation_name)
    
    if existing:
        logger.info(f"Nation '{nation_name}' found via exact match with ID {existing['id']}")
        return existing['id']
    
    # Step 2: Fuzzy name matching
    similar_names = await conn.fetch("""
        SELECT id, name, government_type, matriarchy_level, description,
               similarity(name, $1) as sim
        FROM Nations
        WHERE similarity(name, $1) > 0.6
        ORDER BY sim DESC
        LIMIT 5
    """, nation_name)
    
    if similar_names:
        # Check with validation agent
        validation_agent = CanonValidationAgent()
        for similar in similar_names:
            prompt = f"""
            Are these the same nation?
            
            Proposed: {nation_name} ({government_type or 'unknown government'})
            Existing: {similar['name']} ({similar['government_type']})
            
            Consider name variations, abbreviations, and translations.
            Answer only 'true' or 'false'.
            """
            
            result = await Runner.run(validation_agent.agent, prompt)
            if result.final_output.strip().lower() == 'true':
                logger.info(f"Nation '{nation_name}' matched to existing '{similar['name']}' (ID: {similar['id']})")
                return similar['id']
    
    # Step 3: Semantic similarity check
    description = kwargs.get('description', '')
    embedding_text = f"{nation_name} {government_type or ''} {description}"
    search_vector = await generate_embedding(embedding_text)
    
    semantic_matches = await conn.fetch("""
        SELECT id, name, government_type, description,
               1 - (embedding <=> $1) AS similarity
        FROM Nations
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> $1
        LIMIT 3
    """, search_vector)
    
    for match in semantic_matches:
        if match['similarity'] > 0.85:
            # High similarity - verify with agent
            prompt = f"""
            I found a semantically similar nation. Are these the same?
            
            Proposed Nation:
            - Name: {nation_name}
            - Government: {government_type}
            - Description: {description[:200]}
            
            Existing Nation:
            - Name: {match['name']}
            - Government: {match['government_type']}
            - Description: {match['description'][:200]}
            
            Answer only 'true' or 'false'.
            """
            
            result = await Runner.run(validation_agent.agent, prompt)
            if result.final_output.strip().lower() == 'true':
                logger.info(f"Nation '{nation_name}' semantically matched to '{match['name']}' (ID: {match['id']})")
                return match['id']
    
    # Step 4: Create new nation
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
        search_vector
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
    """
    Find or create a conflict with sophisticated duplicate detection.
    Conflicts between the same nations can be different conflicts.
    """
    # Step 1: Check for very similar active conflicts
    existing_conflicts = await conn.fetch("""
        SELECT c.*, 
               array_agg(n.name ORDER BY n.id) as nation_names
        FROM NationalConflicts c
        LEFT JOIN Nations n ON n.id = ANY(c.involved_nations)
        WHERE c.involved_nations @> $1 
          AND c.involved_nations <@ $1
          AND c.status != 'resolved'
        GROUP BY c.id
    """, involved_nations)
    
    if existing_conflicts:
        # Check each for similarity
        for conflict in existing_conflicts:
            # If same type and similar time period, might be duplicate
            if conflict['conflict_type'] == conflict_type:
                validation_agent = CanonValidationAgent()
                prompt = f"""
                Are these the same conflict?
                
                Existing Conflict:
                - Name: {conflict['name']}
                - Type: {conflict['conflict_type']}
                - Nations: {', '.join(conflict['nation_names'])}
                - Description: {conflict['description'][:200]}
                - Status: {conflict['status']}
                
                Proposed Conflict:
                - Name: {conflict_name}
                - Type: {conflict_type}
                - Description: {kwargs.get('description', '')[:200]}
                
                Consider if this is a continuation, escalation, or separate conflict.
                Answer only 'true' or 'false'.
                """
                
                result = await Runner.run(validation_agent.agent, prompt)
                if result.final_output.strip().lower() == 'true':
                    logger.info(f"Conflict '{conflict_name}' matched to existing conflict ID {conflict['id']}")
                    return conflict['id']
    
    # Step 2: Semantic search for similar conflicts
    embedding_text = f"{conflict_name} {conflict_type} {kwargs.get('description', '')}"
    search_vector = await generate_embedding(embedding_text)
    
    semantic_matches = await conn.fetch("""
        SELECT c.*, 
               1 - (c.embedding <=> $1) AS similarity,
               array_agg(n.name ORDER BY n.id) as nation_names
        FROM NationalConflicts c
        LEFT JOIN Nations n ON n.id = ANY(c.involved_nations)
        WHERE c.embedding IS NOT NULL
        GROUP BY c.id
        HAVING 1 - (c.embedding <=> $1) > 0.8
        ORDER BY c.embedding <=> $1
        LIMIT 5
    """, search_vector)
    
    for match in semantic_matches:
        # Check if nations overlap significantly
        match_nations = set(match['involved_nations'])
        proposed_nations = set(involved_nations)
        overlap = match_nations & proposed_nations
        
        if len(overlap) >= min(len(match_nations), len(proposed_nations)) * 0.5:
            # Significant overlap - verify with agent
            validation_agent = CanonValidationAgent()
            prompt = f"""
            Found a similar conflict. Are these the same?
            
            Existing: {match['name']} ({match['conflict_type']})
            - Nations: {', '.join(match['nation_names'])}
            - Status: {match['status']}
            - Similarity: {match['similarity']:.2f}
            
            Proposed: {conflict_name} ({conflict_type})
            
            Answer only 'true' or 'false'.
            """
            
            result = await Runner.run(validation_agent.agent, prompt)
            if result.final_output.strip().lower() == 'true':
                return match['id']
    
    # Step 3: Create new conflict
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
        search_vector
    )
    
    # Get nation names for the event log
    nation_names = await conn.fetch("""
        SELECT name FROM Nations WHERE id = ANY($1)
    """, involved_nations)
    nation_list = [n['name'] for n in nation_names]
    
    await log_canonical_event(
        ctx, conn,
        f"New conflict erupted: {conflict_name} between {', '.join(nation_list)}",
        tags=['conflict', 'political', conflict_type],
        significance=9
    )
    
    return conflict_id

# lore/core/canon.py - Additional functions needed

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
        'adoption_date': kwargs.get('adoption_date', datetime.now())
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
