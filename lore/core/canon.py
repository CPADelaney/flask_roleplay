# lore/core/canon.py (Upgraded with Semantic Search)

import logging
import json

from db.connection import get_db_connection_context
from nyx.nyx_governance import AgentType
from nyx.governance_helpers import with_governance
from embedding.vector_store import generate_embedding # Assuming you have this
from lore.core.context import CanonicalContext

from typing import List, Dict, Any, Union, Optional
from datetime import datetime
from agents import Runner

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

def ensure_canonical_context(ctx) -> CanonicalContext:
    """Convert any context to a CanonicalContext."""
    if isinstance(ctx, CanonicalContext):
        return ctx
    elif isinstance(ctx, dict):
        return CanonicalContext.from_dict(ctx)
    else:
        return CanonicalContext.from_object(ctx)


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


async def log_canonical_event(ctx, conn, event_text: str, tags: List[str] = None, significance: int = 5):
    """Log a canonical event to establish world history."""
    # Ensure we have a proper context
    ctx = ensure_canonical_context(ctx)
    
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
    """Find or create an urban myth with semantic matching."""
    # Check exact match
    existing = await conn.fetchrow("""
        SELECT id FROM UrbanMyths
        WHERE name = $1
    """, name)
    
    if existing:
        return existing['id']
    
    # Semantic similarity check
    narrative_style = kwargs.get('narrative_style', 'folklore')
    themes = kwargs.get('themes', [])
    embedding_text = f"{name} {description} {narrative_style} {' '.join(themes)}"
    search_vector = await generate_embedding(embedding_text)
    
    similar = await conn.fetchrow("""
        SELECT id, name, description, 1 - (embedding <=> $1) AS similarity
        FROM UrbanMyths
        WHERE 1 - (embedding <=> $1) > 0.80
        ORDER BY embedding <=> $1
        LIMIT 1
    """, search_vector)
    
    if similar:
        validation_agent = CanonValidationAgent()
        is_duplicate = await validation_agent.confirm_is_duplicate_myth(
            conn,
            proposal={"name": name, "description": description, "themes": themes},
            existing_myth_id=similar['id']
        )
        
        if is_duplicate:
            logger.info(f"Urban myth '{name}' matched to existing ID {similar['id']}")
            
            # Update regions if new ones provided
            regions_known = kwargs.get('regions_known', [])
            if regions_known:
                existing_regions = await conn.fetchval(
                    "SELECT regions_known FROM UrbanMyths WHERE id = $1",
                    similar['id']
                )
                new_regions = list(set(existing_regions + regions_known))
                
                await conn.execute("""
                    UPDATE UrbanMyths
                    SET regions_known = $1,
                        spread_rate = GREATEST(spread_rate, $2)
                    WHERE id = $3
                """, new_regions, kwargs.get('spread_rate', 5), similar['id'])
            
            return similar['id']
    
    # Create new myth
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
    search_vector)
    
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
    """
    Find or create a location canonically.
    Returns the location name (locations use name as primary identifier).
    """
    # Check exact match
    existing = await conn.fetchrow("""
        SELECT id, location_name, description
        FROM Locations
        WHERE LOWER(location_name) = LOWER($1)
        AND user_id = $2 AND conversation_id = $3
    """, location_name, ctx.user_id, ctx.conversation_id)
    
    if existing:
        logger.info(f"Location '{location_name}' found via exact match")
        return existing['location_name']
    
    # Semantic similarity check
    description = kwargs.get('description', f"The area known as {location_name}")
    embedding_text = f"{location_name} {description}"
    search_vector = await generate_embedding(embedding_text)
    
    similar = await conn.fetchrow("""
        SELECT id, location_name, description, 1 - (embedding <=> $1) AS similarity
        FROM Locations
        WHERE user_id = $2 AND conversation_id = $3
        AND embedding IS NOT NULL
        AND 1 - (embedding <=> $1) > 0.85
        ORDER BY embedding <=> $1
        LIMIT 1
    """, search_vector, ctx.user_id, ctx.conversation_id)
    
    if similar:
        validation_agent = CanonValidationAgent()
        prompt = f"""
        Are these the same location?
        
        Proposed: {location_name}
        Description: {description}
        
        Existing: {similar['location_name']}
        Description: {similar['description']}
        Similarity: {similar['similarity']:.2f}
        
        Answer only 'true' or 'false'.
        """
        
        result = await Runner.run(validation_agent.agent, prompt)
        if result.final_output.strip().lower() == 'true':
            logger.info(f"Location '{location_name}' matched to existing '{similar['location_name']}'")
            return similar['location_name']
    
    # Create new location
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
        kwargs.get('notable_features', []),
        kwargs.get('hidden_aspects', []),
        kwargs.get('access_restrictions', []),
        kwargs.get('local_customs', []),
        search_vector
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
    
    # Create new figure
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
        kwargs.get('faction_affiliations', []),
        kwargs.get('achievements', []),
        kwargs.get('failures', []),
        kwargs.get('personality_traits', []),
        kwargs.get('public_image', 'neutral'),
        kwargs.get('hidden_aspects', []),
        kwargs.get('influence_areas', []),
        kwargs.get('legacy', ''),
        kwargs.get('controversial_actions', []),
        kwargs.get('relationships', []),
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
    
    result = await lore_system.propose_and_enact_change(
        ctx,
        entity_type=entity_type,
        entity_identifier={"id": entity_id},
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
    Update any entity with governance tracking and validation.
    This is used by Nyx when it needs to make direct canonical changes.
    
    Args:
        ctx: Context with governance info
        conn: Database connection
        entity_type: Type of entity (table name)
        entity_id: ID of the entity
        updates: Dictionary of field updates
        reason: Reason for the update
        significance: Significance level for the event log
        
    Returns:
        Result dictionary with status and details
    """
    try:
        # Build the update query dynamically
        set_clauses = []
        values = []
        for i, (field, value) in enumerate(updates.items()):
            set_clauses.append(f"{field} = ${i+1}")
            values.append(value)
        
        if not set_clauses:
            return {"status": "error", "message": "No updates provided"}
        
        # Add the ID as the last parameter
        values.append(entity_id)
        
        # Construct the query
        query = f"""
            UPDATE {entity_type}
            SET {', '.join(set_clauses)}
            WHERE id = ${len(values)}
            RETURNING *
        """
        
        # Execute the update
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
        
        # Update embedding if the entity has text fields that changed
        text_fields = ['name', 'description', 'title']
        if any(field in updates for field in text_fields):
            # Generate new embedding text
            embedding_parts = []
            for field in text_fields:
                if field in result:
                    embedding_parts.append(str(result[field]))
            
            if embedding_parts:
                embedding_text = ' '.join(embedding_parts)
                embedding = await generate_embedding(embedding_text)
                
                # Update the embedding
                await conn.execute(
                    f"UPDATE {entity_type} SET embedding = $1 WHERE id = $2",
                    embedding, entity_id
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
    """, conversation_id, sender, content, datetime.now(timezone.utc))
    
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
    """
    Find or create a social link between two entities.
    """
    user_id = kwargs.get('user_id', ctx.user_id)
    conversation_id = kwargs.get('conversation_id', ctx.conversation_id)
    entity1_type = kwargs['entity1_type']
    entity1_id = kwargs['entity1_id']
    entity2_type = kwargs['entity2_type']
    entity2_id = kwargs['entity2_id']
    
    # Check if link already exists
    existing = await conn.fetchrow("""
        SELECT link_id FROM SocialLinks
        WHERE user_id = $1 AND conversation_id = $2
        AND entity1_type = $3 AND entity1_id = $4
        AND entity2_type = $5 AND entity2_id = $6
    """, user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id)
    
    if existing:
        return existing['link_id']
    
    # Create new link
    link_id = await conn.fetchval("""
        INSERT INTO SocialLinks (
            user_id, conversation_id,
            entity1_type, entity1_id, entity2_type, entity2_id,
            link_type, link_level, link_history, dynamics,
            experienced_crossroads, experienced_rituals
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11::jsonb, $12::jsonb)
        ON CONFLICT (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id)
        DO UPDATE SET link_id = EXCLUDED.link_id
        RETURNING link_id
    """,
        user_id, conversation_id,
        entity1_type, entity1_id, entity2_type, entity2_id,
        kwargs.get('link_type', 'neutral'),
        kwargs.get('link_level', 0),
        json.dumps(kwargs.get('link_history', [])),
        json.dumps(kwargs.get('dynamics', {})),
        json.dumps(kwargs.get('experienced_crossroads', [])),
        json.dumps(kwargs.get('experienced_rituals', []))
    )
    
    # Log canonical event
    await log_canonical_event(
        ctx, conn,
        f"Social link created between {entity1_type} {entity1_id} and {entity2_type} {entity2_id}",
        tags=['social_link', 'creation'],
        significance=5
    )
    
    return link_id

async def find_or_create_npc_group(ctx, conn, group_data: Dict[str, Any]) -> int:
    """
    Find or create an NPC group.
    """
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
    Create a journal entry canonically.
    """
    entry_id = await conn.fetchval("""
        INSERT INTO PlayerJournal (
            user_id, conversation_id, entry_type, entry_text,
            revelation_types, narrative_moment, fantasy_flag,
            intensity_level, timestamp, entry_metadata,
            importance, tags
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP, $9::jsonb, $10, $11::jsonb)
        RETURNING id
    """,
        ctx.user_id, ctx.conversation_id, entry_type, entry_text,
        kwargs.get('revelation_types'),
        kwargs.get('narrative_moment'),
        kwargs.get('fantasy_flag', False),
        kwargs.get('intensity_level', 0),
        json.dumps(kwargs.get('entry_metadata', {})),
        kwargs.get('importance', 0.5),
        json.dumps(kwargs.get('tags', []))
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

async def find_or_create_player_stats(
    ctx, conn, 
    player_name: str, 
    **initial_stats
) -> None:
    """
    Find or create player stats entry with initial values.
    Does not update existing stats, only creates if missing.
    """
    # Check if player stats exist
    exists = await conn.fetchval("""
        SELECT COUNT(*) FROM PlayerStats
        WHERE user_id = $1 AND conversation_id = $2 AND player_name = $3
    """, ctx.user_id, ctx.conversation_id, player_name)
    
    if not exists:
        # Set defaults for any missing stats
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
        
        # Log canonical event
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
    """Update a player's stat while logging the change canonically."""
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

    await log_stat_change(
        ctx, conn,
        player_name,
        stat_name,
        current_value,
        new_value,
        reason
    )

async def find_or_create_inventory_item(
    ctx, conn,
    item_name: str,
    player_name: str = "Chase",
    **item_data
) -> int:
    """
    Find or create an inventory item for a player.
    """
    # Check if item already exists
    existing = await conn.fetchrow("""
        SELECT item_id, quantity FROM PlayerInventory
        WHERE user_id = $1 AND conversation_id = $2 
        AND player_name = $3 AND item_name = $4
    """, ctx.user_id, ctx.conversation_id, player_name, item_name)
    
    if existing:
        # Update quantity if provided
        if 'quantity' in item_data:
            new_quantity = existing['quantity'] + item_data['quantity']
            await conn.execute("""
                UPDATE PlayerInventory 
                SET quantity = $1, date_acquired = CURRENT_TIMESTAMP
                WHERE item_id = $2
            """, new_quantity, existing['item_id'])
        return existing['item_id']
    
    # Create new item
    item_id = await conn.fetchval("""
        INSERT INTO PlayerInventory (
            user_id, conversation_id, player_name, item_name,
            item_description, item_category, item_properties,
            quantity, equipped, date_acquired
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, CURRENT_TIMESTAMP)
        RETURNING item_id
    """,
        ctx.user_id, ctx.conversation_id, player_name, item_name,
        item_data.get('item_description', ''),
        item_data.get('item_category', 'misc'),
        json.dumps(item_data.get('item_properties', {})),
        item_data.get('quantity', 1),
        item_data.get('equipped', False)
    )
    
    await log_canonical_event(
        ctx, conn,
        f"Player {player_name} acquired new item: {item_name}",
        tags=['inventory', 'item_acquisition', item_data.get('item_category', 'misc')],
        significance=4
    )
    
    return item_id

async def find_or_create_currency(
    ctx, conn,
    currency_name: str,
    **currency_data
) -> int:
    """
    Find or create a currency system.
    """
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
    """
    Find or create an inventory item for a player.
    """
    # Check if item already exists
    existing = await conn.fetchrow("""
        SELECT item_id, quantity FROM PlayerInventory
        WHERE user_id = $1 AND conversation_id = $2 
        AND player_name = $3 AND item_name = $4
    """, ctx.user_id, ctx.conversation_id, player_name, item_name)
    
    if existing:
        return existing['item_id']
    
    # Create new item
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
    # Store environment description
    env_desc = kwargs.get('environment_desc', '')
    env_history = kwargs.get('environment_history', '')
    full_desc = f"{env_desc}\n\nHistory: {env_history}" if env_history else env_desc
    
    await update_current_roleplay(ctx, conn, ctx.user_id, ctx.conversation_id, 
                                 'EnvironmentDesc', full_desc)
    
    # Store setting name
    await update_current_roleplay(ctx, conn, ctx.user_id, ctx.conversation_id,
                                 'CurrentSetting', setting_name)
    
    # Store calendar data if provided
    if 'calendar_data' in kwargs:
        await update_current_roleplay(ctx, conn, ctx.user_id, ctx.conversation_id,
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
    
    await update_current_roleplay(ctx, conn, ctx.user_id, ctx.conversation_id,
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

async def create_opening_message(ctx, conn, sender: str, content: str) -> int:
    """
    Create the opening message canonically.
    """
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
