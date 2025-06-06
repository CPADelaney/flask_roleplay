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
