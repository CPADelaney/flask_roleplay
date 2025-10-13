"""Create canonical entity-card and episodic views."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


description = "Create canonical entity-card and episodic views"


CREATE_ENTITY_CARDS_VIEW = """
CREATE OR REPLACE VIEW public.v_entity_cards AS
WITH npc_cards AS (
    SELECT
        'npc'::text AS entity_type,
        ns.npc_id::text AS entity_id,
        ns.user_id,
        ns.conversation_id,
        jsonb_build_object(
            'npc_id', ns.npc_id,
            'npc_name', ns.npc_name,
            'role', ns.role,
            'description', ns.physical_description,
            'location', ns.current_location,
            'personality_traits', COALESCE(ns.personality_traits, '[]'::jsonb),
            'relationships', COALESCE(ns.relationships, '[]'::jsonb)
        ) AS card,
        to_tsvector(
            'english',
            COALESCE(ns.npc_name, '') || ' ' ||
            COALESCE(ns.physical_description, '') || ' ' ||
            COALESCE(ns.role, '')
        ) AS search_vector,
        ns.embedding,
        ns.created_at::timestamptz AS updated_at
    FROM npcstats AS ns
),
location_cards AS (
    SELECT
        'location'::text AS entity_type,
        l.id::text AS entity_id,
        l.user_id,
        l.conversation_id,
        jsonb_build_object(
            'location_id', l.id,
            'location_name', l.location_name,
            'description', l.description,
            'location_type', l.location_type,
            'notable_features', COALESCE(l.notable_features, '[]'::jsonb)
        ) AS card,
        to_tsvector(
            'english',
            COALESCE(l.location_name, '') || ' ' ||
            COALESCE(l.description, '') || ' ' ||
            COALESCE(l.location_type, '')
        ) AS search_vector,
        l.embedding,
        NULL::timestamptz AS updated_at
    FROM locations AS l
),
memory_cards AS (
    SELECT
        'memory'::text AS entity_type,
        um.id::text AS entity_id,
        um.user_id,
        um.conversation_id,
        jsonb_build_object(
            'memory_id', um.id,
            'entity_type', um.entity_type,
            'entity_id', um.entity_id,
            'content', um.memory_text,
            'memory_type', um.memory_type,
            'importance', um.significance,
            'tags', COALESCE(um.tags, '[]'::jsonb),
            'metadata', COALESCE(um.metadata, '{}'::jsonb)
        ) AS card,
        to_tsvector('english', COALESCE(um.memory_text, '')) AS search_vector,
        um.embedding,
        um.timestamp::timestamptz AS updated_at
    FROM unified_memories AS um
    WHERE um.status = 'active'
      AND COALESCE(um.is_archived, FALSE) = FALSE
)
SELECT * FROM npc_cards
UNION ALL
SELECT * FROM location_cards
UNION ALL
SELECT * FROM memory_cards;
"""


DROP_ENTITY_CARDS_VIEW = """
DROP VIEW IF EXISTS public.v_entity_cards;
"""


CREATE_RECENT_CHUNKS_VIEW = """
CREATE OR REPLACE VIEW public.v_recent_chunks AS
SELECT
    um.id::text AS chunk_id,
    um.user_id,
    um.conversation_id,
    jsonb_build_object(
        'memory_id', um.id,
        'entity_type', um.entity_type,
        'entity_id', um.entity_id,
        'content', um.memory_text,
        'memory_type', um.memory_type,
        'tags', COALESCE(um.tags, '[]'::jsonb),
        'metadata', COALESCE(um.metadata, '{}'::jsonb)
    ) AS chunk,
    to_tsvector('english', COALESCE(um.memory_text, '')) AS search_vector,
    um.embedding,
    um.timestamp::timestamptz AS occurred_at
FROM unified_memories AS um
WHERE um.status = 'active'
  AND COALESCE(um.is_archived, FALSE) = FALSE
  AND um.memory_type = ANY(ARRAY['observation','episodic','experience','reflection']);
"""


DROP_RECENT_CHUNKS_VIEW = """
DROP VIEW IF EXISTS public.v_recent_chunks;
"""


async def upgrade(conn):
    await conn.execute(CREATE_ENTITY_CARDS_VIEW)
    await conn.execute(CREATE_RECENT_CHUNKS_VIEW)
    logger.info("Created v_entity_cards and v_recent_chunks views")


async def downgrade(conn):
    await conn.execute(DROP_ENTITY_CARDS_VIEW)
    await conn.execute(DROP_RECENT_CHUNKS_VIEW)
    logger.info("Dropped v_entity_cards and v_recent_chunks views")
