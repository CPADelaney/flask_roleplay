"""Create projection views backed by the canon schema."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


description = "Create projection views backed by canon schema"


CREATE_CANON_VIEWS = """
CREATE SCHEMA IF NOT EXISTS canon;

DROP VIEW IF EXISTS public.v_scene_context;
DROP VIEW IF EXISTS public.v_entity_cards;
DROP VIEW IF EXISTS public.v_recent_chunks;
DROP VIEW IF EXISTS canon.scene_context_projection;
DROP VIEW IF EXISTS canon.entity_cards_projection;
DROP VIEW IF EXISTS canon.recent_chunks_projection;

CREATE OR REPLACE VIEW canon.scene_context_projection AS
WITH roleplay AS (
    SELECT
        user_id,
        conversation_id,
        jsonb_object_agg(key, to_jsonb(value)) AS current_roleplay
    FROM currentroleplay
    GROUP BY user_id, conversation_id
),
player_stats AS (
    SELECT
        user_id,
        conversation_id,
        jsonb_build_object(
            'corruption', corruption,
            'confidence', confidence,
            'willpower', willpower,
            'obedience', obedience,
            'dependency', dependency,
            'lust', lust,
            'mental_resilience', mental_resilience,
            'physical_endurance', physical_endurance
        ) AS player_stats
    FROM playerstats
),
npcs AS (
    SELECT
        user_id,
        conversation_id,
        jsonb_agg(
            jsonb_build_object(
                'npc_id', npc_id,
                'npc_name', npc_name,
                'physical_description', physical_description,
                'personality_traits', to_jsonb(personality_traits),
                'trust', trust,
                'dominance', dominance,
                'cruelty', cruelty,
                'affection', affection,
                'intensity', intensity,
                'introduced', introduced,
                'current_location', current_location,
                'closeness', closeness,
                'respect', respect
            )
            ORDER BY closeness DESC NULLS LAST, trust DESC NULLS LAST, npc_name
        ) FILTER (WHERE introduced IS TRUE) AS npcs_present
    FROM npcstats
    GROUP BY user_id, conversation_id
),
events AS (
    SELECT
        user_id,
        conversation_id,
        jsonb_agg(
                jsonb_build_object(
                    'event_name', event_name,
                    'description', description,
                    'location', location,
                    'fantasy_level', NULL,
                    'day', day,
                    'time_of_day', time_of_day,
                    'start_time', start_time,
                    'end_time', end_time
                )
        ) AS events_payload
    FROM events
    GROUP BY user_id, conversation_id
),
quests AS (
    SELECT
        user_id,
        conversation_id,
        jsonb_agg(
            jsonb_build_object(
                'quest_id', quest_id,
                'quest_name', quest_name,
                'status', status,
                'progress_detail', progress_detail,
                'quest_giver', quest_giver,
                'reward', reward
            )
        ) AS quests_payload
    FROM quests
    GROUP BY user_id, conversation_id
)
SELECT
    COALESCE(r.user_id, ps.user_id, n.user_id, e.user_id, q.user_id) AS user_id,
    COALESCE(r.conversation_id, ps.conversation_id, n.conversation_id, e.conversation_id, q.conversation_id) AS conversation_id,
    jsonb_build_object(
        'current_roleplay', COALESCE(r.current_roleplay, '{}'::jsonb),
        'player_stats', COALESCE(ps.player_stats, '{}'::jsonb),
        'npcs_present', COALESCE(n.npcs_present, '[]'::jsonb),
        'events', COALESCE(e.events_payload, '[]'::jsonb),
        'quests', COALESCE(q.quests_payload, '[]'::jsonb)
    ) AS scene_context
FROM roleplay r
FULL OUTER JOIN player_stats ps USING (user_id, conversation_id)
FULL OUTER JOIN npcs n USING (user_id, conversation_id)
FULL OUTER JOIN events e USING (user_id, conversation_id)
FULL OUTER JOIN quests q USING (user_id, conversation_id);

CREATE OR REPLACE VIEW canon.entity_cards_projection AS
WITH npc_cards AS (
    SELECT
        'npc'::text AS entity_type,
        npc_id::text AS entity_id,
        user_id,
        conversation_id,
        jsonb_build_object(
            'npc_id', npc_id,
            'npc_name', npc_name,
            'role', role,
            'description', physical_description,
            'location', current_location,
            'personality_traits', COALESCE(personality_traits, '[]'::jsonb),
            'relationships', COALESCE(relationships, '[]'::jsonb)
        ) AS card,
        to_tsvector(
            'english',
            COALESCE(npc_name, '') || ' ' ||
            COALESCE(physical_description, '') || ' ' ||
            COALESCE(role, '')
        ) AS search_vector,
        embedding,
        created_at::timestamptz AS updated_at
    FROM npcstats
),
location_cards AS (
    SELECT
        'location'::text AS entity_type,
        COALESCE(location_id, id)::text AS entity_id,
        user_id,
        conversation_id,
        jsonb_build_object(
            'location_id', COALESCE(location_id, id),
            'location_name', location_name,
            'description', description,
            'location_type', location_type,
            'notable_features', COALESCE(notable_features, '[]'::jsonb)
        ) AS card,
        to_tsvector(
            'english',
            COALESCE(location_name, '') || ' ' ||
            COALESCE(description, '') || ' ' ||
            COALESCE(location_type, '')
        ) AS search_vector,
        embedding,
        NULL::timestamptz AS updated_at
    FROM locations
),
memory_cards AS (
    SELECT
        'memory'::text AS entity_type,
        id::text AS entity_id,
        user_id,
        conversation_id,
        jsonb_build_object(
            'memory_id', id,
            'entity_type', entity_type,
            'entity_id', entity_id,
            'content', memory_text,
            'memory_type', memory_type,
            'importance', significance,
            'tags', COALESCE(tags, '[]'::jsonb),
            'metadata', COALESCE(metadata, '{}'::jsonb)
        ) AS card,
        to_tsvector('english', COALESCE(memory_text, '')) AS search_vector,
        embedding,
        timestamp::timestamptz AS updated_at
    FROM unified_memories
    WHERE status = 'active'
      AND COALESCE(is_archived, FALSE) = FALSE
)
SELECT * FROM npc_cards
UNION ALL
SELECT * FROM location_cards
UNION ALL
SELECT * FROM memory_cards;

CREATE OR REPLACE VIEW canon.recent_chunks_projection AS
SELECT
    id::text AS chunk_id,
    user_id,
    conversation_id,
    jsonb_build_object(
        'memory_id', id,
        'entity_type', entity_type,
        'entity_id', entity_id,
        'content', memory_text,
        'memory_type', memory_type,
        'tags', COALESCE(tags, '[]'::jsonb),
        'metadata', COALESCE(metadata, '{}'::jsonb)
    ) AS chunk,
    to_tsvector('english', COALESCE(memory_text, '')) AS search_vector,
    embedding,
    timestamp::timestamptz AS occurred_at
FROM unified_memories
WHERE status = 'active'
  AND COALESCE(is_archived, FALSE) = FALSE
  AND memory_type = ANY(ARRAY['observation','episodic','experience','reflection']);

CREATE OR REPLACE VIEW public.v_scene_context AS
SELECT * FROM canon.scene_context_projection;

CREATE OR REPLACE VIEW public.v_entity_cards AS
SELECT * FROM canon.entity_cards_projection;

CREATE OR REPLACE VIEW public.v_recent_chunks AS
SELECT * FROM canon.recent_chunks_projection;
"""


DROP_CANON_VIEWS = """
DROP VIEW IF EXISTS public.v_scene_context;
DROP VIEW IF EXISTS public.v_entity_cards;
DROP VIEW IF EXISTS public.v_recent_chunks;
DROP VIEW IF EXISTS canon.scene_context_projection;
DROP VIEW IF EXISTS canon.entity_cards_projection;
DROP VIEW IF EXISTS canon.recent_chunks_projection;

CREATE OR REPLACE VIEW public.v_entity_cards AS
WITH npc_cards AS (
    SELECT
        'npc'::text AS entity_type,
        npc_id::text AS entity_id,
        user_id,
        conversation_id,
        jsonb_build_object(
            'npc_id', npc_id,
            'npc_name', npc_name,
            'role', role,
            'description', physical_description,
            'location', current_location,
            'personality_traits', COALESCE(personality_traits, '[]'::jsonb),
            'relationships', COALESCE(relationships, '[]'::jsonb)
        ) AS card,
        to_tsvector(
            'english',
            COALESCE(npc_name, '') || ' ' ||
            COALESCE(physical_description, '') || ' ' ||
            COALESCE(role, '')
        ) AS search_vector,
        embedding,
        created_at::timestamptz AS updated_at
    FROM npcstats
),
location_cards AS (
    SELECT
        'location'::text AS entity_type,
        id::text AS entity_id,
        user_id,
        conversation_id,
        jsonb_build_object(
            'location_id', id,
            'location_name', location_name,
            'description', description,
            'location_type', location_type,
            'notable_features', COALESCE(notable_features, '[]'::jsonb)
        ) AS card,
        to_tsvector(
            'english',
            COALESCE(location_name, '') || ' ' ||
            COALESCE(description, '') || ' ' ||
            COALESCE(location_type, '')
        ) AS search_vector,
        embedding,
        NULL::timestamptz AS updated_at
    FROM locations
),
memory_cards AS (
    SELECT
        'memory'::text AS entity_type,
        id::text AS entity_id,
        user_id,
        conversation_id,
        jsonb_build_object(
            'memory_id', id,
            'entity_type', entity_type,
            'entity_id', entity_id,
            'content', memory_text,
            'memory_type', memory_type,
            'importance', significance,
            'tags', COALESCE(tags, '[]'::jsonb),
            'metadata', COALESCE(metadata, '{}'::jsonb)
        ) AS card,
        to_tsvector('english', COALESCE(memory_text, '')) AS search_vector,
        embedding,
        timestamp::timestamptz AS updated_at
    FROM unified_memories
    WHERE status = 'active'
      AND COALESCE(is_archived, FALSE) = FALSE
)
SELECT * FROM npc_cards
UNION ALL
SELECT * FROM location_cards
UNION ALL
SELECT * FROM memory_cards;

CREATE OR REPLACE VIEW public.v_recent_chunks AS
SELECT
    id::text AS chunk_id,
    user_id,
    conversation_id,
    jsonb_build_object(
        'memory_id', id,
        'entity_type', entity_type,
        'entity_id', entity_id,
        'content', memory_text,
        'memory_type', memory_type,
        'tags', COALESCE(tags, '[]'::jsonb),
        'metadata', COALESCE(metadata, '{}'::jsonb)
    ) AS chunk,
    to_tsvector('english', COALESCE(memory_text, '')) AS search_vector,
    embedding,
    timestamp::timestamptz AS occurred_at
FROM unified_memories
WHERE status = 'active'
  AND COALESCE(is_archived, FALSE) = FALSE
  AND memory_type = ANY(ARRAY['observation','episodic','experience','reflection']);
"""


async def upgrade(conn):
    await conn.execute(CREATE_CANON_VIEWS)
    logger.info("Created canon-backed projection views")


async def downgrade(conn):
    await conn.execute(DROP_CANON_VIEWS)
    logger.info("Restored legacy projection views")

