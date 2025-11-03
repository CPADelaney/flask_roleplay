"""Add hot-path indexes for scene context and entity cards."""
from __future__ import annotations


CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_v_scene_context_user_conversation
    ON public.v_scene_context (user_id, conversation_id);

CREATE INDEX IF NOT EXISTS idx_v_entity_cards_user_conversation
    ON public.v_entity_cards (user_id, conversation_id);

CREATE INDEX IF NOT EXISTS idx_v_entity_cards_search
    ON public.v_entity_cards USING GIN (search_vector);
"""

DROP_INDEXES = """
DROP INDEX IF EXISTS idx_v_scene_context_user_conversation;
DROP INDEX IF EXISTS idx_v_entity_cards_user_conversation;
DROP INDEX IF EXISTS idx_v_entity_cards_search;
"""


async def upgrade(conn):
    """Apply the migration."""

    await conn.execute(CREATE_INDEXES)


async def downgrade(conn):
    """Revert the migration."""

    await conn.execute(DROP_INDEXES)
