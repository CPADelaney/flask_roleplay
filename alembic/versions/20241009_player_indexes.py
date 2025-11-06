"""Add concurrent indexes for player stats tables."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20241009_player_indexes"
down_revision = "20240927_conflict_resolution"
branch_labels = None
depends_on = None

_INDEX_CREATION_STATEMENTS = (
    sa.text(
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_playerstats_user_conv ON "PlayerStats" (user_id, conversation_id)'
    ),
    sa.text(
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_playervitals_user_conv ON "PlayerVitals" (user_id, conversation_id)'
    ),
)

_INDEX_DROP_STATEMENTS = (
    sa.text("DROP INDEX CONCURRENTLY IF EXISTS idx_playerstats_user_conv"),
    sa.text("DROP INDEX CONCURRENTLY IF EXISTS idx_playervitals_user_conv"),
)


def upgrade() -> None:
    """Create the player stats indexes concurrently to avoid table locks."""
    conn = op.get_bind()
    with op.get_context().autocommit_block():
        for statement in _INDEX_CREATION_STATEMENTS:
            conn.execute(statement)


def downgrade() -> None:
    """Drop the player stats indexes concurrently."""
    conn = op.get_bind()
    with op.get_context().autocommit_block():
        for statement in _INDEX_DROP_STATEMENTS:
            conn.execute(statement)
