"""Create conversation_items table to store scene seals."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20241119_conversation_items"
down_revision = "20241009_player_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversation_items",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("item_type", sa.Text(), nullable=False),
        sa.Column(
            "content",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("conversation_id", "item_type", name="uq_conversation_items_per_type"),
    )
    op.create_index(
        "idx_conversation_items_conversation",
        "conversation_items",
        ["conversation_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_conversation_items_conversation", table_name="conversation_items")
    op.drop_table("conversation_items")
