"""Create conflict_resolution table for persistent conflict pipeline."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20240927_conflict_resolution"
down_revision = "20240918_create_outbox"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conflict_resolution",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("conflict_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("draft_text", sa.Text(), nullable=True),
        sa.Column("eval_score", sa.Float(), nullable=True),
        sa.Column("eval_notes", sa.Text(), nullable=True),
        sa.Column("integrated_changes", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index(
        "uq_conflict_resolution_active",
        "conflict_resolution",
        ["conflict_id", "status"],
        unique=True,
        postgresql_where=sa.text("status IN ('DRAFT','EVAL','CANON','INTEGRATING')"),
        sqlite_where=sa.text("status IN ('DRAFT','EVAL','CANON','INTEGRATING')"),
    )


def downgrade() -> None:
    op.drop_index("uq_conflict_resolution_active", table_name="conflict_resolution")
    op.drop_table("conflict_resolution")
