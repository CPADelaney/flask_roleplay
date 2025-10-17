"""Create Places and PlaceEdges tables for hierarchical location storage."""

from __future__ import annotations

description = "Create Places and PlaceEdges tables"


CREATE_PLACES_SQL = """
CREATE TABLE IF NOT EXISTS Places (
    id BIGSERIAL PRIMARY KEY,
    scope TEXT NOT NULL DEFAULT 'real',
    place_key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    level TEXT NOT NULL,
    admin_path JSONB,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CREATE_PLACES_INDEXES = [
    """
    CREATE INDEX IF NOT EXISTS idx_places_level
    ON Places (level);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_places_scope_level_name
    ON Places (scope, level, normalized_name);
    """,
]

CREATE_PLACE_EDGES_SQL = """
CREATE TABLE IF NOT EXISTS PlaceEdges (
    id BIGSERIAL PRIMARY KEY,
    parent_id BIGINT NOT NULL REFERENCES Places(id) ON DELETE CASCADE,
    child_id BIGINT NOT NULL REFERENCES Places(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    distance_km DOUBLE PRECISION,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (parent_id, child_id, kind)
);
"""

CREATE_PLACE_EDGES_INDEXES = [
    """
    CREATE INDEX IF NOT EXISTS idx_place_edges_parent
    ON PlaceEdges (parent_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_place_edges_child
    ON PlaceEdges (child_id);
    """,
]


DROP_PLACE_EDGES_SQL = """
DROP TABLE IF EXISTS PlaceEdges;
"""

DROP_PLACES_SQL = """
DROP TABLE IF EXISTS Places;
"""


async def upgrade(conn):
    """Install the Places graph tables."""

    await conn.execute(CREATE_PLACES_SQL)
    for stmt in CREATE_PLACES_INDEXES:
        await conn.execute(stmt)

    await conn.execute(CREATE_PLACE_EDGES_SQL)
    for stmt in CREATE_PLACE_EDGES_INDEXES:
        await conn.execute(stmt)


async def downgrade(conn):
    """Remove the Places graph tables."""

    await conn.execute(DROP_PLACE_EDGES_SQL)
    await conn.execute(DROP_PLACES_SQL)
