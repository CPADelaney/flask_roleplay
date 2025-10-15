"""Create geo_cache and world_locations tables for toponym lookups."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

description = "Create geo_cache and world_locations tables"


CREATE_GEO_CACHE_SQL = """
CREATE TABLE IF NOT EXISTS geo_cache (
    provider TEXT NOT NULL,
    query TEXT NOT NULL,
    normalized_query TEXT NOT NULL,
    response JSONB NOT NULL,
    confidence DOUBLE PRECISION,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (provider, normalized_query)
);
"""

CREATE_GEO_CACHE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS geo_cache_expires_at_idx
    ON geo_cache (expires_at)
    WHERE expires_at IS NOT NULL;
"""

CREATE_WORLD_LOCATIONS_SQL = """
CREATE TABLE IF NOT EXISTS world_locations (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    country_code TEXT,
    admin1 TEXT,
    admin2 TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    population BIGINT,
    feature_class TEXT,
    feature_code TEXT,
    data_source TEXT,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


DROP_GEO_CACHE_INDEX_SQL = """
DROP INDEX IF EXISTS geo_cache_expires_at_idx;
"""

DROP_GEO_CACHE_SQL = """
DROP TABLE IF EXISTS geo_cache;
"""

DROP_WORLD_LOCATIONS_SQL = """
DROP TABLE IF EXISTS world_locations;
"""


async def upgrade(conn):
    """Install geo_cache and world_locations tables."""

    await conn.execute(CREATE_GEO_CACHE_SQL)
    await conn.execute(CREATE_GEO_CACHE_INDEX_SQL)
    await conn.execute(CREATE_WORLD_LOCATIONS_SQL)
    logger.info("Created geo_cache and world_locations tables")


async def downgrade(conn):
    """Remove geo_cache and world_locations tables."""

    await conn.execute(DROP_GEO_CACHE_INDEX_SQL)
    await conn.execute(DROP_WORLD_LOCATIONS_SQL)
    await conn.execute(DROP_GEO_CACHE_SQL)
    logger.info("Dropped geo_cache and world_locations tables")
