"""Add world metadata columns to Locations and enforce coordinate types."""

from __future__ import annotations

import logging


description = "Add world metadata columns to Locations"

logger = logging.getLogger(__name__)

ADD_OPTIONAL_COLUMNS = [
    """
    ALTER TABLE Locations
    ADD COLUMN IF NOT EXISTS room TEXT;
    """,
    """
    ALTER TABLE Locations
    ADD COLUMN IF NOT EXISTS building TEXT;
    """,
    """
    ALTER TABLE Locations
    ADD COLUMN IF NOT EXISTS district_type TEXT;
    """,
]

ADD_REQUIRED_COLUMNS = [
    """
    ALTER TABLE Locations
    ADD COLUMN IF NOT EXISTS planet TEXT NOT NULL DEFAULT 'Earth';
    """,
    """
    ALTER TABLE Locations
    ADD COLUMN IF NOT EXISTS galaxy TEXT NOT NULL DEFAULT 'Milky Way';
    """,
    """
    ALTER TABLE Locations
    ADD COLUMN IF NOT EXISTS realm TEXT DEFAULT 'Prime Material';
    """,
    """
    ALTER TABLE Locations
    ADD COLUMN IF NOT EXISTS lat DOUBLE PRECISION;
    """,
    """
    ALTER TABLE Locations
    ADD COLUMN IF NOT EXISTS lon DOUBLE PRECISION;
    """,
    """
    ALTER TABLE Locations
    ADD COLUMN IF NOT EXISTS is_fictional BOOLEAN NOT NULL DEFAULT FALSE;
    """,
]

ENFORCE_COORDINATE_TYPES = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'locations'
          AND column_name = 'lat'
          AND data_type <> 'double precision'
    ) THEN
        ALTER TABLE Locations
            ALTER COLUMN lat TYPE DOUBLE PRECISION
            USING NULLIF(trim(lat::text), '')::double precision;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'locations'
          AND column_name = 'lon'
          AND data_type <> 'double precision'
    ) THEN
        ALTER TABLE Locations
            ALTER COLUMN lon TYPE DOUBLE PRECISION
            USING NULLIF(trim(lon::text), '')::double precision;
    END IF;
END $$;
"""

ENFORCE_REQUIRED_DEFAULTS = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'locations'
          AND column_name = 'planet'
    ) THEN
        UPDATE Locations SET planet = COALESCE(planet, 'Earth');
        ALTER TABLE Locations ALTER COLUMN planet SET DEFAULT 'Earth';
        ALTER TABLE Locations ALTER COLUMN planet SET NOT NULL;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'locations'
          AND column_name = 'galaxy'
    ) THEN
        UPDATE Locations SET galaxy = COALESCE(galaxy, 'Milky Way');
        ALTER TABLE Locations ALTER COLUMN galaxy SET DEFAULT 'Milky Way';
        ALTER TABLE Locations ALTER COLUMN galaxy SET NOT NULL;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'locations'
          AND column_name = 'realm'
    ) THEN
        UPDATE Locations
        SET realm = 'Prime Material'
        WHERE realm IS NULL
           OR btrim(realm) = ''
           OR lower(realm) = 'physical';
        ALTER TABLE Locations ALTER COLUMN realm SET DEFAULT 'Prime Material';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'locations'
          AND column_name = 'is_fictional'
    ) THEN
        UPDATE Locations SET is_fictional = COALESCE(is_fictional, FALSE);
        ALTER TABLE Locations ALTER COLUMN is_fictional SET DEFAULT FALSE;
        ALTER TABLE Locations ALTER COLUMN is_fictional SET NOT NULL;
    END IF;
END $$;
"""

BACKFILL_NOTE = (
    "Default world metadata now backfills existing Locations rows. Review any non-Earth "
    "or non-Prime Material settings and update their planet/galaxy/realm values manually."
)


async def upgrade(conn):
    """Apply the migration."""

    for stmt in ADD_OPTIONAL_COLUMNS:
        await conn.execute(stmt)

    for stmt in ADD_REQUIRED_COLUMNS:
        await conn.execute(stmt)

    await conn.execute(ENFORCE_COORDINATE_TYPES)
    await conn.execute(ENFORCE_REQUIRED_DEFAULTS)

    logger.info(BACKFILL_NOTE)


async def downgrade(conn):
    """Revert the migration."""

    await conn.execute(ENFORCE_COORDINATE_TYPES.replace("double precision", "text"))

    await conn.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'locations'
                  AND column_name = 'planet'
            ) THEN
                ALTER TABLE Locations ALTER COLUMN planet DROP NOT NULL;
                ALTER TABLE Locations ALTER COLUMN planet DROP DEFAULT;
            END IF;

            IF EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'locations'
                  AND column_name = 'galaxy'
            ) THEN
                ALTER TABLE Locations ALTER COLUMN galaxy DROP NOT NULL;
                ALTER TABLE Locations ALTER COLUMN galaxy DROP DEFAULT;
            END IF;

            IF EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'locations'
                  AND column_name = 'realm'
            ) THEN
                ALTER TABLE Locations ALTER COLUMN realm DROP NOT NULL;
                ALTER TABLE Locations ALTER COLUMN realm DROP DEFAULT;
            END IF;

            IF EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'locations'
                  AND column_name = 'is_fictional'
            ) THEN
                ALTER TABLE Locations ALTER COLUMN is_fictional DROP NOT NULL;
                ALTER TABLE Locations ALTER COLUMN is_fictional DROP DEFAULT;
            END IF;
        END $$;
        """
    )

    await conn.execute("ALTER TABLE Locations DROP COLUMN IF EXISTS room;")
    await conn.execute("ALTER TABLE Locations DROP COLUMN IF EXISTS building;")
    await conn.execute("ALTER TABLE Locations DROP COLUMN IF EXISTS district_type;")
    await conn.execute("ALTER TABLE Locations DROP COLUMN IF EXISTS planet;")
    await conn.execute("ALTER TABLE Locations DROP COLUMN IF EXISTS galaxy;")
    await conn.execute("ALTER TABLE Locations DROP COLUMN IF EXISTS realm;")
    await conn.execute("ALTER TABLE Locations DROP COLUMN IF EXISTS lat;")
    await conn.execute("ALTER TABLE Locations DROP COLUMN IF EXISTS lon;")
    await conn.execute("ALTER TABLE Locations DROP COLUMN IF EXISTS is_fictional;")
