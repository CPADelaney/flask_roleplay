"""Install canon.apply_event(jsonb) helper."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

description = "Install canon.apply_event(jsonb) helper"


CANON_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS canon;

CREATE TABLE IF NOT EXISTS canon.events (
    event_id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL UNIQUE,
    payload JSONB NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


APPLY_EVENT_FN = """
CREATE OR REPLACE FUNCTION canon.apply_event(payload jsonb)
RETURNS jsonb
LANGUAGE plpgsql
AS $$
DECLARE
    v_request_id uuid;
    v_existing_event canon.events%ROWTYPE;
    v_event_id bigint;
BEGIN
    IF payload IS NULL THEN
        RAISE EXCEPTION 'payload must not be null';
    END IF;

    v_request_id := (payload ->> 'request_id')::uuid;
    IF v_request_id IS NULL THEN
        RAISE EXCEPTION 'request_id is required';
    END IF;

    SELECT * INTO v_existing_event
    FROM canon.events
    WHERE request_id = v_request_id;

    IF FOUND THEN
        RETURN jsonb_build_object(
            'event_id', v_existing_event.event_id,
            'applied', FALSE,
            'replayed', TRUE,
            'payload', v_existing_event.payload
        );
    END IF;

    INSERT INTO canon.events(request_id, payload)
    VALUES (v_request_id, payload)
    RETURNING event_id INTO v_event_id;

    RETURN jsonb_build_object(
        'event_id', v_event_id,
        'applied', TRUE,
        'replayed', FALSE
    );
END;
$$;
"""


DROP_FUNCTION_SQL = """
DROP FUNCTION IF EXISTS canon.apply_event(jsonb);
"""


async def upgrade(conn):
    """Install the schema, table, and RPC helper."""

    await conn.execute(CANON_SCHEMA_SQL)
    await conn.execute(APPLY_EVENT_FN)
    logger.info("Installed canon.apply_event(jsonb)")


async def downgrade(conn):
    """Remove the RPC helper."""

    await conn.execute(DROP_FUNCTION_SQL)
    logger.info("Dropped canon.apply_event(jsonb)")
