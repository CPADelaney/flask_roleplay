from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lore.lore_generator import BaseGenerator


class DummyConn:
    def __init__(self, column_response):
        self.column_response = column_response
        self.calls = 0

    async def fetchval(self, query, *args):
        if "information_schema.columns" in query.lower():
            self.calls += 1
            return self.column_response
        raise AssertionError("Unexpected query during test")


class DummyConnNoAccess:
    async def fetchval(self, query, *args):
        raise AssertionError("PK lookup should be cached and skip DB access")


@pytest.mark.asyncio
async def test_resolve_faction_pk_prefers_faction_id():
    generator = BaseGenerator()
    conn = DummyConn("faction_id")

    column = await generator._resolve_faction_pk_column(conn)

    assert column == "faction_id"
    assert conn.calls == 1


@pytest.mark.asyncio
async def test_resolve_faction_pk_falls_back_to_id():
    generator = BaseGenerator()
    conn = DummyConn("id")

    column = await generator._resolve_faction_pk_column(conn)

    assert column == "id"
    assert conn.calls == 1


@pytest.mark.asyncio
async def test_resolve_faction_pk_caches_value():
    generator = BaseGenerator()
    primary_conn = DummyConn("faction_id")

    first_column = await generator._resolve_faction_pk_column(primary_conn)
    assert first_column == "faction_id"
    assert primary_conn.calls == 1

    # Once resolved we should not query again even if a different connection is passed.
    cached_column = await generator._resolve_faction_pk_column(DummyConnNoAccess())

    assert cached_column == "faction_id"
