from pathlib import Path
import sys
import json
import asyncio

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from contextlib import asynccontextmanager

from lore.lore_generator import BaseGenerator, WorldBuilder


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


class RecordingConn:
    def __init__(self, response=42):
        self.response = response
        self.calls = 0
        self.query = None
        self.args = None

    async def fetchval(self, query, *args):
        self.calls += 1
        self.query = query
        self.args = args
        return self.response


def test_store_world_lore_serializes_tags(monkeypatch):
    builder = WorldBuilder(user_id=99, conversation_id=123)
    conn = RecordingConn(response=7)

    @asynccontextmanager
    async def fake_connection_context(*args, **kwargs):
        yield conn

    monkeypatch.setattr(
        "lore.lore_generator.get_db_connection_context",
        fake_connection_context,
    )

    async def run_store():
        return await builder._store_world_lore(
            name="Test Lore",
            category="myth",
            description="A legend",
            significance=5,
            tags=["legend", "ancient"],
        )

    lore_id = asyncio.run(run_store())

    assert lore_id == 7
    assert conn.calls == 1
    assert conn.query is not None and "::jsonb" in conn.query
    assert conn.args[6] == json.dumps(["legend", "ancient"])
    assert conn.args[:6] == (
        99,
        123,
        "Test Lore",
        "myth",
        "A legend",
        5,
    )
