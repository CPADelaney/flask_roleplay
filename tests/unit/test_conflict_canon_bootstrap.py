import sys
import typing
from pathlib import Path

import pytest
from typing_extensions import TypedDict as _CompatTypedDict

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Ensure compatibility with pydantic's TypedDict expectations under Python < 3.12.
typing.TypedDict = _CompatTypedDict

from logic.conflict_system import conflict_canon


class _FakeConnection:
    """Minimal async connection stub for canon bootstrap tests."""

    def __init__(self):
        self.insert_calls = 0

    async def fetch(self, query, *args):
        # Canon bootstrap queries parents/related entities; return nothing.
        return []

    async def fetchval(self, query, *args):
        query = query.strip().lower()
        if "count(" in query:
            # Force the bootstrap path by reporting zero existing events.
            return 0
        if query.startswith("insert into canonicalevents"):
            self.insert_calls += 1
            # Simulate returning a primary key.
            return self.insert_calls
        return 0

    async def execute(self, query, *args):
        # Consequence propagation issues follow-up inserts; no-op.
        return None


class _FakeDBContext:
    def __init__(self, registry):
        self._registry = registry
        self._conn = _FakeConnection()
        self._registry.append(self._conn)

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_conflict_bootstrap_skips_memory_persistence(monkeypatch):
    connections = []

    def fake_get_db_connection_context():
        return _FakeDBContext(connections)

    monkeypatch.setattr(
        conflict_canon, "get_db_connection_context", fake_get_db_connection_context
    )

    # Import inside to avoid circular references during test discovery.
    import lore.core.canon as canon_module

    async def _fail_if_called(*args, **kwargs):
        raise AssertionError("Memory orchestrator should not be used during bootstrap")

    monkeypatch.setattr(
        canon_module, "get_canon_memory_orchestrator", _fail_if_called
    )

    subsystem = conflict_canon.ConflictCanonSubsystem(user_id=1, conversation_id=2)

    # Initialize with a simple synthesizer stub; only weak references are stored.
    class _SynthStub:
        pass

    synthesizer_stub = _SynthStub()

    result = await subsystem.initialize(synthesizer_stub)

    assert result is True
    # Ensure bootstrap inserted at least one canonical event, proving the code path ran.
    assert any(conn.insert_calls for conn in connections)
