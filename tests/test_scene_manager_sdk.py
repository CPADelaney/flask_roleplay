import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parent.parent))

from nyx.scene_manager_sdk import get_location_description


class _StubConnection:
    def __init__(self, *, user_id, conversation_id, location_name, description):
        self._user_id = user_id
        self._conversation_id = conversation_id
        self._location_name = location_name
        self._description = description

    async def fetchrow(self, query, *args):
        if "CurrentRoleplay" in query:
            assert args == (self._user_id, self._conversation_id)
            if "'CurrentLocation'" in query:
                return {"value": self._location_name}
            return None

        if "FROM Locations" in query:
            assert args == (self._location_name, self._user_id)
            return {"description": self._description}

        return None


class _StubConnectionContext:
    def __init__(self, connection):
        self._connection = connection

    async def __aenter__(self):
        return self._connection

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


def test_get_location_description_returns_current_location(monkeypatch):
    user_id = 99
    conversation_id = "conv-123"
    expected_description = "A bustling atrium filled with chatter."

    stub_connection = _StubConnection(
        user_id=user_id,
        conversation_id=conversation_id,
        location_name="Atrium",
        description=expected_description,
    )

    def _stub_get_db_connection_context():
        return _StubConnectionContext(stub_connection)

    monkeypatch.setattr(
        "nyx.scene_manager_sdk.get_db_connection_context",
        _stub_get_db_connection_context,
    )

    ctx = SimpleNamespace(
        context=SimpleNamespace(user_id=user_id, conversation_id=conversation_id)
    )

    tool_impl = get_location_description.on_invoke_tool.__closure__[0].cell_contents
    orig_func = tool_impl.__closure__[1].cell_contents

    result = asyncio.run(orig_func(ctx))

    assert result == expected_description
