import asyncio
import sys
from pathlib import Path
from typing import Any, List, Optional
from types import ModuleType, SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parent.parent))


class _StubConnection:
    def __init__(self) -> None:
        self.executed: List[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, query: str, *args: Any) -> Optional[dict[str, Any]]:
        if "CurrentRoleplay" in query and "'CurrentLocation'" in query:
            return None
        if "CurrentRoleplay" in query and "'CurrentTime'" in query:
            return None
        if "PlayerStats" in query:
            return None
        return None

    async def fetch(self, query: str, *args: Any) -> List[dict[str, Any]]:
        return []

    async def execute(self, query: str, *args: Any) -> None:
        self.executed.append((query, args))


class _StubConnectionContext:
    def __init__(self, connection: _StubConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _StubConnection:
        return self._connection

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def test_initialize_game_state_backfills_time_and_location(monkeypatch):
    async def _run_test() -> None:
        expected_time = SimpleNamespace(year=123, month=4, day=5, time_of_day="Evening")
        recorded_updates: List[tuple[int, int, str, str]] = []

        async def fake_get_comprehensive_context(*_args, **_kwargs):
            return {
                "current_location": {"name": " Velvet Sanctum ", "scene_id": "velvet-sanctum"},
                "current_roleplay": {},
            }

        async def fake_get_current_time_model(user_id: int, conversation_id: int):
            assert (user_id, conversation_id) == (42, 9001)
            return expected_time

        async def fake_update_current_roleplay(ctx, conn, key: str, value: str) -> None:  # noqa: ANN001
            recorded_updates.append((ctx.user_id, ctx.conversation_id, key, value))

        class _StubNyxContext:
            @staticmethod
            def _normalize_location_value(value: Any) -> Optional[str]:
                if value is None:
                    return None
                if isinstance(value, dict):
                    token = value.get("name") or value.get("location") or value.get("location_name")
                    if isinstance(token, str):
                        stripped = token.strip()
                        return stripped or None
                    if isinstance(token, (int, float)):
                        stripped = str(token).strip()
                        return stripped or None
                if isinstance(value, str):
                    stripped = value.strip()
                    return stripped or None
                return None

        stub_context_service = ModuleType("context.context_service")
        stub_context_service.get_comprehensive_context = fake_get_comprehensive_context

        stub_time_cycle = ModuleType("logic.time_cycle")
        stub_time_cycle.get_current_time_model = fake_get_current_time_model

        stub_canon = ModuleType("lore.core.canon")
        stub_canon.update_current_roleplay = fake_update_current_roleplay

        stub_lore_core = ModuleType("lore.core")
        stub_lore_core.canon = stub_canon

        stub_lore = ModuleType("lore")
        stub_lore.core = stub_lore_core

        stub_nyx_context = ModuleType("nyx.nyx_agent.context")
        stub_nyx_context.NyxContext = _StubNyxContext

        stub_nyx_agent = ModuleType("nyx.nyx_agent")
        stub_nyx_agent.context = stub_nyx_context

        monkeypatch.setitem(sys.modules, "context.context_service", stub_context_service)
        monkeypatch.setitem(sys.modules, "logic.time_cycle", stub_time_cycle)
        monkeypatch.setitem(sys.modules, "lore", stub_lore)
        monkeypatch.setitem(sys.modules, "lore.core", stub_lore_core)
        monkeypatch.setitem(sys.modules, "lore.core.canon", stub_canon)
        monkeypatch.setitem(sys.modules, "nyx.nyx_agent", stub_nyx_agent)
        monkeypatch.setitem(sys.modules, "nyx.nyx_agent.context", stub_nyx_context)

        from nyx.governance.core import NyxUnifiedGovernor

        stub_connection = _StubConnection()

        monkeypatch.setattr(
            "nyx.governance.core.get_db_connection_context",
            lambda: _StubConnectionContext(stub_connection),
        )

        governor = NyxUnifiedGovernor(user_id=42, conversation_id=9001)

        result = await governor.initialize_game_state(force=True, player_name="TestPlayer")

        assert result["current_location"] == "Velvet Sanctum"
        assert result["current_time"] == "Year 123 4 5 Evening"
        assert (42, 9001, "CurrentTime", "Year 123 4 5 Evening") in recorded_updates

    asyncio.run(_run_test())
