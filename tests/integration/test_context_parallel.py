import asyncio
from typing import Any

from context.context_service import (
    BaseContextData,
    ContextService,
    LocationDetails,
    NPCData,
    PlayerStats,
    RoleplayData,
    TimeInfo,
)
from context.unified_cache import context_cache


def test_initialize_parallel(monkeypatch):
    monkeypatch.setenv("NYX_FLAG_CONTEXT_PARALLEL_INIT", "on")

    call_counts = {"config": 0, "memory": 0}

    async def fake_get_config() -> Any:
        call_counts["config"] += 1
        await asyncio.sleep(0.01)

        class _Config:
            def is_enabled(self, name: str) -> bool:  # pragma: no cover - test stub
                return False

        return _Config()

    async def fake_get_memory_manager(user_id: int, conversation_id: int) -> Any:
        call_counts["memory"] += 1
        await asyncio.sleep(0.01)
        return object()

    async def fake_initialize_agents(self: ContextService) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr("context.context_service.get_config", fake_get_config)
    monkeypatch.setattr("context.context_service.get_memory_manager", fake_get_memory_manager)
    monkeypatch.setattr("context.context_service.get_context_manager", lambda: type("Mgr", (), {"version": 1})())
    monkeypatch.setattr("context.context_service.get_vector_service", lambda *args, **kwargs: asyncio.sleep(0))
    monkeypatch.setattr(ContextService, "_initialize_agents", fake_initialize_agents)

    async def runner():
        service = ContextService(user_id=1, conversation_id=2)
        await asyncio.gather(service.initialize(), service.initialize())
        return service

    service = asyncio.run(runner())

    assert call_counts["config"] == 1
    assert call_counts["memory"] == 1
    assert service.initialized


def test_get_context_parallel(monkeypatch):
    monkeypatch.setenv("NYX_FLAG_CONTEXT_PARALLEL_FETCH", "on")

    service = ContextService(user_id=5, conversation_id=9)
    service.initialized = True
    service.context_manager = type("Mgr", (), {"version": 7})()

    base_context = BaseContextData(
        time_info=TimeInfo(year="2024", month="1", day="2", time_of_day="Morning"),
        player_stats=PlayerStats(),
        current_roleplay=RoleplayData(CurrentLocation="Library"),
        current_location="Library",
    )

    async def fake_base(location: str | None = None) -> BaseContextData:
        await asyncio.sleep(0.01)
        return base_context

    async def fake_npcs(**kwargs: Any):
        await asyncio.sleep(0.01)
        return [
            NPCData(
                npc_id="npc-1",
                npc_name="Test NPC",
                trust=0.5,
                respect=0.5,
            )
        ]

    async def fake_location(location: str | None = None):
        await asyncio.sleep(0.01)
        return LocationDetails(location_name=location or "Library", description="A quiet place")

    async def fake_quests():
        await asyncio.sleep(0.01)
        from context.models import QuestData

        return [
            QuestData(
                quest_id="1",
                quest_name="Find the Tome",
                status="active",
            )
        ]

    monkeypatch.setattr(service, "_get_base_context", fake_base)
    monkeypatch.setattr(service, "_get_relevant_npcs", fake_npcs)
    monkeypatch.setattr(service, "_get_location_details", fake_location)
    monkeypatch.setattr(service, "_get_quest_information", fake_quests)

    async def runner():
        context_cache.l1_cache.clear()
        context_cache.l2_cache.clear()
        context_cache.l3_cache.clear()

        context = await service.get_context(
            input_text="search for clues",
            include_npcs=True,
            include_location=True,
            include_quests=True,
        )
        return context

    context = asyncio.run(runner())

    assert context["npcs"][0]["npc_name"] == "Test NPC"
    assert context["location_details"]["location_name"] == "Library"
    assert context["quests"][0]["quest_name"] == "Find the Tome"
    assert context_cache.l1_cache, "context result should be cached"
