import asyncio
import os
import sys
import time
import types
import typing

import typing_extensions

import pytest


_dummy_models = types.ModuleType("sentence_transformers.models")
_dummy_models.Transformer = lambda *args, **kwargs: None
_dummy_models.Pooling = lambda *args, **kwargs: None


class _DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):  # pragma: no cover - defensive stub
        return [[0.0] * 3 for _ in texts]

    def get_sentence_embedding_dimension(self):  # pragma: no cover - defensive stub
        return 3


_dummy_sentence_transformers = types.ModuleType("sentence_transformers")
_dummy_sentence_transformers.SentenceTransformer = _DummySentenceTransformer
_dummy_sentence_transformers.models = _dummy_models

sys.modules["sentence_transformers"] = _dummy_sentence_transformers
sys.modules["sentence_transformers.models"] = _dummy_models

os.environ.setdefault("OPENAI_API_KEY", "test-key")

typing.TypedDict = typing_extensions.TypedDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nyx.nyx_agent import orchestrator as orchestrator_module


class _StubPackedContext:
    def __init__(self, context: dict[str, typing.Any]):
        self.canonical = {
            "recent_interactions": context.get("recent_interactions", []),
            "recent_turns": context.get("recent_turns", []),
        }


class _StubNyxContext:
    instances: list["_StubNyxContext"] = []

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.seed_context = {
            "present_npcs": [
                {"name": "Vera", "role": "handler"},
                {"name": "Cass", "role": "guard"},
                {"name": "Jules", "role": "bartender"},
            ],
            "recent_interactions": [
                {"sender": "Nyx", "content": "The club hums with bass."},
                {"sender": "You", "content": "I head upstairs."},
            ],
            "current_location": {
                "name": "Club Atrium",
                "id": "loc-1",
                "city": "New Avalon",
                "lat": 37.7749,
                "lon": -122.4194,
            },
            "location_id": "loc-1",
            "location_name": "Club Atrium",
        }
        self.current_context = dict(self.seed_context)
        self.current_location = self.seed_context["current_location"]
        self.current_world_state = None
        self.last_user_input = None
        self.last_packed_context = None
        self.refreshed = False
        self.previous_location_id = None
        _StubNyxContext.instances.append(self)

    async def initialize(self) -> None:
        return None

    async def build_context_for_input(self, user_input: str, base_context: dict[str, typing.Any]):
        merged = dict(self.seed_context)
        merged.update(base_context)
        self.current_context = merged
        self.last_user_input = user_input
        self.last_packed_context = _StubPackedContext(merged)
        return self.last_packed_context

    async def await_orchestrator(self, _name: str) -> bool:
        return False

    async def _refresh_location_from_context(self, previous_location_id=None) -> None:
        self.refreshed = True
        self.previous_location_id = previous_location_id


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_movement_fast_path_uses_router(monkeypatch):
    _StubNyxContext.instances.clear()

    monkeypatch.setattr(orchestrator_module, "NyxContext", _StubNyxContext)

    fast_payload = {
        "overall": {"feasible": True, "strategy": "allow"},
        "per_intent": [
            {"categories": ["movement"], "feasible": True},
            {"categories": ["mundane_action"], "feasible": True},
        ],
        "router_result": {
            "intents": [
                {"categories": ["movement"], "raw_text": "walk"},
                {"categories": ["mundane_action"], "raw_text": "open door"},
            ]
        },
    }

    async def fake_fast(*_args, **_kwargs):
        return fast_payload

    monkeypatch.setattr(
        "nyx.nyx_agent.feasibility.assess_action_feasibility_fast",
        fake_fast,
    )

    assess_calls = 0

    async def fake_assess(*_args, **_kwargs):
        nonlocal assess_calls
        assess_calls += 1
        assert _kwargs.get("router_result") == fast_payload["router_result"]
        return {"overall": {"feasible": True}}

    monkeypatch.setattr(
        "nyx.nyx_agent.feasibility.assess_action_feasibility",
        fake_assess,
    )

    router_calls: list[dict[str, typing.Any]] = []

    async def fake_resolve_place_or_travel(user_text, meta, store, user_key, convo_key):
        router_calls.append(
            {
                "user_text": user_text,
                "meta": meta,
                "store_type": type(store).__name__ if store else None,
                "user_key": user_key,
                "convo_key": convo_key,
            }
        )

        location = types.SimpleNamespace(
            id="loc-2",
            location_id="loc-2",
            location_name="Velvet Staircase",
            location_type="club",
            city="New Avalon",
            region="Central",
            country="Avalon",
            description="Velvet banisters and low amber light.",
            lat=37.7752,
            lon=-122.4189,
        )

        return types.SimpleNamespace(
            status="exact",
            choices=["Velvet Staircase"],
            operations=[{"op": "poi.navigate"}],
            metadata={"router": {"source": "stub"}},
            location=location,
            candidates=[],
        )

    monkeypatch.setattr(
        "nyx.location.router.resolve_place_or_travel",
        fake_resolve_place_or_travel,
    )

    movement_output = {"final_output": "She guides you up the velvet staircase without slowing."}

    async def fake_run_agent(agent, prompt, context, run_config=None, fallback_response=None):
        fake_run_agent.last_call = {
            "agent": agent,
            "prompt": prompt,
            "context": context,
            "run_config": run_config,
        }
        return movement_output

    monkeypatch.setattr(orchestrator_module, "run_agent_safely", fake_run_agent)

    base_context = {
        "present_npcs": [
            {"name": "Vera", "role": "handler"},
            {"name": "Cass", "role": "guard"},
        ],
        "recent_interactions": [
            {"sender": "Nyx", "content": "The club hums with bass."},
            {"sender": "You", "content": "I head upstairs."},
        ],
        "current_location": {
            "name": "Club Atrium",
            "id": "loc-1",
            "city": "New Avalon",
            "lat": 37.7749,
            "lon": -122.4194,
        },
        "location_id": "loc-1",
        "location_name": "Club Atrium",
        "feasibility": fast_payload,
        "router_result": fast_payload["router_result"],
    }

    result = await orchestrator_module.process_user_input(
        user_id=10,
        conversation_id=99,
        user_input="I stride up toward the private lounge.",
        context_data=base_context,
    )

    assert router_calls, "movement fast path should resolve the location"
    assert assess_calls == 0, "full feasibility should be skipped on movement fast path"

    assert result["metadata"]["movement_fast_path"] is True
    assert result["metadata"]["location_transition"]["router_called"] is True
    assert result["metadata"]["location_transition"]["location"]["name"] == "Velvet Staircase"
    assert result["response"] == movement_output["final_output"]
    prompt_text = fake_run_agent.last_call["prompt"]
    assert "Destination:" in prompt_text
    assert "- name: Velvet Staircase" in prompt_text
    assert "Approximate distance:" in prompt_text
    run_config = fake_run_agent.last_call["run_config"]
    expected_limits = orchestrator_module._MOVEMENT_RUN_LIMITS
    assert getattr(run_config, "max_turns", None) == expected_limits["max_turns"]
    assert getattr(run_config, "max_tool_calls", None) == expected_limits["max_tool_calls"]
    assert getattr(run_config, "max_auto_messages", None) == expected_limits["max_auto_messages"]
    assert getattr(run_config, "max_output_tokens", None) == expected_limits["max_output_tokens"]
    assert getattr(run_config, "per_turn_timeout", None) == pytest.approx(expected_limits["per_turn_timeout"])
    assert getattr(run_config, "per_step_timeout", None) == pytest.approx(expected_limits["per_step_timeout"])
    assert result["metadata"]["movement_run_limits"]["max_turns"] == expected_limits["max_turns"]

    movement_meta = result["metadata"]["movement_meta"]
    assert movement_meta["origin"]["name"] == "Club Atrium"
    assert movement_meta["destination"]["name"] == "Velvet Staircase"
    assert movement_meta["distance_class"] == "short"
    assert movement_meta["approx_distance_km"] is not None

    stub_ctx = _StubNyxContext.instances[0]
    assert stub_ctx.refreshed is True
    assert stub_ctx.previous_location_id == "loc-1"


@pytest.mark.anyio
async def test_movement_fast_path_skips_long_distance(monkeypatch):
    ctx = _StubNyxContext(user_id=42, conversation_id=100)
    packed = await ctx.build_context_for_input("I head south", dict(ctx.seed_context))

    fast_payload = {
        "overall": {"feasible": True, "strategy": "allow"},
        "per_intent": [
            {"categories": ["movement"], "feasible": True},
        ],
    }

    long_location = types.SimpleNamespace(
        id="loc-remote",
        location_id="loc-remote",
        location_name="Distant Haven",
        city="Anaheim",
        region="California",
        country="USA",
        lat=33.8121,
        lon=-117.9190,
    )

    async def fake_resolve(*_args, **_kwargs):
        return types.SimpleNamespace(
            status="exact",
            choices=["Distant Haven"],
            operations=[],
            metadata={},
            location=long_location,
            candidates=[],
        )

    monkeypatch.setattr(
        "nyx.location.router.resolve_place_or_travel",
        fake_resolve,
    )

    run_calls: list[typing.Any] = []

    async def fake_run_agent(*args, **kwargs):  # pragma: no cover - should not run
        run_calls.append((args, kwargs))
        return {"final_output": "ignored"}

    monkeypatch.setattr(orchestrator_module, "run_agent_safely", fake_run_agent)

    result = await orchestrator_module._run_movement_transition_fast_path(
        ctx,
        packed,
        "I go to Disneyland",
        fast_payload,
        trace_id="unit",
        start_time=time.time(),
    )

    assert result is None
    assert not run_calls
    assert ctx.refreshed is False
    assert ctx.current_context["current_location"]["name"] == "Club Atrium"
    assert ctx.current_context["location_id"] == "loc-1"
