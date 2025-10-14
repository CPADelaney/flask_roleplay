import asyncio
import json
import os
import sys
import types

import pytest


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DB_DSN", "postgresql://user:pass@localhost/testdb")

dummy_models = types.ModuleType("sentence_transformers.models")


class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def encode(self, sentences, *args, **kwargs):
        if isinstance(sentences, list):
            return [[0.0] * 3 for _ in sentences]
        return [0.0, 0.0, 0.0]

    def get_sentence_embedding_dimension(self):
        return 3


dummy_sentence_transformers = types.ModuleType("sentence_transformers")
dummy_sentence_transformers.SentenceTransformer = DummySentenceTransformer
dummy_sentence_transformers.models = dummy_models

sys.modules.setdefault("sentence_transformers", dummy_sentence_transformers)
sys.modules.setdefault("sentence_transformers.models", dummy_models)

from context.projection_helpers import parse_scene_projection_row
from logic import aggregator_sdk


def _build_scene_row():
    return {
        "user_id": 1,
        "conversation_id": 2,
        "scene_context": {
            "current_roleplay": {
                "CurrentLocation": json.dumps("Chapel of Thorns"),
                "TimeOfDay": json.dumps("Dawn"),
                "CurrentDay": json.dumps(2),
                "CurrentYear": json.dumps(2025),
                "CurrentMonth": json.dumps("May"),
            },
            "player_stats": {
                "corruption": 1,
                "confidence": 2,
                "willpower": 3,
                "obedience": 4,
                "dependency": 5,
                "lust": 6,
                "mental_resilience": 7,
                "physical_endurance": 8,
            },
            "npcs_present": [],
            "events": [],
            "quests": [],
        },
    }


def test_get_aggregated_roleplay_context_exposes_both_spellings(monkeypatch):
    async def fake_read_scene_context(user_id, conversation_id, limit=None):
        assert user_id == 1
        assert conversation_id == 2
        return [_build_scene_row()]

    monkeypatch.setattr(
        aggregator_sdk,
        "read_scene_context",
        fake_read_scene_context,
    )

    async def fake_get_latest_conversation(*, conversation_id, user_id):
        return None

    async def fake_get_active_scene(*, conversation_id):
        return None

    async def fake_get_latest_chatkit_thread(*args, **kwargs):
        return None

    monkeypatch.setattr(
        aggregator_sdk,
        "get_latest_openai_conversation",
        fake_get_latest_conversation,
    )
    monkeypatch.setattr(
        aggregator_sdk,
        "get_latest_chatkit_thread",
        fake_get_latest_chatkit_thread,
    )
    monkeypatch.setattr(
        aggregator_sdk,
        "get_openai_active_scene",
        fake_get_active_scene,
    )

    async def fake_check_preset_story(conversation_id):
        return None

    monkeypatch.setattr(
        aggregator_sdk.PresetStoryManager,
        "check_preset_story",
        staticmethod(fake_check_preset_story),
    )

    from openai_integration import conversations as openai_conversations

    monkeypatch.setattr(
        openai_conversations,
        "get_latest_conversation",
        fake_get_latest_conversation,
    )
    monkeypatch.setattr(
        openai_conversations,
        "get_latest_chatkit_thread",
        fake_get_latest_chatkit_thread,
    )
    monkeypatch.setattr(
        openai_conversations,
        "get_active_scene",
        fake_get_active_scene,
    )

    context = asyncio.run(
        aggregator_sdk.get_aggregated_roleplay_context(1, 2, "Chase")
    )

    assert "currentRoleplay" in context
    assert "current_roleplay" in context
    assert context["currentRoleplay"] is context["current_roleplay"]
    assert context["currentLocation"] == context["current_location"]
    assert context["current_location"] == context["location"]
    assert context["currentRoleplay"]["CurrentLocation"] == "Chapel of Thorns"


def test_get_aggregated_roleplay_context_includes_openai_metadata(monkeypatch):
    async def fake_read_scene_context(user_id, conversation_id, limit=None):
        return [_build_scene_row()]

    monkeypatch.setattr(
        aggregator_sdk,
        "read_scene_context",
        fake_read_scene_context,
    )

    async def fake_check_preset_story(conversation_id):
        return None

    monkeypatch.setattr(
        aggregator_sdk.PresetStoryManager,
        "check_preset_story",
        staticmethod(fake_check_preset_story),
    )

    metadata = {
        "queued_scene": {"scene_title": "Dramatic Entrance"},
        "queued_scene_closing": {"scene_summary": "Fade to black"},
    }

    conversation_row = {
        "id": 42,
        "user_id": 1,
        "conversation_id": 2,
        "openai_assistant_id": "asst_123",
        "openai_thread_id": "thread_abc",
        "openai_run_id": "run_xyz",
        "openai_response_id": "resp_456",
        "status": "pending",
        "last_error": None,
        "metadata": metadata,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }

    active_scene = {"scene_number": 7, "scene_title": "Current Scene"}

    async def fake_get_latest_conversation(*, conversation_id, user_id):
        return conversation_row

    async def fake_get_active_scene(*, conversation_id):
        return active_scene

    monkeypatch.setattr(
        aggregator_sdk,
        "get_latest_openai_conversation",
        fake_get_latest_conversation,
    )
    async def fake_get_latest_chatkit_thread(*args, **kwargs):
        return None

    monkeypatch.setattr(
        aggregator_sdk,
        "get_latest_chatkit_thread",
        fake_get_latest_chatkit_thread,
    )
    monkeypatch.setattr(
        aggregator_sdk,
        "get_openai_active_scene",
        fake_get_active_scene,
    )

    context = asyncio.run(
        aggregator_sdk.get_aggregated_roleplay_context(1, 2, "Chase")
    )

    openai_payload = context.get("openai_integration")
    assert openai_payload is not None
    assert openai_payload["conversation"] == conversation_row


def test_parse_scene_projection_row_handles_string_scene_context():
    scene_row = _build_scene_row()
    scene_row["scene_context"] = json.dumps(scene_row["scene_context"])

    projection = parse_scene_projection_row(scene_row)

    assert projection.current_location() == "Chapel of Thorns"
    assert projection.roleplay_dict()["CurrentLocation"] == "Chapel of Thorns"
