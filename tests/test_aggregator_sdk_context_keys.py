import asyncio
import os
import sys
import json
import types

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

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

from logic import aggregator_sdk


class DummyConnection:
    async def fetch(self, query, *args):
        query_str = " ".join(query.split()) if isinstance(query, str) else str(query)

        if "FROM CurrentRoleplay" in query_str and "NPCStats" not in query_str:
            return [
                {"key": "CurrentLocation", "value": json.dumps("Chapel of Thorns")},
                {"key": "TimeOfDay", "value": json.dumps("Dawn")},
                {"key": "CurrentDay", "value": json.dumps(2)},
                {"key": "CurrentYear", "value": json.dumps(2025)},
                {"key": "CurrentMonth", "value": json.dumps("May")},
            ]

        if "FROM NPCStats" in query_str:
            return []

        if "FROM Events" in query_str:
            return []

        if "FROM Quests" in query_str:
            return []

        return []

    async def fetchrow(self, query, *args):
        query_str = " ".join(query.split()) if isinstance(query, str) else str(query)

        if "FROM PlayerStats" in query_str:
            return {
                "corruption": 1,
                "confidence": 2,
                "willpower": 3,
                "obedience": 4,
                "dependency": 5,
                "lust": 6,
                "mental_resilience": 7,
                "physical_endurance": 8,
            }

        return None


class DummyConnectionContext:
    async def __aenter__(self):
        return DummyConnection()

    async def __aexit__(self, exc_type, exc, tb):
        return False


def dummy_get_db_connection_context():
    return DummyConnectionContext()


def test_get_aggregated_roleplay_context_exposes_both_spellings(monkeypatch):
    monkeypatch.setattr(
        aggregator_sdk,
        "get_db_connection_context",
        dummy_get_db_connection_context,
    )

    async def fake_get_latest_conversation(*, conversation_id, user_id, conn):
        return None

    async def fake_get_active_scene(*, conversation_id, conn):
        return None

    monkeypatch.setattr(
        aggregator_sdk,
        "get_latest_openai_conversation",
        fake_get_latest_conversation,
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
    monkeypatch.setattr(
        aggregator_sdk,
        "get_db_connection_context",
        dummy_get_db_connection_context,
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

    async def fake_get_latest_conversation(*, conversation_id, user_id, conn):
        return conversation_row

    async def fake_get_active_scene(*, conversation_id, conn):
        return active_scene

    monkeypatch.setattr(
        aggregator_sdk,
        "get_latest_openai_conversation",
        fake_get_latest_conversation,
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
    assert openai_payload["thread_id"] == "thread_abc"
    assert openai_payload["run_id"] == "run_xyz"
    assert openai_payload["response_id"] == "resp_456"

    scene_rotation = openai_payload.get("scene_rotation")
    assert scene_rotation["new_scene"] == metadata["queued_scene"]
    assert scene_rotation["closing_scene"] == metadata["queued_scene_closing"]
    assert scene_rotation["active_scene"] == active_scene
