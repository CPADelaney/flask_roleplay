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


@pytest.mark.asyncio
async def test_get_aggregated_roleplay_context_exposes_both_spellings(monkeypatch):
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

    context = await aggregator_sdk.get_aggregated_roleplay_context(1, 2, "Chase")

    assert "currentRoleplay" in context
    assert "current_roleplay" in context
    assert context["currentRoleplay"] is context["current_roleplay"]
    assert context["currentRoleplay"]["CurrentLocation"] == "Chapel of Thorns"
