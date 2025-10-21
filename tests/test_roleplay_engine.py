import os
import sys
import asyncio
import types

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

if "sentence_transformers" not in sys.modules:
    stub_sentence_transformers = types.ModuleType("sentence_transformers")

    class _StubSentenceTransformer:  # pragma: no cover - lightweight compatibility layer
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, *args, **kwargs):
            return []

    stub_sentence_transformers.SentenceTransformer = _StubSentenceTransformer
    stub_sentence_transformers.util = types.SimpleNamespace(cos_sim=lambda *_, **__: 0)
    sys.modules["sentence_transformers"] = stub_sentence_transformers

from logic.roleplay_engine import RoleplayEngine


def test_roleplay_engine_generation(monkeypatch):
    aggregator_text = "Scene summary"

    async def fake_context(user_id, conversation_id, player_name):
        return {"location": "town", "aggregatorText": aggregator_text}

    captured = {}

    async def fake_call_gpt_json(
        conversation_id,
        context,
        prompt,
        model="gpt-5-nano",
        temperature=0.7,
        max_retries=2,
    ):
        captured["context"] = context
        return {
            "narrative": "A brave hero arrives.",
            "updates": {"roleplay_updates": []},
        }

    async def fake_apply(self, user_id, conversation_id, updates):
        self.applied = updates
        return {"success": True}

    import types, sys as _sys

    _sys.modules["logic.gpt_utils"] = types.SimpleNamespace(
        call_gpt_json=fake_call_gpt_json
    )

    monkeypatch.setattr("logic.roleplay_engine.get_aggregated_roleplay_context", fake_context)
    monkeypatch.setattr(RoleplayEngine, "apply_updates", fake_apply)

    async def run():
        engine = RoleplayEngine()
        result = await engine.generate_turn(1, 1, "Player", "Hello")
        assert result["narrative"] == "A brave hero arrives."
        assert engine.applied == {"roleplay_updates": []}

    asyncio.run(run())

    assert captured["context"] == aggregator_text


def test_roleplay_engine_generation_without_aggregator_text(monkeypatch):
    async def fake_context(user_id, conversation_id, player_name):
        return {"location": "forest"}

    captured = {}

    async def fake_call_gpt_json(
        conversation_id,
        context,
        prompt,
        model="gpt-5-nano",
        temperature=0.7,
        max_retries=2,
    ):
        captured["context"] = context
        return {
            "narrative": "The forest whispers.",
            "updates": {},
        }

    async def fake_apply(self, user_id, conversation_id, updates):
        self.applied = updates
        return {"success": True}

    import types, sys as _sys

    _sys.modules["logic.gpt_utils"] = types.SimpleNamespace(
        call_gpt_json=fake_call_gpt_json
    )

    monkeypatch.setattr("logic.roleplay_engine.get_aggregated_roleplay_context", fake_context)
    monkeypatch.setattr(RoleplayEngine, "apply_updates", fake_apply)

    async def run():
        engine = RoleplayEngine()
        result = await engine.generate_turn(1, 2, "Player", "Listen")
        assert result["narrative"] == "The forest whispers."
        assert result["updates"] == {}

    asyncio.run(run())

    assert isinstance(captured["context"], str)
    assert "forest" in captured["context"]
