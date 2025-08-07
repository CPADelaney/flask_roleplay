import os
import sys
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from logic.roleplay_engine import RoleplayEngine


def test_roleplay_engine_generation(monkeypatch):
    async def fake_context(user_id, conversation_id, player_name):
        return {"location": "town"}

    async def fake_call_gpt_json(conversation_id, context, prompt, model="gpt-5-nano", temperature=0.7, max_retries=2):
        assert context["location"] == "town"
        return {"narrative": "A brave hero arrives.", "updates": {"roleplay_updates": []}}

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
