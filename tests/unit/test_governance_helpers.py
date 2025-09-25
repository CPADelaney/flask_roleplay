import pytest

from nyx.governance_helpers import with_governance


class DummyContext:
    def __init__(self, user_id=1, conversation_id=2):
        self.context = {"user_id": user_id, "conversation_id": conversation_id}


@pytest.mark.asyncio
async def test_with_governance_formats_positional_arguments(monkeypatch):
    captured = {}

    async def fake_check_permission(user_id, conversation_id, agent_type, agent_id, action_type, action_details):
        captured["details"] = action_details
        return {
            "approved": True,
            "tracking_id": 101,
            "directive_applied": False,
            "override_action": None,
        }

    async def fake_report_action(user_id, conversation_id, agent_type, agent_id, action, result):
        captured["action"] = action
        return {"reported": True, "report_id": "mock"}

    monkeypatch.setattr("nyx.governance_helpers.check_permission", fake_check_permission)
    monkeypatch.setattr("nyx.governance_helpers.report_action", fake_report_action)

    class DummyAgent:
        agent_id = "npc_agent"

        @with_governance(
            agent_type="npc",
            action_type="speak",
            action_description="NPC {npc_id} says {line}",
        )
        async def act(self, ctx, npc_id, line):
            return {"success": True}

    agent = DummyAgent()
    ctx = DummyContext()

    result = await agent.act(ctx, 42, line="hello there")

    assert captured["details"]["description"] == "NPC 42 says hello there"
    assert captured["action"]["description"] == "NPC 42 says hello there"
    assert result["governance_metadata"]["permission_tracking_id"] == 101
