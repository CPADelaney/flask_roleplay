import importlib
import sys
import types
import pytest

sys.modules['nyx.core.memory.memory_manager'] = types.SimpleNamespace(
    MemoryManager=types.SimpleNamespace(fetch_relevant=lambda *a, **k: [])
)
orchestrator = importlib.import_module('nyx.core.orchestrator')
OrchestratorRunner = importlib.import_module('nyx.core.agents.base_runner').OrchestratorRunner
EventLogger = orchestrator.EventLogger
RewardEvaluator = orchestrator.RewardEvaluator
from agents import Agent
from agents.models.interface import Model, ModelResponse
from agents.usage import Usage
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText

class DummyModel(Model):
    async def get_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs, tracing, *, previous_response_id=None):
        msg = ResponseOutputMessage(
            id="1",
            content=[ResponseOutputText(annotations=[], text="42", type="output_text")],
            role="assistant",
            status="completed",
            type="message",
        )
        return ModelResponse(output=[msg], usage=Usage(), response_id=None)

    def stream_response(self, *args, **kwargs):  # pragma: no cover - Runner.run doesn't call
        async def gen():
            if False:
                yield None
        return gen()


@pytest.mark.asyncio
async def test_orchestrator_integration(monkeypatch):
    logged = []
    monkeypatch.setattr(EventLogger, "LOG_DIR", "/tmp")
    monkeypatch.setattr(EventLogger, "_append_line", lambda path, line: logged.append(line))

    scored = []
    def fake_eval(evt):
        scored.append(evt)
        return 1.0
    monkeypatch.setattr(RewardEvaluator, "evaluate", fake_eval)

    scheduled = []
    import reflection.reflection_agent as ra
    async def fake_schedule():
        scheduled.append(True)
    monkeypatch.setattr(ra, "schedule_reflection", fake_schedule)

    agent = Agent(name="dummy", instructions="You answer 42", model=DummyModel())
    result = await OrchestratorRunner.run(agent, "hi")
    assert result.final_output == "42"

    score = await OrchestratorRunner().log_and_score("unit_test_passed")
    assert score == 1.0
    import json
    assert logged and json.loads(logged[0])["type"] == "unit_test_passed"
    assert scored == ["unit_test_passed"]
    assert scheduled

    del sys.modules['nyx.core.memory.memory_manager']
    sys.modules.pop('agents', None)
    sys.modules.pop('reflection.reflection_agent', None)

