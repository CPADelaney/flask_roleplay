import os
import sys
import os
import sys
import types
from dataclasses import dataclass

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

nyx_pkg = types.ModuleType("nyx")
core_pkg = types.ModuleType("nyx.core")
orchestrator_mod = types.ModuleType("nyx.core.orchestrator")
orchestrator_mod.prepare_context = lambda system_prompt, user_prompt: system_prompt
core_pkg.orchestrator = orchestrator_mod
nyx_pkg.core = core_pkg
sys.modules.setdefault("nyx", nyx_pkg)
sys.modules.setdefault("nyx.core", core_pkg)
sys.modules["nyx.core.orchestrator"] = orchestrator_mod

from logic.chatgpt_integration import sanitize_model_settings


class TemperatureRejectingClient:
    """Simulate a Responses API client that rejects temperature for gpt-5-nano."""

    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        if kwargs.get("temperature") is not None:
            raise ValueError("temperature should be omitted for gpt-5-nano")
        self.last_kwargs = kwargs
        return {"id": "resp-test"}


@dataclass
class DummySettings:
    temperature: float | None = None
    top_p: float | None = None


def test_sanitize_model_settings_removes_temperature_for_nano():
    raw_settings = DummySettings(temperature=0.6)
    client = TemperatureRejectingClient()

    with pytest.raises(ValueError):
        client.create(
            model="gpt-5-nano",
            temperature=raw_settings.temperature,
            top_p=raw_settings.top_p,
        )

    sanitized = sanitize_model_settings("gpt-5-nano", raw_settings)

    assert sanitized is not raw_settings
    assert sanitized.temperature is None
    assert sanitized.top_p == pytest.approx(0.6)

    client.create(
        model="gpt-5-nano",
        temperature=sanitized.temperature,
        top_p=sanitized.top_p,
    )

    assert client.last_kwargs["temperature"] is None
    assert client.last_kwargs["top_p"] == pytest.approx(0.6)
