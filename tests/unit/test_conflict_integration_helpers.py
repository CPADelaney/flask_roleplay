import json
import pathlib
import sys
import typing
import typing_extensions
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover - test environment shim
    sys.path.insert(0, str(ROOT))

if sys.version_info < (3, 12):  # pragma: no cover - compatibility shim
    typing.TypedDict = typing_extensions.TypedDict  # type: ignore[attr-defined]

from nyx.tasks.background.conflict_integration_helpers import (  # noqa: E402
    run_activity_integration,
    run_contextual_conflict_generation,
    run_scene_tension_analysis,
)


@pytest.fixture
def anyio_backend():  # pragma: no cover - test backend shim
    return "asyncio"


@pytest.mark.anyio
async def test_run_scene_tension_analysis_parses_payload(monkeypatch):
    context = {"location": "market", "activity": "browsing", "npcs_present": 2}
    payload = {
        "tensions": [{"source": "ambient", "level": 0.5, "description": "test"}],
        "should_generate_conflict": True,
        "suggested_type": "slice_of_life",
        "manifestation": ["glances"],
    }

    async def fake_run(agent, prompt):  # pragma: no cover - shim
        return SimpleNamespace(result=json.dumps(payload))

    monkeypatch.setattr("nyx.tasks.background.conflict_integration_helpers.Runner.run", fake_run)

    result = await run_scene_tension_analysis(SimpleNamespace(), context)

    assert result["tensions"][0]["source"] == "ambient"
    assert result["context"] == context
    assert result["source"] == "llm"
    assert result["cached_at"]


@pytest.mark.anyio
async def test_run_contextual_conflict_generation_fallback(monkeypatch):
    async def fake_run(agent, prompt):  # pragma: no cover - shim
        return SimpleNamespace(result="not json")

    monkeypatch.setattr("nyx.tasks.background.conflict_integration_helpers.Runner.run", fake_run)

    result = await run_contextual_conflict_generation(SimpleNamespace(), {}, [1, 2])

    assert result["name"]
    assert result["source"] == "llm"
    assert result["cached_at"]


@pytest.mark.anyio
async def test_run_activity_integration_handles_invalid_payload(monkeypatch):
    async def fake_run(agent, prompt):  # pragma: no cover - shim
        raise RuntimeError("failure")

    monkeypatch.setattr("nyx.tasks.background.conflict_integration_helpers.Runner.run", fake_run)

    result = await run_activity_integration(SimpleNamespace(), "cooking", [{"id": 1}])

    assert result["manifestations"]
    assert result["conflicts_active"] is True
    assert result["source"] == "llm"
    assert result["cached_at"]
