from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover - test environment shim
    sys.path.insert(0, str(ROOT))

from tests.unit.test_get_tagged_lore import (  # reuse stubbed orchestrator module
    LoreOrchestrator,
    OrchestratorConfig,
)


@pytest.fixture
def anyio_backend():  # pragma: no cover - test backend shim
    return "asyncio"


@pytest.mark.anyio
async def test_get_scene_bundle_uses_mpf_timeout(monkeypatch):
    config = OrchestratorConfig(
        enable_cache=False,
        enable_validation=False,
        subfetch_timeout=0.25,
        mpf_subfetch_timeout=12.0,
    )
    orchestrator = LoreOrchestrator(user_id=99, conversation_id=1, config=config)

    monkeypatch.setattr(orchestrator, "_generate_scene_cache_key", lambda scope: "scope-key")
    orchestrator._get_cached_bundle = lambda _key: None
    orchestrator._cache_bundle = lambda _key, _value: None

    monkeypatch.setattr(orchestrator, "_fetch_location_lore_for_bundle", AsyncMock(return_value={}))
    monkeypatch.setattr(orchestrator, "_fetch_religions_for_location", AsyncMock(return_value=[]))
    monkeypatch.setattr(orchestrator, "_fetch_myths_for_location", AsyncMock(return_value=[]))
    monkeypatch.setattr(orchestrator, "mpf_generate_core_principles", AsyncMock(return_value=["core"]))
    monkeypatch.setattr(orchestrator, "mpf_generate_power_expressions", AsyncMock(return_value=["expression"]))
    monkeypatch.setattr(orchestrator, "mpf_generate_hierarchical_constraints", AsyncMock(return_value=["constraint"]))

    async def _identity(bundle):
        return bundle

    monkeypatch.setattr(orchestrator, "apply_matriarchal_lens_to_bundle", _identity)

    scope = SimpleNamespace(location_id=1, lore_tags=set(), conflict_ids=set(), nation_ids=set())

    timeouts = []
    original_wait = LoreOrchestrator._wait_with_timeout

    async def recording_wait(self, awaitable, label):
        timeout = self._resolve_timeout_for_label(label)
        timeouts.append((label, timeout))
        return await original_wait(self, awaitable, label)

    monkeypatch.setattr(LoreOrchestrator, "_wait_with_timeout", recording_wait)

    await orchestrator.get_scene_bundle(scope)

    def get_timeout(record_label: str) -> float:
        for label, timeout in timeouts:
            if label == record_label:
                assert timeout is not None, f"Expected timeout for {record_label}"
                return timeout
        raise AssertionError(f"No timeout recorded for {record_label}")

    assert get_timeout("location") == pytest.approx(config.subfetch_timeout)
    assert get_timeout("mpf_core") == pytest.approx(config.mpf_subfetch_timeout)
    assert get_timeout("mpf_expressions") == pytest.approx(config.mpf_subfetch_timeout)
    assert get_timeout("mpf_constraints") == pytest.approx(config.mpf_subfetch_timeout)


def test_resolve_timeout_prefers_mpf_override():
    config = OrchestratorConfig(subfetch_timeout=2.0, mpf_subfetch_timeout=15.0)
    orchestrator = LoreOrchestrator(user_id=1, conversation_id=2, config=config)

    assert orchestrator._resolve_timeout_for_label("mpf_core") == pytest.approx(config.mpf_subfetch_timeout)
    assert orchestrator._resolve_timeout_for_label("location") == pytest.approx(config.subfetch_timeout)

    config = OrchestratorConfig(subfetch_timeout=3.0, mpf_subfetch_timeout=None)
    orchestrator = LoreOrchestrator(user_id=1, conversation_id=2, config=config)
    assert orchestrator._resolve_timeout_for_label("mpf_core") == pytest.approx(config.subfetch_timeout)
