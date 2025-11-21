from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest


def _load_place_enrichment(monkeypatch: pytest.MonkeyPatch):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "nyx" / "tasks" / "background" / "place_enrichment.py"

    # Provide lightweight package stubs so we can import the module without
    # triggering the full Nyx bootstrap (which requires external services).
    stub_nyx = types.ModuleType("nyx")
    stub_nyx.__path__ = [str(repo_root / "nyx")]
    monkeypatch.setitem(sys.modules, "nyx", stub_nyx)

    stub_tasks = types.ModuleType("nyx.tasks")
    stub_tasks.__path__ = [str(repo_root / "nyx" / "tasks")]
    monkeypatch.setitem(sys.modules, "nyx.tasks", stub_tasks)
    stub_nyx.tasks = stub_tasks  # type: ignore[attr-defined]

    stub_background = types.ModuleType("nyx.tasks.background")
    stub_background.__path__ = [str(repo_root / "nyx" / "tasks" / "background")]
    monkeypatch.setitem(sys.modules, "nyx.tasks.background", stub_background)
    stub_tasks.background = stub_background  # type: ignore[attr-defined]

    stub_base = types.ModuleType("nyx.tasks.base")

    class _NyxTask:  # pragma: no cover - minimal stand-in
        abstract = True

    class _TaskWrapper:  # pragma: no cover - simple Celery task stub
        def __init__(self, func):
            self._func = func
            self.run = types.MethodType(func, self)

        def apply_async(self, *args, **kwargs):
            self.last_call = (args, kwargs)
            return None

        def __call__(self, *args, **kwargs):
            return self.run(*args, **kwargs)

    class _App:  # pragma: no cover - decorator stub
        def task(self, *args, **kwargs):
            def _decorator(func):
                task = _TaskWrapper(func)
                task.__name__ = func.__name__
                return task

            return _decorator

    stub_base.NyxTask = _NyxTask  # type: ignore[attr-defined]
    stub_base.app = _App()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nyx.tasks.base", stub_base)

    stub_utils = types.ModuleType("nyx.tasks.utils")

    def run_coro(value):  # pragma: no cover - synchronous shim
        return value

    stub_utils.run_coro = run_coro  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nyx.tasks.utils", stub_utils)

    stub_snapshot = types.ModuleType("nyx.conversation.snapshot_store")

    class ConversationSnapshotStore:  # pragma: no cover - lightweight placeholder
        pass

    stub_snapshot.ConversationSnapshotStore = ConversationSnapshotStore  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nyx.conversation.snapshot_store", stub_snapshot)

    stub_fictional = types.ModuleType("nyx.location.fictional_resolver")

    def resolve_fictional(*args, **kwargs):  # pragma: no cover - never reached
        return types.SimpleNamespace(status="ok", candidates=[])

    stub_fictional.resolve_fictional = resolve_fictional  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nyx.location.fictional_resolver", stub_fictional)

    stub_gemini = types.ModuleType("nyx.location.gemini_maps_adapter")

    def resolve_location_with_gemini(*args, **kwargs):  # pragma: no cover
        return types.SimpleNamespace(status="ok", candidates=[], operations=[])

    stub_gemini.resolve_location_with_gemini = resolve_location_with_gemini  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nyx.location.gemini_maps_adapter", stub_gemini)

    stub_query = types.ModuleType("nyx.location.query")

    class PlaceQuery:  # pragma: no cover - data container for type hints
        def __init__(self, raw_text: str, normalized: str, is_travel: bool, target: str, transport_hint: str | None):
            self.raw_text = raw_text
            self.normalized = normalized
            self.is_travel = is_travel
            self.target = target
            self.transport_hint = transport_hint

    stub_query.PlaceQuery = PlaceQuery  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nyx.location.query", stub_query)

    stub_types = types.ModuleType("nyx.location.types")

    @dataclass
    class GeoAnchor:  # pragma: no cover - mirrors production dataclass shape
        lat: float | None = None
        lon: float | None = None
        neighborhood: str | None = None
        city: str | None = None
        region: str | None = None
        country: str | None = None
        label: str | None = None

    class Anchor:  # pragma: no cover - minimal
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Place:  # pragma: no cover - minimal
        def __init__(self, name: str, level: str, meta: dict | None = None):
            self.name = name
            self.level = level
            self.meta = meta or {}

    class ResolveResult:  # pragma: no cover - minimal placeholder
        def __init__(self, status: str, candidates: list | None = None, operations: list | None = None):
            self.status = status
            self.candidates = candidates or []
            self.operations = operations or []

    stub_types.Anchor = Anchor  # type: ignore[attr-defined]
    stub_types.GeoAnchor = GeoAnchor  # type: ignore[attr-defined]
    stub_types.Place = Place  # type: ignore[attr-defined]
    stub_types.ResolveResult = ResolveResult  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nyx.location.types", stub_types)

    spec = importlib.util.spec_from_file_location(
        "nyx.tasks.background.place_enrichment", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_enrich_accepts_trace_id(monkeypatch: pytest.MonkeyPatch):
    module = _load_place_enrichment(monkeypatch)

    result = module.enrich.run(payload={}, trace_id="trace-123")

    assert result is None


def test_fictional_fallback_accepts_trace_id(monkeypatch: pytest.MonkeyPatch):
    module = _load_place_enrichment(monkeypatch)

    result = module.fictional_fallback.run(payload={}, trace_id="trace-456")

    assert result is None


def test_enqueue_serializes_dataclass_hints(monkeypatch: pytest.MonkeyPatch):
    module = _load_place_enrichment(monkeypatch)

    types_mod = sys.modules["nyx.location.types"]
    query_mod = sys.modules["nyx.location.query"]

    query = query_mod.PlaceQuery(
        raw_text="Travel to park",
        normalized="travel to park",
        is_travel=False,
        target="Travel to park",
        transport_hint=None,
    )

    geo_hint = types_mod.GeoAnchor(lat=10.0, lon=20.0, label="Park")
    anchor = types_mod.Anchor(
        scope="real",
        label="Origin",
        lat=1.0,
        lon=2.0,
        primary_city="Metropolis",
        region="Central",
        country="Example",
        world_name="Prime",
        hints={"geo": geo_hint},
    )

    result = types_mod.ResolveResult(status="ok", candidates=[])

    module.enqueue(
        user_id="user-1",
        conversation_id="conv-1",
        query=query,
        anchor=anchor,
        result=result,
        meta={"allow_fictional_overlay": True},
    )

    assert hasattr(module.enrich, "last_call")
    _, kwargs = module.enrich.last_call
    payload = kwargs["kwargs"]["payload"]

    anchor_payload = payload["anchor"]
    hints_payload = anchor_payload["hints"]

    assert isinstance(hints_payload, dict)
    assert isinstance(hints_payload.get("geo"), dict)
    assert hints_payload["geo"]["lat"] == 10.0
    assert hints_payload["geo"]["lon"] == 20.0
    assert hints_payload["geo"]["label"] == "Park"


def test_enqueue_skips_real_world_without_overlay(monkeypatch: pytest.MonkeyPatch):
    module = _load_place_enrichment(monkeypatch)

    types_mod = sys.modules["nyx.location.types"]
    query_mod = sys.modules["nyx.location.query"]

    query = query_mod.PlaceQuery(
        raw_text="Take me to Disneyland",
        normalized="take me to disneyland",
        is_travel=False,
        target="Disneyland",
        transport_hint=None,
    )

    anchor = types_mod.Anchor(scope="real", label="Disneyland")
    result = types_mod.ResolveResult(status="ok", candidates=[])

    module.enqueue(
        user_id="user-1",
        conversation_id="conv-1",
        query=query,
        anchor=anchor,
        result=result,
        meta={},
    )

    assert not hasattr(module.enrich, "last_call")


def test_fictional_fallback_skips_real_world_without_overlay(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_place_enrichment(monkeypatch)

    types_mod = sys.modules["nyx.location.types"]
    query_mod = sys.modules["nyx.location.query"]

    query = query_mod.PlaceQuery(
        raw_text="Stay in Anaheim",
        normalized="stay in anaheim",
        is_travel=False,
        target="Anaheim",
        transport_hint=None,
    )

    anchor = types_mod.Anchor(scope="real", label="Anaheim")

    module.enqueue_fictional_fallback(
        user_id="user-1",
        conversation_id="conv-1",
        query=query,
        anchor=anchor,
        meta={},
    )

    assert not hasattr(module.fictional_fallback, "last_call")
