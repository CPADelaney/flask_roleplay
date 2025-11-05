"""Unit tests for Nyx heavy memory tasks."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

import nyx.tasks.heavy.memory_tasks as memory_tasks


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear environment overrides that influence vector store selection."""

    monkeypatch.delenv("MEMORY_VECTOR_STORE_TYPE", raising=False)
    monkeypatch.delenv("MEMORY_EMBEDDING_TYPE", raising=False)


def _build_config(*, legacy: bool = False) -> Dict[str, Any]:
    return {
        "vector_store": {
            "type": "chroma",
            "use_legacy_vector_store": legacy,
            "hosted_vector_store_ids": ["vs-1"],
        },
        "embedding": {
            "type": "openai",
        },
    }


def test_hydrate_local_embeddings_short_circuits_for_hosted_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory_tasks, "get_memory_config", lambda: _build_config(legacy=False))
    monkeypatch.setattr(memory_tasks, "get_hosted_vector_store_ids", lambda config=None: ["vs-1"])
    monkeypatch.setattr(memory_tasks, "hosted_vector_store_enabled", lambda *_args, **_kwargs: True)

    result = memory_tasks.hydrate_local_embeddings(user_id=1, conversation_id=2)

    assert result == "skipped:hosted-vector-store"


def test_hydrate_local_embeddings_skips_when_legacy_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory_tasks, "get_memory_config", lambda: _build_config(legacy=False))
    monkeypatch.setattr(memory_tasks, "get_hosted_vector_store_ids", lambda config=None: [])
    monkeypatch.setattr(memory_tasks, "hosted_vector_store_enabled", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(memory_tasks, "legacy_vector_store_enabled", lambda *_args, **_kwargs: False)

    result = memory_tasks.hydrate_local_embeddings(user_id=3, conversation_id=4)

    assert result == "skipped:legacy-disabled"


def test_hydrate_local_embeddings_invokes_memory_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory_tasks, "get_memory_config", lambda: _build_config(legacy=True))
    monkeypatch.setattr(memory_tasks, "get_hosted_vector_store_ids", lambda config=None: [])
    monkeypatch.setattr(memory_tasks, "hosted_vector_store_enabled", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(memory_tasks, "legacy_vector_store_enabled", lambda *_args, **_kwargs: True)

    calls: List[Dict[str, Any]] = []

    class _StubService:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

        async def hydrate_legacy_vector_store(self) -> str:
            return "hydrated"

    monkeypatch.setattr(memory_tasks, "MemoryEmbeddingService", _StubService)

    result = memory_tasks.hydrate_local_embeddings(user_id=5, conversation_id=6)

    assert result == "hydrated"
    assert calls == [
        {
            "user_id": 5,
            "conversation_id": 6,
            "vector_store_type": "chroma",
            "embedding_model": "openai",
            "config": _build_config(legacy=True),
        }
    ]


def test_hydrate_local_embeddings_accepts_trace_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory_tasks, "get_memory_config", lambda: _build_config(legacy=False))
    monkeypatch.setattr(memory_tasks, "get_hosted_vector_store_ids", lambda config=None: ["vs-1"])
    monkeypatch.setattr(memory_tasks, "hosted_vector_store_enabled", lambda *_args, **_kwargs: True)

    result = memory_tasks.hydrate_local_embeddings(
        user_id=7,
        conversation_id=8,
        trace_id="trace-123",
    )

    assert result == "skipped:hosted-vector-store"
