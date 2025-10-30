import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sentence_transformers_stub = ModuleType("sentence_transformers")


class _StubSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, *args, **kwargs):  # pragma: no cover
        return []


sentence_transformers_stub.SentenceTransformer = _StubSentenceTransformer
sentence_transformers_util_stub = ModuleType("sentence_transformers.util")
sentence_transformers_stub.util = sentence_transformers_util_stub
sys.modules["sentence_transformers"] = sentence_transformers_stub
sys.modules["sentence_transformers.util"] = sentence_transformers_util_stub

embedding_stub_root = ModuleType("embedding")
embedding_vector_store_stub = ModuleType("embedding.vector_store")


def _stub_get_sentence_transformer(*_args, **_kwargs):
    return _StubSentenceTransformer()


embedding_vector_store_stub.SentenceTransformer = _StubSentenceTransformer
embedding_vector_store_stub.get_sentence_transformer = _stub_get_sentence_transformer
embedding_stub_root.vector_store = embedding_vector_store_stub
sys.modules.setdefault("embedding", embedding_stub_root)
sys.modules["embedding.vector_store"] = embedding_vector_store_stub

sys.path.append(str(Path(__file__).resolve().parents[3]))

conflict_module_name = "logic.conflict_system.conflict_synthesizer"
if conflict_module_name not in sys.modules:
    conflict_stub = ModuleType(conflict_module_name)

    class _StubSubsystemType:
        EDGE_HANDLER = "edge_handler"

    class _StubSubsystemResponse:
        def __init__(self, subsystem, event_id, success, data, side_effects=None):
            self.subsystem = subsystem
            self.event_id = event_id
            self.success = success
            self.data = data
            self.side_effects = side_effects or []

    conflict_stub.SubsystemType = _StubSubsystemType
    conflict_stub.SubsystemResponse = _StubSubsystemResponse

    def _stub_get_synthesizer(*_args, **_kwargs):  # pragma: no cover - diagnostic only
        raise NotImplementedError("get_synthesizer is not available in test stubs")

    conflict_stub.get_synthesizer = _stub_get_synthesizer
    sys.modules[conflict_module_name] = conflict_stub

from logic.conflict_system.edge_cases import ConflictEdgeCaseSubsystem


def test_health_check_cache_miss_triggers_celery(monkeypatch):
    async def run_test():
        subsystem = ConflictEdgeCaseSubsystem(user_id=123, conversation_id=456)

        redis_mock = AsyncMock()
        redis_mock.get.return_value = None
        redis_mock.set.return_value = True

        async def fake_get_redis_client():
            return redis_mock

        monkeypatch.setattr(
            "logic.conflict_system.edge_cases.get_redis_client",
            fake_get_redis_client,
        )

        celery_mock = MagicMock()
        celery_mock.send_task = MagicMock()
        monkeypatch.setattr(
            "logic.conflict_system.edge_cases.celery_app",
            celery_mock,
        )

        event = SimpleNamespace(event_id="evt-1")

        response = await subsystem._handle_health_check(event)

        celery_mock.send_task.assert_called_once_with(
            "tasks.update_edge_case_scan",
            args=[123, 456],
        )
        assert response.data["status"] == "scan_in_progress"

    asyncio.run(run_test())
