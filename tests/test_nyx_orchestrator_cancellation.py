import asyncio
import os
import sys
import types
import typing
import typing_extensions

import pytest

# Provide lightweight sentence_transformers stub before Nyx imports
_dummy_models = types.ModuleType("sentence_transformers.models")
_dummy_models.Transformer = lambda *args, **kwargs: None
_dummy_models.Pooling = lambda *args, **kwargs: None


class _DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):  # pragma: no cover - defensive stub
        return [[0.0] * 3 for _ in texts]

    def get_sentence_embedding_dimension(self):  # pragma: no cover - defensive stub
        return 3


_dummy_sentence_transformers = types.ModuleType("sentence_transformers")
_dummy_sentence_transformers.SentenceTransformer = _DummySentenceTransformer
_dummy_sentence_transformers.models = _dummy_models

sys.modules["sentence_transformers"] = _dummy_sentence_transformers
sys.modules["sentence_transformers.models"] = _dummy_models

os.environ.setdefault("OPENAI_API_KEY", "test-key")

typing.TypedDict = typing_extensions.TypedDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nyx.nyx_agent import orchestrator as orchestrator_module


class _CancelledNyxContext:
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.current_context = {}
        self.world_director = None
        self.current_location = None

    async def initialize(self) -> None:
        raise asyncio.CancelledError()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_process_user_input_propagates_cancelled_error(monkeypatch):
    async def _noop_fast(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(
        orchestrator_module,
        "NyxContext",
        _CancelledNyxContext,
    )
    monkeypatch.setattr(
        "nyx.nyx_agent.feasibility.assess_action_feasibility_fast",
        _noop_fast,
    )

    with pytest.raises(asyncio.CancelledError):
        await orchestrator_module.process_user_input(
            user_id=1,
            conversation_id=2,
            user_input="test",
        )
