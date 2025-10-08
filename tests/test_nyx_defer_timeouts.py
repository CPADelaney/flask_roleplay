import asyncio
import os
import sys
import time
import types
import pytest
import typing
import typing_extensions

typing.TypedDict = typing_extensions.TypedDict


dummy_models = types.ModuleType("sentence_transformers.models")
dummy_models.Transformer = lambda *args, **kwargs: None
dummy_models.Pooling = lambda *args, **kwargs: None


class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):  # pragma: no cover - defensive stub
        return [[0.0] * 3 for _ in texts]

    def get_sentence_embedding_dimension(self):  # pragma: no cover - defensive stub
        return 3


dummy_sentence_transformers = types.ModuleType("sentence_transformers")
dummy_sentence_transformers.SentenceTransformer = DummySentenceTransformer
dummy_sentence_transformers.models = dummy_models

sys.modules.setdefault("sentence_transformers", dummy_sentence_transformers)
sys.modules.setdefault("sentence_transformers.models", dummy_models)

os.environ.setdefault("OPENAI_API_KEY", "test-key")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nyx.nyx_agent._feasibility_helpers import DeferPromptContext  # noqa: E402
from nyx.nyx_agent import orchestrator  # noqa: E402
from nyx.nyx_agent_sdk import NyxAgentSDK  # noqa: E402


class _DummySnapshotStore:
    def get(self, user_id, conversation_id):  # pragma: no cover - defensive stub
        return {}

    def put(self, user_id, conversation_id, snapshot):  # pragma: no cover - defensive stub
        return None


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _isolate_runner(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("nyx.nyx_agent_sdk.ConversationSnapshotStore", lambda: _DummySnapshotStore())
    monkeypatch.setattr("nyx.nyx_agent_sdk.ResponseFilter", None, raising=False)


@pytest.mark.anyio
async def test_generate_defer_taunt_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    class HangingRunner:
        @staticmethod
        async def run(*args, **kwargs):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                raise

    monkeypatch.setattr(orchestrator, "Runner", HangingRunner)
    monkeypatch.setattr(orchestrator, "nyx_main_agent", object())
    monkeypatch.setattr(orchestrator, "DEFER_RUN_TIMEOUT_SECONDS", 0.05)

    context = DeferPromptContext(
        narrator_guidance="The vault is sealed tight until you fetch the key.",
        leads=["Search the study"],
        violations=[{"rule": "missing_prereq", "reason": "The key is still hidden."}],
        persona_prefix="Oh, pet,",
        reason_phrases=["the key is still hidden"],
    )

    start = time.perf_counter()
    result = await orchestrator._generate_defer_taunt(context, trace_id="timeout-test")
    elapsed = time.perf_counter() - start

    assert result is None
    assert elapsed < 1.0, f"Deferred taunt should timeout quickly, got {elapsed:.2f}s"


@pytest.mark.anyio
async def test_sdk_generate_defer_narrative_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    class HangingRunner:
        @staticmethod
        async def run(*args, **kwargs):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                raise

    monkeypatch.setattr("nyx.nyx_agent_sdk.Runner", HangingRunner)
    monkeypatch.setattr("nyx.nyx_agent_sdk.nyx_main_agent", object())
    monkeypatch.setattr("nyx.nyx_agent_sdk.DEFER_RUN_TIMEOUT_SECONDS", 0.05)

    sdk = NyxAgentSDK()

    context = DeferPromptContext(
        narrator_guidance="Reality isn't bending until you retrieve the access badge.",
        leads=["Ask the captain"],
        violations=[{"rule": "missing_prereq", "reason": "No badge in sight."}],
        persona_prefix="Sweet thing,",
        reason_phrases=["you haven't produced the badge"],
    )

    start = time.perf_counter()
    result = await sdk._generate_defer_narrative(context, trace_id="sdk-timeout")
    elapsed = time.perf_counter() - start

    assert result is None
    assert elapsed < 1.0, f"SDK defer narrative should timeout quickly, got {elapsed:.2f}s"
