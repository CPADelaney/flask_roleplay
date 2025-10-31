import logging
from types import SimpleNamespace

import pytest

from nyx.gateway import llm_gateway
from nyx.gateway.llm_gateway import LLMRequest


class RetryableError(RuntimeError):
    """Custom exception used to simulate retryable failures."""


@pytest.fixture
def stub_runner(monkeypatch):
    class DummyRunner:
        calls: list[str] = []

    monkeypatch.setattr(llm_gateway, "_runner", DummyRunner, raising=False)
    return DummyRunner


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _agent(name: str = "agent") -> SimpleNamespace:
    return SimpleNamespace(name=name, model=object())


@pytest.mark.anyio("asyncio")
async def test_execute_success_returns_raw_and_metadata_and_logs_trace_id(stub_runner, caplog):
    fake_result = SimpleNamespace(
        final_output="all good",
        usage={"prompt_tokens": 5},
        response_id="resp-42",
    )

    async def run_impl(cls, agent, prompt, **kwargs):
        return fake_result

    stub_runner.run = classmethod(run_impl)

    caplog.set_level(logging.INFO, logger=llm_gateway.logger.name)

    request = LLMRequest(
        prompt="Hello there",
        agent=_agent("primary"),
        metadata={"trace_id": "trace-123"},
    )

    result = await llm_gateway.execute(request)

    assert result.text == "all good"
    assert result.raw is fake_result
    assert result.agent_name == "primary"
    assert result.metadata["usage"] == {"prompt_tokens": 5}
    assert result.metadata["trace_id"] == "trace-123"

    start_record = next(r for r in caplog.records if r.message == "nyx.gateway.llm.execute.start")
    assert "trace_id" in start_record.metadata_keys
    success_record = next(r for r in caplog.records if r.message == "nyx.gateway.llm.execute.success")
    assert "trace_id" in success_record.metadata_keys


@pytest.mark.anyio("asyncio")
async def test_execute_retries_with_backoff_then_succeeds(stub_runner, monkeypatch):
    attempts: list[int] = []
    fake_result = SimpleNamespace(final_output="eventual win")

    async def run_impl(cls, agent, prompt, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise RetryableError("temporary")
        return fake_result

    stub_runner.run = classmethod(run_impl)

    sleep_calls: list[float] = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr(llm_gateway.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(llm_gateway.random, "uniform", lambda a, b: 0.1)

    request = LLMRequest(
        prompt="needs retry",
        agent=_agent("primary"),
        metadata={"trace_id": "retry-trace"},
        retryable_exceptions=(RetryableError,),
        max_attempts=3,
        backoff_initial=0.5,
        backoff_multiplier=2.0,
        backoff_jitter=0.2,
    )

    result = await llm_gateway.execute(request)

    assert len(attempts) == 2
    assert sleep_calls == [0.6]
    assert result.text == "eventual win"
    assert result.attempts == 2
    assert result.raw is fake_result


@pytest.mark.anyio("asyncio")
async def test_execute_raises_after_retry_exhaustion(stub_runner, monkeypatch):
    attempts: list[int] = []

    async def run_impl(cls, agent, prompt, **kwargs):
        attempts.append(1)
        raise RetryableError("still failing")

    stub_runner.run = classmethod(run_impl)

    sleep_calls: list[float] = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr(llm_gateway.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(llm_gateway.random, "uniform", lambda a, b: 0.0)

    request = LLMRequest(
        prompt="always fails",
        agent=_agent("primary"),
        retryable_exceptions=(RetryableError,),
        max_attempts=2,
        backoff_initial=0.5,
        backoff_multiplier=2.0,
        backoff_jitter=0.2,
    )

    with pytest.raises(RetryableError):
        await llm_gateway.execute(request)

    assert len(attempts) == 2
    assert sleep_calls == [0.5]


@pytest.mark.anyio("asyncio")
async def test_execute_falls_back_to_secondary_agent(stub_runner):
    stub_runner.calls = []
    fallback_result = SimpleNamespace(final_output="fallback success")

    async def run_impl(cls, agent, prompt, **kwargs):
        stub_runner.calls.append(agent.name)
        if agent.name == "primary":
            raise RetryableError("primary failed")
        return fallback_result

    stub_runner.run = classmethod(run_impl)

    request = LLMRequest(
        prompt="use fallback",
        agent=_agent("primary"),
        fallback_agent=_agent("fallback"),
        retryable_exceptions=(RetryableError,),
        max_attempts=1,
    )

    result = await llm_gateway.execute(request)

    assert stub_runner.calls == ["primary", "fallback"]
    assert result.used_fallback is True
    assert result.agent_name == "fallback"
    assert result.raw is fallback_result
    assert result.attempts == 1


@pytest.mark.anyio("asyncio")
async def test_execute_stream_yields_final_result_preserving_chunk_order(stub_runner):
    chunks = ["one", "two", "three"]
    stream_result = SimpleNamespace(
        final_output="".join(chunks),
        chunks=list(chunks),
        usage={"completion_tokens": 3},
    )

    async def run_impl(cls, agent, prompt, **kwargs):
        assert kwargs.get("stream") is True
        return stream_result

    stub_runner.run = classmethod(run_impl)

    request = LLMRequest(
        prompt="stream please",
        agent=_agent("streamer"),
        runner_kwargs={"stream": True},
        metadata={"trace_id": "stream-trace"},
    )

    results: list[llm_gateway.LLMResult] = []
    async for result in llm_gateway.execute_stream(request):
        results.append(result)

    assert [r.text for r in results] == ["".join(chunks)]
    assert results[0].raw is stream_result
    assert results[0].raw.chunks == chunks
