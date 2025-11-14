"""Utility helpers for constructing Nyx LLM gateway requests."""

from __future__ import annotations

from typing import Any

from nyx.gateway.llm_gateway import LLMRequest


def build_llm_request(*args: Any, **kwargs: Any) -> LLMRequest:
    """Create an :class:`LLMRequest` compatible with legacy ``Runner.run`` calls."""

    runner_kwargs = dict(kwargs)
    context = runner_kwargs.pop("context", None)
    model_override = runner_kwargs.pop("model_override", None)

    agent: Any | None = None
    prompt: Any | None = None

    if args:
        if len(args) > 2:
            raise TypeError("build_llm_request accepts at most two positional arguments")
        agent = args[0]
        if len(args) > 1:
            prompt = args[1]

    if "agent" in runner_kwargs:
        agent = runner_kwargs.pop("agent")
    if "starting_agent" in runner_kwargs:
        agent = runner_kwargs.pop("starting_agent")
    if "prompt" in runner_kwargs:
        prompt = runner_kwargs.pop("prompt")
    if "input" in runner_kwargs:
        prompt = runner_kwargs.pop("input")

    if agent is None:
        raise ValueError("Agent specification is required for build_llm_request")
    if prompt is None:
        raise ValueError("Prompt/input is required for build_llm_request")

    return LLMRequest(
        prompt=prompt,
        agent=agent,
        context=context,
        runner_kwargs=runner_kwargs or None,
        model_override=model_override,
    )
