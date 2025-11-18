"""Public entrypoint for Nyx gateway helpers."""

# Ensure OpenAI client instrumentation is installed as soon as the gateway loads.
from . import openai_instrumentation  # noqa: F401

from .llm_gateway import (
    LLMOperation,
    LLMRequest,
    LLMResult,
    execute,
    execute_stream,
)

__all__ = [
    "LLMOperation",
    "LLMRequest",
    "LLMResult",
    "execute",
    "execute_stream",
]
