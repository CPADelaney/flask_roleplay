import os

import pytest


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def pytest_collection_modifyitems(config, items):
    should_run = _is_truthy(os.getenv("RUN_RAG_VECTOR_TESTS")) and bool(
        os.getenv("OPENAI_API_KEY")
    )
    if should_run:
        return

    skip_marker = pytest.mark.skip(
        reason="Set RUN_RAG_VECTOR_TESTS=1 (and OPENAI_API_KEY) to run hosted vector store tests"
    )
    for item in items:
        if "requires_openai" in item.keywords:
            item.add_marker(skip_marker)
