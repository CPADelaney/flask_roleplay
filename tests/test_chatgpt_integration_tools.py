"""Tests for Responses tool schema preparation."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

# Ensure offline-friendly configuration for optional dependencies.
os.environ.setdefault("SENTENCE_TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

# Provide lightweight stubs for Nyx modules so importing the integration
# layer does not pull heavy runtime dependencies (vector stores, etc.).
nyx_stub = types.ModuleType("nyx")
nyx_stub.__path__ = []  # Mark as namespace package
nyx_core_stub = types.ModuleType("nyx.core")
nyx_core_stub.__path__ = []
nyx_orchestrator_stub = types.ModuleType("nyx.core.orchestrator")
setattr(nyx_orchestrator_stub, "prepare_context", None)

sys.modules.setdefault("nyx", nyx_stub)
sys.modules.setdefault("nyx.core", nyx_core_stub)
sys.modules.setdefault("nyx.core.orchestrator", nyx_orchestrator_stub)
setattr(nyx_stub, "core", nyx_core_stub)
setattr(nyx_core_stub, "orchestrator", nyx_orchestrator_stub)

from logic.chatgpt_integration import ToolSchemaManager


def test_all_tools_have_top_level_name_matching_schema():
    tools = ToolSchemaManager.get_all_tools()
    assert tools, "Expected at least one tool schema to be registered"

    for tool in tools:
        assert "name" in tool, "Tool is missing top-level 'name'"
        assert "function" in tool, "Tool is missing nested 'function' schema"

        function_schema = tool["function"]
        assert isinstance(function_schema, dict), "Function schema must be a dict"

        schema_name = function_schema.get("name")
        assert schema_name, "Function schema is missing its 'name'"

        assert (
            tool["name"] == schema_name
        ), "Top-level tool name must match nested function schema name"
