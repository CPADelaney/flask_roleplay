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


def test_responses_tools_use_function_payload_schema():
    tools = ToolSchemaManager.get_all_tools()
    assert tools, "Expected at least one tool schema to be registered"

    for tool in tools:
        assert tool.get("type") == "function", "Tool payload must use Responses function type"
        assert "function" not in tool, "Legacy function wrapper should not remain in Responses tool payload"

        schema_name = tool.get("name")
        assert schema_name, "Responses tool is missing its 'name'"

        parameters = tool.get("parameters")
        assert isinstance(parameters, dict), "Responses tool must include parameters object"
        assert parameters.get("type") == "object", "Parameters must default to object type"
        assert "properties" in parameters, "Parameters must expose properties map"
