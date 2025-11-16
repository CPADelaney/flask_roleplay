import importlib.util
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "dummy")


def _load_utils_module():
    module_name = "nyx.nyx_agent.utils"
    if module_name in sys.modules:
        return sys.modules[module_name]

    # Provide lightweight stubs for the nyx gateway dependency tree so the module
    # can be imported without initializing the full platform.
    nyx_pkg = sys.modules.setdefault("nyx", types.ModuleType("nyx"))
    nyx_pkg.__path__ = getattr(nyx_pkg, "__path__", [])

    gateway_pkg = sys.modules.setdefault("nyx.gateway", types.ModuleType("nyx.gateway"))
    gateway_pkg.__path__ = getattr(gateway_pkg, "__path__", [])

    llm_gateway_mod = types.ModuleType("nyx.gateway.llm_gateway")
    llm_gateway_mod.LLMRequest = type("LLMRequest", (), {})
    llm_gateway_mod.LLMResult = type("LLMResult", (), {})
    llm_gateway_mod.execute = lambda *_, **__: None
    sys.modules["nyx.gateway.llm_gateway"] = llm_gateway_mod

    gateway_pkg.llm_gateway = llm_gateway_mod
    nyx_pkg.gateway = gateway_pkg

    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).resolve().parents[1] / "nyx" / "nyx_agent" / "utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


nyx_utils = _load_utils_module()


def test_did_call_tool_handles_string_events():
    resp = [
        "not json",
        json.dumps({"type": "function_call", "name": "generate_universal_updates"}),
    ]

    assert nyx_utils._did_call_tool(resp, "generate_universal_updates")


def test_tool_output_parses_stringified_event_payload():
    output_payload = {"foo": "bar", "num": 42}
    resp = [
        json.dumps(
            {
                "type": "function_call_output",
                "name": "generate_universal_updates",
                "output": json.dumps(output_payload),
            }
        )
    ]

    assert nyx_utils._tool_output(resp, "generate_universal_updates") == output_payload
