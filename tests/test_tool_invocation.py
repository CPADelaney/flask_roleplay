from __future__ import annotations

import asyncio
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest


def _load_async_function(path: Path, function_name: str) -> Callable:
    source = path.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(path))
    for node in module_ast.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == function_name:
            extracted_module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(extracted_module)
            namespace: dict[str, Callable] = {}
            exec(compile(extracted_module, str(path), "exec"), namespace)
            return namespace[function_name]
    raise ValueError(f"Function {function_name} not found in {path}")


REPO_ROOT = Path(__file__).resolve().parent.parent

invoke_tool_with_context = _load_async_function(
    REPO_ROOT / "story_agent" / "world_director_agent.py",
    "_invoke_tool_with_context",
)

invoke_function_tool = _load_async_function(
    REPO_ROOT / "logic" / "universal_updater_agent.py",
    "_invoke_function_tool",
)


class _StubFunctionTool:
    def __init__(self):
        self.calls = []

    async def on_invoke_tool(self, ctx, kwargs):
        self.calls.append((ctx, kwargs))
        return {"ctx": ctx, "kwargs": kwargs}


def test_invoke_tool_with_context_uses_on_invoke_tool_without_kwargs():
    tool = _StubFunctionTool()
    ctx = SimpleNamespace(user_id=101)

    result = asyncio.run(invoke_tool_with_context(tool, ctx))

    assert result == {"ctx": ctx, "kwargs": {}}
    assert tool.calls == [(ctx, {})]


def test_invoke_function_tool_uses_on_invoke_tool_with_kwargs():
    tool = _StubFunctionTool()
    ctx = SimpleNamespace()

    result = asyncio.run(invoke_function_tool(tool, ctx, example=123))

    assert result == {"ctx": ctx, "kwargs": {"example": 123}}
    assert tool.calls == [(ctx, {"example": 123})]
