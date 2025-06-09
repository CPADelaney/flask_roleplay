from __future__ import annotations

import math
from agents import function_tool

@function_tool
def calc(expression: str) -> float:
    """Safely evaluate a math expression and return the result."""
    allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed_names.update({"abs": abs, "round": round})
    return eval(expression, {"__builtins__": None}, allowed_names)
