# logic/stat_utils.py
import re
from typing import Any, Dict

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def _guess_delta(raw: Any) -> float:
    """
    Pull the first number out of a free-text modifier string.
    Returns 0.0 if nothing usable is found.
    """
    if isinstance(raw, (int, float)):
        return float(raw)
    if not isinstance(raw, str):
        return 0.0
    m = _NUM_RE.search(raw)
    return float(m.group()) if m else 0.0

def extract_numeric_modifiers(sm: Dict[str, Any]) -> Dict[str, float]:
    """
    Return {stat_name: numeric_delta} no matter which format is stored.

    • New schema  →  {"obedience": {"delta": 20, "description": "…"}}
                     uses the 'delta' key (default 0).

    • Legacy text →  "obedience": "locks at higher levels, …"
                     uses _guess_delta() heuristic (default 0).
    """
    result: Dict[str, float] = {}
    for stat, val in sm.items():
        if isinstance(val, dict):                  # new nested object
            result[stat.lower()] = float(val.get("delta", 0))
        else:                                      # legacy string / number
            result[stat.lower()] = _guess_delta(val)
    return result
