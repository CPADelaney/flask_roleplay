from __future__ import annotations

import json
from typing import Any, Dict


def sanitize_trace_metadata(meta: Dict[str, Any] | None) -> Dict[str, str]:
    """Normalize trace metadata values to strings."""
    if not meta:
        return {}

    sanitized: Dict[str, str] = {}
    for key, value in meta.items():
        if isinstance(value, str):
            sanitized[key] = value
        elif value is None:
            sanitized[key] = ""
        elif isinstance(value, (int, float, bool)):
            sanitized[key] = str(value)
        else:
            try:
                sanitized[key] = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                sanitized[key] = str(value)
    return sanitized


__all__ = ["sanitize_trace_metadata"]
