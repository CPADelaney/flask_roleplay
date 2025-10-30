"""Lightweight helpers for retrieval via the configured RAG backend."""

from __future__ import annotations

import pathlib
import agents as _agents  # type: ignore

_LOCAL_AGENTS_PATH = pathlib.Path(__file__).resolve().parent.parent / "agents"
if _LOCAL_AGENTS_PATH.exists():
    agents_path = getattr(_agents, "__path__", None)
    if isinstance(agents_path, list) and str(_LOCAL_AGENTS_PATH) not in agents_path:
        agents_path.append(str(_LOCAL_AGENTS_PATH))

from .backend import ask, BackendPreference, get_configured_backend

__all__ = ["ask", "BackendPreference", "get_configured_backend"]
