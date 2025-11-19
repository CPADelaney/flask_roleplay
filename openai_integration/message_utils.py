"""Helpers for building Responses API message payloads."""

from __future__ import annotations

from typing import Any, Dict

_ALLOWED_ROLES = {"user", "assistant", "system"}


def _normalize_role(role: str | None) -> str:
    if not role:
        return "user"
    lowered = role.lower()
    if lowered not in _ALLOWED_ROLES:
        return "user"
    return lowered


def _build_content_block(text: Any, role: str) -> Dict[str, str]:
    block_type = "output_text" if role == "assistant" else "input_text"
    return {"type": block_type, "text": "" if text is None else str(text)}


def build_responses_message(role: str, content: Any) -> Dict[str, Any]:
    """Return a message dict compatible with ``responses.create``."""

    normalized_role = _normalize_role(role)
    return {
        "role": normalized_role,
        "content": [_build_content_block(content, normalized_role)],
    }


__all__ = ["build_responses_message"]
