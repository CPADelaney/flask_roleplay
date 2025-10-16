"""Utilities for constructing metadata payloads for ChatKit streaming."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional


def build_metadata_payload(
    *,
    conversation_id: Optional[Any],
    user_id: Optional[Any],
    request_id: Optional[Any] = None,
    assistant_id: Optional[Any] = None,
    openai_conversation_id: Optional[Any] = None,
    thread_id: Optional[Any] = None,
) -> Dict[str, Any]:
    """Normalise metadata values for ChatKit requests."""

    payload: Dict[str, Any] = {
        "conversation_id": str(conversation_id)
        if conversation_id is not None
        else None,
        "user_id": str(user_id) if user_id is not None else None,
    }

    if request_id:
        payload["request_id"] = str(request_id)
    if assistant_id:
        payload["assistant_id"] = str(assistant_id)
    if openai_conversation_id:
        payload["openai_conversation_id"] = str(openai_conversation_id)
    if thread_id:
        payload["thread_id"] = str(thread_id)

    return payload


def encode_safety_metadata(safety_context: Mapping[str, Any]) -> str:
    """Serialise guardrail metadata so ChatKit receives a string value.

    ChatKit expects metadata values to be strings. We JSON-encode the guardrail
    details after coercing each field to a string to avoid type mismatches.
    """

    denial_preview = (safety_context.get("denial_text") or "")[:160]
    guardrail_details = {
        "guardrail": "deny",
        "integrated": "true",
        "denial_text_preview": denial_preview,
    }
    return json.dumps(guardrail_details, ensure_ascii=False)
