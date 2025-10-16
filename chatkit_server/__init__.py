"""Utilities for integrating the OpenAI ChatKit-style Responses API."""

from .metadata import build_metadata_payload, encode_safety_metadata
from .server import RoleplayChatServer
from .streaming import (
    extract_delta_text,
    extract_response_text,
    extract_thread_metadata,
    format_messages_for_chatkit,
    stream_chatkit_tokens,
)

__all__ = [
    "RoleplayChatServer",
    "build_metadata_payload",
    "encode_safety_metadata",
    "extract_delta_text",
    "extract_response_text",
    "extract_thread_metadata",
    "format_messages_for_chatkit",
    "stream_chatkit_tokens",
]
