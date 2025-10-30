"""Lightweight helpers for retrieval via the configured RAG backend."""

from .backend import ask, BackendPreference, get_configured_backend

__all__ = ["ask", "BackendPreference", "get_configured_backend"]
