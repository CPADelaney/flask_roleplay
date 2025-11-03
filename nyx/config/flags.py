"""Runtime feature flags for gradually rolling out Nyx architecture changes."""
from __future__ import annotations

import os


def _read_flag(name: str, default: str = "on") -> str:
    value = os.getenv(name, default)
    return value.lower() if isinstance(value, str) else str(value).lower()


def _is_enabled(value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return True
    normalized = normalized.lower()
    return normalized not in {"0", "false", "off", "disable", "disabled", "no"}


def _flag(name: str, default: str = "on") -> bool:
    return _is_enabled(_read_flag(name, default))


def llm_gateway_enabled() -> bool:
    """Return True when the consolidated LLM gateway should be used."""

    return _flag("NYX_FLAG_LLM_GATEWAY", "on")


def outbox_enabled() -> bool:
    """Return True when the transactional outbox fanout pipeline is active."""

    return _flag("NYX_FLAG_OUTBOX", "on")


def versioned_cache_enabled() -> bool:
    """Return True if versioned snapshot caches should be consulted."""

    return _flag("NYX_FLAG_VERSIONED_CACHE", "on")


def conflict_fsm_enabled() -> bool:
    """Return True when conflict FSM persistence should run."""

    return _flag("NYX_FLAG_CONFLICT_FSM", "on")


def domain_events_enabled() -> bool:
    """Return True when domain events should be emitted to downstream systems."""

    return _flag("NYX_FLAG_DOMAIN_EVENTS", "on")


def output_evals_enabled() -> bool:
    """Return True when automated output evaluation should execute."""

    return _flag("NYX_FLAG_OUTPUT_EVALS", "on")


def context_parallel_init_enabled() -> bool:
    """Return True when context services should initialize subsystems in parallel."""

    return _flag("NYX_FLAG_CONTEXT_PARALLEL_INIT", "on")


def context_parallel_fetch_enabled() -> bool:
    """Return True when per-request context assembly can fan out concurrently."""

    return _flag("NYX_FLAG_CONTEXT_PARALLEL_FETCH", "on")


def context_warmers_enabled() -> bool:
    """Return True when cache warmers/invalidation hooks should execute."""

    return _flag("NYX_FLAG_CONTEXT_WARMERS", "on")


__all__ = [
    "llm_gateway_enabled",
    "outbox_enabled",
    "versioned_cache_enabled",
    "conflict_fsm_enabled",
    "domain_events_enabled",
    "output_evals_enabled",
    "context_parallel_init_enabled",
    "context_parallel_fetch_enabled",
    "context_warmers_enabled",
]
