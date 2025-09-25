"""Compatibility layer for legacy ConflictSystemIntegration imports.

This module provides a thin wrapper around :class:`ConflictSystemInterface`
so that existing call sites (Celery tasks, NPC systems, etc.) can continue to
use the familiar ``ConflictSystemIntegration`` name while the new
``ConflictSystemInterface`` hosts the real implementation.

Only the minimal lifecycle helpers that older code expects (``get_instance``,
``initialize`` and ``generate_conflict``) are implemented explicitly.  All
other attribute access is proxied to the underlying interface so consumers can
progressively migrate to the new API surface without breaking imports.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple
from weakref import WeakValueDictionary

from agents.run_context import RunContextWrapper

from logic.conflict_system.integration import ConflictSystemInterface, IntegrationMode

logger = logging.getLogger(__name__)


class ConflictSystemIntegration:
    """Backwards-compatible wrapper around :class:`ConflictSystemInterface`."""

    _instances: "WeakValueDictionary[Tuple[int, int], ConflictSystemIntegration]" = WeakValueDictionary()
    _instance_lock: asyncio.Lock = asyncio.Lock()

    def __init__(self, user_id: int, conversation_id: int, mode: IntegrationMode | str = IntegrationMode.EMERGENT):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.mode = self._normalize_mode(mode)
        self._interface = ConflictSystemInterface(user_id, conversation_id)
        self._synthesizer = None  # cached synthesizer instance once resolved

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    @classmethod
    async def get_instance(cls, user_id: int, conversation_id: int) -> "ConflictSystemIntegration":
        """Return (and cache) a wrapper for the given conversation."""

        key = (user_id, conversation_id)
        async with cls._instance_lock:
            instance = cls._instances.get(key)
            if instance is None:
                instance = cls(user_id, conversation_id)
                cls._instances[key] = instance
        return instance

    async def initialize(self, mode: IntegrationMode | str | None = None) -> Dict[str, Any]:
        """Initialize the underlying conflict system."""

        if mode is not None:
            self.mode = self._normalize_mode(mode)
        return await self._interface.initialize_system(self.mode)

    async def generate_conflict(
        self,
        ctx: Optional[RunContextWrapper],
        conflict_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate a conflict via the synthesizer, matching legacy semantics."""

        conflict_params = conflict_params or {}
        conflict_type = conflict_params.get("conflict_type", "standard")

        synthesizer = await self._get_synthesizer()
        try:
            await synthesizer.initialize_all_subsystems()
        except Exception:  # pragma: no cover - synthesizer handles idempotency
            logger.debug("Synthesizer subsystem initialization failed", exc_info=True)

        context_payload: Dict[str, Any] = {}
        if ctx is not None:
            if hasattr(ctx, "context") and isinstance(ctx.context, dict):
                context_payload.update(ctx.context)
            if hasattr(ctx, "data") and isinstance(ctx.data, dict):
                context_payload.update(ctx.data)

        context_payload.setdefault("user_id", self.user_id)
        context_payload.setdefault("conversation_id", self.conversation_id)

        # Merge additional parameters (intensity, player involvement, etc.)
        extra_context = {k: v for k, v in conflict_params.items() if k != "conflict_type"}
        context_payload.update(extra_context)

        result = await self._interface.create_conflict(conflict_type, context_payload)
        if not isinstance(result, dict):
            return None

        status = str(result.get("status", "")).lower()
        success = status not in {"failed", "error"}

        return {
            "success": success,
            "conflict_id": result.get("conflict_id"),
            "conflict_type": result.get("conflict_type", conflict_type),
            "message": result.get("message", ""),
            "conflict_details": result.get("conflict_details"),
            "raw_result": result,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _get_synthesizer(self):
        if self._synthesizer is None:
            self._synthesizer = await self._interface._get_synthesizer()
        return self._synthesizer

    @staticmethod
    def _normalize_mode(mode: IntegrationMode | str) -> IntegrationMode:
        if isinstance(mode, IntegrationMode):
            return mode
        try:
            return IntegrationMode(mode)
        except ValueError:
            try:
                return IntegrationMode[mode.upper()]
            except Exception:
                logger.warning("Unknown conflict integration mode '%s', defaulting to EMERGENT", mode)
                return IntegrationMode.EMERGENT

    # ------------------------------------------------------------------
    # Attribute proxying
    # ------------------------------------------------------------------
    def __getattr__(self, item: str) -> Any:  # pragma: no cover - simple delegation
        if hasattr(self._interface, item):
            return getattr(self._interface, item)
        raise AttributeError(item)


__all__ = ["ConflictSystemIntegration"]
