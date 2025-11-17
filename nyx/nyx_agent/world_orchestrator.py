"""World orchestration facade used by :class:`nyx.nyx_agent.context.NyxContext`."""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["WorldOrchestrator"]


def _lazy_load_world_director():
    """Import ``CompleteWorldDirector`` only when required."""

    from story_agent.world_director_agent import CompleteWorldDirector  # type: ignore

    return CompleteWorldDirector


class WorldOrchestrator:
    """Thin wrapper around ``CompleteWorldDirector`` with cache helpers."""

    def __init__(self, user_id: int, conversation_id: int, *, cache_ttl: float = 5.0):
        self.user_id = int(user_id)
        self.conversation_id = int(conversation_id)
        self._cache_ttl = float(cache_ttl)
        self._scene_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._last_scene_key: Optional[str] = None
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._director: Optional[Any] = None
        self._warmed_hint: Optional[bool] = None

    @property
    def director(self) -> Optional[Any]:
        """Expose the underlying director for legacy integrations."""

        return self._director

    async def initialize(self, *, warmed: Optional[bool] = None) -> None:
        """Initialize the underlying director once."""

        if self._initialized and self._director is not None:
            return

        async with self._init_lock:
            if self._initialized and self._director is not None:
                return

            CompleteWorldDirector = _lazy_load_world_director()
            director = CompleteWorldDirector(self.user_id, self.conversation_id)
            await director.initialize(warmed=warmed)
            self._director = director
            self._initialized = True
            self._warmed_hint = warmed
            logger.info(
                "WorldOrchestrator initialized for user_id=%s conversation_id=%s",
                self.user_id,
                self.conversation_id,
            )

    async def dispose(self) -> None:
        """Release cached resources."""

        self._scene_cache.clear()
        director = self._director
        self._director = None
        self._initialized = False
        if director is not None:
            close = getattr(director, "close", None)
            if callable(close):
                maybe = close()
                if asyncio.iscoroutine(maybe):  # pragma: no cover - best effort
                    await maybe

    async def get_scene_bundle(self, scope: Optional[Any] = None) -> Dict[str, Any]:
        """Return a cached world bundle for the provided scope."""

        director = await self._ensure_ready()
        if director is None:
            return {}

        ctx = getattr(director, "context", None)
        if ctx is None or not hasattr(ctx, "get_world_bundle"):
            return {}

        cache_key = self._scope_key(scope)
        cached = self._scene_cache.get(cache_key)
        now = time.time()
        if cached and cached[1] > now:
            return cached[0]

        bundle = await ctx.get_world_bundle(fast=True)
        if isinstance(bundle, dict):
            self._scene_cache[cache_key] = (bundle, now + self._cache_ttl)
            self._last_scene_key = cache_key
            return bundle

        return {}

    def get_cached_state(self) -> Optional[Any]:
        """Return the most recently cached world_state if available."""

        if self._last_scene_key:
            cached = self._scene_cache.get(self._last_scene_key)
            if cached:
                bundle = cached[0]
                if isinstance(bundle, dict) and "world_state" in bundle:
                    return bundle.get("world_state")

        director = self._director
        ctx = getattr(director, "context", None) if director else None
        if ctx is not None and getattr(ctx, "current_world_state", None) is not None:
            return ctx.current_world_state

        return None

    async def get_world_state(self, scope: Optional[Any] = None) -> Optional[Any]:
        """Fetch the current world state via bundle cache or director."""

        cached = self.get_cached_state()
        if cached is not None:
            return cached

        bundle = await self.get_scene_bundle(scope)
        world_state_obj = bundle.get("world_state") if isinstance(bundle, dict) else None
        if world_state_obj is not None:
            return world_state_obj

        director = await self._ensure_ready()
        if director is None:
            return None

        getter = getattr(director, "get_world_state", None)
        if callable(getter):
            return await getter()

        ctx = getattr(director, "context", None)
        if ctx is not None:
            return getattr(ctx, "current_world_state", None)

        return None

    async def expand_state(
        self,
        entities: Optional[Iterable[str]] = None,
        aspects: Optional[Iterable[str]] = None,
        depth: str = "summary",
        scope: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Return a structured payload of the world state for downstream use."""

        bundle = await self.get_scene_bundle(scope)
        world_state_obj = bundle.get("world_state")
        summary = bundle.get("summary") or {}
        conflict_state = bundle.get("conflict_state") or {}
        activities = bundle.get("available_activities") or []
        patterns = bundle.get("patterns")

        serialized_state = self._serialize_world_state(world_state_obj)
        aspect_set = {a.lower() for a in aspects} if aspects else set()

        payload: Dict[str, Any] = {
            "bundle": bundle,
            "world_state": world_state_obj,
        }

        def _wants(key: str) -> bool:
            return not aspect_set or key in aspect_set

        if _wants("summary"):
            payload["summary"] = summary
        if _wants("conflict") or _wants("conflict_state"):
            payload["conflict_state"] = conflict_state
        if _wants("activities") or _wants("available_activities"):
            payload["available_activities"] = activities
        if _wants("patterns") and patterns is not None:
            payload["patterns"] = patterns

        if serialized_state and (_wants("state") or depth == "full"):
            payload["state"] = serialized_state

        if serialized_state and entities:
            requested = {}
            for entity in entities:
                key = str(entity)
                if key in serialized_state:
                    requested[key] = serialized_state[key]
            if requested:
                payload["entities"] = requested

        return payload

    async def handle_world_operation(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        """Generic handler for imperative world operations."""

        if not isinstance(request, Mapping):
            raise ValueError("world operation request must be a mapping")

        operation = str(request.get("operation") or "").lower()
        scope = request.get("scope")

        if operation in {"get_scene_bundle", "get_bundle"}:
            bundle = await self.get_scene_bundle(scope)
            return {"success": True, "bundle": bundle}

        if operation in {"expand_state", "state"}:
            expanded = await self.expand_state(
                entities=request.get("entities"),
                aspects=request.get("aspects"),
                depth=str(request.get("depth", "summary")),
                scope=scope,
            )
            expanded["success"] = True
            return expanded

        director = await self._ensure_ready()
        if director is None:
            return {"success": False, "error": "world director unavailable"}

        handler_name = request.get("handler") or operation
        handler = getattr(director, handler_name, None)
        if handler is None and getattr(director, "context", None):
            handler = getattr(director.context, handler_name, None)

        if handler is None or not callable(handler):
            return {
                "success": False,
                "error": f"Unsupported world operation '{operation or handler_name}'",
            }

        result = handler(request)
        if asyncio.iscoroutine(result):
            result = await result

        return {"success": True, "result": result}

    async def _ensure_ready(self) -> Optional[Any]:
        if not self._initialized or self._director is None:
            await self.initialize(warmed=self._warmed_hint)
        return self._director

    async def advance_time(self, hours: int = 1) -> Optional[Any]:
        """Advance time via the underlying director if possible."""

        director = await self._ensure_ready()
        if director is None:
            return None

        advancer = getattr(director, "advance_time", None)
        if callable(advancer):
            result = advancer(hours)
            if asyncio.iscoroutine(result):
                return await result
            return result
        return None

    @staticmethod
    def _serialize_world_state(state: Any) -> Dict[str, Any]:
        if state is None:
            return {}
        if hasattr(state, "model_dump"):
            try:
                return state.model_dump(mode="python")  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                logger.debug("world state model_dump failed", exc_info=True)
        if dataclasses.is_dataclass(state):
            try:
                return dataclasses.asdict(state)
            except Exception:  # pragma: no cover - defensive
                logger.debug("world state asdict failed", exc_info=True)
        if isinstance(state, Mapping):
            return dict(state)
        return {}

    @staticmethod
    def _scope_key(scope: Optional[Any]) -> str:
        if scope is None:
            return "global"
        if hasattr(scope, "to_cache_key"):
            try:
                return scope.to_cache_key()
            except Exception:  # pragma: no cover - defensive
                logger.debug("SceneScope.to_cache_key failed", exc_info=True)
        if isinstance(scope, Mapping):
            locator = scope.get("location_id") or scope.get("location_name")
            npcs = scope.get("npc_ids") or scope.get("npcs")
            return f"dict:{locator}:{tuple(sorted(npcs)) if npcs else ''}"
        return f"scope:{hash(str(scope))}"
