"""Tiny conversation snapshot store used by the sync path."""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict

try:  # pragma: no cover - optional dependency in tests
    import redis
except Exception:  # pragma: no cover
    redis = None  # type: ignore


class ConversationSnapshotStore:
    """Persist a minimal snapshot of the active scene for a conversation."""

    def __init__(self, namespace: str = "nyx:conversation:snapshot") -> None:
        self._namespace = namespace
        self._local: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._redis = self._build_client()

    def _build_client(self):
        if redis is None:
            return None
        url = os.getenv("NYX_SNAPSHOT_REDIS", os.getenv("REDIS_URL", "redis://localhost:6379/2"))
        try:
            client = redis.Redis.from_url(url)
            client.ping()
            return client
        except Exception:
            return None

    def _key(self, user_id: str, conversation_id: str) -> str:
        return f"{self._namespace}:{user_id}:{conversation_id}"

    def get(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        key = self._key(user_id, conversation_id)
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                if raw:
                    return json.loads(raw)
            except Exception:
                pass
        with self._lock:
            return dict(self._local.get(key, {}))

    def put(self, user_id: str, conversation_id: str, snapshot: Dict[str, Any]) -> None:
        key = self._key(user_id, conversation_id)
        data = json.dumps(snapshot)
        if self._redis is not None:
            try:
                self._redis.setex(key, 3600, data)
            except Exception:
                pass
        with self._lock:
            self._local[key] = dict(snapshot)


__all__ = ["ConversationSnapshotStore"]
