from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytest

from infra import cache as cache_module


@dataclass
class FakeRedis:
    store: Dict[str, str]

    def __init__(self) -> None:
        self.store = {}
        self.calls: list[tuple[str, str]] = []

    def get(self, key: str) -> str | None:
        self.calls.append(("get", key))
        return self.store.get(key)

    def incr(self, key: str) -> int:
        self.calls.append(("incr", key))
        current = int(self.store.get(key, "0")) + 1
        self.store[key] = str(current)
        return current


@pytest.fixture
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> FakeRedis:
    fake = FakeRedis()

    def _get_client() -> FakeRedis:
        return fake

    monkeypatch.setattr(cache_module, "get_redis_client", _get_client)
    return fake


def test_registry_default_version_zero(fake_redis: FakeRedis) -> None:
    registry = cache_module.VersionedKeyRegistry(namespace="test:versions")

    version = registry.get_version("player-1", "conflict")

    assert version == 0
    assert fake_redis.calls == [("get", "test:versions:player-1:conflict")]


def test_registry_bump_is_atomic(fake_redis: FakeRedis) -> None:
    registry = cache_module.VersionedKeyRegistry(namespace="test:versions")

    bumped = registry.bump("player-42", "memory")

    assert bumped == 1
    assert fake_redis.calls == [("incr", "test:versions:player-42:memory")]
    assert fake_redis.store["test:versions:player-42:memory"] == "1"


def test_registry_is_namespaced_per_player(fake_redis: FakeRedis) -> None:
    registry = cache_module.VersionedKeyRegistry(namespace="test:versions")

    first = registry.bump("player-a", "lore")
    second = registry.bump("player-b", "lore")

    assert first == 1
    assert second == 1
    assert fake_redis.store["test:versions:player-a:lore"] == "1"
    assert fake_redis.store["test:versions:player-b:lore"] == "1"


def test_conflict_helper_updates_suffix(monkeypatch: pytest.MonkeyPatch, fake_redis: FakeRedis) -> None:
    registry = cache_module.VersionedKeyRegistry(namespace="test:versions")
    monkeypatch.setattr(cache_module, "_VERSIONED_REGISTRY", registry)

    key_before = cache_module.conflict_cache_key_with_version("player-9", "flow", "init")
    assert key_before.endswith("v0")

    cache_module.bump_conflict_cache_version("player-9")
    key_after = cache_module.conflict_cache_key_with_version("player-9", "flow", "init")

    assert key_after.endswith("v1")


def test_memory_helper_updates_suffix(monkeypatch: pytest.MonkeyPatch, fake_redis: FakeRedis) -> None:
    registry = cache_module.VersionedKeyRegistry(namespace="test:versions")
    monkeypatch.setattr(cache_module, "_VERSIONED_REGISTRY", registry)

    key_before = cache_module.memory_cache_key_with_version("player-3", "bundle", "scene")
    assert key_before.endswith("v0")

    cache_module.bump_memory_cache_version("player-3")
    key_after = cache_module.memory_cache_key_with_version("player-3", "bundle", "scene")

    assert key_after.endswith("v1")


def test_lore_helper_updates_suffix(monkeypatch: pytest.MonkeyPatch, fake_redis: FakeRedis) -> None:
    registry = cache_module.VersionedKeyRegistry(namespace="test:versions")
    monkeypatch.setattr(cache_module, "_VERSIONED_REGISTRY", registry)

    key_before = cache_module.lore_cache_key_with_version("player-7", "scene", "scope")
    assert key_before.endswith("v0")

    cache_module.bump_lore_cache_version("player-7")
    key_after = cache_module.lore_cache_key_with_version("player-7", "scene", "scope")

    assert key_after.endswith("v1")
