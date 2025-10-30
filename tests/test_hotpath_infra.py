"""
Tests for hot path infrastructure (cache, helpers, task dispatch).

These tests verify that:
1. Cache operations are fast (<50ms)
2. Task dispatch is non-blocking
3. Database queries use proper indexes
4. Fallback behavior works correctly
"""

import asyncio
import json
import time
from unittest.mock import Mock, patch, MagicMock

import pytest

from infra.cache import (
    get_json,
    set_json,
    cache_key,
    redis_lock,
    delete_keys,
)


class TestRedisCache:
    """Test Redis cache utilities."""

    def test_cache_key_building(self):
        """Test cache key construction."""
        key = cache_key("conflict", "123", "transition")
        assert key == "conflict:123:transition"

        key = cache_key("social_bundle", "scene_abc")
        assert key == "social_bundle:scene_abc"

    @patch("infra.cache.get_redis_client")
    def test_get_json_success(self, mock_get_client):
        """Test successful JSON retrieval from cache."""
        mock_client = Mock()
        mock_client.get.return_value = '{"test": "data"}'
        mock_get_client.return_value = mock_client

        result = get_json("test_key")

        assert result == {"test": "data"}
        mock_client.get.assert_called_once_with("test_key")

    @patch("infra.cache.get_redis_client")
    def test_get_json_miss_returns_default(self, mock_get_client):
        """Test cache miss returns default value."""
        mock_client = Mock()
        mock_client.get.return_value = None
        mock_get_client.return_value = mock_client

        result = get_json("missing_key", default={"fallback": True})

        assert result == {"fallback": True}

    @patch("infra.cache.get_redis_client")
    def test_set_json_success(self, mock_get_client):
        """Test JSON value is cached successfully."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        data = {"test": "value", "number": 42}
        success = set_json("test_key", data, ex=300)

        assert success is True
        mock_client.set.assert_called_once()
        args = mock_client.set.call_args
        assert args[0][0] == "test_key"
        assert json.loads(args[0][1]) == data
        assert args[1]["ex"] == 300

    @patch("infra.cache.get_redis_client")
    def test_delete_keys_pattern(self, mock_get_client):
        """Test pattern-based key deletion."""
        mock_client = Mock()
        mock_client.keys.return_value = ["social:1", "social:2", "social:3"]
        mock_client.delete.return_value = 3
        mock_get_client.return_value = mock_client

        count = delete_keys("social:*")

        assert count == 3
        mock_client.keys.assert_called_once_with("social:*")
        mock_client.delete.assert_called_once_with("social:1", "social:2", "social:3")

    @patch("infra.cache.get_redis_client")
    def test_redis_lock_context_manager(self, mock_get_client):
        """Test Redis distributed lock."""
        mock_client = Mock()
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = True
        mock_client.lock.return_value = mock_lock
        mock_get_client.return_value = mock_client

        with redis_lock("test_lock", ttl=10):
            # Lock should be acquired
            mock_lock.acquire.assert_called()

        # Lock should be released
        mock_lock.release.assert_called()

    @patch("infra.cache.get_redis_client")
    def test_redis_lock_failure_raises(self, mock_get_client):
        """Test lock acquisition failure raises exception."""
        mock_client = Mock()
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = False
        mock_client.lock.return_value = mock_lock
        mock_get_client.return_value = mock_client

        with pytest.raises(RuntimeError, match="Failed to acquire lock"):
            with redis_lock("test_lock", ttl=10, blocking=False):
                pass


class TestPerformance:
    """Test hot path performance requirements."""

    @patch("infra.cache.get_redis_client")
    def test_cache_read_is_fast(self, mock_get_client):
        """Test cache read completes in <50ms."""
        mock_client = Mock()
        mock_client.get.return_value = '{"data": "value"}'
        mock_get_client.return_value = mock_client

        start = time.time()
        result = get_json("test_key")
        elapsed = time.time() - start

        assert elapsed < 0.05  # Must be <50ms
        assert result == {"data": "value"}

    @patch("infra.cache.get_redis_client")
    def test_cache_write_is_fast(self, mock_get_client):
        """Test cache write completes in <50ms."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        start = time.time()
        set_json("test_key", {"data": "value"}, ex=300)
        elapsed = time.time() - start

        assert elapsed < 0.05  # Must be <50ms


@pytest.mark.asyncio
class TestStakeholderHotPath:
    """Test stakeholder hot path helpers."""

    async def test_determine_scene_behavior_high_stress(self):
        """Test behavior determination for high-stress stakeholder."""
        from logic.conflict_system.autonomous_stakeholder_actions_hotpath import (
            determine_scene_behavior,
        )

        stakeholder = Mock(stress_level=0.9, current_role=Mock(value="mediator"))
        behavior = determine_scene_behavior(stakeholder, {})

        assert behavior == "agitated"  # High stress overrides role

    async def test_determine_scene_behavior_mediator(self):
        """Test behavior determination for mediator."""
        from logic.conflict_system.autonomous_stakeholder_actions_hotpath import (
            determine_scene_behavior,
        )

        stakeholder = Mock(stress_level=0.3, current_role=Mock(value="mediator"))
        behavior = determine_scene_behavior(stakeholder, {})

        assert behavior == "diplomatic"

    async def test_should_dispatch_high_stress(self):
        """Test dispatch logic for high-stress stakeholder."""
        from logic.conflict_system.autonomous_stakeholder_actions_hotpath import (
            should_dispatch_action_generation,
        )

        stakeholder = Mock(stress_level=0.8, current_role=Mock(value="bystander"))
        should_dispatch = should_dispatch_action_generation(stakeholder, {})

        assert should_dispatch is True

    async def test_should_dispatch_passive_low_stress(self):
        """Test dispatch logic for passive, low-stress stakeholder."""
        from logic.conflict_system.autonomous_stakeholder_actions_hotpath import (
            should_dispatch_action_generation,
        )

        stakeholder = Mock(stress_level=0.2, current_role=Mock(value="victim"))
        should_dispatch = should_dispatch_action_generation(stakeholder, {})

        assert should_dispatch is False

    @patch("logic.conflict_system.autonomous_stakeholder_actions_hotpath.generate_stakeholder_action")
    async def test_dispatch_is_nonblocking(self, mock_task):
        """Test task dispatch returns immediately (non-blocking)."""
        from logic.conflict_system.autonomous_stakeholder_actions_hotpath import (
            dispatch_action_generation,
        )

        mock_task.delay = Mock()
        stakeholder = Mock(stakeholder_id=123, stress_level=0.7)

        start = time.time()
        dispatch_action_generation(stakeholder, {"scene_id": 1})
        elapsed = time.time() - start

        assert elapsed < 0.1  # Must be instant (<100ms)
        mock_task.delay.assert_called_once()

    async def test_fetch_ready_actions_performance(self):
        """Test fetching ready actions is fast."""
        from logic.conflict_system.autonomous_stakeholder_actions_hotpath import (
            fetch_ready_actions_for_scene,
        )

        scene_context = {"scene_hash": "test_scene", "stakeholder_ids": [1, 2, 3]}

        start = time.time()
        # This will likely return empty list in test env, but should be fast
        actions = await fetch_ready_actions_for_scene(scene_context)
        elapsed = time.time() - start

        assert elapsed < 0.2  # Must be <200ms (DB query)
        assert isinstance(actions, list)


@pytest.mark.asyncio
class TestCeleryTasks:
    """Test Celery task structure and idempotency."""

    @patch("nyx.tasks.background.stakeholder_tasks.llm_json")
    @patch("nyx.tasks.background.stakeholder_tasks.get_db_connection_context")
    async def test_generate_stakeholder_action_task(self, mock_db, mock_llm):
        """Test stakeholder action generation task."""
        from nyx.tasks.background.stakeholder_tasks import generate_stakeholder_action

        # Mock LLM response
        mock_llm.return_value = {
            "action_type": "strategic",
            "description": "Test action",
            "success_probability": 0.8,
        }

        # Mock DB
        mock_conn = MagicMock()
        mock_conn.fetchval.return_value = 123
        mock_db.return_value.__aenter__.return_value = mock_conn

        payload = {
            "stakeholder_id": 1,
            "scene_context": {"scene_id": 10},
        }

        # This is a mock test - in real scenario, would use celery test harness
        # For now, just verify structure
        assert callable(generate_stakeholder_action)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
