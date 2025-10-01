import asyncio

import pytest

from db import connection as db_connection


@pytest.mark.asyncio
async def test_track_operation_respects_shutdown(monkeypatch):
    """Ensure Celery shutdown path rejects new operations cleanly."""

    # Start from a clean slate for shutdown tracking.
    monkeypatch.setattr(db_connection, "_SHUTTING_DOWN", False)
    monkeypatch.setattr(db_connection, "_SHUTTING_DOWN_PIDS", set())
    monkeypatch.setattr(db_connection, "_pending_operations", set())
    monkeypatch.setattr(db_connection, "_pending_operations_lock", None)

    async def sample_operation():
        await asyncio.sleep(0)
        return "ok"

    # Normal operation should succeed.
    result = await db_connection.track_operation(sample_operation())
    assert result == "ok"

    # Simulate Celery shutdown and ensure new work is rejected.
    db_connection.mark_shutting_down()
    assert db_connection.is_shutting_down() is True

    with pytest.raises(ConnectionError):
        await db_connection.track_operation(sample_operation())

    # No pending work should remain and shutdown wait should be a no-op.
    await db_connection.wait_for_pending_operations()
