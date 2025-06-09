"""Buffer for accumulating reward events with basic concurrency safety."""

import asyncio
from typing import List


class RewardBuffer:
    """Thread-safe buffer for reward events."""

    def __init__(self, batch_size: int = 50):
        self.batch_size = batch_size
        self._events: List[str] = []
        self._lock = asyncio.Lock()

    async def add_event(self, event_type: str) -> None:
        """Add a new event to the buffer."""
        async with self._lock:
            self._events.append(event_type)

    async def next_batch(self) -> List[str]:
        """Retrieve the next batch of events, clearing them from the buffer."""
        async with self._lock:
            if not self._events:
                return []
            batch = self._events[: self.batch_size]
            self._events = self._events[self.batch_size :]
            return batch

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._events)
