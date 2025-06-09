"""Lightweight wrapper around the vector store used by nyx.core."""
from typing import List, Dict

from . import vector_store


class MemoryManager:
    """Interface for storing and retrieving memories."""

    @staticmethod
    async def add(text: str, meta: Dict) -> str:
        return await vector_store.add(text, meta)

    @staticmethod
    async def fetch_relevant(ctx: str, k: int = 5) -> List[Dict]:
        """Return top-k relevant memories for the given query text."""
        return await vector_store.query(ctx, k=k)

