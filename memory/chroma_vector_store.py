# memory/chroma_vector_store.py
"""
ChromaDB Vector Store (New Clients API)

- Uses chromadb.PersistentClient(path=...) per the new architecture.
- No legacy Settings(chroma_db_impl=..., persist_directory=...) usage.
- All blocking Chroma calls are wrapped with asyncio.to_thread.
- Safe batching, robust result-shape handling, and gentle persistence on close.

Compatible with your VectorDatabase interface expected by MemoryEmbeddingService.
"""

from __future__ import annotations

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple

from pathlib import Path
import chromadb
from chromadb.config import Settings  # still OK to pass telemetry flag only
from chromadb.utils import embedding_functions

# Import the abstract VectorDatabase class
from context.optimized_db import VectorDatabase

logger = logging.getLogger(__name__)


class ChromaVectorDatabase(VectorDatabase):
    """ChromaDB vector database integration using the new PersistentClient."""

    # sane defaults
    _DEFAULT_COLLECTION: str = "memories"
    _DEFAULT_BATCH_SIZE: int = 128

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "memories",
        embedding_function=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            persist_directory: Directory to store Chroma data (created if missing).
            collection_name: Default collection name to fall back on.
            embedding_function: Optional chromadb embedding function (only used if you query by text).
            config: Optional config dict:
                - max_retries (int, default 3)
                - retry_delay (float seconds, default 0.5)
                - batch_size (int, default 128)
        """
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.default_collection_name = collection_name or self._DEFAULT_COLLECTION
        self.embedding_function = embedding_function
        self.config = config or {}

        self.client: Optional[chromadb.PersistentClient] = None
        self.collections: Dict[str, Any] = {}

        # Create persist directory if it doesn't exist
        if self.persist_directory:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Retry & batching
        self.max_retries: int = int(self.config.get("max_retries", 3))
        self.retry_delay: float = float(self.config.get("retry_delay", 0.5))
        self.batch_size: int = int(self.config.get("batch_size", self._DEFAULT_BATCH_SIZE))

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize the ChromaDB client (PersistentClient)."""
        if self.client is not None:
            return
        try:
            # Only disable telemetry; do NOT pass legacy db impl or persist dir here.
            client_settings = Settings(anonymized_telemetry=False)

            # New client API
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=client_settings,
            )

            # If you never query by text, this isn't required; harmless to set.
            if not self.embedding_function:
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )

            logger.info("Chroma PersistentClient ready at %s", self.persist_directory)

        except Exception as e:
            logger.error("Error initializing ChromaDB client: %s", e)
            raise

    async def close(self) -> None:
        """Close the ChromaDB connection (best-effort persist for older client shapes)."""
        if self.client:
            try:
                if hasattr(self.client, "persist"):
                    await asyncio.to_thread(self.client.persist)
            except Exception:
                logger.debug("Chroma persist() on close failed", exc_info=True)
            finally:
                self.client = None
                self.collections.clear()

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    async def _retry_sync(self, fn, *args, **kwargs):
        """
        Run a synchronous function in a thread with retry & backoff.
        Returns fn(*args, **kwargs) result or raises last exception.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                return await asyncio.to_thread(fn, *args, **kwargs)
            except Exception as e:
                if attempt >= self.max_retries:
                    raise
                sleep_for = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Chroma op failed (attempt %d/%d): %s; retrying in %.2fs",
                    attempt, self.max_retries, e, sleep_for
                )
                await asyncio.sleep(sleep_for)

    async def _get_collection(self, collection_name: str):
        """Get or create a collection (cached; race-safe enough for single-process)."""
        await self._ensure_client()

        name = collection_name or self.default_collection_name
        if name in self.collections:
            return self.collections[name]

        # New Clients support get_or_create_collection
        collection = await self._retry_sync(
            self.client.get_or_create_collection,  # type: ignore[attr-defined]
            name=name,
            embedding_function=self.embedding_function,
        )
        self.collections[name] = collection
        return collection

    async def _ensure_client(self):
        if self.client is None:
            await self.initialize()

    # -------------------------------------------------------------------------
    # Public API (VectorDatabase)
    # -------------------------------------------------------------------------

    async def create_collection(self, collection_name: str, dimension: int) -> bool:
        """
        Create a new collection in ChromaDB. (dimension is ignored by Chroma;
        it’s kept for interface compatibility with other backends.)
        """
        try:
            await self._get_collection(collection_name)
            return True
        except Exception as e:
            logger.error("Error creating ChromaDB collection %s: %s", collection_name, e)
            return False

    async def insert_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> bool:
        """
        Upsert vectors into a collection. Assumes ids, vectors, metadata are aligned.
        """
        if not ids:
            return True  # nothing to do

        try:
            if not (len(ids) == len(vectors) == len(metadata)):
                raise ValueError("ids, vectors, and metadata lengths must match")

            collection = await self._get_collection(collection_name)
            documents = [""] * len(ids)  # since we provide embeddings directly

            bs = max(1, min(self.batch_size, len(ids)))
            for i in range(0, len(ids), bs):
                batch_ids = ids[i : i + bs]
                batch_vectors = vectors[i : i + bs]
                batch_metadata = metadata[i : i + bs]
                batch_docs = documents[i : i + bs]

                await self._retry_sync(
                    collection.upsert,
                    ids=batch_ids,
                    embeddings=batch_vectors,
                    metadatas=batch_metadata,
                    documents=batch_docs,
                )

            # Persist (PersistentClient generally writes through, but this is safe)
            if self.client and hasattr(self.client, "persist"):
                await asyncio.to_thread(self.client.persist)

            return True

        except Exception as e:
            logger.error("Error inserting vectors into ChromaDB: %s", e)
            return False

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query most similar vectors. Returns a list of dicts:
        {
            "id": str,
            "score": float,  # similarity (1 - distance)
            "metadata": dict,
            "embedding": Optional[List[float]]
        }
        """
        if not query_vector:
            return []

        try:
            collection = await self._get_collection(collection_name)
            where = dict(filter_dict or {})

            results = await self._retry_sync(
                collection.query,
                query_embeddings=[query_vector],
                n_results=max(1, int(top_k)),
                where=where or None,
                include=["metadatas", "distances", "documents", "embeddings"],
            )

            # Expected shape (new clients):
            # {
            #   "ids": [[...]],
            #   "distances": [[...]],
            #   "metadatas": [[...]],
            #   "embeddings": [[...]]  (if requested)
            # }
            formatted: List[Dict[str, Any]] = []
            ids_nested = results.get("ids") or [[]]
            if not ids_nested or not ids_nested[0]:
                return formatted

            ids = ids_nested[0]
            distances = (results.get("distances") or [[]])[0] if results.get("distances") else []
            metadatas = (results.get("metadatas") or [[]])[0] if results.get("metadatas") else []
            embeddings = (results.get("embeddings") or [[]])[0] if results.get("embeddings") else []

            for idx, _id in enumerate(ids):
                dist = float(distances[idx]) if idx < len(distances) else 0.0
                meta = metadatas[idx] if idx < len(metadatas) else {}
                emb = embeddings[idx] if idx < len(embeddings) else None
                formatted.append(
                    {
                        "id": _id,
                        "score": 1.0 - dist,  # convert distance→similarity (approx; metric-dependent)
                        "metadata": meta or {},
                        "embedding": emb,
                    }
                )

            return formatted

        except Exception as e:
            logger.error("Error searching vectors in ChromaDB: %s", e)
            return []

    async def get_by_id(self, collection_name: str, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch documents by IDs. Returns a list of:
        { "id": str, "metadata": dict, "embedding": Optional[List[float]] }
        """
        if not ids:
            return []

        try:
            collection = await self._get_collection(collection_name)
            results = await self._retry_sync(
                collection.get,
                ids=ids,
                include=["metadatas", "documents", "embeddings"],
            )

            # New clients often return flat lists for get()
            got_ids = results.get("ids") or []
            metadatas = results.get("metadatas") or []
            embeddings = results.get("embeddings") or []

            # But be defensive in case a nested shape appears
            if got_ids and isinstance(got_ids[0], list):
                got_ids = got_ids[0]
            if metadatas and isinstance(metadatas[0], list):
                metadatas = metadatas[0]
            if embeddings and isinstance(embeddings[0], list) and len(embeddings) == 1:
                embeddings = embeddings[0]

            out: List[Dict[str, Any]] = []
            for i, _id in enumerate(got_ids):
                meta = metadatas[i] if i < len(metadatas) else {}
                emb = embeddings[i] if i < len(embeddings) else None
                out.append({"id": _id, "metadata": meta or {}, "embedding": emb})
            return out

        except Exception as e:
            logger.error("Error getting vectors by ID from ChromaDB: %s", e)
            return []
