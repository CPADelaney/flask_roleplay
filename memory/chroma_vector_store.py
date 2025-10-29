# memory/chroma_vector_store.py
"""
ChromaDB Vector Store (New Clients API)

- Uses chromadb.PersistentClient(path=...) per the new architecture.
- Disables Chroma telemetry before import to avoid PostHog noise.
- Wraps all blocking Chroma calls with asyncio.to_thread.
- Safe batching, retry w/ backoff, and defensive result parsing.
- DEFAULT EMBEDDINGS: OpenAI text-embedding-3-small (1536 dims).

Note: Chroma collections do not enforce vector dimensionality themselves,
but if you use text-only queries via Chroma's embedding_function, this file
now ensures 1536-dim embeddings by default. If you pass your own embeddings
from upstream (e.g., MemoryEmbeddingService), ensure they are also 1536-dim
for consistency.
"""
from __future__ import annotations

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional

from pathlib import Path

# Disable Chroma telemetry BEFORE importing chromadb (prevents PostHog init/noise)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

import chromadb
from chromadb.config import Settings  # used to disable anonymized telemetry
from chromadb.utils import embedding_functions

# Import the abstract VectorDatabase interface used by your service layer
from context.optimized_db import VectorDatabase

logger = logging.getLogger(__name__)


class ChromaVectorDatabase(VectorDatabase):
    """ChromaDB vector database integration using the new PersistentClient."""

    _DEFAULT_COLLECTION: str = "memories"
    _DEFAULT_BATCH_SIZE: int = 128

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "memories",
        embedding_function=None,
        expected_dimension: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            persist_directory: Directory to store Chroma data (created if missing).
            collection_name: Default collection name.
            embedding_function: Optional Chroma embedding function. If omitted, this class
                                will default to OpenAI text-embedding-3-small (1536 dims).
            config: Optional dict:
                - max_retries: int (default 3)
                - retry_delay: float seconds (default 0.5)
                - batch_size: int (default 128)
        """
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.default_collection_name = collection_name or self._DEFAULT_COLLECTION
        self.embedding_function = embedding_function
        self.config = config or {}

        self.collection_dimensions: Dict[str, int] = {}

        self._configured_dimension = self._coerce_positive_int(expected_dimension)
        if self._configured_dimension is None:
            self._configured_dimension = self._extract_dimension_from_config(self.config)
        if self._configured_dimension:
            self.collection_dimensions[self.default_collection_name] = self._configured_dimension

        policy_flag = self.config.get("use_default_embedding_function")
        self._force_default_embedding: Optional[bool]
        if policy_flag is None:
            self._force_default_embedding = None
        else:
            self._force_default_embedding = bool(policy_flag)

        self.client: Optional[chromadb.PersistentClient] = None
        self.collections: Dict[str, Any] = {}

        # Ensure persistence path exists
        if self.persist_directory:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Tunables
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
            # New client style (no deprecated chroma_db_impl/persist_directory in Settings)
            settings = Settings(anonymized_telemetry=False)
            self.client = chromadb.PersistentClient(path=self.persist_directory, settings=settings)

            should_attach_default = False
            if self.embedding_function is None:
                if self._force_default_embedding is True:
                    should_attach_default = True
                elif self._force_default_embedding is False:
                    should_attach_default = False
                else:
                    # Auto mode – only attach the default if no explicit dimension was provided
                    should_attach_default = self._configured_dimension in (None, 1536)

            if should_attach_default:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        "OPENAI_API_KEY is required to use 1536-dim OpenAI embeddings "
                        "(text-embedding-3-small) as the default embedding_function."
                    )
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=api_key,
                    model_name="text-embedding-3-small",  # 1536 dimensions
                )
                # We now know the expected dimension is 1536
                self._configured_dimension = self._configured_dimension or 1536
            elif self.embedding_function is None and self._configured_dimension is not None:
                logger.info(
                    "Chroma vector store configured for externally supplied embeddings (dimension=%d)",
                    self._configured_dimension,
                )

            logger.info("Chroma PersistentClient ready at %s", self.persist_directory)
        except Exception as e:
            logger.error("Error initializing ChromaDB client: %s", e)
            raise

    async def close(self) -> None:
        """Close the ChromaDB connection (best-effort persist for older shapes)."""
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

    async def _ensure_client(self) -> None:
        if self.client is None:
            await self.initialize()

    async def _retry_sync(self, fn, *args, **kwargs):
        """
        Run a synchronous function in a thread with retry/backoff.
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

    @staticmethod
    def _normalize_where_filter(filter_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Ensure filters conform to Chroma's operator syntax."""

        if not filter_dict:
            return None

        if not isinstance(filter_dict, dict):
            return filter_dict

        def _has_operator(d: Dict[str, Any]) -> bool:
            return any(isinstance(key, str) and key.startswith("$") for key in d.keys())

        if _has_operator(filter_dict):
            return filter_dict

        simple_clauses: List[Dict[str, Any]] = []
        passthrough_clauses: List[Dict[str, Any]] = []

        for key, value in filter_dict.items():
            if isinstance(value, dict):
                passthrough_clauses.append({key: value})
            else:
                simple_clauses.append({key: {"$eq": value}})

        if not simple_clauses:
            return filter_dict

        combined = passthrough_clauses + simple_clauses

        if len(combined) == 1:
            return combined[0]

        return {"$and": combined}

    async def _get_collection(self, collection_name: str):
        """Get or create a collection (cached)."""
        await self._ensure_client()
        name = collection_name or self.default_collection_name

        cached = self.collections.get(name)
        if cached is not None:
            return cached

        metadata: Optional[Dict[str, Any]] = None
        dimension = self.collection_dimensions.get(name) or self._configured_dimension
        coerced_dimension = self._coerce_positive_int(dimension)
        if coerced_dimension:
            metadata = {"dimension": coerced_dimension}

        # New clients support get_or_create_collection — race-safe enough for single process
        collection = await self._retry_sync(
            self.client.get_or_create_collection,  # type: ignore[attr-defined]
            name=name,
            embedding_function=self.embedding_function,
            metadata=metadata or None,
        )
        self.collections[name] = collection
        return collection

    # -------------------------------------------------------------------------
    # Public API (VectorDatabase)
    # -------------------------------------------------------------------------

    async def create_collection(self, collection_name: str, dimension: int) -> bool:
        """
        Create a collection and record the expected embedding dimension if provided.
        """
        try:
            coerced_dimension = self._coerce_positive_int(dimension)
            if coerced_dimension:
                self.collection_dimensions[collection_name] = coerced_dimension
                if self._configured_dimension is None:
                    self._configured_dimension = coerced_dimension
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
            return True

        try:
            if not (len(ids) == len(vectors) == len(metadata)):
                raise ValueError("ids, vectors, and metadata lengths must match")

            collection = await self._get_collection(collection_name)
            docs = [""] * len(ids)  # since we supply embeddings directly

            bs = max(1, min(self.batch_size, len(ids)))
            for i in range(0, len(ids), bs):
                batch_ids = ids[i : i + bs]
                batch_vecs = vectors[i : i + bs]
                batch_meta = metadata[i : i + bs]
                batch_docs = docs[i : i + bs]

                await self._retry_sync(
                    collection.upsert,
                    ids=batch_ids,
                    embeddings=batch_vecs,
                    metadatas=batch_meta,
                    documents=batch_docs,
                )

            # PersistentClient generally writes through; still safe to persist if available
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
        Query similar vectors. Returns list of dicts:
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
            where = self._normalize_where_filter(filter_dict)

            results = await self._retry_sync(
                collection.query,
                query_embeddings=[query_vector],
                n_results=max(1, int(top_k)),
                where=where or None,
                include=["metadatas", "distances", "documents", "embeddings"],
            )

            # Expected new-client shape: nested lists for query results
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
                        "score": 1.0 - dist,  # distance -> similarity (metric-dependent)
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
        Fetch by IDs. Returns a list of:
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

            # Many new-client get() responses are flat lists; be defensive
            got_ids = results.get("ids") or []
            metadatas = results.get("metadatas") or []
            embeddings = results.get("embeddings") or []

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

    @staticmethod
    def _coerce_positive_int(value: Any) -> Optional[int]:
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return None
        return coerced if coerced > 0 else None

    @staticmethod
    def _extract_dimension_from_config(config: Dict[str, Any]) -> Optional[int]:
        if not isinstance(config, dict):
            return None

        candidate_keys = (
            "embedding_dimension",
            "embedding_dim",
            "dimension",
            "vector_dimension",
            "vector_dim",
            "dim",
        )

        for key in candidate_keys:
            value = config.get(key)
            coerced = ChromaVectorDatabase._coerce_positive_int(value)
            if coerced:
                return coerced

        nested_keys = (
            "vector_store",
            "embedding",
            "memory",
            "vector",
        )

        for nested in nested_keys:
            nested_config = config.get(nested)
            if isinstance(nested_config, dict):
                nested_value = ChromaVectorDatabase._extract_dimension_from_config(nested_config)
                if nested_value:
                    return nested_value

        return None
