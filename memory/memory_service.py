# memory/memory_service.py

"""Memory embedding service for hosted Agents and legacy vector stores."""

import os
import logging
from typing import Awaitable, Callable, Dict, List, Any, Optional, Sequence
from datetime import datetime
import uuid


# Import vector store implementations - dynamic for flexibility
from utils.embedding_dimensions import (
    DEFAULT_EMBEDDING_DIMENSION,
    build_zero_vector,
    get_target_embedding_dimension,
)
from rag import ask as rag_ask
from rag.vector_store import (
    get_hosted_vector_store_ids,
    hosted_vector_store_enabled,
    legacy_vector_store_enabled,
    search_hosted_vector_store,
    upsert_hosted_vector_documents,
)
from rag import vector_store as rag_vector_store

try:  # pragma: no cover - optional legacy backends
    from memory.chroma_vector_store import (  # type: ignore
        ChromaVectorDatabase,
        init_chroma_if_present_else_noop,
    )
except Exception:  # pragma: no cover
    ChromaVectorDatabase = None  # type: ignore

    def init_chroma_if_present_else_noop(*_args, **_kwargs) -> None:  # type: ignore
        return None

try:  # pragma: no cover - optional legacy backends
    from memory.faiss_vector_store import FAISSVectorDatabase  # type: ignore
except Exception:  # pragma: no cover
    FAISSVectorDatabase = None  # type: ignore

try:  # pragma: no cover - optional legacy backend
    from context.optimized_db import create_vector_database  # type: ignore
except Exception:  # pragma: no cover
    create_vector_database = None  # type: ignore

# Configure logging
logger = logging.getLogger(__name__)


def _enqueue_heavy_hydration(user_id: int, conversation_id: int) -> None:
    """Dispatch the deferred hydration task without importing Celery modules eagerly."""

    try:
        from nyx.tasks.heavy.memory_tasks import schedule_heavy_hydration
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.debug(
            "Unable to import schedule_heavy_hydration for user_id=%s conversation_id=%s: %s",
            user_id,
            conversation_id,
            exc,
        )
        return

    delay = getattr(schedule_heavy_hydration, "delay", None)
    if callable(delay):
        delay(user_id, conversation_id)
    else:  # pragma: no cover - fallback for non-Celery environments
        schedule_heavy_hydration(user_id, conversation_id)


def _extract_configured_dimension(
    config: Optional[Dict[str, Any]],
) -> Optional[int]:
    """Return an explicit embedding dimension from *config* or the environment."""

    if not isinstance(config, dict):
        config = {}

    search_maps: List[Dict[str, Any]] = [config]
    for key in ("vector_store", "embedding", "memory", "vector"):
        section = config.get(key) if isinstance(config, dict) else None
        if isinstance(section, dict):
            search_maps.append(section)

    candidate_keys = (
        "embedding_dim",
        "dimension",
        "vector_dimension",
        "vector_dim",
        "dim",
    )

    for mapping in search_maps:
        for key in candidate_keys:
            value = mapping.get(key)
            try:
                if value is None:
                    continue
                coerced = int(value)
            except (TypeError, ValueError):
                continue
            if coerced > 0:
                return coerced

    for env_key in (
        "MEMORY_EMBEDDING_DIMENSION",
        "EMBEDDING_DIMENSION",
        "DEFAULT_EMBEDDING_DIMENSION",
    ):
        try:
            value = int(os.getenv(env_key, "") or 0)
        except ValueError:
            continue
        if value > 0:
            return value

    return None

class MemoryEmbeddingService:
    """Embed and retrieve memories using hosted Agents or local vector stores."""
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        vector_store_type: str = "chroma",
        embedding_model: str = "openai",
        persist_directory: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the memory embedding service.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            vector_store_type: Type of vector store ("chroma", "faiss", or "qdrant")
            embedding_model: Embedding model type ("local" or "openai")
            persist_directory: Directory to persist vector store data
            config: Optional configuration
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.vector_store_type = vector_store_type.lower()
        self.embedding_model = embedding_model.lower()
        self.config = config or {}
        self._configured_dimension = _extract_configured_dimension(self.config)
        self._fast_mode_enabled = (
            os.getenv("MEM_FAST_MODE", "on").strip().lower() not in {"0", "false", "off"}
        )

        self._legacy_vector_store_enabled = legacy_vector_store_enabled(self.config)
        self._hosted_vector_store_ids = get_hosted_vector_store_ids(self.config)
        self._use_hosted_vector_store = hosted_vector_store_enabled(
            self._hosted_vector_store_ids,
            config=self.config,
        )
        if self._legacy_vector_store_enabled:
            self._use_hosted_vector_store = False

        agents_import_error = getattr(rag_vector_store, "_AGENTS_IMPORT_ERROR", None)
        agents_setup_missing = getattr(rag_vector_store, "agents_setup", None) is None

        if (
            not self._use_hosted_vector_store
            and not self._legacy_vector_store_enabled
            and self._hosted_vector_store_ids
            and agents_setup_missing
            and agents_import_error is not None
        ):
            logger.warning(
                "Hosted vector store disabled because Agents SDK is unavailable;"
                " falling back to legacy vector store backend."
            )
            vector_store_config = self.config.setdefault("vector_store", {})
            vector_store_config["use_legacy_vector_store"] = True
            self._legacy_vector_store_enabled = True

        if not self._use_hosted_vector_store and not self._legacy_vector_store_enabled:
            raise RuntimeError(
                "Legacy vector store backend disabled; configure Agents hosted vector "
                "stores or set ENABLE_LEGACY_VECTOR_STORE=1 to keep local embeddings."
            )

        self._primary_vector_store_id: Optional[str] = (
            self._hosted_vector_store_ids[0] if self._hosted_vector_store_ids else None
        )

        # Set persist directory for legacy vector stores only.
        if not persist_directory and not self._use_hosted_vector_store:
            persist_base = self.config.get("persist_base_dir", "./vector_stores")
            self.persist_directory = f"{persist_base}/{vector_store_type}/{user_id}_{conversation_id}"
        else:
            self.persist_directory = persist_directory
        
        # Initialize variables to be set up later
        self.vector_db = None
        self._embedding_provider: Optional[Callable[[str], Awaitable[Sequence[float]]]] = None
        self.embedding_source_dimension: Optional[int] = None
        self.target_embedding_dimension: Optional[int] = None
        self.collection_mapping = {
            "memory": "memory_embeddings",
            "npc": "npc_embeddings",
            "location": "location_embeddings",
            "narrative": "narrative_embeddings"
        }
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the embedding service."""
        if self.initialized:
            return
        
        try:
            # 1. Set up the embedding model
            await self._setup_embeddings()

            # 2. Set up the vector store when using the legacy backends
            if not self._use_hosted_vector_store:
                if (
                    self._fast_mode_enabled
                    and self.vector_store_type == "chroma"
                ):
                    init_chroma_if_present_else_noop(self.persist_directory)
                    _enqueue_heavy_hydration(self.user_id, self.conversation_id)
                else:
                    await self._setup_vector_store()

            self.initialized = True
            backend_label = "hosted" if self._use_hosted_vector_store else self.vector_store_type
            logger.info(
                "Memory embedding service initialized with %s vector backend and %s embeddings",
                backend_label,
                self.embedding_model,
            )
            
        except Exception as e:
            logger.error(f"Error initializing memory embedding service: {e}")
            raise
    
    async def _setup_embeddings(self) -> None:
        """Set up the embedding model."""
        embedding_section: Dict[str, Any] = {}
        if isinstance(self.config, dict):
            embedding_section = self.config.get("embedding") or {}

        configured_dimension = self._configured_dimension

        try:
            target_hint = int(get_target_embedding_dimension(config=self.config))
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Unable to resolve target embedding dimension: %s", exc)
            target_hint = 0

        if target_hint <= 0:
            target_hint = configured_dimension or DEFAULT_EMBEDDING_DIMENSION

        requested_dimensions = target_hint or DEFAULT_EMBEDDING_DIMENSION

        async def _call_rag_provider(
            text: str,
            *,
            operation: str,
            model_name: str,
            dimensions: Optional[int] = None,
        ) -> List[float]:
            response = await rag_ask.ask(
                text,
                mode="embedding",
                metadata={
                    "component": "memory.memory_service",
                    "operation": operation,
                    "model": model_name,
                },
                model=model_name,
                dimensions=dimensions,
            )
            vector = response.get("embedding") if isinstance(response, dict) else None
            if vector is None:
                raise ValueError("Embedding response missing vector payload")
            if hasattr(vector, "tolist"):
                vector = vector.tolist()  # type: ignore[assignment]
            return [float(value) for value in vector]

        async def _call_local_provider(text: str) -> List[float]:
            from utils import embedding_service  # local import to avoid circular deps

            vector = await embedding_service.get_embedding(text)
            if hasattr(vector, "tolist"):
                vector = vector.tolist()  # type: ignore[assignment]
            return [float(value) for value in vector]

        if self._fast_mode_enabled:
            model_name = "text-embedding-3-small"
            if isinstance(embedding_section, dict):
                candidate = embedding_section.get("openai_model") or embedding_section.get("model_name")
                if isinstance(candidate, str) and candidate.strip():
                    model_name = candidate
            if isinstance(self.config, dict):
                candidate = self.config.get("openai_model")
                if isinstance(candidate, str) and candidate.strip():
                    model_name = candidate

            async def _fast_provider(text: str) -> Sequence[float]:
                return await _call_rag_provider(
                    text,
                    operation="fast-openai-provider",
                    model_name=model_name,
                    dimensions=requested_dimensions,
                )

            self._embedding_provider = _fast_provider

        elif self.embedding_model == "openai":
            model_name = "text-embedding-3-small"
            if isinstance(embedding_section, dict):
                candidate = embedding_section.get("openai_model") or embedding_section.get("model_name")
                if isinstance(candidate, str) and candidate.strip():
                    model_name = candidate
            if isinstance(self.config, dict):
                candidate = self.config.get("openai_model")
                if isinstance(candidate, str) and candidate.strip():
                    model_name = candidate

            async def _openai_provider(text: str) -> Sequence[float]:
                return await _call_rag_provider(
                    text,
                    operation="openai-provider",
                    model_name=model_name,
                    dimensions=requested_dimensions,
                )

            self._embedding_provider = _openai_provider
        else:
            async def _local_provider(text: str) -> Sequence[float]:
                return await _call_local_provider(text)

            self._embedding_provider = _local_provider

        if self._embedding_provider is None:
            raise RuntimeError("Failed to configure embedding provider")

        try:
            probe_vector = await self._embedding_provider("embedding-dimension-probe")
            probe_list = [float(value) for value in probe_vector]
            self.embedding_source_dimension = len(probe_list)
        except Exception as exc:
            self.embedding_source_dimension = None
            logger.warning(
                "Unable to probe embedding dimension; falling back to configuration. Error: %s",
                exc,
            )

        if self.embedding_source_dimension is not None:
            if (
                configured_dimension is not None
                and configured_dimension != self.embedding_source_dimension
            ):
                logger.info(
                    "Embedding model produced %d dimensions; overriding configured %d to match native size.",
                    self.embedding_source_dimension,
                    configured_dimension,
                )
            self.target_embedding_dimension = self.embedding_source_dimension
        elif configured_dimension is not None:
            self.target_embedding_dimension = configured_dimension
        else:
            self.target_embedding_dimension = target_hint or DEFAULT_EMBEDDING_DIMENSION

        self.embedding_dimension = self.target_embedding_dimension

    async def _setup_vector_store(self) -> None:
        """Set up the vector store."""
        if self._use_hosted_vector_store:
            self.vector_db = None
            return

        if self.vector_store_type == "chroma":
            if ChromaVectorDatabase is None:
                raise RuntimeError("Chroma vector store backend is not available")
            self.vector_db = ChromaVectorDatabase(
                persist_directory=self.persist_directory,
                expected_dimension=self._get_target_dimension(),
                config=self.config,
            )
        elif self.vector_store_type == "faiss":
            if FAISSVectorDatabase is None:
                raise RuntimeError("FAISS vector store backend is not available")
            self.vector_db = FAISSVectorDatabase(
                persist_directory=self.persist_directory,
                config=self.config
            )
        elif self.vector_store_type == "qdrant":
            # Use the existing Qdrant implementation from your codebase
            vector_db_config = {
                "db_type": "qdrant",
                "url": self.config.get("qdrant_url", os.getenv("QDRANT_URL", "http://localhost:6333")),
                "api_key": self.config.get("qdrant_api_key", os.getenv("QDRANT_API_KEY"))
            }
            if create_vector_database is None:
                raise RuntimeError("Qdrant vector store backend is not available")

            self.vector_db = create_vector_database(vector_db_config)
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")

        # Initialize the vector database
        await self.vector_db.initialize()

        # Ensure collections exist
        for collection_name in self.collection_mapping.values():
            await self.vector_db.create_collection(
                collection_name, self._get_target_dimension()
            )

    async def hydrate_legacy_vector_store(self) -> str:
        """Ensure the legacy vector store is fully initialised for this session."""

        if self._use_hosted_vector_store:
            logger.info(
                "Hosted Agents vector store active; skipping legacy hydration for user_id=%s conversation_id=%s",
                self.user_id,
                self.conversation_id,
            )
            return "skipped:hosted-vector-store"

        if not self._legacy_vector_store_enabled:
            logger.info(
                "Legacy vector store disabled; skipping hydration for user_id=%s conversation_id=%s",
                self.user_id,
                self.conversation_id,
            )
            return "skipped:legacy-disabled"

        try:
            await self._setup_embeddings()

            if self.vector_store_type == "chroma":
                init_chroma_if_present_else_noop(self.persist_directory)

            await self._setup_vector_store()

            self.initialized = True

            logger.info(
                "Legacy vector store hydrated for user_id=%s conversation_id=%s",
                self.user_id,
                self.conversation_id,
            )
            return "hydrated"
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error(
                "Legacy vector store hydration failed for user_id=%s conversation_id=%s: %s",
                self.user_id,
                self.conversation_id,
                exc,
            )
            raise

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector
        """
        if not self.initialized:
            await self.initialize()

        try:
            if not self._embedding_provider:
                raise RuntimeError("Embedding provider not configured")

            embedding = await self._embedding_provider(text)

            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            if not isinstance(embedding, Sequence):
                embedding = list(embedding)
            embedding = [float(value) for value in embedding]

            return self._normalise_embedding_length(embedding)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return an empty vector as fallback
            return self._get_empty_embedding()

    def _get_empty_embedding(self) -> List[float]:
        """Get an empty embedding vector with the correct dimension."""
        return build_zero_vector(self._get_target_dimension())

    def _normalise_embedding_length(self, values: Sequence[float]) -> List[float]:
        target = self._get_target_dimension()
        data = list(values)
        if len(data) == target:
            return data
        if len(data) > target:
            logger.debug(
                "Truncating embedding from %d to %d dimensions", len(data), target
            )
            return data[:target]
        logger.debug(
            "Padding embedding from %d to %d dimensions", len(data), target
        )
        return data + [0.0] * (target - len(data))
    
    def _coerce_embedding_input(
        self,
        embedding: Sequence[float],
    ) -> Optional[List[float]]:
        """Convert various embedding containers into a float list."""
        try:
            if hasattr(embedding, "tolist"):
                candidate = embedding.tolist()  # type: ignore[assignment]
            else:
                candidate = list(embedding)
        except TypeError as exc:
            logger.warning("Provided embedding could not be iterated over; regenerating", exc_info=exc)
            return None

        coerced: List[float] = []
        try:
            for value in candidate:
                coerced.append(float(value))
        except (TypeError, ValueError) as exc:
            logger.warning("Provided embedding contained non-numeric values; regenerating", exc_info=exc)
            return None

        return coerced

    async def add_memory(
        self,
        text: str,
        metadata: Dict[str, Any],
        entity_type: str = "memory",
        embedding: Optional[Sequence[float]] = None
    ) -> str:
        """
        Add a memory to the vector store.
        
        Args:
            text: Memory text content
            metadata: Memory metadata
            entity_type: Entity type (memory, npc, location, narrative)
            embedding: Optional pre-computed embedding
            
        Returns:
            Memory ID
        """
        if not self.initialized:
            await self.initialize()
        
        # Generate memory ID if not in metadata
        memory_id = metadata.get("memory_id", f"{entity_type}_{uuid.uuid4()}")
        metadata["memory_id"] = memory_id
        
        # Add user and conversation ID
        metadata.update({
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "entity_type": entity_type,
            "timestamp": datetime.now().isoformat(),
            "content": text
        })

        # Get collection name
        collection_name = self.collection_mapping.get(entity_type, "memory_embeddings")

        if self._use_hosted_vector_store:
            hosted_metadata = dict(metadata)
            hosted_metadata["collection"] = collection_name

            await upsert_hosted_vector_documents(
                [
                    {
                        "id": memory_id,
                        "text": text,
                        "metadata": hosted_metadata,
                    }
                ],
                vector_store_id=self._primary_vector_store_id,
                metadata={
                    "component": "memory.memory_service",
                    "operation": "insert",
                    "user_id": self.user_id,
                    "conversation_id": self.conversation_id,
                },
            )
            return memory_id

        # Generate embedding if not provided
        if embedding is None:
            embedding_vector = await self.generate_embedding(text)
        else:
            coerced_embedding = self._coerce_embedding_input(embedding)
            if coerced_embedding is None:
                embedding_vector = await self.generate_embedding(text)
            else:
                expected_dim = self._get_target_dimension()
                if len(coerced_embedding) != expected_dim:
                    raise ValueError(
                        f"Provided embedding dimension {len(coerced_embedding)} does not match expected {expected_dim}"
                    )
                embedding_vector = coerced_embedding

        # Add to vector store
        success = await self.vector_db.insert_vectors(
            collection_name=collection_name,
            ids=[memory_id],
            vectors=[embedding_vector],
            metadata=[metadata]
        )
        
        if not success:
            logger.error(f"Failed to add memory {memory_id} to vector store")
        
        return memory_id
    
    async def search_memories(
        self,
        query_text: str,
        entity_type: Optional[str] = None,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        fetch_content: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant memories.
        
        Args:
            query_text: Query text
            entity_type: Optional entity type filter
            top_k: Number of results to return
            filter_dict: Optional additional filters
            fetch_content: Whether to include the full content in results
            
        Returns:
            List of relevant memories
        """
        if not self.initialized:
            await self.initialize()
        
        # Apply user and conversation filters
        base_filter = {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        }
        
        # Merge with provided filters
        if filter_dict:
            base_filter.update(filter_dict)
            
        # Add entity type if provided
        if entity_type:
            base_filter["entity_type"] = entity_type
            collection_name = self.collection_mapping.get(entity_type, "memory_embeddings")
        else:
            collection_name = "memory_embeddings"  # Default collection
        
        if self._use_hosted_vector_store:
            hosted_filters = dict(base_filter)
            hosted_filters["collection"] = collection_name
            hosted_results = await search_hosted_vector_store(
                query_text,
                vector_store_ids=self._hosted_vector_store_ids or None,
                max_results=top_k,
                attributes=hosted_filters,
                metadata={
                    "component": "memory.memory_service",
                    "operation": "search",
                    "entity_type": entity_type or "*",
                    "collection": collection_name,
                },
            )

            results: List[Dict[str, Any]] = []
            for record in hosted_results:
                payload = dict(record.get("metadata") or {})
                memory_id = payload.get("memory_id") or record.get("id")
                memory_data = {
                    "id": memory_id,
                    "relevance": record.get("score") or 0.0,
                    "metadata": {},
                }

                for key, value in payload.items():
                    if key == "content" and not fetch_content:
                        continue
                    memory_data["metadata"][key] = value

                if fetch_content:
                    if "content" in payload:
                        memory_data["memory_text"] = payload["content"]
                    elif record.get("text"):
                        memory_data["memory_text"] = record["text"]

                results.append(memory_data)

            return results

        # Generate query embedding
        query_embedding = await self.generate_embedding(query_text)

        # Search vector store
        search_results = await self.vector_db.search_vectors(
            collection_name=collection_name,
            query_vector=query_embedding,
            top_k=top_k,
            filter_dict=base_filter
        )
        
        # Format results
        results = []
        for result in search_results:
            memory_data = {
                "id": result["id"],
                "relevance": result["score"],
                "metadata": {}
            }
            
            # Add metadata
            for key, value in result["metadata"].items():
                if key != "content" or fetch_content:
                    memory_data["metadata"][key] = value
            
            # Add memory_text field for convenience
            if "content" in result["metadata"] and fetch_content:
                memory_data["memory_text"] = result["metadata"]["content"]
            
            results.append(memory_data)
        
        return results
    
    async def get_memory_by_id(
        self,
        memory_id: str,
        entity_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a memory by its ID.
        
        Args:
            memory_id: Memory ID
            entity_type: Optional entity type
            
        Returns:
            Memory data or None if not found
        """
        if not self.initialized:
            await self.initialize()
        
        if self._use_hosted_vector_store:
            attributes = {
                "user_id": self.user_id,
                "conversation_id": self.conversation_id,
                "memory_id": memory_id,
            }
            if entity_type:
                attributes["entity_type"] = entity_type
                attributes["collection"] = self.collection_mapping.get(entity_type, "memory_embeddings")
            else:
                attributes["collection"] = "memory_embeddings"

            hosted_result = await search_hosted_vector_store(
                str(memory_id),
                vector_store_ids=self._hosted_vector_store_ids or None,
                max_results=1,
                attributes=attributes,
                metadata={
                    "component": "memory.memory_service",
                    "operation": "get-by-id",
                    "entity_type": entity_type or "*",
                },
            )

            if not hosted_result:
                return None

            record = hosted_result[0]
            payload = dict(record.get("metadata") or {})
            payload.setdefault("memory_id", memory_id)
            memory_data = {
                "id": payload.get("memory_id") or record.get("id"),
                "metadata": payload,
            }

            if "content" in payload:
                memory_data["memory_text"] = payload["content"]
            elif record.get("text"):
                memory_data["memory_text"] = record["text"]

            return memory_data

        if entity_type:
            collection_name = self.collection_mapping.get(entity_type, "memory_embeddings")
        else:
            collection_name = "memory_embeddings"

        # Get from vector store
        results = await self.vector_db.get_by_id(
            collection_name=collection_name,
            ids=[memory_id]
        )

        if not results:
            return None

        # Format result
        memory_data = {
            "id": results[0]["id"],
            "metadata": results[0]["metadata"],
        }

        # Add memory_text field for convenience
        if "content" in results[0]["metadata"]:
            memory_data["memory_text"] = results[0]["metadata"]["content"]

        return memory_data
    
    async def close(self) -> None:
        """Close the memory embedding service."""
        if self.vector_db:
            await self.vector_db.close()

        self.initialized = False

    def _get_target_dimension(self) -> int:
        if self.target_embedding_dimension is not None:
            return self.target_embedding_dimension

        if self.embedding_source_dimension is not None:
            self.target_embedding_dimension = self.embedding_source_dimension
        elif self._configured_dimension is not None:
            self.target_embedding_dimension = self._configured_dimension
        else:
            self.target_embedding_dimension = DEFAULT_EMBEDDING_DIMENSION

        return self.target_embedding_dimension
