# context/optimized_db.py

import logging
import json
import time
from typing import Dict, List, Any, Optional, Union
import numpy as np
from datetime import datetime
import hashlib
import uuid
import os

# Try to import vector database client
try:
    import qdrant_client
    from qdrant_client.http import models as qmodels
    from qdrant_client.http.exceptions import UnexpectedResponse
    HAVE_VECTOR_DB = True
except ImportError:
    HAVE_VECTOR_DB = False

# Import Render configuration
from context.context_config import get_render_config, configure_render_vector_db

# Set up logging
logger = logging.getLogger(__name__)

class VectorDatabase:
    """Abstract interface for vector database operations"""
    
    async def initialize(self) -> None:
        """Initialize the vector database connection"""
        raise NotImplementedError("Subclasses must implement initialize")
    
    async def close(self) -> None:
        """Close the vector database connection"""
        raise NotImplementedError("Subclasses must implement close")
    
    async def create_collection(self, collection_name: str, dimension: int) -> bool:
        """Create a new collection/index"""
        raise NotImplementedError("Subclasses must implement create_collection")
    
    async def insert_vectors(
        self, 
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """Insert vectors into the database"""
        raise NotImplementedError("Subclasses must implement insert_vectors")
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        raise NotImplementedError("Subclasses must implement search_vectors")
    
    async def get_by_id(
        self,
        collection_name: str,
        ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Get vectors by their IDs"""
        raise NotImplementedError("Subclasses must implement get_by_id")


class QdrantDatabase(VectorDatabase):
    """Qdrant vector database integration with Render optimizations"""
    
    def __init__(
        self, 
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        if not HAVE_VECTOR_DB:
            raise ImportError("qdrant_client is required for QdrantDatabase")
        
        # Get Render configuration
        self.render_config = config or configure_render_vector_db()
        self.vector_config = self.render_config['vector_db']
        
        # Set connection parameters
        self.url = url or self.vector_config['url']
        self.api_key = api_key or self.vector_config['api_key']
        self.client = None
        self.is_render = self.render_config['is_render']
        
        # Connection settings
        self.connection_timeout = self.vector_config['connection_timeout']
        self.operation_timeout = self.vector_config['operation_timeout']
        self.max_retries = self.vector_config['max_retries']
        self.retry_delay = self.vector_config['retry_delay']
    
    async def initialize(self) -> None:
        """Initialize the Qdrant client with Render optimizations"""
        try:
            # Configure client with optimized settings for Render
            client_config = {
                "url": self.url,
                "api_key": self.api_key,
                "prefer_grpc": self.vector_config.get('prefer_grpc', True),
                "timeout": self.connection_timeout
            }
            
            # Add Render-specific optimizations
            if self.is_render:
                client_config.update({
                    "pool_size": self.vector_config['pool_size'],
                    "max_connections": self.vector_config['max_connections']
                })
            
            self.client = qdrant_client.QdrantClient(**client_config)
            
            # Test connection
            self.client.get_collections()
            logger.info("Successfully connected to Qdrant")
            
        except Exception as e:
            logger.error(f"Error initializing Qdrant client: {e}")
            raise
    
    async def close(self) -> None:
        """Close the Qdrant connection"""
        if self.client:
            await self.client.close()
            self.client = None
    
    async def _retry_operation(self, operation, *args, **kwargs):
        """Retry operation with exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                return await operation(*args, **kwargs)
            except UnexpectedResponse as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Retrying operation after error: {e}. Attempt {attempt + 1}/{self.max_retries}")
                time.sleep(wait_time)
    
    async def create_collection(self, collection_name: str, dimension: int) -> bool:
        """Create a new collection in Qdrant with optimized settings"""
        try:
            # Check if collection exists
            collections = await self._retry_operation(self.client.get_collections)
            if collection_name in [c.name for c in collections.collections]:
                logger.info(f"Collection {collection_name} already exists")
                return True
            
            # Optimize collection settings for Render
            vector_params = qmodels.VectorParams(
                size=dimension,
                distance=qmodels.Distance.COSINE
            )
            
            # Add optimizations for Render
            if self.is_render:
                vector_params.on_disk = True  # Store vectors on disk to save memory
                
                # Configure optimized index
                vector_params.hnsw_config = qmodels.HnswConfigDiff(
                    m=16,  # Number of edges per node
                    ef_construct=100,  # Size of the dynamic candidate list
                    full_scan_threshold=10000  # When to switch to full scan
                )
            
            # Create collection
            await self._retry_operation(
                self.client.create_collection,
                collection_name=collection_name,
                vectors_config=vector_params,
                optimizers_config=qmodels.OptimizersConfigDiff(
                    default_segment_number=2,
                    memmap_threshold=10000
                ) if self.is_render else None
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating Qdrant collection: {e}")
            return False
    
    async def insert_vectors(
        self, 
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """Insert vectors into Qdrant with batching and retry logic"""
        try:
            # Convert string IDs to UUID
            uuid_ids = [uuid.uuid5(uuid.NAMESPACE_DNS, id) for id in ids]
            
            # Create points
            points = []
            for i in range(len(ids)):
                points.append(
                    qmodels.PointStruct(
                        id=uuid_ids[i],
                        vector=vectors[i],
                        payload=metadata[i]
                    )
                )
            
            # Insert in optimized batches
            batch_size = 100 if self.is_render else 1000
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                await self._retry_operation(
                    self.client.upsert,
                    collection_name=collection_name,
                    points=batch,
                    wait=True
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error inserting vectors into Qdrant: {e}")
            return False
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Qdrant with optimized settings"""
        try:
            # Convert filter dictionary to Qdrant filter
            filter_obj = None
            if filter_dict:
                filter_conditions = []
                for key, value in filter_dict.items():
                    if isinstance(value, list):
                        filter_conditions.append(
                            qmodels.FieldCondition(
                                key=key,
                                match=qmodels.MatchAny(any=value)
                            )
                        )
                    else:
                        filter_conditions.append(
                            qmodels.FieldCondition(
                                key=key,
                                match=qmodels.MatchValue(value=value)
                            )
                        )
                
                if filter_conditions:
                    filter_obj = qmodels.Filter(must=filter_conditions)
            
            # Optimize search parameters for Render
            search_params = {}
            if self.is_render:
                search_params.update({
                    "exact": False,  # Use approximate search
                    "hnsw_ef": 128,  # Higher values = more accurate but slower
                })
            
            # Perform search with retry
            results = await self._retry_operation(
                self.client.search,
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=filter_obj,
                **search_params
            )
            
            # Format results
            formatted_results = []
            for res in results:
                item = {
                    "id": str(res.id),
                    "score": res.score,
                    "metadata": res.payload
                }
                formatted_results.append(item)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vectors in Qdrant: {e}")
            return []
    
    async def get_by_id(
        self,
        collection_name: str,
        ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Get vectors by their IDs from Qdrant"""
        try:
            # Convert string IDs to UUID
            uuid_ids = [uuid.uuid5(uuid.NAMESPACE_DNS, id) for id in ids]
            
            # Get points
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=uuid_ids,
                with_vectors=True,
                with_payload=True
            )
            
            # Format results
            results = []
            for point in points:
                item = {
                    "id": str(point.id),
                    "vector": point.vector,
                    "metadata": point.payload
                }
                results.append(item)
            
            return results
        except Exception as e:
            logger.error(f"Error getting vectors by ID from Qdrant: {e}")
            return []


class InMemoryVectorDatabase(VectorDatabase):
    """Simple in-memory vector database for testing or local development"""
    
    def __init__(self):
        self.collections = {}
    
    async def initialize(self) -> None:
        """Initialize the in-memory database"""
        pass
    
    async def close(self) -> None:
        """Close the in-memory database"""
        self.collections = {}
    
    async def create_collection(self, collection_name: str, dimension: int) -> bool:
        """Create a new collection in memory"""
        try:
            if collection_name in self.collections:
                logger.info(f"Collection {collection_name} already exists")
                return True
            
            self.collections[collection_name] = {
                "dimension": dimension,
                "vectors": {}  # id -> (vector, metadata)
            }
            
            return True
        except Exception as e:
            logger.error(f"Error creating in-memory collection: {e}")
            return False
    
    async def insert_vectors(
        self, 
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """Insert vectors into memory"""
        try:
            if collection_name not in self.collections:
                logger.error(f"Collection {collection_name} does not exist")
                return False
            
            collection = self.collections[collection_name]
            
            # Check dimension
            for vector in vectors:
                if len(vector) != collection["dimension"]:
                    logger.error(f"Vector dimension mismatch: expected {collection['dimension']}, got {len(vector)}")
                    return False
            
            # Insert vectors
            for i in range(len(ids)):
                collection["vectors"][ids[i]] = (vectors[i], metadata[i])
            
            return True
        except Exception as e:
            logger.error(f"Error inserting vectors into in-memory collection: {e}")
            return False
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in memory"""
        try:
            if collection_name not in self.collections:
                logger.error(f"Collection {collection_name} does not exist")
                return []
            
            collection = self.collections[collection_name]
            
            # Check dimension
            if len(query_vector) != collection["dimension"]:
                logger.error(f"Query vector dimension mismatch: expected {collection['dimension']}, got {len(query_vector)}")
                return []
            
            # Convert query vector to numpy array
            query_array = np.array(query_vector)
            
            # Calculate similarity scores
            results = []
            for id, (vector, metadata) in collection["vectors"].items():
                # Apply filter if provided
                if filter_dict and not self._matches_filter(metadata, filter_dict):
                    continue
                
                # Calculate cosine similarity
                vector_array = np.array(vector)
                similarity = np.dot(query_array, vector_array) / (np.linalg.norm(query_array) * np.linalg.norm(vector_array))
                
                results.append({
                    "id": id,
                    "score": float(similarity),
                    "metadata": metadata
                })
            
            # Sort by similarity (highest first)
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # Apply limit
            return results[:top_k]
        except Exception as e:
            logger.error(f"Error searching vectors in in-memory collection: {e}")
            return []
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter conditions"""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
                
            if isinstance(value, list):
                # For arrays/lists, check if metadata value is in the filter list
                if metadata[key] not in value:
                    return False
            else:
                # For scalar values, check equality
                if metadata[key] != value:
                    return False
        
        return True
    
    async def get_by_id(
        self,
        collection_name: str,
        ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Get vectors by their IDs from memory"""
        try:
            if collection_name not in self.collections:
                logger.error(f"Collection {collection_name} does not exist")
                return []
            
            collection = self.collections[collection_name]
            
            # Get vectors
            results = []
            for id in ids:
                if id in collection["vectors"]:
                    vector, metadata = collection["vectors"][id]
                    results.append({
                        "id": id,
                        "vector": vector,
                        "metadata": metadata
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error getting vectors by ID from in-memory collection: {e}")
            return []


def create_vector_database(config: Dict[str, Any]) -> VectorDatabase:
    """Create a vector database instance based on configuration"""
    # Get Render configuration
    render_config = get_render_config()
    
    db_type = config.get("db_type", "in_memory").lower()
    
    if db_type == "qdrant" and HAVE_VECTOR_DB:
        return QdrantDatabase(
            url=config.get("url"),
            api_key=config.get("api_key"),
            config=render_config
        )
    else:
        # Fallback to in-memory
        return InMemoryVectorDatabase()


class RPGEntityManager:
    """Manager for RPG entity embeddings and retrieval"""
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        vector_db_config: Dict[str, Any],
        embedding_service: Any = None
    ):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize vector database 
        self.vector_db = create_vector_database(vector_db_config)
        
        # Store embedding service for generating embeddings
        self.embedding_service = embedding_service
        
        # Collection name mappings
        self.collections = {
            "npc": "npc_embeddings",
            "location": "location_embeddings",
            "memory": "memory_embeddings",
            "narrative": "narrative_embeddings"
        }
    
    async def initialize(self) -> None:
        """Initialize the entity manager"""
        await self.vector_db.initialize()
        
        # Create collections
        for collection_name in self.collections.values():
            await self.vector_db.create_collection(
                collection_name=collection_name,
                dimension=1536  # Default dimension
            )
    
    async def close(self) -> None:
        """Close the entity manager"""
        await self.vector_db.close()
    
    async def add_entity(
        self,
        entity_type: str,
        entity_id: str,
        content: str,
        embedding: Optional[List[float]] = None,
        **metadata
    ) -> bool:
        """
        Add an entity to the vector database
        
        Args:
            entity_type: Type of entity (npc, location, memory, etc.)
            entity_id: Unique ID for the entity
            content: Text content for the entity
            embedding: Optional pre-computed embedding
            **metadata: Additional metadata for the entity
            
        Returns:
            Success status
        """
        # Check if entity type is supported
        if entity_type not in self.collections:
            logger.error(f"Unsupported entity type: {entity_type}")
            return False
        
        # Get collection name
        collection_name = self.collections[entity_type]
        
        # Generate embedding if not provided
        if embedding is None and self.embedding_service:
            try:
                embedding = await self.embedding_service.get_embedding(content)
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                # Generate random embedding as fallback
                vec = list(np.random.normal(0, 1, 1536))
                embedding = vec / np.linalg.norm(vec)
        elif embedding is None:
            # Generate random embedding as fallback
            vec = list(np.random.normal(0, 1, 1536))
            embedding = vec / np.linalg.norm(vec)
        
        # Add user and conversation ID to metadata
        full_metadata = {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "content": content,
            **metadata
        }
        
        # Store in vector database
        return await self.vector_db.insert_vectors(
            collection_name=collection_name,
            ids=[f"{entity_type}_{entity_id}"],
            vectors=[embedding],
            metadata=[full_metadata]
        )
    
    async def add_memory(
        self,
        memory_id: str,
        content: str,
        memory_type: str = "observation",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        **extra_metadata
    ) -> bool:
        """
        Add a memory to the vector database.
        
        Args:
            memory_id: Unique identifier for the memory
            content: Memory content text  
            memory_type: Type of memory (observation, event, etc.)
            importance: Importance score (0-1)
            tags: Optional list of tags
            embedding: Optional pre-computed embedding
            **extra_metadata: Additional metadata fields
            
        Returns:
            Success status
        """
        # Combine all metadata
        metadata = {
            "memory_type": memory_type,
            "importance": importance,
            "tags": tags or [],
            **extra_metadata
        }
        
        # Use the generic add_entity method with entity_type="memory"
        return await self.add_entity(
            entity_type="memory",
            entity_id=memory_id,
            content=content,
            embedding=embedding,
            **metadata
        )
    
    async def add_npc(
        self,
        npc_id: str,
        npc_name: str,
        description: str = "",
        personality: str = "",
        location: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        **extra_metadata
    ) -> bool:
        """
        Add an NPC to the vector database.
        
        Args:
            npc_id: Unique identifier for the NPC
            npc_name: Name of the NPC
            description: Physical description
            personality: Personality traits
            location: Current location
            embedding: Optional pre-computed embedding
            **extra_metadata: Additional metadata fields
            
        Returns:
            Success status
        """
        # Create content for embedding if not provided
        content = f"NPC: {npc_name}. {description} {personality}"
        
        metadata = {
            "npc_name": npc_name,
            "description": description,
            "personality": personality,
            "location": location,
            **extra_metadata
        }
        
        return await self.add_entity(
            entity_type="npc",
            entity_id=npc_id,
            content=content,
            embedding=embedding,
            **metadata
        )
    
    async def add_location(
        self,
        location_id: str,
        location_name: str,
        description: str = "",
        location_type: Optional[str] = None,
        connected_locations: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        **extra_metadata
    ) -> bool:
        """
        Add a location to the vector database.
        
        Args:
            location_id: Unique identifier for the location
            location_name: Name of the location
            description: Location description
            location_type: Type of location (city, dungeon, etc.)
            connected_locations: List of connected location IDs
            embedding: Optional pre-computed embedding
            **extra_metadata: Additional metadata fields
            
        Returns:
            Success status
        """
        # Create content for embedding if not provided
        content = f"Location: {location_name}. {description}"
        
        metadata = {
            "location_name": location_name,
            "description": description,
            "location_type": location_type,
            "connected_locations": connected_locations or [],
            **extra_metadata
        }
        
        return await self.add_entity(
            entity_type="location",
            entity_id=location_id,
            content=content,
            embedding=embedding,
            **metadata
        )
    
    async def add_narrative(
        self,
        narrative_id: str,
        content: str,
        narrative_type: str = "story",
        importance: float = 0.5,
        embedding: Optional[List[float]] = None,
        **extra_metadata
    ) -> bool:
        """
        Add a narrative element to the vector database.
        
        Args:
            narrative_id: Unique identifier for the narrative
            content: Narrative content text
            narrative_type: Type of narrative (story, quest, etc.)
            importance: Importance score (0-1)
            embedding: Optional pre-computed embedding
            **extra_metadata: Additional metadata fields
            
        Returns:
            Success status
        """
        metadata = {
            "narrative_type": narrative_type,
            "importance": importance,
            **extra_metadata
        }
        
        return await self.add_entity(
            entity_type="narrative",
            entity_id=narrative_id,
            content=content,
            embedding=embedding,
            **metadata
        )
    
    async def search_entities(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        entity_types: Optional[List[str]] = None,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for entities relevant to the query
        
        Args:
            query_text: Query text (used if embedding not provided)
            query_embedding: Optional pre-computed embedding
            entity_types: List of entity types to search
            top_k: Number of results to return
            filter_dict: Optional filter conditions
            
        Returns:
            List of matching entities with scores
        """
        # Use all entity types if not specified
        if entity_types is None:
            entity_types = list(self.collections.keys())
        else:
            # Filter to supported entity types
            entity_types = [et for et in entity_types if et in self.collections]
        
        if not entity_types:
            logger.error("No supported entity types provided")
            return []
        
        # Generate embedding if not provided
        if query_embedding is None and self.embedding_service:
            try:
                query_embedding = await self.embedding_service.get_embedding(query_text)
            except Exception as e:
                logger.error(f"Error generating query embedding: {e}")
                # Generate random embedding as fallback
                vec = list(np.random.normal(0, 1, 1536))
                query_embedding = vec / np.linalg.norm(vec)
        elif query_embedding is None:
            # Generate random embedding as fallback
            vec = list(np.random.normal(0, 1, 1536))
            query_embedding = vec / np.linalg.norm(vec)
        
        # Prepare base filter
        base_filter = {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        }
        
        # Merge with custom filter if provided
        if filter_dict:
            base_filter.update(filter_dict)
        
        # Search each entity type
        all_results = []
        for entity_type in entity_types:
            # Get collection name
            collection_name = self.collections[entity_type]
            
            # Add entity type to filter
            entity_filter = base_filter.copy()
            entity_filter["entity_type"] = entity_type
            
            # Search for similar entities
            results = await self.vector_db.search_vectors(
                collection_name=collection_name,
                query_vector=query_embedding,
                top_k=top_k,
                filter_dict=entity_filter
            )
            
            all_results.extend(results)
        
        # Sort by score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply overall limit
        return all_results[:top_k]
    
    async def get_context_for_input(
        self,
        input_text: str,
        current_location: Optional[str] = None,
        max_items: int = 10
    ) -> Dict[str, Any]:
        """
        Get relevant context for the current input
        
        Args:
            input_text: Current input text
            current_location: Optional current location
            max_items: Maximum number of context items
            
        Returns:
            Relevant context from vector database
        """
        # Combine input with location for better context
        query = input_text
        if current_location:
            query += f" Location: {current_location}"
        
        # Get relevant entities
        results = await self.search_entities(
            query_text=query,
            top_k=max_items,
            entity_types=["npc", "location", "memory", "narrative"]
        )
        
        # Organize by entity type
        context = {
            "npcs": [],
            "locations": [],
            "memories": [],
            "narratives": []
        }
        
        for result in results:
            metadata = result["metadata"]
            entity_type = metadata.get("entity_type")
            
            if entity_type == "npc":
                context["npcs"].append({
                    "npc_id": metadata.get("entity_id"),
                    "npc_name": metadata.get("npc_name"),
                    "description": metadata.get("description", ""),
                    "personality": metadata.get("personality", ""),
                    "location": metadata.get("location"),
                    "relevance": result["score"]
                })
            elif entity_type == "location":
                context["locations"].append({
                    "location_id": metadata.get("entity_id"),
                    "location_name": metadata.get("location_name"),
                    "description": metadata.get("description", ""),
                    "connected_locations": metadata.get("connected_locations", []),
                    "relevance": result["score"]
                })
            elif entity_type == "memory":
                context["memories"].append({
                    "memory_id": metadata.get("entity_id"),
                    "content": metadata.get("content"),
                    "memory_type": metadata.get("memory_type"),
                    "importance": metadata.get("importance", 0.5),
                    "relevance": result["score"]
                })
            elif entity_type == "narrative":
                context["narratives"].append({
                    "narrative_id": metadata.get("entity_id"),
                    "content": metadata.get("content"),
                    "narrative_type": metadata.get("narrative_type"),
                    "importance": metadata.get("importance", 0.5),
                    "relevance": result["score"]
                })
        
        return context
