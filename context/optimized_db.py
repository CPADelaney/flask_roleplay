# context/optimized_db.py

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import numpy as np
from datetime import datetime, timedelta
import hashlib
import uuid

# Try to import different vector database clients with fallbacks
try:
    import qdrant_client
    from qdrant_client.http import models as qmodels
    HAVE_QDRANT = True
except ImportError:
    HAVE_QDRANT = False

try:
    import pinecone
    HAVE_PINECONE = True
except ImportError:
    HAVE_PINECONE = False

try:
    from pymilvus import Collection, connections, utility
    HAVE_MILVUS = True
except ImportError:
    HAVE_MILVUS = False

# Set up logging
logger = logging.getLogger(__name__)

# ----- Vector Database Interface -----

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
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection/index"""
        raise NotImplementedError("Subclasses must implement delete_collection")
    
    async def insert_vectors(
        self, 
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """Insert vectors into the database"""
        raise NotImplementedError("Subclasses must implement insert_vectors")
    
    async def delete_vectors(self, collection_name: str, ids: List[str]) -> bool:
        """Delete vectors from the database"""
        raise NotImplementedError("Subclasses must implement delete_vectors")
    
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
    """
    Qdrant vector database integration
    
    Provides integration with the Qdrant vector database for efficient 
    similarity search and retrieval.
    """
    
    def __init__(
        self, 
        url: str = "http://localhost:6333", 
        api_key: Optional[str] = None
    ):
        if not HAVE_QDRANT:
            raise ImportError("qdrant_client is required for QdrantDatabase")
        
        self.url = url
        self.api_key = api_key
        self.client = None
    
    async def initialize(self) -> None:
        """Initialize the Qdrant client"""
        # Qdrant client is not async, but we keep the async interface for consistency
        self.client = qdrant_client.QdrantClient(
            url=self.url,
            api_key=self.api_key
        )
    
    async def close(self) -> None:
        """Close the Qdrant connection"""
        # No explicit close method in Qdrant client
        self.client = None
    
    async def create_collection(self, collection_name: str, dimension: int) -> bool:
        """
        Create a new collection in Qdrant
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            
        Returns:
            Success status
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections().collections
            if collection_name in [c.name for c in collections]:
                logger.info(f"Collection {collection_name} already exists")
                return True
            
            # Create collection with standard parameters
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=dimension, 
                    distance=qmodels.Distance.COSINE
                )
            )
            
            return True
        except Exception as e:
            logger.error(f"Error creating Qdrant collection: {e}")
            return False
    
    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from Qdrant
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Success status
        """
        try:
            self.client.delete_collection(collection_name=collection_name)
            return True
        except Exception as e:
            logger.error(f"Error deleting Qdrant collection: {e}")
            return False
    
    async def insert_vectors(
        self, 
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """
        Insert vectors into Qdrant
        
        Args:
            collection_name: Name of the collection
            ids: List of unique IDs
            vectors: List of embedding vectors
            metadata: List of metadata dictionaries
            
        Returns:
            Success status
        """
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
            
            # Insert in batches of 100
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
            
            return True
        except Exception as e:
            logger.error(f"Error inserting vectors into Qdrant: {e}")
            return False
    
    async def delete_vectors(self, collection_name: str, ids: List[str]) -> bool:
        """
        Delete vectors from Qdrant
        
        Args:
            collection_name: Name of the collection
            ids: List of IDs to delete
            
        Returns:
            Success status
        """
        try:
            # Convert string IDs to UUID
            uuid_ids = [uuid.uuid5(uuid.NAMESPACE_DNS, id) for id in ids]
            
            # Delete points in batches of 100
            batch_size = 100
            for i in range(0, len(uuid_ids), batch_size):
                batch = uuid_ids[i:i+batch_size]
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=qmodels.PointIdsList(
                        points=batch
                    )
                )
            
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors from Qdrant: {e}")
            return False
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Qdrant
        
        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional filter conditions
            
        Returns:
            List of search results
        """
        try:
            # Convert filter dictionary to Qdrant filter
            filter_obj = None
            if filter_dict:
                filter_conditions = []
                for key, value in filter_dict.items():
                    if isinstance(value, list):
                        # For arrays/lists, use the 'any' match operator
                        filter_conditions.append(
                            qmodels.FieldCondition(
                                key=key,
                                match=qmodels.MatchAny(any=value)
                            )
                        )
                    else:
                        # For scalar values, use the 'match' operator
                        filter_conditions.append(
                            qmodels.FieldCondition(
                                key=key,
                                match=qmodels.MatchValue(value=value)
                            )
                        )
                
                if filter_conditions:
                    filter_obj = qmodels.Filter(
                        must=filter_conditions
                    )
            
            # Perform search
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=filter_obj
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
        """
        Get vectors by their IDs from Qdrant
        
        Args:
            collection_name: Name of the collection
            ids: List of IDs to retrieve
            
        Returns:
            List of retrieved vectors with metadata
        """
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


class PineconeDatabase(VectorDatabase):
    """
    Pinecone vector database integration
    
    Provides integration with the Pinecone vector database for efficient 
    similarity search and retrieval.
    """
    
    def __init__(
        self, 
        api_key: str,
        environment: str
    ):
        if not HAVE_PINECONE:
            raise ImportError("pinecone-client is required for PineconeDatabase")
        
        self.api_key = api_key
        self.environment = environment
        self.indexes = {}
    
    async def initialize(self) -> None:
        """Initialize the Pinecone client"""
        # Pinecone client is not async, but we keep the async interface for consistency
        pinecone.init(
            api_key=self.api_key,
            environment=self.environment
        )
    
    async def close(self) -> None:
        """Close the Pinecone connection"""
        self.indexes = {}
    
    async def create_collection(self, collection_name: str, dimension: int) -> bool:
        """
        Create a new index in Pinecone
        
        Args:
            collection_name: Name of the index
            dimension: Vector dimension
            
        Returns:
            Success status
        """
        try:
            # Check if index already exists
            existing_indexes = pinecone.list_indexes()
            
            if collection_name in existing_indexes:
                # Get the existing index
                self.indexes[collection_name] = pinecone.Index(collection_name)
                logger.info(f"Index {collection_name} already exists")
                return True
            
            # Create the index
            pinecone.create_index(
                name=collection_name,
                dimension=dimension,
                metric="cosine"
            )
            
            # Wait for index to be ready
            while not collection_name in pinecone.list_indexes():
                await asyncio.sleep(1)
            
            # Get the index
            self.indexes[collection_name] = pinecone.Index(collection_name)
            
            return True
        except Exception as e:
            logger.error(f"Error creating Pinecone index: {e}")
            return False
    
    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete an index from Pinecone
        
        Args:
            collection_name: Name of the index
            
        Returns:
            Success status
        """
        try:
            pinecone.delete_index(collection_name)
            
            if collection_name in self.indexes:
                del self.indexes[collection_name]
            
            return True
        except Exception as e:
            logger.error(f"Error deleting Pinecone index: {e}")
            return False
    
    async def insert_vectors(
        self, 
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """
        Insert vectors into Pinecone
        
        Args:
            collection_name: Name of the index
            ids: List of unique IDs
            vectors: List of embedding vectors
            metadata: List of metadata dictionaries
            
        Returns:
            Success status
        """
        try:
            # Get the index
            if collection_name not in self.indexes:
                existing_indexes = pinecone.list_indexes()
                if collection_name in existing_indexes:
                    self.indexes[collection_name] = pinecone.Index(collection_name)
                else:
                    logger.error(f"Index {collection_name} does not exist")
                    return False
            
            index = self.indexes[collection_name]
            
            # Build vector tuples
            vector_tuples = []
            for i in range(len(ids)):
                vector_tuples.append(
                    (ids[i], vectors[i], metadata[i])
                )
            
            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(vector_tuples), batch_size):
                batch = vector_tuples[i:i+batch_size]
                index.upsert(vectors=batch)
            
            return True
        except Exception as e:
            logger.error(f"Error inserting vectors into Pinecone: {e}")
            return False
    
    async def delete_vectors(self, collection_name: str, ids: List[str]) -> bool:
        """
        Delete vectors from Pinecone
        
        Args:
            collection_name: Name of the index
            ids: List of IDs to delete
            
        Returns:
            Success status
        """
        try:
            # Get the index
            if collection_name not in self.indexes:
                existing_indexes = pinecone.list_indexes()
                if collection_name in existing_indexes:
                    self.indexes[collection_name] = pinecone.Index(collection_name)
                else:
                    logger.error(f"Index {collection_name} does not exist")
                    return False
            
            index = self.indexes[collection_name]
            
            # Delete vectors in batches of 100
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch = ids[i:i+batch_size]
                index.delete(ids=batch)
            
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors from Pinecone: {e}")
            return False
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Pinecone
        
        Args:
            collection_name: Name of the index
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional filter conditions
            
        Returns:
            List of search results
        """
        try:
            # Get the index
            if collection_name not in self.indexes:
                existing_indexes = pinecone.list_indexes()
                if collection_name in existing_indexes:
                    self.indexes[collection_name] = pinecone.Index(collection_name)
                else:
                    logger.error(f"Index {collection_name} does not exist")
                    return []
            
            index = self.indexes[collection_name]
            
            # Perform search
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                include_values=False,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                item = {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                formatted_results.append(item)
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching vectors in Pinecone: {e}")
            return []
    
    async def get_by_id(
        self,
        collection_name: str,
        ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get vectors by their IDs from Pinecone
        
        Args:
            collection_name: Name of the index
            ids: List of IDs to retrieve
            
        Returns:
            List of retrieved vectors with metadata
        """
        try:
            # Get the index
            if collection_name not in self.indexes:
                existing_indexes = pinecone.list_indexes()
                if collection_name in existing_indexes:
                    self.indexes[collection_name] = pinecone.Index(collection_name)
                else:
                    logger.error(f"Index {collection_name} does not exist")
                    return []
            
            index = self.indexes[collection_name]
            
            # Fetch vectors in batches of 100
            batch_size = 100
            all_results = []
            
            for i in range(0, len(ids), batch_size):
                batch = ids[i:i+batch_size]
                fetch_response = index.fetch(ids=batch)
                
                for vec_id, vector_data in fetch_response.vectors.items():
                    item = {
                        "id": vec_id,
                        "vector": vector_data.values,
                        "metadata": vector_data.metadata
                    }
                    all_results.append(item)
            
            return all_results
        except Exception as e:
            logger.error(f"Error getting vectors by ID from Pinecone: {e}")
            return []


class MilvusDatabase(VectorDatabase):
    """
    Milvus vector database integration
    
    Provides integration with the Milvus vector database for efficient 
    similarity search and retrieval.
    """
    
    def __init__(
        self, 
        host: str = "localhost", 
        port: str = "19530",
        user: str = "", 
        password: str = ""
    ):
        if not HAVE_MILVUS:
            raise ImportError("pymilvus is required for MilvusDatabase")
        
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.collections = {}
    
    async def initialize(self) -> None:
        """Initialize the Milvus client"""
        # Milvus client is not async, but we keep the async interface for consistency
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password
        )
    
    async def close(self) -> None:
        """Close the Milvus connection"""
        connections.disconnect("default")
        self.collections = {}
    
    async def create_collection(self, collection_name: str, dimension: int) -> bool:
        """
        Create a new collection in Milvus
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            
        Returns:
            Success status
        """
        try:
            # Check if collection already exists
            if utility.has_collection(collection_name):
                # Load existing collection
                self.collections[collection_name] = Collection(collection_name)
                self.collections[collection_name].load()
                logger.info(f"Collection {collection_name} already exists")
                return True
            
            # Define collection schema
            from pymilvus import FieldSchema, CollectionSchema, DataType
            
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            schema = CollectionSchema(fields=fields, description=f"Collection {collection_name}")
            
            # Create collection
            collection = Collection(name=collection_name, schema=schema)
            
            # Create an index for vector field
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            }
            collection.create_index(field_name="vector", index_params=index_params)
            
            # Load collection
            collection.load()
            
            # Store collection
            self.collections[collection_name] = collection
            
            return True
        except Exception as e:
            logger.error(f"Error creating Milvus collection: {e}")
            return False
    
    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from Milvus
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Success status
        """
        try:
            utility.drop_collection(collection_name)
            
            if collection_name in self.collections:
                del self.collections[collection_name]
            
            return True
        except Exception as e:
            logger.error(f"Error deleting Milvus collection: {e}")
            return False
    
    async def insert_vectors(
        self, 
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """
        Insert vectors into Milvus
        
        Args:
            collection_name: Name of the collection
            ids: List of unique IDs
            vectors: List of embedding vectors
            metadata: List of metadata dictionaries
            
        Returns:
            Success status
        """
        try:
            # Get the collection
            if collection_name not in self.collections:
                if utility.has_collection(collection_name):
                    self.collections[collection_name] = Collection(collection_name)
                    self.collections[collection_name].load()
                else:
                    logger.error(f"Collection {collection_name} does not exist")
                    return False
            
            collection = self.collections[collection_name]
            
            # Prepare data
            insert_data = [
                ids,
                vectors,
                metadata
            ]
            
            # Insert data
            collection.insert(insert_data)
            
            return True
        except Exception as e:
            logger.error(f"Error inserting vectors into Milvus: {e}")
            return False
    
    async def delete_vectors(self, collection_name: str, ids: List[str]) -> bool:
        """
        Delete vectors from Milvus
        
        Args:
            collection_name: Name of the collection
            ids: List of IDs to delete
            
        Returns:
            Success status
        """
        try:
            # Get the collection
            if collection_name not in self.collections:
                if utility.has_collection(collection_name):
                    self.collections[collection_name] = Collection(collection_name)
                    self.collections[collection_name].load()
                else:
                    logger.error(f"Collection {collection_name} does not exist")
                    return False
            
            collection = self.collections[collection_name]
            
            # Create delete expression
            expr = f"id in [\"{'\", \"'.join(ids)}\"]"
            
            # Delete vectors
            collection.delete(expr)
            
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors from Milvus: {e}")
            return False
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Milvus
        
        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional filter conditions
            
        Returns:
            List of search results
        """
        try:
            # Get the collection
            if collection_name not in self.collections:
                if utility.has_collection(collection_name):
                    self.collections[collection_name] = Collection(collection_name)
                    self.collections[collection_name].load()
                else:
                    logger.error(f"Collection {collection_name} does not exist")
                    return []
            
            collection = self.collections[collection_name]
            
            # Convert filter dictionary to an expression string
            expr = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    if isinstance(value, list):
                        # For arrays/lists
                        value_str = ", ".join([f'"{item}"' if isinstance(item, str) else str(item) for item in value])
                        conditions.append(f'metadata["{key}"] in [{value_str}]')
                    elif isinstance(value, str):
                        # For strings
                        conditions.append(f'metadata["{key}"] == "{value}"')
                    else:
                        # For numbers and booleans
                        conditions.append(f'metadata["{key}"] == {value}')
                
                if conditions:
                    expr = " && ".join(conditions)
            
            # Define search parameters
            search_params = {"metric_type": "COSINE", "params": {"ef": 100}}
            
            # Execute search
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["metadata"]
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    item = {
                        "id": hit.id,
                        "score": hit.score,
                        "metadata": hit.entity.get("metadata", {})
                    }
                    formatted_results.append(item)
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching vectors in Milvus: {e}")
            return []
    
    async def get_by_id(
        self,
        collection_name: str,
        ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get vectors by their IDs from Milvus
        
        Args:
            collection_name: Name of the collection
            ids: List of IDs to retrieve
            
        Returns:
            List of retrieved vectors with metadata
        """
        try:
            # Get the collection
            if collection_name not in self.collections:
                if utility.has_collection(collection_name):
                    self.collections[collection_name] = Collection(collection_name)
                    self.collections[collection_name].load()
                else:
                    logger.error(f"Collection {collection_name} does not exist")
                    return []
            
            collection = self.collections[collection_name]
            
            # Create query expression
            expr = f"id in [\"{'\", \"'.join(ids)}\"]"
            
            # Execute query
            results = collection.query(
                expr=expr,
                output_fields=["vector", "metadata"]
            )
            
            # Format results
            formatted_results = []
            for entity in results:
                item = {
                    "id": entity["id"],
                    "vector": entity["vector"],
                    "metadata": entity["metadata"]
                }
                formatted_results.append(item)
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error getting vectors by ID from Milvus: {e}")
            return []


class InMemoryVectorDatabase(VectorDatabase):
    """
    Simple in-memory vector database for testing or when external services are unavailable
    
    Provides a basic vector database implementation with in-memory storage
    for development, testing, or when external vector databases are not available.
    """
    
    def __init__(self):
        self.collections = {}
    
    async def initialize(self) -> None:
        """Initialize the in-memory database"""
        # Nothing to do for in-memory database
        pass
    
    async def close(self) -> None:
        """Close the in-memory database"""
        self.collections = {}
    
    async def create_collection(self, collection_name: str, dimension: int) -> bool:
        """
        Create a new collection in memory
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            
        Returns:
            Success status
        """
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
    
    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from memory
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Success status
        """
        try:
            if collection_name in self.collections:
                del self.collections[collection_name]
            
            return True
        except Exception as e:
            logger.error(f"Error deleting in-memory collection: {e}")
            return False
    
    async def insert_vectors(
        self, 
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """
        Insert vectors into memory
        
        Args:
            collection_name: Name of the collection
            ids: List of unique IDs
            vectors: List of embedding vectors
            metadata: List of metadata dictionaries
            
        Returns:
            Success status
        """
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
    
    async def delete_vectors(self, collection_name: str, ids: List[str]) -> bool:
        """
        Delete vectors from memory
        
        Args:
            collection_name: Name of the collection
            ids: List of IDs to delete
            
        Returns:
            Success status
        """
        try:
            if collection_name not in self.collections:
                logger.error(f"Collection {collection_name} does not exist")
                return False
            
            collection = self.collections[collection_name]
            
            # Delete vectors
            for id in ids:
                if id in collection["vectors"]:
                    del collection["vectors"][id]
            
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors from in-memory collection: {e}")
            return False
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in memory
        
        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional filter conditions
            
        Returns:
            List of search results
        """
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
        """
        Get vectors by their IDs from memory
        
        Args:
            collection_name: Name of the collection
            ids: List of IDs to retrieve
            
        Returns:
            List of retrieved vectors with metadata
        """
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


# ----- Vector Database Factory -----

def create_vector_database(config: Dict[str, Any]) -> VectorDatabase:
    """
    Create a vector database instance based on configuration
    
    Args:
        config: Configuration dictionary with db_type and connection parameters
        
    Returns:
        Initialized VectorDatabase instance
    """
    db_type = config.get("db_type", "in_memory").lower()
    
    if db_type == "qdrant":
        if not HAVE_QDRANT:
            logger.warning("Qdrant client not available, falling back to in-memory database")
            return InMemoryVectorDatabase()
        
        return QdrantDatabase(
            url=config.get("url", "http://localhost:6333"),
            api_key=config.get("api_key")
        )
    
    elif db_type == "pinecone":
        if not HAVE_PINECONE:
            logger.warning("Pinecone client not available, falling back to in-memory database")
            return InMemoryVectorDatabase()
        
        api_key = config.get("api_key")
        environment = config.get("environment")
        
        if not api_key or not environment:
            logger.warning("Pinecone API key or environment not provided, falling back to in-memory database")
            return InMemoryVectorDatabase()
        
        return PineconeDatabase(
            api_key=api_key,
            environment=environment
        )
    
    elif db_type == "milvus":
        if not HAVE_MILVUS:
            logger.warning("Milvus client not available, falling back to in-memory database")
            return InMemoryVectorDatabase()
        
        return MilvusDatabase(
            host=config.get("host", "localhost"),
            port=config.get("port", "19530"),
            user=config.get("user", ""),
            password=config.get("password", "")
        )
    
    else:
        return InMemoryVectorDatabase()


# ----- Vector Database Service -----

class VectorDatabaseService:
    """
    High-level service for vector database operations
    
    Provides a unified interface for storing and retrieving vector embeddings
    for various entity types in the RPG game.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        embedding_dimension: int = 384
    ):
        self.config = config
        self.embedding_dimension = embedding_dimension
        self.vector_db = create_vector_database(config)
        
        # Define collection name mappings
        self.collection_mappings = {
            "npc": "npc_embeddings",
            "location": "location_embeddings",
            "narrative": "narrative_embeddings",
            "memory": "memory_embeddings",
            "item": "item_embeddings",
            "quest": "quest_embeddings",
            "conflict": "conflict_embeddings"
        }
    
    async def initialize(self) -> None:
        """Initialize the vector database service"""
        await self.vector_db.initialize()
        
        # Create collections for each entity type
        for collection_name in self.collection_mappings.values():
            await self.vector_db.create_collection(collection_name, self.embedding_dimension)
    
    async def close(self) -> None:
        """Close the vector database service"""
        await self.vector_db.close()
    
    async def store_entity_embedding(
        self,
        entity_type: str,
        entity_id: str,
        vector: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Store an entity embedding in the vector database
        
        Args:
            entity_type: Type of entity (npc, location, etc.)
            entity_id: Unique ID for the entity
            vector: Embedding vector
            metadata: Metadata for the entity
            
        Returns:
            Success status
        """
        # Check if entity type is supported
        if entity_type not in self.collection_mappings:
            logger.error(f"Unsupported entity type: {entity_type}")
            return False
        
        # Add entity type to metadata
        metadata["entity_type"] = entity_type
        
        # Get collection name
        collection_name = self.collection_mappings[entity_type]
        
        # Generate a unique vector ID
        vector_id = f"{entity_type}_{entity_id}"
        
        # Store embedding
        return await self.vector_db.insert_vectors(
            collection_name=collection_name,
            ids=[vector_id],
            vectors=[vector],
            metadata=[metadata]
        )
    
    async def delete_entity_embedding(
        self,
        entity_type: str,
        entity_id: str
    ) -> bool:
        """
        Delete an entity embedding from the vector database
        
        Args:
            entity_type: Type of entity (npc, location, etc.)
            entity_id: Unique ID for the entity
            
        Returns:
            Success status
        """
        # Check if entity type is supported
        if entity_type not in self.collection_mappings:
            logger.error(f"Unsupported entity type: {entity_type}")
            return False
        
        # Get collection name
        collection_name = self.collection_mappings[entity_type]
        
        # Generate vector ID
        vector_id = f"{entity_type}_{entity_id}"
        
        # Delete embedding
        return await self.vector_db.delete_vectors(
            collection_name=collection_name,
            ids=[vector_id]
        )
    
    async def search_similar_entities(
        self,
        entity_type: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar entities in the vector database
        
        Args:
            entity_type: Type of entity (npc, location, etc.)
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional filter conditions
            
        Returns:
            List of similar entities with scores
        """
        # Check if entity type is supported
        if entity_type not in self.collection_mappings:
            logger.error(f"Unsupported entity type: {entity_type}")
            return []
        
        # Get collection name
        collection_name = self.collection_mappings[entity_type]
        
        # Add entity type filter if not already in filter
        if filter_dict is None:
            filter_dict = {"entity_type": entity_type}
        elif "entity_type" not in filter_dict:
            filter_dict["entity_type"] = entity_type
        
        # Search for similar vectors
        return await self.vector_db.search_vectors(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k,
            filter_dict=filter_dict
        )
    
    async def search_across_entity_types(
        self,
        query_vector: List[float],
        entity_types: Optional[List[str]] = None,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar entities across multiple entity types
        
        Args:
            query_vector: Query embedding vector
            entity_types: List of entity types to search (default: all)
            top_k: Number of results per entity type
            filter_dict: Optional filter conditions
            
        Returns:
            List of similar entities with scores
        """
        # Use all entity types if not specified
        if entity_types is None:
            entity_types = list(self.collection_mappings.keys())
        
        # Check if entity types are supported
        for entity_type in entity_types:
            if entity_type not in self.collection_mappings:
                logger.error(f"Unsupported entity type: {entity_type}")
                entity_types.remove(entity_type)
        
        if not entity_types:
            logger.error("No supported entity types provided")
            return []
        
        # Search each entity type
        all_results = []
        for entity_type in entity_types:
            results = await self.search_similar_entities(
                entity_type=entity_type,
                query_vector=query_vector,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            all_results.extend(results)
        
        # Sort all results by score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply overall limit
        return all_results[:top_k]
    
    async def get_entity_by_id(
        self,
        entity_type: str,
        entity_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get an entity by its ID
        
        Args:
            entity_type: Type of entity (npc, location, etc.)
            entity_id: Unique ID for the entity
            
        Returns:
            Entity data or None if not found
        """
        # Check if entity type is supported
        if entity_type not in self.collection_mappings:
            logger.error(f"Unsupported entity type: {entity_type}")
            return None
        
        # Get collection name
        collection_name = self.collection_mappings[entity_type]
        
        # Generate vector ID
        vector_id = f"{entity_type}_{entity_id}"
        
        # Get entity
        results = await self.vector_db.get_by_id(
            collection_name=collection_name,
            ids=[vector_id]
        )
        
        if not results:
            return None
        
        return results[0]


# ----- RPG Entity Manager -----

class RPGEntityManager:
    """
    Manager for RPG entity embeddings and retrieval
    
    Provides high-level methods for managing entity embeddings and retrieving
    relevant entities in the context of an RPG game.
    """
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        vector_db_config: Dict[str, Any],
        embedding_service: Any = None
    ):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize vector database service
        self.vector_db_service = VectorDatabaseService(vector_db_config)
        
        # Store embedding service for generating embeddings
        self.embedding_service = embedding_service
    
    async def initialize(self) -> None:
        """Initialize the entity manager"""
        await self.vector_db_service.initialize()
    
    async def close(self) -> None:
        """Close the entity manager"""
        await self.vector_db_service.close()
    
    async def add_npc(
        self,
        npc_id: int,
        npc_name: str,
        description: str,
        personality: str,
        location: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Add an NPC to the vector database
        
        Args:
            npc_id: Unique ID for the NPC
            npc_name: Name of the NPC
            description: Physical description of the NPC
            personality: Personality description of the NPC
            location: Current location of the NPC
            tags: Optional list of tags for the NPC
            
        Returns:
            Success status
        """
        # Generate embeddings
        if self.embedding_service:
            text_to_embed = f"NPC: {npc_name}. Description: {description}. Personality: {personality}"
            embedding = await self.embedding_service.get_embedding(text_to_embed)
        else:
            # Use random embedding for testing without embedding service
            embedding = list(np.random.normal(0, 1, 384))
            embedding = embedding / np.linalg.norm(embedding)
        
        # Prepare metadata
        metadata = {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "npc_id": npc_id,
            "npc_name": npc_name,
            "description": description,
            "personality": personality,
            "location": location,
            "tags": tags or []
        }
        
        # Store embedding
        return await self.vector_db_service.store_entity_embedding(
            entity_type="npc",
            entity_id=str(npc_id),
            vector=embedding,
            metadata=metadata
        )
    
    async def add_location(
        self,
        location_id: int,
        location_name: str,
        description: str,
        connected_locations: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Add a location to the vector database
        
        Args:
            location_id: Unique ID for the location
            location_name: Name of the location
            description: Description of the location
            connected_locations: List of connected location names
            tags: Optional list of tags for the location
            
        Returns:
            Success status
        """
        # Generate embeddings
        if self.embedding_service:
            text_to_embed = f"Location: {location_name}. Description: {description}"
            embedding = await self.embedding_service.get_embedding(text_to_embed)
        else:
            # Use random embedding for testing without embedding service
            embedding = list(np.random.normal(0, 1, 384))
            embedding = embedding / np.linalg.norm(embedding)
        
        # Prepare metadata
        metadata = {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "location_id": location_id,
            "location_name": location_name,
            "description": description,
            "connected_locations": connected_locations or [],
            "tags": tags or []
        }
        
        # Store embedding
        return await self.vector_db_service.store_entity_embedding(
            entity_type="location",
            entity_id=str(location_id),
            vector=embedding,
            metadata=metadata
        )
    
    async def add_memory(
        self,
        memory_id: str,
        content: str,
        memory_type: str,
        importance: float,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Add a memory to the vector database
        
        Args:
            memory_id: Unique ID for the memory
            content: Memory content
            memory_type: Type of memory
            importance: Importance score (0-1)
            tags: Optional list of tags for the memory
            
        Returns:
            Success status
        """
        # Generate embeddings
        if self.embedding_service:
            embedding = await self.embedding_service.get_embedding(content)
        else:
            # Use random embedding for testing without embedding service
            embedding = list(np.random.normal(0, 1, 384))
            embedding = embedding / np.linalg.norm(embedding)
        
        # Prepare metadata
        metadata = {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "memory_id": memory_id,
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            "tags": tags or []
        }
        
        # Store embedding
        return await self.vector_db_service.store_entity_embedding(
            entity_type="memory",
            entity_id=memory_id,
            vector=embedding,
            metadata=metadata
        )
    
    async def add_narrative(
        self,
        narrative_id: str,
        content: str,
        narrative_type: str,
        importance: float,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Add a narrative element to the vector database
        
        Args:
            narrative_id: Unique ID for the narrative element
            content: Narrative content
            narrative_type: Type of narrative element
            importance: Importance score (0-1)
            tags: Optional list of tags for the narrative element
            
        Returns:
            Success status
        """
        # Generate embeddings
        if self.embedding_service:
            embedding = await self.embedding_service.get_embedding(content)
        else:
            # Use random embedding for testing without embedding service
            embedding = list(np.random.normal(0, 1, 384))
            embedding = embedding / np.linalg.norm(embedding)
        
        # Prepare metadata
        metadata = {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "narrative_id": narrative_id,
            "content": content,
            "narrative_type": narrative_type,
            "importance": importance,
            "tags": tags or []
        }
        
        # Store embedding
        return await self.vector_db_service.store_entity_embedding(
            entity_type="narrative",
            entity_id=narrative_id,
            vector=embedding,
            metadata=metadata
        )
    
    async def get_similar_npcs(
        self,
        query_text: str,
        top_k: int = 5,
        filter_tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get NPCs similar to the query text
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            filter_tags: Optional list of tags to filter by
            
        Returns:
            List of similar NPCs with scores
        """
        # Generate query embedding
        if self.embedding_service:
            query_embedding = await self.embedding_service.get_embedding(query_text)
        else:
            # Use random embedding for testing without embedding service
            query_embedding = list(np.random.normal(0, 1, 384))
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Prepare filter
        filter_dict = {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        }
        
        if filter_tags:
            filter_dict["tags"] = filter_tags
        
        # Search for similar NPCs
        results = await self.vector_db_service.search_similar_entities(
            entity_type="npc",
            query_vector=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        return results
    
    async def get_relevant_entities(
        self,
        query_text: str,
        top_k: int = 10,
        entity_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get entities relevant to the query text across multiple entity types
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            entity_types: Optional list of entity types to search
            
        Returns:
            List of relevant entities with scores
        """
        # Generate query embedding
        if self.embedding_service:
            query_embedding = await self.embedding_service.get_embedding(query_text)
        else:
            # Use random embedding for testing without embedding service
            query_embedding = list(np.random.normal(0, 1, 384))
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Prepare filter for user and conversation
        filter_dict = {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        }
        
        # Search across entity types
        results = await self.vector_db_service.search_across_entity_types(
            query_vector=query_embedding,
            entity_types=entity_types,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        return results
    
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
        results = await self.get_relevant_entities(
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
                    "npc_id": metadata.get("npc_id"),
                    "npc_name": metadata.get("npc_name"),
                    "description": metadata.get("description"),
                    "personality": metadata.get("personality"),
                    "location": metadata.get("location"),
                    "relevance": result["score"]
                })
            elif entity_type == "location":
                context["locations"].append({
                    "location_id": metadata.get("location_id"),
                    "location_name": metadata.get("location_name"),
                    "description": metadata.get("description"),
                    "connected_locations": metadata.get("connected_locations", []),
                    "relevance": result["score"]
                })
            elif entity_type == "memory":
                context["memories"].append({
                    "memory_id": metadata.get("memory_id"),
                    "content": metadata.get("content"),
                    "memory_type": metadata.get("memory_type"),
                    "importance": metadata.get("importance", 0.5),
                    "relevance": result["score"]
                })
            elif entity_type == "narrative":
                context["narratives"].append({
                    "narrative_id": metadata.get("narrative_id"),
                    "content": metadata.get("content"),
                    "narrative_type": metadata.get("narrative_type"),
                    "importance": metadata.get("importance", 0.5),
                    "relevance": result["score"]
                })
        
        return context
