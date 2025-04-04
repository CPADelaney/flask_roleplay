# memory/chroma_vector_store.py

"""
ChromaDB Vector Store Implementation

This module provides a ChromaDB-based implementation of the VectorDatabase interface
for semantic search and memory retrieval in the system.
"""

import os
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import uuid
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pathlib import Path

# Import the abstract VectorDatabase class
from context.optimized_db import VectorDatabase

# Configure logging
logger = logging.getLogger(__name__)

class ChromaVectorDatabase(VectorDatabase):
    """ChromaDB vector database integration with performance optimizations."""
    
    def __init__(
        self, 
        persist_directory: Optional[str] = None,
        collection_name: str = "memories",
        embedding_function = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ChromaDB client.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Default collection name
            embedding_function: Optional custom embedding function
            config: Optional configuration dictionary
        """
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.default_collection_name = collection_name
        self.client = None
        self.collections = {}
        self.embedding_function = embedding_function
        self.config = config or {}
        
        # Create persist directory if it doesn't exist
        if self.persist_directory:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Retry settings
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 0.5)
    
    async def initialize(self) -> None:
        """Initialize the ChromaDB client."""
        try:
            # Configure client with optimized settings
            client_settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_directory,
                anonymized_telemetry=False
            )
            
            # Create the client
            self.client = chromadb.Client(client_settings)
            
            # Set up default embedding function if not provided
            if not self.embedding_function:
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"  # Fast and efficient model for most use cases
                )
            
            logger.info(f"Successfully connected to ChromaDB at {self.persist_directory}")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {e}")
            raise
    
    async def close(self) -> None:
        """Close the ChromaDB connection."""
        if self.client:
            # ChromaDB doesn't have an explicit close method, but we can persist data
            if self.persist_directory:
                self.client.persist()
            self.client = None
            self.collections = {}
    
    async def _retry_operation(self, operation, *args, **kwargs):
        """Retry operation with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Retrying operation after error: {e}. Attempt {attempt + 1}/{self.max_retries}")
                await asyncio.sleep(wait_time)
    
    async def _get_collection(self, collection_name: str):
        """Get or create a collection."""
        # Check if we already have this collection cached
        if collection_name in self.collections:
            return self.collections[collection_name]
            
        # Wrapping the synchronous calls in an asyncio executor
        loop = asyncio.get_event_loop()
        
        try:
            # First try to get the collection
            collection = await loop.run_in_executor(
                None, 
                lambda: self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
            )
        except Exception:
            # If it doesn't exist, create it
            collection = await loop.run_in_executor(
                None, 
                lambda: self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
            )
        
        # Cache the collection
        self.collections[collection_name] = collection
        return collection
    
    async def create_collection(self, collection_name: str, dimension: int) -> bool:
        """
        Create a new collection in ChromaDB.
        
        Args:
            collection_name: Collection name
            dimension: Vector dimension (ignored for ChromaDB as it's determined by the embedding function)
            
        Returns:
            Success status
        """
        try:
            # Get or create collection - ChromaDB handles this automatically
            await self._get_collection(collection_name)
            return True
        except Exception as e:
            logger.error(f"Error creating ChromaDB collection: {e}")
            return False
    
    async def insert_vectors(
        self, 
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """
        Insert vectors into ChromaDB.
        
        Args:
            collection_name: Collection name
            ids: List of document IDs
            vectors: List of embedding vectors
            metadata: List of metadata dictionaries
            
        Returns:
            Success status
        """
        try:
            # Get the collection
            collection = await self._get_collection(collection_name)
            
            # ChromaDB requires documents when not providing embeddings directly
            # Since we're providing embeddings directly, we'll use empty strings
            documents = [""] * len(ids)
            
            # Prepare batch operation
            batch_size = min(len(ids), 100)  # ChromaDB recommends smaller batches
            
            # Process in batches
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_vectors = vectors[i:i+batch_size]
                batch_metadata = metadata[i:i+batch_size]
                batch_documents = documents[i:i+batch_size]
                
                # Execute the batch insert/update
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: collection.upsert(
                        ids=batch_ids,
                        embeddings=batch_vectors,
                        metadatas=batch_metadata,
                        documents=batch_documents
                    )
                )
            
            # Persist if using a persistent directory
            if self.persist_directory:
                await loop.run_in_executor(None, self.client.persist)
            
            return True
        except Exception as e:
            logger.error(f"Error inserting vectors into ChromaDB: {e}")
            return False
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in ChromaDB.
        
        Args:
            collection_name: Collection name
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of search results
        """
        try:
            # Get the collection
            collection = await self._get_collection(collection_name)
            
            # Prepare filter if provided
            where = {}
            if filter_dict:
                where = filter_dict
            
            # Execute the query
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: collection.query(
                    query_embeddings=[query_vector],
                    n_results=top_k,
                    where=where or None,
                    include=["metadatas", "distances", "documents", "embeddings"]
                )
            )
            
            # Format the results to match our interface
            formatted_results = []
            
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "score": 1.0 - float(results["distances"][0][i]),  # Convert distance to similarity
                        "metadata": results["metadatas"][0][i],
                        "embedding": results["embeddings"][0][i] if "embeddings" in results else None
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vectors in ChromaDB: {e}")
            return []
    
    async def get_by_id(
        self,
        collection_name: str,
        ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get vectors by their IDs from ChromaDB.
        
        Args:
            collection_name: Collection name
            ids: List of document IDs
            
        Returns:
            List of retrieved documents
        """
        try:
            # Get the collection
            collection = await self._get_collection(collection_name)
            
            # Execute the get operation
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: collection.get(
                    ids=ids,
                    include=["metadatas", "documents", "embeddings"]
                )
            )
            
            # Format the results
            formatted_results = []
            
            if results["ids"]:
                for i in range(len(results["ids"])):
                    formatted_results.append({
                        "id": results["ids"][i],
                        "metadata": results["metadatas"][i],
                        "embedding": results["embeddings"][i] if "embeddings" in results else None
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error getting vectors by ID from ChromaDB: {e}")
            return []
