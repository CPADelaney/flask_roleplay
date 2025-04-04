# memory/faiss_vector_store.py

"""
FAISS Vector Store Implementation

This module provides a FAISS-based implementation of the VectorDatabase interface
for semantic search and memory retrieval in the system.
"""

import os
import logging
import time
import asyncio
import numpy as np
import faiss
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import json

# Import the abstract VectorDatabase class
from context.optimized_db import VectorDatabase

# Configure logging
logger = logging.getLogger(__name__)

class FAISSVectorDatabase(VectorDatabase):
    """FAISS vector database integration with performance optimizations."""
    
    def __init__(
        self, 
        persist_directory: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the FAISS database.
        
        Args:
            persist_directory: Directory to persist the database
            config: Optional configuration dictionary
        """
        self.persist_directory = persist_directory or os.getenv("FAISS_PERSIST_DIR", "./faiss_db")
        self.indexes = {}  # Map collection names to FAISS indexes
        self.metadata = {}  # Map document IDs to metadata
        self.config = config or {}
        
        # Create persist directory if it doesn't exist
        if self.persist_directory:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Retry settings
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 0.5)
    
    async def initialize(self) -> None:
        """Initialize the FAISS database."""
        try:
            # Load existing indexes and metadata if available
            if self.persist_directory:
                await self._load_collections()
            
            logger.info(f"Successfully initialized FAISS database at {self.persist_directory}")
            
        except Exception as e:
            logger.error(f"Error initializing FAISS database: {e}")
            raise
    
    async def _load_collections(self):
        """Load existing collections from disk."""
        # Run in thread to avoid blocking
        loop = asyncio.get_event_loop()
        
        path = Path(self.persist_directory)
        if not path.exists():
            return
        
        # Load metadata first
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            try:
                content = await loop.run_in_executor(None, metadata_path.read_text)
                self.metadata = json.loads(content)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                self.metadata = {}
        
        # Load FAISS indexes
        index_files = list(path.glob("*.index"))
        for index_file in index_files:
            collection_name = index_file.stem
            try:
                # Load the index
                index = await loop.run_in_executor(
                    None,
                    lambda: faiss.read_index(str(index_file))
                )
                self.indexes[collection_name] = index
                logger.info(f"Loaded FAISS index for collection {collection_name}")
            except Exception as e:
                logger.error(f"Error loading FAISS index {collection_name}: {e}")
    
    async def _save_collection(self, collection_name: str):
        """Save a collection to disk."""
        if not self.persist_directory:
            return
        
        # Run in thread to avoid blocking
        loop = asyncio.get_event_loop()
        
        path = Path(self.persist_directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save the FAISS index
        if collection_name in self.indexes:
            index_path = path / f"{collection_name}.index"
            try:
                await loop.run_in_executor(
                    None,
                    lambda: faiss.write_index(self.indexes[collection_name], str(index_path))
                )
            except Exception as e:
                logger.error(f"Error saving FAISS index {collection_name}: {e}")
        
        # Save metadata
        metadata_path = path / "metadata.json"
        try:
            await loop.run_in_executor(
                None,
                lambda: metadata_path.write_text(json.dumps(self.metadata))
            )
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    async def close(self) -> None:
        """Close the FAISS database."""
        # Save all collections
        save_tasks = []
        for collection_name in self.indexes:
            save_tasks.append(self._save_collection(collection_name))
        
        if save_tasks:
            await asyncio.gather(*save_tasks)
        
        # Clear memory
        self.indexes = {}
        self.metadata = {}
        
        logger.info("FAISS database closed")
    
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
    
    async def create_collection(self, collection_name: str, dimension: int) -> bool:
        """
        Create a new collection in FAISS.
        
        Args:
            collection_name: Collection name
            dimension: Vector dimension
            
        Returns:
            Success status
        """
        try:
            # Check if collection already exists
            if collection_name in self.indexes:
                logger.info(f"Collection {collection_name} already exists")
                return True
            
            # Create a new FAISS index
            # Using IndexFlatIP for inner product similarity (dot product)
            # For cosine similarity, vectors should be normalized before insertion
            loop = asyncio.get_event_loop()
            index = await loop.run_in_executor(
                None,
                lambda: faiss.IndexFlatIP(dimension)
            )
            
            # Store the index
            self.indexes[collection_name] = index
            
            # Initialize metadata for this collection
            if collection_name not in self.metadata:
                self.metadata[collection_name] = {}
            
            # Save the collection
            await self._save_collection(collection_name)
            
            logger.info(f"Created FAISS collection {collection_name} with dimension {dimension}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating FAISS collection: {e}")
            return False
    
    async def insert_vectors(
        self, 
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """
        Insert vectors into FAISS.
        
        Args:
            collection_name: Collection name
            ids: List of document IDs
            vectors: List of embedding vectors
            metadata: List of metadata dictionaries
            
        Returns:
            Success status
        """
        try:
            # Check if collection exists
            if collection_name not in self.indexes:
                dimension = len(vectors[0])
                success = await self.create_collection(collection_name, dimension)
                if not success:
                    return False
            
            # Convert vectors to numpy array
            vectors_np = np.array(vectors, dtype=np.float32)
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(vectors_np)
            
            # Add vectors to the index
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.indexes[collection_name].add(vectors_np)
            )
            
            # Store metadata
            if collection_name not in self.metadata:
                self.metadata[collection_name] = {}
            
            # Map IDs to metadata and internal FAISS IDs
            total_vectors = self.indexes[collection_name].ntotal
            start_idx = total_vectors - len(ids)
            
            for i, doc_id in enumerate(ids):
                faiss_id = start_idx + i
                self.metadata[collection_name][doc_id] = {
                    "metadata": metadata[i],
                    "faiss_id": faiss_id
                }
            
            # Save the collection
            await self._save_collection(collection_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Error inserting vectors into FAISS: {e}")
            return False
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in FAISS.
        
        Args:
            collection_name: Collection name
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of search results
        """
        try:
            # Check if collection exists
            if collection_name not in self.indexes:
                logger.warning(f"Collection {collection_name} does not exist")
                return []
            
            # Convert query vector to numpy array
            query_np = np.array([query_vector], dtype=np.float32)
            
            # Normalize query vector for cosine similarity
            faiss.normalize_L2(query_np)
            
            # Search the index
            loop = asyncio.get_event_loop()
            distances, indices = await loop.run_in_executor(
                None,
                lambda: self.indexes[collection_name].search(query_np, top_k)
            )
            
            # Process results
            results = []
            id_to_faiss_id = {
                meta_entry["faiss_id"]: doc_id
                for doc_id, meta_entry in self.metadata.get(collection_name, {}).items()
            }
            
            for i, idx in enumerate(indices[0]):
                if idx < 0:  # FAISS returns -1 for unused slots
                    continue
                
                # Find the document ID for this FAISS ID
                doc_id = None
                for d_id, faiss_id in id_to_faiss_id.items():
                    if faiss_id == idx:
                        doc_id = d_id
                        break
                
                if not doc_id:
                    continue
                
                # Get metadata
                meta = self.metadata.get(collection_name, {}).get(doc_id, {}).get("metadata", {})
                
                # Apply filter if provided
                if filter_dict:
                    matches_filter = True
                    for k, v in filter_dict.items():
                        if k not in meta or meta[k] != v:
                            matches_filter = False
                            break
                    
                    if not matches_filter:
                        continue
                
                # Add to results
                results.append({
                    "id": doc_id,
                    "score": float(distances[0][i]),  # Similarity score
                    "metadata": meta
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching vectors in FAISS: {e}")
            return []
    
    async def get_by_id(
        self,
        collection_name: str,
        ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get vectors by their IDs from FAISS.
        
        Args:
            collection_name: Collection name
            ids: List of document IDs
            
        Returns:
            List of retrieved documents
        """
        try:
            if collection_name not in self.metadata:
                return []
            
            results = []
            for doc_id in ids:
                if doc_id in self.metadata[collection_name]:
                    results.append({
                        "id": doc_id,
                        "metadata": self.metadata[collection_name][doc_id]["metadata"]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting vectors by ID from FAISS: {e}")
            return []
