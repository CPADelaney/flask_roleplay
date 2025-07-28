# memory/memory_service.py

"""
Memory Embedding Service using LangChain

This module provides memory embedding and retrieval services using LangChain
with multiple vector store backends (ChromaDB, FAISS, or Qdrant).
"""

import os
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import uuid

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceHub, HuggingFacePipeline
from langchain.output_parsers import PydanticOutputParser


# Import vector store implementations - dynamic for flexibility
from memory.chroma_vector_store import ChromaVectorDatabase
from memory.faiss_vector_store import FAISSVectorDatabase
from context.optimized_db import QdrantDatabase, create_vector_database

# Configure logging
logger = logging.getLogger(__name__)

class MemoryEmbeddingService:
    """
    Service for embedding and retrieving memories using LangChain with
    multiple vector store backends.
    """
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        vector_store_type: str = "chroma",
        embedding_model: str = "local",
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
        self.embedding_model = embedding_model
        self.config = config or {}
        
        # Set persist directory
        if not persist_directory:
            persist_base = self.config.get("persist_base_dir", "./vector_stores")
            self.persist_directory = f"{persist_base}/{vector_store_type}/{user_id}_{conversation_id}"
        else:
            self.persist_directory = persist_directory
        
        # Initialize variables to be set up later
        self.vector_db = None
        self.embeddings = None
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
            
            # 2. Set up the vector store
            await self._setup_vector_store()
            
            self.initialized = True
            logger.info(f"Memory embedding service initialized with {self.vector_store_type} and {self.embedding_model} embeddings")
            
        except Exception as e:
            logger.error(f"Error initializing memory embedding service: {e}")
            raise
    
    async def _setup_embeddings(self) -> None:
        """Set up the embedding model."""
        # Use asyncio.to_thread for potentially blocking operations
        loop = asyncio.get_event_loop()
        
        if self.embedding_model == "openai":
            # OpenAI embeddings require an API key
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI embeddings")
            
            self.embeddings = await loop.run_in_executor(
                None,
                lambda: OpenAIEmbeddings(
                    openai_api_key=openai_api_key,
                    model="text-embedding-3-small"  # Modern, efficient model
                )
            )
        else:
            # Default to local HuggingFace embeddings
            model_name = self.config.get("hf_embedding_model", "all-MiniLM-L6-v2")
            
            self.embeddings = await loop.run_in_executor(
                None,
                lambda: HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
            )
    
    async def _setup_vector_store(self) -> None:
        """Set up the vector store."""
        if self.vector_store_type == "chroma":
            self.vector_db = ChromaVectorDatabase(
                persist_directory=self.persist_directory,
                config=self.config
            )
        elif self.vector_store_type == "faiss":
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
            self.vector_db = create_vector_database(vector_db_config)
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
        
        # Initialize the vector database
        await self.vector_db.initialize()
        
        # Ensure collections exist
        for collection_name in self.collection_mapping.values():
            # Get embedding dimension based on model
            dimension = 1536  # Default for all-MiniLM-L6-v2
            if self.embedding_model == "openai":
                dimension = 1536  # Default for text-embedding-3-small
            
            await self.vector_db.create_collection(collection_name, dimension)
    
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
            # Use asyncio.to_thread to make the embedding operation non-blocking
            loop = asyncio.get_event_loop()
            
            # LangChain's embedding interface
            embedding = await loop.run_in_executor(
                None,
                lambda: self.embeddings.embed_query(text)
            )
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return an empty vector as fallback
            return self._get_empty_embedding()
    
    def _get_empty_embedding(self) -> List[float]:
        """Get an empty embedding vector with the correct dimension."""
        if self.embedding_model == "openai":
            return [0.0] * 1536  # OpenAI dimension
        else:
            return [0.0] * 1536  # Default for most HuggingFace models
    
    async def add_memory(
        self,
        text: str,
        metadata: Dict[str, Any],
        entity_type: str = "memory",
        embedding: Optional[List[float]] = None
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
        
        # Generate embedding if not provided
        if embedding is None:
            embedding = await self.generate_embedding(text)
        
        # Add to vector store
        success = await self.vector_db.insert_vectors(
            collection_name=collection_name,
            ids=[memory_id],
            vectors=[embedding],
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
        
        # Generate query embedding
        query_embedding = await self.generate_embedding(query_text)
        
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
