# memory/memory_intregration.py

"""
Memory Integration Helper Module

This module provides utility functions to integrate the memory retrieval system
with the existing application architecture and Celery tasks.
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from functools import wraps

# Import memory components
from memory.memory_service import MemoryEmbeddingService
from memory.memory_retriever import MemoryRetrieverAgent

# Import database connection
from db.connection import get_db_connection_context

# Configure logging
logger = logging.getLogger(__name__)

# Global registry to avoid recreating services
_memory_services = {}
_memory_retrievers = {}

async def get_memory_service(
    user_id: int,
    conversation_id: int,
    vector_store_type: str = "chroma",  # or "faiss" or "qdrant"
    embedding_model: str = "local",     # or "openai"
    config: Optional[Dict[str, Any]] = None
) -> MemoryEmbeddingService:
    """
    Get or create a memory service instance.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        vector_store_type: Type of vector store
        embedding_model: Type of embedding model
        config: Optional configuration
        
    Returns:
        MemoryEmbeddingService instance
    """
    global _memory_services
    
    key = f"{user_id}:{conversation_id}:{vector_store_type}:{embedding_model}"
    
    if key not in _memory_services:
        service = MemoryEmbeddingService(
            user_id=user_id,
            conversation_id=conversation_id,
            vector_store_type=vector_store_type,
            embedding_model=embedding_model,
            config=config
        )
        await service.initialize()
        _memory_services[key] = service
    
    return _memory_services[key]

async def get_memory_retriever(
    user_id: int,
    conversation_id: int,
    llm_type: str = "openai",  # or "huggingface"
    vector_store_type: str = "chroma",
    embedding_model: str = "local",
    config: Optional[Dict[str, Any]] = None
) -> MemoryRetrieverAgent:
    """
    Get or create a memory retriever agent.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        llm_type: Type of LLM to use
        vector_store_type: Type of vector store
        embedding_model: Type of embedding model
        config: Optional configuration
        
    Returns:
        MemoryRetrieverAgent instance
    """
    global _memory_retrievers
    
    key = f"{user_id}:{conversation_id}:{llm_type}:{vector_store_type}:{embedding_model}"
    
    if key not in _memory_retrievers:
        # Get or create memory service
        memory_service = await get_memory_service(
            user_id=user_id,
            conversation_id=conversation_id,
            vector_store_type=vector_store_type,
            embedding_model=embedding_model,
            config=config
        )
        
        # Create retriever
        retriever = MemoryRetrieverAgent(
            user_id=user_id,
            conversation_id=conversation_id,
            llm_type=llm_type,
            memory_service=memory_service,
            config=config
        )
        await retriever.initialize()
        _memory_retrievers[key] = retriever
    
    return _memory_retrievers[key]

async def cleanup_memory_services():
    """Close all memory services."""
    global _memory_services
    
    close_tasks = []
    for key, service in list(_memory_services.items()):
        close_tasks.append(asyncio.create_task(service.close()))
    
    if close_tasks:
        await asyncio.gather(*close_tasks, return_exceptions=True)
    
    _memory_services.clear()

async def cleanup_memory_retrievers():
    """Close all memory retrievers."""
    global _memory_retrievers
    
    close_tasks = []
    for key, retriever in list(_memory_retrievers.items()):
        close_tasks.append(asyncio.create_task(retriever.close()))
    
    if close_tasks:
        await asyncio.gather(*close_tasks, return_exceptions=True)
    
    _memory_retrievers.clear()

async def add_memory_from_message(
    user_id: int,
    conversation_id: int,
    message_text: str,
    entity_type: str = "memory",
    metadata: Optional[Dict[str, Any]] = None,
    vector_store_type: str = "chroma",
    embedding_model: str = "local",
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Add a memory from a message.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        message_text: Message text
        entity_type: Entity type
        metadata: Optional metadata
        vector_store_type: Type of vector store
        embedding_model: Type of embedding model
        config: Optional configuration
        
    Returns:
        Memory ID
    """
    # Initialize metadata if not provided
    metadata = metadata or {}
    
    # Get memory service
    memory_service = await get_memory_service(
        user_id=user_id,
        conversation_id=conversation_id,
        vector_store_type=vector_store_type,
        embedding_model=embedding_model,
        config=config
    )
    
    # Add memory
    memory_id = await memory_service.add_memory(
        text=message_text,
        metadata=metadata,
        entity_type=entity_type
    )
    
    return memory_id

async def retrieve_relevant_memories(
    user_id: int,
    conversation_id: int,
    query_text: str,
    entity_types: Optional[List[str]] = None,
    top_k: int = 5,
    threshold: float = 0.7,
    vector_store_type: str = "chroma",
    embedding_model: str = "local",
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve memories relevant to a query.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        query_text: Query text
        entity_types: List of entity types to search
        top_k: Number of results to return
        threshold: Relevance threshold
        vector_store_type: Type of vector store
        embedding_model: Type of embedding model
        config: Optional configuration
        
    Returns:
        List of relevant memories
    """
    # Get memory service
    memory_service = await get_memory_service(
        user_id=user_id,
        conversation_id=conversation_id,
        vector_store_type=vector_store_type,
        embedding_model=embedding_model,
        config=config
    )
    
    # Default to searching all entity types if none specified
    if not entity_types:
        entity_types = ["memory", "npc", "location", "narrative"]
    
    # Collect memories from all entity types
    all_memories = []
    
    for entity_type in entity_types:
        try:
            # Search for memories of this entity type
            memories = await memory_service.search_memories(
                query_text=query_text,
                entity_type=entity_type,
                top_k=top_k,
                fetch_content=True
            )
            
            # Filter by threshold
            memories = [m for m in memories if m["relevance"] >= threshold]
            
            all_memories.extend(memories)
            
        except Exception as e:
            logger.error(f"Error retrieving {entity_type} memories: {e}")
    
    # Sort by relevance
    all_memories.sort(key=lambda x: x["relevance"], reverse=True)
    
    # Limit to top_k overall
    return all_memories[:top_k]

async def analyze_with_memory(
    user_id: int,
    conversation_id: int,
    query_text: str,
    entity_types: Optional[List[str]] = None,
    top_k: int = 5,
    threshold: float = 0.7,
    llm_type: str = "openai",
    vector_store_type: str = "chroma",
    embedding_model: str = "local",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze a query with relevant memories.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        query_text: Query text
        entity_types: List of entity types to search
        top_k: Number of results to return
        threshold: Relevance threshold
        llm_type: Type of LLM to use
        vector_store_type: Type of vector store
        embedding_model: Type of embedding model
        config: Optional configuration
        
    Returns:
        Dictionary with analysis results
    """
    # Get memory retriever
    retriever = await get_memory_retriever(
        user_id=user_id,
        conversation_id=conversation_id,
        llm_type=llm_type,
        vector_store_type=vector_store_type,
        embedding_model=embedding_model,
        config=config
    )
    
    # Retrieve and analyze memories
    result = await retriever.retrieve_and_analyze(
        query=query_text,
        entity_types=entity_types,
        top_k=top_k,
        threshold=threshold
    )
    
    return result

# Function to integrate with background_chat_task
async def enrich_context_with_memories(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Enrich a context dictionary with relevant memories.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        user_input: User input text
        context: Context dictionary to enrich
        
    Returns:
        Enriched context dictionary
    """
    try:
        # Get memory retriever
        retriever = await get_memory_retriever(
            user_id=user_id,
            conversation_id=conversation_id,
            llm_type="openai",  # Use "huggingface" if preferred
            vector_store_type="chroma",  # Or "faiss" or "qdrant"
            embedding_model="local"  # Or "openai"
        )
        
        # Retrieve and analyze memories
        memory_result = await retriever.retrieve_and_analyze(
            query=user_input,
            entity_types=["memory", "npc", "location", "narrative"],
            top_k=5,
            threshold=0.7
        )
        
        # Add to context
        if memory_result["found_memories"]:
            # Get original context memory structure or create it
            if "memories" not in context:
                context["memories"] = []
            
            # Add retrieved memories
            context["memories"].extend(memory_result["memories"])
            
            # Add analysis
            if "memory_analysis" not in context:
                context["memory_analysis"] = {}
            
            context["memory_analysis"] = {
                "primary_theme": memory_result["analysis"].primary_theme,
                "insights": [insight.dict() for insight in memory_result["analysis"].insights],
                "suggested_response": memory_result["analysis"].suggested_response
            }
        
        return context
    
    except Exception as e:
        logger.error(f"Error enriching context with memories: {e}")
        return context  # Return original context if error occurs

# Celery task wrapper
def memory_celery_task(func):
    """Decorator to handle async memory tasks in Celery."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Run the async function in the asyncio event loop
        result = asyncio.run(func(*args, **kwargs))
        return result
    
    return wrapper

# Example Celery task function
@memory_celery_task
async def process_memory_task(user_id, conversation_id, message_text, entity_type="memory"):
    """
    Celery task to process a memory.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        message_text: Message text
        entity_type: Entity type
        
    Returns:
        Dictionary with task result
    """
    try:
        # Add memory to vector store
        memory_id = await add_memory_from_message(
            user_id=user_id,
            conversation_id=conversation_id,
            message_text=message_text,
            entity_type=entity_type
        )
        
        return {
            "success": True,
            "memory_id": memory_id,
            "message": f"Successfully processed memory for user {user_id}, conversation {conversation_id}"
        }
    
    except Exception as e:
        logger.error(f"Error processing memory task: {e}")
        return {
            "success": False,
            "error": str(e)
        }
