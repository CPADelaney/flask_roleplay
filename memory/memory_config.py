# memory/memory_config.py

"""
Memory System Configuration

This module provides configuration settings for the memory embedding and retrieval system.
"""

import os
from typing import Dict, Any

# Default memory system configuration
DEFAULT_MEMORY_CONFIG = {
    # Vector store settings
    "vector_store": {
        "type": "chroma",  # Options: "chroma", "faiss", "qdrant"
        "persist_base_dir": "./vector_stores",
        "dimension": 1536,  # Default dimension for embeddings
        "similarity_threshold": 0.7,  # Minimum similarity score for retrieval
        "max_results": 10,  # Maximum number of results to return
        "optimized_for_render": True,  # Optimization for Render hosting
        
        # ChromaDB specific settings
        "chroma": {
            "model_name": "all-MiniLM-L6-v2",
            "collection_name": "memories"
        },
        
        # FAISS specific settings
        "faiss": {
            "index_type": "flat",  # Options: flat, ivf, hnsw
            "metric_type": "cosine"  # Options: cosine, l2, ip
        },
        
        # Qdrant specific settings
        "qdrant": {
            "url": os.getenv("QDRANT_URL", "http://localhost:6333"),
            "api_key": os.getenv("QDRANT_API_KEY", ""),
            "prefer_grpc": True
        }
    },
    
    # Embedding model settings
    "embedding": {
        "type": "local",  # Options: "local", "openai"
        "model_name": "all-MiniLM-L6-v2",  # For local embeddings
        "openai_model": "text-embedding-3-small",  # For OpenAI embeddings
        "normalize": True,
        "batch_size": 32,
        "embedding_dim": 1536,  # Dimension of local embeddings
    },
    
    # LLM settings for memory retrieval
    "llm": {
        "type": "openai",  # Options: "openai", "huggingface"
        "openai_model": "gpt-5-nano",  # For OpenAI
        "huggingface_model": "mistralai/Mistral-7B-Instruct-v0.2",  # For HuggingFace
        "temperature": 0.0,  # Low temperature for factual responses
        "max_tokens": 256
    },
    
    # Memory management settings
    "management": {
        "memory_ttl_days": 90,  # Days to keep memories
        "consolidation_interval_days": 7,  # Days between consolidations
        "importance_threshold": 0.5,  # Minimum importance to keep during consolidation
        "max_memories_per_entity": 1000,  # Maximum memories per entity (user, npc, etc.)
        "max_total_memories": 10000,  # Maximum total memories
    },
    
    # Performance settings
    "performance": {
        "max_concurrent_tasks": 5,
        "batch_size": 100,
        "timeout_seconds": 30,
        "cache_ttl_seconds": 300,  # 5 minutes
        "use_async": True,
        "retry_count": 3
    },
    
    # Integration settings
    "integration": {
        "store_message_memories": True,  # Store messages as memories
        "enrich_context_automatically": True,  # Automatically enrich context with memories
        "memory_importance_factors": {
            "ai_response": 0.7,
            "user_input": 0.6,
            "system_event": 0.5,
            "background": 0.3
        }
    }
}

def get_memory_config() -> Dict[str, Any]:
    """
    Get memory system configuration, merging environment variables 
    with default configuration.
    
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_MEMORY_CONFIG.copy()
    
    # Override with environment variables
    if os.getenv("MEMORY_VECTOR_STORE_TYPE"):
        config["vector_store"]["type"] = os.getenv("MEMORY_VECTOR_STORE_TYPE")
    
    if os.getenv("MEMORY_EMBEDDING_TYPE"):
        config["embedding"]["type"] = os.getenv("MEMORY_EMBEDDING_TYPE")
    
    if os.getenv("MEMORY_LLM_TYPE"):
        config["llm"]["type"] = os.getenv("MEMORY_LLM_TYPE")
    
    if os.getenv("MEMORY_TTL_DAYS"):
        config["management"]["memory_ttl_days"] = int(os.getenv("MEMORY_TTL_DAYS"))
    
    if os.getenv("MEMORY_SIMILARITY_THRESHOLD"):
        config["vector_store"]["similarity_threshold"] = float(os.getenv("MEMORY_SIMILARITY_THRESHOLD"))
    
    # Adjust for Render environment if necessary
    if os.getenv("RENDER", "").lower() == "true":
        config["vector_store"]["optimized_for_render"] = True
        
        # Adjust optimization settings for Render
        config["performance"]["max_concurrent_tasks"] = 3
        config["performance"]["batch_size"] = 50
        
        # Use persistent storage path
        if os.getenv("RENDER_PERSIST_DIR"):
            config["vector_store"]["persist_base_dir"] = os.getenv("RENDER_PERSIST_DIR")
    
    return config
