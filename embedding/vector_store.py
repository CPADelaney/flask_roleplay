"""
Vector Store and Embedding Utility Functions

This module provides utilities for generating and working with embeddings
for semantic search in the lore system.
"""

import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Union

from utils.embedding_dimensions import (
    adjust_embedding_vector,
    build_zero_vector,
    get_target_embedding_dimension,
)

try:
    from memory.memory_config import get_memory_config
except Exception:  # pragma: no cover - optional dependency during tests
    get_memory_config = None  # type: ignore
import json

logger = logging.getLogger(__name__)

try:
    _MEMORY_CONFIG: Dict[str, Any] = get_memory_config() if get_memory_config else {}
except Exception:  # pragma: no cover - configuration failures should not block tests
    _MEMORY_CONFIG = {}

# Embeddings produced by this helper should align with the configured target
# dimension for the application so downstream pgvector writes do not fail.
EMBEDDING_DIMENSIONS = get_target_embedding_dimension(config=_MEMORY_CONFIG)

async def generate_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector for the given text.
    
    In a production environment, this would call an embedding service
    like OpenAI's text-embedding-ada-002 or a local model.
    
    For this implementation, we create a deterministic mock embedding
    that's consistent for the same input text.
    
    Args:
        text: The text to generate an embedding for
        
    Returns:
        A list of floats representing the embedding vector
    """
    logger.info(f"Generating embedding for text: {text[:50]}...")
    
    try:
        # Create a deterministic seed from the text
        seed = sum(ord(c) for c in text) % 10000
        
        # Use numpy to generate a deterministic vector
        np.random.seed(seed)
        
        # Generate a normalized embedding vector
        embedding = np.random.normal(0, 1, EMBEDDING_DIMENSIONS)
        normalized_embedding = embedding / np.linalg.norm(embedding)
        
        # Convert to list for JSON serialization and normalise length
        return adjust_embedding_vector(normalized_embedding.tolist(), EMBEDDING_DIMENSIONS)
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        # Return a zero vector as fallback
        return build_zero_vector(EMBEDDING_DIMENSIONS)

async def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        return float(similarity)
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        return 0.0

async def find_most_similar(
    query_embedding: List[float],
    candidate_embeddings: Dict[str, List[float]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Find the most similar embeddings to a query embedding.
    
    Args:
        query_embedding: The query embedding vector
        candidate_embeddings: Dictionary of {id: embedding_vector}
        top_k: Number of top results to return
        
    Returns:
        List of {id, similarity} dictionaries sorted by similarity
    """
    try:
        results = []
        
        for id, embedding in candidate_embeddings.items():
            similarity = await compute_similarity(query_embedding, embedding)
            results.append({"id": id, "similarity": similarity})
        
        # Sort by similarity in descending order
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top_k results
        return results[:top_k]
    except Exception as e:
        logger.error(f"Error finding similar embeddings: {e}")
        return []
