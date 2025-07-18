# utils/embedding_service.py

"""
Embedding service for semantic similarity calculations.
"""

import logging
import numpy as np
from typing import List, Union
import hashlib
import pickle
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Cache for embeddings to reduce API calls
EMBEDDING_CACHE = {}
CACHE_FILE = "embedding_cache.pkl"
CACHE_TTL = timedelta(days=7)  # Cache embeddings for 7 days
LAST_CACHE_SAVE = datetime.now()

def _hash_text(text: str) -> str:
    """Create a hash of the text for cache keys."""
    return hashlib.md5(text.encode()).hexdigest()

def _load_cache():
    """Load embedding cache from disk."""
    global EMBEDDING_CACHE
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
                # Clean expired cache entries
                now = datetime.now()
                clean_cache = {
                    key: (vec, timestamp) 
                    for key, (vec, timestamp) in cache_data.items()
                    if now - timestamp < CACHE_TTL
                }
                EMBEDDING_CACHE = clean_cache
                logger.info(f"Loaded {len(EMBEDDING_CACHE)} embedding cache entries")
    except Exception as e:
        logger.error(f"Error loading embedding cache: {e}")

def _save_cache():
    """Save embedding cache to disk."""
    global LAST_CACHE_SAVE
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(EMBEDDING_CACHE, f)
        LAST_CACHE_SAVE = datetime.now()
        logger.info(f"Saved {len(EMBEDDING_CACHE)} embedding cache entries")
    except Exception as e:
        logger.error(f"Error saving embedding cache: {e}")

# Load cache on module import
_load_cache()

async def get_embedding(text: str) -> np.ndarray:
    """
    Get embedding vector for a text.
    
    Args:
        text: Text to get embedding for
        
    Returns:
        numpy array with embedding vectors
    """
    text_hash = _hash_text(text)
    
    # Check cache
    if text_hash in EMBEDDING_CACHE:
        vector, timestamp = EMBEDDING_CACHE[text_hash]
        # Check if cache entry is still valid
        if datetime.now() - timestamp < CACHE_TTL:
            return vector
    
    try:
        # In a production implementation, you would call an actual embedding service
        # For this example, we'll simulate with a basic embedding approach
        
        from logic.chatgpt_integration import get_text_embedding
        
        # Get embedding from LLM service
        embedding = await get_text_embedding(text)
        
        # Cache the result
        EMBEDDING_CACHE[text_hash] = (embedding, datetime.now())
        
        # Periodically save cache to disk
        if (datetime.now() - LAST_CACHE_SAVE).total_seconds() > 3600:  # Save once per hour
            _save_cache()
            
        return embedding
        
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        # Return a zero vector as fallback
        return np.zeros(1536)  # Standard embedding dimension

async def get_batch_embeddings(texts: List[str]) -> List[np.ndarray]:
    """
    Get embeddings for multiple texts in batch.
    
    Args:
        texts: List of texts to get embeddings for
        
    Returns:
        List of numpy arrays with embedding vectors
    """
    results = []
    
    for text in texts:
        embedding = await get_embedding(text)
        results.append(embedding)
    
    return results
