"""
Embedding Package

This package provides vector embedding functionality for semantic search
and similarity comparison in the lore system.
"""

from embedding.vector_store import generate_embedding, compute_similarity, find_most_similar

__all__ = ['generate_embedding', 'compute_similarity', 'find_most_similar']
