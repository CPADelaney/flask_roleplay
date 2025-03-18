import pytest
import time
import asyncio
import random
from typing import List, Dict
import numpy as np

from memory.core import UnifiedMemoryManager
from memory.maintenance import MemoryMaintenance

@pytest.mark.performance
async def test_memory_insertion_performance():
    """Test bulk memory insertion performance."""
    memory_manager = UnifiedMemoryManager()
    num_memories = 1000
    start_time = time.time()
    
    # Generate test memories
    memories = [
        {
            "content": f"Test memory {i}",
            "importance": random.random(),
            "embedding": np.random.rand(384).tolist(),  # Assuming 384-dim embeddings
            "metadata": {"test_id": i}
        }
        for i in range(num_memories)
    ]
    
    # Batch insert memories
    await memory_manager.batch_insert_memories(memories)
    
    duration = time.time() - start_time
    assert duration < 5.0, f"Bulk insertion took {duration:.2f}s, should be under 5s"
    
    # Verify insertion
    count = await memory_manager.get_memory_count()
    assert count >= num_memories

@pytest.mark.performance
async def test_memory_query_performance():
    """Test memory query performance."""
    memory_manager = UnifiedMemoryManager()
    
    # Setup: Insert test memories
    num_memories = 100
    test_memories = []
    for i in range(num_memories):
        memory = {
            "content": f"Performance test memory {i}",
            "importance": random.random(),
            "embedding": np.random.rand(384).tolist(),
            "metadata": {"test_id": i}
        }
        memory_id = await memory_manager.insert_memory(memory)
        test_memories.append(memory_id)
    
    # Test 1: Vector similarity search
    start_time = time.time()
    query_vector = np.random.rand(384).tolist()
    results = await memory_manager.search_similar_memories(
        query_vector,
        limit=10
    )
    vector_search_time = time.time() - start_time
    assert vector_search_time < 0.5, f"Vector search took {vector_search_time:.2f}s, should be under 0.5s"
    
    # Test 2: Filtered query
    start_time = time.time()
    filtered_results = await memory_manager.query_memories(
        importance_threshold=0.5,
        limit=50
    )
    filter_time = time.time() - start_time
    assert filter_time < 0.2, f"Filtered query took {filter_time:.2f}s, should be under 0.2s"

@pytest.mark.performance
async def test_memory_cleanup_performance():
    """Test memory cleanup performance."""
    maintenance = MemoryMaintenance()
    memory_manager = UnifiedMemoryManager()
    
    # Setup: Insert old memories
    num_memories = 500
    old_date = time.time() - (40 * 24 * 60 * 60)  # 40 days old
    
    for i in range(num_memories):
        memory = {
            "content": f"Old test memory {i}",
            "importance": 0.3 if i % 2 == 0 else 0.7,
            "last_accessed": old_date,
            "metadata": {"test_id": i}
        }
        await memory_manager.insert_memory(memory)
    
    # Run cleanup
    start_time = time.time()
    stats = await maintenance.cleanup_old_memories()
    duration = time.time() - start_time
    
    assert duration < 3.0, f"Cleanup took {duration:.2f}s, should be under 3s"
    assert stats["deleted_count"] > 0, "Should have deleted some memories"
    assert stats["archived_count"] > 0, "Should have archived some memories"

@pytest.mark.performance
async def test_concurrent_memory_operations():
    """Test performance under concurrent operations."""
    memory_manager = UnifiedMemoryManager()
    
    async def worker(worker_id: int, num_ops: int):
        for i in range(num_ops):
            memory = {
                "content": f"Concurrent test memory {worker_id}-{i}",
                "importance": random.random(),
                "embedding": np.random.rand(384).tolist(),
                "metadata": {"worker_id": worker_id, "op_id": i}
            }
            await memory_manager.insert_memory(memory)
    
    num_workers = 5
    ops_per_worker = 100
    start_time = time.time()
    
    # Run concurrent workers
    workers = [worker(i, ops_per_worker) for i in range(num_workers)]
    await asyncio.gather(*workers)
    
    duration = time.time() - start_time
    assert duration < 10.0, f"Concurrent operations took {duration:.2f}s, should be under 10s"
    
    # Verify results
    total_memories = await memory_manager.get_memory_count()
    assert total_memories >= num_workers * ops_per_worker 