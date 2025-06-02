# nyx/core/memory_core_compat.py
"""
Legacy compatibility wrapper for old memory core imports.
This file provides backward compatibility for code that uses old function/class names.
"""

# Import everything from the refactored module
from nyx.core.memory_core import *

# ==================== Legacy Aliases ====================

# Class name aliases (if any were renamed)
MemoryCoreContext = MemoryContext  # Old name -> New name

# ==================== Legacy Function Wrappers ====================

# If old code expects different function signatures or names, wrap them here

async def create_reflection_from_memories(
    ctx,
    topic: str = None,
    memory_ids: List[str] = None
) -> Dict[str, Any]:
    """Legacy wrapper for create_reflection"""
    return await create_reflection(ctx, topic, memory_ids)

async def create_abstraction_from_memories(
    ctx,
    memory_ids: List[str],
    pattern_type: str = "behavior"
) -> Dict[str, Any]:
    """Legacy wrapper for create_abstraction"""
    return await create_abstraction(ctx, memory_ids, pattern_type)

async def construct_narrative_from_memories(
    ctx,
    topic: str,
    chronological: bool = True,
    limit: int = 5
) -> Dict[str, Any]:
    """Legacy wrapper for construct_narrative"""
    return await construct_narrative(ctx, topic, chronological, limit)

async def apply_memory_decay(ctx) -> Dict[str, Any]:
    """Legacy wrapper for apply_decay"""
    return await apply_decay(ctx)

async def consolidate_memory_clusters(ctx) -> Dict[str, Any]:
    """Legacy wrapper for consolidate_memories"""
    return await consolidate_memories(ctx)

# ==================== Legacy Standalone Functions ====================

# If old code expects different parameter names or defaults

async def store_memory(
    text: str,
    type: str = "observation",
    scope: str = "game",
    importance: int = 5,
    tags: List[str] = None,
    metadata: Dict[str, Any] = None,
    user_id: Optional[int] = None,
    conversation_id: Optional[int] = None
) -> str:
    """Legacy wrapper with old parameter names"""
    return await add_memory(
        memory_text=text,
        memory_type=type,
        memory_scope=scope,
        significance=importance,  # Note: old code might use 'importance'
        tags=tags,
        metadata=metadata,
        user_id=user_id,
        conversation_id=conversation_id
    )

async def query_memories(
    query_text: str,
    types: List[str] = None,
    scopes: List[str] = None,
    max_results: int = 10,
    min_importance: int = 3,
    include_archived: bool = False,
    tags: List[str] = None,
    entities: List[str] = None,
    emotional_state: Dict[str, Any] = None,
    user_id: Optional[int] = None,
    conversation_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Legacy wrapper with old parameter names"""
    return await retrieve_memories(
        query=query_text,
        memory_types=types,
        scopes=scopes,
        limit=max_results,  # Note: old code might use 'max_results'
        min_significance=min_importance,  # Note: old code might use 'min_importance'
        include_archived=include_archived,
        tags=tags,
        entities=entities,
        emotional_state=emotional_state,
        user_id=user_id,
        conversation_id=conversation_id
    )

# ==================== Legacy Model Aliases ====================

# If old code expects different model names
MemoryInput = MemoryCreateParams  # Example alias
MemorySearchParams = MemoryQuery  # Example alias

# ==================== Deprecated Functions ====================

async def _apply_reconsolidation(ctx, memory):
    """Deprecated - reconsolidation now happens automatically in search_memories"""
    import warnings
    warnings.warn(
        "_apply_reconsolidation is deprecated. Reconsolidation now happens automatically.",
        DeprecationWarning,
        stacklevel=2
    )
    return memory

async def _apply_reconsolidation_to_results(ctx, memories):
    """Deprecated - reconsolidation now happens automatically in search_memories"""
    import warnings
    warnings.warn(
        "_apply_reconsolidation_to_results is deprecated. Reconsolidation now happens automatically.",
        DeprecationWarning,
        stacklevel=2
    )
    return memories

# ==================== Complete Export List ====================

__all__ = [
    # ===== Original Exports from memory_core =====
    # Classes
    'MemoryCoreAgents',
    'BrainMemoryCore',
    'Memory',
    'MemoryMetadata',
    'EmotionalMemoryContext',
    'MemorySchema',
    'MemoryContext',
    'MemoryStorage',
    
    # Input/Output Models
    'MemoryCreateParams',
    'MemoryUpdateParams',
    'MemoryQuery',
    'MemoryRetrieveResult',
    'MemoryMaintenanceResult',
    'NarrativeResult',
    
    # Standalone functions
    'add_memory',
    'retrieve_memories',
    
    # Tool functions
    'create_memory',
    'search_memories',
    'update_memory',
    'delete_memory',
    'get_memory',
    'get_memory_details',
    'archive_memory',
    'unarchive_memory',
    'crystallize_memory',
    'mark_as_consolidated',
    'create_reflection',
    'create_abstraction',
    'create_semantic_memory',
    'apply_decay',
    'consolidate_memories',
    'get_memory_stats',
    'generate_conversational_recall',
    'construct_narrative',
    'run_maintenance',
    'retrieve_memories_with_formatting',
    'retrieve_relevant_experiences',
    'detect_schema_from_memories',
    'reflect_on_memories',
    
    # ===== Legacy Aliases =====
    'MemoryCoreContext',  # -> MemoryContext
    
    # ===== Legacy Function Names =====
    'create_reflection_from_memories',  # -> create_reflection
    'create_abstraction_from_memories',  # -> create_abstraction
    'construct_narrative_from_memories',  # -> construct_narrative
    'apply_memory_decay',  # -> apply_decay
    'consolidate_memory_clusters',  # -> consolidate_memories
    
    # ===== Legacy Standalone Functions =====
    'store_memory',  # -> add_memory with old param names
    'query_memories',  # -> retrieve_memories with old param names
    
    # ===== Legacy Model Names =====
    'MemoryInput',  # -> MemoryCreateParams
    'MemorySearchParams',  # -> MemoryQuery
    
    # ===== Deprecated (but still available) =====
    '_apply_reconsolidation',
    '_apply_reconsolidation_to_results',
]

# ==================== Monkey Patches (if needed) ====================

# If some old code expects attributes on classes that don't exist anymore,
# you can add them here. For example:

# Add old attribute name to MemoryContext if needed
# MemoryContext.memory_store = property(lambda self: self.storage)

# ==================== Import Hook (optional) ====================

# If you want to intercept and redirect imports automatically:
import sys
import warnings

class MemoryCoreCompatImporter:
    """Custom importer that redirects old imports to new ones"""
    
    def find_module(self, fullname, path=None):
        # Intercept specific old module names if they were renamed
        if fullname == "nyx.core.memory_core_old":
            warnings.warn(
                "nyx.core.memory_core_old is deprecated. Use nyx.core.memory_core instead.",
                DeprecationWarning,
                stacklevel=2
            )
            return self
        return None
    
    def load_module(self, fullname):
        # Redirect to new module
        if fullname == "nyx.core.memory_core_old":
            return sys.modules.get("nyx.core.memory_core_compat")
        return None

# Uncomment to activate the import hook:
# sys.meta_path.append(MemoryCoreCompatImporter())

# ==================== Compatibility Notes ====================

"""
MIGRATION GUIDE:

1. Class Renames:
   - MemoryCoreContext -> MemoryContext

2. Function Renames:
   - create_reflection_from_memories -> create_reflection
   - create_abstraction_from_memories -> create_abstraction
   - construct_narrative_from_memories -> construct_narrative
   - apply_memory_decay -> apply_decay
   - consolidate_memory_clusters -> consolidate_memories

3. Parameter Renames:
   - importance -> significance
   - max_results -> limit
   - min_importance -> min_significance
   - query_text -> query
   - types -> memory_types

4. Deprecated Functions:
   - _apply_reconsolidation (now automatic)
   - _apply_reconsolidation_to_results (now automatic)

5. New Features Available:
   - Query caching (automatic)
   - Tracing support (automatic)
   - Enhanced schema detection
   - Crystallization support
   - Hierarchical memory levels

To use this compatibility layer:
1. Replace: from nyx.core.memory_core import ...
   With:     from nyx.core.memory_core_compat import ...

2. Or add at the top of files with old imports:
   import nyx.core.memory_core_compat as memory_core
"""
