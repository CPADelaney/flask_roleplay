# nyx/core/procedural_memory/__init__.py

# Import core models
from .models import (
    ActionTemplate, ChunkTemplate, ContextPattern, ChunkPrediction,
    ControlMapping, ProcedureTransferRecord, Procedure, StepResult,
    ProcedureStats, TransferStats, HierarchicalProcedure, 
    CausalModel, WorkingMemoryController, ParameterOptimizer, 
    TransferLearningOptimizer
)

# Import specialized modules
from .chunk_selection import ContextAwareChunkSelector
from .generalization import ProceduralChunkLibrary
from .execution import (
    ExecutionStrategy, DeliberateExecutionStrategy, 
    AutomaticExecutionStrategy, AdaptiveExecutionStrategy,
    StrategySelector
)
from .learning import ObservationLearner, ProceduralMemoryConsolidator
from .temporal import TemporalNode, TemporalProcedureGraph, ProcedureGraph

# Import manager
from .manager import (
    ProceduralMemoryManager, 
    EnhancedProceduralMemoryManager,
    demonstrate_cross_game_transfer, 
    demonstrate_procedural_memory
)

# Import function tools
from .function_tools import (
    add_procedure, execute_procedure, transfer_procedure,
    get_procedure_proficiency, list_procedures, get_transfer_statistics,
    identify_chunking_opportunities, apply_chunking,
    generalize_chunk_from_steps, find_matching_chunks,
    transfer_chunk, transfer_with_chunking, find_similar_procedures,
    refine_step
)

# Version
__version__ = "0.1.0"

__all__ = [
    # Core models
    'ActionTemplate', 'ChunkTemplate', 'ContextPattern', 'ChunkPrediction',
    'ControlMapping', 'ProcedureTransferRecord', 'Procedure', 'StepResult',
    'ProcedureStats', 'TransferStats', 'HierarchicalProcedure',
    'CausalModel', 'WorkingMemoryController', 'ParameterOptimizer',
    'TransferLearningOptimizer',
    
    # Specialized components
    'ContextAwareChunkSelector', 'ProceduralChunkLibrary',
    'ExecutionStrategy', 'DeliberateExecutionStrategy',
    'AutomaticExecutionStrategy', 'AdaptiveExecutionStrategy',
    'StrategySelector', 'ObservationLearner', 'ProceduralMemoryConsolidator',
    'TemporalNode', 'TemporalProcedureGraph', 'ProcedureGraph',
    
    # Managers
    'ProceduralMemoryManager', 'EnhancedProceduralMemoryManager',
    
    # Demo functions
    'demonstrate_cross_game_transfer', 'demonstrate_procedural_memory',
    
    # Function tools
    'add_procedure', 'execute_procedure', 'transfer_procedure',
    'get_procedure_proficiency', 'list_procedures', 'get_transfer_statistics',
    'identify_chunking_opportunities', 'apply_chunking',
    'generalize_chunk_from_steps', 'find_matching_chunks',
    'transfer_chunk', 'transfer_with_chunking', 'find_similar_procedures',
    'refine_step'
]
