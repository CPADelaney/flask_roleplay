# nyx/core/procedural_memory/__init__.py

# Import main components
from .models import (
    ActionTemplate, ChunkTemplate, ContextPattern, ChunkPrediction,
    ControlMapping, ProcedureTransferRecord, Procedure, StepResult,
    ProcedureStats, TransferStats
)
from .chunk_selection import ContextAwareChunkSelector
from .generalization import ProceduralChunkLibrary
from .manager import ProceduralMemoryManager

# Import enhanced components
from .enhanced import (
    HierarchicalProcedure, ObservationLearner, CausalModel,
    ExecutionStrategy, TemporalProcedureGraph, ProcedureGraph,
    EnhancedProceduralMemoryManager
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

# Demonstration functions
from .manager import demonstrate_cross_game_transfer, demonstrate_procedural_memory

# Version
__version__ = "0.1.0"

from .models import (
    HierarchicalProcedure, 
    ObservationLearner, 
    CausalModel, 
    TemporalNode, 
    TemporalProcedureGraph,
    ProcedureGraph,
    WorkingMemoryController,
    ParameterOptimizer,
    TransferLearningOptimizer
)

from .execution import (
    ExecutionStrategy,
    DeliberateExecutionStrategy,
    AutomaticExecutionStrategy,
    AdaptiveExecutionStrategy,
    StrategySelector
)

from .learning import (
    ObservationLearner,
    CausalModel,
    ProceduralMemoryConsolidator
)

from .temporal import (
    TemporalNode,
    TemporalProcedureGraph,
    ProcedureGraph
)

from .manager import (
    EnhancedProceduralMemoryManager
)

__all__ = [
    # Models
    'HierarchicalProcedure',
    'ObservationLearner',
    'CausalModel',
    'TemporalNode',
    'TemporalProcedureGraph',
    'ProcedureGraph',
    'WorkingMemoryController',
    'ParameterOptimizer',
    'TransferLearningOptimizer',
    
    # Execution
    'ExecutionStrategy',
    'DeliberateExecutionStrategy',
    'AutomaticExecutionStrategy',
    'AdaptiveExecutionStrategy',
    'StrategySelector',
    
    # Learning
    'ObservationLearner',
    'CausalModel',
    'ProceduralMemoryConsolidator',
    
    # Manager
    'EnhancedProceduralMemoryManager'
]
