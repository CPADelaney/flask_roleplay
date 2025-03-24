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
