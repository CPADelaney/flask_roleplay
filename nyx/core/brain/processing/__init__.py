# nyx/core/brain/processing/__init__.py
from nyx.core.brain.processing.unified_processor import UnifiedProcessor
from nyx.core.brain.processing.legacy_manager import LegacyProcessingManager

# Export with legacy names for compatibility
ProcessingManager = LegacyProcessingManager
BaseProcessor = UnifiedProcessor
SerialProcessor = UnifiedProcessor
ParallelProcessor = UnifiedProcessor
DistributedProcessor = UnifiedProcessor
ReflexiveProcessor = UnifiedProcessor
AgentProcessor = UnifiedProcessor
ModeSelector = None  # No longer needed

__all__ = [
    "UnifiedProcessor",
    "ProcessingManager",
    "BaseProcessor",
    "SerialProcessor",
    "ParallelProcessor", 
    "DistributedProcessor",
    "ReflexiveProcessor",
    "AgentProcessor"
]
