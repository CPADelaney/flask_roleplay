# nyx/core/brain/processing/legacy_manager.py
"""Legacy compatibility wrapper for ProcessingManager"""
import logging
from typing import Dict, Any
from nyx.core.brain.processing.unified_processor import UnifiedProcessor

logger = logging.getLogger(__name__)

class LegacyProcessingManager:
    """Compatibility wrapper that mimics the old ProcessingManager interface"""
    
    def __init__(self, brain):
        self.brain = brain
        self.processor = UnifiedProcessor(brain)
        self.current_mode = "unified"
        self.mode_switch_history = []
        
        # Create fake processors dict for compatibility
        self.processors = {
            "serial": self.processor,
            "parallel": self.processor,
            "distributed": self.processor,
            "reflexive": self.processor,
            "agent": self.processor,
            "integrated": self.processor
        }
    
    async def initialize(self):
        """Initialize the unified processor"""
        await self.processor.initialize()
        logger.info("Legacy processing manager initialized with unified processor")
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process input using unified processor"""
        result = await self.processor.process_input(user_input, context)
        
        # Add legacy mode info
        if "processing_mode" not in result:
            result["processing_mode"] = "unified"
        
        return result
    
    async def generate_response(self, user_input: str, processing_result: Dict[str, Any],
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response using unified processor"""
        return await self.processor.generate_response(user_input, processing_result, context)
    
    async def set_processing_mode(self, mode: str, reason: str = None) -> Dict[str, Any]:
        """Compatibility method - mode setting is now automatic"""
        self.mode_switch_history.append({
            "from": self.current_mode,
            "to": "unified",
            "timestamp": datetime.datetime.now().isoformat(),
            "reason": reason or "Unified processing handles all modes dynamically"
        })
        
        return {
            "success": True,
            "mode": "unified",
            "previous_mode": self.current_mode,
            "message": "Processing mode is now unified and automatic"
        }
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "current_mode": "unified",
            "available_modes": ["unified", "auto"],
            "mode_switches": len(self.mode_switch_history),
            "recent_switches": self.mode_switch_history[-5:],
            "message": "All processing is now handled by unified orchestrator"
        }
