# nyx/core/brain/processing/manager.py
import logging
import datetime
from typing import Dict, Any, Optional, List

from nyx.core.brain.processing.base_processor import BaseProcessor

logger = logging.getLogger(__name__)

class ProcessingManager:
    """Manages different processing strategies"""
    
    def __init__(self, brain):
        self.brain = brain
        self.processors = {}
        self.current_mode = "auto"
        self.mode_switch_history = []
        self.mode_selector = None
        
    async def initialize(self):
        """Initialize all processors"""
        try:
            from nyx.core.brain.processing.serial import SerialProcessor
            from nyx.core.brain.processing.parallel import ParallelProcessor
            from nyx.core.brain.processing.distributed import DistributedProcessor
            from nyx.core.brain.processing.reflexive import ReflexiveProcessor
            from nyx.core.brain.processing.agent import AgentProcessor
            from nyx.core.brain.processing.mode_selector import ModeSelector
            
            # Initialize processors
            self.processors = {
                "serial": SerialProcessor(self.brain),
                "parallel": ParallelProcessor(self.brain),
                "distributed": DistributedProcessor(self.brain),
                "reflexive": ReflexiveProcessor(self.brain),
                "agent": AgentProcessor(self.brain),
                "integrated": AgentProcessor(self.brain, integration_mode=True)
            }
            
            # Initialize each processor
            for name, processor in self.processors.items():
                await processor.initialize()
            
            # Initialize mode selector
            self.mode_selector = ModeSelector(self.brain)
            
            logger.info(f"Processing manager initialized with {len(self.processors)} processors")
            
        except Exception as e:
            logger.error(f"Error initializing processing manager: {str(e)}")
            # Ensure at least serial processor exists
            from nyx.core.brain.processing.serial import SerialProcessor
            self.processors = {"serial": SerialProcessor(self.brain)}
            await self.processors["serial"].initialize()
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process input using appropriate processor"""
        context = context or {}
        
        try:
            # Determine processing mode
            mode = context.get("processing_mode", self.current_mode)
            
            if mode == "auto" and self.mode_selector:
                mode = await self.mode_selector.determine_processing_mode(user_input, context)
            elif mode == "auto":
                mode = "serial"  # Default fallback
            
            # Get processor
            processor = self.processors.get(mode, self.processors.get("serial"))
            
            # Track mode switch
            if mode != self.current_mode:
                self.mode_switch_history.append({
                    "from": self.current_mode,
                    "to": mode,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "input_length": len(user_input)
                })
                self.current_mode = mode
            
            logger.info(f"Processing input using {mode} mode")
            
            # Process
            result = await processor.process_input(user_input, context)
            result["processing_mode"] = mode
            
            # Update metrics if mode selector available  
            if self.mode_selector:
                self.mode_selector.update_mode_metrics(
                    mode=mode,
                    success=not result.get("error"),
                    response_time=result.get("response_time", 0.0)
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in processing: {str(e)}")
            
            # Fallback to serial
            if "serial" in self.processors:
                return await self.processors["serial"].process_input(user_input, context)
            
            # Last resort error response
            return {
                "error": str(e),
                "processing_mode": "error",
                "user_input": user_input,
                "emotional_state": {},
                "memories": [],
                "memory_count": 0,
                "has_experience": False,
                "response_time": 0.0
            }
    
    async def generate_response(self, user_input: str, processing_result: Dict[str, Any],
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response using appropriate processor"""
        mode = processing_result.get("processing_mode", self.current_mode)
        processor = self.processors.get(mode, self.processors.get("serial"))
        
        try:
            return await processor.generate_response(user_input, processing_result, context)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "message": "I apologize, but I encountered an error generating a response.",
                "response_type": "error",
                "error": str(e)
            }
    
    async def set_processing_mode(self, mode: str, reason: str = None) -> Dict[str, Any]:
        """Set processing mode"""
        if mode not in self.processors and mode != "auto":
            return {
                "success": False,
                "error": f"Invalid mode: {mode}"
            }
        
        previous_mode = self.current_mode
        self.current_mode = mode
        
        self.mode_switch_history.append({
            "from": previous_mode,
            "to": mode,
            "timestamp": datetime.datetime.now().isoformat(),
            "reason": reason,
            "user_initiated": True
        })
        
        logger.info(f"Processing mode set to {mode}")
        
        return {
            "success": True,
            "mode": mode,
            "previous_mode": previous_mode
        }
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = {
            "current_mode": self.current_mode,
            "available_modes": list(self.processors.keys()) + ["auto"],
            "mode_switches": len(self.mode_switch_history),
            "recent_switches": self.mode_switch_history[-5:]
        }
        
        if self.mode_selector:
            mode_usage = await self.mode_selector.analyze_mode_usage()
            stats["mode_usage"] = mode_usage
        
        return stats
