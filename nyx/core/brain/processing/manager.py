# nyx/core/brain/processing/manager.py
import logging
import asyncio
from typing import Dict, Any, Optional, List
import random

logger = logging.getLogger(__name__)

class ProcessingManager:
    """Manages different processing strategies for the brain"""
    
    def __init__(self, brain):
        self.brain = brain
        self.processors = {}
        self.current_mode = "auto"
        self.mode_switch_history = []
        self.complexity_threshold = {
            "parallel": 0.6,   # Switch to parallel at this complexity
            "distributed": 0.8 # Switch to distributed at this complexity
        }
    
    async def initialize(self):
        """Initialize all processors"""
        from nyx.core.brain.processing.serial import SerialProcessor
        from nyx.core.brain.processing.parallel import ParallelProcessor
        from nyx.core.brain.processing.distributed import DistributedProcessor
        from nyx.core.brain.processing.reflexive import ReflexiveProcessor
        from nyx.core.brain.processing.agent import AgentProcessor
        
        # Initialize processors
        self.processors = {
            "serial": SerialProcessor(self.brain),
            "parallel": ParallelProcessor(self.brain),
            "distributed": DistributedProcessor(self.brain),
            "reflexive": ReflexiveProcessor(self.brain),
            "agent": AgentProcessor(self.brain),
            "integrated": AgentProcessor(self.brain, integration_mode=True),
        }
        
        # Initialize each processor
        for name, processor in self.processors.items():
            if processor is not None:
                await processor.initialize()
                
        logger.info(f"Processing manager initialized with {len(self.processors)} processors")
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process input using the appropriate processor"""
        context = context or {}
        
        # Determine processing mode
        mode = context.get("processing_mode", self.current_mode)
        
        # Handle auto mode
        if mode == "auto":
            mode = await self._determine_processing_mode(user_input, context)
            
        # Check reflexive first, regardless of mode
        if hasattr(self.brain, "reflexive_system") and self.brain.reflexive_system:
            should_use_reflex, confidence = self.brain.reflexive_system.decision_system.should_use_reflex(
                {"text": user_input}, context, None
            )
            
            if should_use_reflex and confidence > 0.7:
                mode = "reflexive"
                logger.info(f"Switching to reflexive processing (confidence: {confidence:.2f})")
        
        # Use the appropriate processor
        processor = self.processors.get(mode)
        if not processor:
            logger.warning(f"Unknown processing mode '{mode}', falling back to serial")
            processor = self.processors["serial"]
            
        # Track mode switches
        if mode != self.current_mode:
            self.mode_switch_history.append({
                "from": self.current_mode,
                "to": mode,
                "timestamp": datetime.datetime.now().isoformat(),
                "input_length": len(user_input),
                "context": str(context)[:100] + "..." if context and len(str(context)) > 100 else str(context)
            })
            self.current_mode = mode
            
        logger.info(f"Processing input using {mode} mode")
        return await processor.process_input(user_input, context)
    
    async def _determine_processing_mode(self, user_input: str, context: Dict[str, Any]) -> str:
        """Determine optimal processing mode based on input complexity"""
        # Define thresholds
        input_length_threshold = 100  # Characters
        
        # Calculate complexity score based on input and context
        complexity_score = 0.0
        
        # 1. Input length
        input_length_factor = min(1.0, len(user_input) / 500.0)  # Normalize to [0,1]
        complexity_score += input_length_factor * 0.3  # 30% weight
        
        # 2. Content complexity
        words = user_input.lower().split()
        unique_words = len(set(words))
        word_complexity = min(1.0, unique_words / 50.0)  # Normalize to [0,1]
        
        punctuation_count = sum(1 for c in user_input if c in "?!.,;:()[]{}\"'")
        punctuation_complexity = min(1.0, punctuation_count / 20.0)  # Normalize to [0,1]
        
        content_complexity = (word_complexity * 0.7 + punctuation_complexity * 0.3)
        complexity_score += content_complexity * 0.3  # 30% weight
        
        # 3. Context complexity
        context_complexity = 0.0
        if context:
            context_complexity = min(1.0, len(str(context)) / 1000.0)
        complexity_score += context_complexity * 0.2  # 20% weight
        
        # 4. History/state complexity
        history_complexity = min(1.0, self.brain.interaction_count / 50.0)
        complexity_score += history_complexity * 0.2  # 20% weight
        
        # Check for agent indicators first
        agent_indicators = [
            "roleplay", "role play", "acting", "pretend", "scenario",
            "imagine", "fantasy", "act as", "play as", "in-character",
            "story", "scene", "setting", "character", "plot",
            "describe", "tell me about", "what happens",
            "picture", "image", "draw", "show me", "visualize"
        ]
        
        if any(indicator in user_input.lower() for indicator in agent_indicators):
            logger.info(f"Detected agent indicator in input, using agent mode")
            return "agent"
        
        # Select mode based on complexity score
        if complexity_score < self.complexity_threshold["parallel"]:
            # Low complexity, use serial processing
            return "serial"
        elif complexity_score < self.complexity_threshold["distributed"]:
            # Medium complexity, use parallel processing
            return "parallel"
        else:
            # High complexity, use distributed processing
            return "distributed"
    
    async def generate_response(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a response using the appropriate processor"""
        context = context or {}
        
        # Process the input first
        processing_result = await self.process_input(user_input, context)
        
        # Use the same processor to generate the response
        mode = self.current_mode
        processor = self.processors.get(mode, self.processors["serial"])
        
        return await processor.generate_response(user_input, processing_result, context)
    
    async def set_processing_mode(self, mode: str, reason: str = None) -> Dict[str, Any]:
        """
        Set the processing mode for the brain with tracking
        
        Args:
            mode: Processing mode to use
            reason: Reason for the mode change
            
        Returns:
            Status of the mode change
        """
        valid_modes = list(self.processors.keys()) + ["auto"]
        
        if mode not in valid_modes:
            return {
                "success": False,
                "error": f"Invalid mode: {mode}. Valid modes are: {valid_modes}"
            }
        
        # Track previous mode for reflection
        previous_mode = self.current_mode
        
        # Update mode
        self.current_mode = mode
        
        # Track mode change
        self.mode_switch_history.append({
            "from": previous_mode,
            "to": mode,
            "timestamp": datetime.datetime.now().isoformat(),
            "reason": reason
        })
        
        # Create a memory of this decision if reason is provided
        if reason and hasattr(self.brain, "memory_core"):
            await self.brain.memory_core.add_memory(
                memory_text=f"Changed processing mode from {previous_mode} to {mode} because: {reason}",
                memory_type="reflection",
                significance=6,
                tags=["meta_cognition", "processing_mode", mode],
                metadata={
                    "previous_mode": previous_mode,
                    "new_mode": mode,
                    "reason": reason
                }
            )
            
        logger.info(f"Processing mode set to {mode}{f' ({reason})' if reason else ''}")
        
        return {
            "success": True,
            "mode": mode,
            "previous_mode": previous_mode
        }
