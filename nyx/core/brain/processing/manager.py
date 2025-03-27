# nyx/core/brain/processing/manager.py

import logging
import asyncio
import datetime
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
            "distributed": 0.8  # Switch to distributed at this complexity
        }
        
        # Initialize mode selector
        self.mode_selector = None
    
    async def initialize(self):
        """Initialize all processors and mode selector"""
        try:
            from nyx.core.brain.processing.serial import SerialProcessor
            from nyx.core.brain.processing.parallel import ParallelProcessor
            from nyx.core.brain.processing.distributed import DistributedProcessor
            from nyx.core.brain.processing.reflexive import ReflexiveProcessor
            from nyx.core.brain.processing.agent import AgentProcessor
            from nyx.core.brain.processing.mode_selector import ModeSelector
            
            # Initialize mode selector
            self.mode_selector = ModeSelector(self.brain)
            
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
                    
            # Register error handler for processor failures
            self._register_error_handlers()
                    
            logger.info(f"Processing manager initialized with {len(self.processors)} processors")
        except Exception as e:
            logger.error(f"Error initializing processing manager: {str(e)}")
            # Initialize at least the serial processor as fallback
            try:
                from nyx.core.brain.processing.serial import SerialProcessor
                self.processors = {"serial": SerialProcessor(self.brain)}
                await self.processors["serial"].initialize()
                logger.info("Initialized serial processor as fallback")
            except Exception as inner_e:
                logger.critical(f"Failed to initialize even fallback processor: {str(inner_e)}")
    
    def _register_error_handlers(self):
        """Register error handlers for processor failures"""
        # Register handler for processor failures if issue tracker is available
        if hasattr(self.brain, "issue_tracker"):
            for processor_name, processor in self.processors.items():
                # Set brain and error tracking references
                if not hasattr(processor, "brain"):
                    processor.brain = self.brain
                
                # Add _handle_error method if not present
                if not hasattr(processor, "_handle_error"):
                    processor._handle_error = self._create_error_handler(processor_name)
    
    def _create_error_handler(self, processor_name):
        """Create an error handler function for a processor"""
        async def handle_error(error, context):
            # Register issue with issue tracker
            if hasattr(self.brain, "issue_tracker"):
                issue_data = {
                    "title": f"Error in {processor_name} processor",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "component": processor_name,
                    "category": "PROCESSING",
                    "severity": "MEDIUM",
                    "context": context
                }
                return await self.brain.issue_tracker.register_issue(issue_data)
            else:
                # Log error if issue tracker not available
                logger.error(f"Error in {processor_name} processor: {str(error)}")
                return {"registered": False, "error": "Issue tracker not available"}
        return handle_error
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process input using the appropriate processor based on mode selection"""
        context = context or {}
        
        try:
            # Determine processing mode
            mode = context.get("processing_mode", self.current_mode)
            
            # Handle auto mode with mode selector
            if mode == "auto" and self.mode_selector is not None:
                mode = await self.mode_selector.determine_processing_mode(user_input, context)
            elif mode == "auto":
                # Fallback if mode selector not available
                mode = await self._determine_processing_mode(user_input, context)
            
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
            
            # Process using selected processor
            start_time = datetime.datetime.now()
            result = await processor.process_input(user_input, context)
            end_time = datetime.datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Update mode metrics if mode selector available
            if self.mode_selector:
                self.mode_selector.update_mode_metrics(
                    mode=mode,
                    success=True,
                    response_time=processing_time
                )
            
            # Add mode information to result
            result["processing_mode"] = mode
            result["processing_time"] = processing_time
            
            return result
        except Exception as e:
            logger.error(f"Error in processing input: {str(e)}")
            
            # Update mode metrics if mode selector available
            if self.mode_selector:
                self.mode_selector.update_mode_metrics(
                    mode=mode if 'mode' in locals() else self.current_mode,
                    success=False,
                    response_time=0.0
                )
            
            # Handle error with processor's error handler if available
            try:
                processor = self.processors.get("serial")
                if processor and hasattr(processor, "_handle_error"):
                    await processor._handle_error(e, {
                        "user_input": user_input,
                        "context": context,
                        "mode": mode if 'mode' in locals() else "unknown"
                    })
                
                # Fall back to serial processing
                logger.info(f"Falling back to serial processing after error")
                return await self.processors["serial"].process_input(user_input, context)
            except Exception as fallback_error:
                # If even the fallback fails, return a basic error result
                logger.critical(f"Fallback processing failed: {str(fallback_error)}")
                return {
                    "error": f"Processing failed: {str(e)}. Fallback also failed: {str(fallback_error)}",
                    "processing_mode": "error",
                    "user_input": user_input
                }
    
    async def _determine_processing_mode(self, user_input: str, context: Dict[str, Any]) -> str:
            """Legacy method to determine optimal processing mode (used as fallback)"""
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
                if "integrated" in self.processors:
                    return "integrated"
                return "agent"
            
            # Check reflexive first, regardless of mode
            if hasattr(self.brain, "reflexive_system") and self.brain.reflexive_system:
                should_use_reflex, confidence = self.brain.reflexive_system.decision_system.should_use_reflex(
                    {"text": user_input}, context, None
                )
                
                if should_use_reflex and confidence > 0.7:
                    logger.info(f"Detected reflex pattern, using reflexive mode (confidence: {confidence:.2f})")
                    return "reflexive"
            
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
    
    async def generate_response(self, user_input: str, processing_result: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a response using the appropriate processor"""
        context = context or {}
        
        # Get the mode that was used for processing
        mode = processing_result.get("processing_mode", self.current_mode)
        
        # Use the same processor to generate the response
        processor = self.processors.get(mode, self.processors["serial"])
        
        try:
            # Generate response
            start_time = datetime.datetime.now()
            result = await processor.generate_response(user_input, processing_result, context)
            end_time = datetime.datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # Add response timing information
            result["response_time"] = response_time
            result["processing_mode"] = mode
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response with {mode} processor: {str(e)}")
            
            # Handle error with processor's error handler if available
            if hasattr(processor, "_handle_error"):
                await processor._handle_error(e, {
                    "user_input": user_input,
                    "context": context,
                    "mode": mode,
                    "phase": "response_generation"
                })
            
            # Fall back to serial processing
            if mode != "serial":
                logger.info(f"Falling back to serial response generation after error in {mode} mode")
                return await self.processors["serial"].generate_response(user_input, processing_result, context)
            else:
                # If serial processing failed, return error result
                return {
                    "message": f"I encountered an error while generating a response: {str(e)}",
                    "error": str(e),
                    "processing_mode": "error"
                }
    
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
            "reason": reason,
            "user_initiated": True
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
        
        # If using mode selector, update it
        if hasattr(self, "mode_selector") and self.mode_selector:
            # Update complexity thresholds if needed
            if mode == "auto":
                self.mode_selector.update_complexity_thresholds()
        
        logger.info(f"Processing mode set to {mode}{f' ({reason})' if reason else ''}")
        
        return {
            "success": True,
            "mode": mode,
            "previous_mode": previous_mode
        }
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about processing modes"""
        stats = {
            "current_mode": self.current_mode,
            "available_modes": list(self.processors.keys()) + ["auto"],
            "mode_switches": len(self.mode_switch_history),
            "recent_switches": self.mode_switch_history[-5:] if self.mode_switch_history else []
        }
        
        # Add mode selector stats if available
        if hasattr(self, "mode_selector") and self.mode_selector:
            try:
                mode_usage = await self.mode_selector.analyze_mode_usage()
                stats["mode_usage"] = mode_usage
                
                # Include thresholds
                stats["complexity_thresholds"] = self.mode_selector.complexity_thresholds
                
                # Include performance metrics
                stats["mode_metrics"] = {
                    mode: {
                        "success_rate": metrics["success_rate"],
                        "avg_time": metrics["avg_time"],
                        "usage_count": metrics["usage_count"]
                    } for mode, metrics in self.mode_selector.mode_metrics.items()
                }
            except Exception as e:
                logger.error(f"Error getting mode selector stats: {str(e)}")
        
        return stats
    
    async def get_recommended_mode(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get the recommended processing mode without changing the current mode"""
        # Use mode selector if available
        if hasattr(self, "mode_selector") and self.mode_selector:
            try:
                # Get recommendation
                recommended_mode = await self.mode_selector.determine_processing_mode(user_input, context)
                
                # Get reasoning behind recommendation
                selection_record = self.mode_selector.selection_history[-1] if self.mode_selector.selection_history else {}
                
                return {
                    "recommended_mode": recommended_mode,
                    "current_mode": self.current_mode,
                    "reason": selection_record.get("reason", "Unknown reason"),
                    "complexity_score": selection_record.get("complexity_score"),
                    "confidence": selection_record.get("confidence", 0.8)
                }
            except Exception as e:
                logger.error(f"Error getting mode recommendation: {str(e)}")
        
        # Fallback to legacy method
        recommended_mode = await self._determine_processing_mode(user_input, context)
        
        return {
            "recommended_mode": recommended_mode,
            "current_mode": self.current_mode,
            "fallback": True
        }
