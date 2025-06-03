
# nyx/core/brain/processing/parallel.py
import asyncio
import datetime
from typing import Dict, Any, List
from agents import Agent, Runner, trace

from nyx.core.brain.processing.base_processor import BaseProcessor, ProcessingContext, ProcessingResult, ResponseData

logger = logging.getLogger(__name__)

class ParallelProcessor(BaseProcessor):
    """Parallel processing using multiple specialized agents"""
    
    async def _create_agents(self):
        """Create specialized agents for parallel processing"""
        await super()._create_agents()
        
        # Quick response agent
        self._agents["quick_responder"] = Agent(
            name="Quick Responder",
            model="gpt-4.1-nano",
            instructions="Generate quick, contextual responses to user input.",
            output_type=str
        )
        
        # Deep analyzer agent  
        self._agents["deep_analyzer"] = Agent(
            name="Deep Analyzer",
            model="gpt-4.1-nano",
            instructions="Perform deep analysis of user intent and context.",
            tools=[
                function_tool(self._retrieve_memories_tool),
                function_tool(self._analyze_patterns_tool)
            ],
            output_type=Dict[str, Any]
        )
        
        # Response synthesizer
        self._agents["synthesizer"] = Agent(
            name="Response Synthesizer",
            model="gpt-4.1-nano",
            instructions="Synthesize multiple analyses into a coherent response.",
            output_type=str
        )
    
    @function_tool
    async def _analyze_patterns_tool(ctx: RunContextWrapper[ProcessingContext], 
                                   text: str) -> Dict[str, Any]:
        """Analyze patterns in user input"""
        patterns = {
            "is_question": text.endswith("?"),
            "is_command": any(text.lower().startswith(cmd) for cmd in ["do", "make", "create", "show"]),
            "is_emotional": any(word in text.lower() for word in ["feel", "happy", "sad", "angry"]),
            "complexity": len(text.split()) / 10.0  # Simple complexity measure
        }
        return patterns
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process input using parallel agents"""
        if not self._initialized:
            await self.initialize()
        
        with trace(workflow_name="process_input_parallel",
                  group_id=getattr(self.brain, 'trace_group_id', 'default')):
            
            start_time = datetime.datetime.now()
            processing_context = self._create_processing_context(user_input, context)
            
            try:
                # Run multiple agents in parallel
                tasks = [
                    Runner.run(self._agents["emotional_analyzer"], user_input, context=processing_context),
                    Runner.run(self._agents["quick_responder"], user_input, context=processing_context),
                    Runner.run(self._agents["deep_analyzer"], user_input, context=processing_context),
                    Runner.run(self._agents["memory_retriever"], user_input, context=processing_context)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                emotional_state = {}
                quick_response = ""
                deep_analysis = {}
                memories = []
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Task {i} failed: {str(result)}")
                    else:
                        if i == 0:  # Emotional analyzer
                            emotional_state = result.final_output if hasattr(result, 'final_output') else {}
                        elif i == 1:  # Quick responder
                            quick_response = result.final_output if hasattr(result, 'final_output') else ""
                        elif i == 2:  # Deep analyzer
                            deep_analysis = result.final_output if hasattr(result, 'final_output') else {}
                        elif i == 3:  # Memory retriever
                            memories = result.final_output if hasattr(result, 'final_output') else []
                
                # Synthesize final response
                synthesis_input = {
                    "quick_response": quick_response,
                    "deep_analysis": deep_analysis,
                    "emotional_context": emotional_state,
                    "memories": memories
                }
                
                synthesis_result = await Runner.run(
                    self._agents["synthesizer"],
                    str(synthesis_input),
                    context=processing_context
                )
                
                final_response = synthesis_result.final_output if hasattr(synthesis_result, 'final_output') else quick_response
                
                # Store for response generation
                processing_context.metadata["synthesized_response"] = final_response
                
                response_time = (datetime.datetime.now() - start_time).total_seconds()
                
                return ProcessingResult(
                    user_input=user_input,
                    emotional_state=emotional_state,
                    memories=memories,
                    memory_count=len(memories),
                    has_experience=bool(memories),
                    response_time=response_time,
                    processing_mode="parallel"
                ).dict()
                
            except Exception as e:
                logger.error(f"Error in parallel processing: {str(e)}")
                return ProcessingResult(
                    user_input=user_input,
                    emotional_state={},
                    memories=[],
                    memory_count=0,
                    has_experience=False,
                    response_time=0.0,
                    processing_mode="parallel",
                    error=str(e)
                ).dict()
    
    async def generate_response(self, user_input: str, processing_result: Dict[str, Any],
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response from parallel processing"""
        with trace(workflow_name="generate_response_parallel",
                  group_id=getattr(self.brain, 'trace_group_id', 'default')):
            
            message = context.get("synthesized_response", "I've processed your input.") if context else "I've processed your input."
            
            return ResponseData(
                message=message,
                response_type="parallel",
                emotional_state=processing_result.get("emotional_state", {})
            ).dict()
