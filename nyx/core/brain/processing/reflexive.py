# nyx/core/brain/processing/reflexive.py
import time
from typing import Dict, Any
from agents import Agent, Runner, trace

from nyx.core.brain.processing.base_processor import BaseProcessor, ProcessingContext, ProcessingResult, ResponseData

logger = logging.getLogger(__name__)

class ReflexiveResponse(BaseModel):
    """Quick reflexive response"""
    response: str
    pattern_matched: str
    confidence: float
    reaction_time_ms: float

class ReflexiveProcessor(BaseProcessor):
    """Fast reflexive processing for immediate responses"""
    
    async def _create_agents(self):
        """Create reflexive response agent"""
        await super()._create_agents()
        
        # Pattern matcher agent
        self._agents["pattern_matcher"] = Agent(
            name="Pattern Matcher",
            model="gpt-5-nano",
            instructions="""Match input to known patterns and generate immediate responses:
            - Greetings: "hello", "hi", "hey" -> friendly greeting
            - Farewells: "bye", "goodbye", "see you" -> warm farewell  
            - Thanks: "thank you", "thanks" -> acknowledgment
            - Simple questions: provide quick factual answers
            Respond immediately without deep analysis.""",
            output_type=ReflexiveResponse
        )
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process input reflexively"""
        if not self._initialized:
            await self.initialize()
        
        with trace(workflow_name="process_input_reflexive",
                  group_id=getattr(self.brain, 'trace_group_id', 'default')):
            
            start_time = time.time()
            processing_context = self._create_processing_context(user_input, context)
            
            try:
                # Check if reflexive system is available in brain
                if hasattr(self.brain, 'reflexive_system') and self.brain.reflexive_system:
                    # Use brain's reflexive system
                    stimulus = {"text": user_input}
                    reflex_result = await self.brain.reflexive_system.process_stimulus_fast(stimulus)
                    
                    if reflex_result.get("success"):
                        reaction_time = (time.time() - start_time) * 1000
                        
                        return ProcessingResult(
                            user_input=user_input,
                            emotional_state={},
                            memories=[],
                            memory_count=0,
                            has_experience=False,
                            response_time=reaction_time / 1000,
                            processing_mode="reflexive"
                        ).dict()
                
                # Fallback to agent-based reflexive processing
                result = await Runner.run(
                    self._agents["pattern_matcher"],
                    user_input,
                    context=processing_context
                )
                
                reflex_response = result.final_output_as(ReflexiveResponse)
                reaction_time = (time.time() - start_time) * 1000
                
                # Store response for generation phase
                processing_context.metadata["reflex_response"] = reflex_response.response
                processing_context.metadata["pattern_matched"] = reflex_response.pattern_matched
                
                return ProcessingResult(
                    user_input=user_input,
                    emotional_state={},
                    memories=[],
                    memory_count=0,
                    has_experience=False,
                    response_time=reaction_time / 1000,
                    processing_mode="reflexive"
                ).dict()
                
            except Exception as e:
                logger.error(f"Error in reflexive processing: {str(e)}")
                # Fall back to serial processing
                return ProcessingResult(
                    user_input=user_input,
                    emotional_state={},
                    memories=[],
                    memory_count=0,
                    has_experience=False,
                    response_time=0.0,
                    processing_mode="reflexive",
                    error=str(e)
                ).dict()
    
    async def generate_response(self, user_input: str, processing_result: Dict[str, Any],
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate reflexive response"""
        with trace(workflow_name="generate_response_reflexive",
                  group_id=getattr(self.brain, 'trace_group_id', 'default')):
            
            # Get reflexive response
            message = context.get("reflex_response", "I understand.") if context else "I understand."
            pattern = context.get("pattern_matched", "unknown") if context else "unknown"
            
            return ResponseData(
                message=message,
                response_type=f"reflexive_{pattern}",
                emotional_state={}
            ).dict()
