# nyx/core/brain/processing/serial.py
import datetime
from typing import Dict, Any
from agents import Agent, Runner, trace

from nyx.core.brain.processing.base_processor import BaseProcessor, ProcessingContext, ProcessingResult, ResponseData

logger = logging.getLogger(__name__)

class SerialAgent(BaseModel):
    """Output for serial processing agent"""
    analysis: str
    emotional_impact: Dict[str, float]
    relevant_memories: List[str]
    suggested_response: str
    confidence: float

class SerialProcessor(BaseProcessor):
    """Serial processing using a comprehensive analysis agent"""
    
    async def _create_agents(self):
        """Create serial processing agent"""
        await super()._create_agents()
        
        # Comprehensive analysis agent
        self._agents["analyzer"] = Agent(
            name="Serial Analyzer",
            model="gpt-4.1-nano",
            instructions="""Perform comprehensive analysis of user input:
            1. Analyze emotional content and impact
            2. Identify relevant context and implications
            3. Consider appropriate responses
            4. Maintain consistency with past interactions""",
            tools=[
                function_tool(self._analyze_emotion_tool),
                function_tool(self._retrieve_memories_tool),
                function_tool(self._process_temporal_context_tool),
                function_tool(self._check_procedural_knowledge_tool)
            ],
            output_type=SerialAgent
        )
    
    @function_tool
    async def _process_temporal_context_tool(ctx: RunContextWrapper[ProcessingContext]) -> Dict[str, Any]:
        """Process temporal context"""
        if hasattr(ctx.context, 'metadata') and 'brain' in ctx.context.metadata:
            brain = ctx.context.metadata['brain']
            if hasattr(brain, 'temporal_perception') and brain.temporal_perception:
                effects = await brain.temporal_perception.on_interaction_start()
                return effects
        return {}
    
    @function_tool
    async def _check_procedural_knowledge_tool(ctx: RunContextWrapper[ProcessingContext], 
                                             query: str) -> Dict[str, Any]:
        """Check for relevant procedural knowledge"""
        if hasattr(ctx.context, 'metadata') and 'brain' in ctx.context.metadata:
            brain = ctx.context.metadata['brain']
            if hasattr(brain, 'agent_enhanced_memory'):
                procedures = await brain.agent_enhanced_memory.find_similar_procedures(query)
                return {
                    "relevant_procedures": procedures,
                    "can_execute": len(procedures) > 0
                }
        return {"relevant_procedures": [], "can_execute": False}
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process input serially through comprehensive analysis"""
        if not self._initialized:
            await self.initialize()
        
        with trace(workflow_name="process_input_serial",
                  group_id=getattr(self.brain, 'trace_group_id', 'default')):
            
            start_time = datetime.datetime.now()
            processing_context = self._create_processing_context(user_input, context)
            
            try:
                # Run comprehensive analysis
                result = await Runner.run(
                    self._agents["analyzer"],
                    user_input,
                    context=processing_context
                )
                
                analysis = result.final_output_as(SerialAgent)
                
                # Update brain state if needed
                if hasattr(self.brain, 'last_interaction'):
                    self.brain.last_interaction = datetime.datetime.now()
                if hasattr(self.brain, 'interaction_count'):
                    self.brain.interaction_count += 1
                
                # Store suggested response in context for generate_response
                processing_context.metadata["suggested_response"] = analysis.suggested_response
                processing_context.metadata["analysis_confidence"] = analysis.confidence
                
                response_time = (datetime.datetime.now() - start_time).total_seconds()
                
                return ProcessingResult(
                    user_input=user_input,
                    emotional_state=analysis.emotional_impact,
                    memories=[],  # Would be populated from memory retrieval
                    memory_count=len(analysis.relevant_memories),
                    has_experience=False,  # Would check experience system
                    response_time=response_time,
                    processing_mode="serial"
                ).dict()
                
            except Exception as e:
                logger.error(f"Error in serial processing: {str(e)}")
                return ProcessingResult(
                    user_input=user_input,
                    emotional_state={},
                    memories=[],
                    memory_count=0,
                    has_experience=False,
                    response_time=0.0,
                    processing_mode="serial",
                    error=str(e)
                ).dict()
    
    async def generate_response(self, user_input: str, processing_result: Dict[str, Any],
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response from serial analysis"""
        with trace(workflow_name="generate_response_serial",
                  group_id=getattr(self.brain, 'trace_group_id', 'default')):
            
            # Get suggested response from processing
            message = context.get("suggested_response", "I understand your input.") if context else "I understand your input."
            
            return ResponseData(
                message=message,
                response_type="serial",
                emotional_state=processing_result.get("emotional_state", {})
            ).dict()
