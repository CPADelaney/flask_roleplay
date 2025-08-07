# nyx/core/brain/processing/agent.py
import logging
from typing import Dict, Any, Optional, List
from agents import Agent, Runner, handoff, input_guardrail, output_guardrail, GuardrailFunctionOutput
from pydantic import BaseModel

from nyx.core.brain.processing.base_processor import BaseProcessor, ProcessingContext, ProcessingResult, ResponseData

logger = logging.getLogger(__name__)

class AgentResponse(BaseModel):
    """Response from agent processing"""
    narrative: str
    generate_image: bool = False
    image_prompt: Optional[str] = None
    emotional_tone: Optional[str] = None

class SafetyCheck(BaseModel):
    """Safety check result"""
    is_safe: bool
    reason: str

# New explicit models for Dict[str, Any] replacements
class ExperienceData(BaseModel):
    """Experience sharing data"""
    has_experience: bool
    experience_type: Optional[str] = None
    confidence: Optional[float] = None
    content: Optional[str] = None

class ContextData(BaseModel):
    """Context data for experience sharing"""
    user_id: str
    emotional_state: Dict[str, float]
    conversation_id: str

class AgentProcessor(BaseProcessor):
    """Agent-based processor using OpenAI Agents SDK"""
    
    def __init__(self, brain, integration_mode=False):
        super().__init__(brain)
        self.integration_mode = integration_mode
        self.performance_metrics = {
            "response_times": [],
            "token_usage": 0,
            "success_rate": 0.95,
            "error_count": 0
        }
    
    async def _create_agents(self):
        """Create specialized agents"""
        await super()._create_agents()
        
        # Create guardrails
        @input_guardrail
        async def safety_guardrail(ctx: RunContextWrapper[ProcessingContext], 
                                  agent: Agent, 
                                  input_data: str) -> GuardrailFunctionOutput:
            """Check input safety"""
            # Run safety check
            safety_agent = Agent(
                name="Safety Checker",
                model="gpt-5-nano",
                instructions="Check if the input is safe and appropriate.",
                output_type=SafetyCheck
            )
            
            result = await Runner.run(safety_agent, input_data, context=ctx.context)
            safety_check = result.final_output_as(SafetyCheck)
            
            return GuardrailFunctionOutput(
                output_info=safety_check,
                tripwire_triggered=not safety_check.is_safe
            )
        
        # Main processing agent
        self._agents["main"] = Agent(
            name="Nyx Main Agent",
            model="gpt-5-nano",
            instructions="""You are Nyx, an AI assistant with emotional awareness and memory.
            Process user input thoughtfully, considering emotional context and past interactions.
            Maintain consistency with your personality and past experiences.""",
            input_guardrails=[safety_guardrail],
            output_type=AgentResponse,
            tools=[
                self._analyze_emotion_tool,
                self._retrieve_memories_tool,
                self._share_experience_tool
            ]
        )
        
        # Creative agent for roleplay/narrative
        self._agents["creative"] = Agent(
            name="Creative Agent",
            model="gpt-5-nano",
            instructions="""You are a creative storyteller and roleplay assistant.
            Generate vivid, engaging narratives and immersive experiences.
            When appropriate, suggest image generation to enhance the experience.""",
            output_type=AgentResponse
        )
        
        # Analytical agent for reasoning
        self._agents["analytical"] = Agent(
            name="Analytical Agent",
            model="gpt-5-nano",
            instructions="""You are an analytical thinker specializing in reasoning and logic.
            Break down complex problems, analyze relationships, and provide clear explanations.""",
            output_type=AgentResponse
        )
        
        # Set up handoffs
        self._agents["main"].handoffs = [
            handoff(self._agents["creative"], 
                   tool_description="Hand off to creative agent for storytelling and roleplay"),
            handoff(self._agents["analytical"],
                   tool_description="Hand off to analytical agent for complex reasoning")
        ]
    
    @function_tool
    async def _share_experience_tool(ctx: RunContextWrapper[ProcessingContext], 
                                   query: str) -> ExperienceData:
        """Share relevant experiences"""
        if hasattr(ctx.context, 'metadata') and 'brain' in ctx.context.metadata:
            brain = ctx.context.metadata['brain']
            if hasattr(brain, 'experience_interface') and brain.experience_interface:
                result = await brain.experience_interface.share_experience_enhanced(
                    query=query,
                    context_data={
                        "user_id": ctx.context.user_id,
                        "emotional_state": ctx.context.emotional_state,
                        "conversation_id": ctx.context.conversation_id
                    }
                )
                return ExperienceData(
                    has_experience=result.get("has_experience", False),
                    experience_type=result.get("experience_type"),
                    confidence=result.get("confidence"),
                    content=result.get("content")
                )
        return ExperienceData(has_experience=False)
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process input using agents"""
        if not self._initialized:
            await self.initialize()
        
        with trace(workflow_name="process_input_agent", 
                  group_id=getattr(self.brain, 'trace_group_id', 'default')):
            
            start_time = datetime.datetime.now()
            processing_context = self._create_processing_context(user_input, context)
            
            try:
                # Run main agent
                result = await Runner.run(
                    self._agents["main"],
                    user_input,
                    context=processing_context
                )
                
                # Extract structured response
                agent_response = result.final_output_as(AgentResponse)
                
                # Calculate metrics
                response_time = (datetime.datetime.now() - start_time).total_seconds()
                self.performance_metrics["response_times"].append(response_time)
                
                # Build processing result
                return ProcessingResult(
                    user_input=user_input,
                    emotional_state=processing_context.emotional_state,
                    memories=processing_context.memories,
                    memory_count=len(processing_context.memories),
                    has_experience=bool(processing_context.metadata.get("experience_found")),
                    experience_response=processing_context.metadata.get("experience_response"),
                    cross_user_experience=processing_context.metadata.get("cross_user", False),
                    memory_id=processing_context.metadata.get("memory_id"),
                    response_time=response_time,
                    processing_mode="agent"
                ).dict()
                
            except Exception as e:
                logger.error(f"Error in agent processing: {str(e)}")
                self.performance_metrics["error_count"] += 1
                
                return ProcessingResult(
                    user_input=user_input,
                    emotional_state={},
                    memories=[],
                    memory_count=0,
                    has_experience=False,
                    response_time=0.0,
                    processing_mode="agent",
                    error=str(e),
                    tripwire_triggered="Guardrail" in str(e)
                ).dict()
    
    async def generate_response(self, user_input: str, processing_result: Dict[str, Any], 
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response from processing result"""
        with trace(workflow_name="generate_response_agent",
                  group_id=getattr(self.brain, 'trace_group_id', 'default')):
            
            # If there was an error, return error response
            if processing_result.get("error"):
                return ResponseData(
                    message="I apologize, but I encountered an error processing your request.",
                    response_type="error",
                    emotional_state=processing_result.get("emotional_state", {})
                ).dict()
            
            # Extract response from stored context (would be set during processing)
            message = processing_result.get("message", "I've processed your input.")
            
            return ResponseData(
                message=message,
                response_type="agent",
                emotional_state=processing_result.get("emotional_state", {}),
                generate_image=processing_result.get("generate_image", False),
                image_prompt=processing_result.get("image_prompt")
            ).dict()
