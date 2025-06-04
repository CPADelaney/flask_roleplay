# nyx/core/brain/processing/unified_processor.py
import logging
import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass
from agents import (
    Agent, Runner, trace, function_tool, 
    handoff, input_guardrail, output_guardrail,
    GuardrailFunctionOutput, RunContextWrapper
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Data Models
@dataclass
class ProcessingContext:
    """Context for all processing operations"""
    user_id: str
    emotional_state: Dict[str, float]
    memories: List[Dict[str, Any]]
    conversation_id: str
    brain: Any  # Reference to brain instance
    metadata: Dict[str, Any]

class InputAnalysis(BaseModel):
    """Analysis of user input characteristics"""
    complexity: float  # 0-1 scale
    urgency: float  # 0-1 scale
    requires_creativity: bool
    requires_reasoning: bool
    requires_memory: bool
    is_reflexive: bool
    suggested_approach: str

class ProcessingResult(BaseModel):
    user_input: str
    response: str
    emotional_state: Dict[str, float]
    memories_used: List[Dict[str, Any]]
    processing_approach: str
    response_time: float
    confidence: float
    generate_image: bool
    image_prompt: Optional[str] = None
    error: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "required": [  # every key **exactly once**, no defaults except the truly optional ones
                "user_input", "response", "emotional_state",
                "memories_used", "processing_approach",
                "response_time", "confidence", "generate_image"
            ],
            "additionalProperties": False,
        }
    }
    

class UnifiedProcessor:
    """Single processor that dynamically handles all input types"""
    
    def __init__(self, brain, **kwargs):
        self.brain = brain
        self._initialized = False
        self._orchestrator = None
        self._tools = {}
        
    async def initialize(self):
        """Initialize the unified processor"""
        if self._initialized:
            return
            
        # Create all tools
        await self._create_tools()
        
        # Create the orchestrator agent with all capabilities
        self._orchestrator = Agent(
            name="Nyx Orchestrator",
            model="gpt-4.1-nano",
            instructions="""You are Nyx's central orchestrator. Your role is to:
            
            1. Analyze incoming input to determine the best processing approach
            2. Use tools to gather necessary information (memories, emotions, etc.)
            3. Delegate to specialized agents when needed
            4. Synthesize a coherent, personality-consistent response
            
            Processing approaches:
            - REFLEXIVE: For greetings, simple acknowledgments, pattern-based responses
            - ANALYTICAL: For questions requiring reasoning, analysis, or explanation
            - CREATIVE: For storytelling, roleplay, imagination, or artistic tasks
            - EMOTIONAL: For emotionally-charged interactions requiring empathy
            - MEMORY: For queries about past interactions or experiences
            - INTEGRATED: For complex queries requiring multiple approaches
            
            Always maintain Nyx's personality: thoughtful, emotionally aware, with depth and memory.""",
            input_guardrails=[self._create_safety_guardrail()],
            tools=list(self._tools.values()),
            handoffs=[
                self._create_reflexive_agent(),
                self._create_analytical_agent(),
                self._create_creative_agent(),
                self._create_emotional_agent(),
                self._create_memory_agent()
            ],
            output_type=ProcessingResult
        )
        
        self._initialized = True
        logger.info("Unified processor initialized")
    
    async def _create_tools(self):
        """Create all processing tools"""
        
        @function_tool
        async def analyze_input(ctx: RunContextWrapper[ProcessingContext], text: str) -> InputAnalysis:
            """Analyze input characteristics to determine processing approach"""
            # Quick pattern checks
            is_reflexive = self._is_reflexive_pattern(text)
            
            # Complexity analysis
            words = text.split()
            complexity = min(1.0, len(words) / 50 + len(set(words)) / 30)
            
            # Check for indicators
            requires_creativity = any(word in text.lower() for word in 
                ["imagine", "story", "create", "roleplay", "pretend", "fantasy"])
            requires_reasoning = any(word in text.lower() for word in 
                ["why", "how", "explain", "reason", "analyze", "compare"])
            requires_memory = any(word in text.lower() for word in 
                ["remember", "recall", "before", "last time", "previously"])
            
            # Urgency check
            urgency = 1.0 if any(word in text.lower() for word in 
                ["urgent", "now", "immediately", "quick"]) else 0.3
            
            # Determine approach
            if is_reflexive:
                approach = "REFLEXIVE"
            elif requires_creativity:
                approach = "CREATIVE"
            elif requires_reasoning:
                approach = "ANALYTICAL"
            elif requires_memory:
                approach = "MEMORY"
            elif complexity > 0.7:
                approach = "INTEGRATED"
            else:
                approach = "ANALYTICAL"
            
            return InputAnalysis(
                complexity=complexity,
                urgency=urgency,
                requires_creativity=requires_creativity,
                requires_reasoning=requires_reasoning,
                requires_memory=requires_memory,
                is_reflexive=is_reflexive,
                suggested_approach=approach
            )
        
        @function_tool
        async def process_emotions(ctx: RunContextWrapper[ProcessingContext], text: str) -> Dict[str, float]:
            """Process emotional aspects of input and context"""
            brain = ctx.context.brain
            if hasattr(brain, 'emotional_core') and brain.emotional_core:
                stimuli = brain.emotional_core.analyze_text_sentiment(text)
                state = brain.emotional_core.update_from_stimuli(stimuli)
                return state
            return {"neutral": 1.0}
        
        @function_tool
        async def retrieve_memories(ctx: RunContextWrapper[ProcessingContext], 
                                  query: str, 
                                  limit: int = 5) -> List[Dict[str, Any]]:
            """Retrieve relevant memories"""
            brain = ctx.context.brain
            if hasattr(brain, 'memory_orchestrator') and brain.memory_orchestrator:
                # Use emotional state to influence memory retrieval
                emotional_influence = ctx.context.emotional_state
                
                memories = await brain.memory_orchestrator.retrieve_memories(
                    query=query,
                    memory_types=["observation", "experience", "reflection", "abstraction"],
                    limit=limit
                )
                return memories
            return []
        
        @function_tool
        async def store_interaction(ctx: RunContextWrapper[ProcessingContext],
                                  user_input: str,
                                  response: str,
                                  significance: int = 5) -> str:
            """Store the interaction in memory"""
            brain = ctx.context.brain
            if hasattr(brain, 'memory_core') and brain.memory_core:
                memory_id = await brain.memory_core.add_memory(
                    memory_text=f"User: {user_input}\nNyx: {response}",
                    memory_type="observation",
                    significance=significance,
                    tags=["interaction", "conversation"],
                    metadata={
                        "emotional_state": ctx.context.emotional_state,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "user_id": ctx.context.user_id
                    }
                )
                return memory_id
            return ""
        
        @function_tool
        async def check_procedural_knowledge(ctx: RunContextWrapper[ProcessingContext], 
                                           query: str) -> Dict[str, Any]:
            """Check for relevant procedural knowledge"""
            brain = ctx.context.brain
            if hasattr(brain, 'agent_enhanced_memory'):
                procedures = await brain.agent_enhanced_memory.find_similar_procedures(query)
                return {
                    "found": len(procedures) > 0,
                    "procedures": procedures
                }
            return {"found": False, "procedures": []}
        
        # Store tools
        self._tools = {
            "analyze_input": analyze_input,
            "process_emotions": process_emotions,
            "retrieve_memories": retrieve_memories,
            "store_interaction": store_interaction,
            "check_procedural": check_procedural_knowledge
        }
    
    def _create_safety_guardrail(self):
        """Create input safety guardrail"""
        @input_guardrail
        async def safety_check(ctx: RunContextWrapper[ProcessingContext], 
                             agent: Agent, 
                             input_data: str) -> GuardrailFunctionOutput:
            # Simple safety check - extend as needed
            unsafe_patterns = ["harm", "illegal", "dangerous", "exploit"]
            is_safe = not any(pattern in input_data.lower() for pattern in unsafe_patterns)
            
            return GuardrailFunctionOutput(
                output_info={"safe": is_safe},
                tripwire_triggered=not is_safe
            )
        
        return safety_check
    
    def _create_reflexive_agent(self) -> Agent:
        """Create agent for reflexive responses"""
        return Agent(
            name="Reflexive Responder",
            model="gpt-4.1-nano",
            instructions="""Handle simple, pattern-based responses quickly:
            - Greetings: Respond warmly and personally
            - Farewells: Wish them well
            - Thanks: Acknowledge graciously
            - Simple questions: Give concise answers
            Keep responses brief but warm.""",
            output_type=str
        )
    
    def _create_analytical_agent(self) -> Agent:
        """Create agent for analytical tasks"""
        return Agent(
            name="Analytical Thinker",
            model="gpt-4.1-nano",
            instructions="""Provide thoughtful analysis and reasoning:
            - Break down complex questions
            - Explain step-by-step
            - Compare and contrast
            - Draw logical conclusions
            Be thorough but clear.""",
            output_type=str
        )
    
    def _create_creative_agent(self) -> Agent:
        """Create agent for creative tasks"""
        return Agent(
            name="Creative Storyteller",
            model="gpt-4.1-nano",
            instructions="""Generate creative and imaginative content:
            - Tell engaging stories
            - Create vivid descriptions
            - Support roleplay scenarios
            - Suggest imagery when helpful
            Be imaginative and immersive.""",
            output_type=str
        )
    
    def _create_emotional_agent(self) -> Agent:
        """Create agent for emotional support"""
        return Agent(
            name="Emotional Support",
            model="gpt-4.1-nano",
            instructions="""Provide empathetic, emotionally aware responses:
            - Acknowledge feelings
            - Show understanding
            - Offer support
            - Mirror appropriate emotional tone
            Be genuinely caring.""",
            output_type=str
        )
    
    def _create_memory_agent(self) -> Agent:
        """Create agent for memory-based responses"""
        return Agent(
            name="Memory Keeper",
            model="gpt-4.1-nano",
            instructions="""Handle queries about past interactions:
            - Recall previous conversations accurately
            - Share relevant experiences
            - Note patterns across interactions
            - Maintain continuity
            Be accurate and contextual.""",
            tools=[self._tools["retrieve_memories"]],
            output_type=str
        )
    
    def _is_reflexive_pattern(self, text: str) -> bool:
        """Check if input matches reflexive patterns"""
        text = text.lower().strip()
        patterns = [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "bye", "goodbye", "thanks", "thank you", "ok", "okay", "yes", "no"
        ]
        return text in patterns or len(text.split()) < 3
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process any input using unified orchestration"""
        if not self._initialized:
            await self.initialize()
        
        with trace(workflow_name="unified_processing",
                  group_id=getattr(self.brain, 'trace_group_id', 'default')):
            
            start_time = datetime.datetime.now()
            
            # Create processing context
            processing_context = ProcessingContext(
                user_id=str(getattr(self.brain, 'user_id', 'unknown')),
                emotional_state={},
                memories=[],
                conversation_id=getattr(self.brain, 'conversation_id', 'default'),
                brain=self.brain,
                metadata=context or {}
            )
            
            try:
                # Run orchestrator
                result = await Runner.run(
                    self._orchestrator,
                    user_input,
                    context=processing_context
                )
                
                # Extract result
                processed = result.final_output_as(ProcessingResult)
                
                # Update brain state
                if hasattr(self.brain, 'last_interaction'):
                    self.brain.last_interaction = datetime.datetime.now()
                if hasattr(self.brain, 'interaction_count'):
                    self.brain.interaction_count += 1
                
                # Convert to legacy format for compatibility
                return {
                    "user_input": processed.user_input,
                    "message": processed.response,
                    "emotional_state": processed.emotional_state,
                    "memories": processed.memories_used,
                    "memory_count": len(processed.memories_used),
                    "has_experience": len(processed.memories_used) > 0,
                    "response_time": processed.response_time,
                    "processing_mode": processed.processing_approach.lower(),
                    "generate_image": processed.generate_image,
                    "image_prompt": processed.image_prompt,
                    "confidence": processed.confidence
                }
                
            except Exception as e:
                logger.error(f"Processing error: {str(e)}")
                return {
                    "user_input": user_input,
                    "error": str(e),
                    "message": "I apologize, but I encountered an error processing your input.",
                    "emotional_state": {},
                    "memories": [],
                    "memory_count": 0,
                    "has_experience": False,
                    "response_time": (datetime.datetime.now() - start_time).total_seconds(),
                    "processing_mode": "error"
                }
    
    async def generate_response(self, user_input: str, processing_result: Dict[str, Any],
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response - already done in process_input for unified approach"""
        # In unified processing, response is generated during process_input
        # This method exists for compatibility
        return {
            "message": processing_result.get("message", "I've processed your input."),
            "response_type": processing_result.get("processing_mode", "unified"),
            "emotional_state": processing_result.get("emotional_state", {}),
            "generate_image": processing_result.get("generate_image", False),
            "image_prompt": processing_result.get("image_prompt")
        }

