# nyx/core/brain/processing/base_processor.py
import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from agents import Agent, Runner, trace, function_tool, RunContextWrapper
from pydantic import BaseModel

logger = logging.getLogger(__name__)

@dataclass
class ProcessingContext:
    """Context for processing operations"""
    user_id: str
    emotional_state: Dict[str, float]
    memories: List[Dict[str, Any]]
    conversation_id: str
    metadata: Dict[str, Any]

class ProcessingResult(BaseModel):
    """Structured output for processing results"""
    user_input: str
    emotional_state: Dict[str, float]
    memories: List[Dict[str, Any]]
    memory_count: int
    has_experience: bool
    experience_response: Optional[str] = None
    cross_user_experience: bool = False
    memory_id: Optional[str] = None
    response_time: float
    processing_mode: str = "serial"
    error: Optional[str] = None
    tripwire_triggered: bool = False

class ResponseData(BaseModel):
    """Structured output for response generation"""
    message: str
    response_type: str
    emotional_state: Dict[str, float]
    emotional_expression: Optional[str] = None
    generate_image: bool = False
    image_prompt: Optional[str] = None

class BaseProcessor:
    """Base processor with agent integration"""
    
    def __init__(self, brain):
        self.brain = brain
        self._initialized = False
        self._agents = {}
        self._runner = None
        
    async def initialize(self):
        """Initialize processor and create agents"""
        await self._create_agents()
        self._initialized = True
        logger.info(f"{self.__class__.__name__} initialized")
    
    async def _create_agents(self):
        """Create agents for this processor - override in subclasses"""
        # Base emotional analysis agent
        self._agents["emotional_analyzer"] = Agent(
            name="Emotional Analyzer",
            model="gpt-5-nano",
            instructions="Analyze the emotional content and sentiment of user input.",
            tools=[function_tool(self._analyze_emotion_tool)],
            output_type=Dict[str, float]
        )
        
        # Memory retrieval agent
        self._agents["memory_retriever"] = Agent(
            name="Memory Retriever",
            model="gpt-5-nano",
            instructions="Retrieve relevant memories based on user input and context.",
            tools=[function_tool(self._retrieve_memories_tool)],
            output_type=List[Dict[str, Any]]
        )
    
    @function_tool
    async def _analyze_emotion_tool(ctx: RunContextWrapper[ProcessingContext], text: str) -> Dict[str, float]:
        """Analyze emotional content of text"""
        # This would integrate with the brain's emotional core
        if hasattr(ctx.context, 'metadata') and 'brain' in ctx.context.metadata:
            brain = ctx.context.metadata['brain']
            if hasattr(brain, 'emotional_core') and brain.emotional_core:
                stimuli = brain.emotional_core.analyze_text_sentiment(text)
                state = brain.emotional_core.update_from_stimuli(stimuli)
                return state
        
        # Fallback
        return {"neutral": 1.0}
    
    @function_tool
    async def _retrieve_memories_tool(ctx: RunContextWrapper[ProcessingContext], 
                                    query: str, 
                                    memory_types: List[str] = None,
                                    limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories from the brain"""
        if hasattr(ctx.context, 'metadata') and 'brain' in ctx.context.metadata:
            brain = ctx.context.metadata['brain']
            if hasattr(brain, 'memory_orchestrator') and brain.memory_orchestrator:
                memories = await brain.memory_orchestrator.retrieve_memories(
                    query=query,
                    memory_types=memory_types or ["observation", "experience", "reflection"],
                    limit=limit
                )
                return memories
        return []
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user input - implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement process_input")
    
    async def generate_response(self, user_input: str, processing_result: Dict[str, Any], 
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response - implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement generate_response")
    
    def _create_processing_context(self, user_input: str, context: Dict[str, Any] = None) -> ProcessingContext:
        """Create a processing context from input"""
        context = context or {}
        return ProcessingContext(
            user_id=context.get("user_id", str(getattr(self.brain, 'user_id', 'unknown'))),
            emotional_state=context.get("emotional_state", {}),
            memories=context.get("memories", []),
            conversation_id=getattr(self.brain, 'conversation_id', 'default'),
            metadata={
                "brain": self.brain,
                "user_input": user_input,
                **context
            }
        )
