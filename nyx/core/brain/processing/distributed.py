# nyx/core/brain/processing/distributed.py
import asyncio
import datetime
from typing import Dict, Any, List, Optional
from agents import Agent, Runner, trace, handoff

from nyx.core.brain.processing.base_processor import BaseProcessor, ProcessingContext, ProcessingResult, ResponseData

logger = logging.getLogger(__name__)

class TaskResult(BaseModel):
    """Result from a distributed task"""
    task_name: str
    success: bool
    data: Any
    error: Optional[str] = None

class DistributedProcessor(BaseProcessor):
    """Distributed processing using agent orchestration"""
    
    async def _create_agents(self):
        """Create distributed task agents"""
        await super()._create_agents()
        
        # Task coordinator agent
        self._agents["coordinator"] = Agent(
            name="Task Coordinator",
            model="gpt-5-nano",
            instructions="""Coordinate distributed processing tasks:
            1. Identify required subtasks
            2. Delegate to appropriate agents
            3. Collect and synthesize results
            4. Handle failures gracefully""",
            handoffs=[],  # Will be populated after creating other agents
            output_type=Dict[str, Any]
        )
        
        # Specialized task agents
        self._agents["memory_processor"] = Agent(
            name="Memory Processor",
            model="gpt-5-nano",
            instructions="Process and analyze memory-related tasks.",
            tools=[function_tool(self._retrieve_memories_tool)],
            output_type=TaskResult
        )
        
        self._agents["emotion_processor"] = Agent(
            name="Emotion Processor",
            model="gpt-5-nano",
            instructions="Process emotional analysis tasks.",
            tools=[function_tool(self._analyze_emotion_tool)],
            output_type=TaskResult
        )
        
        self._agents["adaptation_processor"] = Agent(
            name="Adaptation Processor",
            model="gpt-5-nano",
            instructions="Handle dynamic adaptation and context changes.",
            output_type=TaskResult
        )
        
        # Set up handoffs for coordinator
        self._agents["coordinator"].handoffs = [
            handoff(self._agents["memory_processor"],
                   tool_description="Process memory-related tasks"),
            handoff(self._agents["emotion_processor"],
                   tool_description="Process emotion-related tasks"),
            handoff(self._agents["adaptation_processor"],
                   tool_description="Process adaptation tasks")
        ]
    
    async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process input using distributed agents"""
        if not self._initialized:
            await self.initialize()
        
        with trace(workflow_name="process_input_distributed",
                  group_id=getattr(self.brain, 'trace_group_id', 'default')):
            
            start_time = datetime.datetime.now()
            processing_context = self._create_processing_context(user_input, context)
            
            try:
                # Let coordinator orchestrate the processing
                coordinator_input = {
                    "user_input": user_input,
                    "required_tasks": [
                        "emotional_analysis",
                        "memory_retrieval", 
                        "pattern_analysis",
                        "adaptation_check"
                    ],
                    "context": context or {}
                }
                
                result = await Runner.run(
                    self._agents["coordinator"],
                    str(coordinator_input),
                    context=processing_context
                )
                
                # Extract coordinated results
                coordinated_results = result.final_output if hasattr(result, 'final_output') else {}
                
                # Process task results
                emotional_state = coordinated_results.get("emotional_analysis", {})
                memories = coordinated_results.get("memories", [])
                adaptation_needed = coordinated_results.get("adaptation_needed", False)
                
                response_time = (datetime.datetime.now() - start_time).total_seconds()
                
                # Update performance metrics
                if hasattr(self.brain, 'performance_metrics'):
                    self.brain.performance_metrics["distributed_tasks_completed"] = \
                        self.brain.performance_metrics.get("distributed_tasks_completed", 0) + 1
                
                return ProcessingResult(
                    user_input=user_input,
                    emotional_state=emotional_state,
                    memories=memories,
                    memory_count=len(memories),
                    has_experience=bool(memories),
                    response_time=response_time,
                    processing_mode="distributed"
                ).dict()
                
            except Exception as e:
                logger.error(f"Error in distributed processing: {str(e)}")
                return ProcessingResult(
                    user_input=user_input,
                    emotional_state={},
                    memories=[],
                    memory_count=0,
                    has_experience=False,
                    response_time=0.0,
                    processing_mode="distributed",
                    error=str(e)
                ).dict()
    
    async def generate_response(self, user_input: str, processing_result: Dict[str, Any],
                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response from distributed processing"""
        with trace(workflow_name="generate_response_distributed",
                  group_id=getattr(self.brain, 'trace_group_id', 'default')):
            
            # Simple response generation for now
            message = "I've processed your input using distributed analysis."
            
            # Check if we have richer results
            if processing_result.get("has_experience"):
                message = processing_result.get("experience_response", message)
            
            return ResponseData(
                message=message,
                response_type="distributed",
                emotional_state=processing_result.get("emotional_state", {})
            ).dict()
