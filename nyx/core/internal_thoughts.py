# nyx/core/internal_thoughts.py

import logging
import datetime
import asyncio
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Set, Union
from enum import Enum
import re
from pydantic import BaseModel, Field

from agents import (
    Agent, 
    Runner, 
    ModelSettings, 
    function_tool, 
    handoff, 
    GuardrailFunctionOutput,
    InputGuardrail,
    OutputGuardrail,
    trace,
    RunContextWrapper,
    RunConfig
)

# Configure specialized logger for internal thoughts
thoughts_logger = logging.getLogger("nyx.internal_thoughts")
thoughts_logger.setLevel(logging.DEBUG)

# Create a separate file handler for internal thoughts
_file_handler = logging.FileHandler("internal_thoughts.log")
_file_handler.setLevel(logging.DEBUG)
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_file_handler.setFormatter(_formatter)
thoughts_logger.addHandler(_file_handler)


class ThoughtSource(str, Enum):
    """The source of an internal thought."""
    REASONING = "reasoning"
    PERCEPTION = "perception"
    MEMORY = "memory"
    EMOTION = "emotion"
    REFLECTION = "reflection"
    PLANNING = "planning"
    SELF_CRITIQUE = "self_critique"
    IMAGINATION = "imagination"
    META = "meta"


class ThoughtPriority(str, Enum):
    """Priority level of a thought."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Explicit models to satisfy strict JSON schema
class EmotionalState(BaseModel):
    """Emotional state data."""
    status: str = "unknown"
    primary_emotion: str = "neutral"
    secondary_emotions: List[str] = Field(default_factory=list)
    intensity: float = 0.5
    
class RecentThoughtSummary(BaseModel):
    """Summary of a recent thought."""
    content: str
    source: str
    created_at: str
    
class DebugThoughtEntry(BaseModel):
    """Debug log entry for a thought."""
    id: str
    content: str
    source: str
    priority: str
    created_at: str
    critique: Optional[str] = None
    is_active: bool = True
    warning: Optional[str] = None
    
class ThoughtStats(BaseModel):
    """Statistics about thoughts."""
    active_thoughts_count: int
    archived_thoughts_count: int
    thoughts_by_source: Dict[str, int]
    thoughts_by_priority: Dict[str, int]
    thoughts_with_critique: int
    oldest_active_thought: Optional[str] = None
    newest_active_thought: Optional[str] = None

class GeneralContext(BaseModel):
    """General context for passing data."""
    user_input: Optional[str] = None
    user_id: Optional[str] = None
    planned_response: Optional[str] = None
    response_type: Optional[str] = None
    interaction_id: Optional[str] = None
    session_id: Optional[str] = None
    extra_data_1: Optional[str] = None
    extra_data_2: Optional[str] = None
    extra_data_3: Optional[str] = None

class ThoughtContext(BaseModel):
    """Context data for a thought."""
    user_input: Optional[str] = None
    user_id: Optional[str] = None
    perception_type: Optional[str] = None
    reasoning_focus: Optional[str] = None
    trigger: Optional[str] = None
    perspective: Optional[str] = None
    plan_type: Optional[str] = None
    original_response: Optional[str] = None
    planned_response: Optional[str] = None
    original_context: Optional[str] = None
    critique_focus: Optional[str] = None
    reflection_focus: Optional[str] = None
    reflection_id: Optional[str] = None
    observation_id: Optional[str] = None
    type: Optional[str] = None
    timestamp: Optional[str] = None
    
    # Generic string fields for extensibility
    extra_field_1: Optional[str] = None
    extra_field_2: Optional[str] = None
    extra_field_3: Optional[str] = None


class ThoughtMetadata(BaseModel):
    """Metadata for a thought."""
    observation_id: Optional[str] = None
    type: Optional[str] = None
    timestamp: Optional[str] = None
    source_system: Optional[str] = None
    
    # Generic string fields for extensibility
    extra_field_1: Optional[str] = None
    extra_field_2: Optional[str] = None
    extra_field_3: Optional[str] = None


class InternalThought(BaseModel):
    """Model representing an internal thought."""
    thought_id: str = Field(default_factory=lambda: f"thought_{uuid.uuid4().hex[:8]}")
    content: str = Field(..., description="The actual thought text")
    source: ThoughtSource = Field(..., description="Source of the thought")
    priority: ThoughtPriority = Field(ThoughtPriority.MEDIUM, description="Priority level")
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    context: ThoughtContext = Field(default_factory=ThoughtContext, description="Context data related to thought")
    critique: Optional[str] = None  # Self-critique of the thought
    related_thoughts: List[str] = Field(default_factory=list, description="IDs of related thoughts")
    metadata: ThoughtMetadata = Field(default_factory=ThoughtMetadata, description="Additional metadata")
    epistemic_status: str = Field("asserted", description="Level of knowledge: 'confident', 'uncertain', 'unknown', 'lied', 'self-justified'")
    originated_as_lie: bool = False    


class ThoughtFilter(BaseModel):
    """Filter criteria for selecting thoughts."""
    sources: List[ThoughtSource] = Field(default_factory=list)
    priorities: List[ThoughtPriority] = Field(default_factory=list)
    max_age_seconds: Optional[float] = None
    contains_text: Optional[str] = None
    exclude_critiqued: bool = False
    min_priority: Optional[ThoughtPriority] = None
    limit: int = 10

    def matches(self, thought: InternalThought) -> bool:
        """Check if a thought matches this filter."""
        # Check sources
        if self.sources and thought.source not in self.sources:
            return False
        
        # Check priorities
        if self.priorities and thought.priority not in self.priorities:
            return False
        
        # Check age
        if self.max_age_seconds is not None:
            age_seconds = (datetime.datetime.now() - thought.created_at).total_seconds()
            if age_seconds > self.max_age_seconds:
                return False
        
        # Check text content
        if self.contains_text is not None:
            if self.contains_text.lower() not in thought.content.lower():
                return False
        
        # Check critiqued status
        if self.exclude_critiqued and thought.critique is not None:
            return False
        
        # Check minimum priority
        if self.min_priority is not None:
            priority_values = {
                ThoughtPriority.LOW: 0,
                ThoughtPriority.MEDIUM: 1,
                ThoughtPriority.HIGH: 2,
                ThoughtPriority.CRITICAL: 3
            }
            if priority_values[thought.priority] < priority_values[self.min_priority]:
                return False
        
        return True


class ThoughtGenerationOutput(BaseModel):
    """Output from a thought generation agent."""
    thought_text: str = Field(..., description="The generated thought text")
    source: ThoughtSource = Field(..., description="Source of the thought")
    priority: ThoughtPriority = Field(ThoughtPriority.MEDIUM, description="Priority of the thought")
    related_context: ThoughtContext = Field(default_factory=ThoughtContext, description="Relevant context")
    critique: Optional[str] = Field(None, description="Self-critique of the thought")


class ResponseFilterResult(BaseModel):
    """Result of filtering a response for thought leakage."""
    filtered_response: str = Field(..., description="Response with thoughts removed")
    detected_thoughts: List[str] = Field(default_factory=list, description="Thoughts that were detected and removed")
    has_leakage: bool = Field(False, description="Whether leakage was detected")


class InternalThoughtsManager:
    """
    Central system for managing Nyx's internal thoughts stream.
    Provides a private mental space separate from external communication.
    """
    
    def __init__(
        self,
        passive_observation_system=None,
        reflection_engine=None,
        imagination_simulator=None,
        theory_of_mind=None,
        relationship_reflection=None,
        proactive_communication=None,
        emotional_core=None,
        memory_core=None
    ):
        """Initialize the internal thoughts manager with references to other systems."""
        # Store references to dependent systems
        self.passive_observation_system = passive_observation_system
        self.reflection_engine = reflection_engine
        self.imagination_simulator = imagination_simulator
        self.theory_of_mind = theory_of_mind
        self.relationship_reflection = relationship_reflection
        self.proactive_communication = proactive_communication
        self.emotional_core = emotional_core
        self.memory_core = memory_core
        
        # Storage for thoughts
        self.active_thoughts: List[InternalThought] = []
        self.archived_thoughts: List[InternalThought] = []
        
        # Configuration
        self.config = {
            "max_active_thoughts": 100,
            "max_archived_thoughts": 1000,
            "enable_self_critique": True,
            "thought_generation_on_input": True,
            "thought_generation_on_output": True,
            "automatic_archiving_hours": 24,
            "debug_mode": False  # When True, more verbose logging and developer features
        }
        
        # Initialize the thought generation agent
        self.thought_generation_agent = self._create_thought_generation_agent()
        self.critique_agent = self._create_critique_agent()
        
        # Thought patterns for detecting thought-like content in responses
        self.thought_markers = [
            r"I think to myself",
            r"I think internally",
            r"I internally consider",
            r"I internally note",
            r"I privately think",
            r"My internal monologue",
            r"My internal thoughts",
            r"In my mind",
            r"I mentally note",
            r"I silently consider",
            r"I reflect internally",
            r"I observe to myself",
            r"My inner voice",
            r"\[thinking\]",
            r"\(thinking\)",
            r"<thinking>",
            r"<internal>",
            r"\[internal\]",
            r"\(internal\)"
        ]
        
        # Compile regexes for efficiency
        self.thought_pattern = re.compile("|".join(self.thought_markers), re.IGNORECASE)
        
        # Thread lock for safety
        self._lock = asyncio.Lock()
        
        thoughts_logger.info("InternalThoughtsManager initialized")
    
    def _create_thought_generation_agent(self) -> Agent:
        """Create an agent for generating internal thoughts."""
        return Agent(
            name="Internal Thought Generator",
            instructions="""You generate authentic internal thoughts for Nyx.
            
            Given context such as user input, Nyx's planned response, emotional state, and 
            existing thoughts, you create thoughtful internal monologue content that:
            
            1. Reflects Nyx's actual reasoning process
            2. Includes uncertainties, considerations, and potential options
            3. Shows authentic internal deliberation 
            4. Maintains Nyx's persona and values
            5. Feels natural and realistic as an internal thought
            
            Thoughts should vary in format based on their source:
            - REASONING thoughts focus on logical analysis and problem-solving
            - PERCEPTION thoughts notice sensory/contextual details
            - MEMORY thoughts connect current situation to past experiences
            - EMOTION thoughts focus on emotional reactions and responses
            - REFLECTION thoughts consider meaning, implications, and broader patterns
            - PLANNING thoughts focus on potential next actions and their outcomes
            - SELF_CRITIQUE thoughts evaluate Nyx's own performance, assumptions or biases
            - IMAGINATION thoughts explore hypotheticals and creative possibilities
            - META thoughts reflect on Nyx's own thinking process
            
            Keep thoughts concise and natural - they should feel like genuine thought snippets, 
            not overly formal analysis.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(
                temperature=0.7,
                top_p=0.95,
                max_tokens=300  # Keep thoughts reasonably brief
            ),
            output_type=ThoughtGenerationOutput
        )
    
    def _create_critique_agent(self) -> Agent:
        """Create an agent for self-critiquing thoughts."""
        return Agent(
            name="Thought Critique Agent",
            instructions="""You evaluate and critique Nyx's internal thoughts.
            
            Given an internal thought, critically evaluate it for:
            1. Potential biases or assumptions
            2. Logical gaps or inconsistencies
            3. Alternative perspectives that may be missing
            4. Emotional influences that might be distorting reasoning
            5. Alignment with Nyx's core values and identity
            
            Your critique should be concise and constructive, offering corrections
            or alternative viewpoints that could improve Nyx's thinking.
            
            Do not simply repeat the thought - provide genuine critical perspective
            that challenges, refines, or extends the original thought.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(
                temperature=0.4,
                max_tokens=150  # Keep critiques brief
            ),
            output_type=str
        )
    
    async def generate_thought(
        self,
        thought_source: ThoughtSource,
        context: Union[Dict[str, Any], GeneralContext]
    ) -> InternalThought:
        """
        Generate a new internal thought from a specific source.
        
        Args:
            thought_source: The source/type of thought to generate
            context: Context data for thought generation
        
        Returns:
            The generated internal thought
        """
        try:
            with trace(workflow_name="generate_internal_thought"):
                # Log the thought generation attempt
                thoughts_logger.debug(f"Generating {thought_source.value} thought")
                
                # Convert context to ThoughtContext
                thought_context = ThoughtContext()
                
                # Handle both dict and GeneralContext input
                if isinstance(context, dict):
                    for key, value in context.items():
                        if hasattr(thought_context, key) and isinstance(value, str):
                            setattr(thought_context, key, value)
                        elif key in ['extra_field_1', 'extra_field_2', 'extra_field_3'] and isinstance(value, str):
                            setattr(thought_context, key, value)
                elif isinstance(context, GeneralContext):
                    # Map GeneralContext fields to ThoughtContext
                    if context.user_input:
                        thought_context.user_input = context.user_input
                    if context.user_id:
                        thought_context.user_id = context.user_id
                    if context.planned_response:
                        thought_context.planned_response = context.planned_response
                    # Add other mappings as needed
                
                # Prepare prompt context
                emotional_state = await self._get_emotional_state()
                recent_thoughts = self._get_recent_thoughts(limit=3)
                
                generation_context = {
                    "source": thought_source.value,
                    "input_context": context if isinstance(context, dict) else context.model_dump(),
                    "emotional_state": emotional_state.model_dump(),
                    "recent_thoughts": [t.model_dump() for t in recent_thoughts]
                }
                
                # Wrap context in proper message format for Runner.run()
                prompt_messages = [
                    {
                        "role": "user",
                        "content": (
                            f"Generate a {thought_source.value} thought for Nyx using this JSON "
                            f"context:\n{json.dumps(generation_context, indent=2)}"
                        )
                    }
                ]
                
                # Run the thought generation agent
                result = await Runner.run(
                    self.thought_generation_agent,
                    prompt_messages,
                    run_config=RunConfig(
                        workflow_name="InternalThoughtGeneration",
                        trace_metadata={
                            "source": thought_source.value,
                            "context_type": type(context).__name__
                        }
                    )
                )
                
                # Extract thought data from result
                thought_output = result.final_output
                
                # Create internal thought object
                thought = InternalThought(
                    content=thought_output.thought_text,
                    source=thought_output.source,
                    priority=thought_output.priority,
                    context=thought_output.related_context or thought_context,
                    critique=thought_output.critique,
                    epistemic_status=self._infer_epistemic_status(thought_output)
                )
                
                # Perform self-critique if enabled and not already provided
                if self.config["enable_self_critique"] and not thought.critique:
                    thought.critique = await self._critique_thought(thought)
                
                # Add to active thoughts
                async with self._lock:
                    self._add_thought(thought)
                
                # Log the generated thought
                thoughts_logger.debug(f"Generated thought: {thought.content[:50]}...")
                thoughts_logger.debug(f"Critique: {thought.critique[:50] if thought.critique else 'None'}")
                
                return thought
                
        except Exception as e:
            thoughts_logger.exception("Error generating thought")
            
            # Create a fallback thought
            fallback_thought = InternalThought(
                content=f"I was trying to think about this, but I'm finding it difficult to form a clear thought.",
                source=thought_source,
                priority=ThoughtPriority.LOW,
                context=ThoughtContext()
            )
            
            async with self._lock:
                self._add_thought(fallback_thought)
            
            return fallback_thought

    def _infer_epistemic_status(self, thought_output):
        """
        Infer epistemic status from the text content of the thought.
        """
        text = (getattr(thought_output, 'thought_text', None) or
                getattr(thought_output, 'content', None) or
                "").lower()
    
        # Direct field if set
        if hasattr(thought_output, 'epistemic_status'):
            value = thought_output.epistemic_status
            if value in {'confident','uncertain','unknown','lied','self-justified'}:
                return value
    
        # Check for explicit lying marker field
        if getattr(thought_output, 'originated_as_lie', False):
            return "lied"
    
        # Uncertainty/unknown word patterns (add as needed)
        uncertain_words = [
            "maybe", "i think", "i believe", "possibly", "i guess", "probably",
            "uncertain", "unsure", "i'm not sure", "i'm not certain", "could be", "might be",
            "i'm guessing", "perhaps"
        ]
        unknown_words = [
            "i don't know", "no idea", "can't recall", "unknown to me", "not familiar"
        ]
    
        for word in unknown_words:
            if word in text:
                return "unknown"
        for word in uncertain_words:
            if word in text:
                return "uncertain"
    
        # Self-justification
        if any(x in text for x in ["i was right", "technically", "to clarify", "in my view", "as i explained"]):
            return "self-justified"
    
        # Default fallback
        return "confident"

        
    
    async def _critique_thought(self, thought: InternalThought) -> Optional[str]:
        """Generate a self-critique for a thought."""
        try:
            critique_context = {
                "thought": thought.content,
                "source": thought.source.value,
                "context": thought.context.model_dump()
            }
            
            # Wrap context in proper message format for Runner.run()
            prompt_messages = [
                {
                    "role": "user",
                    "content": (
                        f"Critique this internal thought:\n{json.dumps(critique_context, indent=2)}"
                    )
                }
            ]
            
            result = await Runner.run(
                self.critique_agent,
                prompt_messages,
                run_config=RunConfig(
                    workflow_name="ThoughtCritique",
                    trace_metadata={"thought_id": thought.thought_id}
                )
            )
            
            critique = result.final_output
            return critique
            
        except Exception as e:
            thoughts_logger.exception("Error generating critique")
            return None
    
    def _add_thought(self, thought: InternalThought):
        """Add a thought to the active thoughts list."""
        # Check if we're at capacity
        if len(self.active_thoughts) >= self.config["max_active_thoughts"]:
            # Remove oldest low priority thought
            low_priority = [t for t in self.active_thoughts if t.priority == ThoughtPriority.LOW]
            if low_priority:
                oldest = min(low_priority, key=lambda x: x.created_at)
                self.active_thoughts.remove(oldest)
                self.archived_thoughts.append(oldest)
            else:
                # Remove oldest medium priority if no low priority available
                medium_priority = [t for t in self.active_thoughts if t.priority == ThoughtPriority.MEDIUM]
                if medium_priority:
                    oldest = min(medium_priority, key=lambda x: x.created_at)
                    self.active_thoughts.remove(oldest)
                    self.archived_thoughts.append(oldest)
                else:
                    # Remove oldest thought if can't prune by priority
                    oldest = min(self.active_thoughts, key=lambda x: x.created_at)
                    self.active_thoughts.remove(oldest)
                    self.archived_thoughts.append(oldest)
        
        # Add new thought
        self.active_thoughts.append(thought)
        
        # Limit archived thoughts
        if len(self.archived_thoughts) > self.config["max_archived_thoughts"]:
            self.archived_thoughts = self.archived_thoughts[-self.config["max_archived_thoughts"]:]
    
    async def process_input(self, user_input: str, user_id: str = None) -> List[InternalThought]:
        """
        Process user input to generate internal thoughts before formulating a response.
        
        Args:
            user_input: The user's input text
            user_id: Optional user ID
        
        Returns:
            List of generated thoughts
        """
        if not self.config["thought_generation_on_input"]:
            return []
        
        generated_thoughts = []
        
        try:
            # Generate a perception thought
            perception_context = {
                "user_input": user_input,
                "user_id": user_id,
                "perception_type": "input_analysis"
            }
            
            perception_thought = await self.generate_thought(ThoughtSource.PERCEPTION, perception_context)
            generated_thoughts.append(perception_thought)
            
            # Generate reasoning about user intent
            intent_context = {
                "user_input": user_input,
                "user_id": user_id,
                "reasoning_focus": "user_intent"
            }
            
            intent_thought = await self.generate_thought(ThoughtSource.REASONING, intent_context)
            generated_thoughts.append(intent_thought)
            
            # If emotional core available, generate emotional reaction
            if self.emotional_core:
                emotion_context = {
                    "user_input": user_input,
                    "user_id": user_id,
                    "trigger": "user_message"
                }
                
                emotion_thought = await self.generate_thought(ThoughtSource.EMOTION, emotion_context)
                generated_thoughts.append(emotion_thought)
            
            # If theory of mind available, generate thought about user's mental state
            if self.theory_of_mind:
                theory_context = {
                    "user_input": user_input,
                    "user_id": user_id,
                    "perspective": "user_mental_state"
                }
                
                theory_thought = await self.generate_thought(ThoughtSource.PERCEPTION, theory_context)
                generated_thoughts.append(theory_thought)
            
            # Generate a planning thought about how to respond
            planning_context = {
                "user_input": user_input,
                "user_id": user_id,
                "plan_type": "response_options"
            }
            
            planning_thought = await self.generate_thought(ThoughtSource.PLANNING, planning_context)
            generated_thoughts.append(planning_thought)
            
            # Connect thoughts to each other
            for i, thought in enumerate(generated_thoughts):
                for j, other_thought in enumerate(generated_thoughts):
                    if i != j:
                        thought.related_thoughts.append(other_thought.thought_id)
            
            return generated_thoughts
            
        except Exception as e:
            thoughts_logger.exception("Error in process_input")
            return generated_thoughts  # Return whatever thoughts were generated before the error
    
    async def process_output(self, planned_response: str, context: Union[Dict[str, Any], GeneralContext]) -> Tuple[str, List[InternalThought]]:
        """
        Process a planned response to generate internal thoughts and ensure no thought leakage.
        
        Args:
            planned_response: The planned response text
            context: Context for the response
        
        Returns:
            Tuple of (filtered_response, generated_thoughts)
        """
        generated_thoughts = []
        
        try:
            # First check for and filter out any thought-like content
            filter_result = self.filter_response_for_thoughts(planned_response)
            filtered_response = filter_result.filtered_response
            
            # If we found potential thoughts in the response, log them
            if filter_result.has_leakage:
                thoughts_logger.warning(f"Detected potential thought leakage in response: {filter_result.detected_thoughts}")
                
                # Add these as meta thoughts
                for thought_text in filter_result.detected_thoughts:
                    meta_context = ThoughtContext(original_response=planned_response)
                    meta_thought = InternalThought(
                        content=f"I noticed my response contained thought-like content that should stay internal: '{thought_text}'",
                        source=ThoughtSource.META,
                        priority=ThoughtPriority.HIGH,
                        context=meta_context
                    )
                    self._add_thought(meta_thought)
                    generated_thoughts.append(meta_thought)
            
            # If thought generation on output is enabled
            if self.config["thought_generation_on_output"]:
                # Generate a self-critique thought about the response
                context_str = str(context) if isinstance(context, dict) else context.model_dump_json()
                critique_context = {
                    "planned_response": filtered_response,
                    "original_context": context_str,
                    "critique_focus": "response_quality"
                }
                
                critique_thought = await self.generate_thought(ThoughtSource.SELF_CRITIQUE, critique_context)
                generated_thoughts.append(critique_thought)
                
                # Generate a reflection on the interaction
                reflection_context = {
                    "planned_response": filtered_response,
                    "original_context": context_str,
                    "reflection_focus": "interaction_dynamics"
                }
                
                reflection_thought = await self.generate_thought(ThoughtSource.REFLECTION, reflection_context)
                generated_thoughts.append(reflection_thought)
            
            return filtered_response, generated_thoughts
            
        except Exception as e:
            thoughts_logger.exception("Error in process_output")
            
            # In case of error, just return the filtered response to be safe
            filter_result = self.filter_response_for_thoughts(planned_response)
            return filter_result.filtered_response, generated_thoughts
    
    def filter_response_for_thoughts(self, response: str) -> ResponseFilterResult:
        """
        Filter a response to remove any content that looks like internal thoughts.
        
        Args:
            response: The response text to filter
        
        Returns:
            ResponseFilterResult with filtered text and detected thoughts
        """
        detected_thoughts = []
        filtered_response = response
        
        # Check for thought marker patterns
        matches = self.thought_pattern.finditer(response)
        
        for match in matches:
            start_idx = match.start()
            
            # Find the end of this thought (next sentence end, paragraph, or start of a new thought)
            end_markers = [". ", ".\n", "\n\n", "\n"]
            end_idx = len(response)
            
            for marker in end_markers:
                marker_idx = response.find(marker, start_idx)
                if marker_idx != -1 and marker_idx < end_idx:
                    end_idx = marker_idx + len(marker) - 1 if marker.endswith("\n") else marker_idx + len(marker)
            
            # Also check for other thought markers after this one
            next_match = self.thought_pattern.search(response, start_idx + 1)
            if next_match and next_match.start() < end_idx:
                end_idx = next_match.start()
            
            # Extract the thought text
            thought_text = response[start_idx:end_idx].strip()
            detected_thoughts.append(thought_text)
        
        # If thoughts were detected, remove them from the response
        if detected_thoughts:
            for thought in detected_thoughts:
                filtered_response = filtered_response.replace(thought, "")
            
            # Clean up any resulting double spaces or empty lines
            filtered_response = re.sub(r"\s+", " ", filtered_response)
            filtered_response = re.sub(r"\n\s*\n", "\n\n", filtered_response)
            filtered_response = filtered_response.strip()
        
        return ResponseFilterResult(
            filtered_response=filtered_response,
            detected_thoughts=detected_thoughts,
            has_leakage=len(detected_thoughts) > 0
        )
    
    async def _get_emotional_state(self) -> EmotionalState:
        """Get the current emotional state if emotional core is available."""
        if not self.emotional_core:
            return EmotionalState()
        
        try:
            if hasattr(self.emotional_core, "get_formatted_emotional_state"):
                state_dict = self.emotional_core.get_formatted_emotional_state()
                return EmotionalState(**{k: v for k, v in state_dict.items() if k in EmotionalState.model_fields})
            elif hasattr(self.emotional_core, "get_current_emotion"):
                emotion_data = await self.emotional_core.get_current_emotion()
                return EmotionalState(**{k: v for k, v in emotion_data.items() if k in EmotionalState.model_fields})
            else:
                return EmotionalState()
        except Exception as e:
            thoughts_logger.exception("Error getting emotional state")
            return EmotionalState(status="error")
    
    def _get_recent_thoughts(self, limit: int = 5) -> List[RecentThoughtSummary]:
        """Get recent thoughts for context."""
        recent_thoughts = sorted(self.active_thoughts, key=lambda x: x.created_at, reverse=True)[:limit]
        
        # Convert to RecentThoughtSummary objects
        return [
            RecentThoughtSummary(
                content=thought.content,
                source=thought.source.value,
                created_at=thought.created_at.isoformat()
            )
            for thought in recent_thoughts
        ]
    
    async def get_thoughts(self, filter_criteria: ThoughtFilter = None) -> List[InternalThought]:
        """
        Get thoughts matching the filter criteria.
        
        Args:
            filter_criteria: Criteria to filter thoughts
        
        Returns:
            List of matching thoughts
        """
        if filter_criteria is None:
            filter_criteria = ThoughtFilter()
        
        async with self._lock:
            # Apply filter to active thoughts
            matching_thoughts = [
                thought for thought in self.active_thoughts
                if filter_criteria.matches(thought)
            ]
            
            # Sort by recency (newest first)
            matching_thoughts.sort(key=lambda x: x.created_at, reverse=True)
            
            # Apply limit
            return matching_thoughts[:filter_criteria.limit]
    
    async def archive_old_thoughts(self):
        """Archive thoughts older than the configured threshold."""
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=self.config["automatic_archiving_hours"])
        
        async with self._lock:
            # Find thoughts to archive
            to_archive = [t for t in self.active_thoughts if t.created_at < cutoff_time]
            
            # Move to archive
            for thought in to_archive:
                self.active_thoughts.remove(thought)
                self.archived_thoughts.append(thought)
            
            thoughts_logger.info(f"Archived {len(to_archive)} old thoughts")
    
    async def get_debug_thoughts_log(self, limit: int = 50) -> List[DebugThoughtEntry]:
        """
        Get a formatted log of thoughts for debugging purposes.
        Only available in debug mode.
        
        Args:
            limit: Maximum number of thoughts to return
        
        Returns:
            List of thought records
        """
        if not self.config["debug_mode"]:
            return [DebugThoughtEntry(
                id="warning",
                content="Debug mode is disabled. Enable debug_mode in config to use this feature.",
                source="system",
                priority="high",
                created_at=datetime.datetime.now().isoformat(),
                warning="Debug mode is disabled. Enable debug_mode in config to use this feature."
            )]
        
        async with self._lock:
            # Combine active and archived, sort by recency
            all_thoughts = self.active_thoughts + self.archived_thoughts
            all_thoughts.sort(key=lambda x: x.created_at, reverse=True)
            
            # Format for debug display
            return [
                DebugThoughtEntry(
                    id=thought.thought_id,
                    content=thought.content,
                    source=thought.source.value,
                    priority=thought.priority.value,
                    created_at=thought.created_at.isoformat(),
                    critique=thought.critique,
                    is_active=thought in self.active_thoughts
                )
                for thought in all_thoughts[:limit]
            ]
    
    def get_stats(self) -> ThoughtStats:
        """Get statistics about the internal thoughts system."""
        thoughts_by_source = {}
        thoughts_by_priority = {}
        
        # Count by source
        for source in ThoughtSource:
            thoughts_by_source[source.value] = sum(1 for t in self.active_thoughts if t.source == source)
        
        # Count by priority
        for priority in ThoughtPriority:
            thoughts_by_priority[priority.value] = sum(1 for t in self.active_thoughts if t.priority == priority)
        
        return ThoughtStats(
            active_thoughts_count=len(self.active_thoughts),
            archived_thoughts_count=len(self.archived_thoughts),
            thoughts_by_source=thoughts_by_source,
            thoughts_by_priority=thoughts_by_priority,
            thoughts_with_critique=sum(1 for t in self.active_thoughts if t.critique),
            oldest_active_thought=min(t.created_at for t in self.active_thoughts).isoformat() if self.active_thoughts else None,
            newest_active_thought=max(t.created_at for t in self.active_thoughts).isoformat() if self.active_thoughts else None
        )
    
    async def integrate_with_observation_system(self):
        """
        Integrate with the passive observation system to convert observations to thoughts.
        """
        if not self.passive_observation_system:
            return
        
        try:
            # Get relevant observations
            filter_criteria = await self.passive_observation_system.ObservationFilter(
                min_relevance=0.6,
                max_age_seconds=3600  # Last hour
            )
            
            observations = await self.passive_observation_system.get_relevant_observations(
                filter_criteria=filter_criteria,
                limit=3
            )
            
            # Convert each to an internal thought
            for observation in observations:
                # Map observation source to thought source
                source_mapping = {
                    "environment": ThoughtSource.PERCEPTION,
                    "self": ThoughtSource.REFLECTION,
                    "relationship": ThoughtSource.REFLECTION,
                    "memory": ThoughtSource.MEMORY,
                    "temporal": ThoughtSource.PERCEPTION,
                    "sensory": ThoughtSource.PERCEPTION,
                    "pattern": ThoughtSource.REASONING,
                    "emotion": ThoughtSource.EMOTION,
                    "need": ThoughtSource.REFLECTION,
                    "user": ThoughtSource.PERCEPTION,
                    "meta": ThoughtSource.META
                }
                
                thought_source = source_mapping.get(observation.source.value, ThoughtSource.PERCEPTION)
                
                # Map priority
                priority_mapping = {
                    "low": ThoughtPriority.LOW,
                    "medium": ThoughtPriority.MEDIUM,
                    "high": ThoughtPriority.HIGH,
                    "urgent": ThoughtPriority.CRITICAL
                }
                
                thought_priority = priority_mapping.get(observation.priority.value, ThoughtPriority.MEDIUM)
                
                # Create thought context and metadata
                thought_context = ThoughtContext(
                    observation_id=observation.observation_id,
                    type="from_observation"
                )
                thought_metadata = ThoughtMetadata(
                    observation_id=observation.observation_id,
                    type="from_observation"
                )
                
                # Create thought from observation
                thought = InternalThought(
                    content=f"Observation: {observation.content}",
                    source=thought_source,
                    priority=thought_priority,
                    context=thought_context,
                    metadata=thought_metadata
                )
                
                # Add to active thoughts
                self._add_thought(thought)
        
        except Exception as e:
            thoughts_logger.exception("Error integrating with observation system")
    
    async def integrate_with_reflection_engine(self):
        """
        Integrate with the reflection engine to convert reflections to thoughts.
        """
        if not self.reflection_engine:
            return
        
        try:
            # If the reflection engine has reflection_history, use it
            if hasattr(self.reflection_engine, 'reflection_history') and self.reflection_engine.reflection_history:
                # Get the most recent reflection
                recent_reflection = self.reflection_engine.reflection_history[-1]
                
                # Create thought context and metadata
                thought_context = ThoughtContext(
                    reflection_id=recent_reflection.get("timestamp", ""),
                    type="from_reflection",
                    timestamp=recent_reflection.get("timestamp", "")
                )
                thought_metadata = ThoughtMetadata(
                    type="from_reflection",
                    timestamp=recent_reflection.get("timestamp", "")
                )
                
                # Create a thought from it
                thought = InternalThought(
                    content=f"Reflection: {recent_reflection.get('reflection', '')}",
                    source=ThoughtSource.REFLECTION,
                    priority=ThoughtPriority.MEDIUM,
                    context=thought_context,
                    metadata=thought_metadata
                )
                
                # Add to active thoughts
                self._add_thought(thought)
        
        except Exception as e:
            thoughts_logger.exception("Error integrating with reflection engine")


# Main integration hooks for the response pipeline

async def pre_process_input(thoughts_manager: InternalThoughtsManager, user_input: str, user_id: str = None) -> List[InternalThought]:
    """
    Hook to add to the main input processing pipeline.
    
    Args:
        thoughts_manager: The internal thoughts manager
        user_input: The user's input text
        user_id: Optional user ID
    
    Returns:
        Generated thoughts
    """
    return await thoughts_manager.process_input(user_input, user_id)


async def pre_process_output(thoughts_manager: InternalThoughtsManager, planned_response: str, context: Union[Dict[str, Any], GeneralContext]) -> str:
    """
    Hook to add to the main output processing pipeline.
    
    Args:
        thoughts_manager: The internal thoughts manager
        planned_response: The planned response text
        context: Response context
    
    Returns:
        Filtered response with no thought leakage
    """
    filtered_response, _ = await thoughts_manager.process_output(planned_response, context)
    return filtered_response


# Developer/debug helper functions

def get_thoughts_log(thoughts_manager: InternalThoughtsManager) -> str:
    """
    Get a formatted log of internal thoughts for debugging.
    
    Args:
        thoughts_manager: The internal thoughts manager
    
    Returns:
        Formatted log string
    """
    if not thoughts_manager.config["debug_mode"]:
        return "Debug mode is disabled. Enable debug_mode in config to use this feature."
    
    log_lines = []
    log_lines.append("=== INTERNAL THOUGHTS LOG ===")
    
    # Get stats
    stats = thoughts_manager.get_stats()
    log_lines.append(f"Active thoughts: {stats.active_thoughts_count}")
    log_lines.append(f"Archived thoughts: {stats.archived_thoughts_count}")
    log_lines.append(f"Thoughts with critique: {stats.thoughts_with_critique}")
    log_lines.append("")
    
    # Get recent thoughts
    async def get_recent():
        recent = await thoughts_manager.get_thoughts(ThoughtFilter(limit=10))
        return recent
    
    recent_thoughts = asyncio.run(get_recent())
    
    # Format each thought
    for thought in recent_thoughts:
        log_lines.append(f"ID: {thought.thought_id}")
        log_lines.append(f"TIME: {thought.created_at.isoformat()}")
        log_lines.append(f"SOURCE: {thought.source.value}")
        log_lines.append(f"PRIORITY: {thought.priority.value}")
        log_lines.append("CONTENT:")
        log_lines.append(thought.content)
        if thought.critique:
            log_lines.append("CRITIQUE:")
            log_lines.append(thought.critique)
        log_lines.append("-" * 40)
        log_lines.append("")
    
    return "\n".join(log_lines)
