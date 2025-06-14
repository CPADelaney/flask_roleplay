# nyx/core/recognition_memory.py

import logging
import asyncio
import datetime
import random
import uuid
import json
from typing import Dict, List, Any, Optional, Set, Union, Tuple, TypedDict
from pydantic import BaseModel, Field, ConfigDict

# OpenAI Agents SDK imports
from agents import (
    Agent, 
    Runner, 
    ModelSettings, 
    trace, 
    function_tool, 
    handoff, 
    RunContextWrapper,
    RunConfig,
    InputGuardrail,
    GuardrailFunctionOutput
)

logger = logging.getLogger(__name__)

# ============================================================================
# Simple data classes to avoid Dict issues
# ============================================================================

class ContextRequirement(BaseModel):
    """Single context requirement"""
    model_config = ConfigDict(extra='forbid')
    
    key: str
    value: str

class ConfidenceLevel(BaseModel):
    """Confidence level for a specific aspect"""
    model_config = ConfigDict(extra='forbid')
    
    aspect: str
    confidence: float

class SimilarTrigger(BaseModel):
    """Similar trigger information"""
    model_config = ConfigDict(extra='forbid')
    
    trigger_id: str
    trigger_value: str
    similarity: float

class RelevanceHierarchyItem(BaseModel):
    """Item in relevance hierarchy"""
    model_config = ConfigDict(extra='forbid')
    
    memory_id: str
    relevance: float
    rank: int

class ConceptItem(BaseModel):
    """Simple concept representation"""
    model_config = ConfigDict(extra='forbid')
    
    concept: str
    description: str

class PatternItem(BaseModel):
    """Pattern representation"""
    model_config = ConfigDict(extra='forbid')
    
    pattern_type: str
    pattern_value: str

# Keep TypedDict for backwards compatibility but minimize their use
class TriggerDict(TypedDict):
    """Simplified trigger dict for backwards compatibility"""
    trigger_id: str
    trigger_type: str
    trigger_value: str
    relevance_threshold: float
    activation_strength: float
    source: str

class MemoryDict(TypedDict):
    """Simplified memory dict for backwards compatibility"""
    id: str
    memory_text: str
    memory_type: str
    relevance: float
    significance: int

class EntityDict(TypedDict):
    """Entity analysis result"""
    entity: str
    type: str
    salience: float
    position: int

class EmotionDict(TypedDict):
    """Emotion analysis result"""
    emotion: str
    trigger_word: str
    intensity: float
    position: int

class TopicDict(TypedDict):
    """Topic analysis result"""
    topic: str
    count: int
    salience: float
    context: str

# ============================================================================
# New Pydantic models to replace Dict[str, Any]
# ============================================================================

class RecognitionTimestamp(BaseModel):
    """Model for recognition timestamp"""
    model_config = ConfigDict(extra='forbid')
    
    memory_id: str
    recognized_at: str

class MessageContext(BaseModel):
    """Model for conversation messages"""
    model_config = ConfigDict(extra='forbid')
    
    text: str
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    context_items: List[ContextRequirement] = Field(default_factory=list)

class MemoryData(BaseModel):
    """Model for memory data"""
    model_config = ConfigDict(extra='forbid')
    
    id: str
    memory_text: str
    memory_type: str
    relevance: float = 0.5
    significance: int = 5
    confidence: float = 0.5
    activation_trigger: Optional[TriggerDict] = None
    metadata_items: List[ContextRequirement] = Field(default_factory=list)

class TriggerPerformance(BaseModel):
    """Model for trigger performance tracking"""
    model_config = ConfigDict(extra='forbid')
    
    trigger_id: str
    query_count: int = 0
    total_memories: int = 0
    avg_relevance: float = 0.0
    needs_optimization: bool = False
    last_updated: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

class ContextState(BaseModel):
    """Model for context state analysis"""
    model_config = ConfigDict(extra='forbid')
    
    recognition_sensitivity: float
    max_recognitions: int
    context_complexity: int
    active_triggers_count: int

class RecognitionHistory(BaseModel):
    """Model for recognition history check"""
    model_config = ConfigDict(extra='forbid')
    
    was_recognized: bool
    seconds_ago: Optional[float] = None
    cooldown_period: float
    cooldown_remaining: float
    in_cooldown: bool

class EntityAnalysis(BaseModel):
    """Model for entity analysis results"""
    model_config = ConfigDict(extra='forbid')
    
    entity: str
    type: str
    salience: float
    position: int

class EmotionAnalysis(BaseModel):
    """Model for emotion analysis results"""
    model_config = ConfigDict(extra='forbid')
    
    emotion: str
    trigger_word: Optional[str] = None
    trigger_pattern: Optional[str] = None
    intensity: float
    position: int

class TopicAnalysis(BaseModel):
    """Model for topic analysis results"""
    model_config = ConfigDict(extra='forbid')
    
    topic: str
    count: int
    salience: float
    context: str

class NarrativeElement(BaseModel):
    """Model for narrative element analysis"""
    model_config = ConfigDict(extra='forbid')
    
    element_type: str
    marker: Optional[str] = None
    indicator: Optional[str] = None
    context: str
    salience: float
    position: int

class ContextAwarenessResult(BaseModel):
    """Model for context awareness results"""
    model_config = ConfigDict(extra='forbid')
    
    entities: List[EntityAnalysis]
    topics: List[TopicAnalysis]
    emotions: List[EmotionAnalysis]

class BlendResult(BaseModel):
    """Model for memory blending results"""
    model_config = ConfigDict(extra='forbid')
    
    blended_snippet: str
    blend_score: float

class TriggerQuality(BaseModel):
    """Model for trigger quality assessment"""
    model_config = ConfigDict(extra='forbid')
    
    quality_score: float
    is_generic: bool
    similar_triggers: List[SimilarTrigger]
    specificity: float
    recommended_threshold: float

class TriggerParameters(BaseModel):
    """Model for calibrated trigger parameters"""
    model_config = ConfigDict(extra='forbid')
    
    relevance_threshold: float
    activation_strength: float

class ConversationalImpact(BaseModel):
    """Model for conversational impact assessment"""
    model_config = ConfigDict(extra='forbid')
    
    impact_score: float
    novelty: float
    relevance: float
    coherence: float
    impact_type: str

class CausalityAssessment(BaseModel):
    """Model for causality assessment"""
    model_config = ConfigDict(extra='forbid')
    
    causality_strength: float
    causality_markers_found: List[str]

class ParallelQueryResultItem(BaseModel):
    """Single result from parallel query"""
    model_config = ConfigDict(extra='forbid')
    
    trigger_id: str
    memories: List[MemoryData]

class ParallelQueryResult(BaseModel):
    """Model for parallel query results"""
    model_config = ConfigDict(extra='forbid')
    
    results: List[ParallelQueryResultItem]

# ============================================================================
# Existing Pydantic models for recognition memory
# ============================================================================

class ContextualTrigger(BaseModel):
    """Schema for a contextual trigger that activates recognition memory"""
    model_config = ConfigDict(extra='forbid')
    
    trigger_id: str = Field(default_factory=lambda: f"trigger_{uuid.uuid4().hex[:8]}")
    trigger_type: str  # e.g., "entity", "concept", "event", "emotion", "pattern"
    trigger_value: str
    relevance_threshold: float = 0.6
    activation_strength: float = 1.0
    context_requirements: List[ContextRequirement] = Field(default_factory=list)
    source: str = "system"  # Where this trigger came from

class RecognitionResult(BaseModel):
    """Schema for a recognition memory result"""
    model_config = ConfigDict(extra='forbid')
    
    memory_id: str
    memory_text: str
    memory_type: str
    relevance_score: float
    confidence: float
    activation_reason: str
    activation_trigger: str
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

class ContextualCue(BaseModel):
    """Schema for a contextual cue extracted from conversation"""
    model_config = ConfigDict(extra='forbid')
    
    cue_type: str  # e.g., "entity", "keyword", "phrase", "emotion", "topic"
    cue_value: str
    salience: float
    source_text: str
    extracted_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

class SalienceAnalysisOutput(BaseModel):
    """Output schema for conversation salience analysis"""
    model_config = ConfigDict(extra='forbid')
    
    salient_entities: List[EntityDict]
    salient_concepts: List[ConceptItem]
    emotional_markers: List[EmotionDict]
    key_topics: List[TopicDict]
    notable_patterns: List[PatternItem]

class TriggerExtractionOutput(BaseModel):
    """Output schema for trigger extraction"""
    model_config = ConfigDict(extra='forbid')
    
    contextual_triggers: List[ContextualTrigger]
    extraction_confidence: float
    trigger_count: int
    potential_triggers: List[TriggerDict]

class RecognitionFilterOutput(BaseModel):
    """Output schema for recognition filtering"""
    model_config = ConfigDict(extra='forbid')
    
    selected_memories: List[MemoryDict]
    selection_rationale: str
    confidence_levels: List[ConfidenceLevel]
    relevance_hierarchy: List[RelevanceHierarchyItem]

class RecognitionMemoryContext:
    """Context object for recognition memory operations"""
    
    def __init__(self, memory_core=None, context_awareness=None, reasoning_core=None):
        self.memory_core = memory_core
        self.context_awareness = context_awareness
        self.reasoning_core = reasoning_core
        
        # Recent context tracking
        self.recent_conversation: List[MessageContext] = []
        self.current_context_distribution = None
        
        # Active triggers registry - store as serialized JSON
        self._active_triggers_storage: Dict[str, str] = {}
        
        # Recently recognized memory IDs with timestamps
        self.recent_recognitions: List[Tuple[str, datetime.datetime]] = []
        
        # Contextual cues collection
        self.contextual_cues: List[ContextualCue] = []
        
        # Trace ID for connecting traces
        self.trace_id = f"recognition_memory_{datetime.datetime.now().isoformat()}"
        
        # Control parameters
        self.max_triggers_per_turn = 5
        self.max_recognitions_per_turn = 3
        self.recognition_cooldown = 600  # Seconds before a memory can be recognized again
        self.context_history_size = 5  # Number of turns to keep in history
        self.cue_retention_period = 24 * 60 * 60  # Seconds to retain contextual cues (1 day)
        self.recognition_sensitivity = 0.7
        self.max_memories_per_trigger = 3
        self.max_memories_to_return = 3
        
        # Trigger types
        self.trigger_types = [
            "entity", "concept", "keyword", "phrase", 
            "emotion", "topic", "event", "pattern"
        ]
        
        # Default triggers (can be extended)
        self.default_triggers: List[ContextualTrigger] = []
        
        # Trigger performance tracking
        self.trigger_performance: List[TriggerPerformance] = []
    
    @property
    def active_triggers(self) -> Dict[str, ContextualTrigger]:
        """Get active triggers by deserializing from storage"""
        triggers = {}
        for trigger_id, trigger_json in self._active_triggers_storage.items():
            try:
                triggers[trigger_id] = ContextualTrigger.model_validate_json(trigger_json)
            except Exception:
                pass
        return triggers
    
    def add_trigger(self, trigger: ContextualTrigger) -> None:
        """Add a trigger by serializing it"""
        self._active_triggers_storage[trigger.trigger_id] = trigger.model_dump_json()
    
    def remove_trigger(self, trigger_id: str) -> bool:
        """Remove a trigger"""
        if trigger_id in self._active_triggers_storage:
            del self._active_triggers_storage[trigger_id]
            return True
        return False

class RecognitionMemorySystem:
    """
    System that automatically retrieves contextually relevant memories
    based on recognition triggers rather than explicit queries.
    """
    
    def __init__(self, memory_core=None, context_awareness=None, reasoning_core=None):
        """Initialize the recognition memory system with required components"""
        # Create context
        self.context = RecognitionMemoryContext(
            memory_core=memory_core,
            context_awareness=context_awareness,
            reasoning_core=reasoning_core
        )
        
        # Initialize agent system
        self.recognition_agent = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the recognition memory system and its agents"""
        if self.initialized:
            return
            
        logger.info("Initializing RecognitionMemorySystem")
        
        with trace(workflow_name="RecognitionMemory Initialization", group_id=self.context.trace_id):
            # Initialize memory core interface if needed
            if self.context.memory_core and not hasattr(self.context.memory_core, "initialized"):
                await self.context.memory_core.initialize()
                
            # Initialize reasoning core if available
            if self.context.reasoning_core and not hasattr(self.context.reasoning_core, "initialized"):
                # Initialize with minimal configuration for recognition purposes
                pass
                
            self._initialize_agents()
            self._initialize_default_triggers()
            self.initialized = True
            logger.info("RecognitionMemorySystem initialized with Agents SDK")
    
    def _initialize_agents(self):
        """Initialize all specialized agents needed for the recognition memory system"""
        # Create specialized agents
        self.salience_detector_agent = self._create_salience_detector_agent()
        self.trigger_extraction_agent = self._create_trigger_extraction_agent()
        self.memory_query_agent = self._create_memory_query_agent()
        self.relevance_filter_agent = self._create_relevance_filter_agent()
        
        # Create the main recognition memory agent with handoffs
        self.recognition_agent = self._create_recognition_agent()
        
        logger.info("Recognition memory agents initialized")
    
    def _initialize_default_triggers(self):
        """Initialize default recognition triggers"""
        # Create some default triggers that should always be active
        default_triggers = [
            ContextualTrigger(
                trigger_type="entity",
                trigger_value="user",
                relevance_threshold=0.5,
                activation_strength=0.8,
                source="system_default"
            ),
            ContextualTrigger(
                trigger_type="emotion",
                trigger_value="happiness",
                relevance_threshold=0.6,
                activation_strength=0.7,
                source="system_default"
            ),
            ContextualTrigger(
                trigger_type="emotion",
                trigger_value="anger",
                relevance_threshold=0.6,
                activation_strength=0.9,
                source="system_default"
            ),
            ContextualTrigger(
                trigger_type="concept",
                trigger_value="trust",
                relevance_threshold=0.5,
                activation_strength=0.8,
                source="system_default"
            )
        ]
        
        # Add to default triggers list
        self.context.default_triggers.extend(default_triggers)
        
        # Add to active triggers
        for trigger in default_triggers:
            self.context.add_trigger(trigger)
    
    def _create_recognition_agent(self) -> Agent:
        """Create optimized main orchestrator agent for recognition memory"""
        return Agent[RecognitionMemoryContext](
            name="Recognition Memory Orchestrator",
            instructions="""You are the recognition memory orchestration system for the Nyx AI.
            
            Your role is to identify memories that should be automatically recognized based
            on the current conversation context without explicit retrieval requests.
            
            Recognition memory works differently from free recall:
            - In free recall, memories are explicitly searched for with a query
            - In recognition memory, relevant memories are automatically surfaced when
              contextual elements trigger pattern matching in memory
              
            Always prioritize:
            1. Quality over quantity - only surface truly relevant memories
            2. Conversational appropriateness - memories should add value
            3. Cognitive plausibility - recognition should feel natural
            4. Contextual depth - consider both surface and deeper contextual factors
            
            Follow a recognition memory process that mimics human cognition:
            1. Detect salient elements in the conversation
            2. Extract contextual triggers from these elements
            3. Match triggers against memory store
            4. Filter and rank matched memories for relevance
            5. Return only the most valuable matches (max 3)
            
            Remember that each stage builds on the previous one, forming a
            recognition pipeline that leads to high-quality memory retrieval.
            """,
            tools=[
                _get_active_triggers,
                _get_recent_recognitions,
                _update_context_history,
                _analyze_context_state
            ],
            handoffs=[
                handoff(
                    self.salience_detector_agent,
                    tool_name_override="detect_salient_elements",
                    tool_description_override="Detect salient elements in conversation that could trigger memory recognition",
                    on_handoff=prepare_for_salience_detection
                ),
                handoff(
                    self.trigger_extraction_agent,
                    tool_name_override="extract_contextual_triggers",
                    tool_description_override="Extract contextual triggers from the salient elements detected in conversation",
                    on_handoff=prepare_for_trigger_extraction
                ),
                handoff(
                    self.memory_query_agent,
                    tool_name_override="query_based_on_triggers",
                    tool_description_override="Query memory system based on the extracted contextual triggers",
                    on_handoff=prepare_for_memory_query
                ),
                handoff(
                    self.relevance_filter_agent,
                    tool_name_override="filter_recognition_results",
                    tool_description_override="Filter memory recognition results for contextual relevance and quality",
                    on_handoff=prepare_for_relevance_filtering
                )
            ],
            output_type=List[RecognitionResult],
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.4)
        )
    
    def _create_salience_detector_agent(self) -> Agent:
        """Create enhanced agent for detecting salient elements in conversation"""
        return Agent[RecognitionMemoryContext](
            name="Enhanced Salience Detector",
            instructions="""You are specialized in detecting cognitively salient elements in conversation
            that could trigger recognition memory, mimicking human attention patterns.
            
            Your task is to identify elements that naturally stand out in conversation:
            
            1. Entities with psychological significance (people, places, objects with emotional weight)
            2. Conceptual hooks (ideas that naturally connect to other memories/knowledge)
            3. Emotional markers (both explicit and implicit emotional content)
            4. Narrative elements (key developments in conversational storyline)
            5. Pattern activators (elements that fit or break established patterns)
            
            Focus on elements with high attentional capture - those that would naturally
            trigger associative memory in human cognition. Consider both explicit content
            and implicit/subtext elements.
            
            Prioritize elements with:
            - Emotional salience
            - Personal relevance
            - Novelty/surprise value
            - Conceptual richness
            - Narrative importance
            """,
            tools=[
                _analyze_entities,
                _analyze_emotions,
                _analyze_topics,
                _analyze_narrative_elements,
                _leverage_context_awareness
            ],
            output_type=SalienceAnalysisOutput,
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.3)
        )
        
    def _create_memory_query_agent(self) -> Agent:
        """Create enhanced agent for querying memory based on triggers"""
        return Agent[RecognitionMemoryContext](
            name="Enhanced Memory Query Agent",
            instructions="""You are specialized in querying the memory system based on
            contextual triggers using advanced retrieval strategies.
            
            Your task is to:
            1. Formulate sophisticated queries for each trigger
            2. Leverage specialized memory retrieval functions
            3. Balance parallel and sequential retrieval strategies
            4. Apply prioritization across memory types
            5. Adjust retrieval parameters based on trigger properties
            6. Track query performance for ongoing optimization
            
            Focus on finding memories that are cognitively plausible matches for
            the current conversational context. Balance diversification (finding
            varied memory types) with precision (finding highly relevant memories).
            
            When querying:
            - Use prioritization to emphasize experiences and reflections
            - Adjust relevance thresholds based on trigger activation strength
            - Consider memory significance levels in retrieval strategy
            - Apply memory type-specific retrieval techniques
            """,
            tools=[
                _query_memory,
                _query_with_prioritization,
                _query_memories_parallel,
                _combine_query_results,
                _track_query_performance
            ],
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.3)
        )
        
    def _create_trigger_extraction_agent(self) -> Agent:
        """Create enhanced agent for extracting contextual triggers"""
        return Agent[RecognitionMemoryContext](
            name="Enhanced Trigger Extraction Agent",
            instructions="""You are specialized in extracting and formulating contextual
            triggers from salient elements identified in conversation.
            
            Your task is to:
            1. Convert salient elements into formal recognition triggers
            2. Calibrate relevance thresholds based on element salience
            3. Set appropriate activation strengths
            4. Determine specific context requirements for trigger activation
            5. Balance specificity vs. generality for optimal trigger effectiveness
            6. Consider trigger interactions and potential overlaps
            
            Focus on creating cognitively plausible triggers that mirror human
            memory recognition patterns. A good trigger should:
            - Be specific enough to avoid excessive false positives
            - Be general enough to catch meaningful associations
            - Have appropriate activation parameters based on salience
            - Account for context dependencies in recognition
            - Work well in combination with other active triggers
            """,
            tools=[
                _get_trigger_types,
                _format_trigger,
                _assess_trigger_quality,
                _calibrate_trigger_parameters
            ],
            output_type=TriggerExtractionOutput,
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.4)
        )
        
    def _create_relevance_filter_agent(self) -> Agent:
        """Create enhanced agent for filtering recognition results"""
        return Agent[RecognitionMemoryContext](
            name="Enhanced Relevance Filter Agent",
            instructions="""You are specialized in filtering recognition memory results
            for cognitive relevance, conversational value, and psychological plausibility.
            
            Your task is to:
            1. Evaluate memories for deep contextual relevance, not just surface matching
            2. Assess conversational appropriateness and timing
            3. Consider causal connections between context and memories
            4. Evaluate potential for valuable conceptual blending
            5. Filter to prevent repetitive or redundant recognitions
            6. Ensure memory selection feels psychologically authentic
            
            Focus on selecting memories that would naturally emerge in human
            cognition during this conversational context. Balance:
            - Relevance (contextual fit)
            - Novelty (adds new information)
            - Value (enhances the conversation)
            - Plausibility (feels like natural memory recognition)
            - Diversity (appropriate mix of memory types)
            
            Prioritize memories that enable insights through their connection
            to the current context.
            """,
            tools=[
                _calculate_contextual_relevance,
                _assess_memory_causality,
                _blend_memory_with_context,
                _check_recognition_history,
                _assess_conversational_impact
            ],
            output_type=RecognitionFilterOutput,
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.3)
        )
    
    # Main public methods
    
    async def process_conversation_turn(
        self,
        conversation_text: str,
        current_context: Dict[str, Any] = None
    ) -> List[RecognitionResult]:
        """
        Process a conversation turn to trigger recognition memory
        
        Args:
            conversation_text: Text from the current conversation turn
            current_context: Current context information
            
        Returns:
            List of recognized memories
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
        
        # Format as message for context history
        context_items = []
        if current_context:
            for key, value in current_context.items():
                context_items.append(ContextRequirement(key=key, value=str(value)))
        
        message = MessageContext(
            text=conversation_text,
            context_items=context_items
        )
        
        with trace(workflow_name="Process Conversation Turn", group_id=self.context.trace_id):
            # Update context history
            await _update_context_history(
                RunContextWrapper(context=self.context),
                message
            )
            
            # Prepare prompt for recognition agent
            prompt = f"""Process this conversation turn for recognition memory:
            
            Current message: {conversation_text}
            
            Identify salient elements that could trigger recognition memory,
            extract contextual triggers, query the memory system, and filter
            the results for relevance and quality.
            """
            
            # Configure run with tracing
            run_config = RunConfig(
                workflow_name="Recognition Memory Processing",
                group_id=self.context.trace_id,
                trace_metadata={
                    "message_length": len(conversation_text),
                    "active_triggers": len(self.context.active_triggers)
                }
            )
            
            # Run through recognition agent
            result = await Runner.run(
                self.recognition_agent,
                prompt,
                context=self.context,
                run_config=run_config
            )
            
            recognition_results = result.final_output
            
            # Update recent recognitions
            self._update_recent_recognitions(recognition_results)
            
            return recognition_results
    
    def _update_recent_recognitions(self, recognition_results: List[RecognitionResult]) -> None:
        """
        Update the list of recently recognized memories
        
        Args:
            recognition_results: New recognition results
        """
        now = datetime.datetime.now()
        
        # Add new recognitions
        for result in recognition_results:
            self.context.recent_recognitions.append((result.memory_id, now))
        
        # Prune old recognitions beyond cooldown period
        current_recognitions = []
        for memory_id, timestamp in self.context.recent_recognitions:
            if (now - timestamp).total_seconds() < self.context.recognition_cooldown:
                current_recognitions.append((memory_id, timestamp))
        
        self.context.recent_recognitions = current_recognitions
    
    async def add_contextual_trigger(
        self,
        trigger_type: str,
        trigger_value: str,
        relevance_threshold: float = 0.6,
        activation_strength: float = 1.0,
        context_requirements: Optional[List[ContextRequirement]] = None,
        source: str = "user"
    ) -> Optional[str]:
        """
        Add a new contextual trigger
        
        Args:
            trigger_type: Type of trigger
            trigger_value: Value of the trigger
            relevance_threshold: Minimum relevance for activation
            activation_strength: Strength of activation
            context_requirements: Additional context requirements
            source: Source of the trigger
            
        Returns:
            ID of the created trigger
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
        
        try:
            trigger = await _format_trigger(
                RunContextWrapper(context=self.context),
                trigger_type=trigger_type,
                trigger_value=trigger_value,
                relevance_threshold=relevance_threshold,
                activation_strength=activation_strength,
                context_requirements=context_requirements or {}
            )
            
            # Set source
            trigger.source = source
            
            # Add to active triggers
            self.context.add_trigger(trigger)
            
            return trigger.trigger_id
            
        except Exception as e:
            logger.error(f"Error adding contextual trigger: {e}")
            return None
    
    async def remove_contextual_trigger(self, trigger_id: str) -> bool:
        """
        Remove a contextual trigger
        
        Args:
            trigger_id: ID of the trigger to remove
            
        Returns:
            True if successfully removed, False otherwise
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
        
        return self.context.remove_trigger(trigger_id)
    
    async def get_active_triggers(self) -> Dict[str, ContextualTrigger]:
        """
        Get currently active contextual triggers
        
        Returns:
            Dictionary of active triggers
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
        
        return self.context.active_triggers
    
    async def register_contextual_cue(
        self,
        cue_type: str,
        cue_value: str,
        salience: float,
        source_text: str
    ) -> ContextualCue:
        """
        Register a contextual cue for later processing
        
        Args:
            cue_type: Type of cue
            cue_value: Value of the cue
            salience: Salience/importance of the cue
            source_text: Source text containing the cue
            
        Returns:
            Created contextual cue
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
        
        cue = ContextualCue(
            cue_type=cue_type,
            cue_value=cue_value,
            salience=salience,
            source_text=source_text
        )
        
        self.context.contextual_cues.append(cue)
        
        # Prune old cues
        now = datetime.datetime.now()
        current_cues = []
        for old_cue in self.context.contextual_cues:
            try:
                cue_time = datetime.datetime.fromisoformat(old_cue.extracted_at)
                if (now - cue_time).total_seconds() < self.context.cue_retention_period:
                    current_cues.append(old_cue)
            except Exception:
                # Keep cues with invalid timestamps
                current_cues.append(old_cue)
        
        self.context.contextual_cues = current_cues
        
        return cue
    
    async def process_accumulated_cues(self) -> List[ContextualTrigger]:
        """
        Process accumulated contextual cues to generate triggers
        
        Returns:
            List of generated contextual triggers
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
        
        if not self.context.contextual_cues:
            return []
        
        with trace(workflow_name="Process Accumulated Cues", group_id=self.context.trace_id):
            # Extract cue information
            cue_data = []
            for cue in self.context.contextual_cues:
                cue_data.append({
                    "cue_type": cue.cue_type,
                    "cue_value": cue.cue_value,
                    "salience": cue.salience,
                    "source_text": cue.source_text,
                    "extracted_at": cue.extracted_at
                })
            
            # Run trigger extraction on accumulated cues
            prompt = f"""Process these accumulated contextual cues to generate recognition triggers:
            
            Cue data: {json.dumps(cue_data, indent=2)}
            
            Extract contextual triggers that can be used for memory recognition.
            Focus on patterns and themes across multiple cues.
            """
            
            result = await Runner.run(
                self.trigger_extraction_agent,
                prompt,
                context=self.context
            )
            
            extraction_result = result.final_output
            
            # Add extracted triggers to active triggers
            generated_triggers = []
            if hasattr(extraction_result, "contextual_triggers"):
                for trigger in extraction_result.contextual_triggers:
                    trigger.source = "accumulated_cues"
                    self.context.add_trigger(trigger)
                    generated_triggers.append(trigger)
            
            return generated_triggers
    
    async def integrate_with_novelty_engine(
        self,
        conversation_text: str,
        novelty_engine=None
    ) -> Dict[str, Any]:
        """
        Integrate recognition memory with novelty engine
        
        Args:
            conversation_text: Current conversation text
            novelty_engine: Reference to novelty engine if available
            
        Returns:
            Integration results
        """
        if not novelty_engine:
            return {"error": "Novelty engine not provided"}
        
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="Recognition-Novelty Integration", group_id=self.context.trace_id):
            # Get recognized memories
            recognition_results = await self.process_conversation_turn(conversation_text)
            
            if not recognition_results:
                return {"status": "no_recognitions", "result": None}
            
            # Extract concepts from recognized memories
            memory_concepts = []
            for recog in recognition_results:
                # Extract simple concept from memory text
                words = recog.memory_text.split()
                if len(words) > 5:
                    memory_concepts.append(" ".join(words[:5]))
                else:
                    memory_concepts.append(recog.memory_text)
            
            # Extract concept from current conversation
            conversation_concept = conversation_text[:50] if len(conversation_text) > 50 else conversation_text
            
            # Use bisociation to connect conversation with recognized memory
            try:
                novel_idea = await novelty_engine.generate_novel_idea(
                    technique="bisociation",
                    concepts=[conversation_concept] + memory_concepts[:1]  # Use first memory concept
                )
                
                return {
                    "status": "success",
                    "integration_type": "bisociation",
                    "conversation_concept": conversation_concept,
                    "memory_concept": memory_concepts[0] if memory_concepts else None,
                    "novel_idea": novel_idea
                }
            except Exception as e:
                logger.error(f"Error integrating with novelty engine: {e}")
                return {"status": "error", "error": str(e)}
    
    # Helper methods
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple string similarity"""
        if not str1 or not str2:
            return 0.0
            
        # Convert to sets of words
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

# ============================================================================
# Tool Functions - All with RunContextWrapper as first parameter and Pydantic models
# ============================================================================

@function_tool
async def _blend_memory_with_context(
    ctx: RunContextWrapper[RecognitionMemoryContext],
    memory: MemoryData,
    conversation_context: List[MessageContext]
) -> BlendResult:
    """
    Create a lightweight "blend" of a memory and the current conversation context
    so the filter agent can present how they interrelate.
    """
    mem_text = memory.memory_text
    convo_text = " ".join(msg.text for msg in conversation_context)

    # Pick a sentence from memory that overlaps with context, or fallback
    overlap_words = set(mem_text.lower().split()) & set(convo_text.lower().split())
    snippet = mem_text[:100].strip()
    
    if overlap_words:
        # find first sentence containing any overlap word
        for sentence in mem_text.split("."):
            if any(w in sentence.lower() for w in overlap_words):
                snippet = sentence.strip()
                break

    # A very simple "blend score" based on proportion of shared words
    shared = len(overlap_words)
    total = len(set(mem_text.lower().split()) | set(convo_text.lower().split()))
    blend_score = round(shared / total, 2) if total > 0 else 0.0

    return BlendResult(
        blended_snippet=snippet,
        blend_score=blend_score
    )

@function_tool
async def _get_active_triggers(ctx: RunContextWrapper[RecognitionMemoryContext]) -> List[TriggerDict]:
    """
    Get the list of currently active triggers
    
    Returns:
        List of active triggers as dicts
    """
    triggers = []
    for trigger_id, trigger in ctx.context.active_triggers.items():
        triggers.append(TriggerDict(
            trigger_id=trigger.trigger_id,
            trigger_type=trigger.trigger_type,
            trigger_value=trigger.trigger_value,
            relevance_threshold=trigger.relevance_threshold,
            activation_strength=trigger.activation_strength,
            source=trigger.source
        ))
    return triggers

@function_tool
async def _get_recent_recognitions(ctx: RunContextWrapper[RecognitionMemoryContext]) -> List[RecognitionTimestamp]:
    """
    Get recently recognized memories
    
    Returns:
        List of recent memory recognitions with timestamps
    """
    recent = []
    for memory_id, timestamp in ctx.context.recent_recognitions:
        recent.append(RecognitionTimestamp(
            memory_id=memory_id,
            recognized_at=timestamp.isoformat()
        ))
    return recent

@function_tool
async def _update_context_history(
    ctx: RunContextWrapper[RecognitionMemoryContext],
    new_message: MessageContext
) -> int:
    """
    Update conversation context history with a new message
    
    Args:
        new_message: New message to add to context history
        
    Returns:
        Current size of context history
    """
    # Add new message to history
    ctx.context.recent_conversation.append(new_message)
    
    # Trim to max size
    if len(ctx.context.recent_conversation) > ctx.context.context_history_size:
        ctx.context.recent_conversation = ctx.context.recent_conversation[-ctx.context.context_history_size:]
    
    return len(ctx.context.recent_conversation)

@function_tool
async def _get_trigger_types(ctx: RunContextWrapper[RecognitionMemoryContext]) -> List[str]:
    """
    Get the list of valid trigger types
    
    Returns:
        List of trigger types
    """
    return ctx.context.trigger_types

@function_tool
async def _analyze_entities(
    ctx: RunContextWrapper[RecognitionMemoryContext],
    text: str
) -> List[EntityAnalysis]:
    """
    Analyze text for salient entities
    
    Args:
        text: Text to analyze
        
    Returns:
        List of detected entities with metadata
    """
    entities = []
    
    # Extract simple entities based on capitalization
    words = text.split()
    capitalized_words = [word for word in words if word and word[0].isupper()]
    
    for word in capitalized_words:
        # Clean up punctuation
        clean_word = word.strip(".,;:!?")
        if len(clean_word) > 1:  # Skip single letters
            entity_type = "person" if random.random() > 0.5 else "organization"
            entities.append(EntityAnalysis(
                entity=clean_word,
                type=entity_type,
                salience=round(random.uniform(0.6, 0.9), 2),
                position=text.find(clean_word)
            ))
    
    # Check for simple location patterns
    location_patterns = ["in ", "at ", "to ", "from "]
    for pattern in location_patterns:
        idx = text.lower().find(pattern)
        if idx >= 0:
            # Get the word after the pattern
            end_idx = text.find(" ", idx + len(pattern))
            if end_idx < 0:
                end_idx = len(text)
            location = text[idx + len(pattern):end_idx].strip(".,;:!?")
            if location and len(location) > 1 and location[0].isupper():
                entities.append(EntityAnalysis(
                    entity=location,
                    type="location",
                    salience=round(random.uniform(0.5, 0.8), 2),
                    position=idx + len(pattern)
                ))
    
    return entities

@function_tool
async def _analyze_emotions(
    ctx: RunContextWrapper[RecognitionMemoryContext],
    text: str
) -> List[EmotionAnalysis]:
    """
    Analyze text for emotional markers
    
    Args:
        text: Text to analyze
        
    Returns:
        List of detected emotions with metadata
    """
    emotion_keywords = {
        "happy": ["happy", "joy", "delighted", "pleased", "glad", "excited"],
        "sad": ["sad", "unhappy", "disappointed", "upset", "depressed", "miserable"],
        "angry": ["angry", "mad", "furious", "irritated", "annoyed", "frustrated"],
        "afraid": ["afraid", "scared", "frightened", "terrified", "anxious", "worried"],
        "surprised": ["surprised", "shocked", "amazed", "astonished", "stunned"],
        "disgusted": ["disgusted", "repulsed", "revolted", "appalled"]
    }
    
    detected_emotions = []
    
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            if keyword in text.lower():
                detected_emotions.append(EmotionAnalysis(
                    emotion=emotion,
                    trigger_word=keyword,
                    intensity=round(random.uniform(0.5, 0.9), 2),
                    position=text.lower().find(keyword)
                ))
    
    # If no emotions were detected, try to infer from patterns
    if not detected_emotions:
        if "!" in text:
            # Excitement or anger
            if any(word in text.lower() for word in ["wow", "amazing", "awesome", "cool"]):
                detected_emotions.append(EmotionAnalysis(
                    emotion="surprised",
                    trigger_pattern="exclamation",
                    intensity=0.7,
                    position=text.find("!")
                ))
            else:
                detected_emotions.append(EmotionAnalysis(
                    emotion="angry",
                    trigger_pattern="exclamation",
                    intensity=0.6,
                    position=text.find("!")
                ))
        
        if "?" in text:
            # Curiosity or confusion
            detected_emotions.append(EmotionAnalysis(
                emotion="curious",
                trigger_pattern="question",
                intensity=0.5,
                position=text.find("?")
            ))
    
    return detected_emotions

@function_tool
async def _analyze_topics(
    ctx: RunContextWrapper[RecognitionMemoryContext],
    text: str
) -> List[TopicAnalysis]:
    """
    Analyze text for key topics
    
    Args:
        text: Text to analyze
        
    Returns:
        List of detected topics with metadata
    """
    # Count word frequencies (ignoring common words)
    common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", 
                   "about", "that", "this", "these", "those", "it", "they", "we", "you", "i", "he", "she"}
    
    word_counts = {}
    words = text.lower().split()
    
    for word in words:
        # Clean word
        word = word.strip(".,;:!?\"'()[]{}").lower()
        if word and len(word) > 3 and word not in common_words:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1
    
    # Find top words
    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    topics = []
    for word, count in top_words:
        context_start = max(0, text.lower().find(word) - 10)
        context_end = min(len(text), text.lower().find(word) + len(word) + 10)
        
        topics.append(TopicAnalysis(
            topic=word,
            count=count,
            salience=min(1.0, 0.5 + (count / 10)),
            context=text[context_start:context_end]
        ))
    
    return topics

@function_tool
async def _format_trigger(
    ctx: RunContextWrapper[RecognitionMemoryContext],
    trigger_type: str,
    trigger_value: str,
    relevance_threshold: float = 0.6,
    activation_strength: float = 1.0,
    context_requirements: Optional[List[ContextRequirement]] = None
) -> ContextualTrigger:
    """
    Format a contextual trigger
    
    Args:
        trigger_type: Type of trigger
        trigger_value: Value of the trigger
        relevance_threshold: Minimum relevance for activation
        activation_strength: Strength of activation
        context_requirements: Additional context requirements
        
    Returns:
        Formatted ContextualTrigger
    """
    if trigger_type not in ctx.context.trigger_types:
        raise ValueError(f"Invalid trigger type: {trigger_type}")
    
    if relevance_threshold < 0 or relevance_threshold > 1:
        relevance_threshold = max(0, min(1, relevance_threshold))
    
    if activation_strength < 0 or activation_strength > 1:
        activation_strength = max(0, min(1, activation_strength))
    
    return ContextualTrigger(
        trigger_type=trigger_type,
        trigger_value=trigger_value,
        relevance_threshold=relevance_threshold,
        activation_strength=activation_strength,
        context_requirements=context_requirements or [],
        source="agent_generated"
    )

@function_tool
async def _query_memory(
    ctx: RunContextWrapper[RecognitionMemoryContext],
    trigger: TriggerDict,
    memory_types: Optional[List[str]] = None,
    limit: int = 3
) -> List[MemoryData]:
    """
    Query memory system based on a contextual trigger
    
    Args:
        trigger: Contextual trigger to use
        memory_types: Types of memories to retrieve
        limit: Maximum number of memories to return
        
    Returns:
        List of matching memories
    """
    # Ensure we have a memory core
    if not ctx.context.memory_core:
        return []
    
    try:
        # Construct query based on trigger
        query = trigger.get("trigger_value", "")
        
        # Default memory types if not specified
        if not memory_types:
            memory_types = ["observation", "experience", "reflection"]
        
        # Call memory_core.retrieve_memories
        memories = await ctx.context.memory_core.retrieve_memories(
            query=query,
            memory_types=memory_types,
            limit=limit,
            min_significance=3,
            entities=[trigger.get("trigger_value")] if trigger.get("trigger_type") == "entity" else None
        )
        
        # Convert to MemoryData objects
        memory_data_list = []
        for memory in memories:
            # Convert metadata dict to list of ContextRequirement
            metadata_items = []
            if "metadata" in memory and memory["metadata"]:
                for key, value in memory["metadata"].items():
                    metadata_items.append(ContextRequirement(key=key, value=str(value)))
            
            memory_data = MemoryData(
                id=memory.get("id", ""),
                memory_text=memory.get("memory_text", ""),
                memory_type=memory.get("memory_type", ""),
                relevance=memory.get("relevance", 0.5),
                significance=memory.get("significance", 5),
                confidence=memory.get("confidence", 0.5),
                activation_trigger=trigger,
                metadata_items=metadata_items
            )
            memory_data_list.append(memory_data)
        
        return memory_data_list
        
    except Exception as e:
        logger.error(f"Error querying memory: {e}")
        return []

@function_tool
async def _combine_query_results(
    ctx: RunContextWrapper[RecognitionMemoryContext],
    memory_sets: List[List[MemoryData]],
    max_results: int = 10
) -> List[MemoryData]:
    """
    Combine and deduplicate memory query results
    
    Args:
        memory_sets: Lists of memories from different queries
        max_results: Maximum number of results to return
        
    Returns:
        Combined and sorted list of memories
    """
    # Combine all memories
    all_memories = []
    memory_ids = set()
    
    for memory_set in memory_sets:
        for memory in memory_set:
            if memory.id and memory.id not in memory_ids:
                all_memories.append(memory)
                memory_ids.add(memory.id)
    
    # Sort by relevance if available
    all_memories.sort(key=lambda x: x.relevance, reverse=True)
    
    return all_memories[:max_results]

@function_tool
async def _calculate_contextual_relevance(
    ctx: RunContextWrapper[RecognitionMemoryContext],
    memory: MemoryData,
    conversation_context: List[MessageContext]
) -> float:
    """
    Calculate contextual relevance of a memory to current conversation
    
    Args:
        memory: Memory to evaluate
        conversation_context: Recent conversation context
        
    Returns:
        Contextual relevance score
    """
    # Start with the memory's relevance score if available
    base_relevance = memory.relevance
    
    # Get memory text and type
    memory_text = memory.memory_text
    memory_type = memory.memory_type
    
    # Extract conversation text
    conversation_text = " ".join(msg.text for msg in conversation_context)
    
    # Simple text similarity (word overlap)
    memory_words = set(memory_text.lower().split())
    conversation_words = set(conversation_text.lower().split())
    if memory_words and conversation_words:
        overlap = len(memory_words.intersection(conversation_words))
        overlap_score = min(1.0, overlap / 5)  # Cap at 1.0
    else:
        overlap_score = 0.0
    
    # Apply type-specific boosts
    type_boost = 0.0
    if memory_type == "experience":
        type_boost = 0.1  # Boost experiences
    elif memory_type == "reflection":
        type_boost = 0.05  # Small boost for reflections
    
    # Recency boost
    recency_boost = 0.0
    if memory.metadata_items:
        # Look for timestamp in metadata items
        timestamp_item = None
        for item in memory.metadata_items:
            if item.key == "timestamp":
                timestamp_item = item
                break
        
        if timestamp_item:
            try:
                timestamp = datetime.datetime.fromisoformat(
                    timestamp_item.value.replace("Z", "+00:00")
                )
                days_old = (datetime.datetime.now() - timestamp).days
                if days_old < 7:
                    recency_boost = 0.1  # Boost very recent memories
                elif days_old < 30:
                    recency_boost = 0.05  # Small boost for recent memories
            except Exception:
                pass
    
    # Calculate final relevance
    contextual_relevance = min(1.0, base_relevance + overlap_score + type_boost + recency_boost)
    
    return contextual_relevance

@function_tool
async def _check_recognition_history(
    ctx: RunContextWrapper[RecognitionMemoryContext],
    memory_id: str
) -> RecognitionHistory:
    """
    Check if a memory was recently recognized
    
    Args:
        memory_id: ID of memory to check
        
    Returns:
        Information about recent recognition if any
    """
    now = datetime.datetime.now()
    
    for past_id, timestamp in ctx.context.recent_recognitions:
        if past_id == memory_id:
            seconds_ago = (now - timestamp).total_seconds()
            cooldown = ctx.context.recognition_cooldown
            
            return RecognitionHistory(
                was_recognized=True,
                seconds_ago=seconds_ago,
                cooldown_period=cooldown,
                cooldown_remaining=max(0, cooldown - seconds_ago),
                in_cooldown=seconds_ago < cooldown
            )
    
    return RecognitionHistory(
        was_recognized=False,
        cooldown_period=ctx.context.recognition_cooldown,
        cooldown_remaining=0,
        in_cooldown=False
    )

@function_tool
async def _analyze_context_state(
    ctx: RunContextWrapper[RecognitionMemoryContext]
) -> ContextState:
    """
    Analyze current context state for recognition parameters
    
    Returns:
        Context state analysis
    """
    # Get conversation context
    conversation = ctx.context.recent_conversation
    
    # Default parameters
    recognition_sensitivity = 0.7
    max_recognitions = 3
    
    # Adjust based on conversation state
    if len(conversation) > 0:
        # Check for questions (may indicate higher interest in memories)
        last_message = conversation[-1].text
        if "?" in last_message:
            recognition_sensitivity += 0.1
            max_recognitions += 1
            
        # Check for emotional content (may trigger more memories)
        if any(word in last_message.lower() for word in ["feel", "happy", "sad", "angry", "excited"]):
            recognition_sensitivity += 0.1
            
        # Check for memory-related terms
        if any(word in last_message.lower() for word in ["remember", "recall", "memory", "forget", "remembered"]):
            recognition_sensitivity += 0.2
            max_recognitions += 1
    
    # Update context parameters
    ctx.context.recognition_sensitivity = min(1.0, recognition_sensitivity)
    ctx.context.max_recognitions_per_turn = min(5, max_recognitions)
    
    return ContextState(
        recognition_sensitivity=ctx.context.recognition_sensitivity,
        max_recognitions=ctx.context.max_recognitions_per_turn,
        context_complexity=len(conversation),
        active_triggers_count=len(ctx.context.active_triggers)
    )

@function_tool
async def _analyze_narrative_elements(
    ctx: RunContextWrapper[RecognitionMemoryContext],
    text: str
) -> List[NarrativeElement]:
    """
    Analyze text for narrative elements that might trigger recognition
    
    Args:
        text: Text to analyze
        
    Returns:
        List of narrative elements
    """
    narrative_elements = []
    
    # Check for narrative markers
    narrative_markers = [
        "then", "after that", "before", "while", "during",
        "first", "second", "finally", "lastly", "next"
    ]
    
    for marker in narrative_markers:
        marker_position = text.lower().find(f" {marker} ")
        if marker_position >= 0:
            # Get surrounding context
            start = max(0, marker_position - 20)
            end = min(len(text), marker_position + 20)
            context = text[start:end]
            
            narrative_elements.append(NarrativeElement(
                element_type="narrative_marker",
                marker=marker,
                context=context,
                salience=0.7,
                position=marker_position
            ))
    
    # Check for event descriptions
    event_indicators = ["happened", "occurred", "took place", "experienced", "went to"]
    
    for indicator in event_indicators:
        indicator_position = text.lower().find(indicator)
        if indicator_position >= 0:
            # Get surrounding context
            start = max(0, indicator_position - 25)
            end = min(len(text), indicator_position + 25)
            context = text[start:end]
            
            narrative_elements.append(NarrativeElement(
                element_type="event_description",
                indicator=indicator,
                context=context,
                salience=0.8,
                position=indicator_position
            ))
    
    return narrative_elements

@function_tool
async def _leverage_context_awareness(
    ctx: RunContextWrapper[RecognitionMemoryContext],
    text: str
) -> ContextAwarenessResult:
    """
    Leverage context awareness system for enhanced salience detection
    
    Args:
        text: Text to analyze
        
    Returns:
        Context awareness results
    """
    if not ctx.context.context_awareness:
        return ContextAwarenessResult(
            entities=[],
            topics=[],
            emotions=[]
        )
    
    try:
        # Use context awareness system
        awareness_result = await ctx.context.context_awareness.analyze_content(text)
        
        # Extract elements
        entities = []
        for entity in awareness_result.get("entities", []):
            entities.append(EntityAnalysis(
                entity=entity.get("text", ""),
                type=entity.get("type", "unknown"),
                salience=entity.get("salience", 0.5),
                position=0  # Position not provided by context awareness
            ))
            
        topics = []
        for topic in awareness_result.get("topics", []):
            topics.append(TopicAnalysis(
                topic=topic.get("name", ""),
                count=1,  # Count not provided
                salience=topic.get("confidence", 0.5),
                context=""  # Context not provided
            ))
            
        emotions = []
        for emotion in awareness_result.get("emotions", []):
            emotions.append(EmotionAnalysis(
                emotion=emotion.get("name", ""),
                intensity=emotion.get("score", 0.5),
                position=0  # Position not provided
            ))
            
        return ContextAwarenessResult(
            entities=entities,
            topics=topics,
            emotions=emotions
        )
        
    except Exception as e:
        logger.error(f"Error leveraging context awareness: {e}")
        return ContextAwarenessResult(
            entities=[],
            topics=[],
            emotions=[]
        )

@function_tool
async def _query_with_prioritization(
    ctx: RunContextWrapper[RecognitionMemoryContext],
    query: str,
    trigger: Optional[TriggerDict] = None,
    memory_types: Optional[List[str]] = None,
    prioritization: Optional[Dict[str, float]] = None,
    limit: int = 5
) -> List[MemoryData]:
    """
    Query memory system with type prioritization
    
    Args:
        query: Search query
        trigger: Trigger information
        memory_types: Types of memories to retrieve
        prioritization: Priority weights for different memory types
        limit: Maximum number of memories to return
        
    Returns:
        List of matching memories with prioritization applied
    """
    # Ensure memory core exists
    if not ctx.context.memory_core:
        return []
        
    # Default memory types
    if not memory_types:
        memory_types = ["experience", "reflection", "abstraction", "observation"]
        
    # Default prioritization
    if not prioritization:
        prioritization = {
            "experience": 0.4,
            "reflection": 0.3,
            "abstraction": 0.2,
            "observation": 0.1
        }
        
    try:
        # Use memory_core's prioritized retrieval
        if hasattr(ctx.context.memory_core, "retrieve_memories_with_prioritization"):
            memories = await ctx.context.memory_core.retrieve_memories_with_prioritization(
                query=query,
                memory_types=memory_types,
                prioritization=prioritization,
                limit=limit
            )
        else:
            # Fall back to standard retrieval
            memories = await ctx.context.memory_core.retrieve_memories(
                query=query,
                memory_types=memory_types,
                limit=limit
            )
            
        # Convert to MemoryData objects
        memory_data_list = []
        for memory in memories:
            # Convert metadata dict to list of ContextRequirement
            metadata_items = []
            if "metadata" in memory and memory["metadata"]:
                for key, value in memory["metadata"].items():
                    metadata_items.append(ContextRequirement(key=key, value=str(value)))
            
            memory_data = MemoryData(
                id=memory.get("id", ""),
                memory_text=memory.get("memory_text", ""),
                memory_type=memory.get("memory_type", ""),
                relevance=memory.get("relevance", 0.5),
                significance=memory.get("significance", 5),
                confidence=memory.get("confidence", 0.5),
                activation_trigger=trigger,
                metadata_items=metadata_items
            )
            memory_data_list.append(memory_data)
                
        return memory_data_list
        
    except Exception as e:
        logger.error(f"Error in prioritized memory query: {e}")
        return []

@function_tool
async def _query_memories_parallel(
    ctx: RunContextWrapper[RecognitionMemoryContext],
    triggers: List[TriggerDict],
    memory_types: Optional[List[str]] = None,
    limit_per_trigger: int = 3
) -> ParallelQueryResult:
    """
    Query memory system in parallel for multiple triggers
    
    Args:
        triggers: List of triggers to query
        memory_types: Types of memories to retrieve
        limit_per_trigger: Maximum number of memories per trigger
        
    Returns:
        Dictionary of trigger_id -> memories
    """
    # Ensure memory core exists
    if not ctx.context.memory_core:
        return ParallelQueryResult(results={})
        
    # Default memory types
    if not memory_types:
        memory_types = ["experience", "reflection", "abstraction", "observation"]
        
    try:
        # Create tasks for each trigger
        trigger_tasks = {}
        for trigger in triggers:
            trigger_id = trigger.get("trigger_id", str(uuid.uuid4()))
            trigger_value = trigger.get("trigger_value", "")
            
            # Create retrieval task
            trigger_tasks[trigger_id] = ctx.context.memory_core.retrieve_memories(
                query=trigger_value,
                memory_types=memory_types,
                limit=limit_per_trigger
            )
        
        # Wait for all tasks to complete
        result_items = []
        for trigger_id, task in trigger_tasks.items():
            try:
                memories = await task
                # Convert to MemoryData objects
                memory_data_list = []
                for memory in memories:
                    # Convert metadata dict to list of ContextRequirement
                    metadata_items = []
                    if "metadata" in memory and memory["metadata"]:
                        for key, value in memory["metadata"].items():
                            metadata_items.append(ContextRequirement(key=key, value=str(value)))
                    
                    memory_data = MemoryData(
                        id=memory.get("id", ""),
                        memory_text=memory.get("memory_text", ""),
                        memory_type=memory.get("memory_type", ""),
                        relevance=memory.get("relevance", 0.5),
                        significance=memory.get("significance", 5),
                        confidence=memory.get("confidence", 0.5),
                        metadata_items=metadata_items
                    )
                    memory_data_list.append(memory_data)
                
                result_items.append(ParallelQueryResultItem(
                    trigger_id=trigger_id,
                    memories=memory_data_list
                ))
            except Exception as e:
                logger.error(f"Error in parallel query for trigger {trigger_id}: {e}")
                result_items.append(ParallelQueryResultItem(
                    trigger_id=trigger_id,
                    memories=[]
                ))
                
        return ParallelQueryResult(results=result_items)
        
    except Exception as e:
        logger.error(f"Error in parallel memory queries: {e}")
        return ParallelQueryResult(results={})

@function_tool
async def _track_query_performance(
    ctx: RunContextWrapper[RecognitionMemoryContext],
    trigger_id: str,
    memories_found: int,
    avg_relevance: float
) -> TriggerPerformance:
    """
    Track query performance for trigger optimization
    
    Args:
        trigger_id: ID of the trigger
        memories_found: Number of memories found
        avg_relevance: Average relevance of retrieved memories
        
    Returns:
        Performance metrics
    """
    # Find existing performance entry
    performance = None
    for perf in ctx.context.trigger_performance:
        if perf.trigger_id == trigger_id:
            performance = perf
            break
    
    # Create new entry if not found
    if performance is None:
        performance = TriggerPerformance(trigger_id=trigger_id)
        ctx.context.trigger_performance.append(performance)
        
    # Update performance metrics
    performance.query_count += 1
    performance.total_memories += memories_found
    
    # Calculate new average relevance
    old_avg = performance.avg_relevance
    old_count = performance.query_count - 1  # Subtract the one we just added
    
    if old_count > 0:
        performance.avg_relevance = (old_avg * old_count + avg_relevance) / (old_count + 1)
    else:
        performance.avg_relevance = avg_relevance
        
    performance.last_updated = datetime.datetime.now().isoformat()
    
    # Check if trigger needs optimization
    needs_optimization = False
    
    # If too many queries with no results
    if performance.query_count >= 3 and performance.total_memories == 0:
        needs_optimization = True
        
    # If consistently low relevance
    if performance.query_count >= 5 and performance.avg_relevance < 0.3:
        needs_optimization = True
        
    performance.needs_optimization = needs_optimization
    
    return performance

@function_tool
async def _assess_trigger_quality(
    ctx: RunContextWrapper[RecognitionMemoryContext],
    trigger_type: str,
    trigger_value: str
) -> TriggerQuality:
    """
    Assess quality of a potential trigger
    
    Args:
        trigger_type: Type of trigger
        trigger_value: Value of the trigger
        
    Returns:
        Quality assessment
    """
    # Check if trigger is too generic
    generic_terms = {
        "entity": ["person", "place", "thing", "someone", "something"],
        "concept": ["idea", "thought", "concept", "notion"],
        "emotion": ["feeling", "emotion", "mood"],
        "event": ["happening", "occurrence", "incident"],
        "topic": ["subject", "matter", "issue"]
    }
    
    is_generic = trigger_value.lower() in generic_terms.get(trigger_type, [])
    
    # Check if similar to existing triggers
    similar_triggers = []
    for existing_id, existing in ctx.context.active_triggers.items():
        if existing.trigger_type == trigger_type:
            # Simple string similarity
            similarity = _calculate_string_similarity(
                existing.trigger_value.lower(),
                trigger_value.lower()
            )
            
            if similarity > 0.7:  # High similarity
                similar_triggers.append(SimilarTrigger(
                    trigger_id=existing_id,
                    trigger_value=existing.trigger_value,
                    similarity=similarity
                ))
    
    # Check specificity
    words = trigger_value.split()
    specificity = min(1.0, len(words) / 3)  # More words = more specific
    
    # Overall quality score
    quality = 0.5
    
    if is_generic:
        quality -= 0.3
    
    if len(similar_triggers) > 0:
        quality -= 0.2
        
    quality += specificity * 0.3
    
    # Bound score
    quality = max(0.1, min(1.0, quality))
    
    return TriggerQuality(
        quality_score=quality,
        is_generic=is_generic,
        similar_triggers=similar_triggers,
        specificity=specificity,
        recommended_threshold=max(0.5, 0.7 - (quality * 0.2))  # Adjust threshold based on quality
    )

@function_tool
async def _assess_memory_causality(
    ctx: RunContextWrapper[RecognitionMemoryContext],
    memory: MemoryData,
    conversation_context: List[MessageContext]
) -> CausalityAssessment:
    """
    Assess causal connections between a recognized memory and the current conversation.
    Returns a simple causality strength score and any trigger keywords found.
    """
    # Combine memory text and recent conversation
    mem_text = memory.memory_text
    convo_text = " ".join(msg.text for msg in conversation_context)
    full_text = f"{mem_text} {convo_text}".lower()

    # Look for basic causal markers
    causality_markers = ["because", "due to", "as a result", "therefore", "so that"]
    found = [kw for kw in causality_markers if kw in full_text]

    # Simple strength: proportion of markers found capped at 1.0
    strength = min(1.0, len(found) / len(causality_markers))

    return CausalityAssessment(
        causality_strength=strength,
        causality_markers_found=found
    )

@function_tool
async def _calibrate_trigger_parameters(
    ctx: RunContextWrapper[RecognitionMemoryContext],
    trigger_type: str,
    trigger_value: str,
    quality_assessment: TriggerQuality
) -> TriggerParameters:
    """
    Calibrate trigger parameters based on quality assessment
    
    Args:
        trigger_type: Type of trigger
        trigger_value: Value of the trigger
        quality_assessment: Quality assessment data
        
    Returns:
        Calibrated parameters
    """
    quality = quality_assessment.quality_score
    is_generic = quality_assessment.is_generic
    specificity = quality_assessment.specificity
    
    # Base parameters
    relevance_threshold = 0.6
    activation_strength = 0.8
    
    # Adjust threshold based on quality
    if quality < 0.3:
        # Low quality triggers need higher threshold to avoid false positives
        relevance_threshold = 0.8
    elif quality < 0.6:
        relevance_threshold = 0.7
    else:
        # High quality triggers can use lower threshold
        relevance_threshold = 0.6
        
    # Adjust strength based on trigger type and specificity
    if trigger_type == "entity":
        # Entities typically have higher activation
        activation_strength = 0.9
    elif trigger_type == "emotion":
        # Emotions can be powerful triggers
        activation_strength = 0.85
    else:
        # Base strength on specificity
        activation_strength = 0.7 + (specificity * 0.2)
        
    # Adjustments for generic triggers
    if is_generic:
        activation_strength *= 0.8  # Reduce strength for generic triggers
        
    return TriggerParameters(
        relevance_threshold=relevance_threshold,
        activation_strength=activation_strength
    )

@function_tool
async def _assess_conversational_impact(
    ctx: RunContextWrapper[RecognitionMemoryContext],
    memory: MemoryData,
    conversation_context: List[MessageContext]
) -> ConversationalImpact:
    """
    Assess potential conversational impact of a recognized memory
    
    Args:
        memory: Memory to evaluate
        conversation_context: Recent conversation context
        
    Returns:
        Impact assessment
    """
    # Get memory properties
    memory_text = memory.memory_text
    memory_type = memory.memory_type
    memory_significance = memory.significance
    
    # Extract last message
    last_message = ""
    if conversation_context:
        last_message = conversation_context[-1].text
        
    # Default impact scores
    novelty = 0.5
    relevance = 0.5
    coherence = 0.5
    
    # Assess novelty
    words_last_message = set(last_message.lower().split())
    words_memory = set(memory_text.lower().split())
    
    # New information rate
    if words_last_message:
        unique_to_memory = words_memory - words_last_message
        novelty = min(1.0, len(unique_to_memory) / len(words_memory)) if words_memory else 0.5
        
    # Assess coherence
    if words_last_message and words_memory:
        overlap = words_last_message.intersection(words_memory)
        coherence = min(1.0, len(overlap) / min(len(words_last_message), len(words_memory)))
        
    # Assess relevance based on memory type and significance
    if memory_type == "experience":
        relevance += 0.1  # Slight boost for experiences
    
    # Significance boost
    relevance += (memory_significance / 10) * 0.2
    
    # Calculate overall impact
    impact_score = (novelty * 0.4) + (relevance * 0.4) + (coherence * 0.2)
    
    # Assess specific impact types
    elaboration = coherence > 0.3 and novelty > 0.3
    contrast = coherence < 0.3 and novelty > 0.7
    reinforcement = coherence > 0.7 and novelty < 0.3
    
    return ConversationalImpact(
        impact_score=impact_score,
        novelty=novelty,
        relevance=relevance,
        coherence=coherence,
        impact_type="elaboration" if elaboration else "contrast" if contrast else "reinforcement" if reinforcement else "mixed"
    )

# Helper function for string similarity
def _calculate_string_similarity(str1: str, str2: str) -> float:
    """Calculate simple string similarity"""
    if not str1 or not str2:
        return 0.0
        
    # Convert to sets of words
    words1 = set(str1.lower().split())
    words2 = set(str2.lower().split())
    
    if not words1 or not words2:
        return 0.0
        
    # Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

# Handoff preparation functions
async def prepare_for_salience_detection(ctx: RunContextWrapper[RecognitionMemoryContext]):
    """Prepare context for salience detection handoff"""
    logger.debug("Preparing context for Salience Detector Agent Handoff.")
    recent_convo = ctx.context.recent_conversation
    logger.debug(f"Standalone: Recent conversation length: {len(recent_convo)}")
    return None

async def prepare_for_trigger_extraction(ctx: RunContextWrapper[RecognitionMemoryContext]):
    """Prepare context for trigger extraction handoff"""
    logger.debug("Preparing context for Trigger Extraction Agent Handoff.")
    return None

async def prepare_for_memory_query(ctx: RunContextWrapper[RecognitionMemoryContext]):
    """Prepare context for memory query handoff"""
    logger.debug("Preparing context for Memory Query Agent Handoff.")
    max_memories_per_trigger = 3
    if len(ctx.context.active_triggers) > 5:
        max_memories_per_trigger = 2
    ctx.context.max_memories_per_trigger = max_memories_per_trigger
    logger.debug(f"Standalone: Set max_memories_per_trigger to: {ctx.context.max_memories_per_trigger}")
    return None

async def prepare_for_relevance_filtering(ctx: RunContextWrapper[RecognitionMemoryContext]):
    """Prepare context for relevance filtering handoff"""
    logger.debug("Preparing context for Relevance Filter Agent Handoff.")
    ctx.context.max_memories_to_return = ctx.context.max_recognitions_per_turn
    logger.debug(f"Standalone: Set max_memories_to_return to: {ctx.context.max_memories_to_return}")
    return None
