# nyx/core/recognition_memory.py

import logging
import asyncio
import datetime
import random
import uuid
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from pydantic import BaseModel, Field

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

# Pydantic models for recognition memory
class ContextualTrigger(BaseModel):
    """Schema for a contextual trigger that activates recognition memory"""
    trigger_id: str = Field(default_factory=lambda: f"trigger_{uuid.uuid4().hex[:8]}")
    trigger_type: str  # e.g., "entity", "concept", "event", "emotion", "pattern"
    trigger_value: str
    relevance_threshold: float = 0.6
    activation_strength: float = 1.0
    context_requirements: Dict[str, Any] = Field(default_factory=dict)
    source: str = "system"  # Where this trigger came from

class RecognitionResult(BaseModel):
    """Schema for a recognition memory result"""
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
    cue_type: str  # e.g., "entity", "keyword", "phrase", "emotion", "topic"
    cue_value: str
    salience: float
    source_text: str
    extracted_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

class SalienceAnalysisOutput(BaseModel):
    """Output schema for conversation salience analysis"""
    salient_entities: List[Dict[str, Any]]
    salient_concepts: List[Dict[str, Any]]
    emotional_markers: List[Dict[str, Any]]
    key_topics: List[Dict[str, Any]]
    notable_patterns: List[Dict[str, Any]]

class TriggerExtractionOutput(BaseModel):
    """Output schema for trigger extraction"""
    contextual_triggers: List[ContextualTrigger]
    extraction_confidence: float
    trigger_count: int
    potential_triggers: List[Dict[str, Any]]

class RecognitionFilterOutput(BaseModel):
    """Output schema for recognition filtering"""
    selected_memories: List[Dict[str, Any]]
    selection_rationale: str
    confidence_levels: Dict[str, float]
    relevance_hierarchy: List[Dict[str, Any]]

class RecognitionMemoryContext:
    """Context object for recognition memory operations"""
    
    def __init__(self, memory_core=None, context_awareness=None):
        self.memory_core = memory_core
        self.context_awareness = context_awareness
        
        # Recent context tracking
        self.recent_conversation: List[Dict[str, Any]] = []
        self.current_context_distribution = None
        
        # Active triggers registry
        self.active_triggers: Dict[str, ContextualTrigger] = {}
        
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
        
        # Trigger types
        self.trigger_types = [
            "entity", "concept", "keyword", "phrase", 
            "emotion", "topic", "event", "pattern"
        ]
        
        # Default triggers (can be extended)
        self.default_triggers: List[ContextualTrigger] = []

class RecognitionMemorySystem:
    """
    System that automatically retrieves contextually relevant memories
    based on recognition triggers rather than explicit queries.
    """
    
    def __init__(self, memory_core=None, context_awareness=None):
        """Initialize the recognition memory system with required components"""
        # Create context
        self.context = RecognitionMemoryContext(
            memory_core=memory_core,
            context_awareness=context_awareness
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
            self.context.active_triggers[trigger.trigger_id] = trigger
    
    def _create_recognition_agent(self) -> Agent:
        """Create the main orchestrator agent for recognition memory"""
        return Agent[RecognitionMemoryContext](
            name="Recognition Memory Orchestrator",
            instructions="""You are the recognition memory orchestration system for the Nyx AI.
            
            Your role is to identify memories that should be automatically recognized based
            on the current conversation context without explicit retrieval requests. You
            coordinate specialized agents to extract contextual triggers, query memory,
            and filter relevant recognitions.
            
            Recognition memory works differently from free recall:
            - In free recall, memories are explicitly searched for with a query
            - In recognition memory, relevant memories are automatically surfaced when
              triggered by contextual cues
            
            Your task is to detect salient elements in the conversation that could trigger
            recognition, find relevant memories, and filter them for quality and relevance.
            The goal is to surface memories that add value to the conversation at just the
            right time.
            """,
            tools=[
                function_tool(self._get_active_triggers),
                function_tool(self._get_recent_recognitions),
                function_tool(self._update_context_history)
            ],
            handoffs=[
                handoff(self.salience_detector_agent,
                      tool_name_override="detect_salient_elements",
                      tool_description_override="Detect salient elements in conversation"),
                handoff(self.trigger_extraction_agent,
                      tool_name_override="extract_contextual_triggers",
                      tool_description_override="Extract contextual triggers from salient elements"),
                handoff(self.memory_query_agent,
                      tool_name_override="query_based_on_triggers",
                      tool_description_override="Query memory system based on extracted triggers"),
                handoff(self.relevance_filter_agent,
                      tool_name_override="filter_recognition_results",
                      tool_description_override="Filter recognition results for relevance")
            ],
            output_type=List[RecognitionResult],
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.4)
        )
    
    def _create_salience_detector_agent(self) -> Agent:
        """Create specialized agent for detecting salient elements in conversation"""
        return Agent[RecognitionMemoryContext](
            name="Salience Detector",
            instructions="""You are specialized in detecting salient elements in conversation
            that could trigger recognition memory.
            
            Your task is to analyze conversation text to identify:
            1. Salient entities (people, places, organizations, objects)
            2. Salient concepts (abstract ideas, themes, topics)
            3. Emotional markers (expressed or implied emotions)
            4. Key topics (main subjects of discussion)
            5. Notable patterns (recurring themes, linguistic patterns)
            
            Focus on elements with high salience - those that stand out and could
            serve as effective memory triggers. Consider both explicit mentions
            and implicit references.
            """,
            tools=[
                function_tool(self._analyze_entities),
                function_tool(self._analyze_emotions),
                function_tool(self._analyze_topics)
            ],
            output_type=SalienceAnalysisOutput,
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.3)
        )
    
    def _create_trigger_extraction_agent(self) -> Agent:
        """Create specialized agent for extracting contextual triggers"""
        return Agent[RecognitionMemoryContext](
            name="Trigger Extraction Agent",
            instructions="""You are specialized in extracting contextual triggers from
            salient elements identified in conversation.
            
            Your task is to:
            1. Convert salient elements into formal recognition triggers
            2. Assign appropriate relevance thresholds and activation strengths
            3. Determine any specific context requirements
            4. Ensure triggers are specific enough to be effective
            
            Focus on creating triggers that will effectively activate recognition memory
            without being too broad (causing too many activations) or too narrow
            (failing to activate when appropriate).
            """,
            tools=[
                function_tool(self._get_trigger_types),
                function_tool(self._format_trigger)
            ],
            output_type=TriggerExtractionOutput,
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.4)
        )
    
    def _create_memory_query_agent(self) -> Agent:
        """Create specialized agent for querying memory based on triggers"""
        return Agent[RecognitionMemoryContext](
            name="Memory Query Agent",
            instructions="""You are specialized in querying the memory system based on
            contextual triggers.
            
            Your task is to:
            1. Formulate effective queries for each trigger
            2. Execute memory searches with appropriate parameters
            3. Combine results from multiple triggers when appropriate
            4. Track query performance for trigger refinement
            
            Focus on finding memories that are genuinely relevant to the current
            conversation context. Balance breadth (finding diverse memories) with
            precision (finding highly relevant memories).
            """,
            tools=[
                function_tool(self._query_memory),
                function_tool(self._combine_query_results)
            ],
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.3)
        )
    
    def _create_relevance_filter_agent(self) -> Agent:
        """Create specialized agent for filtering recognition results"""
        return Agent[RecognitionMemoryContext](
            name="Relevance Filter Agent",
            instructions="""You are specialized in filtering recognition memory results
            for relevance, quality, and appropriateness.
            
            Your task is to:
            1. Evaluate each candidate memory for relevance to current context
            2. Consider conversational appropriateness and timing
            3. Assess potential value-add to the conversation
            4. Filter to prevent repetitive or redundant recognitions
            5. Ensure diversity of memory types when appropriate
            
            Focus on selecting memories that will enhance the conversation in
            meaningful ways. Balance relevance with novelty and value.
            """,
            tools=[
                function_tool(self._calculate_contextual_relevance),
                function_tool(self._check_recognition_history)
            ],
            output_type=RecognitionFilterOutput,
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.3)
        )
    
    # Tool functions for the various agents
    
    @function_tool
    async def _get_active_triggers(self, ctx: RunContextWrapper[RecognitionMemoryContext]) -> Dict[str, ContextualTrigger]:
        """
        Get the dictionary of currently active triggers
        
        Returns:
            Dictionary of trigger_id -> ContextualTrigger
        """
        return ctx.context.active_triggers
    
    @function_tool
    async def _get_recent_recognitions(self, ctx: RunContextWrapper[RecognitionMemoryContext]) -> List[Dict[str, Any]]:
        """
        Get recently recognized memories
        
        Returns:
            List of recent memory recognitions with timestamps
        """
        recent = []
        for memory_id, timestamp in ctx.context.recent_recognitions:
            recent.append({
                "memory_id": memory_id,
                "recognized_at": timestamp.isoformat()
            })
        return recent
    
    @function_tool
    async def _update_context_history(
        self,
        ctx: RunContextWrapper[RecognitionMemoryContext],
        new_message: Dict[str, Any]
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
    async def _get_trigger_types(self, ctx: RunContextWrapper[RecognitionMemoryContext]) -> List[str]:
        """
        Get the list of valid trigger types
        
        Returns:
            List of trigger types
        """
        return ctx.context.trigger_types
    
    @function_tool
    async def _analyze_entities(
        self,
        ctx: RunContextWrapper[RecognitionMemoryContext],
        text: str
    ) -> List[Dict[str, Any]]:
        """
        Analyze text for salient entities
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected entities with metadata
        """
        # In a full implementation, this might use a named entity recognition system
        # For now, we'll return a simplified version with mock entities
        
        entities = []
        
        # Extract simple entities based on capitalization
        words = text.split()
        capitalized_words = [word for word in words if word and word[0].isupper()]
        
        for word in capitalized_words:
            # Clean up punctuation
            clean_word = word.strip(".,;:!?")
            if len(clean_word) > 1:  # Skip single letters
                entity_type = "person" if random.random() > 0.5 else "organization"
                entities.append({
                    "entity": clean_word,
                    "type": entity_type,
                    "salience": round(random.uniform(0.6, 0.9), 2),
                    "position": text.find(clean_word)
                })
        
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
                    entities.append({
                        "entity": location,
                        "type": "location",
                        "salience": round(random.uniform(0.5, 0.8), 2),
                        "position": idx + len(pattern)
                    })
        
        return entities
    
    @function_tool
    async def _analyze_emotions(
        self,
        ctx: RunContextWrapper[RecognitionMemoryContext],
        text: str
    ) -> List[Dict[str, Any]]:
        """
        Analyze text for emotional markers
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected emotions with metadata
        """
        # In a full implementation, this might use a sentiment analysis system
        # For now, we'll use a simple keyword approach
        
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
                    detected_emotions.append({
                        "emotion": emotion,
                        "trigger_word": keyword,
                        "intensity": round(random.uniform(0.5, 0.9), 2),
                        "position": text.lower().find(keyword)
                    })
        
        # If no emotions were detected, try to infer from patterns
        if not detected_emotions:
            if "!" in text:
                # Excitement or anger
                if any(word in text.lower() for word in ["wow", "amazing", "awesome", "cool"]):
                    detected_emotions.append({
                        "emotion": "surprised",
                        "trigger_pattern": "exclamation",
                        "intensity": 0.7,
                        "position": text.find("!")
                    })
                else:
                    detected_emotions.append({
                        "emotion": "angry",
                        "trigger_pattern": "exclamation",
                        "intensity": 0.6,
                        "position": text.find("!")
                    })
            
            if "?" in text:
                # Curiosity or confusion
                detected_emotions.append({
                    "emotion": "curious",
                    "trigger_pattern": "question",
                    "intensity": 0.5,
                    "position": text.find("?")
                })
        
        return detected_emotions
    
    @function_tool
    async def _analyze_topics(
        self,
        ctx: RunContextWrapper[RecognitionMemoryContext],
        text: str
    ) -> List[Dict[str, Any]]:
        """
        Analyze text for key topics
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected topics with metadata
        """
        # In a full implementation, this would use topic modeling
        # For now, we'll use a simple approach
        
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
            topics.append({
                "topic": word,
                "count": count,
                "salience": min(1.0, 0.5 + (count / 10)),
                "context": text[max(0, text.lower().find(word) - 10):text.lower().find(word) + len(word) + 10]
            })
        
        return topics
    
    @function_tool
    async def _format_trigger(
        self,
        ctx: RunContextWrapper[RecognitionMemoryContext],
        trigger_type: str,
        trigger_value: str,
        relevance_threshold: float = 0.6,
        activation_strength: float = 1.0,
        context_requirements: Dict[str, Any] = None
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
            context_requirements=context_requirements or {},
            source="agent_generated"
        )
    
    @function_tool
    async def _query_memory(
        self,
        ctx: RunContextWrapper[RecognitionMemoryContext],
        trigger: ContextualTrigger,
        memory_types: List[str] = None,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
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
            query = trigger.trigger_value
            
            # Default memory types if not specified
            if not memory_types:
                memory_types = ["observation", "experience", "reflection"]
            
            # Call memory_core.retrieve_memories
            # We're assuming a method signature similar to the one in our document
            memories = await ctx.context.memory_core.retrieve_memories(
                query=query,
                memory_types=memory_types,
                limit=limit,
                min_significance=3,
                entities=[trigger.trigger_value] if trigger.trigger_type == "entity" else None
            )
            
            # Add trigger information to results
            for memory in memories:
                memory["activation_trigger"] = {
                    "trigger_id": trigger.trigger_id,
                    "trigger_type": trigger.trigger_type,
                    "trigger_value": trigger.trigger_value
                }
            
            return memories
            
        except Exception as e:
            logger.error(f"Error querying memory: {e}")
            return []
    
    @function_tool
    async def _combine_query_results(
        self,
        ctx: RunContextWrapper[RecognitionMemoryContext],
        memory_sets: List[List[Dict[str, Any]]],
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
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
                memory_id = memory.get("id")
                if memory_id and memory_id not in memory_ids:
                    all_memories.append(memory)
                    memory_ids.add(memory_id)
        
        # Sort by relevance if available
        if all_memories and "relevance" in all_memories[0]:
            all_memories.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        return all_memories[:max_results]
    
    @function_tool
    async def _calculate_contextual_relevance(
        self,
        ctx: RunContextWrapper[RecognitionMemoryContext],
        memory: Dict[str, Any],
        conversation_context: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate contextual relevance of a memory to current conversation
        
        Args:
            memory: Memory to evaluate
            conversation_context: Recent conversation context
            
        Returns:
            Contextual relevance score
        """
        # In a full implementation, this would perform sophisticated relevance calculation
        # Simplified version:
        
        # Start with the memory's relevance score if available
        base_relevance = memory.get("relevance", 0.5)
        
        # Get memory text and type
        memory_text = memory.get("memory_text", "")
        memory_type = memory.get("memory_type", "")
        
        # Extract conversation text
        conversation_text = ""
        for message in conversation_context:
            if "text" in message:
                conversation_text += " " + message["text"]
            elif "content" in message:
                conversation_text += " " + message["content"]
        
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
        if "timestamp" in memory.get("metadata", {}):
            try:
                timestamp = datetime.datetime.fromisoformat(
                    memory["metadata"]["timestamp"].replace("Z", "+00:00")
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
        self,
        ctx: RunContextWrapper[RecognitionMemoryContext],
        memory_id: str
    ) -> Dict[str, Any]:
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
                
                return {
                    "was_recognized": True,
                    "seconds_ago": seconds_ago,
                    "cooldown_period": cooldown,
                    "cooldown_remaining": max(0, cooldown - seconds_ago),
                    "in_cooldown": seconds_ago < cooldown
                }
        
        return {
            "was_recognized": False,
            "cooldown_period": ctx.context.recognition_cooldown,
            "cooldown_remaining": 0,
            "in_cooldown": False
        }
    
    # Main public methods for using the recognition memory system
    
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
        message = {
            "text": conversation_text,
            "timestamp": datetime.datetime.now().isoformat(),
            "context": current_context or {}
        }
        
        with trace(workflow_name="Process Conversation Turn", group_id=self.context.trace_id):
            # Update context history
            await self._update_context_history(
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
        context_requirements: Dict[str, Any] = None,
        source: str = "user"
    ) -> str:
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
            trigger = await self._format_trigger(
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
            self.context.active_triggers[trigger.trigger_id] = trigger
            
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
        
        if trigger_id in self.context.active_triggers:
            del self.context.active_triggers[trigger_id]
            return True
        
        return False
    
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
                    self.context.active_triggers[trigger.trigger_id] = trigger
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
