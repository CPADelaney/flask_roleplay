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
    
    def __init__(self, memory_core=None, context_awareness=None, reasoning_core=None):
        self.memory_core = memory_core
        self.context_awareness = context_awareness
        self.reasoning_core = reasoning_core  # Add this line
        
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
            self.context.active_triggers[trigger.trigger_id] = trigger
    
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
                function_tool(self._get_active_triggers),
                function_tool(self._get_recent_recognitions),
                function_tool(self._update_context_history),
                function_tool(self._analyze_context_state)
            ],
            handoffs=[
                handoff(
                    self.salience_detector_agent,
                    tool_name_override="detect_salient_elements",
                    tool_description_override="Detect salient elements in conversation that could trigger memory recognition",
                    on_handoff=self._prepare_for_salience_detection
                ),
                handoff(
                    self.trigger_extraction_agent,
                    tool_name_override="extract_contextual_triggers",
                    tool_description_override="Extract contextual triggers from the salient elements detected in conversation",
                    on_handoff=self._prepare_for_trigger_extraction
                ),
                handoff(
                    self.memory_query_agent,
                    tool_name_override="query_based_on_triggers",
                    tool_description_override="Query memory system based on the extracted contextual triggers",
                    on_handoff=self._prepare_for_memory_query
                ),
                handoff(
                    self.relevance_filter_agent,
                    tool_name_override="filter_recognition_results",
                    tool_description_override="Filter memory recognition results for contextual relevance and quality",
                    on_handoff=self._prepare_for_relevance_filtering
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
                function_tool(self._analyze_entities),
                function_tool(self._analyze_emotions),
                function_tool(self._analyze_topics),
                function_tool(self._analyze_narrative_elements),
                function_tool(self._leverage_context_awareness)
            ],
            output_type=SalienceAnalysisOutput,
            model="gpt-4o-mini",
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
                function_tool(self._query_memory),
                function_tool(self._query_with_prioritization),
                function_tool(self._query_memories_parallel),
                function_tool(self._combine_query_results),
                function_tool(self._track_query_performance)
            ],
            model="gpt-4o-mini",
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
                function_tool(self._get_trigger_types),
                function_tool(self._format_trigger),
                function_tool(self._assess_trigger_quality),
                function_tool(self._calibrate_trigger_parameters)
            ],
            output_type=TriggerExtractionOutput,
            model="gpt-4o-mini",
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
                function_tool(self._calculate_contextual_relevance),
                function_tool(self._assess_memory_causality),
                function_tool(self._blend_memory_with_context),
                function_tool(self._check_recognition_history),
                function_tool(self._assess_conversational_impact)
            ],
            output_type=RecognitionFilterOutput,
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.3)
        )
    
    # Tool functions for the various agents

    @staticmethod
    @function_tool
    async def _blend_memory_with_context(
        ctx: RunContextWrapper[RecognitionMemoryContext],
        memory: Dict[str, Any],
        conversation_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a lightweight “blend” of a memory and the current conversation context
        so the filter agent can present how they interrelate.
        """
        mem_text = memory.get("memory_text", "")
        convo_text = " ".join(msg.get("text", "") for msg in conversation_context)

        # Pick a sentence from memory that overlaps with context, or fallback
        overlap_words = set(mem_text.lower().split()) & set(convo_text.lower().split())
        if overlap_words:
            # find first sentence containing any overlap word
            for sentence in mem_text.split("."):
                if any(w in sentence.lower() for w in overlap_words):
                    snippet = sentence.strip()
                    break
            else:
                snippet = mem_text[:100].strip()
        else:
            snippet = mem_text[:100].strip()

        # A very simple “blend score” based on proportion of shared words
        shared = len(overlap_words)
        total = len(set(mem_text.lower().split()) | set(convo_text.lower().split()))
        blend_score = round(shared / total, 2) if total > 0 else 0.0

        return {
            "blended_snippet": snippet,
            "blend_score": blend_score
        }


    @staticmethod
    @function_tool
    async def _get_active_triggers(ctx: RunContextWrapper[RecognitionMemoryContext]) -> Dict[str, ContextualTrigger]:
        """
        Get the dictionary of currently active triggers
        
        Returns:
            Dictionary of trigger_id -> ContextualTrigger
        """
        return ctx.context.active_triggers

    @staticmethod
    @function_tool
    async def _get_recent_recognitions(ctx: RunContextWrapper[RecognitionMemoryContext]) -> List[Dict[str, Any]]:
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

    @staticmethod
    @function_tool
    async def _update_context_history(
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

    @staticmethod
    @function_tool
    async def _get_trigger_types(ctx: RunContextWrapper[RecognitionMemoryContext]) -> List[str]:
        """
        Get the list of valid trigger types
        
        Returns:
            List of trigger types
        """
        return ctx.context.trigger_types

    @staticmethod
    @function_tool
    async def _analyze_entities(
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

    @staticmethod
    @function_tool
    async def _analyze_emotions(
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

    @staticmethod
    @function_tool
    async def _analyze_topics(
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

    @staticmethod
    @function_tool
    async def _format_trigger(
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

    @staticmethod
    @function_tool
    async def _query_memory(
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

    @staticmethod
    @function_tool
    async def _combine_query_results(
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

    @staticmethod
    @function_tool
    async def _calculate_contextual_relevance(
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

    @staticmethod
    @function_tool
    async def _check_recognition_history(
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

    @staticmethod
    @function_tool
    async def _analyze_context_state(
        ctx: RunContextWrapper[RecognitionMemoryContext]
    ) -> Dict[str, Any]:
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
            last_message = conversation[-1].get("text", "")
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
        
        return {
            "recognition_sensitivity": ctx.context.recognition_sensitivity,
            "max_recognitions": ctx.context.max_recognitions_per_turn,
            "context_complexity": len(conversation),
            "active_triggers_count": len(ctx.context.active_triggers)
        }

    @staticmethod
    @function_tool
    async def _analyze_narrative_elements(
        ctx: RunContextWrapper[RecognitionMemoryContext],
        text: str
    ) -> List[Dict[str, Any]]:
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
                
                narrative_elements.append({
                    "element_type": "narrative_marker",
                    "marker": marker,
                    "context": context,
                    "salience": 0.7,
                    "position": marker_position
                })
        
        # Check for event descriptions
        event_indicators = ["happened", "occurred", "took place", "experienced", "went to"]
        
        for indicator in event_indicators:
            indicator_position = text.lower().find(indicator)
            if indicator_position >= 0:
                # Get surrounding context
                start = max(0, indicator_position - 25)
                end = min(len(text), indicator_position + 25)
                context = text[start:end]
                
                narrative_elements.append({
                    "element_type": "event_description",
                    "indicator": indicator,
                    "context": context,
                    "salience": 0.8,
                    "position": indicator_position
                })
        
        return narrative_elements

    @staticmethod
    @function_tool
    async def _leverage_context_awareness(
        ctx: RunContextWrapper[RecognitionMemoryContext],
        text: str
    ) -> Dict[str, Any]:
        """
        Leverage context awareness system for enhanced salience detection
        
        Args:
            text: Text to analyze
            
        Returns:
            Context awareness results
        """
        if not ctx.context.context_awareness:
            return {
                "entities": [],
                "topics": [],
                "emotions": []
            }
        
        try:
            # Use context awareness system
            awareness_result = await ctx.context.context_awareness.analyze_content(text)
            
            # Extract elements
            entities = []
            for entity in awareness_result.get("entities", []):
                entities.append({
                    "entity": entity.get("text", ""),
                    "type": entity.get("type", "unknown"),
                    "salience": entity.get("salience", 0.5)
                })
                
            topics = []
            for topic in awareness_result.get("topics", []):
                topics.append({
                    "topic": topic.get("name", ""),
                    "confidence": topic.get("confidence", 0.5),
                    "keywords": topic.get("keywords", [])
                })
                
            emotions = []
            for emotion in awareness_result.get("emotions", []):
                emotions.append({
                    "emotion": emotion.get("name", ""),
                    "intensity": emotion.get("score", 0.5)
                })
                
            return {
                "entities": entities,
                "topics": topics,
                "emotions": emotions
            }
            
        except Exception as e:
            logger.error(f"Error leveraging context awareness: {e}")
            return {
                "entities": [],
                "topics": [],
                "emotions": []
            }

    @staticmethod
    @function_tool
    async def _query_with_prioritization(
        ctx: RunContextWrapper[RecognitionMemoryContext],
        query: str,
        trigger: Dict[str, Any] = None,
        memory_types: List[str] = None,
        prioritization: Dict[str, float] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
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
                
            # Add trigger information
            if trigger:
                for memory in memories:
                    memory["activation_trigger"] = trigger
                    
            return memories
            
        except Exception as e:
            logger.error(f"Error in prioritized memory query: {e}")
            return []

    @staticmethod
    @function_tool
    async def _query_memories_parallel(
        ctx: RunContextWrapper[RecognitionMemoryContext],
        triggers: List[Dict[str, Any]],
        memory_types: List[str] = None,
        limit_per_trigger: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
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
            return {}
            
        # Default memory types
        if not memory_types:
            memory_types = ["experience", "reflection", "abstraction", "observation"]
            
        try:
            # Create tasks for each trigger
            trigger_tasks = {}
            for trigger in triggers:
                trigger_id = trigger.get("trigger_id", str(uuid.uuid4()))
                trigger_value = trigger.get("trigger_value", "")
                trigger_type = trigger.get("trigger_type", "unknown")
                
                # Create retrieval task
                if hasattr(ctx.context.memory_core, "retrieve_memories_parallel"):
                    # Use parallel retrieval if available
                    trigger_tasks[trigger_id] = ctx.context.memory_core.retrieve_memories(
                        query=trigger_value,
                        memory_types=memory_types,
                        limit=limit_per_trigger
                    )
                else:
                    # Fall back to standard retrieval
                    trigger_tasks[trigger_id] = self._query_memory(
                        ctx,
                        ContextualTrigger(
                            trigger_id=trigger_id,
                            trigger_type=trigger_type,
                            trigger_value=trigger_value
                        ),
                        memory_types,
                        limit_per_trigger
                    )
            
            # Wait for all tasks to complete
            results = {}
            for trigger_id, task in trigger_tasks.items():
                try:
                    results[trigger_id] = await task
                except Exception as e:
                    logger.error(f"Error in parallel query for trigger {trigger_id}: {e}")
                    results[trigger_id] = []
                    
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel memory queries: {e}")
            return {}

    @staticmethod
    @function_tool
    async def _track_query_performance(
        ctx: RunContextWrapper[RecognitionMemoryContext],
        trigger_id: str,
        memories_found: int,
        avg_relevance: float
    ) -> Dict[str, Any]:
        """
        Track query performance for trigger optimization
        
        Args:
            trigger_id: ID of the trigger
            memories_found: Number of memories found
            avg_relevance: Average relevance of retrieved memories
            
        Returns:
            Performance metrics
        """
        # Create performance entry if it doesn't exist
        if trigger_id not in ctx.context.trigger_performance:
            ctx.context.trigger_performance[trigger_id] = {
                "query_count": 0,
                "total_memories": 0,
                "avg_relevance": 0.0,
                "last_updated": datetime.datetime.now().isoformat()
            }
            
        # Update performance metrics
        performance = ctx.context.trigger_performance[trigger_id]
        performance["query_count"] += 1
        performance["total_memories"] += memories_found
        
        # Calculate new average relevance
        old_avg = performance.get("avg_relevance", 0.0)
        old_count = performance.get("query_count", 1) - 1  # Subtract the one we just added
        
        if old_count > 0:
            performance["avg_relevance"] = (old_avg * old_count + avg_relevance) / (old_count + 1)
        else:
            performance["avg_relevance"] = avg_relevance
            
        performance["last_updated"] = datetime.datetime.now().isoformat()
        
        # Check if trigger needs optimization
        needs_optimization = False
        
        # If too many queries with no results
        if performance["query_count"] >= 3 and performance["total_memories"] == 0:
            needs_optimization = True
            
        # If consistently low relevance
        if performance["query_count"] >= 5 and performance["avg_relevance"] < 0.3:
            needs_optimization = True
            
        return {
            "trigger_id": trigger_id,
            "query_count": performance["query_count"],
            "total_memories": performance["total_memories"],
            "avg_relevance": performance["avg_relevance"],
            "needs_optimization": needs_optimization
        }

    @staticmethod
    @function_tool
    async def _assess_trigger_quality(
        ctx: RunContextWrapper[RecognitionMemoryContext],
        trigger_type: str,
        trigger_value: str
    ) -> Dict[str, Any]:
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
                similarity = self._calculate_string_similarity(
                    existing.trigger_value.lower(),
                    trigger_value.lower()
                )
                
                if similarity > 0.7:  # High similarity
                    similar_triggers.append({
                        "trigger_id": existing_id,
                        "trigger_value": existing.trigger_value,
                        "similarity": similarity
                    })
        
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
        
        return {
            "quality_score": quality,
            "is_generic": is_generic,
            "similar_triggers": similar_triggers,
            "specificity": specificity,
            "recommended_threshold": max(0.5, 0.7 - (quality * 0.2))  # Adjust threshold based on quality
        }

    @staticmethod
    @function_tool
    async def _assess_memory_causality(
        ctx: RunContextWrapper[RecognitionMemoryContext],
        memory: Dict[str, Any],
        conversation_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Assess causal connections between a recognized memory and the current conversation.
        Returns a simple causality strength score and any trigger keywords found.
        """
        # Combine memory text and recent conversation
        mem_text = memory.get("memory_text", "")
        convo_text = " ".join(msg.get("text", "") for msg in conversation_context)
        full_text = f"{mem_text} {convo_text}".lower()

        # Look for basic causal markers
        causality_markers = ["because", "due to", "as a result", "therefore", "so that"]
        found = [kw for kw in causality_markers if kw in full_text]

        # Simple strength: proportion of markers found capped at 1.0
        strength = min(1.0, len(found) / len(causality_markers))

        return {
            "causality_strength": strength,
            "causality_markers_found": found
        }

    @staticmethod
    @function_tool
    async def _calibrate_trigger_parameters(
        ctx: RunContextWrapper[RecognitionMemoryContext],
        trigger_type: str,
        trigger_value: str,
        quality_assessment: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calibrate trigger parameters based on quality assessment
        
        Args:
            trigger_type: Type of trigger
            trigger_value: Value of the trigger
            quality_assessment: Quality assessment data
            
        Returns:
            Calibrated parameters
        """
        quality = quality_assessment.get("quality_score", 0.5)
        is_generic = quality_assessment.get("is_generic", False)
        specificity = quality_assessment.get("specificity", 0.5)
        
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
            
        return {
            "relevance_threshold": relevance_threshold,
            "activation_strength": activation_strength
        }

    @staticmethod
    @function_tool
    async def _assess_conversational_impact(
        ctx: RunContextWrapper[RecognitionMemoryContext],
        memory: Dict[str, Any],
        conversation_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Assess potential conversational impact of a recognized memory
        
        Args:
            memory: Memory to evaluate
            conversation_context: Recent conversation context
            
        Returns:
            Impact assessment
        """
        # Get memory properties
        memory_text = memory.get("memory_text", "")
        memory_type = memory.get("memory_type", "")
        memory_significance = memory.get("significance", 5)
        
        # Extract last message
        last_message = ""
        if conversation_context:
            last_message = conversation_context[-1].get("text", "")
            
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
            novelty = min(1.0, len(unique_to_memory) / len(words_memory))
            
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
        
        return {
            "impact_score": impact_score,
            "novelty": novelty,
            "relevance": relevance,
            "coherence": coherence,
            "impact_type": "elaboration" if elaboration else "contrast" if contrast else "reinforcement" if reinforcement else "mixed"
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

    async def _prepare_for_salience_detection(self, ctx, agent, input_data):
        """Prepare context for salience detection handoff"""
        # Extract content and current context
        text = ""
        if isinstance(input_data, str):
            text = input_data
        else:
            # Try to find text in a list of messages
            for item in input_data:
                if isinstance(item, dict) and "content" in item:
                    text += item["content"] + " "
                elif isinstance(item, dict) and "text" in item:
                    text += item["text"] + " "
        
        # Reset context fields for this analysis
        ctx.context.context_topics = []
        ctx.context.context_entities = []
        ctx.context.context_emotions = []
        
        return input_data
    
    async def _prepare_for_trigger_extraction(self, ctx, agent, input_data):
        """Prepare context for trigger extraction handoff"""
        # Set the maximum number of triggers to extract
        ctx.context.max_triggers_current = min(
            ctx.context.max_triggers_per_turn,
            len(ctx.context.context_entities) + len(ctx.context.context_topics) + len(ctx.context.context_emotions)
        )
        
        return input_data
    
    async def _prepare_for_memory_query(self, ctx, agent, input_data):
        """Prepare context for memory query handoff"""
        # Set maximum number of memories to retrieve per trigger
        max_memories_per_trigger = 3
        if len(ctx.context.active_triggers) > 5:
            max_memories_per_trigger = 2
        
        # Store in context
        ctx.context.max_memories_per_trigger = max_memories_per_trigger
        
        return input_data
    
    async def _prepare_for_relevance_filtering(self, ctx, agent, input_data):
        """Prepare context for relevance filtering handoff"""
        # Set maximum number of memories to return after filtering
        ctx.context.max_memories_to_return = ctx.context.max_recognitions_per_turn
        
        return input_data
    
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
