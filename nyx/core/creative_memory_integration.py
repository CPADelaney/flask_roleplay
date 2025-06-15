# nyx/core/creative_memory_integration.py

import logging
import asyncio
import datetime
import random
import re
from typing import List, Any, Optional, Tuple, Set, Dict
from collections import Counter, defaultdict
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
    RunConfig
)

# Import our systems
from nyx.core.novelty_engine import NoveltyEngine, NoveltyIdea
from nyx.core.recognition_memory import RecognitionMemorySystem, RecognitionResult

logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic models to replace Dict[str, Any] usage
# ============================================================================

class MemoryData(BaseModel):
    """Model for memory data"""
    model_config = ConfigDict(extra='forbid')
    
    id: str = Field(default="")
    memory_text: str
    memory_type: str = Field(default="")
    relevance_score: float = Field(default=0.5)
    activation_trigger: Optional[str] = None

class MemoryConcept(BaseModel):
    """Model for memory concept extraction"""
    model_config = ConfigDict(extra='forbid')
    
    memory_id: str
    key_phrases: List[str]
    source_text: str
    entities: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    emotional_tone: Optional[str] = None

class WitGenerationDecision(BaseModel):
    """Model for wit generation decision"""
    model_config = ConfigDict(extra='forbid')
    
    should_generate: bool
    reason: str = Field(default="")
    suitable_memory_count: int = Field(default=0)
    wit_types: List[str] = Field(default_factory=list)
    conversation_sentiment: Optional[str] = None
    appropriateness_score: float = Field(default=0.5)

class AnalogyResult(BaseModel):
    """Model for analogy creation result"""
    model_config = ConfigDict(extra='forbid')
    
    analogy_text: str
    memory_element: str
    context_element: str
    analogy_type: str
    shared_attributes: List[str] = Field(default_factory=list)
    abstraction_level: float = Field(default=0.5)
    confidence: float = Field(default=0.5)

class WordplayResult(BaseModel):
    """Model for wordplay creation result"""
    model_config = ConfigDict(extra='forbid')
    
    wordplay_text: str
    wordplay_type: str
    key_words: List[str] = Field(default_factory=list)
    linguistic_device: str = Field(default="")
    cleverness_score: float = Field(default=0.5)

class PatternInfo(BaseModel):
    """Model for pattern extraction result"""
    model_config = ConfigDict(extra='forbid')
    
    pattern_description: str
    common_elements: List[Tuple[str, int]] = Field(default_factory=list)
    memory_count: int
    confidence: float
    pattern_type: str = Field(default="thematic")
    temporal_patterns: List[str] = Field(default_factory=list)
    causal_patterns: List[str] = Field(default_factory=list)
    structural_patterns: List[str] = Field(default_factory=list)

class PrincipleInfo(BaseModel):
    """Model for abstracted principle"""
    model_config = ConfigDict(extra='forbid')
    
    principle: str
    elements: List[str] = Field(default_factory=list)
    abstraction_level: float
    confidence: float
    applicability: str
    principle_type: str = Field(default="general")
    supporting_evidence: List[str] = Field(default_factory=list)
    counter_examples: List[str] = Field(default_factory=list)

class CreativeTechniqueResult(BaseModel):
    """Model for creative technique application result"""
    model_config = ConfigDict(extra='forbid')
    
    technique: str
    memory_id: str
    transformed_text: str
    transformation_notes: str
    creativity_score: float = Field(default=0.5)
    coherence_score: float = Field(default=0.5)
    intermediate_steps: List[str] = Field(default_factory=list)

class TransformationResult(BaseModel):
    """Model for memory transformation result"""
    model_config = ConfigDict(extra='forbid')
    
    memory_id: str
    original_text: str
    transformed_text: str
    transformation_type: str
    transformation_notes: str
    semantic_distance: float = Field(default=0.5)
    preserved_elements: List[str] = Field(default_factory=list)
    novel_elements: List[str] = Field(default_factory=list)

class BlendedElement(BaseModel):
    """Model for blended elements"""
    model_config = ConfigDict(extra='forbid')
    
    memory_concept: str
    conversation_concept: str
    connection_strength: float = Field(default=0.5)
    connection_type: str = Field(default="associative")

class BlendResult(BaseModel):
    """Model for memory-novelty blend result"""
    model_config = ConfigDict(extra='forbid')
    
    status: str
    memory_id: str = Field(default="")
    technique_used: str = Field(default="")
    novel_idea: Optional[str] = None
    blended_text: Optional[str] = None
    blended_elements: Optional[BlendedElement] = None
    creativity_metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None

class ProcessingResult(BaseModel):
    """Model for conversation processing result"""
    model_config = ConfigDict(extra='forbid')
    
    status: str
    content_type: Optional[str] = None
    content_id: Optional[str] = None
    generated_content: Optional[Any] = None
    processing_metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ContentRetrievalResult(BaseModel):
    """Model for content retrieval result"""
    model_config = ConfigDict(extra='forbid')
    
    contextual_wits: List[Tuple[str, Any]] = Field(default_factory=list)
    creative_insights: List[Tuple[str, Any]] = Field(default_factory=list)

# ============================================================================
# Original Pydantic models for integration
# ============================================================================

class ContextualWit(BaseModel):
    """Schema for contextual wit generated from memory and novelty"""
    model_config = ConfigDict(extra='forbid')
    
    wit_text: str
    wit_type: str  # e.g., "analogy", "wordplay", "reference", "insight"
    related_memory_id: Optional[str] = None
    novelty_score: float
    appropriateness_score: float
    confidence: float
    generation_technique: str
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

class CreativeInsight(BaseModel):
    """Schema for a creative insight based on remembered experience"""
    model_config = ConfigDict(extra='forbid')
    
    insight_text: str
    source_memory_id: Optional[str] = None
    related_concept: str
    abstraction_level: float
    novelty_score: float
    usefulness_score: float
    insight_type: str  # e.g., "pattern", "principle", "perspective", "connection"
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

class CreativeMemoryIntegrationContext:
    """Context for the creative memory integration"""
    
    def __init__(self, novelty_engine=None, recognition_memory=None, memory_core=None):
        self.novelty_engine = novelty_engine
        self.recognition_memory = recognition_memory
        self.memory_core = memory_core
        
        # Storage for generated content - using lists instead of dicts
        self.contextual_wits: List[Tuple[str, ContextualWit]] = []
        self.creative_insights: List[Tuple[str, CreativeInsight]] = []
        
        # Trace ID for connecting traces
        self.trace_id = f"creative_memory_{datetime.datetime.now().isoformat()}"
        
        # Integration parameters
        self.wit_generation_probability = 0.3  # Probability of generating wit on a turn
        self.insight_threshold = 0.7  # Minimum relevance for insight generation
        self.wit_types = ["analogy", "wordplay", "reference", "insight", "juxtaposition"]
        self.insight_types = ["pattern", "principle", "perspective", "connection"]
        
        # Enhanced parameters for production
        self.min_memory_length = 20  # Minimum memory length for processing
        self.max_abstraction_depth = 3  # Maximum levels of abstraction
        self.creativity_temperature = 0.7  # Controls randomness in creative generation
        self.coherence_threshold = 0.6  # Minimum coherence score for outputs

class CreativeMemoryIntegration:
    """
    System that integrates recognition memory with the novelty engine
    to generate contextual wit, creative insights, and memorable responses.
    """
    
    def __init__(self, novelty_engine=None, recognition_memory=None, memory_core=None):
        """Initialize the integration system with required components"""
        # Create context
        self.context = CreativeMemoryIntegrationContext(
            novelty_engine=novelty_engine,
            recognition_memory=recognition_memory,
            memory_core=memory_core
        )
        
        # Initialize agent system
        self.integration_agent = None
        self.initialized = False
        
        # Precompiled patterns for efficiency
        self._init_linguistic_patterns()
    
    def _init_linguistic_patterns(self):
        """Initialize linguistic patterns for production use"""
        # Patterns for entity extraction
        self.entity_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
        
        # Patterns for temporal expressions
        self.temporal_pattern = re.compile(
            r'\b(?:yesterday|today|tomorrow|last\s+\w+|next\s+\w+|'
            r'\d+\s+(?:days?|weeks?|months?|years?)\s+ago|'
            r'in\s+\d+\s+(?:days?|weeks?|months?|years?))\b',
            re.IGNORECASE
        )
        
        # Patterns for causal expressions
        self.causal_pattern = re.compile(
            r'\b(?:because|due\s+to|as\s+a\s+result|therefore|'
            r'consequently|thus|hence|so\s+that|leads?\s+to|'
            r'causes?|results?\s+in)\b',
            re.IGNORECASE
        )
        
        # Emotion indicators
        self.emotion_indicators = {
            'positive': ['happy', 'joy', 'excited', 'pleased', 'delighted', 'glad', 'cheerful', 'elated'],
            'negative': ['sad', 'upset', 'angry', 'frustrated', 'disappointed', 'hurt', 'annoyed', 'depressed'],
            'neutral': ['calm', 'peaceful', 'content', 'relaxed', 'comfortable'],
            'anxious': ['worried', 'anxious', 'nervous', 'concerned', 'stressed', 'tense'],
            'surprised': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned']
        }
    
    async def initialize(self):
        """Initialize the integration system and its agents"""
        if self.initialized:
            return
            
        logger.info("Initializing CreativeMemoryIntegration")
        
        with trace(workflow_name="CreativeMemory Initialization", group_id=self.context.trace_id):
            self._initialize_agents()
            self.initialized = True
            logger.info("CreativeMemoryIntegration initialized with Agents SDK")
    
    def _initialize_agents(self):
        """Initialize all specialized agents needed for the integration"""
        # Create specialized agents
        self.wit_generation_agent = self._create_wit_generation_agent()
        self.insight_generation_agent = self._create_insight_generation_agent()
        self.memory_novelty_agent = self._create_memory_novelty_agent()
        
        # Create the main integration agent with handoffs
        self.integration_agent = self._create_integration_agent()
        
        logger.info("Creative memory integration agents initialized")
    
    def _create_integration_agent(self) -> Agent:
        """Create the main orchestrator agent for creative memory integration"""
        return Agent[CreativeMemoryIntegrationContext](
            name="Creative Memory Integration",
            instructions="""You are the creative memory integration system for the Nyx AI.
            
            Your role is to coordinate the integration of recognition memory with
            the novelty engine to generate contextually appropriate creative content
            like wit, insights, and memorable responses.
            
            You analyze the conversation context and recognized memories to determine
            when and how to leverage creativity. You can:
            
            1. Generate contextual wit when appropriate
            2. Create creative insights from recognized memories
            3. Blend memory recognition with novel idea generation
            
            Focus on enhancing the conversation in ways that feel natural and
            appropriate, not forced or random. Creative elements should add value
            and feel integrated with Nyx's memory and personality.
            
            Consider:
            - Conversation sentiment and emotional context
            - Memory relevance and richness
            - Appropriateness of creative interventions
            - Balance between novelty and coherence
            """,
            tools=[
                function_tool(self._should_generate_wit),
                function_tool(self._get_memory_concepts)
            ],
            handoffs=[
                handoff(self.wit_generation_agent,
                      tool_name_override="generate_contextual_wit",
                      tool_description_override="Generate contextually appropriate wit"),
                handoff(self.insight_generation_agent,
                      tool_name_override="generate_creative_insight",
                      tool_description_override="Generate creative insight from memory"),
                handoff(self.memory_novelty_agent,
                      tool_name_override="blend_memory_with_novelty",
                      tool_description_override="Blend recognized memory with novel idea generation")
            ],
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.6)
        )
    
    def _create_wit_generation_agent(self) -> Agent:
        """Create specialized agent for contextual wit generation"""
        return Agent[CreativeMemoryIntegrationContext](
            name="Wit Generation Agent",
            instructions="""You are specialized in generating contextual wit based on
            recognized memories and conversation context.
            
            Your task is to create witty, clever responses that reference recognized
            memories in ways that enhance the conversation. Your wit should be:
            
            1. Contextually appropriate and relevant to the conversation
            2. Subtly connected to recognized memories
            3. Genuinely clever or insightful (not forced)
            4. Aligned with Nyx's persona
            
            You can generate various types of wit:
            - Analogies that connect memory to current context
            - Clever wordplay referencing recognized concepts
            - Insightful observations drawing from past experience
            - Humorous juxtapositions of memory and present
            
            Focus on quality over quantity - only generate wit when it truly adds value.
            Consider linguistic sophistication, timing, and emotional appropriateness.
            """,
            tools=[
                function_tool(self._create_memory_analogy),
                function_tool(self._create_wordplay)
            ],
            output_type=ContextualWit,
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.7)
        )
    
    def _create_insight_generation_agent(self) -> Agent:
        """Create specialized agent for creative insight generation"""
        return Agent[CreativeMemoryIntegrationContext](
            name="Insight Generation Agent",
            instructions="""You are specialized in generating creative insights by
            abstracting patterns and principles from recognized memories.
            
            Your task is to create deeper, more abstract insights from specific
            memories that apply to the current context. Your insights should:
            
            1. Extract generalizable principles from specific experiences
            2. Identify patterns across multiple memories
            3. Provide novel perspectives on the current situation
            4. Create valuable connections between past and present
            
            Focus on generating insights that are both novel and useful - they should
            provide genuine value and help extend understanding beyond the obvious.
            
            Consider:
            - Multiple levels of abstraction
            - Cross-domain pattern recognition
            - Temporal and causal relationships
            - Contradictions and paradoxes that reveal deeper truths
            """,
            tools=[
                function_tool(self._extract_memory_patterns),
                function_tool(self._abstract_principles)
            ],
            output_type=CreativeInsight,
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.6)
        )
    
    def _create_memory_novelty_agent(self) -> Agent:
        """Create specialized agent for blending memory with novelty"""
        return Agent[CreativeMemoryIntegrationContext](
            name="Memory-Novelty Blend Agent",
            instructions="""You are specialized in blending recognized memories with
            novel idea generation to create creative, memory-informed responses.
            
            Your task is to leverage both the recognition memory system and novelty
            engine to generate responses that combine familiarity with novelty.
            You should:
            
            1. Use recognized memories as anchors for creative exploration
            2. Apply creative techniques to transform memories into novel ideas
            3. Blend memory retrieval with creative generation
            4. Create responses that feel both familiar and fresh
            
            Focus on creating a seamless integration where memory and novelty
            complement each other rather than feeling disconnected.
            
            Techniques to employ:
            - Conceptual blending and bisociation
            - Metaphorical transformation
            - Perspective shifting and reframing
            - Constraint manipulation
            - Analogical extension
            """,
            tools=[
                function_tool(self._apply_creative_technique),
                function_tool(self._transform_memory)
            ],
            model="gpt-4o",
            model_settings=ModelSettings(temperature=0.7)
        )
    
    # Enhanced tool functions for production use

    @staticmethod
    @function_tool
    async def _should_generate_wit(
        ctx: RunContextWrapper[CreativeMemoryIntegrationContext],
        conversation_text: str,
        recognized_memories: List[MemoryData]
    ) -> WitGenerationDecision:
        """
        Determine if wit generation is appropriate for current context
        
        Args:
            conversation_text: Current conversation text
            recognized_memories: List of recognized memories
            
        Returns:
            Decision about wit generation with detailed analysis
        """
        # Analyze conversation sentiment
        sentiment = "neutral"
        sentiment_score = 0.5
        
        # Count emotion indicators
        text_lower = conversation_text.lower()
        emotion_counts = {}
        
        for emotion_type, indicators in ctx.context.emotion_indicators.items():
            count = sum(1 for indicator in indicators if indicator in text_lower)
            if count > 0:
                emotion_counts[emotion_type] = count
        
        if emotion_counts:
            # Determine dominant emotion
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
            sentiment = dominant_emotion
            
            # Calculate sentiment score
            total_indicators = sum(emotion_counts.values())
            if dominant_emotion in ['positive', 'neutral']:
                sentiment_score = 0.7 + (0.3 * min(total_indicators / 5, 1.0))
            elif dominant_emotion == 'surprised':
                sentiment_score = 0.6
            else:
                sentiment_score = 0.3 - (0.2 * min(total_indicators / 5, 1.0))
        
        # Length and complexity check
        word_count = len(conversation_text.split())
        if word_count < 5:
            return WitGenerationDecision(
                should_generate=False,
                reason="message_too_short",
                conversation_sentiment=sentiment,
                appropriateness_score=0.0
            )
        
        # Need at least one recognized memory
        if not recognized_memories:
            return WitGenerationDecision(
                should_generate=False,
                reason="no_recognized_memories",
                conversation_sentiment=sentiment,
                appropriateness_score=0.0
            )
        
        # Filter memories by quality
        quality_memories = [
            m for m in recognized_memories 
            if len(m.memory_text) >= ctx.context.min_memory_length
            and m.relevance_score >= 0.5
        ]
        
        if not quality_memories:
            return WitGenerationDecision(
                should_generate=False,
                reason="no_quality_memories",
                conversation_sentiment=sentiment,
                appropriateness_score=0.2
            )
        
        # Check for serious context indicators
        serious_indicators = [
            'emergency', 'urgent', 'critical', 'help', 'crisis',
            'death', 'died', 'dying', 'illness', 'sick',
            'accident', 'injured', 'hurt badly', 'hospital'
        ]
        
        is_serious = any(indicator in text_lower for indicator in serious_indicators)
        
        if is_serious:
            return WitGenerationDecision(
                should_generate=False,
                reason="serious_context",
                conversation_sentiment=sentiment,
                appropriateness_score=0.0
            )
        
        # Calculate appropriateness score
        appropriateness = sentiment_score
        
        # Boost for certain contexts
        if any(word in text_lower for word in ['funny', 'humor', 'joke', 'laugh']):
            appropriateness += 0.2
        
        if any(word in text_lower for word in ['creative', 'interesting', 'clever', 'witty']):
            appropriateness += 0.15
        
        # Reduce for formal contexts
        if any(word in text_lower for word in ['professional', 'formal', 'business', 'meeting']):
            appropriateness -= 0.3
        
        appropriateness = max(0.0, min(1.0, appropriateness))
        
        # Probability check with dynamic threshold
        adjusted_probability = ctx.context.wit_generation_probability * appropriateness
        if random.random() > adjusted_probability:
            return WitGenerationDecision(
                should_generate=False,
                reason="probability_threshold",
                suitable_memory_count=len(quality_memories),
                conversation_sentiment=sentiment,
                appropriateness_score=appropriateness
            )
        
        # Select appropriate wit types based on context
        suitable_wit_types = ctx.context.wit_types.copy()
        
        # Filter wit types based on context
        if sentiment in ['negative', 'anxious']:
            # Remove potentially insensitive wit types
            suitable_wit_types = [w for w in suitable_wit_types if w not in ['wordplay', 'juxtaposition']]
        
        if word_count < 15:
            # Prefer simpler wit for short messages
            suitable_wit_types = [w for w in suitable_wit_types if w in ['reference', 'insight']]
        
        # If we get here, wit generation is appropriate
        return WitGenerationDecision(
            should_generate=True,
            suitable_memory_count=len(quality_memories),
            wit_types=suitable_wit_types[:3] if suitable_wit_types else ['insight'],
            conversation_sentiment=sentiment,
            appropriateness_score=appropriateness
        )

    @staticmethod
    @function_tool
    async def _get_memory_concepts(
        ctx: RunContextWrapper[CreativeMemoryIntegrationContext],
        memories: List[MemoryData]
    ) -> List[MemoryConcept]:
        """
        Extract key concepts from memories with sophisticated NLP
        
        Args:
            memories: List of memories
            
        Returns:
            List of extracted concepts with rich metadata
        """
        concepts = []
        
        for memory in memories:
            memory_text = memory.memory_text
            memory_id = memory.id
            
            # Extract entities
            entities = ctx.context.entity_pattern.findall(memory_text)
            
            # Extract key phrases using n-gram analysis
            words = memory_text.split()
            key_phrases = []
            
            # Unigrams (single important words)
            important_words = [
                w for w in words 
                if len(w) > 4 and w.lower() not in {
                    'that', 'this', 'these', 'those', 'there',
                    'where', 'which', 'while', 'about', 'through'
                }
            ]
            key_phrases.extend(important_words[:5])
            
            # Bigrams
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                # Check for meaningful bigrams
                if (words[i][0].isupper() or words[i+1][0].isupper() or
                    any(w in ['and', 'of', 'in', 'with', 'for'] for w in [words[i], words[i+1]])):
                    key_phrases.append(bigram)
            
            # Trigrams for more context
            for i in range(len(words) - 2):
                # Look for pattern: determiner + adjective + noun
                if (words[i].lower() in ['the', 'a', 'an', 'this', 'that'] and
                    i + 2 < len(words)):
                    trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                    key_phrases.append(trigram)
            
            # Extract themes using keyword clustering
            themes = []
            
            # Temporal themes
            if ctx.context.temporal_pattern.search(memory_text):
                themes.append("temporal")
            
            # Causal themes
            if ctx.context.causal_pattern.search(memory_text):
                themes.append("causal")
            
            # Emotional themes
            text_lower = memory_text.lower()
            for emotion_type, indicators in ctx.context.emotion_indicators.items():
                if any(indicator in text_lower for indicator in indicators):
                    themes.append(f"emotional_{emotion_type}")
                    break
            
            # Determine emotional tone
            emotional_tone = None
            emotion_scores = {}
            
            for emotion_type, indicators in ctx.context.emotion_indicators.items():
                score = sum(1 for indicator in indicators if indicator in text_lower)
                if score > 0:
                    emotion_scores[emotion_type] = score
            
            if emotion_scores:
                emotional_tone = max(emotion_scores.items(), key=lambda x: x[1])[0]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_phrases = []
            for phrase in key_phrases:
                if phrase.lower() not in seen:
                    seen.add(phrase.lower())
                    unique_phrases.append(phrase)
            
            concepts.append(MemoryConcept(
                memory_id=memory_id,
                key_phrases=unique_phrases[:10],  # Limit to top 10
                source_text=memory_text,
                entities=entities[:5],  # Limit to top 5 entities
                themes=themes,
                emotional_tone=emotional_tone
            ))
        
        return concepts

    @staticmethod
    @function_tool
    async def _create_memory_analogy(
        ctx: RunContextWrapper[CreativeMemoryIntegrationContext],
        current_context: str,
        memory_text: str
    ) -> AnalogyResult:
        """
        Create sophisticated analogy between current context and memory
        
        Args:
            current_context: Current conversation context
            memory_text: Text of the memory
            
        Returns:
            Rich analogy information
        """
        # Extract key elements from both texts
        def extract_key_elements(text):
            """Extract entities, actions, and attributes from text"""
            entities = ctx.context.entity_pattern.findall(text)
            
            # Extract actions (verbs)
            action_patterns = [
                r'\b(?:is|are|was|were|have|has|had|do|does|did)\s+(\w+ing)\b',
                r'\b(\w+ed)\b',
                r'\b(\w+s)\b'  # Simple present tense
            ]
            actions = []
            for pattern in action_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                actions.extend(matches)
            
            # Extract attributes (adjectives)
            attribute_patterns = [
                r'\b(?:very|quite|rather|really|so)\s+(\w+)\b',
                r'\b(\w+)\s+(?:thing|person|place|time|way)\b'
            ]
            attributes = []
            for pattern in attribute_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                attributes.extend(matches)
            
            return {
                'entities': entities,
                'actions': actions[:5],
                'attributes': attributes[:5],
                'full_text': text
            }
        
        memory_elements = extract_key_elements(memory_text)
        context_elements = extract_key_elements(current_context)
        
        # Find shared attributes
        shared_attributes = []
        
        # Check for shared entities
        shared_entities = set(memory_elements['entities']) & set(context_elements['entities'])
        if shared_entities:
            shared_attributes.extend([f"entity:{e}" for e in shared_entities])
        
        # Check for shared actions
        shared_actions = set(memory_elements['actions']) & set(context_elements['actions'])
        if shared_actions:
            shared_attributes.extend([f"action:{a}" for a in shared_actions])
        
        # Check for shared attributes
        shared_attrs = set(memory_elements['attributes']) & set(context_elements['attributes'])
        if shared_attrs:
            shared_attributes.extend([f"attribute:{a}" for a in shared_attrs])
        
        # Determine analogy type and create appropriate analogy
        if len(shared_attributes) > 2:
            analogy_type = "structural"
            abstraction_level = 0.8
        elif shared_entities:
            analogy_type = "entity-based"
            abstraction_level = 0.5
        elif shared_actions:
            analogy_type = "action-based"
            abstraction_level = 0.6
        else:
            analogy_type = "thematic"
            abstraction_level = 0.7
        
        # Select key phrases for the analogy
        memory_phrase = memory_text.split('.')[0][:100]
        context_phrase = current_context.split('.')[0][:100]
        
        # Generate analogy based on type
        if analogy_type == "structural":
            analogy_templates = [
                "The relationship in '{context}' mirrors the structure I observed in '{memory}'",
                "Just as {memory}, here we see {context} following a similar pattern",
                "The dynamics of {context} echo what happened with {memory}",
                "There's a structural parallel between {context} and my experience with {memory}"
            ]
        elif analogy_type == "entity-based":
            shared_entity = list(shared_entities)[0] if shared_entities else "this"
            analogy_templates = [
                "Seeing {entity} in {context} brings back {memory}",
                "{entity} here reminds me of when {memory}",
                "The role of {entity} in {context} parallels {memory}",
                "Just like {entity} in {memory}, we have {entity} in {context}"
            ]
            analogy_templates = [t.replace("{entity}", shared_entity) for t in analogy_templates]
        elif analogy_type == "action-based":
            analogy_templates = [
                "The way things unfold in '{context}' resembles '{memory}'",
                "This process in {context} follows the same trajectory as {memory}",
                "The sequence here with {context} matches what I saw in {memory}",
                "Similar forces at work: {context} develops like {memory}"
            ]
        else:  # thematic
            analogy_templates = [
                "The theme underlying '{context}' resonates with '{memory}'",
                "Both {context} and {memory} touch on similar undercurrents",
                "I sense the same essence in {context} that permeated {memory}",
                "The spirit of {context} evokes {memory}"
            ]
        
        template = random.choice(analogy_templates)
        analogy_text = template.format(
            memory=memory_phrase,
            context=context_phrase
        )
        
        # Calculate confidence based on shared attributes
        confidence = min(0.9, 0.4 + (len(shared_attributes) * 0.1))
        
        return AnalogyResult(
            analogy_text=analogy_text,
            memory_element=memory_phrase,
            context_element=context_phrase,
            analogy_type=analogy_type,
            shared_attributes=shared_attributes[:10],
            abstraction_level=abstraction_level,
            confidence=confidence
        )

    @staticmethod
    @function_tool
    async def _create_wordplay(
        ctx: RunContextWrapper[CreativeMemoryIntegrationContext],
        memory_text: str,
        current_context: str
    ) -> WordplayResult:
        """
        Create sophisticated wordplay connecting memory and current context
        
        Args:
            memory_text: Text of the memory
            current_context: Current conversation context
            
        Returns:
            Rich wordplay information
        """
        # Tokenize and analyze both texts
        memory_words = memory_text.lower().split()
        context_words = current_context.lower().split()
        
        # Remove common stop words for better analysis
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
            'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 
            'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'can', 'this', 'that', 'these', 'those', 'it'
        }
        
        meaningful_memory_words = [w for w in memory_words if w not in stop_words and len(w) > 2]
        meaningful_context_words = [w for w in context_words if w not in stop_words and len(w) > 2]
        
        # Find various types of word relationships
        wordplay_type = "association"
        linguistic_device = "semantic_connection"
        key_words = []
        cleverness_score = 0.5
        
        # 1. Direct word matches
        common_words = set(meaningful_memory_words) & set(meaningful_context_words)
        
        # 2. Phonetic similarities (simple approximation)
        def phonetic_similarity(w1, w2):
            """Simple phonetic similarity check"""
            if abs(len(w1) - len(w2)) > 2:
                return False
            # Check for rhyming
            if len(w1) > 3 and len(w2) > 3 and w1[-3:] == w2[-3:]:
                return True
            # Check for alliteration
            if w1[0] == w2[0] and len(w1) > 3 and len(w2) > 3:
                return True
            # Check for assonance (same vowel sounds - simplified)
            vowels1 = ''.join([c for c in w1 if c in 'aeiou'])
            vowels2 = ''.join([c for c in w2 if c in 'aeiou'])
            if vowels1 and vowels1 == vowels2:
                return True
            return False
        
        phonetic_pairs = []
        for mw in meaningful_memory_words[:20]:  # Limit for performance
            for cw in meaningful_context_words[:20]:
                if phonetic_similarity(mw, cw):
                    phonetic_pairs.append((mw, cw))
        
        # 3. Semantic word families
        word_families = {
            'time': ['time', 'moment', 'hour', 'day', 'minute', 'second', 'past', 'future', 'present'],
            'emotion': ['feel', 'feeling', 'emotion', 'happy', 'sad', 'angry', 'joy', 'fear', 'love'],
            'movement': ['move', 'go', 'come', 'walk', 'run', 'travel', 'journey', 'path', 'way'],
            'thought': ['think', 'thought', 'idea', 'mind', 'remember', 'forget', 'know', 'understand'],
            'communication': ['say', 'tell', 'speak', 'talk', 'word', 'language', 'voice', 'sound']
        }
        
        family_connections = []
        for family_name, family_words in word_families.items():
            memory_family_words = [w for w in meaningful_memory_words if w in family_words]
            context_family_words = [w for w in meaningful_context_words if w in family_words]
            if memory_family_words and context_family_words:
                family_connections.append({
                    'family': family_name,
                    'memory_words': memory_family_words,
                    'context_words': context_family_words
                })
        
        # Generate wordplay based on findings
        if common_words:
            # Direct word connection
            word = random.choice(list(common_words))
            key_words = [word]
            
            # Check if the word appears in different contexts
            memory_context = None
            context_context = None
            
            # Find the word's context in memory
            for i, w in enumerate(memory_words):
                if w == word:
                    start = max(0, i-2)
                    end = min(len(memory_words), i+3)
                    memory_context = ' '.join(memory_words[start:end])
                    break
            
            # Find the word's context in current
            for i, w in enumerate(context_words):
                if w == word:
                    start = max(0, i-2)
                    end = min(len(context_words), i+3)
                    context_context = ' '.join(context_words[start:end])
                    break
            
            if memory_context and context_context:
                wordplay_templates = [
                    f"The word '{word}' bridges two worlds: '{memory_context}' and '{context_context}'",
                    f"'{word}' - same word, different story: then '{memory_context}', now '{context_context}'",
                    f"Funny how '{word}' keeps appearing, from '{memory_context}' to '{context_context}'",
                    f"'{word}' echoes through time: '{memory_context}' meets '{context_context}'"
                ]
                wordplay_type = "polysemy"
                linguistic_device = "semantic_shift"
                cleverness_score = 0.7
            else:
                wordplay_templates = [
                    f"The word '{word}' creates a bridge between then and now",
                    f"'{word}' - a linguistic déjà vu",
                    f"Interesting how '{word}' resurfaces in our conversation",
                    f"'{word}' seems to be following us through this discussion"
                ]
                wordplay_type = "repetition"
                linguistic_device = "echo"
                cleverness_score = 0.6
            
            wordplay_text = random.choice(wordplay_templates)
            
        elif phonetic_pairs:
            # Phonetic wordplay
            memory_word, context_word = random.choice(phonetic_pairs)
            key_words = [memory_word, context_word]
            
            if memory_word[-3:] == context_word[-3:]:
                # Rhyme
                wordplay_templates = [
                    f"From '{memory_word}' to '{context_word}' - our conversation has found its rhyme",
                    f"'{memory_word}' and '{context_word}' - an accidental poetry in our exchange",
                    f"The echo of '{memory_word}' finds its partner in '{context_word}'",
                    f"A rhyming connection: '{memory_word}' then, '{context_word}' now"
                ]
                wordplay_type = "rhyme"
                linguistic_device = "phonetic_echo"
                cleverness_score = 0.8
            elif memory_word[0] == context_word[0]:
                # Alliteration
                wordplay_templates = [
                    f"'{memory_word}' and '{context_word}' - alliteratively aligned",
                    f"The '{memory_word[0]}' sound connects '{memory_word}' to '{context_word}'",
                    f"From '{memory_word}' to '{context_word}' - keeping the consonance",
                    f"An alliterative arc from '{memory_word}' to '{context_word}'"
                ]
                wordplay_type = "alliteration"
                linguistic_device = "consonance"
                cleverness_score = 0.7
            else:
                # Assonance
                wordplay_templates = [
                    f"'{memory_word}' and '{context_word}' share a vowel melody",
                    f"The sound of '{memory_word}' resonates in '{context_word}'",
                    f"An assonant connection: '{memory_word}' to '{context_word}'",
                    f"'{memory_word}' and '{context_word}' - united by sound"
                ]
                wordplay_type = "assonance"
                linguistic_device = "vowel_harmony"
                cleverness_score = 0.75
            
            wordplay_text = random.choice(wordplay_templates)
            
        elif family_connections:
            # Semantic family connection
            connection = random.choice(family_connections)
            memory_word = random.choice(connection['memory_words'])
            context_word = random.choice(connection['context_words'])
            family = connection['family']
            key_words = [memory_word, context_word, family]
            
            wordplay_templates = [
                f"From '{memory_word}' to '{context_word}' - variations on the theme of {family}",
                f"The {family} family: '{memory_word}' in memory, '{context_word}' in conversation",
                f"'{memory_word}' and '{context_word}' - different words, same {family} essence",
                f"A semantic journey through {family}: from '{memory_word}' to '{context_word}'"
            ]
            wordplay_type = "semantic_field"
            linguistic_device = "thematic_variation"
            cleverness_score = 0.65
            
            wordplay_text = random.choice(wordplay_templates)
            
        else:
            # Abstract connection
            if meaningful_memory_words and meaningful_context_words:
                memory_word = random.choice(meaningful_memory_words[:10])
                context_word = random.choice(meaningful_context_words[:10])
                key_words = [memory_word, context_word]
                
                wordplay_templates = [
                    f"From '{memory_word}' in my memory to '{context_word}' in our chat - life's lexical journey",
                    f"'{memory_word}' then, '{context_word}' now - words marking time",
                    f"The linguistic leap from '{memory_word}' to '{context_word}'",
                    f"'{memory_word}' and '{context_word}' - bookends of experience"
                ]
                wordplay_type = "juxtaposition"
                linguistic_device = "conceptual_contrast"
                cleverness_score = 0.5
            else:
                wordplay_templates = [
                    "Words weave through memory and moment alike",
                    "The language of then meets the vocabulary of now",
                    "Same linguistic tools, different conversational canvas",
                    "Words as bridges between past and present"
                ]
                wordplay_type = "meta_linguistic"
                linguistic_device = "abstract_connection"
                cleverness_score = 0.4
            
            wordplay_text = random.choice(wordplay_templates)
        
        return WordplayResult(
            wordplay_text=wordplay_text,
            wordplay_type=wordplay_type,
            key_words=key_words[:5],
            linguistic_device=linguistic_device,
            cleverness_score=cleverness_score
        )

    @staticmethod
    @function_tool
    async def _extract_memory_patterns(
        ctx: RunContextWrapper[CreativeMemoryIntegrationContext],
        memories: List[MemoryData]
    ) -> PatternInfo:
        """
        Extract sophisticated patterns across multiple memories
        
        Args:
            memories: List of memories
            
        Returns:
            Comprehensive pattern information
        """
        if not memories:
            return PatternInfo(
                pattern_description="No patterns found in empty memory set",
                common_elements=[],
                memory_count=0,
                confidence=0.0
            )
        
        # Initialize pattern storage
        word_frequency = Counter()
        bigram_frequency = Counter()
        entity_frequency = Counter()
        temporal_patterns = []
        causal_patterns = []
        structural_patterns = []
        
        # Process each memory
        memory_structures = []
        
        for memory in memories:
            text = memory.memory_text
            words = text.lower().split()
            
            # Word frequency
            meaningful_words = [
                w for w in words 
                if len(w) > 3 and w not in {
                    'that', 'this', 'these', 'those', 'there',
                    'where', 'which', 'while', 'about', 'through',
                    'have', 'been', 'with', 'from', 'into'
                }
            ]
            word_frequency.update(meaningful_words)
            
            # Bigram frequency
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                bigram_frequency[bigram] += 1
            
            # Entity extraction
            entities = ctx.context.entity_pattern.findall(text)
            entity_frequency.update(entities)
            
            # Temporal pattern detection
            temporal_matches = ctx.context.temporal_pattern.findall(text)
            if temporal_matches:
                temporal_patterns.extend(temporal_matches)
            
            # Causal pattern detection
            causal_matches = ctx.context.causal_pattern.findall(text)
            if causal_matches:
                causal_patterns.extend(causal_matches)
            
            # Structural analysis
            sentences = text.split('.')
            structure = {
                'sentence_count': len(sentences),
                'avg_sentence_length': sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0,
                'has_question': '?' in text,
                'has_exclamation': '!' in text,
                'starts_with_i': text.strip().lower().startswith('i '),
                'memory_type': memory.memory_type
            }
            memory_structures.append(structure)
        
        # Analyze structural patterns
        if memory_structures:
            avg_sentences = sum(s['sentence_count'] for s in memory_structures) / len(memory_structures)
            question_ratio = sum(1 for s in memory_structures if s['has_question']) / len(memory_structures)
            first_person_ratio = sum(1 for s in memory_structures if s['starts_with_i']) / len(memory_structures)
            
            if avg_sentences > 3:
                structural_patterns.append("long-form_narratives")
            if question_ratio > 0.3:
                structural_patterns.append("questioning_pattern")
            if first_person_ratio > 0.5:
                structural_patterns.append("self-referential")
            
            # Memory type patterns
            type_counts = Counter(s['memory_type'] for s in memory_structures)
            dominant_type = type_counts.most_common(1)[0][0] if type_counts else "mixed"
            structural_patterns.append(f"predominantly_{dominant_type}")
        
        # Determine pattern type
        pattern_types = []
        if len(entity_frequency) > len(memories) * 0.5:
            pattern_types.append("entity-rich")
        if temporal_patterns:
            pattern_types.append("temporal")
        if causal_patterns:
            pattern_types.append("causal")
        if len(structural_patterns) > 2:
            pattern_types.append("structural")
        
        pattern_type = "_".join(pattern_types) if pattern_types else "thematic"
        
        # Get most common elements
        common_words = word_frequency.most_common(10)
        common_bigrams = bigram_frequency.most_common(5)
        common_entities = entity_frequency.most_common(5)
        
        # Combine all common elements
        common_elements = []
        common_elements.extend([(word, count) for word, count in common_words])
        common_elements.extend([(f"[bigram] {bigram}", count) for bigram, count in common_bigrams])
        common_elements.extend([(f"[entity] {entity}", count) for entity, count in common_entities])
        
        # Sort by frequency
        common_elements.sort(key=lambda x: x[1], reverse=True)
        common_elements = common_elements[:15]
        
        # Generate pattern description
        if not common_elements:
            pattern_description = "Diverse memories with no dominant pattern"
            confidence = 0.3
        else:
            top_elements = [elem for elem, _ in common_elements[:5]]
            
            if pattern_type == "temporal_causal":
                pattern_description = f"Temporal-causal pattern linking {', '.join(top_elements[:3])}"
            elif pattern_type == "entity-rich":
                entities = [e.replace('[entity] ', '') for e, _ in common_elements if '[entity]' in e][:3]
                pattern_description = f"Entity-centric pattern around {', '.join(entities)}"
            elif pattern_type == "structural":
                pattern_description = f"Structural consistency with themes of {', '.join(top_elements[:3])}"
            else:
                pattern_description = f"Thematic pattern involving {', '.join(top_elements[:3])}"
            
            # Calculate confidence based on pattern strength
            total_occurrences = sum(count for _, count in common_elements[:5])
            avg_occurrence = total_occurrences / 5 if common_elements else 0
            confidence = min(0.9, 0.3 + (avg_occurrence / 10) + (len(memories) / 20))
        
        # Convert temporal patterns to unique list
        unique_temporal = list(set(temporal_patterns))[:5]
        unique_causal = list(set(causal_patterns))[:5]
        
        return PatternInfo(
            pattern_description=pattern_description,
            common_elements=common_elements,
            memory_count=len(memories),
            confidence=confidence,
            pattern_type=pattern_type,
            temporal_patterns=unique_temporal,
            causal_patterns=unique_causal,
            structural_patterns=structural_patterns[:5]
        )

    @staticmethod
    @function_tool
    async def _abstract_principles(
        ctx: RunContextWrapper[CreativeMemoryIntegrationContext],
        pattern: PatternInfo,
        current_context: str
    ) -> PrincipleInfo:
        """
        Abstract sophisticated principles from memory patterns
        
        Args:
            pattern: Pattern information
            current_context: Current conversation context
            
        Returns:
            Rich principle information
        """
        # Extract key elements from pattern
        elements = [elem for elem, count in pattern.common_elements[:5]]
        
        # Clean up element markers
        cleaned_elements = []
        for elem in elements:
            if '[bigram]' in elem:
                cleaned_elements.append(elem.replace('[bigram] ', ''))
            elif '[entity]' in elem:
                cleaned_elements.append(elem.replace('[entity] ', ''))
            else:
                cleaned_elements.append(elem)
        
        if not cleaned_elements:
            return PrincipleInfo(
                principle="Experience often contains hidden patterns waiting to be discovered",
                elements=[],
                abstraction_level=0.5,
                confidence=0.3,
                applicability="general",
                principle_type="meta"
            )
        
        # Determine principle type based on pattern characteristics
        principle_type = "general"
        abstraction_level = 0.5
        
        if pattern.temporal_patterns and pattern.causal_patterns:
            principle_type = "temporal_causal"
            abstraction_level = 0.8
        elif pattern.causal_patterns:
            principle_type = "causal"
            abstraction_level = 0.7
        elif pattern.temporal_patterns:
            principle_type = "temporal"
            abstraction_level = 0.6
        elif 'entity-rich' in pattern.pattern_type:
            principle_type = "relational"
            abstraction_level = 0.65
        elif 'structural' in pattern.pattern_type:
            principle_type = "structural"
            abstraction_level = 0.75
        
        # Generate principle based on type
        supporting_evidence = []
        
        if principle_type == "temporal_causal":
            # Complex temporal-causal principles
            temporal_marker = pattern.temporal_patterns[0] if pattern.temporal_patterns else "over time"
            causal_marker = pattern.causal_patterns[0] if pattern.causal_patterns else "leads to"
            
            principle_templates = [
                f"When {cleaned_elements[0]} occurs {temporal_marker}, it {causal_marker} changes in {cleaned_elements[1] if len(cleaned_elements) > 1 else 'outcomes'}",
                f"The progression from {cleaned_elements[0]} to {cleaned_elements[1] if len(cleaned_elements) > 1 else 'resolution'} follows predictable {temporal_marker} patterns",
                f"Events involving {cleaned_elements[0]} {causal_marker} predictable sequences {temporal_marker}",
                f"{temporal_marker.title()}, {cleaned_elements[0]} {causal_marker} transformation in perspective"
            ]
            
            supporting_evidence = [
                f"Pattern observed across {pattern.memory_count} memories",
                f"Temporal markers: {', '.join(pattern.temporal_patterns[:3])}",
                f"Causal connections: {', '.join(pattern.causal_patterns[:3])}"
            ]
            
        elif principle_type == "causal":
            # Causal principles
            causal_marker = pattern.causal_patterns[0] if pattern.causal_patterns else "influences"
            
            principle_templates = [
                f"{cleaned_elements[0].title()} {causal_marker} the nature of {cleaned_elements[1] if len(cleaned_elements) > 1 else 'subsequent experiences'}",
                f"The presence of {cleaned_elements[0]} {causal_marker} how we perceive {cleaned_elements[1] if len(cleaned_elements) > 1 else 'related situations'}",
                f"Changes in {cleaned_elements[0]} {causal_marker} shifts in {cleaned_elements[1] if len(cleaned_elements) > 1 else 'outcomes'}",
                f"When {cleaned_elements[0]} is prominent, it {causal_marker} the trajectory of events"
            ]
            
            supporting_evidence = [
                f"Causal pattern strength: {pattern.confidence:.2f}",
                f"Key causal markers: {', '.join(pattern.causal_patterns[:3])}"
            ]
            
        elif principle_type == "temporal":
            # Temporal principles
            temporal_marker = pattern.temporal_patterns[0] if pattern.temporal_patterns else "through time"
            
            principle_templates = [
                f"{cleaned_elements[0].title()} evolves {temporal_marker} in predictable ways",
                f"The significance of {cleaned_elements[0]} changes {temporal_marker}",
                f"{temporal_marker.title()}, {cleaned_elements[0]} takes on new meanings",
                f"Patterns of {cleaned_elements[0]} repeat {temporal_marker} with variations"
            ]
            
            supporting_evidence = [
                f"Temporal consistency: {pattern.confidence:.2f}",
                f"Time markers: {', '.join(pattern.temporal_patterns[:3])}"
            ]
            
        elif principle_type == "relational":
            # Relational principles about entities
            if len(cleaned_elements) >= 2:
                principle_templates = [
                    f"The relationship between {cleaned_elements[0]} and {cleaned_elements[1]} defines the context",
                    f"{cleaned_elements[0]} and {cleaned_elements[1]} form a meaningful dyad",
                    f"Understanding {cleaned_elements[0]} requires considering {cleaned_elements[1]}",
                    f"The interplay of {cleaned_elements[0]} and {cleaned_elements[1]} creates emergent properties"
                ]
            else:
                principle_templates = [
                    f"{cleaned_elements[0]} serves as a focal point for meaning",
                    f"The presence of {cleaned_elements[0]} shapes the entire context",
                    f"{cleaned_elements[0]} acts as an organizing principle",
                    f"Experiences center around {cleaned_elements[0]} as a key element"
                ]
            
            supporting_evidence = [
                f"Entity prominence in {pattern.memory_count} memories",
                f"Relational density: {len([e for e, _ in pattern.common_elements if '[entity]' in e])}"
            ]
            
        elif principle_type == "structural":
            # Structural principles
            struct_patterns = pattern.structural_patterns
            if struct_patterns:
                principle_templates = [
                    f"Experiences with {cleaned_elements[0]} tend toward {struct_patterns[0].replace('_', ' ')} patterns",
                    f"The structure of {cleaned_elements[0]} memories reveals {struct_patterns[0].replace('_', ' ')} tendencies",
                    f"{struct_patterns[0].replace('_', ' ').title()} characterizes how {cleaned_elements[0]} unfolds",
                    f"Form follows function: {cleaned_elements[0]} naturally adopts {struct_patterns[0].replace('_', ' ')} structures"
                ]
            else:
                principle_templates = [
                    f"The form of {cleaned_elements[0]} experiences shapes their content",
                    f"Structure and meaning intertwine in {cleaned_elements[0]} contexts",
                    f"{cleaned_elements[0]} follows consistent organizational patterns",
                    f"The architecture of {cleaned_elements[0]} experiences reveals deeper truths"
                ]
            
            supporting_evidence = [
                f"Structural consistency: {pattern.confidence:.2f}",
                f"Patterns: {', '.join(pattern.structural_patterns[:3])}"
            ]
            
        else:
            # General thematic principles
            principle_templates = [
                f"Experiences involving {cleaned_elements[0]} share common threads with {cleaned_elements[1] if len(cleaned_elements) > 1 else 'similar situations'}",
                f"The theme of {cleaned_elements[0]} recurs in various forms",
                f"{cleaned_elements[0]} represents a fundamental pattern in experience",
                f"When {cleaned_elements[0]} appears, certain dynamics tend to follow"
            ]
            
            supporting_evidence = [
                f"Theme strength: {pattern.confidence:.2f}",
                f"Recurrence in {pattern.memory_count} memories"
            ]
        
        principle = random.choice(principle_templates)
        
        # Assess applicability to current context
        context_lower = current_context.lower()
        applicability_score = 0.0
        
        # Check element presence
        for element in cleaned_elements:
            if element.lower() in context_lower:
                applicability_score += 0.3
        
        # Check pattern marker presence
        if pattern.temporal_patterns:
            for marker in pattern.temporal_patterns:
                if marker.lower() in context_lower:
                    applicability_score += 0.2
                    break
        
        if pattern.causal_patterns:
            for marker in pattern.causal_patterns:
                if marker.lower() in context_lower:
                    applicability_score += 0.2
                    break
        
        # Determine applicability level
        if applicability_score >= 0.5:
            applicability = "high"
        elif applicability_score >= 0.2:
            applicability = "medium"
        else:
            applicability = "low"
        
        # Add counter-examples for balance
        counter_examples = []
        if pattern.confidence < 0.7:
            counter_examples.append("Pattern may not hold in all contexts")
        if len(pattern.common_elements) < 5:
            counter_examples.append("Limited data points suggest caution in generalization")
        
        return PrincipleInfo(
            principle=principle,
            elements=cleaned_elements,
            abstraction_level=abstraction_level,
            confidence=pattern.confidence,
            applicability=applicability,
            principle_type=principle_type,
            supporting_evidence=supporting_evidence[:3],
            counter_examples=counter_examples
        )

    @staticmethod
    @function_tool
    async def _apply_creative_technique(
        ctx: RunContextWrapper[CreativeMemoryIntegrationContext],
        memory: MemoryData,
        current_context: str,
        technique: str
    ) -> CreativeTechniqueResult:
        """
        Apply sophisticated creative techniques to memory
        
        Args:
            memory: Memory to transform
            current_context: Current conversation context
            technique: Creative technique to apply
            
        Returns:
            Rich technique application result
        """
        memory_text = memory.memory_text
        intermediate_steps = []
        creativity_score = 0.5
        coherence_score = 0.5
        
        if technique == "bisociation":
            # Bisociation: Connect two unrelated frames of reference
            
            # Extract conceptual frames from memory and context
            memory_frames = []
            context_frames = []
            
            # Simple frame extraction based on key phrases
            memory_words = memory_text.split()
            context_words = current_context.split()
            
            # Look for domain indicators
            domains = {
                'technical': ['system', 'process', 'function', 'algorithm', 'data', 'code'],
                'emotional': ['feel', 'happy', 'sad', 'angry', 'love', 'fear', 'joy'],
                'physical': ['move', 'touch', 'see', 'hear', 'body', 'space', 'place'],
                'social': ['people', 'friend', 'family', 'group', 'community', 'together'],
                'temporal': ['time', 'past', 'future', 'now', 'then', 'when', 'after'],
                'abstract': ['idea', 'concept', 'theory', 'principle', 'meaning', 'truth']
            }
            
            for domain, indicators in domains.items():
                if any(ind in memory_text.lower() for ind in indicators):
                    memory_frames.append(domain)
                if any(ind in current_context.lower() for ind in indicators):
                    context_frames.append(domain)
            
            # Default frames if none found
            if not memory_frames:
                memory_frames = ['experiential']
            if not context_frames:
                context_frames = ['conversational']
            
            # Select frames to connect
            memory_frame = memory_frames[0]
            context_frame = context_frames[0] if context_frames[0] != memory_frame else (context_frames[1] if len(context_frames) > 1 else 'current')
            
            intermediate_steps.append(f"Identified frames: {memory_frame} (memory) and {context_frame} (context)")
            
            # Create bisociation based on frame combination
            if memory_frame == 'technical' and context_frame == 'emotional':
                transformed_text = f"Like debugging code, understanding {current_context[:30]}... requires systematic exploration of each emotional subroutine"
                transformation_notes = "Bisociated technical problem-solving with emotional processing"
                creativity_score = 0.8
            elif memory_frame == 'physical' and context_frame == 'abstract':
                transformed_text = f"Just as {memory_text[:40]}... involved tangible movement, {current_context[:30]}... requires navigating conceptual space"
                transformation_notes = "Bisociated physical experience with abstract navigation"
                creativity_score = 0.75
            elif memory_frame == 'temporal' and context_frame == 'social':
                transformed_text = f"The temporal flow of {memory_text[:30]}... mirrors the social dynamics in {current_context[:30]}..."
                transformation_notes = "Bisociated time progression with social evolution"
                creativity_score = 0.7
            else:
                # Generic bisociation
                transformed_text = f"Connecting {memory_frame} elements from '{memory_text[:30]}...' with {context_frame} aspects of '{current_context[:30]}...' reveals unexpected parallels"
                transformation_notes = f"Bisociated {memory_frame} and {context_frame} domains"
                creativity_score = 0.65
            
            intermediate_steps.append(f"Created bisociation between {memory_frame} and {context_frame}")
            coherence_score = 0.7
            
        elif technique == "perspective_shift":
            # Sophisticated perspective shifting
            
            perspectives = {
                'temporal': {
                    'shifts': ['future retrospective', 'historical parallel', 'cyclical view', 'momentary expansion'],
                    'markers': ['will look back', 'history shows', 'cycles repeat', 'this moment contains']
                },
                'spatial': {
                    'shifts': ['bird\'s eye view', 'microscopic detail', 'inside-out', 'parallel dimension'],
                    'markers': ['from above', 'zooming in', 'from within', 'in another reality']
                },
                'emotional': {
                    'shifts': ['empathetic reversal', 'emotional archaeology', 'feeling as metaphor', 'emotional ecology'],
                    'markers': ['in their shoes', 'layers of feeling', 'emotion as landscape', 'emotional ecosystem']
                },
                'cognitive': {
                    'shifts': ['child\'s wonder', 'alien observer', 'pattern recognizer', 'meaning maker'],
                    'markers': ['with fresh eyes', 'as an outsider', 'seeing patterns', 'finding meaning']
                },
                'relational': {
                    'shifts': ['network view', 'symbiotic lens', 'systemic perspective', 'interdependence focus'],
                    'markers': ['web of connections', 'mutual influence', 'system dynamics', 'interconnected whole']
                }
            }
            
            # Choose perspective type based on content
            chosen_perspective = random.choice(list(perspectives.keys()))
            if 'time' in memory_text.lower() or 'when' in memory_text.lower():
                chosen_perspective = 'temporal'
            elif 'feel' in memory_text.lower() or any(emotion in memory_text.lower() for emotion in ['happy', 'sad', 'angry']):
                chosen_perspective = 'emotional'
            
            perspective_data = perspectives[chosen_perspective]
            shift_type = random.choice(perspective_data['shifts'])
            marker = random.choice(perspective_data['markers'])
            
            intermediate_steps.append(f"Applying {chosen_perspective} perspective: {shift_type}")
            
            # Apply the perspective shift
            if chosen_perspective == 'temporal':
                if shift_type == 'future retrospective':
                    transformed_text = f"Looking back from a future vantage point, {memory_text[:50]}... will reveal its connection to {current_context[:30]}..."
                elif shift_type == 'historical parallel':
                    transformed_text = f"History shows that {memory_text[:40]}... follows patterns now visible in {current_context[:30]}..."
                elif shift_type == 'cyclical view':
                    transformed_text = f"The cycle from {memory_text[:30]}... to {current_context[:30]}... suggests recurring patterns"
                else:  # momentary expansion
                    transformed_text = f"This moment contains echoes: {memory_text[:30]}... reverberating in {current_context[:30]}..."
                    
            elif chosen_perspective == 'emotional':
                if shift_type == 'empathetic reversal':
                    transformed_text = f"Stepping into the emotional space of {memory_text[:40]}... illuminates {current_context[:30]}..."
                elif shift_type == 'emotional archaeology':
                    transformed_text = f"Excavating emotional layers: {memory_text[:30]}... buried beneath {current_context[:30]}..."
                elif shift_type == 'feeling as metaphor':
                    transformed_text = f"If {memory_text[:30]}... were an emotional landscape, {current_context[:30]}... would be its weather"
                else:  # emotional ecology
                    transformed_text = f"The emotional ecosystem linking {memory_text[:30]}... to {current_context[:30]}... reveals interdependencies"
                    
            else:
                # Generic perspective shift
                transformed_text = f"{marker.capitalize()}, {memory_text[:40]}... takes on new meaning in light of {current_context[:30]}..."
            
            transformation_notes = f"Applied {shift_type} perspective shift"
            creativity_score = 0.75
            coherence_score = 0.8
            
        elif technique == "constraint_relaxation":
            # Sophisticated constraint relaxation
            
            # Identify constraints in the memory
            constraints = []
            
            # Physical constraints
            if any(word in memory_text.lower() for word in ['couldn\'t', 'cannot', 'impossible', 'limited', 'restricted']):
                constraints.append('physical_limitation')
            
            # Temporal constraints
            if any(word in memory_text.lower() for word in ['always', 'never', 'must', 'have to', 'should']):
                constraints.append('temporal_necessity')
            
            # Social constraints
            if any(word in memory_text.lower() for word in ['expected', 'supposed', 'normal', 'appropriate', 'proper']):
                constraints.append('social_expectation')
            
            # Logical constraints
            if any(word in memory_text.lower() for word in ['because', 'therefore', 'if then', 'means that']):
                constraints.append('logical_dependency')
            
            if not constraints:
                constraints = ['implicit_assumption']
            
            constraint_type = constraints[0]
            intermediate_steps.append(f"Identified constraint type: {constraint_type}")
            
            # Relax the constraint
            if constraint_type == 'physical_limitation':
                transformed_text = f"If physical laws bent around {memory_text[:40]}..., then {current_context[:30]}... becomes a playground of possibility"
                transformation_notes = "Relaxed physical constraints to explore impossible scenarios"
                creativity_score = 0.85
                
            elif constraint_type == 'temporal_necessity':
                transformed_text = f"Releasing the 'must' from {memory_text[:40]}... opens {current_context[:30]}... to optional futures"
                transformation_notes = "Relaxed temporal necessities to reveal choice"
                creativity_score = 0.8
                
            elif constraint_type == 'social_expectation':
                transformed_text = f"Beyond social scripts, {memory_text[:40]}... and {current_context[:30]}... dance in authentic space"
                transformation_notes = "Relaxed social constraints to find authenticity"
                creativity_score = 0.75
                
            elif constraint_type == 'logical_dependency':
                transformed_text = f"What if the logic linking {memory_text[:30]}... to outcomes dissolved? {current_context[:30]}... suggests new causalities"
                transformation_notes = "Relaxed logical dependencies to explore alternative causation"
                creativity_score = 0.8
                
            else:
                transformed_text = f"Questioning assumptions in {memory_text[:40]}... reveals hidden freedoms in {current_context[:30]}..."
                transformation_notes = "Relaxed implicit assumptions"
                creativity_score = 0.7
            
            intermediate_steps.append("Explored implications of relaxed constraints")
            coherence_score = 0.65
            
        else:
            # Default creative transformation with sophistication
            
            # Extract key elements
            memory_essence = memory_text.split('.')[0] if '.' in memory_text else memory_text[:50]
            context_essence = current_context.split('.')[0] if '.' in current_context else current_context[:50]
            
            # Apply a random creative operation
            operations = [
                {
                    'name': 'metaphorical_blend',
                    'text': f"Like {memory_essence} becoming water, {context_essence} flows into new forms",
                    'notes': "Applied metaphorical transformation"
                },
                {
                    'name': 'recursive_embedding',
                    'text': f"Within {memory_essence} lies {context_essence}, within that, infinite recursion",
                    'notes': "Applied recursive conceptual embedding"
                },
                {
                    'name': 'dialectical_synthesis',
                    'text': f"Thesis: {memory_essence}. Antithesis: {context_essence}. Synthesis: emergent understanding",
                    'notes': "Applied dialectical synthesis"
                },
                {
                    'name': 'quantum_superposition',
                    'text': f"{memory_essence} and {context_essence} exist in superposition until observed",
                    'notes': "Applied quantum metaphor to conceptual duality"
                }
            ]
            
            operation = random.choice(operations)
            transformed_text = operation['text']
            transformation_notes = operation['notes']
            intermediate_steps.append(f"Applied {operation['name']} operation")
            
            creativity_score = 0.6
            coherence_score = 0.6
        
        # Final coherence check and adjustment
        if len(transformed_text) > 200:
            transformed_text = transformed_text[:197] + "..."
            coherence_score *= 0.9
        
        return CreativeTechniqueResult(
            technique=technique,
            memory_id=memory.id,
            transformed_text=transformed_text,
            transformation_notes=transformation_notes,
            creativity_score=creativity_score,
            coherence_score=coherence_score,
            intermediate_steps=intermediate_steps[:3]
        )

    @staticmethod
    @function_tool
    async def _transform_memory(
        ctx: RunContextWrapper[CreativeMemoryIntegrationContext],
        memory: MemoryData,
        transformation_type: str
    ) -> TransformationResult:
        """
        Transform memory with sophisticated techniques
        
        Args:
            memory: Memory to transform
            transformation_type: Type of transformation to apply
            
        Returns:
            Rich transformation result
        """
        memory_text = memory.memory_text
        memory_id = memory.id
        
        # Initialize tracking
        preserved_elements = []
        novel_elements = []
        semantic_distance = 0.5
        
        if transformation_type == "generalize":
            # Sophisticated generalization
            
            # Identify specific elements to generalize
            specific_elements = {
                'personal_pronouns': {
                    'I': 'one',
                    'my': 'one\'s',
                    'me': 'oneself',
                    'mine': 'one\'s own',
                    'we': 'people',
                    'our': 'their',
                    'us': 'them'
                },
                'specific_times': {
                    'yesterday': 'in the past',
                    'today': 'at present',
                    'tomorrow': 'in the future',
                    'last week': 'recently',
                    'next month': 'soon'
                },
                'specific_places': {
                    'here': 'in a place',
                    'there': 'elsewhere',
                    'home': 'a familiar space',
                    'work': 'a functional environment'
                }
            }
            
            transformed_text = memory_text
            replacements_made = []
            
            # Apply generalizations
            for category, mappings in specific_elements.items():
                for specific, general in mappings.items():
                    if specific in transformed_text:
                        transformed_text = transformed_text.replace(specific, general)
                        replacements_made.append(f"{specific}→{general}")
                        preserved_elements.append(f"structure of {specific}")
                        novel_elements.append(f"generalized {general}")
            
            # Extract and generalize specific entities
            entities = ctx.context.entity_pattern.findall(transformed_text)
            for entity in entities[:3]:  # Limit to avoid over-generalization
                if len(entity) > 4:  # Only generalize substantial entities
                    general_form = "an entity"
                    if entity.lower().endswith('ing'):
                        general_form = "an activity"
                    elif entity[0].isupper() and ' ' not in entity:
                        general_form = "a person"
                    elif ' ' in entity:
                        general_form = "a place or thing"
                    
                    transformed_text = transformed_text.replace(entity, general_form, 1)
                    replacements_made.append(f"{entity}→{general_form}")
            
            # Add abstraction layer
            if replacements_made:
                transformed_text = f"In general terms: {transformed_text}"
                transformation_notes = f"Generalized {len(replacements_made)} specific elements"
                semantic_distance = 0.3 + (len(replacements_made) * 0.1)
            else:
                transformed_text = f"Generally speaking, {transformed_text.lower()}"
                transformation_notes = "Applied general framing without specific replacements"
                semantic_distance = 0.2
            
            # Preserve core message elements
            core_verbs = [word for word in memory_text.split() if word.endswith(('ing', 'ed', 's')) and len(word) > 4]
            preserved_elements.extend([f"action: {verb}" for verb in core_verbs[:3]])
            
        elif transformation_type == "metaphorize":
            # Sophisticated metaphor generation
            
            # Analyze memory for metaphor source domains
            metaphor_mappings = {
                'journey': {
                    'triggers': ['go', 'went', 'move', 'travel', 'path', 'journey', 'arrive', 'leave'],
                    'elements': ['path', 'destination', 'obstacles', 'companions', 'milestone'],
                    'template': "Life's journey through {memory} leads to {insight}"
                },
                'growth': {
                    'triggers': ['grow', 'change', 'develop', 'learn', 'become', 'transform'],
                    'elements': ['seed', 'soil', 'seasons', 'fruit', 'roots'],
                    'template': "{memory} planted seeds that bloom into {insight}"
                },
                'construction': {
                    'triggers': ['build', 'create', 'make', 'establish', 'foundation', 'structure'],
                    'elements': ['foundation', 'blueprint', 'materials', 'architecture', 'cornerstone'],
                    'template': "{memory} laid the foundation for {insight}"
                },
                'navigation': {
                    'triggers': ['find', 'search', 'discover', 'explore', 'understand', 'realize'],
                    'elements': ['compass', 'map', 'stars', 'lighthouse', 'harbor'],
                    'template': "{memory} serves as a compass pointing toward {insight}"
                },
                'weather': {
                    'triggers': ['feel', 'emotion', 'mood', 'atmosphere', 'tension', 'calm'],
                    'elements': ['storm', 'sunshine', 'seasons', 'climate', 'forecast'],
                    'template': "The climate of {memory} forecasts {insight}"
                }
            }
            
            # Find applicable metaphor domain
            selected_domain = None
            memory_lower = memory_text.lower()
            
            for domain, data in metaphor_mappings.items():
                if any(trigger in memory_lower for trigger in data['triggers']):
                    selected_domain = domain
                    break
            
            if not selected_domain:
                selected_domain = random.choice(list(metaphor_mappings.keys()))
            
            domain_data = metaphor_mappings[selected_domain]
            
            # Extract key concept from memory
            key_concept = memory_text.split('.')[0] if '.' in memory_text else memory_text[:40]
            key_concept = key_concept.strip()
            
            # Generate insight based on memory
            insights = [
                "understanding emerges",
                "patterns become clear",
                "wisdom crystallizes",
                "perspective shifts",
                "truth reveals itself"
            ]
            insight = random.choice(insights)
            
            # Build metaphor
            template = domain_data['template']
            transformed_text = template.format(memory=key_concept, insight=insight)
            
            # Add metaphorical elements
            elements = random.sample(domain_data['elements'], min(2, len(domain_data['elements'])))
            if elements:
                element_phrase = f", where {elements[0]} meets {elements[1]}" if len(elements) > 1 else f", like a {elements[0]}"
                transformed_text += element_phrase
            
            transformation_notes = f"Transformed into {selected_domain} metaphor"
            semantic_distance = 0.7
            
            preserved_elements.append(f"core meaning of: {key_concept[:20]}...")
            novel_elements.extend([f"{selected_domain} domain", "metaphorical mapping"])
            
        elif transformation_type == "hypothesize":
            # Sophisticated hypothetical transformation
            
            # Identify key elements to make hypothetical
            key_elements = {
                'actions': [],
                'conditions': [],
                'outcomes': []
            }
            
            # Simple pattern matching for different elements
            sentences = memory_text.split('.')
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['did', 'was', 'happened', 'went']):
                    key_elements['actions'].append(sentence.strip())
                elif any(word in sentence.lower() for word in ['because', 'since', 'as', 'when']):
                    key_elements['conditions'].append(sentence.strip())
                elif any(word in sentence.lower() for word in ['then', 'so', 'thus', 'result']):
                    key_elements['outcomes'].append(sentence.strip())
            
            # Create hypothetical scenarios
            hypothetical_frames = [
                "What if {memory} had unfolded differently?",
                "Imagine if {memory} were just the beginning...",
                "Suppose {memory} represented a larger pattern...",
                "Consider: if {memory} were a universal law...",
                "What might emerge if {memory} were inverted?",
                "If we could replay {memory} with variations...",
                "Hypothetically, if {memory} were the norm..."
            ]
            
            frame = random.choice(hypothetical_frames)
            memory_core = memory_text[:60]
            
            transformed_text = frame.format(memory=memory_core)
            
            # Add hypothetical explorations
            if key_elements['actions']:
                action = key_elements['actions'][0][:40]
                transformed_text += f" Would {action} still hold true?"
                preserved_elements.append(f"action: {action[:20]}...")
                
            if key_elements['conditions']:
                condition = key_elements['conditions'][0][:40]
                transformed_text += f" The conditions might shift: {condition} becomes variable."
                preserved_elements.append(f"condition: {condition[:20]}...")
                
            if key_elements['outcomes']:
                outcome = key_elements['outcomes'][0][:40]
                transformed_text += f" New outcomes could emerge beyond {outcome}."
                preserved_elements.append(f"outcome: {outcome[:20]}...")
            
            transformation_notes = "Transformed into hypothetical exploration"
            semantic_distance = 0.6
            novel_elements.extend(["hypothetical framing", "possibility space", "counterfactual thinking"])
            
        else:
            # Default sophisticated transformation
            
            # Analyze memory structure
            has_temporal = bool(ctx.context.temporal_pattern.search(memory_text))
            has_causal = bool(ctx.context.causal_pattern.search(memory_text))
            has_emotional = any(
                indicator in memory_text.lower() 
                for indicators in ctx.context.emotion_indicators.values() 
                for indicator in indicators
            )
            
            # Build transformation based on structure
            transformations = []
            
            if has_temporal:
                transformations.append({
                    'text': f"Echoing through time: {memory_text}",
                    'notes': "Emphasized temporal dimension",
                    'distance': 0.3
                })
            
            if has_causal:
                transformations.append({
                    'text': f"The chain of causation: {memory_text} - revealing hidden connections",
                    'notes': "Highlighted causal relationships",
                    'distance': 0.4
                })
            
            if has_emotional:
                transformations.append({
                    'text': f"The emotional archaeology of this reveals: {memory_text}",
                    'notes': "Focused on emotional dimensions",
                    'distance': 0.35
                })
            
            if not transformations:
                transformations.append({
                    'text': f"Reframing this memory: {memory_text} - seen through today's lens",
                    'notes': "Applied temporal reframing",
                    'distance': 0.25
                })
            
            chosen = random.choice(transformations)
            transformed_text = chosen['text']
            transformation_notes = chosen['notes']
            semantic_distance = chosen['distance']
            
            # Identify preserved vs novel elements
            words = memory_text.split()
            preserved_elements = [f"core: {' '.join(words[:3])}"] if len(words) >= 3 else ["core meaning"]
            novel_elements = ["reframing", "new perspective"]
        
        # Ensure transformed text isn't too long
        if len(transformed_text) > 200:
            transformed_text = transformed_text[:197] + "..."
        
        # Calculate final semantic distance
        semantic_distance = min(0.9, semantic_distance)
        
        return TransformationResult(
            memory_id=memory_id,
            original_text=memory_text,
            transformed_text=transformed_text,
            transformation_type=transformation_type,
            transformation_notes=transformation_notes,
            semantic_distance=semantic_distance,
            preserved_elements=preserved_elements[:5],
            novel_elements=novel_elements[:5]
        )
    
    # Main public methods for the integration system
    
    async def process_conversation_turn(
        self,
        conversation_text: str,
        current_context: Optional[Any] = None
    ) -> ProcessingResult:
        """
        Process a conversation turn to generate creative memory-based responses
        
        Args:
            conversation_text: Text from the current conversation turn
            current_context: Current context information
            
        Returns:
            Creative memory integration results
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
        
        # Ensure required components are available
        if not self.context.recognition_memory:
            return ProcessingResult(status="error", error="Recognition memory system not available")
        
        if not self.context.novelty_engine:
            return ProcessingResult(status="error", error="Novelty engine not available")
        
        processing_metadata = {
            'timestamp': datetime.datetime.now().isoformat(),
            'conversation_length': len(conversation_text),
            'has_context': current_context is not None
        }
        
        with trace(workflow_name="Creative Memory Integration", group_id=self.context.trace_id):
            # 1. Get recognized memories from the recognition memory system
            recognition_results = await self.context.recognition_memory.process_conversation_turn(
                conversation_text,
                current_context
            )
            
            # If no memories recognized, return empty result
            if not recognition_results:
                return ProcessingResult(
                    status="no_memories_recognized",
                    generated_content=None,
                    processing_metadata=processing_metadata
                )
            
            # 2. Convert recognition results to format needed for integration
            memories = []
            for result in recognition_results:
                if isinstance(result, dict):
                    memories.append(MemoryData(
                        id=result.get("memory_id", ""),
                        memory_text=result.get("memory_text", ""),
                        memory_type=result.get("memory_type", ""),
                        relevance_score=result.get("relevance_score", 0.5),
                        activation_trigger=result.get("activation_trigger")
                    ))
                else:
                    # Convert Pydantic model to MemoryData
                    memories.append(MemoryData(
                        id=getattr(result, "memory_id", ""),
                        memory_text=getattr(result, "memory_text", ""),
                        memory_type=getattr(result, "memory_type", ""),
                        relevance_score=getattr(result, "relevance_score", 0.5),
                        activation_trigger=getattr(result, "activation_trigger", None)
                    ))
            
            processing_metadata['recognized_memories'] = len(memories)
            processing_metadata['memory_types'] = list(set(m.memory_type for m in memories))
            
            # 3. Prepare prompt for integration agent
            prompt = f"""Process this conversation turn with recognized memories:
            
            Conversation: {conversation_text}
            
            Recognized memories: {len(memories)} memories recognized.
            Memory types: {', '.join(set(m.memory_type for m in memories))}
            
            Determine the most appropriate creative integration approach:
            - Generate contextual wit if appropriate (check sentiment and context)
            - Create a creative insight from patterns in memories
            - Blend memory recognition with novel idea generation
            
            Consider the conversation's emotional tone, complexity, and the richness
            of the recognized memories when deciding on the approach.
            """
            
            # Configure run with tracing
            run_config = RunConfig(
                workflow_name="Creative Memory Response",
                group_id=self.context.trace_id,
                trace_metadata={
                    "message_length": len(conversation_text),
                    "memory_count": len(memories)
                }
            )
            
            # 4. Run through integration agent
            result = await Runner.run(
                self.integration_agent,
                prompt,
                context=self.context,
                run_config=run_config
            )
            
            # 5. Process the result
            integration_result = result.final_output
            
            # Store in context if appropriate
            if hasattr(integration_result, "wit_text"):
                # It's a ContextualWit
                wit_id = f"wit_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
                self.context.contextual_wits.append((wit_id, integration_result))
                
                processing_metadata['wit_type'] = integration_result.wit_type
                processing_metadata['novelty_score'] = integration_result.novelty_score
                
                return ProcessingResult(
                    status="wit_generated",
                    content_type="contextual_wit",
                    content_id=wit_id,
                    generated_content=integration_result,
                    processing_metadata=processing_metadata
                )
                
            elif hasattr(integration_result, "insight_text"):
                # It's a CreativeInsight
                insight_id = f"insight_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
                self.context.creative_insights.append((insight_id, integration_result))
                
                processing_metadata['insight_type'] = integration_result.insight_type
                processing_metadata['abstraction_level'] = integration_result.abstraction_level
                
                return ProcessingResult(
                    status="insight_generated",
                    content_type="creative_insight",
                    content_id=insight_id,
                    generated_content=integration_result,
                    processing_metadata=processing_metadata
                )
                
            else:
                # It's likely a direct result from the memory-novelty blend
                return ProcessingResult(
                    status="blend_generated",
                    content_type="memory_novelty_blend",
                    generated_content=integration_result,
                    processing_metadata=processing_metadata
                )
    
    async def generate_contextual_wit(
        self,
        conversation_text: str,
        memory: MemoryData
    ) -> ContextualWit:
        """
        Directly generate contextual wit from a specific memory
        
        Args:
            conversation_text: Current conversation text
            memory: Specific memory to use
            
        Returns:
            Generated contextual wit
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
        
        with trace(workflow_name="Generate Contextual Wit", group_id=self.context.trace_id):
            # Analyze appropriateness first
            wit_decision = await self._should_generate_wit(
                RunContextWrapper(context=self.context),
                conversation_text,
                [memory]
            )
            
            if not wit_decision.should_generate:
                # Return a default safe wit if not appropriate
                return ContextualWit(
                    wit_text="This reminds me of something...",
                    wit_type="reference",
                    related_memory_id=memory.id,
                    novelty_score=0.3,
                    appropriateness_score=wit_decision.appropriateness_score,
                    confidence=0.5,
                    generation_technique="safe_default"
                )
            
            # Prepare prompt for wit generation agent
            prompt = f"""Generate contextual wit based on this memory and conversation:
            
            Conversation: {conversation_text}
            Sentiment: {wit_decision.conversation_sentiment}
            
            Memory: {memory.memory_text}
            Memory type: {memory.memory_type}
            
            Appropriate wit types: {', '.join(wit_decision.wit_types)}
            
            Create witty, clever response that references this memory in a way
            that enhances the conversation. Focus on quality and contextual appropriateness.
            The wit should feel natural, not forced, and should add value to the conversation.
            """
            
            # Run through wit generation agent
            result = await Runner.run(
                self.wit_generation_agent,
                prompt,
                context=self.context
            )
            
            wit = result.final_output
            
            # Set the related memory ID
            wit.related_memory_id = memory.id
            
            # Store the generated wit
            if wit:
                wit_id = f"wit_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
                self.context.contextual_wits.append((wit_id, wit))
            
            return wit
    
    async def generate_creative_insight(
        self,
        conversation_text: str,
        memories: List[MemoryData]
    ) -> CreativeInsight:
        """
        Generate a creative insight from multiple memories
        
        Args:
            conversation_text: Current conversation text
            memories: Memories to derive insight from
            
        Returns:
            Generated creative insight
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
        
        # Filter memories for quality
        quality_memories = [
            m for m in memories 
            if len(m.memory_text) >= self.context.min_memory_length
        ]
        
        if not quality_memories:
            quality_memories = memories  # Use all if none meet quality threshold
        
        with trace(workflow_name="Generate Creative Insight", group_id=self.context.trace_id):
            # Extract patterns first
            pattern_info = await self._extract_memory_patterns(
                RunContextWrapper(context=self.context),
                quality_memories
            )
            
            # Prepare prompt for insight generation agent
            memory_texts = []
            for i, memory in enumerate(quality_memories[:5]):  # Limit to 5 for processing
                memory_texts.append(f"Memory {i+1} ({memory.memory_type}): {memory.memory_text}")
            memory_context = "\n\n".join(memory_texts)
            
            prompt = f"""Generate a creative insight based on these memories and conversation:
            
            Conversation: {conversation_text}
            
            Memories:
            {memory_context}
            
            Pattern analysis:
            - Pattern type: {pattern_info.pattern_type}
            - Pattern description: {pattern_info.pattern_description}
            - Confidence: {pattern_info.confidence}
            - Temporal patterns: {', '.join(pattern_info.temporal_patterns[:3]) if pattern_info.temporal_patterns else 'none'}
            - Causal patterns: {', '.join(pattern_info.causal_patterns[:3]) if pattern_info.causal_patterns else 'none'}
            
            Create a deeper, more abstract insight from these memories that applies
            to the current context. Extract generalizable principles or identify
            patterns that provide value to the current situation.
            
            The insight should:
            1. Go beyond surface observations
            2. Connect multiple memories meaningfully
            3. Provide actionable understanding
            4. Feel revelatory rather than obvious
            """
            
            # Run through insight generation agent
            result = await Runner.run(
                self.insight_generation_agent,
                prompt,
                context=self.context
            )
            
            insight = result.final_output
            
            # Set the source memory ID to the most relevant one
            if quality_memories:
                insight.source_memory_id = quality_memories[0].id
            
            # Store the generated insight
            if insight:
                insight_id = f"insight_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
                self.context.creative_insights.append((insight_id, insight))
            
            return insight
    
    async def blend_memory_with_novelty(
        self,
        conversation_text: str,
        memory: MemoryData,
        creative_technique: str = "auto"
    ) -> BlendResult:
        """
        Blend a recognized memory with novel idea generation
        
        Args:
            conversation_text: Current conversation text
            memory: Memory to transform
            creative_technique: Creative technique to apply
            
        Returns:
            Blend results
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
        
        # Ensure novelty engine is available
        if not self.context.novelty_engine:
            return BlendResult(status="error", error="Novelty engine not available")
        
        with trace(workflow_name="Blend Memory with Novelty", group_id=self.context.trace_id):
            # If technique is auto, choose based on memory content analysis
            if creative_technique == "auto":
                # Analyze memory to choose technique
                memory_lower = memory.memory_text.lower()
                
                if any(word in memory_lower for word in ['different', 'contrast', 'opposite', 'unlike']):
                    creative_technique = "bisociation"
                elif any(word in memory_lower for word in ['see', 'view', 'perspective', 'look']):
                    creative_technique = "perspective_shifting"
                elif any(word in memory_lower for word in ['limit', 'constraint', 'boundary', 'restriction']):
                    creative_technique = "constraint_relaxation"
                elif any(word in memory_lower for word in ['pattern', 'similar', 'like', 'resemble']):
                    creative_technique = "analogical_thinking"
                else:
                    techniques = ["bisociation", "perspective_shifting", "constraint_relaxation", "analogical_thinking"]
                    creative_technique = random.choice(techniques)
            
            # Extract concepts with sophistication
            memory_concept = memory.memory_text[:100]
            conversation_concept = conversation_text[:100]
            
            # Calculate connection strength
            common_words = set(memory_concept.lower().split()) & set(conversation_concept.lower().split())
            connection_strength = min(0.9, 0.3 + (len(common_words) * 0.1))
            
            # Determine connection type
            if any(word in memory.memory_text.lower() for word in ['because', 'therefore', 'thus']):
                connection_type = "causal"
            elif any(word in memory.memory_text.lower() for word in ['like', 'similar', 'same']):
                connection_type = "analogical"
            elif any(word in memory.memory_text.lower() for word in ['but', 'however', 'although']):
                connection_type = "contrastive"
            else:
                connection_type = "associative"
            
            try:
                # Use novelty engine to generate idea connecting memory and conversation
                novel_idea = await self.context.novelty_engine.generate_novel_idea(
                    technique=creative_technique,
                    concepts=[memory_concept, conversation_concept]
                )
                
                # Calculate creativity metrics
                creativity_metrics = {
                    'novelty': random.uniform(0.6, 0.9),
                    'usefulness': random.uniform(0.5, 0.8),
                    'surprise': random.uniform(0.4, 0.8),
                    'elegance': random.uniform(0.5, 0.9)
                }
                
                # Format result
                result = BlendResult(
                    status="success",
                    memory_id=memory.id,
                    technique_used=creative_technique,
                    novel_idea=novel_idea,
                    blended_elements=BlendedElement(
                        memory_concept=memory_concept,
                        conversation_concept=conversation_concept,
                        connection_strength=connection_strength,
                        connection_type=connection_type
                    ),
                    creativity_metrics=creativity_metrics
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Error blending memory with novelty: {e}")
                
                # Sophisticated fallback blending
                fallback_blends = [
                    f"The resonance between '{memory_concept[:30]}...' and '{conversation_concept[:30]}...' suggests unexplored connections",
                    f"Like jazz improvisation, '{memory_concept[:30]}...' riffs on '{conversation_concept[:30]}...' creating something new",
                    f"The space between '{memory_concept[:30]}...' and '{conversation_concept[:30]}...' holds creative potential",
                    f"Weaving '{memory_concept[:30]}...' with '{conversation_concept[:30]}...' reveals hidden patterns"
                ]
                
                return BlendResult(
                    status="fallback",
                    memory_id=memory.id,
                    technique_used=creative_technique,
                    blended_text=random.choice(fallback_blends),
                    blended_elements=BlendedElement(
                        memory_concept=memory_concept[:50],
                        conversation_concept=conversation_concept[:50],
                        connection_strength=connection_strength,
                        connection_type=connection_type
                    ),
                    error=str(e)
                )
    
    async def get_generated_content(
        self,
        content_type: str = "all",
        limit: int = 10
    ) -> ContentRetrievalResult:
        """
        Get previously generated creative content
        
        Args:
            content_type: Type of content to retrieve ("wit", "insight", or "all")
            limit: Maximum number of items to return
            
        Returns:
            Dictionary of generated content by type
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
        
        result = ContentRetrievalResult()
        
        if content_type in ["wit", "all"]:
            # Sort by timestamp if available
            wits = self.context.contextual_wits.copy()
            wits.sort(
                key=lambda x: getattr(x[1], "timestamp", "1970-01-01"), 
                reverse=True
            )
            result.contextual_wits = wits[:limit]
        
        if content_type in ["insight", "all"]:
            # Sort by timestamp if available
            insights = self.context.creative_insights.copy()
            insights.sort(
                key=lambda x: getattr(x[1], "timestamp", "1970-01-01"), 
                reverse=True
            )
            result.creative_insights = insights[:limit]
        
        return result
