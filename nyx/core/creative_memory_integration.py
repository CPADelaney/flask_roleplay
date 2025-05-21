# nyx/core/creative_memory_integration.py

import logging
import asyncio
import datetime
import random
from typing import Dict, List, Any, Optional, Tuple
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
    RunConfig
)

# Import our systems
from nyx.core.novelty_engine import NoveltyEngine, NoveltyIdea
from nyx.core.recognition_memory import RecognitionMemorySystem, RecognitionResult

logger = logging.getLogger(__name__)

# Pydantic models for integration
class ContextualWit(BaseModel):
    """Schema for contextual wit generated from memory and novelty"""
    wit_text: str
    wit_type: str  # e.g., "analogy", "wordplay", "reference", "insight"
    related_memory_id: Optional[str] = None
    novelty_score: float
    appropriateness_score: float
    confidence: float
    generation_technique: str

class CreativeInsight(BaseModel):
    """Schema for a creative insight based on remembered experience"""
    insight_text: str
    source_memory_id: Optional[str] = None
    related_concept: str
    abstraction_level: float
    novelty_score: float
    usefulness_score: float
    insight_type: str  # e.g., "pattern", "principle", "perspective", "connection"

class CreativeMemoryIntegrationContext:
    """Context for the creative memory integration"""
    
    def __init__(self, novelty_engine=None, recognition_memory=None, memory_core=None):
        self.novelty_engine = novelty_engine
        self.recognition_memory = recognition_memory
        self.memory_core = memory_core
        
        # Storage for generated content
        self.contextual_wits = {}
        self.creative_insights = {}
        
        # Trace ID for connecting traces
        self.trace_id = f"creative_memory_{datetime.datetime.now().isoformat()}"
        
        # Integration parameters
        self.wit_generation_probability = 0.3  # Probability of generating wit on a turn
        self.insight_threshold = 0.7  # Minimum relevance for insight generation
        self.wit_types = ["analogy", "wordplay", "reference", "insight", "juxtaposition"]
        self.insight_types = ["pattern", "principle", "perspective", "connection"]

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
            model="gpt-4.1-nano",
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
            """,
            tools=[
                function_tool(self._create_memory_analogy),
                function_tool(self._create_wordplay)
            ],
            output_type=ContextualWit,
            model="gpt-4.1-nano",
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
            """,
            tools=[
                function_tool(self._extract_memory_patterns),
                function_tool(self._abstract_principles)
            ],
            output_type=CreativeInsight,
            model="gpt-4.1-nano",
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
            """,
            tools=[
                function_tool(self._apply_creative_technique),
                function_tool(self._transform_memory)
            ],
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.7)
        )
    
    # Tool functions for the various agents

    @staticmethod
    @function_tool
    async def _should_generate_wit(
        ctx: RunContextWrapper[CreativeMemoryIntegrationContext],
        conversation_text: str,
        recognized_memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Determine if wit generation is appropriate for current context
        
        Args:
            conversation_text: Current conversation text
            recognized_memories: List of recognized memories
            
        Returns:
            Decision about wit generation
        """
        # Simple heuristics for determining if wit is appropriate
        
        # Length check - very short messages may not provide enough context
        if len(conversation_text) < 10:
            return {"should_generate": False, "reason": "message_too_short"}
        
        # Need at least one recognized memory
        if not recognized_memories:
            return {"should_generate": False, "reason": "no_recognized_memories"}
        
        # Check for signals that wit might be inappropriate
        serious_indicators = ["sad", "upset", "worried", "concerned", "hurt", "angry", "problem", "issue"]
        is_serious = any(word in conversation_text.lower() for word in serious_indicators)
        
        if is_serious:
            return {"should_generate": False, "reason": "serious_context"}
        
        # Random probability check
        if random.random() > ctx.context.wit_generation_probability:
            return {"should_generate": False, "reason": "probability_threshold"}
        
        # If we get here, wit generation is appropriate
        return {
            "should_generate": True,
            "suitable_memory_count": len(recognized_memories),
            "wit_types": random.sample(ctx.context.wit_types, min(3, len(ctx.context.wit_types)))
        }

    @staticmethod
    @function_tool
    async def _get_memory_concepts(
        ctx: RunContextWrapper[CreativeMemoryIntegrationContext],
        memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract key concepts from memories
        
        Args:
            memories: List of memories
            
        Returns:
            List of extracted concepts
        """
        concepts = []
        
        for memory in memories:
            memory_text = memory.get("memory_text", "")
            memory_id = memory.get("id", "")
            
            # Very simple concept extraction
            words = memory_text.split()
            key_phrases = []
            
            if len(words) > 5:
                key_phrases.append(" ".join(words[:5]))
            
            if len(words) > 10:
                key_phrases.append(" ".join(words[5:10]))
            
            # Extract noun phrases (simplified)
            # In a real implementation, use NLP for better extraction
            for i in range(len(words) - 2):
                if i > 0 and words[i-1].lower() in ["a", "the", "this", "that"] and words[i][0].islower():
                    key_phrases.append(" ".join(words[i:i+2]))
            
            concepts.append({
                "memory_id": memory_id,
                "key_phrases": key_phrases,
                "source_text": memory_text
            })
        
        return concepts

    @staticmethod
    @function_tool
    async def _create_memory_analogy(
        ctx: RunContextWrapper[CreativeMemoryIntegrationContext],
        current_context: str,
        memory_text: str
    ) -> Dict[str, str]:
        """
        Create an analogy between current context and memory
        
        Args:
            current_context: Current conversation context
            memory_text: Text of the memory
            
        Returns:
            Analogy information
        """
        # This would ideally use sophisticated analogy generation
        # Simplified version:
        
        # Extract key phrases from memory
        memory_phrases = memory_text.split(".")[:2]
        memory_phrase = memory_phrases[0] if memory_phrases else memory_text[:50]
        
        # Extract key phrase from context
        context_phrase = current_context.split(".")[:1][0] if "." in current_context else current_context[:50]
        
        # Template-based analogy generation 
        templates = [
            "Just like {memory}, this reminds me of {context}.",
            "{context} is similar to {memory} in a way.",
            "This situation with {context} parallels my experience with {memory}.",
            "I'm seeing a pattern between {context} and {memory}.",
            "There's an interesting parallel between {context} and {memory}."
        ]
        
        template = random.choice(templates)
        analogy = template.format(memory=memory_phrase, context=context_phrase)
        
        return {
            "analogy_text": analogy,
            "memory_element": memory_phrase,
            "context_element": context_phrase,
            "analogy_type": "direct" if random.random() > 0.5 else "structural"
        }

    @staticmethod
    @function_tool
    async def _create_wordplay(
        ctx: RunContextWrapper[CreativeMemoryIntegrationContext],
        memory_text: str,
        current_context: str
    ) -> Dict[str, str]:
        """
        Create wordplay connecting memory and current context
        
        Args:
            memory_text: Text of the memory
            current_context: Current conversation context
            
        Returns:
            Wordplay information
        """
        # This would ideally use sophisticated wordplay generation
        # Simplified version:
        
        # Extract words from both sources
        memory_words = memory_text.split()
        context_words = current_context.split()
        
        # Look for matching words
        common_words = set([w.lower() for w in memory_words]) & set([w.lower() for w in context_words])
        common_words = [w for w in common_words if len(w) > 3]  # Only consider substantial words
        
        if common_words:
            word = random.choice(list(common_words))
            
            templates = [
                f"Speaking of {word}, that brings to mind...",
                f"That {word} reminds me of...",
                f"There's that {word} again, which makes me think of...",
                f"Interesting use of {word}, which connects to..."
            ]
            
            wordplay = random.choice(templates)
        else:
            # No common words, create simple wordplay
            if memory_words and context_words:
                memory_word = random.choice([w for w in memory_words if len(w) > 3]) if any(len(w) > 3 for w in memory_words) else random.choice(memory_words)
                context_word = random.choice([w for w in context_words if len(w) > 3]) if any(len(w) > 3 for w in context_words) else random.choice(context_words)
                
                templates = [
                    f"Your {context_word} is to my {memory_word} as...",
                    f"From {memory_word} to {context_word}, there's a connection...",
                    f"That {context_word} has a similar feel to my experience with {memory_word}..."
                ]
                
                wordplay = random.choice(templates)
            else:
                wordplay = "This brings to mind a similar situation..."
        
        return {
            "wordplay_text": wordplay,
            "wordplay_type": "common_word" if common_words else "word_association",
            "key_words": list(common_words) if common_words else []
        }

    @staticmethod
    @function_tool
    async def _extract_memory_patterns(
        ctx: RunContextWrapper[CreativeMemoryIntegrationContext],
        memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract patterns across multiple memories
        
        Args:
            memories: List of memories
            
        Returns:
            Extracted pattern information
        """
        # This would ideally use sophisticated pattern recognition
        # Simplified version:
        
        # Get memory texts
        memory_texts = [memory.get("memory_text", "") for memory in memories]
        
        # Simple word frequency analysis
        all_words = []
        for text in memory_texts:
            all_words.extend([word.lower() for word in text.split() if len(word) > 3])
        
        # Count words
        word_counts = {}
        for word in all_words:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1
        
        # Find most common words
        common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Pattern description
        if common_words:
            pattern_words = [word for word, count in common_words]
            pattern_description = f"Pattern involving {', '.join(pattern_words)}"
        else:
            pattern_description = "No clear pattern detected"
        
        return {
            "pattern_description": pattern_description,
            "common_elements": dict(common_words),
            "memory_count": len(memories),
            "confidence": min(1.0, 0.4 + (len(memories) / 10))
        }

    @staticmethod
    @function_tool
    async def _abstract_principles(
        ctx: RunContextWrapper[CreativeMemoryIntegrationContext],
        pattern: Dict[str, Any],
        current_context: str
    ) -> Dict[str, Any]:
        """
        Abstract principles from memory patterns
        
        Args:
            pattern: Pattern information
            current_context: Current conversation context
            
        Returns:
            Abstracted principle information
        """
        # This would ideally use sophisticated abstraction
        # Simplified version:
        
        # Extract key elements from pattern
        elements = list(pattern.get("common_elements", {}).keys())
        
        if not elements:
            return {
                "principle": "Some experiences share common elements",
                "abstraction_level": 0.5,
                "confidence": 0.3,
                "applicability": "low"
            }
        
        # Create a principle based on elements
        principle_templates = [
            f"Situations involving {elements[0]} often lead to {elements[1] if len(elements) > 1 else 'predictable outcomes'}",
            f"When {elements[0]} is present, {elements[1] if len(elements) > 1 else 'certain patterns'} tend to emerge",
            f"The presence of {elements[0]} suggests a tendency toward {elements[1] if len(elements) > 1 else 'specific behaviors'}",
            f"Experiences with {elements[0]} typically involve {elements[1] if len(elements) > 1 else 'consistent elements'}"
        ]
        
        principle = random.choice(principle_templates)
        
        # Assess applicability to current context
        applicability = "medium"
        for element in elements:
            if element in current_context.lower():
                applicability = "high"
                break
        
        return {
            "principle": principle,
            "elements": elements,
            "abstraction_level": 0.7,
            "confidence": pattern.get("confidence", 0.5),
            "applicability": applicability
        }

    @staticmethod
    @function_tool
    async def _apply_creative_technique(
        ctx: RunContextWrapper[CreativeMemoryIntegrationContext],
        memory: Dict[str, Any],
        current_context: str,
        technique: str
    ) -> Dict[str, Any]:
        """
        Apply a specific creative technique to memory
        
        Args:
            memory: Memory to transform
            current_context: Current conversation context
            technique: Creative technique to apply
            
        Returns:
            Result of applying the technique
        """
        # This would ideally use the novelty engine's techniques
        # Simplified version:
        
        memory_text = memory.get("memory_text", "")
        
        result = {
            "technique": technique,
            "memory_id": memory.get("id", ""),
            "transformed_text": "",
            "transformation_notes": ""
        }
        
        if technique == "bisociation":
            # Connect memory domain with current context domain
            result["transformed_text"] = f"Connecting your current situation with my memory of {memory_text[:30]}..."
            result["transformation_notes"] = "Connected two conceptual domains"
        
        elif technique == "perspective_shift":
            # View memory from different perspective
            perspectives = ["temporal", "spatial", "emotional", "counterfactual"]
            perspective = random.choice(perspectives)
            
            if perspective == "temporal":
                result["transformed_text"] = f"Looking back on {memory_text[:30]}... from today's perspective..."
            elif perspective == "emotional":
                result["transformed_text"] = f"Considering how I felt during {memory_text[:30]}... compared to your current situation..."
            else:
                result["transformed_text"] = f"Seeing {memory_text[:30]}... from a different angle..."
                
            result["transformation_notes"] = f"Applied {perspective} perspective shift"
        
        elif technique == "constraint_relaxation":
            # Remove assumed constraints
            result["transformed_text"] = f"If we remove the usual limitations from {memory_text[:30]}..."
            result["transformation_notes"] = "Relaxed implicit constraints"
        
        else:
            # Default transformation
            result["transformed_text"] = f"This reminds me of {memory_text[:30]}... but with a twist..."
            result["transformation_notes"] = "Applied general creative transformation"
        
        return result

    @staticmethod
    @function_tool
    async def _transform_memory(
        ctx: RunContextWrapper[CreativeMemoryIntegrationContext],
        memory: Dict[str, Any],
        transformation_type: str
    ) -> Dict[str, Any]:
        """
        Transform a memory into a novel form
        
        Args:
            memory: Memory to transform
            transformation_type: Type of transformation to apply
            
        Returns:
            Transformed memory
        """
        # This would ideally use sophisticated NLG transformation
        # Simplified version:
        
        memory_text = memory.get("memory_text", "")
        memory_id = memory.get("id", "")
        
        transformed_text = memory_text
        transformation_notes = ""
        
        if transformation_type == "generalize":
            # Generalize specific memory
            specific_terms = ["I", "my", "me", "mine"]
            generalized_text = memory_text
            
            for term in specific_terms:
                generalized_text = generalized_text.replace(f" {term} ", " one ")
            
            transformed_text = f"Generally speaking, {generalized_text}"
            transformation_notes = "Generalized from specific to general"
        
        elif transformation_type == "metaphorize":
            # Turn memory into metaphor
            transformed_text = f"It's like {memory_text[:30]}..."
            transformation_notes = "Transformed into metaphorical form"
        
        elif transformation_type == "hypothesize":
            # Turn memory into hypothesis
            transformed_text = f"What if {memory_text}... but in a different context?"
            transformation_notes = "Transformed into hypothetical form"
        
        else:
            # Default transformation
            transformed_text = f"This reminds me of a situation where {memory_text}"
            transformation_notes = "Applied standard transformation"
        
        return {
            "memory_id": memory_id,
            "original_text": memory_text,
            "transformed_text": transformed_text,
            "transformation_type": transformation_type,
            "transformation_notes": transformation_notes
        }
    
    # Main public methods for the integration system
    
    async def process_conversation_turn(
        self,
        conversation_text: str,
        current_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
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
            return {"error": "Recognition memory system not available"}
        
        if not self.context.novelty_engine:
            return {"error": "Novelty engine not available"}
        
        with trace(workflow_name="Creative Memory Integration", group_id=self.context.trace_id):
            # 1. Get recognized memories from the recognition memory system
            recognition_results = await self.context.recognition_memory.process_conversation_turn(
                conversation_text,
                current_context
            )
            
            # If no memories recognized, return empty result
            if not recognition_results:
                return {
                    "status": "no_memories_recognized",
                    "generated_content": None
                }
            
            # 2. Convert recognition results to format needed for integration
            memories = []
            for result in recognition_results:
                if isinstance(result, dict):
                    memories.append(result)
                else:
                    # Convert Pydantic model to dict if needed
                    memories.append({
                        "memory_id": result.memory_id,
                        "memory_text": result.memory_text,
                        "memory_type": result.memory_type,
                        "relevance_score": result.relevance_score,
                        "activation_trigger": result.activation_trigger
                    })
            
            # 3. Prepare prompt for integration agent
            prompt = f"""Process this conversation turn with recognized memories:
            
            Conversation: {conversation_text}
            
            Recognized memories: {len(memories)} memories recognized.
            
            Determine the most appropriate creative integration approach:
            - Generate contextual wit if appropriate
            - Create a creative insight from patterns in memories
            - Blend memory recognition with novel idea generation
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
                wit_id = f"wit_{random.randint(1000, 9999)}"
                self.context.contextual_wits[wit_id] = integration_result
                
                return {
                    "status": "wit_generated",
                    "content_type": "contextual_wit",
                    "content_id": wit_id,
                    "generated_content": integration_result
                }
                
            elif hasattr(integration_result, "insight_text"):
                # It's a CreativeInsight
                insight_id = f"insight_{random.randint(1000, 9999)}"
                self.context.creative_insights[insight_id] = integration_result
                
                return {
                    "status": "insight_generated",
                    "content_type": "creative_insight",
                    "content_id": insight_id,
                    "generated_content": integration_result
                }
                
            else:
                # It's likely a direct result from the memory-novelty blend
                return {
                    "status": "blend_generated",
                    "content_type": "memory_novelty_blend",
                    "generated_content": integration_result
                }
    
    async def generate_contextual_wit(
        self,
        conversation_text: str,
        memory: Dict[str, Any]
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
            # Prepare prompt for wit generation agent
            prompt = f"""Generate contextual wit based on this memory and conversation:
            
            Conversation: {conversation_text}
            
            Memory: {memory.get("memory_text", "")}
            
            Create witty, clever response that references this memory in a way
            that enhances the conversation. Focus on quality and contextual appropriateness.
            """
            
            # Run through wit generation agent
            result = await Runner.run(
                self.wit_generation_agent,
                prompt,
                context=self.context
            )
            
            wit = result.final_output
            
            # Store the generated wit
            if wit:
                wit_id = f"wit_{random.randint(1000, 9999)}"
                self.context.contextual_wits[wit_id] = wit
            
            return wit
    
    async def generate_creative_insight(
        self,
        conversation_text: str,
        memories: List[Dict[str, Any]]
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
        
        with trace(workflow_name="Generate Creative Insight", group_id=self.context.trace_id):
            # Prepare prompt for insight generation agent
            memory_texts = [f"Memory {i+1}: {memory.get('memory_text', '')}" for i, memory in enumerate(memories)]
            memory_context = "\n\n".join(memory_texts)
            
            prompt = f"""Generate a creative insight based on these memories and conversation:
            
            Conversation: {conversation_text}
            
            Memories:
            {memory_context}
            
            Create a deeper, more abstract insight from these memories that applies
            to the current context. Extract generalizable principles or identify
            patterns that provide value to the current situation.
            """
            
            # Run through insight generation agent
            result = await Runner.run(
                self.insight_generation_agent,
                prompt,
                context=self.context
            )
            
            insight = result.final_output
            
            # Store the generated insight
            if insight:
                insight_id = f"insight_{random.randint(1000, 9999)}"
                self.context.creative_insights[insight_id] = insight
            
            return insight
    
    async def blend_memory_with_novelty(
        self,
        conversation_text: str,
        memory: Dict[str, Any],
        creative_technique: str = "auto"
    ) -> Dict[str, Any]:
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
            return {"error": "Novelty engine not available"}
        
        with trace(workflow_name="Blend Memory with Novelty", group_id=self.context.trace_id):
            # If technique is auto, choose based on memory content
            if creative_technique == "auto":
                techniques = ["bisociation", "perspective_shifting", "constraint_relaxation"]
                creative_technique = random.choice(techniques)
            
            # Extract concepts
            memory_concept = memory.get("memory_text", "")[:50]
            conversation_concept = conversation_text[:50]
            
            try:
                # Use novelty engine to generate idea connecting memory and conversation
                novel_idea = await self.context.novelty_engine.generate_novel_idea(
                    technique=creative_technique,
                    concepts=[memory_concept, conversation_concept]
                )
                
                # Format result
                result = {
                    "status": "success",
                    "memory_id": memory.get("id", ""),
                    "technique_used": creative_technique,
                    "novel_idea": novel_idea,
                    "blended_elements": {
                        "memory_concept": memory_concept,
                        "conversation_concept": conversation_concept
                    }
                }
                
                return result
                
            except Exception as e:
                logger.error(f"Error blending memory with novelty: {e}")
                
                # Fallback to simple blending
                return {
                    "status": "fallback",
                    "memory_id": memory.get("id", ""),
                    "blended_text": f"This reminds me of {memory_concept}, which makes me think about {conversation_concept} differently...",
                    "error": str(e)
                }
    
    async def get_generated_content(
        self,
        content_type: str = "all",
        limit: int = 10
    ) -> Dict[str, List[Any]]:
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
        
        result = {}
        
        if content_type in ["wit", "all"]:
            wits = list(self.context.contextual_wits.items())
            wits.sort(key=lambda x: getattr(x[1], "timestamp", ""), reverse=True)
            result["contextual_wits"] = [(wit_id, wit) for wit_id, wit in wits[:limit]]
        
        if content_type in ["insight", "all"]:
            insights = list(self.context.creative_insights.items())
            insights.sort(key=lambda x: getattr(x[1], "timestamp", ""), reverse=True)
            result["creative_insights"] = [(insight_id, insight) for insight_id, insight in insights[:limit]]
        
        return result
