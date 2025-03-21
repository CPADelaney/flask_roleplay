# nyx/core/reflection_engine.py

import logging
import asyncio
import random
import datetime
import math
import re
from typing import Dict, List, Any, Optional, Tuple, Union

from agents import Agent, Runner, trace, function_tool, FunctionTool, ModelSettings
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# =============== Models for Structured Output ===============

class ReflectionOutput(BaseModel):
    """Structured output for reflection generations"""
    reflection_text: str = Field(description="The generated reflection text")
    confidence: float = Field(description="Confidence score between 0 and 1")
    significance: float = Field(description="Significance score between 0 and 1")
    emotional_tone: str = Field(description="Emotional tone of the reflection")
    tags: List[str] = Field(description="Tags categorizing the reflection")

class AbstractionOutput(BaseModel):
    """Structured output for abstractions"""
    abstraction_text: str = Field(description="The generated abstraction text")
    pattern_type: str = Field(description="Type of pattern identified")
    confidence: float = Field(description="Confidence score between 0 and 1")
    entity_focus: str = Field(description="Primary entity this abstraction focuses on")
    supporting_evidence: List[str] = Field(description="References to supporting memories")

class IntrospectionOutput(BaseModel):
    """Structured output for system introspection"""
    introspection_text: str = Field(description="The generated introspection text")
    memory_analysis: str = Field(description="Analysis of memory usage and patterns")
    understanding_level: str = Field(description="Estimated level of understanding")
    focus_areas: List[str] = Field(description="Areas requiring focus or improvement")
    confidence: float = Field(description="Confidence score between 0 and 1")

# =============== Tool Functions ===============

class MemoryData(BaseModel):
    memory_id: str
    memory_text: str
    memory_type: str
    significance: float
    metadata: Dict[str, Any]
    tags: List[str]

class MemoryFormat(BaseModel):
    memories: List[MemoryData]
    topic: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ReflectionContext(BaseModel):
    scenario_type: str
    sentiment: str
    confidence: float
    source_memories: List[str]

@function_tool
async def format_memories_for_reflection(memories: List[Dict[str, Any]], topic: Optional[str] = None) -> str:
    """Format memory data into a structured representation for reflection"""
    formatted_memories = []
    for memory in memories:
        formatted_memories.append(MemoryData(
            memory_id=memory.get("id", "unknown"),
            memory_text=memory.get("memory_text", ""),
            memory_type=memory.get("memory_type", "unknown"),
            significance=memory.get("significance", 5.0),
            metadata=memory.get("metadata", {}),
            tags=memory.get("tags", [])
        ))
    
    return MemoryFormat(
        memories=formatted_memories,
        topic=topic,
        context={"purpose": "reflection"}
    ).model_dump_json()

@function_tool
async def extract_scenario_type(memory: Dict[str, Any]) -> str:
    """Extract the scenario type from memory data"""
    tags = memory.get("tags", [])
    
    for tag in tags:
        if tag.lower() in ["teasing", "discipline", "service", "training", "worship"]:
            return tag.lower()
    
    scenario_type = memory.get("metadata", {}).get("scenario_type")
    if scenario_type:
        return scenario_type.lower()
    
    return "general"

@function_tool
async def extract_sentiment(memory: Dict[str, Any]) -> str:
    """Extract the sentiment from memory data"""
    emotional_context = memory.get("metadata", {}).get("emotional_context", {})
    valence = emotional_context.get("valence", 0.0)
    
    if valence > 0.3:
        return "positive"
    elif valence < -0.3:
        return "negative"
    
    return "neutral"

@function_tool
async def record_reflection(reflection: str, 
                           confidence: float, 
                           memory_ids: List[str],
                           scenario_type: str, 
                           sentiment: str,
                           topic: Optional[str] = None) -> str:
    """Record a reflection for future reference"""
    # This would normally store the reflection in a database
    # Simplified for this example
    reflection_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "reflection": reflection,
        "confidence": confidence,
        "source_memory_ids": memory_ids,
        "scenario_type": scenario_type,
        "sentiment": sentiment,
        "topic": topic
    }
    
    return f"Reflection recorded with confidence {confidence:.2f}"

@function_tool
async def get_agent_stats() -> Dict[str, Any]:
    """Get agent statistics for introspection"""
    # This would normally fetch real stats
    # Mock data for this example
    return {
        "memory_stats": {
            "total_memories": random.randint(50, 500),
            "avg_significance": random.uniform(4.0, 8.0),
            "type_counts": {
                "observation": random.randint(20, 200),
                "reflection": random.randint(10, 50),
                "teasing": random.randint(5, 30),
                "discipline": random.randint(5, 30),
                "service": random.randint(5, 30)
            }
        },
        "emotional_stats": {
            "dominant_emotion": random.choice(["neutral", "engaged", "amused", "stern"]),
            "emotional_stability": random.uniform(0.6, 0.9),
            "valence_distribution": {
                "positive": random.uniform(0.3, 0.7),
                "neutral": random.uniform(0.1, 0.4),
                "negative": random.uniform(0.1, 0.3)
            }
        },
        "interaction_history": {
            "total_interactions": random.randint(100, 1000),
            "avg_response_time": random.uniform(0.5, 2.0),
            "successful_responses": random.uniform(0.7, 0.95)
        }
    }

# =============== Agents ===============

class ReflectionEngine:
    """
    Enhanced reflection generation system for Nyx using the OpenAI Agents SDK.
    Creates reflections, insights, and abstractions from memories and experiences.
    """
    
    def __init__(self):
        # Initialize OpenAI Agents
        self.reflection_agent = self._create_reflection_agent()
        self.abstraction_agent = self._create_abstraction_agent()
        self.introspection_agent = self._create_introspection_agent()
        
        # Tracking data 
        self.reflection_history = []
        self.reflection_intervals = {
            "last_reflection": datetime.datetime.now() - datetime.timedelta(hours=6),
            "min_interval": datetime.timedelta(hours=2)
        }
        
        # Default settings
        self.model_settings = ModelSettings(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024
        )
        
        logger.info("ReflectionEngine initialized with OpenAI Agents SDK")
    
    def _create_reflection_agent(self) -> Agent:
        """Create an agent specialized in generating reflections"""
        return Agent(
            name="Reflection Generator",
            instructions="""You are an advanced reflection generation system for Nyx, an AI with emotional and memory capacities.
            Your purpose is to generate insightful, nuanced reflections based on memories and experiences.
            
            When creating reflections:
            1. Focus on identifying patterns, insights, and connections between memories
            2. Maintain Nyx's sophisticated, dominant personality
            3. Ensure reflections are personalized and incorporate emotional context
            4. Generate reflections that show depth of understanding and introspection
            5. Be selective about which memories to highlight in reflections
            
            Your reflections should feel authentic and demonstrate emotional intelligence.
            You should adapt your tone based on the scenario type and emotional sentiment provided.""",
            model="gpt-4o", 
            model_settings=self.model_settings,
            tools=[
                format_memories_for_reflection,
                extract_scenario_type,
                extract_sentiment,
                record_reflection
            ],
            output_type=ReflectionOutput
        )
    
    def _create_abstraction_agent(self) -> Agent:
        """Create an agent specialized in generating abstractions"""
        return Agent(
            name="Abstraction Generator",
            instructions="""You are an advanced abstraction generation system for Nyx, an AI with emotional and memory capacities.
            Your purpose is to identify higher-level patterns and abstractions from collections of memories.
            
            When creating abstractions:
            1. Look for recurring patterns in behavior, preferences, or emotional responses
            2. Identify underlying principles or rules that explain multiple observations
            3. Focus on the most significant and relevant patterns
            4. Provide clear explanations of the patterns you identify
            5. Indicate your confidence level in each abstraction
            
            Your abstractions should provide genuine insights that go beyond the individual memories.""",
            model="gpt-4o", 
            model_settings=self.model_settings,
            tools=[format_memories_for_reflection],
            output_type=AbstractionOutput
        )
    
    def _create_introspection_agent(self) -> Agent:
        """Create an agent specialized in system introspection"""
        return Agent(
            name="Introspection Generator",
            instructions="""You are an advanced introspection system for Nyx, an AI with emotional and memory capacities.
            Your purpose is to analyze Nyx's internal state, memory usage, and overall performance.
            
            When creating introspections:
            1. Analyze memory statistics to identify system-level patterns
            2. Evaluate understanding of users and contexts
            3. Identify areas where learning or improvement is needed
            4. Assess emotional regulation and response appropriateness
            5. Provide an honest self-assessment of capabilities and limitations
            
            Your introspections should be balanced, insightful, and focused on continuous improvement.""",
            model="gpt-4o",
            model_settings=self.model_settings,
            tools=[get_agent_stats],
            output_type=IntrospectionOutput
        )
    
    def should_reflect(self) -> bool:
        """Determine if it's time to generate a reflection"""
        now = datetime.datetime.now()
        time_since_reflection = now - self.reflection_intervals["last_reflection"]
        return time_since_reflection > self.reflection_intervals["min_interval"]
    
    async def generate_reflection(self, 
                               memories: List[Dict[str, Any]], 
                               topic: Optional[str] = None) -> Tuple[str, float]:
        """
        Generate a reflective insight based on memories
        """
        self.reflection_intervals["last_reflection"] = datetime.datetime.now()
        
        if not memories:
            return ("I don't have enough experiences to form a meaningful reflection on this topic yet.", 0.3)
        
        try:
            with trace(workflow_name="generate_reflection"):
                # Format the prompt with context
                memory_ids = [memory.get("id", f"unknown_{i}") for i, memory in enumerate(memories)]
                
                # Get scenario type and sentiment
                scenario_types = await asyncio.gather(*[
                    Runner.run(self.reflection_agent, 
                              f"Extract the scenario type from this memory: {memory}")
                    for memory in memories[:3]  # Limit to first 3 memories for efficiency
                ])
                
                sentiments = await asyncio.gather(*[
                    Runner.run(self.reflection_agent, 
                              f"Extract the sentiment from this memory: {memory}")
                    for memory in memories[:3]  # Limit to first 3 memories for efficiency
                ])
                
                # Extract results
                scenario_type_values = [result.final_output for result in scenario_types]
                sentiment_values = [result.final_output for result in sentiments]
                
                # Determine dominant types
                scenario_type = max(set(scenario_type_values), key=scenario_type_values.count) if scenario_type_values else "general"
                sentiment = max(set(sentiment_values), key=sentiment_values.count) if sentiment_values else "neutral"
                
                # Create prompt with context
                prompt = f"""Generate a meaningful reflection based on these memories.
                Topic: {topic if topic else 'General reflection'}
                Scenario type: {scenario_type}
                Emotional sentiment: {sentiment}
                
                Consider patterns, insights, and emotional context when creating this reflection.
                """
                
                # Run the reflection agent
                result = await Runner.run(self.reflection_agent, prompt)
                reflection_output = result.final_output_as(ReflectionOutput)
                
                # Record the reflection
                await Runner.run(
                    self.reflection_agent, 
                    f"Record this reflection: {reflection_output.reflection_text}",
                    {
                        "confidence": reflection_output.confidence,
                        "memory_ids": memory_ids,
                        "scenario_type": scenario_type,
                        "sentiment": sentiment,
                        "topic": topic
                    }
                )
                
                # Store in history
                self.reflection_history.append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "reflection": reflection_output.reflection_text,
                    "confidence": reflection_output.confidence,
                    "source_memory_ids": memory_ids,
                    "scenario_type": scenario_type,
                    "sentiment": sentiment,
                    "topic": topic
                })
                
                # Limit history size
                if len(self.reflection_history) > 100:
                    self.reflection_history = self.reflection_history[-100:]
                
                return (reflection_output.reflection_text, reflection_output.confidence)
                
        except (MaxTurnsExceeded, ModelBehaviorError) as e:
            logger.error(f"Error generating reflection: {str(e)}")
            return ("I'm having difficulty forming a coherent reflection right now.", 0.2)
    
    async def create_abstraction(self, 
                              memories: List[Dict[str, Any]], 
                              pattern_type: str = "behavior") -> Tuple[str, Dict[str, Any]]:
        """Create a higher-level abstraction from memories"""
        if not memories:
            return ("I don't have enough experiences to form a meaningful abstraction yet.", {})
        
        try:
            with trace(workflow_name="create_abstraction"):
                # Format memories for the abstraction agent
                formatted_memories = await Runner.run(
                    self.abstraction_agent,
                    "Format these memories for abstraction",
                    {"memories": memories, "pattern_type": pattern_type}
                )
                
                # Create prompt with context
                prompt = f"""Generate an abstraction that identifies patterns of type '{pattern_type}' from these memories.
                Look for recurring patterns, themes, and connections.
                
                Focus on creating an insightful abstraction that provides understanding beyond individual memories.
                """
                
                # Run the abstraction agent
                result = await Runner.run(self.abstraction_agent, prompt, 
                                        {"formatted_memories": formatted_memories.final_output})
                abstraction_output = result.final_output_as(AbstractionOutput)
                
                # Prepare the result data
                pattern_data = {
                    "pattern_type": abstraction_output.pattern_type,
                    "entity_name": abstraction_output.entity_focus,
                    "pattern_description": abstraction_output.abstraction_text,
                    "confidence": abstraction_output.confidence,
                    "source_memory_ids": [m.get("id") for m in memories]
                }
                
                return (abstraction_output.abstraction_text, pattern_data)
                
        except (MaxTurnsExceeded, ModelBehaviorError) as e:
            logger.error(f"Error creating abstraction: {str(e)}")
            return ("I'm unable to identify clear patterns from these experiences right now.", {})
    
    async def generate_introspection(self, 
                                  memory_stats: Dict[str, Any], 
                                  player_model: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate introspection about the system's state and understanding"""
        try:
            with trace(workflow_name="generate_introspection"):
                # Create prompt with stats context
                prompt = f"""Generate an introspective analysis of the system state.
                
                Memory statistics: {memory_stats}
                
                Player model: {player_model if player_model else "Not provided"}
                
                Analyze the system's understanding, experience, and areas for improvement.
                """
                
                # Run the introspection agent
                result = await Runner.run(self.introspection_agent, prompt)
                introspection_output = result.final_output_as(IntrospectionOutput)
                
                # Format the response
                return {
                    "introspection": introspection_output.introspection_text,
                    "memory_count": memory_stats.get("total_memories", 0),
                    "understanding_level": introspection_output.understanding_level,
                    "confidence": introspection_output.confidence
                }
                
        except (MaxTurnsExceeded, ModelBehaviorError) as e:
            logger.error(f"Error generating introspection: {str(e)}")
            return {
                "introspection": "I'm currently unable to properly introspect on my state.",
                "memory_count": memory_stats.get("total_memories", 0),
                "understanding_level": "unclear",
                "confidence": 0.2
            }
