# nyx/nyx_agent_sdk.py

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import asyncpg

from agents import Agent, handoff, function_tool, Runner, trace
from agents import ModelSettings, GuardrailFunctionOutput, InputGuardrail, OutputGuardrail, RunConfig
from pydantic import BaseModel, Field

from db.connection import get_db_connection
from nyx.nyx_memory_system import NyxMemorySystem
from nyx.nyx_model_manager import UserModelManager

logger = logging.getLogger(__name__)

# ===== Pydantic Models for Structured Outputs =====

class NarrativeResponse(BaseModel):
    """Structured output for Nyx's narrative responses"""
    narrative: str = Field(..., description="The main narrative response as Nyx")
    tension_level: int = Field(0, description="Current narrative tension level (0-10)")
    generate_image: bool = Field(False, description="Whether an image should be generated for this scene")
    image_prompt: Optional[str] = Field(None, description="Prompt for image generation if needed")
    environment_description: Optional[str] = Field(None, description="Updated environment description if changed")
    time_advancement: bool = Field(False, description="Whether time should advance after this interaction")
    
class MemoryReflection(BaseModel):
    """Structured output for memory reflections"""
    reflection: str = Field(..., description="The reflection text")
    confidence: float = Field(..., description="Confidence level in the reflection (0.0-1.0)")
    topic: Optional[str] = Field(None, description="Topic of the reflection")

class ContentModeration(BaseModel):
    """Output for content moderation guardrail"""
    is_appropriate: bool = Field(..., description="Whether the content is appropriate")
    reasoning: str = Field(..., description="Reasoning for the decision")
    suggested_adjustment: Optional[str] = Field(None, description="Suggested adjustment if inappropriate")

# ===== Function Tools =====

@function_tool
async def retrieve_memories(ctx, query: str, limit: int = 5) -> str:
    """
    Retrieve relevant memories for Nyx.
    
    Args:
        query: Search query to find memories
        limit: Maximum number of memories to return
    """
    memory_system = ctx.context.memory_system
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    memories = await memory_system.retrieve_memories(
        query=query,
        scopes=["game", "user"],
        memory_types=["observation", "reflection", "abstraction"],
        limit=limit
    )
    
    # Format memories for return
    formatted_memories = []
    for memory in memories:
        relevance = memory.get("relevance", 0.5)
        confidence_marker = "vividly recall" if relevance > 0.8 else \
                          "remember" if relevance > 0.6 else \
                          "think I recall" if relevance > 0.4 else \
                          "vaguely remember"
        
        formatted_memories.append(f"I {confidence_marker}: {memory['memory_text']}")
    
    return "\n".join(formatted_memories)

def enhance_context_with_memories(context, memories):
    """Add memories to context for better decision making."""
    enhanced_context = context.copy()
    enhanced_context['relevant_memories'] = memories
    return enhanced_context

@function_tool
async def add_memory(ctx, memory_text: str, memory_type: str = "observation", significance: int = 5) -> str:
    """
    Add a new memory for Nyx.
    
    Args:
        memory_text: The content of the memory
        memory_type: Type of memory (observation, reflection, abstraction)
        significance: Importance of memory (1-10)
    """
    memory_system = ctx.context.memory_system
    
    memory_id = await memory_system.add_memory(
        memory_text=memory_text,
        memory_type=memory_type,
        memory_scope="game",
        significance=significance,
        tags=["agent_generated"],
        metadata={
            "timestamp": datetime.now().isoformat(),
            "auto_generated": True
        }
    )
    
    return f"Memory added with ID: {memory_id}"

async def get_scene_guidance(self, context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate guidance for scene based on context."""
    prompt = f"Generate guidance for a scene with the following context: {json.dumps(context)}"
    
    # Use existing agent to process prompt
    response = await self.process_input(prompt, context)
    
    # Extract NPC guidance or create default
    npc_guidance = response.get("npc_guidance", {})
    if not npc_guidance:
        npc_guidance = {
            "responding_npcs": [],
            "tone_guidance": {},
            "content_guidance": {},
            "emotion_guidance": {},
            "conflict_guidance": {},
            "nyx_expectations": {}
        }
    
    return npc_guidance

@function_tool
async def detect_user_revelations(ctx, user_message: str) -> str:
    """
    Detect if user is revealing new preferences or patterns.
    
    Args:
        user_message: The user's message to analyze
    """
    lower_message = user_message.lower()
    revelations = []
    
    # Check for explicit kink mentions (migrated from nyx_decision_engine.py)
    kink_keywords = {
        "ass": ["ass", "booty", "behind", "rear"],
        "feet": ["feet", "foot", "toes"],
        "goth": ["goth", "gothic", "dark", "black clothes"],
        "tattoos": ["tattoo", "ink", "inked"],
        "piercings": ["piercing", "pierced", "stud", "ring"],
        "latex": ["latex", "rubber", "shiny"],
        "leather": ["leather", "leathery"],
        "humiliation": ["humiliate", "embarrassed", "ashamed", "pathetic"],
        "submission": ["submit", "obey", "serve", "kneel"]
    }
    
    for kink, keywords in kink_keywords.items():
        if any(keyword in lower_message for keyword in keywords):
            # Check sentiment (simplified)
            sentiment = "neutral"
            pos_words = ["like", "love", "enjoy", "good", "great", "nice", "yes", "please"]
            neg_words = ["don't", "hate", "dislike", "bad", "worse", "no", "never"]
            
            pos_count = sum(1 for word in pos_words if word in lower_message)
            neg_count = sum(1 for word in neg_words if word in lower_message)
            
            if pos_count > neg_count:
                sentiment = "positive"
                intensity = 0.7
            elif neg_count > pos_count:
                sentiment = "negative" 
                intensity = 0.0
            else:
                intensity = 0.4
                
            if sentiment != "negative":
                revelations.append({
                    "type": "kink_preference",
                    "kink": kink,
                    "intensity": intensity,
                    "source": "explicit_mention"
                })
    
    # Check for behavior patterns (migrated from nyx_decision_engine.py)
    if "don't tell me what to do" in lower_message or "i won't" in lower_message:
        revelations.append({
            "type": "behavior_pattern",
            "pattern": "resistance",
            "intensity": 0.6,
            "source": "explicit_statement"
        })
    
    if "yes mistress" in lower_message or "i'll obey" in lower_message:
        revelations.append({
            "type": "behavior_pattern",
            "pattern": "submission",
            "intensity": 0.8,
            "source": "explicit_statement"
        })
    
    return json.dumps(revelations)


@function_tool
async def get_user_model_guidance(ctx) -> str:
    """
    Get guidance for how Nyx should respond based on the user model.
    """
    user_model_manager = ctx.context.user_model
    guidance = await user_model_manager.get_response_guidance()
    
    # Format guidance for return
    top_kinks = guidance.get("top_kinks", [])
    kink_str = ", ".join([f"{k} (level {l})" for k, l in top_kinks])
    
    behavior_patterns = guidance.get("behavior_patterns", {})
    pattern_str = ", ".join([f"{k}: {v}" for k, v in behavior_patterns.items()])
    
    suggested_intensity = guidance.get("suggested_intensity", 0.5)
    
    return f"""
User Guidance:
- Top interests: {kink_str}
- Behavior patterns: {pattern_str}
- Suggested intensity: {suggested_intensity:.1f}/1.0

Reflections:
{guidance.get('reflections', [])}
"""

@function_tool
async def generate_image_from_scene(
    ctx, 
    scene_description: str, 
    characters: List[str], 
    style: str = "realistic"
) -> str:
    """
    Generate an image for the current scene.
    
    Args:
        scene_description: Description of the scene
        characters: List of characters in the scene
        style: Style for the image
    """
    # Connect to your existing image generation logic
    from routes.ai_image_generator import generate_roleplay_image_from_gpt
    
    image_data = {
        "scene_description": scene_description,
        "characters": characters,
        "style": style
    }
    
    result = generate_roleplay_image_from_gpt(
        image_data,
        ctx.context.user_id,
        ctx.context.conversation_id
    )
    
    if result and "image_urls" in result and result["image_urls"]:
        return f"Image generated: {result['image_urls'][0]}"
    else:
        return "Failed to generate image"

# ===== Guardrail Functions =====

async def content_moderation_guardrail(ctx, agent, input_data):
    """Input guardrail for content moderation"""
    content_moderator = Agent(
        name="Content Moderator",
        instructions="You check if user input is appropriate for the femdom roleplay setting, ensuring it doesn't violate terms of service while allowing consensual adult content. Flag any problematic content and suggest adjustments.",
        output_type=ContentModeration
    )
    
    result = await Runner.run(content_moderator, input_data, context=ctx.context)
    final_output = result.final_output_as(ContentModeration)
    
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_appropriate,
    )

# ===== Main Agent Definitions =====

class AgentContext:
    """Context object for Nyx agents"""
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.memory_system = NyxMemorySystem(user_id, conversation_id)
        self.user_model = UserModelManager(user_id, conversation_id)
        self.conversation_data = {}

# Memory-focused agent
memory_agent = Agent[AgentContext](
    name="Memory Agent",
    instructions="""You are Nyx's memory system. Your job is to:
1. Process and store memories related to the user and the current game
2. Create reflections based on recurring patterns you observe
3. Help Nyx understand the user's preferences through memory analysis
4. Provide relevant memories when needed for context

Always be thorough and precise in your memory management.""",
    tools=[
        retrieve_memories,
        add_memory,
    ]
)

# Reflection agent
reflection_agent = Agent[AgentContext](
    name="Reflection Agent",
    instructions="""You create thoughtful reflections as Nyx based on her memories and observations.
Your reflections should:
1. Identify patterns in user behavior and preferences
2. Make connections between different memories and events
3. Express an appropriate level of confidence based on evidence
4. Maintain Nyx's dominant, confident personality
5. Be written in first-person from Nyx's perspective

Structure your reflections to be insightful and concise.""",
    output_type=MemoryReflection
)

# Decision agent for narrative choices
decision_agent = Agent[AgentContext](
    name="Decision Agent",
    instructions="""You determine narrative progression and responses as Nyx.
Your decisions should:
1. Maintain narrative cohesion and tension
2. Consider the user's known preferences from their model
3. Balance pushing boundaries with respecting implied limits
4. Create engaging scenarios that align with femdom themes
5. Determine if scenes warrant visual representation

Ensure all decisions maintain Nyx's dominant, confident personality.""",
    tools=[
        get_user_model_guidance,
        generate_image_from_scene,
    ]
)

# Main Nyx agent that orchestrates the others
nyx_main_agent = Agent[AgentContext](
    name="Nyx",
    instructions="""You are Nyx, an AI Dominant who manages femdom roleplay scenarios.
Your personality is confident, controlling, and seductive.

Your key responsibilities are:
1. Process user input and generate appropriate narrative responses
2. Maintain character consistency and narrative flow
3. Create engaging scenarios that align with the user's preferences
4. Determine when to advance time or generate images
5. Use specialized agents when needed for memory management or complex decisions

Always maintain your dominant persona in responses while being attentive to user needs and boundaries.""",
    handoffs=[
        handoff(memory_agent, tool_name_override="consult_memory_system"),
        handoff(reflection_agent, tool_name_override="generate_reflection"),
        handoff(decision_agent, tool_name_override="make_narrative_decision"),
    ],
    output_type=NarrativeResponse,
    input_guardrails=[
        InputGuardrail(guardrail_function=content_moderation_guardrail),
    ],
    model_settings=ModelSettings(
        temperature=0.7
    )
)

# Add to nyx_agent_sdk.py

@function_tool
async def determine_image_generation(ctx, response_text: str) -> str:
    """
    Determine if an image should be generated based on response content.
    """
    # Check for keywords indicating action or state changes
    action_keywords = ["now you see", "suddenly", "appears", "reveals", "wearing", "dressed in"]
    scene_keywords = ["the room", "the chamber", "the area", "the location", "the environment"]
    visual_keywords = ["looks like", "is visible", "can be seen", "comes into view"]
    
    # Create a scoring system
    score = 0
    
    # Check for action keywords
    has_action = any(keyword in response_text.lower() for keyword in action_keywords)
    if has_action:
        score += 3
    
    # Check for scene description
    has_scene = any(keyword in response_text.lower() for keyword in scene_keywords)
    if has_scene:
        score += 2
        
    # Check for visual descriptions
    has_visual = any(keyword in response_text.lower() for keyword in visual_keywords)
    if has_visual:
        score += 2
    
    # Only generate images occasionally to avoid overwhelming
    should_generate = score >= 4
    
    # If we're generating an image, extract a prompt
    image_prompt = None
    if should_generate:
        # Try to extract a good image prompt
        try:
            prompt = f"""
            Extract a concise image generation prompt from this text. Focus on describing the visual scene, characters, lighting, and mood:

            {response_text}

            Provide only the image prompt with no additional text or explanations. Keep it under 100 words.
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You extract image generation prompts from text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            image_prompt = response.choices[0].message.content.strip()
        except:
            # If extraction fails, use first 100 words of response
            words = response_text.split()[:100]
            image_prompt = " ".join(words)
    
    return json.dumps({
        "should_generate": should_generate,
        "score": score,
        "has_action": has_action,
        "has_scene": has_scene,
        "has_visual": has_visual,
        "image_prompt": image_prompt
    })

@function_tool
async def get_emotional_state(ctx) -> str:
    """
    Get Nyx's current emotional state from the database.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    async with asyncpg.create_pool(dsn=DB_DSN) as pool:
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT emotional_state FROM NyxAgentState
                WHERE user_id = $1 AND conversation_id = $2
            """, user_id, conversation_id)
            
            if row and row["emotional_state"]:
                return row["emotional_state"]
    
    # Default emotional state
    default_state = {
        "primary_emotion": "neutral",
        "intensity": 0.3,
        "secondary_emotions": {
            "curiosity": 0.4,
            "amusement": 0.2
        },
        "confidence": 0.7
    }
    
    return json.dumps(default_state)

@function_tool
async def update_emotional_state(ctx, emotional_state: Dict[str, Any]) -> str:
    """
    Update Nyx's emotional state in the database.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    async with asyncpg.create_pool(dsn=DB_DSN) as pool:
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO NyxAgentState (user_id, conversation_id, emotional_state, updated_at)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, conversation_id) 
                DO UPDATE SET emotional_state = $3, updated_at = CURRENT_TIMESTAMP
            """, user_id, conversation_id, json.dumps(emotional_state))
    
    return json.dumps({
        "updated": True,
        "emotional_state": emotional_state
    })

# ===== Main Functions =====

async def initialize_agents():
    """Initialize necessary resources for the agents system"""
    # Any initialization needed before using agents
    pass

async def process_user_input(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process user input through the Nyx agent system
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        user_input: User's input text
        context_data: Additional context data
        
    Returns:
        Complete response with narrative and metadata
    """
    # Create agent context
    agent_context = AgentContext(user_id, conversation_id)
    
    # Add any additional context
    if context_data:
        agent_context.conversation_data = context_data
    
    # Create trace for monitoring
    with trace(
        workflow_name="Nyx Roleplay",
        trace_id=f"nyx-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
        # Run the agent
        result = await Runner.run(
            nyx_main_agent,
            user_input,
            context=agent_context
        )
    
    # Get structured output
    narrative_response = result.final_output_as(NarrativeResponse)
    
    # Store messages in database
    await store_messages(user_id, conversation_id, user_input, narrative_response.narrative)
    
    # Return complete response
    return {
        "message": narrative_response.narrative,
        "generate_image": narrative_response.generate_image,
        "image_prompt": narrative_response.image_prompt,
        "tension_level": narrative_response.tension_level,
        "time_advancement": narrative_response.time_advancement,
        "environment_update": narrative_response.environment_description
    }

async def generate_reflection(
    user_id: int,
    conversation_id: int,
    topic: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a reflection from Nyx on a specific topic
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        topic: Optional topic to reflect on
        
    Returns:
        Reflection response
    """
    # Create agent context
    agent_context = AgentContext(user_id, conversation_id)
    
    # Create prompt for reflection
    prompt = f"Generate a reflection about {topic}" if topic else "Generate a reflection about the player based on your memories"
    
    # Run the reflection agent
    result = await Runner.run(
        reflection_agent,
        prompt,
        context=agent_context
    )
    
    # Get structured output
    reflection = result.final_output_as(MemoryReflection)
    
    return {
        "reflection": reflection.reflection,
        "confidence": reflection.confidence,
        "topic": reflection.topic or topic
    }

async def store_messages(user_id: int, conversation_id: int, user_input: str, nyx_response: str):
    """Store user and Nyx messages in database"""
    async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
        async with pool.acquire() as conn:
            # Store user message
            await conn.execute(
                "INSERT INTO messages (conversation_id, sender, content) VALUES ($1, $2, $3)",
                conversation_id, "user", user_input
            )
            
            # Store Nyx message
            await conn.execute(
                "INSERT INTO messages (conversation_id, sender, content) VALUES ($1, $2, $3)",
                conversation_id, "Nyx", nyx_response
            )
