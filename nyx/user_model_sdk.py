# nyx/user_model_sdk.py

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from agents import Agent, function_tool, Runner, trace
from agents import ModelSettings, RunConfig
from pydantic import BaseModel, Field

from db.connection import get_db_connection_context
from nyx.nyx_model_manager import UserModelManager
from utils.caching import USER_MODEL_CACHE

logger = logging.getLogger(__name__)

# ===== Pydantic Models for Structured Outputs =====

class UserPreference(BaseModel):
    """Detected user preference"""
    preference_type: str = Field(..., description="Type of preference (kink, narrative, etc.)")
    preference_name: str = Field(..., description="Name of the preference")
    intensity: float = Field(..., description="Intensity of the preference (0.0-1.0)")
    confidence: float = Field(..., description="Confidence in this detection (0.0-1.0)")
    source: str = Field(..., description="Source of detection (explicit mention, reaction, etc.)")

class BehaviorPattern(BaseModel):
    """Detected behavior pattern"""
    pattern_type: str = Field(..., description="Type of pattern (response style, aggression, etc.)")
    pattern_value: str = Field(..., description="The specific value or nature of the pattern")
    occurrence_count: int = Field(1, description="Number of times this pattern has been observed")
    intensity: float = Field(0.5, description="Intensity of the pattern (0.0-1.0)")
    confidence: float = Field(0.5, description="Confidence in this detection (0.0-1.0)")

class ResponseGuidance(BaseModel):
    """Guidance for Nyx's response based on user model"""
    suggested_intensity: float = Field(..., description="Suggested intensity level (0.0-1.0)")
    suggested_dominance: float = Field(..., description="Suggested dominance level (0.0-1.0)")
    top_interests: List[Dict[str, Any]] = Field(default_factory=list, description="Top interests to focus on")
    avoid_topics: List[str] = Field(default_factory=list, description="Topics to avoid")
    behavior_patterns: Dict[str, Any] = Field(default_factory=dict, description="Relevant behavior patterns")
    custom_guidance: Optional[str] = Field(None, description="Additional custom guidance")

class UserModelAnalysis(BaseModel):
    """Analysis of user input for model updates"""
    detected_preferences: List[UserPreference] = Field(default_factory=list, description="Detected preferences")
    detected_patterns: List[BehaviorPattern] = Field(default_factory=list, description="Detected behavior patterns")
    confidence: float = Field(0.5, description="Overall confidence in analysis")
    suggestions: Optional[str] = Field(None, description="Suggestions for future interactions")

# ===== User Model Context Object =====

class UserModelContext:
    """Context object for user model agents"""
    def __init__(self, user_id: int, conversation_id: int = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.user_model_manager = UserModelManager(user_id, conversation_id)
        self.context_data = {}

# ===== Function Tools =====

@function_tool
async def get_user_model(ctx) -> str:
    """
    Get the current user model.
    """
    user_model_manager = ctx.context.user_model_manager
    
    # Get user model with cache handling
    user_model = await user_model_manager.get_user_model()
    
    return json.dumps(user_model)

@function_tool
async def track_kink_preference(
    ctx,
    kink_name: str,
    intensity: float = 0.5,
    detected_from: str = "analysis"
) -> str:
    """
    Track a detected kink preference.
    
    Args:
        kink_name: Name of the kink (e.g., "ass", "goth", "tattoos")
        intensity: Detected intensity of preference (0.0-1.0)
        detected_from: Source of detection (e.g., "explicit_mention", "reaction")
    """
    user_model_manager = ctx.context.user_model_manager
    
    # Track the preference
    updated_model = await user_model_manager.track_kink_preference(
        kink_name=kink_name,
        intensity=intensity,
        detected_from=detected_from
    )
    
    # Extract the level from the updated model
    level = updated_model.get("kink_profile", {}).get(kink_name, {}).get("level", 0)
    
    return f"Tracked kink preference: {kink_name} with intensity {intensity:.2f} from {detected_from}. Current level: {level}"

@function_tool
async def track_behavior_pattern(
    ctx,
    pattern_type: str,
    pattern_value: str,
    intensity: float = 0.5
) -> str:
    """
    Track a detected behavior pattern.
    
    Args:
        pattern_type: Type of pattern (e.g., "response_style", "aggression")
        pattern_value: Specific value or nature of the pattern
        intensity: Strength of the pattern (0.0-1.0)
    """
    user_model_manager = ctx.context.user_model_manager
    
    # Track the behavior pattern
    updated_model = await user_model_manager.track_behavior_pattern(
        pattern_type=pattern_type,
        pattern_value=pattern_value,
        intensity=intensity
    )
    
    return f"Tracked behavior pattern: {pattern_type}={pattern_value} with intensity {intensity:.2f}"

@function_tool
async def get_response_guidance(ctx) -> str:
    """
    Get guidance for how Nyx should respond based on the user model.
    """
    user_model_manager = ctx.context.user_model_manager
    
    # Get response guidance
    guidance = await user_model_manager.get_response_guidance()
    
    return json.dumps(guidance)

@function_tool
async def update_personality_assessment(
    ctx,
    dominance_preference: int = None,
    intensity_preference: int = None,
    humiliation_tolerance: int = None,
    creative_tolerance: int = None
) -> str:
    """
    Update the personality assessment in the user model.
    
    Args:
        dominance_preference: Preference for domination (-100 to 100 scale)
        intensity_preference: Preference for intensity (0 to 100 scale)
        humiliation_tolerance: Tolerance for humiliation (0 to 100 scale)
        creative_tolerance: Tolerance for creative/surreal content (0 to 100 scale)
    """
    user_model_manager = ctx.context.user_model_manager
    
    # Get current model
    user_model = await user_model_manager.get_user_model()
    
    # Prepare updates
    personality_assessment = user_model.get("personality_assessment", {})
    
    updates = {}
    if dominance_preference is not None:
        updates["dominance_preference"] = max(-100, min(100, dominance_preference))
    if intensity_preference is not None:
        updates["intensity_preference"] = max(0, min(100, intensity_preference))
    if humiliation_tolerance is not None:
        updates["humiliation_tolerance"] = max(0, min(100, humiliation_tolerance))
    if creative_tolerance is not None:
        updates["creative_tolerance"] = max(0, min(100, creative_tolerance))
    
    # Apply updates
    personality_assessment.update(updates)
    
    # Update user model
    await user_model_manager.update_user_model({"personality_assessment": personality_assessment})
    
    return f"Updated personality assessment: {json.dumps(updates)}"

# ===== User Model Agents =====

# Preference Detection Agent
preference_detection_agent = Agent[UserModelContext](
    name="Preference Detection Agent",
    instructions="""You analyze user messages to detect preferences relevant to a femdom roleplay context.
    
Your role is to:
1. Identify explicit preferences directly stated by the user
2. Detect implicit preferences based on reactions and context
3. Assess the intensity and confidence for each detected preference
4. Focus on preferences relevant to femdom roleplay contexts
5. Categorize preferences appropriately (kinks, narrative styles, etc.)

Be attentive to subtle cues and context, but maintain appropriate confidence
levels based on the evidence available.""",
    output_type=UserPreference
)

# Behavior Pattern Analysis Agent
behavior_analysis_agent = Agent[UserModelContext](
    name="Behavior Pattern Analysis Agent",
    instructions="""You analyze user behavior patterns in conversations.
    
Your role is to:
1. Identify recurring patterns in how the user responds
2. Detect patterns related to submission, resistance, etc.
3. Analyze communication style preferences
4. Assess interaction patterns with NPCs and the environment
5. Determine patterns related to narrative preferences

Focus on patterns that will help Nyx understand how to better engage with the user.""",
    output_type=BehaviorPattern,
    tools=[get_user_model]
)

# Response Guidance Agent
response_guidance_agent = Agent[UserModelContext](
    name="Response Guidance Agent",
    instructions="""You generate guidance for how Nyx should respond based on the user model.
    
Your role is to:
1. Recommend appropriate intensity levels for responses
2. Suggest preferred themes based on user interests
3. Identify topics or approaches to avoid
4. Provide guidance on tone, language, and style
5. Balance pushing boundaries with respecting implied limits

Your guidance should help Nyx create responses that align with user preferences
while maintaining Nyx's dominant, confident personality.""",
    output_type=ResponseGuidance,
    tools=[get_user_model, get_response_guidance]
)

# User Model Manager Agent (Orchestrator)
user_model_manager_agent = Agent[UserModelContext](
    name="User Model Manager Agent",
    instructions="""You orchestrate the user modeling system for Nyx.
    
Your role is to:
1. Analyze user input for preference and behavior revelations
2. Coordinate with specialized agents to update the user model
3. Maintain a cohesive and consistent model of the user
4. Ensure appropriate confidence levels for model components
5. Generate guidance for Nyx based on the user model

Manage the user model to help Nyx understand the user's preferences,
boundaries, and interaction patterns.""",
    handoffs=[
        handoff(preference_detection_agent, tool_name_override="detect_preferences"),
        handoff(behavior_analysis_agent, tool_name_override="analyze_behavior"),
        handoff(response_guidance_agent, tool_name_override="get_response_guidance")
    ],
    tools=[
        get_user_model,
        track_kink_preference,
        track_behavior_pattern,
        update_personality_assessment
    ],
    output_type=UserModelAnalysis
)

# ===== Main Functions =====

async def process_user_input_for_model(
    user_id: int,
    conversation_id: int,
    user_input: str,
    nyx_response: str = None,
    context_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process user input to update the user model
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        user_input: User's input text
        nyx_response: Optional response from Nyx to evaluate user reaction
        context_data: Additional context data
        
    Returns:
        Update results and model changes
    """
    # Create user model context
    user_model_context = UserModelContext(user_id, conversation_id)
    
    # Add any additional context
    if context_data:
        user_model_context.context_data = context_data
    
    # Create prompt based on available information
    if nyx_response:
        prompt = f"""
Analyze this interaction to update the user model:

User: {user_input}

Nyx: {nyx_response}

Consider both the user's message and their response to Nyx.
"""
    else:
        prompt = f"""
Analyze this user message to update the user model:

User: {user_input}

Focus on detecting preferences, boundaries, and behavior patterns.
"""
    
    # Create trace for monitoring
    with trace(
        workflow_name="User Model",
        trace_id=f"user-model-{conversation_id}-{int(datetime.now().timestamp())}",
        group_id=f"user-{user_id}"
    ):
        # Run the user model manager agent
        result = await Runner.run(
            user_model_manager_agent,
            prompt,
            context=user_model_context
        )
    
    # Get structured output
    analysis = result.final_output_as(UserModelAnalysis)
    
    # Create summary of changes
    changes = {
        "preferences_detected": [
            {
                "type": pref.preference_type,
                "name": pref.preference_name,
                "intensity": pref.intensity,
                "confidence": pref.confidence
            }
            for pref in analysis.detected_preferences
        ],
        "patterns_detected": [
            {
                "type": pattern.pattern_type,
                "value": pattern.pattern_value,
                "intensity": pattern.intensity,
                "confidence": pattern.confidence
            }
            for pattern in analysis.detected_patterns
        ],
        "overall_confidence": analysis.confidence,
        "suggestions": analysis.suggestions
    }
    
    return changes

async def get_response_guidance_for_user(
    user_id: int, 
    conversation_id: int,
    context_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Get response guidance based on the user model
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        context_data: Additional context data
        
    Returns:
        Response guidance
    """
    # Create user model context
    user_model_context = UserModelContext(user_id, conversation_id)
    
    # Add any additional context
    if context_data:
        user_model_context.context_data = context_data
    
    # Run the response guidance agent
    result = await Runner.run(
        response_guidance_agent,
        "Generate response guidance based on the current user model",
        context=user_model_context
    )
    
    # Get structured output
    guidance = result.final_output_as(ResponseGuidance)
    
    return {
        "suggested_intensity": guidance.suggested_intensity,
        "suggested_dominance": guidance.suggested_dominance,
        "top_interests": guidance.top_interests,
        "avoid_topics": guidance.avoid_topics,
        "behavior_patterns": guidance.behavior_patterns,
        "custom_guidance": guidance.custom_guidance
    }

@function_tool
async def calculate_suggested_intensity(ctx, user_model: Dict[str, Any]) -> str:
    """Calculate suggested intensity level based on user model."""
    # Start with base preference from personality assessment
    base_intensity = user_model.get("personality_assessment", {}).get("intensity_preference", 50) / 100.0
    
    # Adjust based on behavior patterns
    behavior = user_model.get("behavior_patterns", {})
    
    # If user has shown positive response to intensity, increase
    aggression = behavior.get("aggression", {}).get("occurrences", 0)
    submission = behavior.get("submission", {}).get("occurrences", 0)
    
    # More aggressive users might want slightly higher intensity
    if aggression > submission:
        base_intensity += 0.1
    
    # Cap between 0.2 and 0.9
    intensity = max(0.2, min(0.9, base_intensity))
    
    return json.dumps({"suggested_intensity": intensity})

@function_tool
async def format_behavior_patterns(ctx, behavior_patterns: Dict[str, Any]) -> str:
    """Format behavior patterns for guidance."""
    result = {}
    
    for category, data in behavior_patterns.items():
        if "values" in data:
            # Find most common value
            most_common = sorted(
                [(v, d["count"]) for v, d in data["values"].items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            if most_common:
                result[category] = most_common[0][0]
    
    return json.dumps(result)

# Add to user_model_sdk.py

@function_tool
async def track_conversation_response(
    ctx,
    user_message: str,
    nyx_response: str,
    user_reaction: str = None,
    conversation_context: Dict[str, Any] = None
) -> str:
    """
    Track a conversation interaction to learn from user responses.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    user_model_manager = ctx.context.user_model_manager
    
    # Get current model
    user_model = await user_model_manager.get_user_model()
    
    # Get conversation history
    conversation_patterns = user_model.get("conversation_patterns", {
        "response_types": {},
        "reaction_patterns": {},
        "tracked_conversations": []
    })
    
    # Basic tracking of conversation
    tracked_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_message": user_message[:100],  # Truncate for storage
        "nyx_response_sample": nyx_response[:100],  # Truncate
        "context": conversation_context,
        "user_reaction": user_reaction
    }
    
    conversation_patterns["tracked_conversations"].append(tracked_entry)
    
    # Keep only most recent 20 conversations
    if len(conversation_patterns["tracked_conversations"]) > 20:
        conversation_patterns["tracked_conversations"] = conversation_patterns["tracked_conversations"][-20:]
    
    # Store memory
    await user_model_manager.memory_system.add_memory(
        memory_text=f"Conversation interaction - User: '{user_message[:50]}...' Nyx: '{nyx_response[:50]}...'",
        memory_type="observation",
        memory_scope="user",
        significance=3,
        tags=["conversation_pattern"],
        metadata={
            "user_message": user_message[:200],
            "nyx_response": nyx_response[:200],
            "user_reaction": user_reaction,
            "context": conversation_context
        }
    )
    
    # Update user model
    await user_model_manager.update_user_model({"conversation_patterns": conversation_patterns})
    
    return f"Tracked conversation response for user {user_id}"

@function_tool
async def analyze_user_revelations(
    ctx,
    user_message: str,
    nyx_response: str = None,
    context: Dict[str, Any] = None
) -> str:
    """
    Analyze user message for potential preference revelations and behavior patterns.
    """
    lower_message = user_message.lower()
    revelations = []
    
    # Check for explicit kink mentions
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
    
    # Check for behavior patterns
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
        
    # If no revelations found through simple means, use more sophisticated analysis
    if not revelations and len(user_message.split()) > 5:
        # Create a prompt for detecting implicit preferences
        prompt = f"""
        Analyze this user message for potential preferences or behavior patterns in a femdom roleplay context:
        
        User: "{user_message}"
        
        Identify:
        1. Any explicit or implicit kink preferences
        2. Behavioral tendencies (submission, resistance, etc.)
        3. Communication style preferences
        4. Response to dominant themes
        
        Format your response as JSON with 'revelations' as a list of objects, each containing:
        - type: "kink_preference" or "behavior_pattern"
        - name/pattern: The specific preference or pattern
        - intensity: A value from 0 to 1 indicating strength
        - source: How this was detected
        - confidence: How confident you are in this detection (0-1)
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You analyze user messages to detect preferences and patterns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            # Try to parse JSON response
            try:
                analysis = json.loads(response.choices[0].message.content)
                if "revelations" in analysis:
                    revelations.extend(analysis["revelations"])
            except:
                # If JSON parsing fails, extract what we can
                pass
        except Exception as e:
            pass
    
    # Track any detected revelations
    if revelations:
        user_model_manager = ctx.context.user_model_manager
        
        for revelation in revelations:
            if revelation["type"] == "kink_preference":
                await user_model_manager.track_kink_preference(
                    kink_name=revelation["kink"],
                    intensity=revelation["intensity"],
                    detected_from=revelation["source"]
                )
            elif revelation["type"] == "behavior_pattern":
                await user_model_manager.track_behavior_pattern(
                    pattern_type="response_style",
                    pattern_value=revelation["pattern"],
                    intensity=revelation["intensity"]
                )
    
    return json.dumps({"revelations": revelations})

async def initialize_user_model(user_id: int) -> Dict[str, Any]:
    """
    Initialize a new user model if one doesn't exist
    
    Args:
        user_id: User ID
        
    Returns:
        Initialized user model
    """
    # Create user model context
    user_model_context = UserModelContext(user_id)
    
    # Get user model (which initializes a new one if needed)
    user_model = await user_model_context.user_model_manager.get_user_model()
    
    return user_model

def _process_user_interaction(self, interaction: Dict[str, Any]):
    """Process a user interaction for model updates"""
    try:
        # Extract features
        features = self._extract_interaction_features(interaction)
        
        # Update behavioral model
        self._update_behavioral_model(features)
        
        # Update preference model
        self._update_preference_model(features)
        
        # Update emotional model
        self._update_emotional_model(features)
        
        # Save updates
        self._save_model_updates()
        
    except Exception as e:
        logger.error(f"Failed to process user interaction: {e}")
        raise

def _process_model_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
    """Process a query against the user model"""
    try:
        # Parse query
        query_type = query.get("type", "general")
        query_params = query.get("params", {})
        
        # Get relevant model components
        components = self._get_relevant_components(query_type)
        
        # Generate response
        response = self._generate_model_response(components, query_params)
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to process model query: {e}")
        raise
