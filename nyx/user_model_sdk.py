# nyx/user_model_sdk.py

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from agents import Agent, function_tool, Runner, trace, handoff
from agents import ModelSettings, RunConfig
from pydantic import BaseModel, Field

from db.connection import get_db_connection_context
from nyx.nyx_profile_agents import ProfilingAgent, ResponseAnalysisAgent
from nyx.nyx_profile_integration import ProfileIntegration
from nyx.nyx_reinforcement import ReinforcementAgent
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

# ===== User Model Manager Class =====

class UserModelManager:
    """
    Manages user modeling by integrating various Nyx profiling systems
    """
    _instances = {}
    
    @classmethod
    async def get_instance(cls, user_id: int, conversation_id: int = None) -> 'UserModelManager':
        """
        Get or create a UserModelManager instance for the specified user and conversation.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID (optional)
            
        Returns:
            UserModelManager instance
        """
        # Create a unique key for this user/conversation combination
        key = f"{user_id}:{conversation_id if conversation_id is not None else 'global'}"
        
        # Check if an instance already exists for this key
        if key not in cls._instances:
            # Create a new instance if none exists
            cls._instances[key] = cls(user_id, conversation_id)
            
        return cls._instances[key]
    
    def __init__(self, user_id: int, conversation_id: int = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.profile_integration = ProfileIntegration()
        self.profiling_agent = ProfilingAgent()
        self.response_agent = ResponseAnalysisAgent()
        self.reinforcement_agent = ReinforcementAgent()
        self.user_model_cache = {}
        self.last_model_update = datetime.now()
        
    async def get_user_model(self) -> Dict[str, Any]:
        """Get the current user model"""
        # Check cache first
        cache_key = f"user_model:{self.user_id}"
        cached_model = USER_MODEL_CACHE.get(cache_key)
        
        if cached_model:
            return cached_model
            
        # Fetch from database
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT model_data FROM UserModels 
                WHERE user_id = $1
            """, self.user_id)
            
            if row:
                user_model = json.loads(row["model_data"])
                # Cache the model
                USER_MODEL_CACHE.set(cache_key, user_model, 300)  # 5 minute TTL
                return user_model
                
        # Initialize a new model if not found
        return await self._initialize_new_model()
    
    async def track_kink_preference(
        self,
        kink_name: str,
        intensity: float = 0.5,
        detected_from: str = "analysis"
    ) -> Dict[str, Any]:
        """Track a detected kink preference"""
        # Get current model
        user_model = await self.get_user_model()
        
        # Initialize kink profile if not exists
        if "kink_profile" not in user_model:
            user_model["kink_profile"] = {}
            
        # Initialize specific kink if not exists
        if kink_name not in user_model["kink_profile"]:
            user_model["kink_profile"][kink_name] = {
                "level": 0,
                "confidence": 0,
                "detections": []
            }
            
        # Add detection
        detection = {
            "timestamp": datetime.now().isoformat(),
            "intensity": intensity,
            "source": detected_from
        }
        user_model["kink_profile"][kink_name]["detections"].append(detection)
        
        # Update level using weighted average
        detections = user_model["kink_profile"][kink_name]["detections"]
        total_weight = 0
        weighted_sum = 0
        
        for i, det in enumerate(detections):
            # More recent detections have higher weight
            weight = 1 + (i * 0.1)
            weighted_sum += det["intensity"] * weight
            total_weight += weight
            
        if total_weight > 0:
            user_model["kink_profile"][kink_name]["level"] = weighted_sum / total_weight
            
        # Update confidence based on number of detections
        detection_count = len(detections)
        user_model["kink_profile"][kink_name]["confidence"] = min(0.9, 0.3 + (detection_count * 0.1))
        
        # Update the model
        await self.update_user_model(user_model)
        
        return user_model
    
    async def track_behavior_pattern(
        self,
        pattern_type: str,
        pattern_value: str,
        intensity: float = 0.5
    ) -> Dict[str, Any]:
        """Track a detected behavior pattern"""
        # Get current model
        user_model = await self.get_user_model()
        
        # Initialize behavior patterns if not exists
        if "behavior_patterns" not in user_model:
            user_model["behavior_patterns"] = {}
            
        # Initialize specific pattern type if not exists
        if pattern_type not in user_model["behavior_patterns"]:
            user_model["behavior_patterns"][pattern_type] = {
                "values": {},
                "last_updated": datetime.now().isoformat()
            }
            
        # Initialize specific value if not exists
        if pattern_value not in user_model["behavior_patterns"][pattern_type]["values"]:
            user_model["behavior_patterns"][pattern_type]["values"][pattern_value] = {
                "count": 0,
                "intensity": 0,
                "last_seen": None
            }
            
        # Update pattern
        pattern_data = user_model["behavior_patterns"][pattern_type]["values"][pattern_value]
        pattern_data["count"] += 1
        
        # Update intensity using exponential moving average
        alpha = 0.3  # Weight for new observation
        pattern_data["intensity"] = (alpha * intensity) + ((1 - alpha) * pattern_data["intensity"])
        
        pattern_data["last_seen"] = datetime.now().isoformat()
        user_model["behavior_patterns"][pattern_type]["last_updated"] = datetime.now().isoformat()
        
        # Update the model
        await self.update_user_model(user_model)
        
        return user_model
    
    async def get_response_guidance(self) -> Dict[str, Any]:
        """Get guidance for Nyx's response based on user model"""
        # Get current model
        user_model = await self.get_user_model()
        
        # Get insights from profile integration
        current_profile = await self._convert_model_to_profile(user_model)
        profile_insights = await self.profile_integration.get_profile_insights(current_profile, {})
        
        # Use reinforcement agent to enhance guidance
        reinforcement_state = {
            "player_desperation": self.reinforcement_agent.player_desperation,
            "emotional_state": user_model.get("emotional_state", {})
        }
        
        # Calculate suggested intensity
        intensity = await self._calculate_suggested_intensity(user_model)
        
        # Get top interests
        kink_profile = user_model.get("kink_profile", {})
        top_interests = sorted(
            [(k, v["level"]) for k, v in kink_profile.items() if v["confidence"] > 0.5],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Get behavior patterns
        behavior_patterns = await self._format_behavior_patterns(user_model.get("behavior_patterns", {}))
        
        # Put together guidance
        guidance = {
            "suggested_intensity": intensity,
            "suggested_dominance": profile_insights.get("personality_preferences", {}).get("dominant", 0.7),
            "top_interests": [{"name": name, "level": level} for name, level in top_interests],
            "avoid_topics": user_model.get("avoid_topics", []),
            "behavior_patterns": behavior_patterns,
            "reinforcement_state": reinforcement_state
        }
        
        # Add custom guidance based on detected patterns
        if "dominant" in behavior_patterns.get("response_style", ""):
            guidance["custom_guidance"] = "User responds well to firm, direct commands."
        elif "submissive" in behavior_patterns.get("response_style", ""):
            guidance["custom_guidance"] = "User responds well to praise and gentle dominance."
            
        return guidance
    
    async def update_user_model(self, updated_model: Dict[str, Any]) -> Dict[str, Any]:
        """Update the user model"""
        # Merge if partial update
        if not isinstance(updated_model, dict):
            raise ValueError("Updated model must be a dictionary")
            
        full_model = await self.get_user_model()
        
        # Apply updates
        for key, value in updated_model.items():
            if isinstance(value, dict) and key in full_model and isinstance(full_model[key], dict):
                # Merge nested dictionaries
                full_model[key].update(value)
            else:
                # Replace or add values
                full_model[key] = value
                
        # Add metadata
        full_model["last_updated"] = datetime.now().isoformat()
        
        # Save to database
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO UserModels (user_id, model_data, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (user_id) 
                DO UPDATE SET model_data = $2, updated_at = NOW()
            """, self.user_id, json.dumps(full_model))
            
        # Update cache
        cache_key = f"user_model:{self.user_id}"
        USER_MODEL_CACHE.set(cache_key, full_model, 300)  # 5 minute TTL
        
        return full_model
    
    async def analyze_text_for_preferences(self, text: str) -> Dict[str, Any]:
        """Analyze text for preferences using profiling agent"""
        # Create event and context
        event = {
            "content": text,
            "type": "user_message"
        }
        
        current_profile = await self._get_current_profile()
        
        # Use profiling agent to analyze
        profile_analysis = await self.profiling_agent.analyze_interaction(event, current_profile)
        
        # Extract revelations
        revelations = []
        
        # Process new insights
        for category, insight in profile_analysis.get("new_insights", {}).items():
            # Extract kink preferences
            if category in ["bdsm", "roleplay", "exhibition"]:
                revelations.append({
                    "type": "kink_preference",
                    "kink": category,
                    "intensity": insight.get("intensity", 0.5),
                    "source": "profile_analysis"
                })
                
            # Extract personality preferences
            elif category in ["dominant", "submissive", "playful"]:
                revelations.append({
                    "type": "behavior_pattern",
                    "pattern": category,
                    "intensity": insight.get("intensity", 0.5),
                    "source": "profile_analysis"
                })
                
        return {
            "revelations": revelations,
            "raw_analysis": profile_analysis
        }
        
    async def _initialize_new_model(self) -> Dict[str, Any]:
        """Initialize a new user model"""
        # Create basic model structure
        model = {
            "user_id": self.user_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "kink_profile": {},
            "behavior_patterns": {},
            "personality_assessment": {
                "dominance_preference": 50,
                "intensity_preference": 50,
                "humiliation_tolerance": 50,
                "creative_tolerance": 50
            },
            "conversation_patterns": {
                "response_types": {},
                "reaction_patterns": {},
                "tracked_conversations": []
            },
            "avoid_topics": []
        }
        
        # Save to database
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO UserModels (user_id, model_data, created_at, updated_at)
                VALUES ($1, $2, NOW(), NOW())
                ON CONFLICT (user_id) DO NOTHING
            """, self.user_id, json.dumps(model))
            
        # Cache the model
        cache_key = f"user_model:{self.user_id}"
        USER_MODEL_CACHE.set(cache_key, model, 300)  # 5 minute TTL
        
        return model
    
    async def _get_current_profile(self) -> Dict[str, Any]:
        """Get current profile in format expected by profiling agent"""
        # Get user model
        user_model = await self.get_user_model()
        
        # Convert to profile format
        profile = await self._convert_model_to_profile(user_model)
        
        return profile
    
    async def _convert_model_to_profile(self, user_model: Dict[str, Any]) -> Dict[str, Any]:
        """Convert user model to profile format for compatibility with profiling agent"""
        profile = {
            "kink_preferences": {},
            "physical_preferences": {},
            "personality_preferences": {},
            "observed_patterns": []
        }
        
        # Convert kink preferences
        for kink, data in user_model.get("kink_profile", {}).items():
            if data.get("confidence", 0) > 0.4:  # Only include preferences with reasonable confidence
                profile["kink_preferences"][kink] = data.get("level", 0.5)
                
        # Convert behavior patterns to personality preferences
        for pattern_type, pattern_data in user_model.get("behavior_patterns", {}).items():
            if pattern_type == "response_style":
                for value, data in pattern_data.get("values", {}).items():
                    if value in ["dominant", "submissive", "playful", "serious"]:
                        profile["personality_preferences"][value] = data.get("intensity", 0.5)
            
        # Add timestamps for compatibility
        now = datetime.now().isoformat()
        pattern = {
            "timestamp": now,
            "observations": {}
        }
        
        for category, values in profile.items():
            if isinstance(values, dict):
                pattern["observations"][category] = values
                
        profile["observed_patterns"].append(pattern)
        
        return profile
    
    async def _calculate_suggested_intensity(self, user_model: Dict[str, Any]) -> float:
        """Calculate suggested intensity level based on user model"""
        # Start with base preference from personality assessment
        base_intensity = user_model.get("personality_assessment", {}).get("intensity_preference", 50) / 100.0
        
        # Adjust based on behavior patterns
        behavior = user_model.get("behavior_patterns", {})
        
        # If user has shown positive response to intensity, increase
        aggression_data = behavior.get("aggression", {}).get("values", {}).get("high", {})
        submission_data = behavior.get("response_style", {}).get("values", {}).get("submissive", {})
        
        aggression = aggression_data.get("count", 0) if aggression_data else 0
        submission = submission_data.get("count", 0) if submission_data else 0
        
        # More aggressive users might want slightly higher intensity
        if aggression > submission:
            base_intensity += 0.1
        
        # Cap between 0.2 and 0.9
        intensity = max(0.2, min(0.9, base_intensity))
        
        return intensity
    
    async def _format_behavior_patterns(self, behavior_patterns: Dict[str, Any]) -> Dict[str, str]:
        """Format behavior patterns for guidance"""
        result = {}
        
        for category, data in behavior_patterns.items():
            if "values" in data:
                # Find most common value
                most_common = sorted(
                    [(v, d.get("count", 0)) for v, d in data["values"].items()],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                if most_common:
                    result[category] = most_common[0][0]
        
        return result
    
    async def add_memory(self, memory_text: str, memory_type: str, memory_scope: str,
                        significance: int, tags: List[str], metadata: Dict[str, Any] = None):
        """Compatibility with memory system"""
        # This would integrate with a memory system
        # For now, just add as part of user model
        user_model = await self.get_user_model()
        
        if "memories" not in user_model:
            user_model["memories"] = []
            
        memory = {
            "text": memory_text,
            "type": memory_type,
            "scope": memory_scope,
            "significance": significance,
            "tags": tags,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        user_model["memories"].append(memory)
        
        # Keep only most important memories
        if len(user_model["memories"]) > 50:
            user_model["memories"] = sorted(
                user_model["memories"],
                key=lambda x: x["significance"],
                reverse=True
            )[:50]
            
        await self.update_user_model({"memories": user_model["memories"]})

# ===== User Model Context =====

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
    await user_model_manager.add_memory(
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
        user_model_manager = ctx.context.user_model_manager
        
        # Use the LLM for more sophisticated analysis through UserModelManager
        sophisticated_analysis = await user_model_manager.analyze_text_for_preferences(user_message)
        if sophisticated_analysis and sophisticated_analysis.get("revelations"):
            revelations.extend(sophisticated_analysis["revelations"])
    
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
