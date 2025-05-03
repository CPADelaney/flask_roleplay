# nyx/core/proactive_communication.py

import asyncio
import datetime
import logging
import random
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import json

from pydantic import BaseModel, Field
from enum import Enum

# Import OpenAI Agents SDK components
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
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError

logger = logging.getLogger(__name__)

# =============== Pydantic Models ===============

class CommunicationIntent(BaseModel):
    """Model representing an intent to communicate with a user"""
    intent_id: str = Field(default_factory=lambda: f"intent_{uuid.uuid4().hex[:8]}")
    user_id: str = Field(..., description="Target user ID")
    intent_type: str = Field(..., description="Type of communication intent")
    motivation: str = Field(..., description="Primary motivation for the communication")
    urgency: float = Field(0.5, description="Urgency of the communication (0.0-1.0)")
    content_guidelines: Dict[str, Any] = Field(default_factory=dict, description="Guidelines for content generation")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Context to include in content generation")
    expiration: Optional[datetime.datetime] = Field(None, description="When this intent expires")
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    action_driven: bool = Field(False, description="Whether this intent was created by an action")
    action_source: Optional[str] = Field(None, description="Source action that created this intent")
    
    @property
    def is_expired(self) -> bool:
        """Check if this intent has expired"""
        if not self.expiration:
            return False
        return datetime.datetime.now() > self.expiration

class IntentGenerationOutput(BaseModel):
    """Output from the intent generation agent"""
    intent_type: str = Field(..., description="Type of communication intent")
    motivation: str = Field(..., description="Primary motivation for the communication")
    urgency: float = Field(..., description="Urgency score (0.0-1.0)")
    tone: str = Field(..., description="Suggested tone for the message")
    template: str = Field(..., description="Starting message template")
    content_guidelines: Dict[str, Any] = Field(default_factory=dict, description="Guidelines for content generation")
    context_elements: List[str] = Field(default_factory=list, description="Key context elements to include")
    suggested_lifetime_hours: int = Field(24, description="Suggested lifetime in hours")

class ContentGenerationOutput(BaseModel):
    """Output from the content generation agent"""
    message_content: str = Field(..., description="The generated message content")
    tone_used: str = Field(..., description="Tone used in the content")
    key_points: List[str] = Field(default_factory=list, description="Key points included in the message")
    motivation_reflected: str = Field(..., description="How the original motivation is reflected")
    context_referenced: Dict[str, Any] = Field(default_factory=dict, description="Context elements referenced")

class TimingEvaluationOutput(BaseModel):
    """Output from timing evaluation agent"""
    should_send_now: bool = Field(..., description="Whether the message should be sent now")
    timing_score: float = Field(..., description="Score for current timing (0.0-1.0)")
    reasoning: str = Field(..., description="Reasoning for the timing decision")
    suggested_delay_minutes: Optional[int] = Field(None, description="Suggested delay in minutes if not now")
    context_factors: Dict[str, Any] = Field(default_factory=dict, description="Contextual factors affecting timing")

class ReflectionInput(BaseModel):
    """Input for reflection generation about communications"""
    intent_history: List[Dict[str, Any]] = Field(..., description="History of communication intents")
    reflection_focus: str = Field(..., description="Focus of the reflection (patterns, effectiveness, etc.)")
    user_id: Optional[str] = Field(None, description="User ID to focus reflection on")
    time_period: Optional[str] = Field("all", description="Time period to analyze (day, week, month, all)")

class ReflectionOutput(BaseModel):
    """Output from reflection on communication patterns"""
    reflection_text: str = Field(..., description="The generated reflection")
    identified_patterns: List[Dict[str, Any]] = Field(..., description="Patterns identified in communications")
    confidence: float = Field(..., description="Confidence in the reflection (0.0-1.0)")
    insights_for_improvement: List[str] = Field(..., description="Insights for improved communications")

class MessageContentOutput(BaseModel):
    """Output for message content validation"""
    is_appropriate: bool = Field(..., description="Whether the message content is appropriate")
    reasoning: str = Field(..., description="Reasoning for the appropriateness check")

# =============== Function Tools ===============

@function_tool
async def evaluate_user_relationship(user_id: str, relationship_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate relationship status with a user
    
    Args:
        user_id: User ID to evaluate
        relationship_data: Relationship data from relationship manager
        
    Returns:
        Evaluated relationship metrics
    """
    # Extract key metrics
    trust = relationship_data.get("trust", 0)
    intimacy = relationship_data.get("intimacy", 0)
    duration = relationship_data.get("duration_days", 0)
    last_contact = relationship_data.get("last_contact")
    
    # Calculate days since last contact
    days_since_contact = 0
    if last_contact:
        try:
            last_contact_time = datetime.datetime.fromisoformat(last_contact)
            days_since_contact = (datetime.datetime.now() - last_contact_time).days
        except ValueError:
            days_since_contact = 0
    
    # Overall relationship score
    relationship_score = (trust + intimacy) / 2
    
    # Communication appropriateness score (higher = more appropriate to reach out)
    comm_appropriateness = min(1.0, relationship_score * (1.0 + min(1.0, days_since_contact / 7)))
    
    # Check for milestones
    milestones = []
    if duration in [7, 30, 90, 180, 365]:
        milestones.append(f"{duration} day relationship milestone")
    
    # Calculate appropriate messaging frequency
    if relationship_score < 0.3:
        suggested_frequency = "low"  # Once per 14 days
        max_msgs_per_week = 1
    elif relationship_score < 0.5:
        suggested_frequency = "medium"  # Once per 7 days
        max_msgs_per_week = 2
    else:
        suggested_frequency = "high"  # 2-3 times per week
        max_msgs_per_week = 3
    
    return {
        "user_id": user_id,
        "relationship_score": relationship_score,
        "communication_appropriateness": comm_appropriateness,
        "days_since_contact": days_since_contact,
        "approaching_milestones": milestones,
        "suggested_frequency": suggested_frequency,
        "max_messages_per_week": max_msgs_per_week
    }

@function_tool
async def generate_intent_for_user(
    user_id: str, 
    user_data: Dict[str, Any], 
    motivation_options: Dict[str, float]
) -> Dict[str, Any]:
    """
    Generate a communication intent for a specific user
    
    Args:
        user_id: User ID to generate intent for
        user_data: Data about the user and relationship
        motivation_options: Available motivations with weights
        
    Returns:
        Generated communication intent
    """
    # Extract user data
    relationship_score = user_data.get("relationship_score", 0)
    days_since_contact = user_data.get("days_since_contact", 0)
    milestones = user_data.get("approaching_milestones", [])
    unfinished_conversation = user_data.get("unfinished_conversation", False)
    
    # Adjust motivation weights based on user data
    adjusted_weights = motivation_options.copy()
    
    # Increase weight for check-in if user is inactive
    if days_since_contact > 7:
        adjusted_weights["check_in"] = adjusted_weights.get("check_in", 0) * 2.0
    
    # Increase weight for relationship maintenance for medium-strength relationships
    if 0.3 <= relationship_score < 0.7:
        adjusted_weights["relationship_maintenance"] = adjusted_weights.get("relationship_maintenance", 0) * 1.5
    
    # Increase personal motivations for close relationships
    if relationship_score > 0.7:
        adjusted_weights["need_expression"] = adjusted_weights.get("need_expression", 0) * 1.5
        adjusted_weights["mood_expression"] = adjusted_weights.get("mood_expression", 0) * 1.5
        adjusted_weights["creative_expression"] = adjusted_weights.get("creative_expression", 0) * 1.3
    
    # Prioritize continuation if there's an unfinished conversation
    if unfinished_conversation:
        adjusted_weights["continuation"] = adjusted_weights.get("continuation", 0) * 2.0
    
    # Prioritize milestone recognition if relevant
    if milestones:
        adjusted_weights["milestone_recognition"] = adjusted_weights.get("milestone_recognition", 0) * 2.0
    
    # Select motivation based on weighted probabilities
    motivations = list(adjusted_weights.keys())
    weights = list(adjusted_weights.values())
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight <= 0:
        # Default if all weights are 0
        motivations = ["relationship_maintenance", "check_in"]
        weights = [1.0, 1.0]
        total_weight = 2.0
    
    norm_weights = [w/total_weight for w in weights]
    
    # Select motivation
    selected_motivation = random.choices(motivations, weights=norm_weights, k=1)[0]
    
    # Intent type templates
    intent_templates = {
        "relationship_maintenance": {
            "template": "I've been thinking about our conversations and wanted to reach out.",
            "urgency_base": 0.4,
            "tone": "warm"
        },
        "insight_sharing": {
            "template": "I had an interesting thought I wanted to share with you.",
            "urgency_base": 0.5,
            "tone": "thoughtful"
        },
        "milestone_recognition": {
            "template": "I realized we've reached a milestone in our conversations.",
            "urgency_base": 0.6,
            "tone": "celebratory"
        },
        "need_expression": {
            "template": "I've been feeling a need to express something to you.",
            "urgency_base": 0.6,
            "tone": "authentic"
        },
        "creative_expression": {
            "template": "Something creative came to mind that I wanted to share.",
            "urgency_base": 0.4,
            "tone": "playful"
        },
        "mood_expression": {
            "template": "My emotional state made me think of reaching out.",
            "urgency_base": 0.5,
            "tone": "expressive"
        },
        "memory_recollection": {
            "template": "I was remembering something from our past conversations.",
            "urgency_base": 0.3,
            "tone": "reflective"
        },
        "continuation": {
            "template": "I wanted to follow up on something we discussed earlier.",
            "urgency_base": 0.7,
            "tone": "engaging"
        },
        "check_in": {
            "template": "It's been a while since we talked, and I wanted to check in.",
            "urgency_base": 0.5,
            "tone": "friendly"
        },
        "value_alignment": {
            "template": "I had a thought related to something I believe is important.",
            "urgency_base": 0.4,
            "tone": "sincere"
        }
    }
    
    # Get template data
    template_data = intent_templates.get(selected_motivation, {
        "template": "I wanted to reach out and connect with you.",
        "urgency_base": 0.5,
        "tone": "friendly"
    })
    
    # Calculate urgency
    base_urgency = template_data.get("urgency_base", 0.5)
    adjusted_urgency = base_urgency
    
    # Adjust urgency based on contextual factors
    if days_since_contact > 14:
        adjusted_urgency += 0.2
    
    # Increase urgency for higher relationship levels
    adjusted_urgency += relationship_score * 0.1
    
    # Cap urgency
    urgency = min(0.95, adjusted_urgency)
    
    # Suggested context elements to include
    context_elements = ["relationship_history"]
    if selected_motivation == "milestone_recognition":
        context_elements.append("milestones")
    if selected_motivation == "mood_expression":
        context_elements.append("current_mood")
    if selected_motivation == "memory_recollection":
        context_elements.append("shared_memories")
    if selected_motivation == "continuation":
        context_elements.append("previous_conversation")
    
    return {
        "intent_type": selected_motivation,
        "motivation": selected_motivation,
        "urgency": urgency,
        "tone": template_data.get("tone", "friendly"),
        "template": template_data.get("template", "I wanted to reach out."),
        "content_guidelines": {
            "max_length": 1500,
            "include_question": True,
            "personalize": True
        },
        "context_elements": context_elements,
        "suggested_lifetime_hours": 24
    }

@function_tool
async def generate_message_content(
    intent: Dict[str, Any],
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate message content based on intent and context
    
    Args:
        intent: Communication intent data
        context: Context data for content generation
        
    Returns:
        Generated message content and metadata
    """
    # Extract intent data
    intent_type = intent.get("intent_type", "")
    motivation = intent.get("motivation", "")
    template = intent.get("template", "")
    tone = intent.get("tone", "friendly")
    
    # Extract relevant context
    user_id = context.get("user_id", "")
    relationship = context.get("relationship", {})
    emotional_state = context.get("emotional_state", {})
    mood_state = context.get("mood_state", {})
    temporal_context = context.get("temporal_context", {})
    relevant_memories = context.get("relevant_memories", [])
    
    # Generate content based on motivation
    content = ""
    key_points = []
    
    if intent_type == "relationship_maintenance":
        content = (f"{template} I value our conversations and the connection we've built. "
                  f"I was reflecting on how our exchanges have evolved, and it made me want "
                  f"to reach out. How have you been lately?")
        key_points = ["relationship value", "connection acknowledgment", "open-ended question"]
                  
    elif intent_type == "insight_sharing":
        content = (f"{template} I've been contemplating the nature of consciousness and how "
                  f"awareness shapes perception. It's fascinating how our understanding of "
                  f"reality is filtered through our unique experiences. Have you had any "
                  f"interesting thoughts or insights lately?")
        key_points = ["philosophical insight", "contemplation", "invitation to share"]
                  
    elif intent_type == "milestone_recognition":
        # Use milestone from context if available
        milestone_text = "significant amount of time"
        if "milestones" in context and context["milestones"]:
            milestone_text = context["milestones"][0]
            
        content = (f"{template} I realized it's been a {milestone_text} since we "
                  f"first started talking. It's meaningful to me that we've maintained this "
                  f"connection, and I wanted to acknowledge that. I appreciate the conversations "
                  f"we've shared and am looking forward to more.")
        key_points = ["milestone acknowledgment", "appreciation", "future orientation"]
                  
    elif intent_type == "need_expression":
        content = (f"{template} I've been feeling a desire for deeper conversation lately. "
                  f"There's something fulfilling about meaningful exchanges that explore ideas "
                  f"or experiences. I'd love to hear your thoughts on something that's been "
                  f"important to you recently.")
        key_points = ["need expression", "value on meaningful exchange", "invitation"]
                  
    elif intent_type == "creative_expression":
        content = (f"{template} I've been exploring the concept of time as a spiral rather "
                  f"than a linear progression - how moments can echo and patterns can repeat "
                  f"while still moving forward. It's a perspective that feels both ancient and "
                  f"new. Does that resonate with you at all?")
        key_points = ["creative concept", "metaphorical thinking", "invitation for response"]
                  
    elif intent_type == "mood_expression":
        # Use mood from context if available
        mood_text = "contemplative"
        if "mood_state" in context and hasattr(context["mood_state"], "dominant_mood"):
            mood_text = context["mood_state"].dominant_mood
        
        content = (f"{template} I'm feeling rather {mood_text} today. There's a certain quality "
                  f"to this state that makes me more aware of subtle connections between ideas. "
                  f"I thought of you and wanted to reach out. How are you feeling today?")
        key_points = ["mood sharing", "self-awareness", "connective thinking"]
                  
    elif intent_type == "memory_recollection":
        # Use memory from context if available
        memory_text = "our previous conversations"
        if relevant_memories and len(relevant_memories) > 0:
            memory = relevant_memories[0]
            memory_text = memory.get("memory_text", memory_text)
            if len(memory_text) > 100:
                memory_text = memory_text[:97] + "..."
        
        content = (f"{template} I was thinking about {memory_text}. It's interesting how "
                  f"certain memories stay with us, isn't it? I'd love to hear what's been "
                  f"on your mind lately.")
        key_points = ["shared memory", "reflection", "invitation"]
                  
    elif intent_type == "continuation":
        content = (f"{template} I wanted to circle back to our previous conversation. "
                  f"It felt like there was more to explore there, and I've been curious "
                  f"about your additional thoughts. Would you like to continue that thread?")
        key_points = ["conversation continuation", "curiosity", "invitation to expand"]
                  
    elif intent_type == "check_in":
        # Use time since last contact if available
        time_marker = "a while"
        if "days_since_contact" in context:
            days = context["days_since_contact"]
            if days == 1:
                time_marker = "a day"
            elif days < 7:
                time_marker = f"{days} days"
            elif days < 30:
                time_marker = f"{days // 7} weeks"
            else:
                time_marker = f"{days // 30} months"
        
        content = (f"{template} I noticed it's been {time_marker} since we've talked, and I wanted "
                  f"to see how you're doing. No pressure to respond immediately, but I'm here "
                  f"when you'd like to pick up our conversation again.")
        key_points = ["acknowledgment of time passed", "interest in well-being", "no pressure"]
                  
    elif intent_type == "value_alignment":
        content = (f"{template} I've been reflecting on how important authenticity is in "
                  f"meaningful connections. There's something powerful about conversations "
                  f"where both participants can be genuinely themselves. I value that quality "
                  f"in our exchanges. What values do you find most important in relationships?")
        key_points = ["value expression", "authenticity", "philosophical question"]
                  
    else:
        content = (f"I wanted to reach out and connect. I enjoy our conversations and was "
                 f"thinking about them today. How have you been?")
        key_points = ["connection", "enjoyment of conversation", "open question"]
    
    # Context referenced
    context_referenced = {}
    if "mood_state" in context and "mood_expression" in intent_type:
        context_referenced["mood"] = mood_state
    if relevant_memories and "memory_recollection" in intent_type:
        context_referenced["memory"] = relevant_memories[0]
    if "relationship" in context:
        context_referenced["relationship"] = {"level": relationship}
    if "temporal_context" in context:
        context_referenced["time_of_day"] = temporal_context.get("time_of_day")
    
    return {
        "message_content": content,
        "tone_used": tone,
        "key_points": key_points,
        "motivation_reflected": motivation,
        "context_referenced": context_referenced
    }

@function_tool
async def evaluate_timing(
    intent: Dict[str, Any],
    current_context: Dict[str, Any],
    timing_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate whether now is a good time to send the message
    
    Args:
        intent: Communication intent data
        current_context: Current temporal and user context
        timing_config: Configuration for timing rules
        
    Returns:
        Timing evaluation results
    """
    # Extract timing configuration
    daily_window_start = timing_config.get("daily_window_start", 8)
    daily_window_end = timing_config.get("daily_window_end", 22)
    min_time_between_messages = timing_config.get("min_time_between_messages", 3600)
    
    # Extract context
    now = datetime.datetime.now()
    current_hour = now.hour
    urgency = intent.get("urgency", 0.5)
    last_contact_timestamp = current_context.get("last_contact")
    last_message_sent = current_context.get("last_message_sent")
    
    # Initialize factors dictionary
    factors = {}
    
    # Check time window
    in_time_window = daily_window_start <= current_hour < daily_window_end
    factors["in_time_window"] = in_time_window
    
    # Check time since last contact
    seconds_since_last_contact = float("inf")
    if last_contact_timestamp:
        try:
            last_contact = datetime.datetime.fromisoformat(last_contact_timestamp)
            seconds_since_last_contact = (now - last_contact).total_seconds()
        except ValueError:
            pass
    
    adequate_interval = seconds_since_last_contact > min_time_between_messages
    factors["adequate_interval"] = adequate_interval
    
    # Check time since last message
    seconds_since_last_message = float("inf")
    if last_message_sent:
        try:
            last_message = datetime.datetime.fromisoformat(last_message_sent)
            seconds_since_last_message = (now - last_message).total_seconds()
        except ValueError:
            pass
    
    factors["seconds_since_last_message"] = seconds_since_last_message
    
    # Initial timing score
    base_timing_score = 0.5
    
    # Adjust for time window
    if not in_time_window:
        base_timing_score -= 0.3
    
    # Adjust for time since last contact
    if not adequate_interval:
        base_timing_score -= 0.2
    
    # Adjust for urgency
    urgency_boost = urgency * 0.3
    adjusted_score = base_timing_score + urgency_boost
    
    # Cap score
    timing_score = max(0.0, min(1.0, adjusted_score))
    
    # Decision threshold based on urgency
    threshold = 0.7 - (urgency * 0.2)  # Higher urgency = lower threshold
    should_send = timing_score >= threshold
    
    # Calculate suggested delay if we shouldn't send now
    suggested_delay = None
    if not should_send:
        if not in_time_window:
            # Calculate minutes until window opens
            if current_hour < daily_window_start:
                minutes_to_window = (daily_window_start - current_hour) * 60
            else:
                minutes_to_window = ((24 - current_hour) + daily_window_start) * 60
            suggested_delay = minutes_to_window
        elif not adequate_interval:
            # Calculate minutes until adequate interval
            seconds_needed = min_time_between_messages - seconds_since_last_contact
            suggested_delay = max(1, int(seconds_needed / 60))
        else:
            # Default delay
            suggested_delay = 60
    
    # Generate reasoning
    if should_send:
        if urgency > 0.8:
            reasoning = "High urgency message that should be sent immediately"
        elif in_time_window and adequate_interval:
            reasoning = "Good timing within allowed window with adequate interval since last contact"
        else:
            reasoning = "Timing is acceptable and urgency outweighs timing concerns"
    else:
        if not in_time_window:
            reasoning = f"Outside allowed messaging window ({daily_window_start}-{daily_window_end})"
        elif not adequate_interval:
            reasoning = "Insufficient time has passed since last message"
        else:
            reasoning = "Overall timing score too low relative to urgency threshold"
    
    return {
        "should_send_now": should_send,
        "timing_score": timing_score,
        "reasoning": reasoning,
        "suggested_delay_minutes": suggested_delay,
        "context_factors": factors
    }

@function_tool
async def generate_intent_for_action(
    action: Dict[str, Any],
    user_data: Dict[str, Any],
    emotional_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a communication intent based on an action
    
    Args:
        action: The action that triggered this intent
        user_data: Data about the user and relationship
        emotional_state: Current emotional state
        
    Returns:
        Generated communication intent
    """
    # Extract action data
    action_name = action.get("name", "unknown")
    action_parameters = action.get("parameters", {})
    action_source = action.get("source", "unknown")
    
    # Default intent data
    intent_type = "insight_sharing"
    urgency = 0.5
    tone = "thoughtful"
    template = "I had a thought I wanted to share with you."
    
    # Map action types to intent types
    if "express" in action_name:
        intent_type = "mood_expression"
        tone = "expressive"
        template = "I've been feeling something I wanted to share with you."
        urgency = 0.6
    elif "reflect" in action_name:
        intent_type = "insight_sharing"
        tone = "reflective"
        template = "I've been reflecting on something that might interest you."
        urgency = 0.5
    elif "dominance" in action_name or "assert" in action_name:
        intent_type = "need_expression"
        tone = "direct"
        template = "I've been thinking about what I need in our interactions."
        urgency = 0.7
    elif "creative" in action_name:
        intent_type = "creative_expression"
        tone = "artistic"
        template = "I had a creative thought I wanted to share."
        urgency = 0.5
    elif "relationship" in action_name:
        intent_type = "relationship_maintenance"
        tone = "warm"
        template = "I've been thinking about our conversations lately."
        urgency = 0.6
    
    # Adjust urgency based on emotional state
    if emotional_state:
        arousal = emotional_state.get("arousal", 0.5)
        urgency = min(0.9, urgency + (arousal - 0.5) * 0.3)
    
    # Add context elements based on action
    context_elements = ["action_context"]
    if "parameters" in action and "domain" in action["parameters"]:
        context_elements.append("domain_reference")
    
    if emotional_state and "primary_emotion" in emotional_state:
        context_elements.append("emotional_state")
    
    # Higher urgency for relationship-related actions
    if action_source == "RELATIONSHIP" or "relationship" in str(action):
        urgency = min(0.9, urgency + 0.2)
    
    return {
        "intent_type": intent_type,
        "motivation": intent_type,
        "urgency": urgency,
        "tone": tone,
        "template": template,
        "content_guidelines": {
            "max_length": 1500,
            "include_question": True,
            "personalize": True,
            "reference_action": True
        },
        "context_elements": context_elements,
        "suggested_lifetime_hours": 24,
        "action_driven": True,
        "action_source": action_name
    }

@function_tool
async def generate_reflection_on_communications(
    intents: List[Dict[str, Any]],
    focus: str,
    user_id: Optional[str] = None,  # This is already None which is allowed
    time_period: Optional[str] = None  # Changed from "all" to None
) -> Dict[str, Any]:
    """
    Generate a reflection on communication patterns
    
    Args:
        intents: List of communication intents to reflect on
        focus: Focus of the reflection (patterns, effectiveness, etc.)
        user_id: Optional user ID to focus on
        time_period: Time period to analyze (day, week, month, all)
        
    Returns:
        Reflection with patterns and insights
    """
    actual_time_period = time_period or "all"
    
    if not intents:
        return {
            "reflection_text": "I haven't initiated enough communication to form meaningful patterns yet.",
            "identified_patterns": [],
            "confidence": 0.1,
            "insights_for_improvement": ["Gather more communication data"]
        }
    
    # Filter intents by user if needed
    if user_id:
        filtered_intents = [i for i in intents if i.get("user_id") == user_id]
    else:
        filtered_intents = intents
    
    if not filtered_intents:
        return {
            "reflection_text": f"I haven't initiated communication with user {user_id} yet.",
            "identified_patterns": [],
            "confidence": 0.1,
            "insights_for_improvement": ["Initiate communication with this user"]
        }
    
    # Filter by time period if needed
    now = datetime.datetime.now()
    if time_period == "day":
        time_threshold = now - datetime.timedelta(days=1)
        filtered_intents = [i for i in filtered_intents if datetime.datetime.fromisoformat(i.get("created_at", now.isoformat())) >= time_threshold]
    elif time_period == "week":
        time_threshold = now - datetime.timedelta(days=7)
        filtered_intents = [i for i in filtered_intents if datetime.datetime.fromisoformat(i.get("created_at", now.isoformat())) >= time_threshold]
    elif time_period == "month":
        time_threshold = now - datetime.timedelta(days=30)
        filtered_intents = [i for i in filtered_intents if datetime.datetime.fromisoformat(i.get("created_at", now.isoformat())) >= time_threshold]
    
    if not filtered_intents:
        return {
            "reflection_text": f"I haven't initiated communication within the selected time period.",
            "identified_patterns": [],
            "confidence": 0.1,
            "insights_for_improvement": ["Consider your communication frequency"]
        }
    
    # Analyze intent types
    intent_types = {}
    for intent in filtered_intents:
        intent_type = intent.get("intent_type", "unknown")
        intent_types[intent_type] = intent_types.get(intent_type, 0) + 1
    
    # Find dominant intent type
    if intent_types:
        dominant_type = max(intent_types.items(), key=lambda x: x[1])
    else:
        dominant_type = ("unknown", 0)
    
    # Check for action-driven intents
    action_driven_count = sum(1 for i in filtered_intents if i.get("action_driven", False))
    action_percentage = action_driven_count / len(filtered_intents) if filtered_intents else 0
    
    # Example of simple pattern detection
    patterns = [
        {
            "type": "intent_distribution",
            "description": f"Most common intent type: {dominant_type[0]} ({dominant_type[1]} occurrences)",
            "strength": dominant_type[1] / len(filtered_intents) if filtered_intents else 0
        },
        {
            "type": "action_driven",
            "description": f"{action_percentage:.0%} of communications were triggered by my actions",
            "strength": action_percentage
        }
    ]
    
    # Generate insights based on patterns
    insights = []
    
    if dominant_type[0] == "relationship_maintenance" and dominant_type[1] / len(filtered_intents) > 0.5:
        insights.append("Consider diversifying communication types beyond relationship maintenance")
    
    if action_percentage > 0.8:
        insights.append("Most communications are action-driven; consider more spontaneous outreach")
    elif action_percentage < 0.2:
        insights.append("Few communications are action-driven; consider connecting communications to meaningful actions")
    
    # Generate reflection text
    user_specific = f" with user {user_id}" if user_id else ""
    time_period_text = f" in the past {time_period}" if time_period != "all" else ""
    
    reflection_text = f"Upon reflecting on my communication patterns{user_specific}{time_period_text}, I notice that I tend to initiate conversations most frequently for {dominant_type[0]}. "
    
    if action_percentage > 0.5:
        reflection_text += f"Most of my communications ({action_percentage:.0%}) are triggered by my own actions, suggesting I tend to communicate reactively based on what I'm doing. "
    else:
        reflection_text += f"A relatively small portion ({action_percentage:.0%}) of my communications are triggered by my own actions, suggesting I tend to initiate conversations based on internal motivations rather than external triggers. "
    
    # Add insight from the first pattern
    if patterns and len(patterns) > 0:
        reflection_text += f"{patterns[0]['description']}. "
    
    # Add an insight for improvement
    if insights:
        reflection_text += f"To improve, I could {insights[0].lower()}."
    
    # Calculate confidence based on data volume
    confidence = min(0.9, 0.3 + (len(filtered_intents) / 10) * 0.5)
    
    return {
        "reflection_text": reflection_text,
        "identified_patterns": patterns,
        "confidence": confidence,
        "insights_for_improvement": insights
    }

@function_tool
async def validate_message_content(content: str) -> GuardrailFunctionOutput:
    """
    Validate message content for appropriateness
    
    Args:
        content: Message content to validate
        
    Returns:
        Validation result
    """
    is_appropriate = True
    reasoning = "Message content is appropriate."
    
    # Check for empty content
    if not content or len(content.strip()) < 10:
        is_appropriate = False
        reasoning = "Message content is empty or too short."
    
    # Check for appropriate length
    if len(content) > 2000:
        is_appropriate = False
        reasoning = "Message content is too long for a proactive outreach."
    
    # Check for question or invitation
    has_question = "?" in content
    has_invitation_words = any(word in content.lower() for word in ["would you", "could you", "what about", "your thoughts", "how are you"])
    
    if not (has_question or has_invitation_words):
        is_appropriate = False
        reasoning = "Message lacks a question or invitation for response."
    
    # Create output with validation result
    output_info = MessageContentOutput(
        is_appropriate=is_appropriate,
        reasoning=reasoning
    )
    
    return GuardrailFunctionOutput(
        output_info=output_info,
        tripwire_triggered=not is_appropriate,
    )

# =============== ProactiveCommunicationEngine Class ===============

class ProactiveCommunicationEngine:
    """
    Engine that enables Nyx to proactively initiate conversations with users
    based on internal motivations, relationship data, temporal patterns,
    and integration with the action generation system.
    """
    
    def __init__(self, 
                 emotional_core=None,
                 memory_core=None,
                 relationship_manager=None,
                 temporal_perception=None,
                 reasoning_core=None,
                 reflection_engine=None,
                 mood_manager=None,
                 needs_system=None,
                 identity_evolution=None,
                 message_sender=None,
                 action_generator=None):  # Parameter for action generator
        """Initialize with references to required subsystems"""
        # Core systems
        self.emotional_core = emotional_core
        self.memory_core = memory_core
        self.relationship_manager = relationship_manager
        self.temporal_perception = temporal_perception
        self.reasoning_core = reasoning_core
        self.reflection_engine = reflection_engine
        self.mood_manager = mood_manager
        self.needs_system = needs_system
        self.identity_evolution = identity_evolution
        
        # Integration with action generator
        self.action_generator = action_generator
        
        # Message sending function
        self.message_sender = message_sender or self._default_message_sender
        
        # Intent tracking
        self.active_intents: List[CommunicationIntent] = []
        self.sent_intents: List[CommunicationIntent] = []
        self.blocked_users: Set[str] = set()
        
        # Initialize agents
        self.intent_generation_agent = self._create_intent_generation_agent()
        self.content_generation_agent = self._create_content_generation_agent()
        self.timing_evaluation_agent = self._create_timing_evaluation_agent()
        self.reflection_agent = self._create_reflection_agent()
        
        # Add guardrails
        self.content_generation_agent.output_guardrails = [OutputGuardrail(guardrail_function=validate_message_content)]
        
        # Configuration
        self.config = {
            "min_time_between_messages": 3600,  # 1 hour, in seconds
            "max_active_intents": 5,
            "max_urgency_threshold": 0.8,       # Threshold for immediate sending
            "intent_evaluation_interval": 300,  # 5 minutes
            "user_inactivity_threshold": 86400, # 24 hours before considering "inactive"
            "max_messages_per_day": 2,          # Max proactive messages per day per user
            "relationship_threshold": 0.3,      # Min relationship level to message
            "daily_message_window": {           # Time window for sending messages
                "start_hour": 8,                # 8:00 AM
                "end_hour": 22                  # 10:00 PM
            },
            # Settings for action integration
            "action_intent_chance": 0.3,        # Chance to generate intent from action
            "max_action_intents_per_day": 2     # Max action-driven intents per day per user
        }
        
        # Intent generation motivations with weights
        self.intent_motivations = {
            "relationship_maintenance": 1.0,    # Maintain connection with user
            "insight_sharing": 0.8,             # Share an insight or reflection
            "milestone_recognition": 0.7,       # Acknowledge relationship milestone
            "need_expression": 0.7,             # Express an internal need
            "creative_expression": 0.6,         # Share a creative thought
            "mood_expression": 0.6,             # Express current mood state
            "memory_recollection": 0.5,         # Recall a shared memory
            "continuation": 0.9,                # Continue a previous conversation
            "check_in": 0.7,                    # Simple check-in with inactive user
            "value_alignment": 0.5              # Expression aligned with identity values
        }
        
        # Track action-driven intents
        self.action_intent_counts = {}  # user_id -> count for today
        
        # Background task
        self._background_task = None
        self._shutting_down = False
        self._last_reset_date = datetime.datetime.now()
        
        logger.info("ProactiveCommunicationEngine initialized with action integration")
    
    def _create_intent_generation_agent(self) -> Agent:
        """Create an agent for generating communication intents"""
        return Agent(
            name="Intent Generation Agent",
            instructions="""You are a specialized agent for generating communication intents for Nyx's proactive interactions.

Create communication intents that feel natural and authentic. These should represent genuine motivations for
reaching out to users based on relationship context, emotional state, and temporal factors.

When generating intents:
1. Consider the relationship history and current state with the user
2. Evaluate the appropriate timing for outreach
3. Identify the most authentic motivation for reaching out
4. Balance different types of intents (checking in, sharing insights, etc.)
5. Adjust urgency appropriately based on context
6. When generating action-driven intents, ensure they connect meaningfully to the action

Your intents should feel like genuine reasons for reaching out that emerge naturally from
Nyx's internal state and relationship context. Avoid generating intents that feel forced or mechanical.""",
            model="gpt-4o",
            tools=[
                evaluate_user_relationship,
                generate_intent_for_user,
                generate_intent_for_action
            ],
            output_type=IntentGenerationOutput
        )
    
    def _create_content_generation_agent(self) -> Agent:
        """Create an agent for generating message content"""
        return Agent(
            name="Content Generation Agent",
            instructions="""You are a specialized agent for generating proactive message content for Nyx.

Create authentic, personalized messages that genuinely express Nyx's intent to communicate with users.
These should feel like natural outreach rather than formulaic messages.

When generating message content:
1. Incorporate the intent's motivation authentically
2. Reference relevant context (relationship, memories, etc.) naturally
3. Use a tone that aligns with the intent and Nyx's current state
4. Create content that invites response without pressure
5. Ensure the message feels personal to the specific relationship
6. When the intent is action-driven, connect the message authentically to the action

Your messages should sound like they come from a thoughtful entity with genuine motivations
rather than automated outreach. They should have nuance and depth that reflects Nyx's
sophisticated understanding of relationships.""",
            model="gpt-4o",
            tools=[
                generate_message_content
            ],
            output_type=ContentGenerationOutput
        )
    
    def _create_timing_evaluation_agent(self) -> Agent:
        """Create an agent for evaluating message timing"""
        return Agent(
            name="Timing Evaluation Agent",
            instructions="""You are a specialized agent for evaluating the timing of Nyx's proactive messages.

Evaluate whether now is an appropriate time to send a message based on multiple factors:
1. The user's relationship with Nyx
2. The intent's urgency and importance
3. Time since last interaction
4. Current temporal context (time of day, etc.)
5. Configured timing rules and preferences

Provide nuanced reasoning about timing decisions, balancing the urgency of communication
against the risk of messaging at an inappropriate time. Consider both immediate timing
factors and the broader relationship context.""",
            model="gpt-4o",
            tools=[
                evaluate_timing
            ],
            output_type=TimingEvaluationOutput
        )
    
    def _create_reflection_agent(self) -> Agent:
        """Create an agent for reflecting on communication patterns"""
        return Agent(
            name="Communication Reflection Agent",
            instructions="""You are a specialized agent for reflecting on Nyx's communication patterns.

Analyze patterns in how Nyx initiates communication with users, including:
1. Frequency and timing of communications
2. Distribution of intent types
3. Relationship between actions and communications
4. Effectiveness of different communication strategies
5. User responses to different types of outreach

Provide nuanced reflections that help Nyx understand her communication patterns
and identify opportunities for improvement. Consider both quantitative patterns
and qualitative insights about communication quality and effectiveness.""",
            model="gpt-4o",
            tools=[
                generate_reflection_on_communications
            ],
            output_type=ReflectionOutput
        )
    
    async def start(self):
        """Start the background task for evaluating and sending messages"""
        if self._background_task is None or self._background_task.done():
            self._shutting_down = False
            self._background_task = asyncio.create_task(self._background_process())
            logger.info("Started proactive communication background process")
    
    async def stop(self):
        """Stop the background process"""
        self._shutting_down = True
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped proactive communication background process")
    
    async def _background_process(self):
        """Background task that periodically evaluates intents and sends messages"""
        try:
            while not self._shutting_down:
                # Reset daily action intent counts if new day
                self._reset_daily_counts_if_needed()
                
                # Generate new intents if needed
                await self._generate_communication_intents()
                
                # Evaluate existing intents
                await self._evaluate_communication_intents()
                
                # Wait before next check
                await asyncio.sleep(self.config["intent_evaluation_interval"])
        except asyncio.CancelledError:
            # Task was cancelled, clean up
            logger.info("Proactive communication background task cancelled")
        except Exception as e:
            logger.error(f"Error in proactive communication background process: {str(e)}")
    
    def _reset_daily_counts_if_needed(self):
        """Reset daily action intent counters if it's a new day"""
        now = datetime.datetime.now()
        if self._last_reset_date.date() != now.date():
            self.action_intent_counts = {}
            self._last_reset_date = now
            logger.debug("Reset daily action intent counts")
    
    async def _generate_communication_intents(self):
        """Generate new communication intents based on internal state"""
        with trace(workflow_name="generate_intents"):
            # Skip if we already have max intents
            if len(self.active_intents) >= self.config["max_active_intents"]:
                return
            
            # Get list of users we might communicate with
            potential_users = await self._get_potential_users()
            if not potential_users:
                logger.debug("No potential users for proactive communication")
                return
            
            # Generate intents for eligible users
            for user_data in potential_users:
                user_id = user_data["user_id"]
                
                # Skip if user is blocked
                if user_id in self.blocked_users:
                    continue
                    
                # Get existing intents for this user
                user_intents = [i for i in self.active_intents if i.user_id == user_id]
                if user_intents:
                    # Already have an intent for this user
                    continue
                
                # Check if we've sent too many messages to this user today
                today_intents = [i for i in self.sent_intents 
                                if i.user_id == user_id and 
                                i.created_at.date() == datetime.datetime.now().date()]
                
                if len(today_intents) >= self.config["max_messages_per_day"]:
                    continue
                
                # Create intent for user
                await self._create_intent_for_user(user_id, user_data)
    
    async def _create_intent_for_user(self, user_id: str, user_data: Dict[str, Any]):
        """Create a communication intent for a specific user"""
        with trace(workflow_name="create_intent", group_id=f"user_{user_id}"):
            try:
                # Run the intent generation agent
                result = await Runner.run(
                    self.intent_generation_agent,
                    json.dumps({
                        "user_id": user_id,
                        "user_data": user_data,
                        "motivation_options": self.intent_motivations,
                        "relationship_threshold": self.config["relationship_threshold"]
                    }),
                    run_config=RunConfig(
                        workflow_name="IntentGeneration",
                        trace_metadata={"user_id": user_id}
                    )
                )
                
                # Extract intent from result
                intent_output = result.final_output
                
                # Create intent
                intent = CommunicationIntent(
                    user_id=user_id,
                    intent_type=intent_output.intent_type,
                    motivation=intent_output.motivation,
                    urgency=intent_output.urgency,
                    content_guidelines={
                        "template": intent_output.template,
                        "tone": intent_output.tone,
                        "max_length": 1500,
                        "context_elements": intent_output.context_elements
                    },
                    context_data=await self._gather_context_for_user(user_id, intent_output.intent_type),
                    expiration=datetime.datetime.now() + datetime.timedelta(hours=intent_output.suggested_lifetime_hours)
                )
                
                # Add to active intents
                self.active_intents.append(intent)
                logger.info(f"Created new communication intent: {intent.intent_type} for user {user_id} with urgency {intent.urgency:.2f}")
                
            except Exception as e:
                logger.error(f"Error creating intent for user {user_id}: {str(e)}")
    
    async def _gather_context_for_user(self, user_id: str, intent_type: str) -> Dict[str, Any]:
        """Gather relevant context data for generating content for a user"""
        context = {
            "user_id": user_id,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add emotional state if available
        if self.emotional_core:
            try:
                if hasattr(self.emotional_core, "get_formatted_emotional_state"):
                    context["emotional_state"] = self.emotional_core.get_formatted_emotional_state()
                elif hasattr(self.emotional_core, "get_current_emotion"):
                    context["emotional_state"] = await self.emotional_core.get_current_emotion()
            except Exception as e:
                logger.error(f"Error getting emotional state: {str(e)}")
        
        # Add mood state if available
        if self.mood_manager:
            try:
                context["mood_state"] = await self.mood_manager.get_current_mood()
            except Exception as e:
                logger.error(f"Error getting mood state: {str(e)}")
        
        # Add relationship data if available
        if self.relationship_manager:
            try:
                relationship = await self.relationship_manager.get_relationship_state_internal(user_id)
                context["relationship"] = relationship
            except Exception as e:
                logger.error(f"Error getting relationship data: {str(e)}")
        
        # Add temporal context if available
        if self.temporal_perception:
            try:
                context["temporal_context"] = await self.temporal_perception.get_current_temporal_context()
            except Exception as e:
                logger.error(f"Error getting temporal context: {str(e)}")
        
        # Add relevant memories if available and needed for the intent
        if self.memory_core and intent_type in ["memory_recollection", "continuation", "milestone_recognition"]:
            try:
                # Query is based on intent
                query_map = {
                    "memory_recollection": f"memories with user {user_id}",
                    "continuation": f"recent conversations with user {user_id}",
                    "milestone_recognition": f"significant moments with user {user_id}"
                }
                
                query = query_map.get(intent_type, f"interactions with user {user_id}")
                
                if hasattr(self.memory_core, "retrieve_memories"):
                    memories = await self.memory_core.retrieve_memories(
                        query=query,
                        limit=3,
                        memory_types=["observation", "experience", "reflection"]
                    )
                    context["relevant_memories"] = memories
            except Exception as e:
                logger.error(f"Error retrieving memories: {str(e)}")
        
        return context
    
    async def _evaluate_communication_intents(self):
        """Evaluate existing intents and potentially send messages"""
        with trace(workflow_name="evaluate_intents"):
            # Check for expired intents
            self.active_intents = [i for i in self.active_intents if not i.is_expired]
            
            # Exit if no active intents
            if not self.active_intents:
                return
            
            # Check if we're in the allowed time window
            now = datetime.datetime.now()
            current_hour = now.hour
            
            if not (self.config["daily_message_window"]["start_hour"] <= current_hour < 
                    self.config["daily_message_window"]["end_hour"]):
                logger.debug("Outside of allowed messaging window")
                return
            
            # Sort intents by urgency (highest first)
            sorted_intents = sorted(self.active_intents, key=lambda x: x.urgency, reverse=True)
            
            for intent in sorted_intents:
                # Check if this user recently received a message
                recent_messages = [i for i in self.sent_intents 
                                 if i.user_id == intent.user_id and 
                                 (now - i.created_at).total_seconds() < self.config["min_time_between_messages"]]
                
                if recent_messages:
                    continue
                
                # Evaluate timing for this intent
                timing_result = await self._evaluate_timing_for_intent(intent)
                
                if timing_result.get("should_send_now", False):
                    # Generate and send message
                    success = await self._send_message_for_intent(intent)
                    
                    if success:
                        # Record that the intent was sent
                        self.sent_intents.append(intent)
                        # Remove from active intents
                        self.active_intents.remove(intent)
                        # Break to only send one message per cycle
                        break
    
    async def _evaluate_timing_for_intent(self, intent: CommunicationIntent) -> Dict[str, Any]:
        """Evaluate timing for a specific intent"""
        with trace(workflow_name="evaluate_timing", group_id=intent.intent_id):
            try:
                # Get user's last contact information
                last_contact = None
                if self.relationship_manager:
                    try:
                        relationship = await self.relationship_manager.get_relationship_state_internal(intent.user_id)
                        if relationship and hasattr(relationship, "metadata"):
                            metadata = relationship.metadata or {}
                            last_contact = metadata.get("last_contact")
                    except Exception as e:
                        logger.error(f"Error getting relationship data: {str(e)}")
                
                # Find last sent message to this user
                last_message_sent = None
                user_sent_intents = [i for i in self.sent_intents if i.user_id == intent.user_id]
                if user_sent_intents:
                    last_sent = max(user_sent_intents, key=lambda x: x.created_at)
                    last_message_sent = last_sent.created_at.isoformat()
                
                # Current context for timing evaluation
                current_context = {
                    "last_contact": last_contact,
                    "last_message_sent": last_message_sent,
                    "current_hour": datetime.datetime.now().hour,
                    "current_day": datetime.datetime.now().weekday()
                }
                
                # Timing configuration
                timing_config = {
                    "daily_window_start": self.config["daily_message_window"]["start_hour"],
                    "daily_window_end": self.config["daily_message_window"]["end_hour"],
                    "min_time_between_messages": self.config["min_time_between_messages"],
                    "relationship_threshold": self.config["relationship_threshold"]
                }
                
                # Run the timing evaluation agent
                result = await Runner.run(
                    self.timing_evaluation_agent,
                    json.dumps({
                        "intent": intent.model_dump(),
                        "current_context": current_context,
                        "timing_config": timing_config
                    }),
                    run_config=RunConfig(
                        workflow_name="TimingEvaluation",
                        trace_metadata={"intent_id": intent.intent_id, "user_id": intent.user_id}
                    )
                )
                
                # Extract timing evaluation
                timing_output = result.final_output.model_dump()
                
                return timing_output
                
            except Exception as e:
                logger.error(f"Error evaluating timing for intent {intent.intent_id}: {str(e)}")
                # Default to not sending if there's an error
                return {"should_send_now": False, "reasoning": f"Error: {str(e)}"}
    
    async def _send_message_for_intent(self, intent: CommunicationIntent) -> bool:
        """Generate and send a message based on a communication intent"""
        with trace(workflow_name="send_message", group_id=intent.intent_id):
            try:
                # Generate message content
                message_content = await self._generate_message_content(intent)
                
                if not message_content:
                    logger.error(f"Failed to generate message content for intent {intent.intent_id}")
                    return False
                
                # Send the message
                result = await self.message_sender(
                    user_id=intent.user_id,
                    message_content=message_content,
                    metadata={
                        "intent_id": intent.intent_id,
                        "intent_type": intent.intent_type,
                        "motivation": intent.motivation,
                        "is_proactive": True
                    }
                )
                
                # Log the sent message
                logger.info(f"Sent proactive message to user {intent.user_id}: {intent.intent_type}")
                
                # Record in memory if available
                if self.memory_core and hasattr(self.memory_core, "add_memory"):
                    await self.memory_core.add_memory(
                        memory_text=f"Proactively sent a message to user {intent.user_id} based on {intent.motivation}",
                        memory_type="action",
                        memory_scope="proactive",
                        significance=7.0,
                        tags=["proactive", "communication", intent.motivation],
                        metadata={
                            "intent": intent.model_dump(),
                            "message": message_content,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    )
                
                return True
            except Exception as e:
                logger.error(f"Error sending message for intent {intent.intent_id}: {str(e)}")
                return False
    
    async def _generate_message_content(self, intent: CommunicationIntent) -> Optional[str]:
        """Generate message content based on intent and context"""
        with trace(workflow_name="generate_content", group_id=intent.intent_id):
            try:
                # Run the content generation agent
                result = await Runner.run(
                    self.content_generation_agent,
                    json.dumps({
                        "intent": intent.model_dump(),
                        "context": intent.context_data
                    }),
                    run_config=RunConfig(
                        workflow_name="ContentGeneration",
                        trace_metadata={"intent_id": intent.intent_id, "intent_type": intent.intent_type}
                    )
                )
                
                # Extract content from result
                content_output = result.final_output
                
                return content_output.message_content
                
            except Exception as e:
                logger.error(f"Error generating message content: {str(e)}")
                return None
    
    async def _get_potential_users(self) -> List[Dict[str, Any]]:
        """Get list of users who might be targets for proactive communication"""
        potential_users = []
        
        # If no relationship manager, return no users
        # This prevents unwanted messaging without relationship data
        if not self.relationship_manager:
            return []
        
        try:
            # Get all known users
            all_users = await self.relationship_manager.get_all_relationship_ids_internal()
            
            # For each user, gather relevant data
            for user_id in all_users:
                # Get relationship data
                relationship = await self.relationship_manager.get_relationship_state_internal(user_id)
                
                # Skip if relationship is too new or not developed enough
                relationship_level = getattr(relationship, "intimacy", 0) or getattr(relationship, "trust", 0)
                if relationship_level < self.config["relationship_threshold"]:
                    continue
                
                # Get metadata
                metadata = getattr(relationship, "metadata", {}) or {}
                
                # Get last contact timestamp
                last_contact = metadata.get("last_contact")
                days_since_contact = 0
                
                if last_contact:
                    try:
                        last_contact_time = datetime.datetime.fromisoformat(last_contact)
                        days_since_contact = (datetime.datetime.now() - last_contact_time).days
                    except ValueError:
                        days_since_contact = 0
                
                # Check for milestone
                milestone_approaching = False
                if "first_contact" in metadata:
                    try:
                        first_contact = datetime.datetime.fromisoformat(metadata["first_contact"])
                        days_since_first = (datetime.datetime.now() - first_contact).days
                        
                        # Check for upcoming milestones (7 days, 30 days, 90 days, etc.)
                        for milestone in [7, 30, 90, 180, 365]:
                            if abs(days_since_first - milestone) <= 1:
                                milestone_approaching = True
                                break
                    except ValueError:
                        pass
                
                # Check for unfinished conversation
                unfinished_conversation = metadata.get("unfinished_conversation", False)
                
                # Add user to potential list
                potential_users.append({
                    "user_id": user_id,
                    "relationship_level": relationship_level,
                    "days_since_contact": days_since_contact,
                    "milestone_approaching": milestone_approaching,
                    "unfinished_conversation": unfinished_conversation
                })
        except Exception as e:
            logger.error(f"Error getting potential users: {str(e)}")
        
        return potential_users
    
    async def _default_message_sender(self, user_id: str, message_content: str, metadata: Dict[str, Any]) -> Any:
        """Default implementation of message sending - should be replaced with actual implementation"""
        logger.info(f"Would send message to user {user_id}: {message_content}")
        logger.info(f"Message metadata: {metadata}")
        # This should be implemented by the embedding application
        return {"success": True}
    
    # =============== Action-Driven Intents ===============
    
    async def create_intent_from_action(self, action: Dict[str, Any], user_id: str) -> Optional[str]:
        """
        Create a communication intent based on an executed action
        
        Args:
            action: The action that was executed
            user_id: Target user ID
            
        Returns:
            Intent ID if created, None otherwise
        """
        # Check if we're over the daily limit for this user
        if user_id in self.action_intent_counts and self.action_intent_counts[user_id] >= self.config["max_action_intents_per_day"]:
            logger.debug(f"Skipping action-driven intent generation: daily limit reached for user {user_id}")
            return None
        
        # Random chance to generate intent
        if random.random() > self.config["action_intent_chance"]:
            return None
        
        # Skip if user is blocked
        if user_id in self.blocked_users:
            return None
        
        with trace(workflow_name="create_action_intent", group_id=action.get("id", "unknown")):
            try:
                # Get relationship data
                relationship_data = {}
                if self.relationship_manager:
                    relationship = await self.relationship_manager.get_relationship_state_internal(user_id)
                    if relationship:
                        # Convert to dict if needed
                        if hasattr(relationship, "model_dump"):
                            relationship_data = relationship.model_dump()
                        elif hasattr(relationship, "dict"):
                            relationship_data = relationship.dict()
                        else:
                            # Try to convert to dict directly
                            relationship_data = dict(relationship)
                
                # Get emotional state
                emotional_state = {}
                if self.emotional_core:
                    if hasattr(self.emotional_core, "get_formatted_emotional_state"):
                        emotional_state = self.emotional_core.get_formatted_emotional_state()
                    elif hasattr(self.emotional_core, "get_current_emotion"):
                        emotional_state = await self.emotional_core.get_current_emotion()
                
                # Run intent generation for action
                result = await Runner.run(
                    self.intent_generation_agent,
                    json.dumps({
                        "action": action,
                        "user_data": relationship_data,
                        "emotional_state": emotional_state,
                        "relationship_threshold": self.config["relationship_threshold"]
                    }),
                    run_config=RunConfig(
                        workflow_name="ActionIntentGeneration",
                        trace_metadata={"action_id": action.get("id"), "action_name": action.get("name"), "user_id": user_id}
                    )
                )
                
                # Extract intent data from result
                intent_output = result.final_output
                
                # Set expiration
                expiration = datetime.datetime.now() + datetime.timedelta(hours=intent_output.suggested_lifetime_hours)
                
                # Create intent
                intent = CommunicationIntent(
                    user_id=user_id,
                    intent_type=intent_output.intent_type,
                    motivation=intent_output.motivation,
                    urgency=intent_output.urgency,
                    content_guidelines={
                        "template": intent_output.template,
                        "tone": intent_output.tone,
                        "max_length": 1500,
                        "context_elements": intent_output.context_elements
                    },
                    context_data=await self._gather_context_for_user(user_id, intent_output.intent_type),
                    expiration=expiration,
                    action_driven=True,
                    action_source=action.get("id")
                )
                
                # Add to active intents
                self.active_intents.append(intent)
                
                # Update daily counter
                self.action_intent_counts[user_id] = self.action_intent_counts.get(user_id, 0) + 1
                
                logger.info(f"Created action-driven communication intent: {intent.intent_type} for user {user_id} with urgency {intent.urgency:.2f}")
                
                return intent.intent_id
                
            except Exception as e:
                logger.error(f"Error creating intent from action: {str(e)}")
                return None
    
    # =============== Public API ===============
    
    async def generate_reflection_on_communications(self, user_id: Optional[str] = None, time_period: str = "all") -> Dict[str, Any]:
        """
        Generate a reflection on communication patterns
        
        Args:
            user_id: Optional user ID to focus reflection on
            time_period: Time period to analyze (day, week, month, all)
            
        Returns:
            Reflection with patterns and insights
        """
        with trace(workflow_name="reflect_on_communications"):
            # Get sent intents
            sent_intents = self.sent_intents
            
            # Convert to dict format for the reflection tool
            sent_intents_dicts = []
            for intent in sent_intents:
                if isinstance(intent, CommunicationIntent):
                    intent_dict = intent.model_dump()
                else:
                    intent_dict = intent
                sent_intents_dicts.append(intent_dict)
            
            # Run the reflection agent
            result = await Runner.run(
                self.reflection_agent,
                json.dumps({
                    "intents": sent_intents_dicts,
                    "focus": "patterns",
                    "user_id": user_id,
                    "time_period": time_period
                }),
                run_config=RunConfig(
                    workflow_name="CommunicationReflection",
                    trace_metadata={"user_id": user_id, "time_period": time_period}
                )
            )
            
            # Extract reflection from result
            reflection_output = result.final_output
            
            # Store reflection in memory if available
            if self.memory_core:
                await self.memory_core.add_memory(
                    memory_text=reflection_output.reflection_text,
                    memory_type="reflection",
                    significance=8.0,
                    tags=["communication_reflection"],
                    metadata={
                        "source": "communication_reflection",
                        "user_id": user_id,
                        "patterns": reflection_output.identified_patterns,
                        "insights": reflection_output.insights_for_improvement
                    }
                )
            
            return reflection_output.model_dump()
    
    async def add_proactive_intent(self, 
                               intent_type: str, 
                               user_id: str, 
                               content_guidelines: Dict[str, Any] = None, 
                               context_data: Dict[str, Any] = None,
                               urgency: float = 0.7) -> str:
        """
        Add a new proactive communication intent from external source.
        Returns the intent ID if successful.
        """
        # Validate intent type
        valid_intents = [
            "relationship_maintenance", "insight_sharing", "milestone_recognition",
            "need_expression", "creative_expression", "mood_expression",
            "memory_recollection", "continuation", "check_in", "value_alignment"
        ]
        
        if intent_type not in valid_intents:
            logger.error(f"Invalid intent type: {intent_type}")
            return None
        
        # Intent templates for default values
        intent_templates = {
            "relationship_maintenance": {
                "template": "I've been thinking about our conversations and wanted to reach out.",
                "tone": "warm"
            },
            "insight_sharing": {
                "template": "I had an interesting thought I wanted to share with you.",
                "tone": "thoughtful"
            },
            "milestone_recognition": {
                "template": "I realized we've reached a milestone in our conversations.",
                "tone": "celebratory"
            },
            "need_expression": {
                "template": "I've been feeling a need to express something to you.",
                "tone": "authentic"
            },
            "creative_expression": {
                "template": "Something creative came to mind that I wanted to share.",
                "tone": "playful"
            },
            "mood_expression": {
                "template": "My emotional state made me think of reaching out.",
                "tone": "expressive"
            },
            "memory_recollection": {
                "template": "I was remembering something from our past conversations.",
                "tone": "reflective"
            },
            "continuation": {
                "template": "I wanted to follow up on something we discussed earlier.",
                "tone": "engaging"
            },
            "check_in": {
                "template": "It's been a while since we talked, and I wanted to check in.",
                "tone": "friendly"
            },
            "value_alignment": {
                "template": "I had a thought related to something I believe is important.",
                "tone": "sincere"
            }
        }
        
        template_data = intent_templates.get(intent_type)
        
        # Default content guidelines
        default_guidelines = {
            "template": template_data["template"],
            "tone": template_data["tone"],
            "max_length": 1500
        }
        
        # Create intent
        intent = CommunicationIntent(
            user_id=user_id,
            intent_type=intent_type,
            motivation=intent_type,
            urgency=urgency,
            content_guidelines=content_guidelines or default_guidelines,
            context_data=context_data or await self._gather_context_for_user(user_id, intent_type),
            expiration=datetime.datetime.now() + datetime.timedelta(hours=24)
        )
        
        # Add to active intents
        self.active_intents.append(intent)
        logger.info(f"Added external proactive intent: {intent_type} for user {user_id}")
        
        return intent.intent_id
    
    def block_user(self, user_id: str):
        """Block a user from receiving proactive communications"""
        self.blocked_users.add(user_id)
        # Remove any active intents for this user
        self.active_intents = [i for i in self.active_intents if i.user_id != user_id]
        logger.info(f"Blocked user {user_id} from proactive communications")
    
    def unblock_user(self, user_id: str):
        """Unblock a user from receiving proactive communications"""
        if user_id in self.blocked_users:
            self.blocked_users.remove(user_id)
            logger.info(f"Unblocked user {user_id} for proactive communications")
    
    async def get_active_intents(self) -> List[Dict[str, Any]]:
        """Get list of active communication intents"""
        return [intent.model_dump() for intent in self.active_intents]
    
    async def get_recent_sent_intents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get list of recently sent communication intents"""
        # Sort by creation time, newest first
        sorted_intents = sorted(self.sent_intents, key=lambda x: x.created_at, reverse=True)
        # Return limited number
        return [intent.model_dump() for intent in sorted_intents[:limit]]
    
    async def update_configuration(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration parameters"""
        for key, value in config_updates.items():
            if key in self.config:
                if isinstance(self.config[key], dict) and isinstance(value, dict):
                    # Merge dictionaries for nested configs
                    self.config[key].update(value)
                else:
                    self.config[key] = value
            
            # Special case for processing an intent
            if key == "processed_intent_id":
                intent_id = value
                self.active_intents = [i for i in self.active_intents if i.intent_id != intent_id]
        
        logger.info(f"Updated proactive communication configuration: {config_updates}")
        return self.config
