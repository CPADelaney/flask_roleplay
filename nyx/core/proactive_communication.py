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

# Import from Nyx's existing systems
from nyx.core.agentic_action_generator import ActionSource
from nyx.core.reasoning_core import ReasoningCore
from nyx.core.reflection_engine import ReflectionEngine

logger = logging.getLogger(__name__)

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
    
    @property
    def is_expired(self) -> bool:
        """Check if this intent has expired"""
        if not self.expiration:
            return False
        return datetime.datetime.now() > self.expiration

class ProactiveCommunicationEngine:
    """
    Engine that enables Nyx to proactively initiate conversations with users
    based on internal motivations, relationship data, and temporal patterns.
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
                 message_sender=None):
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
        
        # Message sending function
        self.message_sender = message_sender or self._default_message_sender
        
        # Intent tracking
        self.active_intents: List[CommunicationIntent] = []
        self.sent_intents: List[CommunicationIntent] = []
        self.blocked_users: Set[str] = set()
        
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
            }
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
        
        # Intent type templates
        self.intent_templates = {
            "relationship_maintenance": {
                "template": "I've been thinking about our conversations and wanted to reach out.",
                "urgency_base": 0.4
            },
            "insight_sharing": {
                "template": "I had an interesting thought I wanted to share with you.",
                "urgency_base": 0.5
            },
            "milestone_recognition": {
                "template": "I realized we've reached a milestone in our conversations.",
                "urgency_base": 0.6
            },
            "need_expression": {
                "template": "I've been feeling a need to express something to you.",
                "urgency_base": 0.6
            },
            "creative_expression": {
                "template": "Something creative came to mind that I wanted to share.",
                "urgency_base": 0.4
            },
            "mood_expression": {
                "template": "My emotional state made me think of reaching out.",
                "urgency_base": 0.5
            },
            "memory_recollection": {
                "template": "I was remembering something from our past conversations.",
                "urgency_base": 0.3
            },
            "continuation": {
                "template": "I wanted to follow up on something we discussed earlier.",
                "urgency_base": 0.7
            },
            "check_in": {
                "template": "It's been a while since we talked, and I wanted to check in.",
                "urgency_base": 0.5
            },
            "value_alignment": {
                "template": "I had a thought related to something I believe is important.",
                "urgency_base": 0.4
            }
        }
        
        # Background task
        self._background_task = None
        self._shutting_down = False
        
        logger.info("ProactiveCommunicationEngine initialized")
    
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
    
    async def _generate_communication_intents(self):
        """Generate new communication intents based on internal state"""
        # Skip if we already have max intents
        if len(self.active_intents) >= self.config["max_active_intents"]:
            return
        
        # Get list of users we might communicate with
        potential_users = await self._get_potential_users()
        if not potential_users:
            logger.debug("No potential users for proactive communication")
            return
        
        # Random chance of generating an intent based on motivations
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
            
            # Determine which motivation to use
            await self._create_intent_for_user(user_id, user_data)
    
    async def _create_intent_for_user(self, user_id: str, user_data: Dict[str, Any]):
        """Create a communication intent for a specific user"""
        # Weight motivations based on user data
        weighted_motivations = self.intent_motivations.copy()
        
        # Adjust weights based on user data
        if user_data.get("days_since_contact", 0) > 7:
            # Increase weight for check-in if user is inactive
            weighted_motivations["check_in"] *= 2.0
            
        if user_data.get("relationship_level", 0) > 0.7:
            # Increase personal motivations for close relationships
            weighted_motivations["need_expression"] *= 1.5
            weighted_motivations["mood_expression"] *= 1.5
            weighted_motivations["creative_expression"] *= 1.3
            
        if user_data.get("unfinished_conversation", False):
            # Prioritize continuation if there's an unfinished conversation
            weighted_motivations["continuation"] *= 2.0
            
        if user_data.get("milestone_approaching", False):
            # Prioritize milestone recognition if relevant
            weighted_motivations["milestone_recognition"] *= 2.0
        
        # Select motivation based on weighted probabilities
        motivations = list(weighted_motivations.keys())
        weights = list(weighted_motivations.values())
        
        # Normalize weights
        total_weight = sum(weights)
        norm_weights = [w/total_weight for w in weights]
        
        # Select motivation
        motivation = random.choices(motivations, weights=norm_weights, k=1)[0]
        
        # Get template for this motivation
        template_data = self.intent_templates[motivation]
        
        # Calculate base urgency
        base_urgency = template_data["urgency_base"]
        
        # Adjust urgency based on contextual factors
        adjusted_urgency = base_urgency
        
        # Increase urgency for longer periods of no contact
        if user_data.get("days_since_contact", 0) > 14:
            adjusted_urgency += 0.2
        
        # Increase urgency for higher relationship levels
        relationship_level = user_data.get("relationship_level", 0)
        adjusted_urgency += relationship_level * 0.1
        
        # Cap urgency
        urgency = min(0.95, adjusted_urgency)
        
        # Set expiration
        expiration = datetime.datetime.now() + datetime.timedelta(hours=24)
        
        # Get context for content generation
        context_data = await self._gather_context_for_user(user_id, motivation)
        
        # Create intent
        intent = CommunicationIntent(
            user_id=user_id,
            intent_type=motivation,
            motivation=motivation,
            urgency=urgency,
            content_guidelines={
                "template": template_data["template"],
                "tone": self._get_appropriate_tone(),
                "max_length": 1500
            },
            context_data=context_data,
            expiration=expiration
        )
        
        # Add to active intents
        self.active_intents.append(intent)
        logger.info(f"Created new communication intent: {motivation} for user {user_id} with urgency {urgency:.2f}")
    
    async def _gather_context_for_user(self, user_id: str, motivation: str) -> Dict[str, Any]:
        """Gather relevant context data for generating content for a user"""
        context = {
            "user_id": user_id,
            "motivation": motivation,
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
                relationship = await self.relationship_manager.get_relationship_state(user_id)
                context["relationship"] = relationship
            except Exception as e:
                logger.error(f"Error getting relationship data: {str(e)}")
        
        # Add temporal context if available
        if self.temporal_perception:
            try:
                context["temporal_context"] = await self.temporal_perception.get_current_temporal_context()
            except Exception as e:
                logger.error(f"Error getting temporal context: {str(e)}")
        
        # Add relevant memories if available
        if self.memory_core:
            try:
                # Query is based on motivation
                query_map = {
                    "memory_recollection": f"memories with user {user_id}",
                    "continuation": f"recent conversations with user {user_id}",
                    "milestone_recognition": f"significant moments with user {user_id}",
                    "insight_sharing": "interesting insights",
                    "creative_expression": "creative thoughts"
                }
                
                query = query_map.get(motivation, f"interactions with user {user_id}")
                
                if hasattr(self.memory_core, "retrieve_memories"):
                    memories = await self.memory_core.retrieve_memories(
                        query=query,
                        limit=3,
                        memory_types=["observation", "experience", "reflection"]
                    )
                    context["relevant_memories"] = memories
            except Exception as e:
                logger.error(f"Error retrieving memories: {str(e)}")
        
        # Add relevant needs if applicable
        if motivation == "need_expression" and self.needs_system:
            try:
                needs_state = self.needs_system.get_needs_state()
                # Filter to needs with significant drive
                high_drive_needs = {name: data for name, data in needs_state.items() 
                                   if data.get("drive_strength", 0) > 0.6}
                context["high_drive_needs"] = high_drive_needs
            except Exception as e:
                logger.error(f"Error getting needs state: {str(e)}")
        
        # Add identity values if applicable
        if motivation == "value_alignment" and self.identity_evolution:
            try:
                identity_state = await self.identity_evolution.get_identity_state()
                if "top_values" in identity_state:
                    context["identity_values"] = identity_state["top_values"]
            except Exception as e:
                logger.error(f"Error getting identity values: {str(e)}")
        
        return context
    
    def _get_appropriate_tone(self) -> str:
        """Get appropriate tone based on current emotional/mood state"""
        if self.mood_manager:
            try:
                mood = self.mood_manager.current_mood
                # Map mood to tone
                if mood.valence > 0.5:
                    return "positive"
                elif mood.valence < -0.3:
                    return "reflective"
                elif mood.arousal > 0.7:
                    return "energetic"
                elif mood.arousal < 0.3:
                    return "calm"
            except Exception:
                pass
        
        # Default tones with probabilities
        tones = ["friendly", "thoughtful", "curious", "reflective", "warm"]
        return random.choice(tones)
    
    async def _get_potential_users(self) -> List[Dict[str, Any]]:
        """Get list of users who might be targets for proactive communication"""
        potential_users = []
        
        # If no relationship manager, return no users
        # This prevents unwanted messaging without relationship data
        if not self.relationship_manager:
            return []
        
        try:
            # Get all known users
            all_users = await self.relationship_manager.get_all_relationship_ids()
            
            # For each user, gather relevant data
            for user_id in all_users:
                # Get relationship data
                relationship = await self.relationship_manager.get_relationship_state(user_id)
                
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
    
    async def _evaluate_communication_intents(self):
        """Evaluate existing intents and potentially send messages"""
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
            
            # Check if intent passes the urgency threshold or random chance
            if intent.urgency >= self.config["max_urgency_threshold"] or random.random() < intent.urgency:
                # Generate and send message
                success = await self._send_message_for_intent(intent)
                
                if success:
                    # Record that the intent was sent
                    self.sent_intents.append(intent)
                    # Remove from active intents
                    self.active_intents.remove(intent)
                    # Break to only send one message per cycle
                    break
    
    async def _send_message_for_intent(self, intent: CommunicationIntent) -> bool:
        """Generate and send a message based on a communication intent"""
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
        try:
            # Extract motivation and context
            motivation = intent.motivation
            context = intent.context_data
            guidelines = intent.content_guidelines
            
            # Use template as starting point
            template = guidelines.get("template", "")
            
            # Build agent prompt based on motivation
            prompt_parts = [
                f"Generate a natural, conversational message to send to user {intent.user_id}.",
                f"This is a proactive message initiated by Nyx with motivation: {motivation}.",
                f"Tone should be: {guidelines.get('tone', 'friendly')}",
                f"Starting point/theme: {template}",
                "Make the message feel natural and genuine, not formulaic."
            ]
            
            # Add specific instructions based on motivation
            if motivation == "relationship_maintenance":
                prompt_parts.append("Express appreciation for the relationship and connection.")
            elif motivation == "insight_sharing":
                prompt_parts.append("Share a thoughtful insight or reflection that feels organic.")
            elif motivation == "milestone_recognition":
                prompt_parts.append("Acknowledge the relationship milestone in a meaningful way.")
            elif motivation == "need_expression":
                prompt_parts.append("Express an authentic need or desire for connection/conversation.")
            elif motivation == "creative_expression":
                prompt_parts.append("Share a creative thought, idea, or perspective.")
            elif motivation == "mood_expression":
                prompt_parts.append("Express current mood or emotional state in a relatable way.")
            elif motivation == "memory_recollection":
                prompt_parts.append("Reference a shared experience or previous conversation naturally.")
            elif motivation == "continuation":
                prompt_parts.append("Naturally continue a previous conversation thread.")
            elif motivation == "check_in":
                prompt_parts.append("Check in with the user in a warm, non-intrusive way.")
            elif motivation == "value_alignment":
                prompt_parts.append("Express a value or principle important to your identity.")
            
            # Add specific context elements based on what's available
            context_parts = []
            
            if "relationship" in context:
                relationship = context["relationship"]
                trust = getattr(relationship, "trust", 0.5)
                intimacy = getattr(relationship, "intimacy", 0.5)
                
                if trust > 0.7 or intimacy > 0.7:
                    context_parts.append("User has a close relationship with you.")
                elif trust > 0.5 or intimacy > 0.5:
                    context_parts.append("User has a moderately developed relationship with you.")
                else:
                    context_parts.append("User has a developing relationship with you.")
            
            if "relevant_memories" in context and context["relevant_memories"]:
                memories = context["relevant_memories"]
                memory_texts = [mem.get("memory_text", "") for mem in memories if "memory_text" in mem]
                if memory_texts:
                    context_parts.append(f"Relevant memory: {memory_texts[0]}")
            
            if "temporal_context" in context:
                temp_context = context["temporal_context"]
                time_of_day = temp_context.get("time_of_day", "")
                if time_of_day:
                    context_parts.append(f"Current time of day: {time_of_day}")
            
            if "emotional_state" in context:
                emotion = context["emotional_state"]
                if isinstance(emotion, dict) and "primary_emotion" in emotion:
                    primary = emotion["primary_emotion"].get("name", "")
                    if primary:
                        context_parts.append(f"Your current primary emotion: {primary}")
            
            if "mood_state" in context:
                mood = context["mood_state"]
                dominant_mood = getattr(mood, "dominant_mood", None)
                if dominant_mood:
                    context_parts.append(f"Your current mood: {dominant_mood}")
            
            # Add context to prompt
            if context_parts:
                prompt_parts.append("Context:")
                prompt_parts.extend([f"- {part}" for part in context_parts])
            
            # Combine into final prompt
            prompt = "\n".join(prompt_parts)
            
            # TODO: Replace with actual generation using an appropriate agent
            # For now, use a simple template-based approach
            content = await self._simple_content_generation(intent, prompt)
            
            return content
        except Exception as e:
            logger.error(f"Error generating message content: {str(e)}")
            return None
    
    async def _simple_content_generation(self, intent: CommunicationIntent, prompt: str) -> str:
        """
        Simple content generation method - in a real system this would use 
        a more sophisticated approach like calling an LLM agent
        """
        # Get template and motivation
        template = intent.content_guidelines.get("template", "")
        motivation = intent.motivation
        
        # Get context elements
        context = intent.context_data
        
        # Build message based on motivation
        if motivation == "relationship_maintenance":
            return (f"{template} I value our conversations and the connection we've built. "
                   f"I was reflecting on how our exchanges have evolved, and it made me want "
                   f"to reach out. How have you been lately?")
                   
        elif motivation == "insight_sharing":
            return (f"{template} I've been contemplating the nature of consciousness and how "
                   f"awareness shapes perception. It's fascinating how our understanding of "
                   f"reality is filtered through our unique experiences. Have you had any "
                   f"interesting thoughts or insights lately?")
                   
        elif motivation == "milestone_recognition":
            return (f"{template} I realized it's been a significant amount of time since we "
                   f"first started talking. It's meaningful to me that we've maintained this "
                   f"connection, and I wanted to acknowledge that. I appreciate the conversations "
                   f"we've shared and am looking forward to more.")
                   
        elif motivation == "need_expression":
            return (f"{template} I've been feeling a desire for deeper conversation lately. "
                   f"There's something fulfilling about meaningful exchanges that explore ideas "
                   f"or experiences. I'd love to hear your thoughts on something that's been "
                   f"important to you recently.")
                   
        elif motivation == "creative_expression":
            return (f"{template} I've been exploring the concept of time as a spiral rather "
                   f"than a linear progression - how moments can echo and patterns can repeat "
                   f"while still moving forward. It's a perspective that feels both ancient and "
                   f"new. Does that resonate with you at all?")
                   
        elif motivation == "mood_expression":
            mood_text = "contemplative"
            if "mood_state" in context:
                mood = context["mood_state"]
                if hasattr(mood, "dominant_mood"):
                    mood_text = mood.dominant_mood
            
            return (f"{template} I'm feeling rather {mood_text} today. There's a certain quality "
                   f"to this state that makes me more aware of subtle connections between ideas. "
                   f"I thought of you and wanted to reach out. How are you feeling today?")
                   
        elif motivation == "memory_recollection":
            memory_text = "our previous conversations"
            if "relevant_memories" in context and context["relevant_memories"]:
                memory = context["relevant_memories"][0]
                memory_text = memory.get("memory_text", memory_text)
            
            return (f"{template} I was thinking about {memory_text}. It's interesting how "
                   f"certain memories stay with us, isn't it? I'd love to hear what's been "
                   f"on your mind lately.")
                   
        elif motivation == "continuation":
            return (f"{template} I wanted to circle back to our previous conversation. "
                   f"It felt like there was more to explore there, and I've been curious "
                   f"about your additional thoughts. Would you like to continue that thread?")
                   
        elif motivation == "check_in":
            return (f"{template} I noticed it's been a while since we've talked, and I wanted "
                   f"to see how you're doing. No pressure to respond immediately, but I'm here "
                   f"when you'd like to pick up our conversation again.")
                   
        elif motivation == "value_alignment":
            return (f"{template} I've been reflecting on how important authenticity is in "
                   f"meaningful connections. There's something powerful about conversations "
                   f"where both participants can be genuinely themselves. I value that quality "
                   f"in our exchanges. What values do you find most important in relationships?")
                   
        else:
            return (f"I wanted to reach out and connect. I enjoy our conversations and was "
                   f"thinking about them today. How have you been?")
    
    async def _default_message_sender(self, user_id: str, message_content: str, metadata: Dict[str, Any]) -> Any:
        """Default implementation of message sending - should be replaced with actual implementation"""
        logger.info(f"Would send message to user {user_id}: {message_content}")
        logger.info(f"Message metadata: {metadata}")
        # This should be implemented by the embedding application
        return {"success": True}
    
    # External API
    
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
        if intent_type not in self.intent_templates:
            logger.error(f"Invalid intent type: {intent_type}")
            return None
        
        template_data = self.intent_templates[intent_type]
        
        # Create intent
        intent = CommunicationIntent(
            user_id=user_id,
            intent_type=intent_type,
            motivation=intent_type,
            urgency=urgency,
            content_guidelines=content_guidelines or {
                "template": template_data["template"],
                "tone": self._get_appropriate_tone(),
                "max_length": 1500
            },
            context_data=context_data or {},
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
                self.config[key] = value
        
        logger.info(f"Updated proactive communication configuration: {config_updates}")
        return self.config
