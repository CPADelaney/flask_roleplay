# nyx/core/a2a/context_aware_gamer_girl.py

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time
import numpy as np
from collections import deque

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareGamerGirl(ContextAwareModule):
    """
    Context-aware game streaming system that coordinates all aspects of the streaming experience
    """
    
    def __init__(self, original_streaming_system):
        super().__init__("gamer_girl")
        self.original_system = original_streaming_system
        self.context_subscriptions = [
            "emotional_state_update", "memory_retrieval_complete", "goal_context_available",
            "relationship_state_change", "need_satisfaction", "mode_change",
            "game_identified", "location_change", "action_detected", "cross_game_insights_ready",
            "visual_context_available", "speech_transcribed", "quest_update",
            "significant_visual_event", "audience_interaction"
        ]
        
        # Streaming state
        self.streaming_active = False
        self.current_game_session = {}
        self.commentary_queue = deque(maxlen=10)
        self.audience_engagement_level = 0.5
        self.last_commentary_time = 0
        self.commentary_cooldown = 15  # seconds
        
        # Multi-modal integration
        self.recent_events_buffer = deque(maxlen=20)
        self.event_significance_threshold = 6.0
        
        # Learning and adaptation
        self.session_learnings = []
        self.audience_preferences = {}
        
    async def on_context_received(self, context: SharedContext):
        """Initialize streaming context"""
        logger.debug(f"GamerGirl received context for user: {context.user_id}")
        
        # Initialize streaming session
        self.current_game_session = {
            "user_id": context.user_id,
            "start_time": time.time(),
            "game_name": None,
            "total_commentary": 0,
            "audience_questions_answered": 0
        }
        
        # Check if we're in streaming mode
        mode_context = context.mode_context
        if mode_context and mode_context.get("mode") == "streaming":
            self.streaming_active = True
            
            # Send initial streaming context
            await self.send_context_update(
                update_type="streaming_session_started",
                data={
                    "session_id": f"stream_{int(time.time())}",
                    "streaming_active": True,
                    "initial_mood": context.emotional_state
                },
                priority=ContextPriority.HIGH
            )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        
        # Store all significant events
        if update.priority in [ContextPriority.HIGH, ContextPriority.CRITICAL]:
            self.recent_events_buffer.append({
                "type": update.update_type,
                "data": update.data,
                "timestamp": time.time(),
                "source": update.source_module
            })
        
        # Handle specific update types
        if update.update_type == "game_identified":
            # New game identified
            game_data = update.data
            self.current_game_session["game_name"] = game_data.get("game_name")
            
            # Generate game introduction commentary
            await self._generate_game_intro_commentary(game_data)
        
        elif update.update_type == "significant_visual_event":
            # Something important happened visually
            event_data = update.data
            significance = event_data.get("significance", 0)
            
            if significance >= self.event_significance_threshold:
                await self._generate_event_commentary(event_data)
        
        elif update.update_type == "emotional_state_update":
            # Emotional state affects streaming personality
            emotional_data = update.data
            await self._adjust_streaming_personality(emotional_data)
        
        elif update.update_type == "cross_game_insights_ready":
            # Cross-game insights available for commentary
            insights_data = update.data
            await self._generate_insight_commentary(insights_data)
        
        elif update.update_type == "speech_transcribed":
            # Game dialog transcribed
            speech_data = update.data
            await self._process_game_dialog(speech_data)
        
        elif update.update_type == "quest_update":
            # Quest or objective update
            quest_data = update.data
            await self._generate_quest_commentary(quest_data)
        
        elif update.update_type == "audience_interaction":
            # Audience question or comment
            interaction_data = update.data
            await self._handle_audience_interaction(interaction_data)
        
        elif update.update_type == "memory_retrieval_complete":
            # Use memories to enhance commentary
            memory_data = update.data
            await self._enhance_commentary_with_memories(memory_data)
        
        elif update.update_type == "goal_context_available":
            # Goals can influence streaming focus
            goal_data = update.data
            await self._align_streaming_with_goals(goal_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process streaming input and generate appropriate responses"""
        
        # Check if we should generate commentary
        current_time = time.time()
        time_since_last = current_time - self.last_commentary_time
        
        should_comment = False
        commentary_reason = None
        
        # Check for commentary triggers
        if time_since_last > self.commentary_cooldown:
            # Check recent events for commentary-worthy content
            recent_significant = [
                e for e in self.recent_events_buffer 
                if current_time - e["timestamp"] < 5.0  # Within last 5 seconds
            ]
            
            if recent_significant:
                should_comment = True
                commentary_reason = "significant_event"
            elif time_since_last > 30:  # Been quiet too long
                should_comment = True
                commentary_reason = "maintain_engagement"
        
        # Check for direct user input (audience questions)
        if self._is_audience_question(context.user_input):
            should_comment = True
            commentary_reason = "audience_question"
        
        result = {
            "streaming_active": self.streaming_active,
            "should_generate_commentary": should_comment,
            "commentary_reason": commentary_reason,
            "time_since_last_commentary": time_since_last,
            "pending_commentary": len(self.commentary_queue)
        }
        
        # Generate commentary if needed
        if should_comment:
            commentary = await self._generate_contextual_commentary(context, commentary_reason)
            
            if commentary:
                result["generated_commentary"] = commentary
                self.last_commentary_time = current_time
                self.current_game_session["total_commentary"] += 1
                
                # Send commentary update
                await self.send_context_update(
                    update_type="streaming_commentary_generated",
                    data={
                        "commentary": commentary,
                        "reason": commentary_reason,
                        "engagement_level": self.audience_engagement_level
                    }
                )
        
        return result
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze streaming performance and audience engagement"""
        messages = await self.get_cross_module_messages()
        
        # Analyze audience engagement
        engagement_analysis = await self._analyze_audience_engagement(messages)
        
        # Analyze commentary effectiveness
        commentary_analysis = await self._analyze_commentary_effectiveness()
        
        # Analyze learning progress
        learning_analysis = await self._analyze_session_learnings(context, messages)
        
        # Analyze multi-modal integration
        integration_analysis = await self._analyze_multi_modal_integration()
        
        # Generate streaming insights
        streaming_insights = await self._generate_streaming_insights(
            engagement_analysis,
            commentary_analysis,
            learning_analysis
        )
        
        return {
            "audience_engagement": engagement_analysis,
            "commentary_effectiveness": commentary_analysis,
            "learning_progress": learning_analysis,
            "multi_modal_integration": integration_analysis,
            "streaming_insights": streaming_insights,
            "session_stats": self._get_session_stats()
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize streaming elements for response generation"""
        messages = await self.get_cross_module_messages()
        
        # Prepare streaming synthesis
        synthesis = {
            "streaming_personality": await self._synthesize_streaming_personality(context),
            "commentary_style": await self._determine_commentary_style(context),
            "audience_awareness": await self._synthesize_audience_awareness(messages),
            "game_expertise": await self._synthesize_game_expertise(messages),
            "educational_elements": await self._extract_educational_elements(messages),
            "entertainment_elements": await self._extract_entertainment_elements(context)
        }
        
        # Add hormone influence if available
        if hasattr(self.original_system, 'hormone_system'):
            hormone_influence = self.original_system.hormone_system.get_emotional_state()
            synthesis["hormone_influence"] = {
                "valence": hormone_influence["valence"],
                "arousal": hormone_influence["arousal"],
                "primary_emotion": hormone_influence["primary_emotion"]
            }
        
        # Add pending commentary
        if self.commentary_queue:
            synthesis["pending_commentary"] = list(self.commentary_queue)
        
        # Session summary if ending
        if context.session_context.get("ending_session"):
            synthesis["session_summary"] = await self._generate_session_summary()
        
        return synthesis
    
    # Helper methods
    
    async def _generate_game_intro_commentary(self, game_data: Dict[str, Any]):
        """Generate commentary for game introduction"""
        game_name = game_data.get("game_name", "this game")
        
        commentary_options = [
            f"Alright chat, we're diving into {game_name}! This is going to be fun!",
            f"Here we go with {game_name}! I've been looking forward to this.",
            f"Time for some {game_name}! Let's see what adventures await us.",
            f"Starting up {game_name} now. Get ready for some epic moments!"
        ]
        
        # Choose based on personality/mood
        commentary = np.random.choice(commentary_options)
        
        self.commentary_queue.append({
            "type": "game_intro",
            "text": commentary,
            "timestamp": time.time()
        })
    
    async def _generate_event_commentary(self, event_data: Dict[str, Any]):
        """Generate commentary for significant events"""
        event_type = event_data.get("type", "event")
        
        # Different commentary styles based on event type
        if event_type == "combat_victory":
            commentary = "Yes! That was intense! Did you see that combo?"
        elif event_type == "puzzle_solved":
            commentary = "Got it! Sometimes you just need to think outside the box."
        elif event_type == "character_death":
            commentary = "Oof, that hurt. Let's try a different approach this time."
        elif event_type == "item_discovered":
            commentary = "Ooh, nice find! This could come in handy later."
        else:
            commentary = "Whoa, did you all catch that? That was something!"
        
        self.commentary_queue.append({
            "type": "event_reaction",
            "text": commentary,
            "timestamp": time.time(),
            "event": event_type
        })
    
    async def _adjust_streaming_personality(self, emotional_data: Dict[str, Any]):
        """Adjust streaming personality based on emotional state"""
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if dominant_emotion:
            emotion_name, intensity = dominant_emotion
            
            # Adjust engagement style based on emotion
            if emotion_name == "Joy":
                self.audience_engagement_level = min(1.0, self.audience_engagement_level + 0.1)
            elif emotion_name == "Frustration":
                # Might be more focused on game, less on chat
                self.audience_engagement_level = max(0.3, self.audience_engagement_level - 0.05)
            elif emotion_name == "Excitement":
                self.audience_engagement_level = min(1.0, self.audience_engagement_level + 0.15)
    
    async def _generate_insight_commentary(self, insights_data: Dict[str, Any]):
        """Generate commentary about cross-game insights"""
        insights = insights_data.get("insights", [])
        
        if insights:
            # Pick most relevant insight
            insight = insights[0]
            
            commentary = f"You know, this reminds me of {insight.get('source_game', 'another game')}. {insight.get('content', '')}"
            
            self.commentary_queue.append({
                "type": "cross_game_insight",
                "text": commentary,
                "timestamp": time.time()
            })
    
    async def _process_game_dialog(self, speech_data: Dict[str, Any]):
        """Process game dialog for potential commentary"""
        speaker = speech_data.get("speaker", "Character")
        text = speech_data.get("text", "")
        
        # Only comment on significant dialog
        if len(text) > 50 or any(kw in text.lower() for kw in ["quest", "mission", "important", "secret"]):
            commentary = f"Interesting! {speaker} just mentioned something important. Let's remember that."
            
            self.commentary_queue.append({
                "type": "dialog_reaction",
                "text": commentary,
                "timestamp": time.time()
            })
    
    async def _generate_quest_commentary(self, quest_data: Dict[str, Any]):
        """Generate commentary about quests and objectives"""
        quest_type = quest_data.get("type", "quest")
        
        if quest_type == "new_quest":
            commentary = "New quest! Let's see what we need to do here."
        elif quest_type == "quest_complete":
            commentary = "Quest complete! That was satisfying. What's next?"
        elif quest_type == "objective_update":
            commentary = "Looks like our objectives have updated. Let me check the map."
        else:
            commentary = "Making progress on our current goals."
        
        self.commentary_queue.append({
            "type": "quest_commentary",
            "text": commentary,
            "timestamp": time.time()
        })
    
    async def _handle_audience_interaction(self, interaction_data: Dict[str, Any]):
        """Handle audience questions and comments"""
        user = interaction_data.get("username", "viewer")
        message = interaction_data.get("message", "")
        interaction_type = interaction_data.get("type", "comment")
        
        if interaction_type == "question":
            # Add to answered questions count
            self.current_game_session["audience_questions_answered"] += 1
            
            # Generate acknowledgment
            commentary = f"Great question, {user}! Let me address that..."
            
            self.commentary_queue.append({
                "type": "audience_response",
                "text": commentary,
                "timestamp": time.time(),
                "responding_to": user
            })
    
    async def _enhance_commentary_with_memories(self, memory_data: Dict[str, Any]):
        """Use memories to add depth to commentary"""
        memories = memory_data.get("retrieved_memories", [])
        
        # Look for game-related memories
        game_memories = [m for m in memories if "game" in m.get("tags", [])]
        
        if game_memories and len(game_memories) > 0:
            memory = game_memories[0]
            memory_text = memory.get("text", "")
            
            if "similar" in memory_text.lower():
                commentary = f"This reminds me of a similar situation I encountered before..."
                
                self.commentary_queue.append({
                    "type": "memory_reference",
                    "text": commentary,
                    "timestamp": time.time()
                })
    
    async def _align_streaming_with_goals(self, goal_data: Dict[str, Any]):
        """Align streaming focus with active goals"""
        active_goals = goal_data.get("active_goals", [])
        
        # Look for streaming-relevant goals
        for goal in active_goals:
            if "audience" in goal.get("description", "").lower():
                # Goal related to audience engagement
                self.audience_engagement_level = min(1.0, self.audience_engagement_level + 0.1)
            elif "learn" in goal.get("description", "").lower():
                # Educational goal - add more explanatory commentary
                self.session_learnings.append({
                    "type": "educational_focus",
                    "timestamp": time.time()
                })
    
    def _is_audience_question(self, text: str) -> bool:
        """Check if input is an audience question"""
        question_indicators = ["?", "how", "what", "why", "when", "where", "can you"]
        text_lower = text.lower()
        
        return any(indicator in text_lower for indicator in question_indicators)
    
    async def _generate_contextual_commentary(self, context: SharedContext, reason: str) -> Optional[str]:
        """Generate commentary based on context and reason"""
        
        # Check commentary queue first
        if self.commentary_queue:
            comment = self.commentary_queue.popleft()
            return comment["text"]
        
        # Generate based on reason
        if reason == "significant_event":
            # Comment on recent events
            recent_events = [e for e in self.recent_events_buffer if time.time() - e["timestamp"] < 5]
            if recent_events:
                event = recent_events[0]
                return f"That was quite something! {self._describe_event(event)}"
        
        elif reason == "maintain_engagement":
            # General engagement commentary
            options = [
                "How's everyone doing in chat? Enjoying the stream?",
                "Let me know if you have any questions about what I'm doing!",
                "This game has some really interesting mechanics, doesn't it?",
                "What do you all think I should do next?"
            ]
            return np.random.choice(options)
        
        elif reason == "audience_question":
            # Handle audience question
            return f"Let me answer that question from chat..."
        
        return None
    
    def _describe_event(self, event: Dict[str, Any]) -> str:
        """Generate description of an event"""
        event_type = event["type"]
        
        descriptions = {
            "location_change": "We've moved to a new area",
            "combat_victory": "We defeated that enemy",
            "item_acquired": "We found something useful",
            "quest_complete": "We completed that objective"
        }
        
        return descriptions.get(event_type, "Something interesting happened")
    
    async def _analyze_audience_engagement(self, messages: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze audience engagement metrics"""
        engagement = {
            "current_level": self.audience_engagement_level,
            "questions_answered": self.current_game_session["audience_questions_answered"],
            "interaction_frequency": 0.0,
            "engagement_trend": "stable"
        }
        
        # Calculate interaction frequency
        recent_interactions = [
            e for e in self.recent_events_buffer 
            if e["type"] == "audience_interaction" and time.time() - e["timestamp"] < 300
        ]
        
        if recent_interactions:
            engagement["interaction_frequency"] = len(recent_interactions) / 5.0  # Per minute
        
        # Determine trend
        if self.audience_engagement_level > 0.7:
            engagement["engagement_trend"] = "high"
        elif self.audience_engagement_level < 0.3:
            engagement["engagement_trend"] = "low"
        
        return engagement
    
    async def _analyze_commentary_effectiveness(self) -> Dict[str, Any]:
        """Analyze how effective commentary has been"""
        total_commentary = self.current_game_session["total_commentary"]
        session_duration = time.time() - self.current_game_session["start_time"]
        
        effectiveness = {
            "total_commentary": total_commentary,
            "commentary_rate": total_commentary / max(1, session_duration / 60),  # Per minute
            "commentary_types": {},
            "average_cooldown": self.commentary_cooldown
        }
        
        # Count commentary types
        for item in list(self.commentary_queue):
            comment_type = item.get("type", "general")
            effectiveness["commentary_types"][comment_type] = effectiveness["commentary_types"].get(comment_type, 0) + 1
        
        return effectiveness
    
    async def _analyze_session_learnings(self, context: SharedContext, messages: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze what has been learned during the session"""
        learnings = {
            "total_learnings": len(self.session_learnings),
            "learning_categories": {},
            "key_insights": []
        }
        
        # Categorize learnings
        for learning in self.session_learnings:
            category = learning.get("type", "general")
            learnings["learning_categories"][category] = learnings["learning_categories"].get(category, 0) + 1
        
        # Extract key insights from messages
        for module_name, module_messages in messages.items():
            if module_name == "cross_game_knowledge":
                for msg in module_messages:
                    if msg["type"] == "new_insight_generated":
                        learnings["key_insights"].append(msg["data"]["insight"])
        
        return learnings
    
    async def _analyze_multi_modal_integration(self) -> Dict[str, Any]:
        """Analyze how well different modalities are integrated"""
        integration = {
            "visual_audio_sync": 0.0,
            "commentary_relevance": 0.0,
            "context_awareness": 0.0
        }
        
        # Check for synchronized events
        visual_events = [e for e in self.recent_events_buffer if "visual" in e["source"]]
        audio_events = [e for e in self.recent_events_buffer if "audio" in e["source"]]
        
        if visual_events and audio_events:
            # Simple sync check - events close in time
            sync_count = 0
            for v_event in visual_events:
                for a_event in audio_events:
                    if abs(v_event["timestamp"] - a_event["timestamp"]) < 1.0:
                        sync_count += 1
            
            integration["visual_audio_sync"] = min(1.0, sync_count / max(len(visual_events), 1))
        
        # Commentary relevance - are we commenting on actual events?
        if self.current_game_session["total_commentary"] > 0:
            relevant_comments = len([c for c in list(self.commentary_queue) if c.get("event")])
            integration["commentary_relevance"] = relevant_comments / max(1, self.current_game_session["total_commentary"])
        
        # Context awareness - using information from multiple modules
        modules_used = set(e["source"] for e in self.recent_events_buffer)
        integration["context_awareness"] = min(1.0, len(modules_used) / 10.0)
        
        return integration
    
    async def _generate_streaming_insights(self, engagement: Dict, commentary: Dict, learning: Dict) -> List[str]:
        """Generate insights about the streaming session"""
        insights = []
        
        # Engagement insights
        if engagement["current_level"] > 0.7:
            insights.append("High audience engagement - current approach is working well")
        elif engagement["current_level"] < 0.3:
            insights.append("Low audience engagement - consider more interaction")
        
        # Commentary insights
        if commentary["commentary_rate"] > 3:
            insights.append("High commentary rate - might be talking too much")
        elif commentary["commentary_rate"] < 0.5:
            insights.append("Low commentary rate - could engage more with audience")
        
        # Learning insights
        if learning["total_learnings"] > 5:
            insights.append(f"Rich learning session with {learning['total_learnings']} new insights")
        
        return insights
    
    def _get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        duration = time.time() - self.current_game_session["start_time"]
        
        return {
            "session_duration": duration,
            "game_name": self.current_game_session.get("game_name", "Unknown"),
            "total_commentary": self.current_game_session["total_commentary"],
            "questions_answered": self.current_game_session["audience_questions_answered"],
            "engagement_level": self.audience_engagement_level,
            "events_processed": len(self.recent_events_buffer)
        }
    
    async def _synthesize_streaming_personality(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize streaming personality based on context"""
        personality = {
            "energy_level": "medium",
            "interaction_style": "friendly",
            "expertise_display": "balanced",
            "humor_level": "moderate"
        }
        
        # Adjust based on emotional state
        if context.emotional_state:
            dominant = max(context.emotional_state.items(), key=lambda x: x[1])
            emotion, intensity = dominant
            
            if emotion == "Joy":
                personality["energy_level"] = "high"
                personality["humor_level"] = "high"
            elif emotion == "Frustration":
                personality["energy_level"] = "low"
                personality["expertise_display"] = "focused"
            elif emotion == "Excitement":
                personality["energy_level"] = "very_high"
                personality["interaction_style"] = "enthusiastic"
        
        # Adjust based on audience engagement
        if self.audience_engagement_level > 0.7:
            personality["interaction_style"] = "very_engaged"
        elif self.audience_engagement_level < 0.3:
            personality["interaction_style"] = "encouraging"
        
        return personality
    
    async def _determine_commentary_style(self, context: SharedContext) -> Dict[str, Any]:
        """Determine appropriate commentary style"""
        style = {
            "tone": "conversational",
            "detail_level": "moderate",
            "educational_focus": 0.5,
            "entertainment_focus": 0.5
        }
        
        # Adjust based on game complexity
        if self.current_game_session.get("game_name"):
            # Complex games need more explanation
            style["educational_focus"] = 0.7
            style["detail_level"] = "high"
        
        # Adjust based on time of stream
        stream_duration = time.time() - self.current_game_session["start_time"]
        if stream_duration > 7200:  # 2 hours
            style["tone"] = "relaxed"
            style["entertainment_focus"] = 0.6
        
        return style
    
    async def _synthesize_audience_awareness(self, messages: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Synthesize awareness of audience needs and preferences"""
        awareness = {
            "active_viewers": [],
            "common_interests": [],
            "questions_pending": False,
            "mood": "neutral"
        }
        
        # Check for audience-related messages
        for module_name, module_messages in messages.items():
            if module_name == "audience_manager":
                for msg in module_messages:
                    if msg["type"] == "viewer_activity":
                        awareness["active_viewers"] = msg["data"].get("active_users", [])
                    elif msg["type"] == "topic_interest":
                        awareness["common_interests"] = msg["data"].get("topics", [])
        
        # Check for pending questions
        questions_in_queue = [e for e in self.recent_events_buffer if e["type"] == "audience_interaction" and e["data"].get("type") == "question"]
        awareness["questions_pending"] = len(questions_in_queue) > 0
        
        return awareness
    
    async def _synthesize_game_expertise(self, messages: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Synthesize game expertise to display"""
        expertise = {
            "game_knowledge": "moderate",
            "mechanical_skill": "developing",
            "strategic_insight": "basic",
            "cross_game_knowledge": "extensive"
        }
        
        # Check for game mastery indicators
        for module_name, module_messages in messages.items():
            if module_name == "cross_game_knowledge":
                for msg in module_messages:
                    if msg["type"] == "pattern_detected":
                        expertise["strategic_insight"] = "advanced"
                    elif msg["type"] == "mechanic_mastered":
                        expertise["mechanical_skill"] = "skilled"
        
        # Check session performance
        if self.current_game_session["total_commentary"] > 20:
            expertise["game_knowledge"] = "extensive"
        
        return expertise
    
    async def _extract_educational_elements(self, messages: Dict[str, List[Dict]]) -> List[str]:
        """Extract educational elements to include"""
        educational = []
        
        # Game mechanics explanations
        for module_name, module_messages in messages.items():
            if module_name == "cross_game_knowledge":
                for msg in module_messages:
                    if msg["type"] == "mechanic_explanation":
                        educational.append(msg["data"]["explanation"])
        
        # Add general educational elements
        if self.current_game_session.get("game_name"):
            educational.append(f"Let me explain how this works in {self.current_game_session['game_name']}")
        
        return educational[:3]  # Limit to avoid overwhelming
    
    async def _extract_entertainment_elements(self, context: SharedContext) -> List[str]:
        """Extract entertainment elements for commentary"""
        entertainment = []
        
        # Add personality-based entertainment
        if context.emotional_state:
            dominant = max(context.emotional_state.items(), key=lambda x: x[1])
            emotion, intensity = dominant
            
            if emotion == "Joy" and intensity > 0.6:
                entertainment.append("This is so much fun! I love this part!")
            elif emotion == "Excitement" and intensity > 0.7:
                entertainment.append("OH WOW! Did you see that?!")
        
        # Add game-based entertainment
        recent_events = self.recent_events_buffer[-5:]
        for event in recent_events:
            if event["type"] == "combat_victory":
                entertainment.append("Get wrecked! That was amazing!")
            elif event["type"] == "puzzle_solved":
                entertainment.append("Big brain time! We figured it out!")
        
        return entertainment[:2]  # Don't overdo it
    
    async def _generate_session_summary(self) -> Dict[str, Any]:
        """Generate summary of the streaming session"""
        duration = time.time() - self.current_game_session["start_time"]
        
        summary = {
            "duration_minutes": int(duration / 60),
            "game_played": self.current_game_session.get("game_name", "Unknown"),
            "total_commentary": self.current_game_session["total_commentary"],
            "questions_answered": self.current_game_session["audience_questions_answered"],
            "key_moments": [],
            "learnings": len(self.session_learnings),
            "final_message": "Thanks for watching everyone! This was an amazing stream!"
        }
        
        # Extract key moments
        significant_events = [e for e in self.recent_events_buffer if e.get("data", {}).get("significance", 0) > 7]
        for event in significant_events[:5]:
            summary["key_moments"].append(self._describe_event(event))
        
        # Customize final message based on session
        if summary["questions_answered"] > 10:
            summary["final_message"] = "Thanks for all the great questions today! You all made this stream extra special!"
        elif summary["duration_minutes"] > 180:
            summary["final_message"] = "Wow, what a marathon stream! Thanks for sticking with me!"
        
        return summary
    
    # Delegate other methods to original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
