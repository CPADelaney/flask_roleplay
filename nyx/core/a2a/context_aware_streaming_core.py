# nyx/core/a2a/context_aware_streaming_core.py

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareStreamingCore(ContextAwareModule):
    """
    Advanced StreamingCore with full context distribution capabilities
    """
    
    def __init__(self, original_streaming_core):
        super().__init__("streaming_core")
        self.original_core = original_streaming_core
        self.context_subscriptions = [
            "emotional_state_update", "memory_retrieval_complete", 
            "goal_progress", "relationship_state_change",
            "hormonal_state_update", "cross_game_insight",
            "audience_reaction", "identity_update"
        ]
        
        # Track streaming-specific context
        self.streaming_context = {
            "current_game": None,
            "session_active": False,
            "audience_engagement": 0.5,
            "commentary_style": "balanced",
            "recent_insights": []
        }
    
    async def on_context_received(self, context: SharedContext):
        """Initialize streaming processing for this context"""
        logger.debug(f"StreamingCore received context for user: {context.user_id}")
        
        # Check if this is a streaming-related interaction
        streaming_relevant = await self._analyze_streaming_relevance(context.user_input)
        
        if streaming_relevant or self.streaming_context["session_active"]:
            # Send streaming context to other modules
            await self.send_context_update(
                update_type="streaming_context_available",
                data={
                    "streaming_active": self.streaming_context["session_active"],
                    "current_game": self.streaming_context["current_game"],
                    "audience_engagement": self.streaming_context["audience_engagement"],
                    "streaming_relevant": streaming_relevant
                },
                priority=ContextPriority.HIGH
            )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules with streaming implications"""
        
        if update.update_type == "emotional_state_update":
            # Adjust commentary style based on emotions
            emotional_data = update.data
            await self._adjust_commentary_from_emotion(emotional_data)
        
        elif update.update_type == "hormonal_state_update":
            # Sync with hormone system for streaming reactions
            hormone_data = update.data
            if self.streaming_context["session_active"]:
                await self._process_hormonal_influence(hormone_data)
        
        elif update.update_type == "memory_retrieval_complete":
            # Use retrieved memories for commentary enhancement
            memory_data = update.data
            if self.streaming_context["session_active"]:
                await self._enhance_commentary_with_memories(memory_data)
        
        elif update.update_type == "goal_progress":
            # Incorporate goal progress into streaming narrative
            goal_data = update.data
            if self.streaming_context["session_active"]:
                await self._incorporate_goal_progress(goal_data)
        
        elif update.update_type == "relationship_state_change":
            # Adjust audience interaction based on relationship understanding
            relationship_data = update.data
            await self._adjust_audience_interaction(relationship_data)
        
        elif update.update_type == "cross_game_insight":
            # Store and utilize cross-game insights
            insight_data = update.data
            self.streaming_context["recent_insights"].append(insight_data)
            
            # Limit recent insights
            if len(self.streaming_context["recent_insights"]) > 10:
                self.streaming_context["recent_insights"] = self.streaming_context["recent_insights"][-10:]
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with streaming awareness"""
        # Check if this is a streaming command
        streaming_command = await self._parse_streaming_command(context.user_input)
        
        result = {}
        if streaming_command:
            # Execute streaming command
            result = await self._execute_streaming_command(streaming_command, context)
            
            # Send update about streaming state change
            await self.send_context_update(
                update_type="streaming_state_change",
                data={
                    "command": streaming_command,
                    "result": result,
                    "session_active": self.streaming_context["session_active"]
                }
            )
        
        # Get cross-module messages for context
        messages = await self.get_cross_module_messages()
        
        # Process any streaming-relevant messages
        streaming_effects = await self._process_streaming_messages(messages)
        
        return {
            "streaming_processed": True,
            "command": streaming_command,
            "result": result,
            "cross_module_effects": streaming_effects,
            "session_active": self.streaming_context["session_active"]
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze streaming state in context"""
        # Get current streaming metrics if active
        streaming_metrics = {}
        if self.streaming_context["session_active"]:
            streaming_metrics = await self._get_streaming_metrics()
        
        # Analyze streaming patterns
        streaming_patterns = await self._analyze_streaming_patterns(context)
        
        # Analyze audience engagement
        engagement_analysis = await self._analyze_audience_engagement(context)
        
        # Analyze cross-game connections
        cross_game_analysis = await self._analyze_cross_game_connections(context)
        
        return {
            "streaming_metrics": streaming_metrics,
            "streaming_patterns": streaming_patterns,
            "engagement_analysis": engagement_analysis,
            "cross_game_analysis": cross_game_analysis,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize streaming components for response"""
        messages = await self.get_cross_module_messages()
        
        # Generate streaming influence on response
        streaming_influence = {
            "should_mention_streaming": await self._should_mention_streaming(context),
            "streaming_insights": await self._get_relevant_streaming_insights(context),
            "commentary_suggestions": await self._generate_commentary_suggestions(context, messages),
            "audience_considerations": await self._get_audience_considerations(context),
            "streaming_narrative": await self._generate_streaming_narrative(context)
        }
        
        # If streaming is active, send performance update
        if self.streaming_context["session_active"]:
            performance_data = await self._get_performance_data()
            await self.send_context_update(
                update_type="streaming_performance_update",
                data=performance_data,
                priority=ContextPriority.NORMAL
            )
        
        return {
            "streaming_influence": streaming_influence,
            "synthesis_complete": True,
            "session_active": self.streaming_context["session_active"]
        }
    
    # ========================================================================================
    # STREAMING-SPECIFIC METHODS
    # ========================================================================================
    
    async def _analyze_streaming_relevance(self, user_input: str) -> bool:
        """Determine if input is relevant to streaming"""
        input_lower = user_input.lower()
        
        streaming_keywords = [
            "stream", "streaming", "game", "play", "audience", "viewer",
            "commentary", "broadcast", "live", "gaming"
        ]
        
        return any(keyword in input_lower for keyword in streaming_keywords)
    
    async def _parse_streaming_command(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Parse streaming commands from input"""
        input_lower = user_input.lower()
        
        if any(phrase in input_lower for phrase in ["start stream", "begin streaming", "let's stream"]):
            return {"action": "start_streaming"}
        elif any(phrase in input_lower for phrase in ["stop stream", "end streaming", "finish streaming"]):
            return {"action": "stop_streaming"}
        elif "add question" in input_lower or "audience asked" in input_lower:
            return {"action": "add_audience_question", "question": user_input}
        elif "streaming stats" in input_lower or "how's the stream" in input_lower:
            return {"action": "get_streaming_stats"}
        
        return None
    
    async def _execute_streaming_command(self, command: Dict[str, Any], context: SharedContext) -> Dict[str, Any]:
        """Execute a streaming command"""
        action = command.get("action")
        
        if action == "start_streaming":
            result = await self.original_core.start_streaming()
            if result.get("status") == "streaming_started":
                self.streaming_context["session_active"] = True
                self.streaming_context["current_game"] = "Unknown"
        
        elif action == "stop_streaming":
            result = await self.original_core.stop_streaming()
            if result.get("status") == "streaming_stopped":
                self.streaming_context["session_active"] = False
                self.streaming_context["current_game"] = None
        
        elif action == "add_audience_question":
            # Extract question details
            question = command.get("question", "")
            result = await self.original_core.add_audience_question(
                user_id=context.user_id or "unknown",
                username="User",
                question=question
            )
        
        elif action == "get_streaming_stats":
            result = await self.original_core.get_streaming_stats()
        
        else:
            result = {"error": "Unknown streaming command"}
        
        return result
    
    async def _adjust_commentary_from_emotion(self, emotional_data: Dict[str, Any]):
        """Adjust commentary style based on emotional state"""
        emotional_state = emotional_data.get("emotional_state", {})
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if not dominant_emotion:
            return
        
        emotion_name, strength = dominant_emotion
        
        # Map emotions to commentary styles
        emotion_commentary_map = {
            "Excitement": "enthusiastic",
            "Joy": "upbeat",
            "Curiosity": "analytical",
            "Frustration": "determined",
            "Calm": "measured",
            "Anxiety": "cautious"
        }
        
        if emotion_name in emotion_commentary_map and strength > 0.5:
            self.streaming_context["commentary_style"] = emotion_commentary_map[emotion_name]
            
            # Send update about commentary style change
            await self.send_context_update(
                update_type="commentary_style_change",
                data={
                    "new_style": self.streaming_context["commentary_style"],
                    "influenced_by": emotion_name,
                    "strength": strength
                },
                scope=ContextScope.GLOBAL
            )
    
    async def _process_hormonal_influence(self, hormone_data: Dict[str, Any]):
        """Process hormonal influence on streaming"""
        if hasattr(self.original_core, "hormone_system"):
            # Update hormone system with brain hormone data
            await self.original_core.hormone_system.sync_with_brain_hormone_system()
            
            # Get commentary influence
            influence = self.original_core.hormone_system.get_commentary_influence()
            
            if "dominant" in influence:
                dominant = influence["dominant"]
                self.streaming_context["commentary_style"] = dominant.get("style", "balanced")
    
    async def _enhance_commentary_with_memories(self, memory_data: Dict[str, Any]):
        """Enhance commentary using retrieved memories"""
        memories = memory_data.get("retrieved_memories", [])
        
        if not memories or not self.streaming_context["session_active"]:
            return
        
        # Store relevant memories for commentary use
        if hasattr(self.original_core, "streaming_system"):
            game_state = self.original_core.streaming_system.game_state
            if game_state:
                game_state.retrieved_memories = memories[:3]  # Top 3 most relevant
    
    async def _incorporate_goal_progress(self, goal_data: Dict[str, Any]):
        """Incorporate goal progress into streaming narrative"""
        if not self.streaming_context["session_active"]:
            return
        
        goal_context = goal_data.get("goal_context", {})
        active_goals = goal_context.get("active_goals", [])
        
        # Find gaming-related goals
        gaming_goals = [
            goal for goal in active_goals 
            if any(keyword in goal.get("description", "").lower() 
                   for keyword in ["game", "stream", "play", "skill"])
        ]
        
        if gaming_goals and hasattr(self.original_core, "streaming_system"):
            # Add goal context to game state
            game_state = self.original_core.streaming_system.game_state
            if game_state:
                game_state.active_goals = gaming_goals
    
    async def _adjust_audience_interaction(self, relationship_data: Dict[str, Any]):
        """Adjust audience interaction based on relationship understanding"""
        relationship_context = relationship_data.get("relationship_context", {})
        trust_level = relationship_context.get("trust", 0.5)
        intimacy_level = relationship_context.get("intimacy", 0.5)
        
        # Calculate audience engagement based on relationship metrics
        engagement = (trust_level + intimacy_level) / 2
        self.streaming_context["audience_engagement"] = engagement
        
        # Update environmental factors if streaming is active
        if self.streaming_context["session_active"] and hasattr(self.original_core, "hormone_system"):
            await self.original_core.hormone_system.update_environmental_factors({
                "audience_engagement": engagement
            })
    
    async def _process_streaming_messages(self, messages: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Process cross-module messages for streaming relevance"""
        effects = []
        
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                if msg["type"] == "cross_game_insight":
                    # Store cross-game insight
                    self.streaming_context["recent_insights"].append(msg["data"])
                    effects.append({
                        "module": module_name,
                        "effect": "stored_cross_game_insight"
                    })
                
                elif msg["type"] == "audience_reaction":
                    # Process audience reaction
                    reaction_data = msg["data"]
                    effects.append({
                        "module": module_name,
                        "effect": "processed_audience_reaction",
                        "data": reaction_data
                    })
        
        return effects
    
    async def _get_streaming_metrics(self) -> Dict[str, Any]:
        """Get current streaming metrics"""
        if hasattr(self.original_core, "get_streaming_stats"):
            return await self.original_core.get_streaming_stats()
        return {}
    
    async def _analyze_streaming_patterns(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze patterns in streaming behavior"""
        patterns = {
            "commentary_consistency": 0.8,
            "audience_responsiveness": self.streaming_context["audience_engagement"],
            "insight_frequency": len(self.streaming_context["recent_insights"]) / 10.0,
            "style_stability": 0.7
        }
        
        return patterns
    
    async def _analyze_audience_engagement(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze audience engagement patterns"""
        return {
            "current_engagement": self.streaming_context["audience_engagement"],
            "engagement_trend": "stable",
            "peak_engagement_topics": ["gameplay", "commentary", "insights"]
        }
    
    async def _analyze_cross_game_connections(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze cross-game connections"""
        insights = self.streaming_context["recent_insights"]
        
        if not insights:
            return {"connections": [], "pattern_count": 0}
        
        # Extract patterns from insights
        game_mentions = {}
        mechanic_mentions = {}
        
        for insight in insights:
            source_game = insight.get("source_game", "")
            if source_game:
                game_mentions[source_game] = game_mentions.get(source_game, 0) + 1
            
            mechanic = insight.get("mechanic", "")
            if mechanic:
                mechanic_mentions[mechanic] = mechanic_mentions.get(mechanic, 0) + 1
        
        return {
            "connections": insights[:3],
            "pattern_count": len(insights),
            "frequent_games": game_mentions,
            "common_mechanics": mechanic_mentions
        }
    
    async def _should_mention_streaming(self, context: SharedContext) -> bool:
        """Determine if streaming should be mentioned in response"""
        # Check if user asked about streaming
        if await self._analyze_streaming_relevance(context.user_input):
            return True
        
        # Check if streaming is active
        if self.streaming_context["session_active"]:
            return True
        
        # Check if recent streaming experience is relevant
        messages = await self.get_cross_module_messages()
        for module_messages in messages.values():
            for msg in module_messages:
                if msg["type"] in ["streaming_memory", "streaming_reflection"]:
                    return True
        
        return False
    
    async def _get_relevant_streaming_insights(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Get streaming insights relevant to current context"""
        relevant_insights = []
        
        for insight in self.streaming_context["recent_insights"]:
            # Check relevance to user input
            if any(keyword in context.user_input.lower() 
                   for keyword in [insight.get("source_game", "").lower(), 
                                  insight.get("mechanic", "").lower()]):
                relevant_insights.append(insight)
        
        return relevant_insights[:2]  # Top 2 most relevant
    
    async def _generate_commentary_suggestions(self, context: SharedContext, messages: Dict) -> List[str]:
        """Generate commentary suggestions based on context"""
        suggestions = []
        
        # Base suggestions on commentary style
        style = self.streaming_context["commentary_style"]
        
        style_suggestions = {
            "enthusiastic": ["Express excitement about discoveries", "Share enthusiasm for gameplay moments"],
            "analytical": ["Break down game mechanics", "Analyze strategic decisions"],
            "upbeat": ["Keep energy positive", "Celebrate small victories"],
            "measured": ["Provide thoughtful observations", "Take time to explain"],
            "cautious": ["Note potential challenges", "Prepare for difficulties"]
        }
        
        suggestions.extend(style_suggestions.get(style, ["Maintain engaging commentary"]))
        
        return suggestions
    
    async def _get_audience_considerations(self, context: SharedContext) -> Dict[str, Any]:
        """Get considerations for audience interaction"""
        return {
            "engagement_level": self.streaming_context["audience_engagement"],
            "recommended_interaction": "high" if self.streaming_context["audience_engagement"] > 0.7 else "moderate",
            "topics_of_interest": ["gameplay", "strategy", "cross-game insights"]
        }
    
    async def _generate_streaming_narrative(self, context: SharedContext) -> str:
        """Generate a narrative element for streaming context"""
        if not self.streaming_context["session_active"]:
            return ""
        
        game = self.streaming_context["current_game"] or "this game"
        style = self.streaming_context["commentary_style"]
        
        narratives = {
            "enthusiastic": f"I'm really enjoying streaming {game}!",
            "analytical": f"There are interesting mechanics to explore in {game}.",
            "upbeat": f"Having a great time with {game} today!",
            "measured": f"Let's thoughtfully explore what {game} has to offer.",
            "cautious": f"Carefully navigating through {game}."
        }
        
        return narratives.get(style, f"Currently streaming {game}.")
    
    async def _get_performance_data(self) -> Dict[str, Any]:
        """Get streaming performance data"""
        if hasattr(self.original_core, "get_performance_metrics"):
            return self.original_core.get_performance_metrics()
        
        return {
            "fps": 30,
            "processing_time": 0.1,
            "resource_usage": {"cpu": 50, "memory": 40}
        }
    
    # Delegate all other methods to the original core
    def __getattr__(self, name):
        """Delegate any missing methods to the original core"""
        return getattr(self.original_core, name)
