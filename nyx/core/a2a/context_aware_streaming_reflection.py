# nyx/core/a2a/context_aware_streaming_reflection.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareStreamingReflectionEngine(ContextAwareModule):
    """
    Advanced StreamingReflectionEngine with full context distribution capabilities
    """
    
    def __init__(self, original_reflection_engine):
        super().__init__("streaming_reflection_engine")
        self.original_engine = original_reflection_engine
        self.context_subscriptions = [
            "streaming_state_change", "memory_consolidation",
            "goal_completion", "emotional_state_update",
            "identity_update", "cross_game_insight",
            "audience_milestone", "streaming_performance_update",
            "experience_stored", "abstraction_created"
        ]
        
        # Track reflection-specific context
        self.reflection_context = {
            "pending_reflections": [],
            "recent_insights": [],
            "consolidation_candidates": [],
            "reflection_depth": 0.7,  # How deep reflections should be
            "cross_domain_awareness": 0.8  # Awareness of non-streaming experiences
        }
    
    async def on_context_received(self, context: SharedContext):
        """Initialize reflection processing for this context"""
        logger.debug(f"StreamingReflectionEngine received context")
        
        # Check if reflection is warranted
        reflection_triggers = await self._identify_reflection_triggers(context)
        
        if reflection_triggers:
            # Send reflection context to other modules
            await self.send_context_update(
                update_type="reflection_context_available",
                data={
                    "reflection_triggers": reflection_triggers,
                    "pending_count": len(self.reflection_context["pending_reflections"]),
                    "consolidation_ready": len(self.reflection_context["consolidation_candidates"]) >= 3
                },
                priority=ContextPriority.NORMAL
            )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that trigger reflection"""
        
        if update.update_type == "streaming_state_change":
            # Handle streaming state changes
            streaming_data = update.data
            if streaming_data.get("command", {}).get("action") == "stop_streaming":
                # Trigger session reflection
                await self._queue_session_reflection(streaming_data)
        
        elif update.update_type == "memory_consolidation":
            # Handle memory consolidation events
            consolidation_data = update.data
            await self._process_memory_consolidation(consolidation_data)
        
        elif update.update_type == "goal_completion":
            # Reflect on completed goals
            goal_data = update.data
            if self._is_streaming_related_goal(goal_data):
                await self._queue_goal_reflection(goal_data)
        
        elif update.update_type == "emotional_state_update":
            # Track emotional patterns for reflection
            emotional_data = update.data
            await self._track_emotional_pattern(emotional_data)
        
        elif update.update_type == "identity_update":
            # Reflect on identity changes from streaming
            identity_data = update.data
            if "streaming" in identity_data.get("source", ""):
                await self._queue_identity_reflection(identity_data)
        
        elif update.update_type == "cross_game_insight":
            # Store insights for pattern reflection
            insight_data = update.data
            self.reflection_context["recent_insights"].append(insight_data)
            
            # Trigger pattern reflection if enough insights
            if len(self.reflection_context["recent_insights"]) >= 5:
                await self._queue_pattern_reflection()
        
        elif update.update_type == "audience_milestone":
            # Reflect on audience milestones
            milestone_data = update.data
            await self._queue_milestone_reflection(milestone_data)
        
        elif update.update_type == "experience_stored":
            # Track experiences for consolidation
            experience_data = update.data
            if "streaming" in experience_data.get("tags", []):
                self.reflection_context["consolidation_candidates"].append(experience_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for reflection opportunities"""
        # Check if user is asking for reflection
        reflection_request = await self._parse_reflection_request(context.user_input)
        
        result = {}
        if reflection_request:
            # Generate requested reflection
            result = await self._generate_requested_reflection(reflection_request, context)
            
            # Send reflection generated update
            await self.send_context_update(
                update_type="reflection_generated",
                data={
                    "reflection_type": reflection_request.get("type"),
                    "reflection_id": result.get("reflection_id"),
                    "confidence": result.get("confidence", 0.7)
                }
            )
        
        # Process pending reflections if any
        if self.reflection_context["pending_reflections"]:
            pending_result = await self._process_pending_reflections()
            result["pending_processed"] = pending_result
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        reflection_influences = await self._process_reflection_influences(messages)
        
        return {
            "reflection_processing_complete": True,
            "reflection_request": reflection_request,
            "result": result,
            "pending_count": len(self.reflection_context["pending_reflections"]),
            "cross_module_influences": len(reflection_influences)
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze reflection needs and patterns"""
        # Analyze reflection coverage
        coverage_analysis = await self._analyze_reflection_coverage()
        
        # Analyze consolidation opportunities
        consolidation_analysis = await self._analyze_consolidation_opportunities()
        
        # Analyze cross-domain connections
        cross_domain_analysis = await self._analyze_cross_domain_connections(context)
        
        # Analyze reflection quality
        quality_analysis = await self._analyze_reflection_quality()
        
        return {
            "coverage_analysis": coverage_analysis,
            "consolidation_analysis": consolidation_analysis,
            "cross_domain_analysis": cross_domain_analysis,
            "quality_analysis": quality_analysis,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize reflection insights for response"""
        messages = await self.get_cross_module_messages()
        
        # Generate reflection synthesis
        reflection_synthesis = {
            "relevant_reflections": await self._get_relevant_reflections(context),
            "emerging_patterns": await self._identify_emerging_patterns(),
            "consolidation_suggestions": await self._suggest_consolidations(),
            "reflection_insights": await self._generate_reflection_insights(context, messages),
            "meta_reflection": await self._generate_meta_reflection(context)
        }
        
        # Check if deep reflection is needed
        if await self._should_generate_deep_reflection(context):
            deep_reflection = await self._generate_deep_reflection_synthesis(context)
            reflection_synthesis["deep_reflection"] = deep_reflection
            
            # Send deep reflection notification
            await self.send_context_update(
                update_type="deep_reflection_available",
                data={
                    "reflection": deep_reflection,
                    "depth": self.reflection_context["reflection_depth"]
                },
                priority=ContextPriority.HIGH
            )
        
        return {
            "reflection_synthesis": reflection_synthesis,
            "synthesis_complete": True
        }
    
    # ========================================================================================
    # REFLECTION-SPECIFIC HELPER METHODS
    # ========================================================================================
    
    async def _identify_reflection_triggers(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify triggers for reflection in the current context"""
        triggers = []
        
        # Check for reflection keywords
        reflection_keywords = ["reflect", "think about", "consider", "realize", "understand", "learn"]
        if any(keyword in context.user_input.lower() for keyword in reflection_keywords):
            triggers.append({
                "type": "explicit_request",
                "source": "user_input"
            })
        
        # Check session context for reflection opportunities
        if context.session_context.get("streaming_session_ended"):
            triggers.append({
                "type": "session_end",
                "source": "session_context"
            })
        
        # Check for significant events
        if context.session_context.get("significant_event_count", 0) >= 3:
            triggers.append({
                "type": "significant_events",
                "source": "event_accumulation"
            })
        
        return triggers
    
    async def _queue_session_reflection(self, streaming_data: Dict[str, Any]):
        """Queue a reflection for streaming session end"""
        session_stats = streaming_data.get("result", {}).get("stats", {})
        
        reflection_item = {
            "type": "session_reflection",
            "priority": 0.8,
            "data": {
                "games_played": session_stats.get("games_played", []),
                "duration": session_stats.get("duration", 0),
                "highlights": session_stats.get("commentary_count", 0),
                "audience_interaction": session_stats.get("questions_answered", 0)
            },
            "queued_at": datetime.now()
        }
        
        self.reflection_context["pending_reflections"].append(reflection_item)
        
        # Limit queue size
        if len(self.reflection_context["pending_reflections"]) > 10:
            self.reflection_context["pending_reflections"] = self.reflection_context["pending_reflections"][-10:]
    
    async def _process_memory_consolidation(self, consolidation_data: Dict[str, Any]):
        """Process memory consolidation for reflection"""
        if consolidation_data.get("domain") == "streaming":
            # Track consolidation for meta-reflection
            self.reflection_context["consolidation_candidates"].extend(
                consolidation_data.get("memories", [])
            )
    
    def _is_streaming_related_goal(self, goal_data: Dict[str, Any]) -> bool:
        """Check if a goal is related to streaming"""
        goal_description = goal_data.get("goal_context", {}).get("description", "").lower()
        streaming_keywords = ["stream", "game", "play", "audience", "commentary"]
        
        return any(keyword in goal_description for keyword in streaming_keywords)
    
    async def _queue_goal_reflection(self, goal_data: Dict[str, Any]):
        """Queue a reflection for completed streaming goal"""
        reflection_item = {
            "type": "goal_reflection",
            "priority": 0.7,
            "data": goal_data.get("goal_context", {}),
            "queued_at": datetime.now()
        }
        
        self.reflection_context["pending_reflections"].append(reflection_item)
    
    async def _track_emotional_pattern(self, emotional_data: Dict[str, Any]):
        """Track emotional patterns for later reflection"""
        # This would track patterns over time
        # For now, just note significant emotional states
        dominant_emotion = emotional_data.get("dominant_emotion")
        if dominant_emotion and dominant_emotion[1] > 0.7:  # High intensity
            pattern = {
                "emotion": dominant_emotion[0],
                "intensity": dominant_emotion[1],
                "timestamp": datetime.now(),
                "context": "streaming"
            }
            # Store for pattern analysis
    
    async def _queue_identity_reflection(self, identity_data: Dict[str, Any]):
        """Queue reflection on identity changes"""
        reflection_item = {
            "type": "identity_reflection",
            "priority": 0.9,
            "data": {
                "traits_modified": identity_data.get("traits_modified", []),
                "preferences_modified": identity_data.get("preferences_modified", []),
                "source": "streaming"
            },
            "queued_at": datetime.now()
        }
        
        self.reflection_context["pending_reflections"].append(reflection_item)
    
    async def _queue_pattern_reflection(self):
        """Queue reflection on emerging patterns"""
        insights = self.reflection_context["recent_insights"]
        
        if insights:
            reflection_item = {
                "type": "pattern_reflection",
                "priority": 0.75,
                "data": {
                    "insight_count": len(insights),
                    "insights": insights[:5]  # Top 5 insights
                },
                "queued_at": datetime.now()
            }
            
            self.reflection_context["pending_reflections"].append(reflection_item)
            
            # Clear processed insights
            self.reflection_context["recent_insights"] = self.reflection_context["recent_insights"][5:]
    
    async def _queue_milestone_reflection(self, milestone_data: Dict[str, Any]):
        """Queue reflection on audience milestone"""
        reflection_item = {
            "type": "milestone_reflection",
            "priority": 0.6,
            "data": milestone_data,
            "queued_at": datetime.now()
        }
        
        self.reflection_context["pending_reflections"].append(reflection_item)
    
    async def _parse_reflection_request(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Parse user input for reflection requests"""
        input_lower = user_input.lower()
        
        if "reflect on" in input_lower:
            # Extract what to reflect on
            parts = input_lower.split("reflect on")
            if len(parts) > 1:
                topic = parts[1].strip()
                return {"type": "topic_reflection", "topic": topic}
        
        elif "what did i learn" in input_lower:
            return {"type": "learning_reflection"}
        
        elif "how was my streaming" in input_lower:
            return {"type": "performance_reflection"}
        
        elif "streaming insights" in input_lower:
            return {"type": "insight_reflection"}
        
        return None
    
    async def _generate_requested_reflection(self, request: Dict[str, Any], context: SharedContext) -> Dict[str, Any]:
        """Generate a requested reflection"""
        reflection_type = request.get("type")
        
        if reflection_type == "topic_reflection":
            topic = request.get("topic", "streaming")
            return await self.original_engine.generate_deep_reflection(
                game_name="General",
                aspect=topic
            )
        
        elif reflection_type == "learning_reflection":
            # Generate learning summary
            if hasattr(self.original_engine, "brain") and hasattr(self.original_engine.brain, "streaming_core"):
                summary = await self.original_engine.brain.streaming_core.summarize_session_learnings()
                return {
                    "reflection": summary.get("summary", "No recent learning to reflect on."),
                    "confidence": 0.8
                }
        
        elif reflection_type == "performance_reflection":
            # Reflect on streaming performance
            recent_games = context.session_context.get("recent_games", [])
            if recent_games:
                return await self.original_engine.generate_comparative_reflection(recent_games)
        
        elif reflection_type == "insight_reflection":
            # Reflect on insights
            insights = self.reflection_context["recent_insights"]
            if insights:
                return {
                    "reflection": f"I've discovered {len(insights)} insights across games, showing patterns in mechanics and player experiences.",
                    "insights": insights[:3],
                    "confidence": 0.7
                }
        
        return {
            "reflection": "I don't have enough context for that reflection.",
            "confidence": 0.3
        }
    
    async def _process_pending_reflections(self) -> Dict[str, Any]:
        """Process pending reflections"""
        if not self.reflection_context["pending_reflections"]:
            return {"processed": 0}
        
        # Sort by priority
        self.reflection_context["pending_reflections"].sort(
            key=lambda x: x["priority"], 
            reverse=True
        )
        
        # Process highest priority
        reflection_item = self.reflection_context["pending_reflections"].pop(0)
        
        result = await self._process_single_reflection(reflection_item)
        
        return {
            "processed": 1,
            "type": reflection_item["type"],
            "result": result,
            "remaining": len(self.reflection_context["pending_reflections"])
        }
    
    async def _process_single_reflection(self, reflection_item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single reflection item"""
        reflection_type = reflection_item["type"]
        data = reflection_item["data"]
        
        if reflection_type == "session_reflection":
            games = data.get("games_played", [])
            if games:
                game_name = games[0] if len(games) == 1 else "Multiple Games"
                return await self.original_engine.process_streaming_session(
                    game_name=game_name,
                    session_data=data
                )
        
        elif reflection_type == "pattern_reflection":
            # Generate pattern reflection
            insights = data.get("insights", [])
            return {
                "reflection": f"Patterns emerging from {len(insights)} insights across games.",
                "pattern_count": len(insights)
            }
        
        # Default
        return {"reflection": f"Processed {reflection_type} reflection"}
    
    async def _process_reflection_influences(self, messages: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Process reflection influences from other modules"""
        influences = []
        
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                if msg["type"] in ["memory_created", "experience_stored", "abstraction_created"]:
                    influences.append({
                        "module": module_name,
                        "type": msg["type"],
                        "influence": "consolidation_candidate"
                    })
        
        return influences
    
    async def _analyze_reflection_coverage(self) -> Dict[str, Any]:
        """Analyze coverage of reflections"""
        # Check reflection history
        history = self.original_engine.reflection_history
        
        coverage = {
            "total_reflections": len(history),
            "recent_reflections": len([r for r in history if datetime.fromisoformat(r["timestamp"]) > datetime.now() - timedelta(days=7)]),
            "coverage_areas": {}
        }
        
        # Analyze coverage by game
        games_reflected = {}
        for reflection in history:
            game = reflection.get("game_name", "Unknown")
            games_reflected[game] = games_reflected.get(game, 0) + 1
        
        coverage["games_reflected"] = games_reflected
        
        return coverage
    
    async def _analyze_consolidation_opportunities(self) -> Dict[str, Any]:
        """Analyze opportunities for experience consolidation"""
        candidates = self.reflection_context["consolidation_candidates"]
        
        analysis = {
            "candidate_count": len(candidates),
            "ready_for_consolidation": len(candidates) >= 3,
            "consolidation_themes": []
        }
        
        # Group by themes
        theme_groups = {}
        for candidate in candidates:
            tags = candidate.get("tags", [])
            for tag in tags:
                if tag not in ["streaming", "experience"]:
                    if tag not in theme_groups:
                        theme_groups[tag] = 0
                    theme_groups[tag] += 1
        
        # Find themes with enough experiences
        for theme, count in theme_groups.items():
            if count >= 3:
                analysis["consolidation_themes"].append({
                    "theme": theme,
                    "experience_count": count
                })
        
        return analysis
    
    async def _analyze_cross_domain_connections(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze connections between streaming and other domains"""
        cross_domain_score = self.reflection_context["cross_domain_awareness"]
        
        analysis = {
            "awareness_level": cross_domain_score,
            "identified_connections": [],
            "potential_connections": []
        }
        
        # Check for existing connections
        messages = await self.get_cross_module_messages()
        
        for module_name, module_messages in messages.items():
            if module_name not in ["streaming_core", "streaming_hormone_system"]:
                for msg in module_messages:
                    if "streaming" in str(msg.get("data", {})):
                        analysis["identified_connections"].append({
                            "domain": module_name,
                            "connection_type": msg["type"]
                        })
        
        # Suggest potential connections
        if "goal_manager" in messages:
            analysis["potential_connections"].append({
                "domain": "goals",
                "suggestion": "Connect streaming achievements to personal goals"
            })
        
        return analysis
    
    async def _analyze_reflection_quality(self) -> Dict[str, Any]:
        """Analyze quality of reflections"""
        return {
            "depth_score": self.reflection_context["reflection_depth"],
            "cross_domain_integration": self.reflection_context["cross_domain_awareness"],
            "pattern_recognition": len(self.reflection_context["recent_insights"]) > 0,
            "consolidation_active": len(self.reflection_context["consolidation_candidates"]) >= 3
        }
    
    async def _get_relevant_reflections(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Get reflections relevant to current context"""
        # This would retrieve relevant past reflections
        # For now, return recent reflection summaries
        history = self.original_engine.reflection_history[-3:]  # Last 3 reflections
        
        relevant = []
        for reflection in history:
            relevant.append({
                "game": reflection.get("game_name"),
                "timestamp": reflection.get("timestamp"),
                "type": "streaming_reflection"
            })
        
        return relevant
    
    async def _identify_emerging_patterns(self) -> List[Dict[str, Any]]:
        """Identify emerging patterns from recent experiences"""
        patterns = []
        
        # Analyze recent insights
        insights = self.reflection_context["recent_insights"]
        
        if len(insights) >= 3:
            # Find common mechanics
            mechanics = {}
            for insight in insights:
                mechanic = insight.get("mechanic", "")
                if mechanic:
                    mechanics[mechanic] = mechanics.get(mechanic, 0) + 1
            
            for mechanic, count in mechanics.items():
                if count >= 2:
                    patterns.append({
                        "type": "recurring_mechanic",
                        "pattern": mechanic,
                        "frequency": count
                    })
        
        return patterns
    
    async def _suggest_consolidations(self) -> List[Dict[str, Any]]:
        """Suggest experience consolidations"""
        suggestions = []
        
        analysis = await self._analyze_consolidation_opportunities()
        
        for theme_data in analysis["consolidation_themes"]:
            suggestions.append({
                "action": "consolidate",
                "theme": theme_data["theme"],
                "experience_count": theme_data["experience_count"],
                "benefit": "Create higher-level understanding"
            })
        
        return suggestions
    
    async def _generate_reflection_insights(self, context: SharedContext, messages: Dict) -> List[str]:
        """Generate insights from reflection analysis"""
        insights = []
        
        # Insight about reflection coverage
        coverage = await self._analyze_reflection_coverage()
        if coverage["total_reflections"] > 10:
            insights.append(f"Built {coverage['total_reflections']} reflections across streaming experiences")
        
        # Insight about patterns
        patterns = await self._identify_emerging_patterns()
        if patterns:
            insights.append(f"Noticed {len(patterns)} recurring patterns in gameplay")
        
        # Insight about consolidation
        consolidation = await self._analyze_consolidation_opportunities()
        if consolidation["ready_for_consolidation"]:
            insights.append("Ready to consolidate experiences into deeper understanding")
        
        return insights
    
    async def _generate_meta_reflection(self, context: SharedContext) -> str:
        """Generate meta-reflection about the reflection process itself"""
        pending_count = len(self.reflection_context["pending_reflections"])
        depth = self.reflection_context["reflection_depth"]
        
        if pending_count > 5:
            return "I have many experiences to reflect on, each offering unique insights."
        elif depth > 0.8:
            return "My reflections are becoming deeper, connecting streaming to broader patterns."
        else:
            return ""
    
    async def _should_generate_deep_reflection(self, context: SharedContext) -> bool:
        """Determine if deep reflection should be generated"""
        # Check for deep reflection triggers
        if "deep" in context.user_input.lower() and "reflect" in context.user_input.lower():
            return True
        
        # Check if enough time has passed
        if hasattr(self.original_engine, "last_deep_reflection_time"):
            time_since = datetime.now() - self.original_engine.last_deep_reflection_time
            if time_since > timedelta(minutes=30):
                return True
        
        # Check if significant experiences accumulated
        if len(self.reflection_context["consolidation_candidates"]) >= 5:
            return True
        
        return False
    
    async def _generate_deep_reflection_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Generate deep reflection synthesis"""
        # Use enhanced deep reflection
        if hasattr(self.original_engine, "generate_deep_reflection"):
            # Get most relevant game
            recent_games = context.session_context.get("recent_games", ["General"])
            game_name = recent_games[0] if recent_games else "General"
            
            return await self.original_engine.generate_deep_reflection(
                game_name=game_name,
                aspect="overall streaming experience"
            )
        
        return {
            "reflection": "Deep patterns emerge from streaming experiences, revealing growth and understanding.",
            "depth": self.reflection_context["reflection_depth"]
        }
    
    # Delegate all other methods to the original engine
    def __getattr__(self, name):
        """Delegate any missing methods to the original engine"""
        return getattr(self.original_engine, name)
