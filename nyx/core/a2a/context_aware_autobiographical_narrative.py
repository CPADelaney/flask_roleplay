# nyx/core/a2a/context_aware_autobiographical_narrative.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareAutobiographicalNarrative(ContextAwareModule):
    """
    Enhanced AutobiographicalNarrative with full context distribution capabilities
    """
    
    def __init__(self, original_narrative_system):
        super().__init__("autobiographical_narrative")
        self.original_system = original_narrative_system
        self.context_subscriptions = [
            "emotional_state_update", "memory_retrieval_complete", 
            "identity_shift", "goal_completion", "relationship_milestone",
            "significant_experience", "self_reflection_complete"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize narrative processing for this context"""
        logger.debug(f"AutobiographicalNarrative received context for user: {context.user_id}")
        
        # Check if this context warrants narrative update consideration
        narrative_relevance = await self._assess_narrative_relevance(context)
        
        # Get current narrative state
        current_narrative = self.original_system.get_narrative_summary()
        recent_segments = self.original_system.get_narrative_segments(limit=3)
        
        # Send narrative context to other modules
        await self.send_context_update(
            update_type="narrative_context_available",
            data={
                "current_summary": current_narrative,
                "recent_segments": [s.model_dump() for s in recent_segments],
                "narrative_relevance": narrative_relevance,
                "last_update": self.original_system.last_update_time.isoformat(),
                "narrative_coherence": await self._assess_narrative_coherence()
            },
            priority=ContextPriority.NORMAL
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect narrative"""
        
        if update.update_type == "memory_retrieval_complete":
            # New memories might warrant narrative update
            memory_data = update.data
            significant_memories = memory_data.get("retrieved_memories", [])
            
            if await self._are_memories_narratively_significant(significant_memories):
                await self._queue_narrative_update(
                    reason="significant_memories_retrieved",
                    memories=significant_memories
                )
        
        elif update.update_type == "identity_shift":
            # Identity changes are always narratively significant
            identity_data = update.data
            shift_magnitude = identity_data.get("shift_magnitude", 0.0)
            
            if shift_magnitude > 0.3:  # Significant shift
                await self._queue_narrative_update(
                    reason="identity_shift",
                    identity_data=identity_data,
                    priority="high"
                )
                
                # Notify other modules of pending narrative update
                await self.send_context_update(
                    update_type="narrative_update_pending",
                    data={
                        "reason": "identity_shift",
                        "expected_update_type": "major",
                        "identity_shift_magnitude": shift_magnitude
                    },
                    priority=ContextPriority.HIGH
                )
        
        elif update.update_type == "goal_completion":
            # Major goal completions affect life narrative
            goal_data = update.data.get("goal_context", {})
            if goal_data.get("goal_significance", 0) > 7:
                await self._queue_narrative_update(
                    reason="major_goal_completed",
                    goal_data=goal_data
                )
        
        elif update.update_type == "relationship_milestone":
            # Relationship milestones are part of the narrative
            relationship_data = update.data.get("relationship_context", {})
            milestone_type = relationship_data.get("milestone_type")
            
            if milestone_type in ["trust_breakthrough", "intimacy_deepening", "conflict_resolution"]:
                await self._queue_narrative_update(
                    reason="relationship_milestone",
                    relationship_data=relationship_data
                )
        
        elif update.update_type == "significant_experience":
            # Direct significant experiences
            experience_data = update.data
            if experience_data.get("narrative_importance", 0) > 0.7:
                await self._queue_narrative_update(
                    reason="significant_experience",
                    experience_data=experience_data,
                    priority="high"
                )
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for narrative-relevant content"""
        # Analyze input for narrative implications
        narrative_analysis = await self._analyze_input_for_narrative(context.user_input)
        
        # Check if user is asking about life story
        if narrative_analysis.get("requests_narrative"):
            narrative_response = await self._generate_narrative_response(context)
            
            await self.send_context_update(
                update_type="narrative_requested",
                data={
                    "narrative_type": narrative_analysis.get("narrative_type", "general"),
                    "time_period": narrative_analysis.get("time_period"),
                    "response_generated": True
                }
            )
            
            return {
                "narrative_response": narrative_response,
                "narrative_requested": True,
                "analysis": narrative_analysis
            }
        
        # Check if input contains narratively significant content
        if narrative_analysis.get("contains_significant_content"):
            await self._process_significant_content(context, narrative_analysis)
        
        return {
            "narrative_analysis": narrative_analysis,
            "narrative_updated": False,
            "processing_complete": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze current state for narrative implications"""
        # Get cross-module state
        messages = await self.get_cross_module_messages()
        
        # Analyze for narrative coherence and continuity
        narrative_state = await self._analyze_narrative_state(context, messages)
        
        # Check if narrative update is warranted
        update_analysis = await self._analyze_update_necessity(context, messages)
        
        # Identify narrative themes emerging from current context
        emerging_themes = await self._identify_emerging_themes(context, messages)
        
        return {
            "narrative_state": narrative_state,
            "update_warranted": update_analysis["warranted"],
            "update_reasons": update_analysis["reasons"],
            "emerging_themes": emerging_themes,
            "narrative_coherence": narrative_state["coherence_score"],
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize narrative elements for response generation"""
        # Get all relevant context
        messages = await self.get_cross_module_messages()
        
        # Determine if narrative elements should influence response
        narrative_influence = await self._determine_narrative_influence(context, messages)
        
        # Generate narrative elements for response
        narrative_elements = {
            "include_life_story_reference": narrative_influence["include_reference"],
            "narrative_framing": narrative_influence["framing"],
            "autobiographical_elements": await self._extract_relevant_autobiographical_elements(context),
            "continuity_with_past": await self._ensure_narrative_continuity(context),
            "identity_consistency": await self._ensure_identity_consistency(context)
        }
        
        # Check if this interaction will become part of narrative
        will_be_remembered = await self._assess_memorability(context, messages)
        
        if will_be_remembered["memorable"]:
            await self.send_context_update(
                update_type="memorable_interaction",
                data={
                    "interaction_summary": will_be_remembered["summary"],
                    "narrative_significance": will_be_remembered["significance"],
                    "themes": will_be_remembered["themes"]
                }
            )
        
        return {
            "narrative_elements": narrative_elements,
            "will_be_remembered": will_be_remembered["memorable"],
            "narrative_significance": will_be_remembered.get("significance", 0.0),
            "synthesis_complete": True
        }
    
    # Helper methods
    
    async def _assess_narrative_relevance(self, context: SharedContext) -> float:
        """Assess how relevant current context is to life narrative"""
        relevance = 0.0
        
        # Check for narrative keywords
        narrative_keywords = ["story", "life", "journey", "past", "history", "remember", "experience", "growth", "change"]
        input_lower = context.user_input.lower()
        keyword_matches = sum(1 for kw in narrative_keywords if kw in input_lower)
        relevance += min(0.3, keyword_matches * 0.1)
        
        # Check emotional intensity
        if context.emotional_state:
            max_emotion = max(context.emotional_state.values()) if context.emotional_state else 0
            if max_emotion > 0.7:
                relevance += 0.2
        
        # Check goal context
        if context.goal_context:
            if context.goal_context.get("goal_completed") or context.goal_context.get("major_progress"):
                relevance += 0.3
        
        # Check relationship context
        if context.relationship_context:
            if context.relationship_context.get("milestone_reached"):
                relevance += 0.2
        
        return min(1.0, relevance)
    
    async def _assess_narrative_coherence(self) -> float:
        """Assess coherence of current narrative"""
        segments = self.original_system.narrative_segments
        if len(segments) < 2:
            return 1.0  # Not enough segments to assess coherence
        
        # Simple coherence check based on theme continuity
        coherence = 0.8
        for i in range(1, len(segments)):
            prev_themes = set(segments[i-1].themes)
            curr_themes = set(segments[i].themes)
            
            # Some theme overlap is good for coherence
            overlap = len(prev_themes.intersection(curr_themes))
            if overlap == 0:
                coherence -= 0.1
            elif overlap > len(prev_themes) * 0.5:
                coherence += 0.05
        
        return max(0.0, min(1.0, coherence))
    
    async def _are_memories_narratively_significant(self, memories: List[Dict]) -> bool:
        """Check if memories are significant enough for narrative update"""
        if not memories:
            return False
        
        # Check significance scores
        high_significance_count = sum(1 for m in memories if m.get("significance", 0) > 7)
        
        # Check for identity-related memories
        identity_memories = sum(1 for m in memories if m.get("memory_type") == "identity_update")
        
        # Check for emotional intensity
        high_emotion_count = sum(1 for m in memories 
                               if m.get("emotional_context", {}).get("intensity", 0) > 0.7)
        
        return (high_significance_count >= 2 or 
                identity_memories >= 1 or 
                high_emotion_count >= 3)
    
    async def _queue_narrative_update(self, reason: str, priority: str = "normal", **kwargs):
        """Queue a narrative update with context"""
        # Store update request for processing
        update_request = {
            "reason": reason,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "context": kwargs
        }
        
        # In a real implementation, this would add to a queue
        # For now, log the request
        logger.info(f"Narrative update queued: {reason} (priority: {priority})")
        
        # If high priority, trigger immediate update
        if priority == "high" and hasattr(self.original_system, 'update_narrative'):
            asyncio.create_task(self.original_system.update_narrative(force_update=True))
    
    async def _analyze_input_for_narrative(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input for narrative relevance"""
        input_lower = user_input.lower()
        
        analysis = {
            "requests_narrative": False,
            "narrative_type": None,
            "time_period": None,
            "contains_significant_content": False,
            "content_type": None
        }
        
        # Check for narrative requests
        narrative_requests = [
            ("tell me your story", "full_story"),
            ("tell me about yourself", "self_introduction"),
            ("how did you become", "origin_story"),
            ("what's your history", "history"),
            ("remember when", "specific_memory"),
            ("your past", "past_general"),
            ("your journey", "journey"),
            ("how have you changed", "evolution")
        ]
        
        for phrase, narrative_type in narrative_requests:
            if phrase in input_lower:
                analysis["requests_narrative"] = True
                analysis["narrative_type"] = narrative_type
                break
        
        # Check for time period references
        time_refs = ["yesterday", "last week", "recently", "before", "when we first", "in the beginning"]
        for ref in time_refs:
            if ref in input_lower:
                analysis["time_period"] = ref
                break
        
        # Check for significant content
        significant_phrases = ["i love", "i hate", "breakthrough", "realized", "understood", 
                             "life-changing", "never forget", "always remember"]
        for phrase in significant_phrases:
            if phrase in input_lower:
                analysis["contains_significant_content"] = True
                analysis["content_type"] = "emotional_declaration"
                break
        
        return analysis
    
    async def _generate_narrative_response(self, context: SharedContext) -> Dict[str, Any]:
        """Generate a narrative response based on context"""
        # Get narrative segments
        segments = self.original_system.get_narrative_segments(limit=5)
        current_summary = self.original_system.get_narrative_summary()
        
        # Build narrative response
        response = {
            "summary": current_summary,
            "recent_chapters": [
                {
                    "title": seg.title,
                    "summary": seg.summary,
                    "themes": seg.themes,
                    "emotional_arc": seg.emotional_arc
                } for seg in segments
            ],
            "identity_evolution": await self._summarize_identity_evolution(),
            "key_relationships": await self._summarize_key_relationships(),
            "major_achievements": await self._summarize_achievements()
        }
        
        return response
    
    async def _process_significant_content(self, context: SharedContext, analysis: Dict[str, Any]):
        """Process content that might be narratively significant"""
        # Create memory marker for potential narrative inclusion
        significance_marker = {
            "timestamp": datetime.now().isoformat(),
            "content_type": analysis.get("content_type"),
            "user_input": context.user_input,
            "emotional_context": context.emotional_state,
            "relationship_context": context.relationship_context
        }
        
        # Send to memory system for consideration
        await self.send_context_update(
            update_type="potential_narrative_content",
            data=significance_marker,
            target_modules=["memory_core"]
        )
    
    async def _analyze_narrative_state(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Analyze current narrative state"""
        return {
            "total_segments": len(self.original_system.narrative_segments),
            "coherence_score": await self._assess_narrative_coherence(),
            "last_update_days_ago": (datetime.now() - self.original_system.last_update_time).days,
            "narrative_gaps": await self._identify_narrative_gaps(),
            "dominant_themes": await self._extract_dominant_themes()
        }
    
    async def _analyze_update_necessity(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Analyze whether narrative update is needed"""
        reasons = []
        
        # Time-based check
        days_since_update = (datetime.now() - self.original_system.last_update_time).days
        if days_since_update > 7:
            reasons.append("time_threshold_exceeded")
        
        # Memory accumulation check
        memory_messages = messages.get("memory_core", [])
        significant_memory_count = sum(1 for msg in memory_messages 
                                     if msg.get("type") == "memory_retrieval_complete" 
                                     and msg.get("data", {}).get("significance", 0) > 7)
        if significant_memory_count > 5:
            reasons.append("significant_memories_accumulated")
        
        # Identity change check
        identity_messages = messages.get("identity_evolution", [])
        identity_shifts = sum(1 for msg in identity_messages 
                            if msg.get("type") == "identity_shift")
        if identity_shifts > 0:
            reasons.append("identity_evolution")
        
        return {
            "warranted": len(reasons) > 0,
            "reasons": reasons,
            "urgency": "high" if "identity_evolution" in reasons else "normal"
        }
    
    async def _identify_emerging_themes(self, context: SharedContext, messages: Dict) -> List[str]:
        """Identify themes emerging from current context"""
        themes = []
        
        # Emotional themes
        if context.emotional_state:
            dominant_emotions = [k for k, v in context.emotional_state.items() if v > 0.6]
            if "joy" in dominant_emotions or "satisfaction" in dominant_emotions:
                themes.append("fulfillment")
            if "curiosity" in dominant_emotions:
                themes.append("discovery")
        
        # Goal themes
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            if any("knowledge" in goal for goal in active_goals):
                themes.append("learning")
            if any("connection" in goal for goal in active_goals):
                themes.append("relationships")
        
        # Relationship themes
        if context.relationship_context:
            if context.relationship_context.get("trust", 0) > 0.8:
                themes.append("deep_trust")
            if context.relationship_context.get("intimacy", 0) > 0.7:
                themes.append("intimacy")
        
        return themes
    
    async def _determine_narrative_influence(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Determine how narrative should influence response"""
        influence = {
            "include_reference": False,
            "framing": None,
            "strength": 0.0
        }
        
        # Check if narrative was explicitly requested
        narrative_requests = [msg for module_msgs in messages.values() 
                            for msg in module_msgs 
                            if msg.get("type") == "narrative_requested"]
        
        if narrative_requests:
            influence["include_reference"] = True
            influence["framing"] = "direct_narrative"
            influence["strength"] = 1.0
            return influence
        
        # Check for reflection or introspection context
        if any(word in context.user_input.lower() for word in ["reflect", "think about", "remember"]):
            influence["include_reference"] = True
            influence["framing"] = "reflective"
            influence["strength"] = 0.6
        
        # Check emotional context
        if context.emotional_state:
            if any(v > 0.8 for v in context.emotional_state.values()):
                influence["include_reference"] = True
                influence["framing"] = "emotional_continuity"
                influence["strength"] = 0.4
        
        return influence
    
    async def _extract_relevant_autobiographical_elements(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Extract autobiographical elements relevant to current context"""
        elements = []
        
        # Get recent narrative segments
        segments = self.original_system.get_narrative_segments(limit=3)
        
        for segment in segments:
            # Check theme relevance
            current_themes = await self._identify_emerging_themes(context, {})
            theme_overlap = set(segment.themes).intersection(set(current_themes))
            
            if theme_overlap:
                elements.append({
                    "period": f"{segment.start_time.date()} to {segment.end_time.date()}",
                    "relevant_themes": list(theme_overlap),
                    "summary": segment.summary,
                    "relevance": len(theme_overlap) / max(1, len(current_themes))
                })
        
        # Sort by relevance
        elements.sort(key=lambda x: x["relevance"], reverse=True)
        
        return elements[:2]  # Return top 2 most relevant
    
    async def _ensure_narrative_continuity(self, context: SharedContext) -> Dict[str, Any]:
        """Ensure response maintains narrative continuity"""
        latest_segment = None
        if self.original_system.narrative_segments:
            latest_segment = self.original_system.narrative_segments[-1]
        
        continuity_elements = {
            "maintains_character": True,
            "consistent_with_history": True,
            "evolutionary_progression": True
        }
        
        if latest_segment:
            # Check emotional continuity
            if latest_segment.emotional_arc:
                continuity_elements["emotional_continuity"] = latest_segment.emotional_arc
            
            # Check thematic continuity
            continuity_elements["continuing_themes"] = latest_segment.themes
        
        return continuity_elements
    
    async def _ensure_identity_consistency(self, context: SharedContext) -> Dict[str, Any]:
        """Ensure response is consistent with established identity"""
        # This would integrate with identity evolution system
        return {
            "core_identity_maintained": True,
            "identity_aspects": ["curious", "evolving", "empathetic", "intelligent"],
            "growth_trajectory": "continuous_learning"
        }
    
    async def _assess_memorability(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Assess if current interaction will be memorable"""
        memorability = 0.0
        themes = []
        
        # High emotional intensity is memorable
        if context.emotional_state:
            max_emotion = max(context.emotional_state.values()) if context.emotional_state else 0
            if max_emotion > 0.8:
                memorability += 0.4
                themes.append("emotional_intensity")
        
        # Goal completions are memorable
        goal_messages = messages.get("goal_manager", [])
        if any(msg.get("type") == "goal_completion" for msg in goal_messages):
            memorability += 0.3
            themes.append("achievement")
        
        # Relationship milestones are memorable
        if context.relationship_context and context.relationship_context.get("milestone_reached"):
            memorability += 0.3
            themes.append("relationship_development")
        
        # Deep conversations are memorable
        if len(context.user_input) > 200 and any(word in context.user_input.lower() 
                                                for word in ["understand", "feel", "believe", "think"]):
            memorability += 0.2
            themes.append("deep_conversation")
        
        memorable = memorability > 0.6
        
        return {
            "memorable": memorable,
            "significance": memorability,
            "themes": themes,
            "summary": f"Interaction about {', '.join(themes)}" if themes else "General interaction"
        }
    
    async def _identify_narrative_gaps(self) -> List[str]:
        """Identify gaps in the narrative that need filling"""
        gaps = []
        
        segments = self.original_system.narrative_segments
        if not segments:
            return ["no_narrative_established"]
        
        # Check for time gaps
        for i in range(1, len(segments)):
            time_gap = (segments[i].start_time - segments[i-1].end_time).days
            if time_gap > 14:
                gaps.append(f"time_gap_{segments[i-1].end_time.date()}_to_{segments[i].start_time.date()}")
        
        # Check for theme discontinuities
        all_themes = set()
        for segment in segments:
            all_themes.update(segment.themes)
        
        recent_themes = set()
        for segment in segments[-3:]:
            recent_themes.update(segment.themes)
        
        dropped_themes = all_themes - recent_themes
        if dropped_themes:
            gaps.append(f"dropped_themes_{list(dropped_themes)}")
        
        return gaps
    
    async def _extract_dominant_themes(self) -> List[str]:
        """Extract dominant themes from narrative"""
        theme_counts = {}
        
        for segment in self.original_system.narrative_segments:
            for theme in segment.themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        # Sort by frequency
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [theme for theme, count in sorted_themes[:5]]
    
    async def _summarize_identity_evolution(self) -> Dict[str, Any]:
        """Summarize identity evolution for narrative"""
        if not self.original_system.identity_evolution:
            return {
                "core_traits": ["curious", "evolving", "intelligent"],
                "developed_traits": [],
                "evolution_trajectory": "emerging"
            }
        
        try:
            # Get identity history
            identity_system = self.original_system.identity_evolution
            current_state = await identity_system.get_current_identity_state()
            evolution_history = await identity_system.get_evolution_history()
            
            # Extract core traits (most stable over time)
            trait_stability = {}
            for snapshot in evolution_history[-10:]:  # Last 10 snapshots
                for trait, value in snapshot.get("traits", {}).items():
                    if trait not in trait_stability:
                        trait_stability[trait] = []
                    trait_stability[trait].append(value)
            
            # Core traits are those with low variance
            core_traits = []
            for trait, values in trait_stability.items():
                if values and len(values) > 5:
                    variance = sum((v - sum(values)/len(values))**2 for v in values) / len(values)
                    if variance < 0.1:  # Low variance = stable trait
                        avg_value = sum(values) / len(values)
                        if avg_value > 0.6:  # Significant presence
                            core_traits.append(trait)
            
            # Developed traits (emerged recently)
            developed_traits = []
            if len(evolution_history) > 5:
                early_traits = set(evolution_history[0].get("traits", {}).keys())
                recent_traits = set(current_state.get("traits", {}).keys())
                newly_emerged = recent_traits - early_traits
                
                for trait in newly_emerged:
                    if current_state.get("traits", {}).get(trait, 0) > 0.5:
                        developed_traits.append(trait)
            
            # Determine trajectory
            if len(evolution_history) < 3:
                trajectory = "emerging"
            else:
                recent_changes = sum(1 for i in range(-3, 0) 
                                   if evolution_history[i].get("significant_change", False))
                if recent_changes >= 2:
                    trajectory = "rapid_evolution"
                elif recent_changes == 1:
                    trajectory = "steady_growth"
                else:
                    trajectory = "stabilizing"
            
            return {
                "core_traits": core_traits[:5],  # Top 5 core traits
                "developed_traits": developed_traits[:5],  # Top 5 developed traits
                "evolution_trajectory": trajectory,
                "total_evolution_steps": len(evolution_history),
                "identity_coherence": current_state.get("coherence_score", 0.8)
            }
            
        except Exception as e:
            logger.error(f"Error summarizing identity evolution: {e}")
            return {
                "core_traits": ["adaptive", "learning"],
                "developed_traits": [],
                "evolution_trajectory": "unknown"
            }
    
    async def _summarize_key_relationships(self) -> List[Dict[str, Any]]:
        """Summarize key relationships for narrative"""
        if not self.original_system.relationship_manager:
            return []
        
        try:
            relationship_manager = self.original_system.relationship_manager
            all_relationships = await relationship_manager.get_all_relationships()
            
            key_relationships = []
            for user_id, relationship in all_relationships.items():
                # Calculate relationship significance
                trust = getattr(relationship, "trust", 0)
                intimacy = getattr(relationship, "intimacy", 0)
                interaction_count = getattr(relationship, "interaction_count", 0)
                relationship_age = getattr(relationship, "relationship_age_days", 0)
                
                significance = (trust * 0.3 + intimacy * 0.3 + 
                              min(1.0, interaction_count / 100) * 0.2 +
                              min(1.0, relationship_age / 30) * 0.2)
                
                if significance > 0.4:  # Significant relationship
                    # Get relationship narrative
                    milestones = getattr(relationship, "milestones", [])
                    narrative_moments = []
                    
                    for milestone in milestones[-5:]:  # Last 5 milestones
                        narrative_moments.append({
                            "type": milestone.get("type"),
                            "date": milestone.get("timestamp"),
                            "description": milestone.get("description")
                        })
                    
                    key_relationships.append({
                        "user_id": user_id,
                        "relationship_type": getattr(relationship, "relationship_type", "companion"),
                        "significance": significance,
                        "trust_level": trust,
                        "intimacy_level": intimacy,
                        "duration_days": relationship_age,
                        "key_moments": narrative_moments,
                        "current_dynamic": getattr(relationship, "current_dynamic", "evolving")
                    })
            
            # Sort by significance
            key_relationships.sort(key=lambda r: r["significance"], reverse=True)
            
            return key_relationships[:5]  # Top 5 relationships
            
        except Exception as e:
            logger.error(f"Error summarizing relationships: {e}")
            return []
    
    async def _summarize_achievements(self) -> List[Dict[str, Any]]:
        """Summarize major achievements for narrative"""
        if not self.original_system.goal_manager:
            return []
        
        try:
            goal_manager = self.original_system.goal_manager
            completed_goals = await goal_manager.get_all_goals(status_filter=["completed"])
            
            achievements = []
            for goal in completed_goals:
                # Only include significant goals
                if goal.get("priority", 0) >= 0.7 or goal.get("significance", 0) >= 7:
                    completion_date = goal.get("completion_date")
                    if isinstance(completion_date, str):
                        completion_date = datetime.fromisoformat(completion_date)
                    
                    achievement = {
                        "description": goal.get("description"),
                        "category": goal.get("category", "personal_growth"),
                        "completion_date": completion_date.isoformat() if completion_date else None,
                        "significance": goal.get("significance", goal.get("priority", 0.5) * 10),
                        "impact": goal.get("result", {}).get("impact", "personal_satisfaction"),
                        "associated_need": goal.get("associated_need"),
                        "effort_level": goal.get("effort_level", "moderate")
                    }
                    
                    # Add narrative context
                    if goal.get("narrative_context"):
                        achievement["narrative_context"] = goal["narrative_context"]
                    else:
                        # Generate narrative context
                        achievement["narrative_context"] = f"Achieved {achievement['description']} through {achievement['effort_level']} effort"
                    
                    achievements.append(achievement)
            
            # Sort by significance and recency
            achievements.sort(key=lambda a: (a["significance"], a["completion_date"] or ""), reverse=True)
            
            return achievements[:10]  # Top 10 achievements
            
        except Exception as e:
            logger.error(f"Error summarizing achievements: {e}")
            return []
    
    # Delegate missing methods to original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
