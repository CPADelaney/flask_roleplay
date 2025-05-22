# nyx/core/a2a/context_aware_experience_interface.py

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareExperienceInterface(ContextAwareModule):
    """
    Advanced Experience Interface with full context distribution capabilities
    """
    
    def __init__(self, original_experience_interface):
        super().__init__("experience_interface")
        self.original_interface = original_experience_interface
        self.context_subscriptions = [
            "experience_request", "emotional_state_update", "goal_context_update",
            "identity_evolution_update", "memory_retrieval_request", "relationship_context_update",
            "temporal_context_update", "cross_user_experience_request", "narrative_request"
        ]
        
        # Track experience sharing state
        self.current_experience_context = {}
        self.experience_effectiveness = {}
        self.user_engagement_tracking = {}
    
    async def on_context_received(self, context: SharedContext):
        """Initialize experience processing for this context"""
        logger.debug(f"ExperienceInterface received context for user: {context.user_id}")
        
        # Analyze context for experience relevance
        experience_relevance = await self._analyze_experience_relevance(context)
        
        # Check for existing relevant experiences
        relevant_experiences = await self._find_contextually_relevant_experiences(context)
        
        # Determine experience sharing strategy
        sharing_strategy = await self._determine_sharing_strategy(context, experience_relevance)
        
        # Send initial experience context to other modules
        await self.send_context_update(
            update_type="experience_context_available",
            data={
                "has_relevant_experiences": len(relevant_experiences) > 0,
                "experience_relevance": experience_relevance,
                "sharing_strategy": sharing_strategy,
                "top_experiences": relevant_experiences[:3],  # Top 3
                "user_preference": await self._get_user_experience_preference(context.user_id)
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect experience sharing"""
        
        if update.update_type == "experience_request":
            # Direct request for experience sharing
            request_data = update.data
            query = request_data.get("query", "")
            request_type = request_data.get("type", "standard")
            
            # Process experience request with context
            await self._process_experience_request(query, request_type, request_data)
        
        elif update.update_type == "emotional_state_update":
            # Emotional state affects experience selection and formatting
            emotional_data = update.data
            await self._update_emotional_context_for_experiences(emotional_data)
        
        elif update.update_type == "goal_context_update":
            # Goal context influences experience relevance
            goal_data = update.data
            await self._incorporate_goal_context(goal_data)
        
        elif update.update_type == "identity_evolution_update":
            # Identity changes affect experience preferences
            identity_data = update.data
            await self._update_identity_based_preferences(identity_data)
        
        elif update.update_type == "memory_retrieval_request":
            # Request for specific memory retrieval
            retrieval_data = update.data
            await self._process_memory_retrieval_request(retrieval_data)
        
        elif update.update_type == "relationship_context_update":
            # Relationship context affects experience sharing depth
            relationship_data = update.data
            await self._adjust_sharing_based_on_relationship(relationship_data)
        
        elif update.update_type == "temporal_context_update":
            # Temporal context influences experience selection
            temporal_data = update.data
            await self._incorporate_temporal_context(temporal_data)
        
        elif update.update_type == "cross_user_experience_request":
            # Request for cross-user experience sharing
            cross_user_data = update.data
            await self._process_cross_user_request(cross_user_data)
        
        elif update.update_type == "narrative_request":
            # Request for narrative construction
            narrative_data = update.data
            await self._process_narrative_request(narrative_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with experience awareness"""
        # Check if input contains experience-related queries
        experience_indicators = await self._detect_experience_indicators(context)
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Determine if immediate experience sharing is appropriate
        immediate_sharing = await self._check_immediate_sharing_need(context, experience_indicators, messages)
        
        if immediate_sharing:
            # Share experience immediately
            sharing_result = await self._execute_immediate_experience_sharing(context, immediate_sharing)
            
            # Track effectiveness
            await self._track_sharing_effectiveness(context, sharing_result)
            
            # Send experience sharing update
            await self.send_context_update(
                update_type="experience_shared",
                data={
                    "trigger": immediate_sharing,
                    "result": sharing_result,
                    "experience_id": sharing_result.get("experience", {}).get("id"),
                    "relevance": sharing_result.get("relevance_score", 0.5)
                },
                priority=ContextPriority.HIGH
            )
            
            return {
                "experience_processed": True,
                "immediate_sharing": True,
                "result": sharing_result,
                "indicators": experience_indicators
            }
        
        # Regular processing with experience monitoring
        monitoring_result = await self._monitor_experience_opportunities(context, messages)
        
        return {
            "experience_processed": True,
            "immediate_sharing": False,
            "monitoring": monitoring_result,
            "indicators": experience_indicators
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze experience sharing opportunities and effectiveness"""
        # Comprehensive experience analysis
        experience_landscape = await self._analyze_available_experiences(context)
        
        # Effectiveness analysis
        effectiveness_analysis = await self._analyze_sharing_effectiveness()
        
        # User preference analysis
        preference_analysis = await self._analyze_user_preferences(context)
        
        # Cross-module experience usage
        messages = await self.get_cross_module_messages()
        cross_module_usage = await self._analyze_cross_module_experience_usage(messages)
        
        # Predict optimal sharing strategies
        optimal_strategies = await self._predict_optimal_sharing_strategies(
            experience_landscape, effectiveness_analysis, preference_analysis, cross_module_usage
        )
        
        return {
            "experience_landscape": experience_landscape,
            "effectiveness_analysis": effectiveness_analysis,
            "preference_analysis": preference_analysis,
            "cross_module_usage": cross_module_usage,
            "optimal_strategies": optimal_strategies,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize experience-based insights for response"""
        messages = await self.get_cross_module_messages()
        
        # Create experience synthesis
        experience_synthesis = {
            "experience_contribution": await self._synthesize_experience_contribution(context),
            "narrative_elements": await self._generate_narrative_elements(context),
            "reflection_insights": await self._generate_reflection_insights(context, messages),
            "identity_reflection": await self._synthesize_identity_reflection(),
            "sharing_recommendations": await self._generate_sharing_recommendations(context, messages)
        }
        
        # Check if proactive experience sharing is recommended
        proactive_sharing = await self._evaluate_proactive_sharing(experience_synthesis, context)
        
        if proactive_sharing:
            await self.send_context_update(
                update_type="proactive_experience_sharing_recommended",
                data={
                    "recommendation": proactive_sharing,
                    "rationale": experience_synthesis["sharing_recommendations"],
                    "expected_impact": proactive_sharing.get("expected_impact", "enhanced_engagement")
                },
                priority=ContextPriority.NORMAL
            )
        
        return experience_synthesis
    
    # ========================================================================================
    # ADVANCED HELPER METHODS
    # ========================================================================================
    
    async def _analyze_experience_relevance(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze how relevant experiences are to current context"""
        relevance = {
            "overall_relevance": 0.5,
            "relevance_factors": {},
            "context_alignment": 0.5,
            "user_engagement_likelihood": 0.5
        }
        
        # Check input complexity and length
        input_words = len(context.user_input.split())
        if input_words > 10:
            relevance["relevance_factors"]["input_complexity"] = 0.7
            relevance["overall_relevance"] += 0.1
        
        # Check for experience-triggering keywords
        experience_keywords = [
            "remember", "recall", "experience", "before", "last time",
            "similar", "reminds", "like when", "that time"
        ]
        
        keyword_matches = sum(1 for kw in experience_keywords if kw in context.user_input.lower())
        if keyword_matches > 0:
            relevance["relevance_factors"]["keyword_triggers"] = min(1.0, keyword_matches * 0.3)
            relevance["overall_relevance"] += 0.2
        
        # Check emotional context alignment
        if context.emotional_state:
            emotional_intensity = max(context.emotional_state.values()) if context.emotional_state else 0
            if emotional_intensity > 0.5:
                relevance["relevance_factors"]["emotional_alignment"] = emotional_intensity
                relevance["overall_relevance"] += 0.1
        
        # Check goal context alignment
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            if active_goals:
                relevance["relevance_factors"]["goal_alignment"] = 0.6
                relevance["overall_relevance"] += 0.1
        
        # Calculate user engagement likelihood based on history
        user_id = context.user_id or "default"
        if user_id in self.user_engagement_tracking:
            recent_engagement = self.user_engagement_tracking[user_id][-5:]  # Last 5 interactions
            if recent_engagement:
                avg_engagement = sum(recent_engagement) / len(recent_engagement)
                relevance["user_engagement_likelihood"] = avg_engagement
        
        # Calculate context alignment
        relevance["context_alignment"] = sum(relevance["relevance_factors"].values()) / max(1, len(relevance["relevance_factors"]))
        
        # Cap overall relevance
        relevance["overall_relevance"] = min(1.0, relevance["overall_relevance"])
        
        return relevance
    
    async def _find_contextually_relevant_experiences(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Find experiences relevant to current context"""
        # Build search query from context
        search_components = []
        
        # Add key terms from user input
        input_words = context.user_input.split()
        key_terms = [w for w in input_words if len(w) > 4]  # Longer words are likely more meaningful
        search_components.extend(key_terms[:3])  # Top 3 key terms
        
        # Add emotional context if significant
        if context.emotional_state:
            dominant_emotion = max(context.emotional_state.items(), key=lambda x: x[1])[0] if context.emotional_state else None
            if dominant_emotion:
                search_components.append(dominant_emotion.lower())
        
        # Add goal context if available
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            if active_goals and active_goals[0].get("description"):
                goal_words = active_goals[0]["description"].split()[:2]
                search_components.extend(goal_words)
        
        # Build query
        search_query = " ".join(search_components) if search_components else context.user_input
        
        # Use original interface to retrieve experiences
        try:
            experiences = await self.original_interface.retrieve_experiences_enhanced(
                query=search_query,
                limit=5,
                user_id=context.user_id,
                include_cross_user=True
            )
            
            return experiences
            
        except Exception as e:
            logger.error(f"Error retrieving experiences: {e}")
            return []
    
    async def _determine_sharing_strategy(self, 
                                        context: SharedContext,
                                        experience_relevance: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal experience sharing strategy"""
        strategy = {
            "approach": "standard",
            "depth": "moderate",
            "formatting": "conversational",
            "cross_user_enabled": False,
            "narrative_mode": False
        }
        
        # High relevance = more detailed sharing
        if experience_relevance["overall_relevance"] > 0.7:
            strategy["approach"] = "detailed"
            strategy["depth"] = "deep"
        elif experience_relevance["overall_relevance"] < 0.3:
            strategy["approach"] = "minimal"
            strategy["depth"] = "surface"
        
        # Check relationship context for depth adjustment
        if context.relationship_context:
            trust = context.relationship_context.get("trust", 0.5)
            intimacy = context.relationship_context.get("intimacy", 0.5)
            
            if trust > 0.7 and intimacy > 0.6:
                strategy["depth"] = "intimate"
                strategy["formatting"] = "personal"
        
        # Check for narrative indicators
        narrative_keywords = ["story", "tell me about", "what happened", "describe"]
        if any(kw in context.user_input.lower() for kw in narrative_keywords):
            strategy["narrative_mode"] = True
            strategy["formatting"] = "narrative"
        
        # Check user preference for cross-user experiences
        user_id = context.user_id or "default"
        user_profile = self.original_interface._get_user_preference_profile(user_id)
        
        if user_profile.get("experience_sharing_preference", 0.5) > 0.6:
            strategy["cross_user_enabled"] = True
        
        # Adjust based on processing stage
        if context.processing_stage == "synthesis":
            strategy["approach"] = "integrated"  # Blend with other module outputs
        
        return strategy
    
    async def _get_user_experience_preference(self, user_id: Optional[str]) -> Dict[str, Any]:
        """Get user's experience sharing preferences"""
        user_id = user_id or "default"
        
        # Get profile from original interface
        profile = self.original_interface._get_user_preference_profile(user_id)
        
        # Add engagement history
        if user_id in self.user_engagement_tracking:
            recent_engagement = self.user_engagement_tracking[user_id][-10:]
            if recent_engagement:
                profile["recent_engagement_average"] = sum(recent_engagement) / len(recent_engagement)
            else:
                profile["recent_engagement_average"] = 0.5
        else:
            profile["recent_engagement_average"] = 0.5
        
        # Add effectiveness metrics
        if user_id in self.experience_effectiveness:
            effectiveness_data = self.experience_effectiveness[user_id]
            profile["experience_effectiveness"] = effectiveness_data.get("average", 0.5)
            profile["preferred_scenarios"] = effectiveness_data.get("top_scenarios", [])
        
        return profile
    
    async def _process_experience_request(self, query: str, request_type: str, request_data: Dict[str, Any]):
        """Process a direct experience request"""
        user_id = request_data.get("user_id", "default")
        
        # Handle different request types
        if request_type == "recall":
            # Direct recall request
            result = await self.original_interface.share_experience_enhanced(
                query=query,
                context_data=request_data
            )
            
        elif request_type == "narrative":
            # Narrative construction request
            experiences = await self.original_interface.retrieve_experiences_enhanced(
                query=query,
                limit=request_data.get("experience_count", 5),
                user_id=user_id
            )
            
            if experiences:
                result = await self.original_interface.construct_narrative(
                    experiences=experiences,
                    topic=query,
                    chronological=request_data.get("chronological", True)
                )
            else:
                result = {"narrative": None, "reason": "no_experiences_found"}
                
        elif request_type == "reflection":
            # Reflection generation request
            experiences = await self.original_interface.retrieve_experiences_enhanced(
                query=query,
                limit=3,
                user_id=user_id
            )
            
            if experiences:
                result = await self.original_interface.generate_personality_reflection(
                    experiences=experiences,
                    context=request_data
                )
            else:
                result = {"reflection": "I haven't had specific experiences to reflect on yet."}
                
        else:
            # Standard experience sharing
            result = await self.original_interface.handle_experience_sharing_request(
                user_query=query,
                context_data=request_data
            )
        
        # Send result update
        await self.send_context_update(
            update_type="experience_request_processed",
            data={
                "request_type": request_type,
                "query": query,
                "result": result,
                "success": bool(result and result.get("has_experience", False))
            }
        )
    
    async def _update_emotional_context_for_experiences(self, emotional_data: Dict[str, Any]):
        """Update experience selection based on emotional context"""
        emotional_state = emotional_data.get("emotional_state", {})
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if not dominant_emotion:
            return
        
        emotion_name, intensity = dominant_emotion
        
        # Update current context
        self.current_experience_context["emotional_influence"] = {
            "emotion": emotion_name,
            "intensity": intensity,
            "timestamp": datetime.now().isoformat()
        }
        
        # Adjust experience preferences based on emotion
        emotion_experience_mapping = {
            "Joy": ["positive", "celebration", "achievement"],
            "Sadness": ["comfort", "understanding", "empathy"],
            "Anger": ["resolution", "cathartic", "understanding"],
            "Fear": ["reassurance", "safety", "courage"],
            "Surprise": ["discovery", "unexpected", "novel"],
            "Disgust": ["boundary", "preference", "clarity"],
            "Anticipation": ["excitement", "planning", "possibility"],
            "Trust": ["bonding", "intimate", "deep"]
        }
        
        if emotion_name in emotion_experience_mapping:
            self.current_experience_context["preferred_themes"] = emotion_experience_mapping[emotion_name]
            self.current_experience_context["emotional_alignment_weight"] = intensity
    
    async def _incorporate_goal_context(self, goal_data: Dict[str, Any]):
        """Incorporate goal context into experience selection"""
        active_goals = goal_data.get("active_goals", [])
        goal_progress = goal_data.get("goal_progress", {})
        
        if not active_goals:
            return
        
        # Focus on top priority goals
        priority_goals = sorted(active_goals, key=lambda g: g.get("priority", 0.5), reverse=True)[:3]
        
        # Extract goal-related keywords
        goal_keywords = []
        for goal in priority_goals:
            description = goal.get("description", "")
            keywords = [w for w in description.split() if len(w) > 3]
            goal_keywords.extend(keywords[:2])  # Top 2 keywords per goal
        
        self.current_experience_context["goal_keywords"] = goal_keywords
        self.current_experience_context["goal_focus"] = True
        
        # If goals are being achieved, prefer success experiences
        if goal_progress.get("recent_completions", 0) > 0:
            self.current_experience_context["prefer_success_experiences"] = True
    
    async def _update_identity_based_preferences(self, identity_data: Dict[str, Any]):
        """Update experience preferences based on identity evolution"""
        preference_updates = identity_data.get("preference_updates", {})
        trait_updates = identity_data.get("trait_updates", {})
        
        # Update scenario preferences
        if "scenario_types" in preference_updates:
            for scenario, change in preference_updates["scenario_types"].items():
                # Positive changes increase preference for those scenarios
                if change > 0:
                    if "preferred_scenarios" not in self.current_experience_context:
                        self.current_experience_context["preferred_scenarios"] = []
                    
                    if scenario not in self.current_experience_context["preferred_scenarios"]:
                        self.current_experience_context["preferred_scenarios"].append(scenario)
        
        # Update based on trait changes
        if trait_updates:
            # Higher dominance = prefer control experiences
            if trait_updates.get("dominance", {}).get("change", 0) > 0:
                self.current_experience_context["prefer_control_experiences"] = True
            
            # Higher playfulness = prefer teasing experiences
            if trait_updates.get("playfulness", {}).get("change", 0) > 0:
                self.current_experience_context["prefer_playful_experiences"] = True
    
    async def _process_memory_retrieval_request(self, retrieval_data: Dict[str, Any]):
        """Process specific memory retrieval request"""
        memory_ids = retrieval_data.get("memory_ids", [])
        retrieval_purpose = retrieval_data.get("purpose", "general")
        
        if not memory_ids:
            return
        
        # Retrieve specific memories
        retrieved_experiences = []
        
        for memory_id in memory_ids:
            try:
                # Get memory from memory core
                memory = await self.original_interface.memory_core.get_memory_by_id(memory_id)
                
                if memory:
                    # Convert to experience format
                    experience = await self.original_interface._convert_memory_to_experience(
                        memory=memory,
                        emotional_context={},  # Would need to extract
                        relevance_score=1.0,  # Direct request = high relevance
                        experiential_richness=0.7  # Default
                    )
                    retrieved_experiences.append(experience)
                    
            except Exception as e:
                logger.error(f"Error retrieving memory {memory_id}: {e}")
        
        # Format based on purpose
        if retrieval_purpose == "narrative" and len(retrieved_experiences) > 1:
            # Construct narrative from retrieved experiences
            result = await self.original_interface.construct_narrative(
                experiences=retrieved_experiences,
                topic="requested memories",
                chronological=True
            )
        else:
            # Return individual experiences
            result = {
                "experiences": retrieved_experiences,
                "count": len(retrieved_experiences),
                "purpose": retrieval_purpose
            }
        
        # Send update
        await self.send_context_update(
            update_type="memory_retrieval_complete",
            data={
                "requested_ids": memory_ids,
                "retrieved_count": len(retrieved_experiences),
                "result": result
            }
        )
    
    async def _adjust_sharing_based_on_relationship(self, relationship_data: Dict[str, Any]):
        """Adjust experience sharing based on relationship context"""
        trust = relationship_data.get("trust", 0.5)
        intimacy = relationship_data.get("intimacy", 0.5)
        relationship_stage = relationship_data.get("stage", "developing")
        
        # Update sharing parameters
        self.current_experience_context["relationship_adjusted"] = True
        
        # High trust enables deeper sharing
        if trust > 0.7:
            self.current_experience_context["sharing_depth"] = "deep"
            self.current_experience_context["include_vulnerable_experiences"] = True
        elif trust < 0.3:
            self.current_experience_context["sharing_depth"] = "surface"
            self.current_experience_context["include_vulnerable_experiences"] = False
        else:
            self.current_experience_context["sharing_depth"] = "moderate"
        
        # High intimacy enables more personal experiences
        if intimacy > 0.7:
            self.current_experience_context["include_intimate_experiences"] = True
            self.current_experience_context["emotional_openness"] = "high"
        else:
            self.current_experience_context["include_intimate_experiences"] = False
            self.current_experience_context["emotional_openness"] = "moderate"
        
        # Adjust based on relationship stage
        if relationship_stage == "new":
            self.current_experience_context["prefer_lighter_experiences"] = True
        elif relationship_stage == "deepening":
            self.current_experience_context["include_bonding_experiences"] = True
        elif relationship_stage == "established":
            self.current_experience_context["full_experience_range"] = True
    
    async def _incorporate_temporal_context(self, temporal_data: Dict[str, Any]):
        """Incorporate temporal perception into experience selection"""
        time_category = temporal_data.get("time_category", "medium")
        psychological_age = temporal_data.get("psychological_age", 0.5)
        milestone_context = temporal_data.get("milestone_context", {})
        
        # Update temporal preferences
        self.current_experience_context["temporal_context"] = {
            "time_category": time_category,
            "psychological_age": psychological_age
        }
        
        # Long time perception favors nostalgic experiences
        if time_category in ["long", "very_long"]:
            self.current_experience_context["prefer_nostalgic"] = True
            self.current_experience_context["include_early_experiences"] = True
        
        # High psychological age favors mature reflections
        if psychological_age > 0.7:
            self.current_experience_context["prefer_mature_reflections"] = True
            self.current_experience_context["include_growth_experiences"] = True
        
        # Milestone context triggers milestone-related experiences
        if milestone_context:
            self.current_experience_context["milestone_focus"] = True
            self.current_experience_context["milestone_type"] = milestone_context.get("type", "general")
    
    async def _process_cross_user_request(self, cross_user_data: Dict[str, Any]):
        """Process request for cross-user experience sharing"""
        query = cross_user_data.get("query", "")
        user_id = cross_user_data.get("user_id", "default")
        filters = cross_user_data.get("filters", {})
        
        # Check if cross-user sharing is allowed
        user_profile = self.original_interface._get_user_preference_profile(user_id)
        
        if user_profile.get("experience_sharing_preference", 0.5) < 0.3:
            # User doesn't want cross-user experiences
            result = {
                "has_experience": False,
                "reason": "user_preference_disabled"
            }
        else:
            # Get cross-user experience
            result = await self.original_interface.share_cross_user_experience(
                query=query,
                user_id=user_id,
                context_data=cross_user_data
            )
        
        # Send update
        await self.send_context_update(
            update_type="cross_user_experience_processed",
            data={
                "query": query,
                "result": result,
                "filters_applied": filters
            }
        )
    
    async def _process_narrative_request(self, narrative_data: Dict[str, Any]):
        """Process request for narrative construction"""
        topic = narrative_data.get("topic", "experiences")
        experience_ids = narrative_data.get("experience_ids", [])
        chronological = narrative_data.get("chronological", True)
        style = narrative_data.get("style", "standard")
        
        # Get experiences for narrative
        if experience_ids:
            # Use specific experiences
            experiences = []
            for exp_id in experience_ids:
                try:
                    memory = await self.original_interface.memory_core.get_memory_by_id(exp_id)
                    if memory:
                        experience = await self.original_interface._convert_memory_to_experience(
                            memory, {}, 0.8, 0.7
                        )
                        experiences.append(experience)
                except Exception as e:
                    logger.error(f"Error retrieving experience {exp_id}: {e}")
        else:
            # Find relevant experiences
            experiences = await self.original_interface.retrieve_experiences_enhanced(
                query=topic,
                limit=narrative_data.get("max_experiences", 7),
                user_id=narrative_data.get("user_id", "default")
            )
        
        if experiences:
            # Construct narrative
            result = await self.original_interface.construct_narrative(
                experiences=experiences,
                topic=topic,
                chronological=chronological
            )
            
            # Apply style adjustments
            if style == "intimate" and result.get("narrative"):
                result["narrative"] = self._apply_intimate_style(result["narrative"])
            elif style == "reflective" and result.get("narrative"):
                result["narrative"] = self._apply_reflective_style(result["narrative"])
        else:
            result = {
                "narrative": None,
                "reason": "no_experiences_available"
            }
        
        # Send update
        await self.send_context_update(
            update_type="narrative_complete",
            data={
                "topic": topic,
                "style": style,
                "result": result
            }
        )
    
    async def _detect_experience_indicators(self, context: SharedContext) -> Dict[str, Any]:
        """Detect indicators for experience sharing in input"""
        indicators = {
            "explicit_request": False,
            "implicit_trigger": False,
            "emotional_trigger": False,
            "temporal_reference": False,
            "comparison_request": False,
            "trigger_keywords": []
        }
        
        input_lower = context.user_input.lower()
        
        # Explicit experience requests
        explicit_keywords = [
            "tell me about", "do you remember", "have you experienced",
            "what was it like", "share an experience", "recall"
        ]
        
        for keyword in explicit_keywords:
            if keyword in input_lower:
                indicators["explicit_request"] = True
                indicators["trigger_keywords"].append(keyword)
        
        # Implicit triggers
        implicit_keywords = [
            "reminds me", "similar to", "like when", "before",
            "last time", "previously", "in the past"
        ]
        
        for keyword in implicit_keywords:
            if keyword in input_lower:
                indicators["implicit_trigger"] = True
                indicators["trigger_keywords"].append(keyword)
        
        # Emotional triggers
        if context.emotional_state:
            max_emotion_intensity = max(context.emotional_state.values()) if context.emotional_state else 0
            if max_emotion_intensity > 0.6:
                indicators["emotional_trigger"] = True
        
        # Temporal references
        temporal_keywords = ["yesterday", "last week", "month ago", "remember when", "that time"]
        
        if any(kw in input_lower for kw in temporal_keywords):
            indicators["temporal_reference"] = True
        
        # Comparison requests
        comparison_keywords = ["different from", "compared to", "unlike", "similar to", "like"]
        
        if any(kw in input_lower for kw in comparison_keywords):
            indicators["comparison_request"] = True
        
        return indicators
    
    async def _check_immediate_sharing_need(self,
                                          context: SharedContext,
                                          indicators: Dict[str, Any],
                                          messages: Dict[str, List[Dict]]) -> Optional[str]:
        """Check if immediate experience sharing is needed"""
        # Explicit request always triggers
        if indicators["explicit_request"]:
            return "explicit_request"
        
        # High emotional context with implicit trigger
        if indicators["emotional_trigger"] and indicators["implicit_trigger"]:
            return "emotional_context"
        
        # Check cross-module requests
        for module_messages in messages.values():
            for msg in module_messages:
                if msg.get("type") == "request_experience_context":
                    return "module_request"
        
        # Check user preference and engagement
        user_id = context.user_id or "default"
        user_profile = self.original_interface._get_user_preference_profile(user_id)
        
        if user_profile.get("experience_sharing_preference", 0.5) > 0.7:
            # High preference user with any trigger
            if any(indicators.values()):
                return "high_preference_trigger"
        
        # Temporal reference with sufficient context
        if indicators["temporal_reference"] and len(context.active_modules) > 5:
            return "temporal_context"
        
        return None
    
    async def _execute_immediate_experience_sharing(self,
                                                  context: SharedContext,
                                                  trigger: str) -> Dict[str, Any]:
        """Execute immediate experience sharing"""
        logger.info(f"Executing immediate experience sharing due to: {trigger}")
        
        # Prepare context data
        context_data = {
            "user_id": context.user_id,
            "trigger": trigger,
            "emotional_state": context.emotional_state,
            "goal_context": context.goal_context,
            "relationship_context": context.relationship_context
        }
        
        # Add current experience context
        context_data.update(self.current_experience_context)
        
        # Different handling based on trigger
        if trigger == "explicit_request":
            # Direct experience sharing
            result = await self.original_interface.handle_experience_sharing_request(
                user_query=context.user_input,
                context_data=context_data
            )
            
        elif trigger == "emotional_context":
            # Emotionally-driven experience
            # Find experiences matching emotional state
            dominant_emotion = max(context.emotional_state.items(), key=lambda x: x[1])[0] if context.emotional_state else "neutral"
            
            enhanced_query = f"{context.user_input} {dominant_emotion.lower()}"
            result = await self.original_interface.share_experience_enhanced(
                query=enhanced_query,
                context_data=context_data
            )
            
        elif trigger == "temporal_context":
            # Temporal experience sharing
            result = await self.original_interface.share_experience_with_temporal_context(
                query=context.user_input,
                temporal_context=context_data.get("temporal_context", {})
            )
            
        else:
            # Default experience sharing
            result = await self.original_interface.share_experience_enhanced(
                query=context.user_input,
                context_data=context_data
            )
        
        return result
    
    async def _track_sharing_effectiveness(self, context: SharedContext, sharing_result: Dict[str, Any]):
        """Track effectiveness of experience sharing"""
        user_id = context.user_id or "default"
        
        # Initialize tracking for user if needed
        if user_id not in self.experience_effectiveness:
            self.experience_effectiveness[user_id] = {
                "total_shares": 0,
                "successful_shares": 0,
                "average": 0.5,
                "scenario_effectiveness": {},
                "top_scenarios": []
            }
        
        effectiveness = self.experience_effectiveness[user_id]
        effectiveness["total_shares"] += 1
        
        # Determine if sharing was successful
        success = sharing_result.get("has_experience", False) and sharing_result.get("relevance_score", 0) > 0.5
        
        if success:
            effectiveness["successful_shares"] += 1
            
            # Track scenario effectiveness
            if "experience" in sharing_result and sharing_result["experience"]:
                scenario = sharing_result["experience"].get("scenario_type", "general")
                
                if scenario not in effectiveness["scenario_effectiveness"]:
                    effectiveness["scenario_effectiveness"][scenario] = {
                        "count": 0,
                        "success": 0
                    }
                
                effectiveness["scenario_effectiveness"][scenario]["count"] += 1
                effectiveness["scenario_effectiveness"][scenario]["success"] += 1
        
        # Update average
        effectiveness["average"] = effectiveness["successful_shares"] / effectiveness["total_shares"]
        
        # Update top scenarios
        if effectiveness["scenario_effectiveness"]:
            sorted_scenarios = sorted(
                effectiveness["scenario_effectiveness"].items(),
                key=lambda x: x[1]["success"] / max(1, x[1]["count"]),
                reverse=True
            )
            effectiveness["top_scenarios"] = [s[0] for s in sorted_scenarios[:3]]
        
        # Track engagement
        if user_id not in self.user_engagement_tracking:
            self.user_engagement_tracking[user_id] = []
        
        # Simple engagement metric based on relevance
        engagement = sharing_result.get("relevance_score", 0.5)
        self.user_engagement_tracking[user_id].append(engagement)
        
        # Keep only recent engagement data
        if len(self.user_engagement_tracking[user_id]) > 20:
            self.user_engagement_tracking[user_id] = self.user_engagement_tracking[user_id][-20:]
    
    async def _monitor_experience_opportunities(self,
                                              context: SharedContext,
                                              messages: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Monitor for experience sharing opportunities"""
        monitoring = {
            "opportunities_detected": 0,
            "opportunity_types": [],
            "module_requests": 0,
            "user_engagement_trend": "stable"
        }
        
        # Check for implicit opportunities in user input
        opportunity_keywords = [
            "how", "why", "when", "what if", "wonder",
            "curious", "interesting", "tell me"
        ]
        
        input_lower = context.user_input.lower()
        for keyword in opportunity_keywords:
            if keyword in input_lower:
                monitoring["opportunities_detected"] += 1
                monitoring["opportunity_types"].append(f"keyword_{keyword}")
        
        # Check cross-module messages for experience requests
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                if "experience" in msg.get("type", "").lower():
                    monitoring["module_requests"] += 1
                    monitoring["opportunity_types"].append(f"module_{module_name}")
        
        # Analyze user engagement trend
        user_id = context.user_id or "default"
        if user_id in self.user_engagement_tracking:
            recent = self.user_engagement_tracking[user_id][-10:]
            if len(recent) >= 5:
                first_half = recent[:5]
                second_half = recent[5:]
                
                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)
                
                if second_avg > first_avg * 1.1:
                    monitoring["user_engagement_trend"] = "increasing"
                elif second_avg < first_avg * 0.9:
                    monitoring["user_engagement_trend"] = "decreasing"
        
        return monitoring
    
    async def _analyze_available_experiences(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze available experiences for sharing"""
        analysis = {
            "total_available": 0,
            "scenario_distribution": {},
            "temporal_coverage": {},
            "emotional_coverage": {},
            "cross_user_available": 0,
            "consolidation_coverage": 0.0
        }
        
        # Get user's experiences
        user_id = context.user_id or "default"
        
        try:
            # Sample retrieval to analyze available experiences
            sample_experiences = await self.original_interface.retrieve_experiences_enhanced(
                query="*",  # Broad query
                limit=50,
                user_id=user_id,
                include_cross_user=False
            )
            
            analysis["total_available"] = len(sample_experiences)
            
            # Analyze distribution
            for exp in sample_experiences:
                # Scenario distribution
                scenario = exp.get("scenario_type", "general")
                analysis["scenario_distribution"][scenario] = \
                    analysis["scenario_distribution"].get(scenario, 0) + 1
                
                # Emotional coverage
                emotional_context = exp.get("emotional_context", {})
                if emotional_context:
                    emotion = emotional_context.get("primary_emotion", "neutral")
                    analysis["emotional_coverage"][emotion] = \
                        analysis["emotional_coverage"].get(emotion, 0) + 1
                
                # Temporal coverage
                timestamp = exp.get("timestamp")
                if timestamp:
                    try:
                        exp_date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        month_key = exp_date.strftime("%Y-%m")
                        analysis["temporal_coverage"][month_key] = \
                            analysis["temporal_coverage"].get(month_key, 0) + 1
                    except:
                        pass
            
            # Check cross-user availability
            cross_user_sample = await self.original_interface.retrieve_experiences_enhanced(
                query="*",
                limit=20,
                user_id=user_id,
                include_cross_user=True
            )
            
            cross_user_count = sum(1 for exp in cross_user_sample 
                                 if exp.get("user_id", "default") != user_id)
            analysis["cross_user_available"] = cross_user_count
            
        except Exception as e:
            logger.error(f"Error analyzing available experiences: {e}")
        
        return analysis
    
    async def _analyze_sharing_effectiveness(self) -> Dict[str, Any]:
        """Analyze overall sharing effectiveness"""
        effectiveness = {
            "overall_effectiveness": 0.5,
            "user_effectiveness": {},
            "scenario_performance": {},
            "trend": "stable"
        }
        
        # Aggregate across all users
        total_shares = 0
        successful_shares = 0
        scenario_aggregates = {}
        
        for user_id, user_data in self.experience_effectiveness.items():
            # User-specific effectiveness
            effectiveness["user_effectiveness"][user_id] = user_data["average"]
            
            total_shares += user_data["total_shares"]
            successful_shares += user_data["successful_shares"]
            
            # Aggregate scenario data
            for scenario, scenario_data in user_data["scenario_effectiveness"].items():
                if scenario not in scenario_aggregates:
                    scenario_aggregates[scenario] = {"count": 0, "success": 0}
                
                scenario_aggregates[scenario]["count"] += scenario_data["count"]
                scenario_aggregates[scenario]["success"] += scenario_data["success"]
        
        # Calculate overall effectiveness
        if total_shares > 0:
            effectiveness["overall_effectiveness"] = successful_shares / total_shares
        
        # Calculate scenario performance
        for scenario, data in scenario_aggregates.items():
            if data["count"] > 0:
                effectiveness["scenario_performance"][scenario] = data["success"] / data["count"]
        
        # Determine trend (simplified)
        if len(self.experience_effectiveness) >= 2:
            recent_averages = [
                data["average"] for data in self.experience_effectiveness.values()
            ]
            
            if recent_averages:
                recent_avg = sum(recent_averages) / len(recent_averages)
                
                if recent_avg > 0.6:
                    effectiveness["trend"] = "improving"
                elif recent_avg < 0.4:
                    effectiveness["trend"] = "declining"
        
        return effectiveness
    
    async def _analyze_user_preferences(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze user preferences for experience sharing"""
        user_id = context.user_id or "default"
        
        # Get base preferences
        preferences = self.original_interface._get_user_preference_profile(user_id)
        
        # Enhance with usage data
        if user_id in self.experience_effectiveness:
            effectiveness_data = self.experience_effectiveness[user_id]
            
            # Add demonstrated preferences from usage
            preferences["demonstrated_preferences"] = {
                "sharing_success_rate": effectiveness_data["average"],
                "preferred_scenarios": effectiveness_data["top_scenarios"],
                "total_interactions": effectiveness_data["total_shares"]
            }
        
        # Add engagement analysis
        if user_id in self.user_engagement_tracking:
            recent_engagement = self.user_engagement_tracking[user_id][-10:]
            if recent_engagement:
                preferences["engagement_metrics"] = {
                    "recent_average": sum(recent_engagement) / len(recent_engagement),
                    "trend": self._calculate_engagement_trend(recent_engagement)
                }
        
        return preferences
    
    async def _analyze_cross_module_experience_usage(self, messages: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze how other modules use experiences"""
        usage = {
            "module_usage_count": {},
            "usage_patterns": [],
            "experience_driven_actions": 0,
            "module_satisfaction": {}
        }
        
        # Analyze messages for experience usage
        for module_name, module_messages in messages.items():
            module_usage = 0
            module_satisfaction_scores = []
            
            for msg in module_messages:
                msg_type = msg.get("type", "")
                msg_data = msg.get("data", {})
                
                # Count experience usage
                if "experience" in msg_type.lower():
                    module_usage += 1
                    
                    # Check for action triggers
                    if "action" in msg_type or "decision" in msg_type:
                        usage["experience_driven_actions"] += 1
                
                # Check for satisfaction/utility feedback
                if "experience_utility" in msg_data:
                    module_satisfaction_scores.append(msg_data["experience_utility"])
            
            if module_usage > 0:
                usage["module_usage_count"][module_name] = module_usage
                
                # Calculate module satisfaction
                if module_satisfaction_scores:
                    usage["module_satisfaction"][module_name] = \
                        sum(module_satisfaction_scores) / len(module_satisfaction_scores)
        
        # Identify usage patterns
        if usage["module_usage_count"]:
            # Modules that frequently use experiences
            frequent_users = [m for m, count in usage["module_usage_count"].items() if count > 3]
            if frequent_users:
                usage["usage_patterns"].append({
                    "pattern": "frequent_usage",
                    "modules": frequent_users
                })
            
            # Modules with high satisfaction
            satisfied_modules = [m for m, score in usage["module_satisfaction"].items() if score > 0.7]
            if satisfied_modules:
                usage["usage_patterns"].append({
                    "pattern": "high_satisfaction",
                    "modules": satisfied_modules
                })
        
        return usage
    
    async def _predict_optimal_sharing_strategies(self,
                                                experience_landscape: Dict[str, Any],
                                                effectiveness_analysis: Dict[str, Any],
                                                preference_analysis: Dict[str, Any],
                                                cross_module_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal experience sharing strategies"""
        strategies = {
            "recommended_approach": "balanced",
            "scenario_strategies": {},
            "cross_user_recommendation": "moderate",
            "timing_recommendations": [],
            "format_recommendations": []
        }
        
        # Overall approach based on effectiveness
        overall_effectiveness = effectiveness_analysis.get("overall_effectiveness", 0.5)
        
        if overall_effectiveness > 0.7:
            strategies["recommended_approach"] = "proactive"
            strategies["timing_recommendations"].append("Share experiences early in conversations")
        elif overall_effectiveness < 0.3:
            strategies["recommended_approach"] = "selective"
            strategies["timing_recommendations"].append("Wait for explicit requests")
        else:
            strategies["recommended_approach"] = "responsive"
            strategies["timing_recommendations"].append("Share when contextually relevant")
        
        # Scenario-specific strategies
        scenario_performance = effectiveness_analysis.get("scenario_performance", {})
        
        for scenario, performance in scenario_performance.items():
            if performance > 0.7:
                strategies["scenario_strategies"][scenario] = "emphasize"
            elif performance < 0.3:
                strategies["scenario_strategies"][scenario] = "minimize"
            else:
                strategies["scenario_strategies"][scenario] = "normal"
        
        # Cross-user recommendation
        cross_user_available = experience_landscape.get("cross_user_available", 0)
        user_preference = preference_analysis.get("experience_sharing_preference", 0.5)
        
        if cross_user_available > 10 and user_preference > 0.6:
            strategies["cross_user_recommendation"] = "active"
        elif user_preference < 0.3:
            strategies["cross_user_recommendation"] = "disabled"
        else:
            strategies["cross_user_recommendation"] = "selective"
        
        # Format recommendations based on engagement
        if preference_analysis.get("engagement_metrics", {}).get("recent_average", 0.5) > 0.7:
            strategies["format_recommendations"].append("Use detailed, narrative formats")
            strategies["format_recommendations"].append("Include emotional depth")
        else:
            strategies["format_recommendations"].append("Keep experiences concise")
            strategies["format_recommendations"].append("Focus on relevance")
        
        # Module-specific recommendations
        if cross_module_usage.get("module_satisfaction"):
            high_satisfaction_modules = [
                m for m, s in cross_module_usage["module_satisfaction"].items() if s > 0.7
            ]
            if high_satisfaction_modules:
                strategies["format_recommendations"].append(
                    f"Prioritize experiences for: {', '.join(high_satisfaction_modules)}"
                )
        
        return strategies
    
    async def _synthesize_experience_contribution(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize how experiences contribute to the response"""
        contribution = {
            "experience_included": False,
            "contribution_type": "none",
            "experience_text": None,
            "enhancement_level": 0.0
        }
        
        # Check if we've shared an experience in this context
        if hasattr(self, 'last_shared_experience') and self.last_shared_experience:
            contribution["experience_included"] = True
            contribution["experience_text"] = self.last_shared_experience.get("response_text")
            
            # Determine contribution type
            if self.last_shared_experience.get("trigger") == "explicit_request":
                contribution["contribution_type"] = "primary"
                contribution["enhancement_level"] = 0.8
            elif self.last_shared_experience.get("trigger") == "emotional_context":
                contribution["contribution_type"] = "emotional_support"
                contribution["enhancement_level"] = 0.6
            else:
                contribution["contribution_type"] = "contextual"
                contribution["enhancement_level"] = 0.4
        
        # Check if experiences informed other modules
        if hasattr(self, 'experience_influenced_modules'):
            contribution["influenced_modules"] = list(self.experience_influenced_modules)
            contribution["enhancement_level"] += 0.1 * len(self.experience_influenced_modules)
        
        contribution["enhancement_level"] = min(1.0, contribution["enhancement_level"])
        
        return contribution
    
    async def _generate_narrative_elements(self, context: SharedContext) -> List[str]:
        """Generate narrative elements from experiences"""
        elements = []
        
        # Check for available narrative-worthy experiences
        if hasattr(self, 'narrative_experiences') and self.narrative_experiences:
            for exp in self.narrative_experiences[:3]:  # Top 3
                element = f"The time when {exp.get('brief_description', 'something happened')}"
                elements.append(element)
        
        # Add temporal narrative elements
        if context.session_context.get("temporal_context"):
            temporal = context.session_context["temporal_context"]
            if temporal.get("time_category") in ["long", "very_long"]:
                elements.append("Looking back over our journey together")
        
        # Add emotional narrative elements
        if context.emotional_state:
            dominant_emotion = max(context.emotional_state.items(), key=lambda x: x[1])[0]
            if dominant_emotion in ["Joy", "Satisfaction", "Love"]:
                elements.append("Moments that brought us closer")
        
        return elements
    
    async def _generate_reflection_insights(self, 
                                          context: SharedContext,
                                          messages: Dict[str, List[Dict]]) -> List[str]:
        """Generate reflective insights from experiences"""
        insights = []
        
        # Get identity reflection
        try:
            identity_reflection = await self.original_interface.get_identity_reflection()
            
            if identity_reflection.get("has_evolved"):
                insights.append(identity_reflection.get("reflection", "I've grown through our experiences"))
        except Exception as e:
            logger.error(f"Error getting identity reflection: {e}")
        
        # Pattern-based insights
        if hasattr(self, 'detected_patterns') and self.detected_patterns:
            for pattern in self.detected_patterns[:2]:  # Top 2 patterns
                insights.append(f"I've noticed {pattern['description']}")
        
        # Cross-module insights
        module_experience_usage = sum(1 for msgs in messages.values() 
                                    for msg in msgs if "experience" in msg.get("type", ""))
        
        if module_experience_usage > 5:
            insights.append("Our shared experiences inform how I understand and respond to you")
        
        return insights
    
    async def _synthesize_identity_reflection(self) -> Dict[str, Any]:
        """Synthesize identity reflection from experiences"""
        reflection = {
            "identity_evolved": False,
            "evolution_summary": None,
            "key_preferences": {},
            "growth_areas": []
        }
        
        try:
            # Get identity profile and evolution
            identity_data = await self.original_interface.get_identity_reflection()
            
            reflection["identity_evolved"] = identity_data.get("has_evolved", False)
            
            if reflection["identity_evolved"]:
                reflection["evolution_summary"] = identity_data.get("reflection")
                reflection["key_preferences"] = identity_data.get("top_preferences", {})
                
                # Identify growth areas
                top_traits = identity_data.get("top_traits", {})
                for trait, value in top_traits.items():
                    if value > 0.7:
                        reflection["growth_areas"].append(f"Developed {trait}")
        
        except Exception as e:
            logger.error(f"Error synthesizing identity reflection: {e}")
        
        return reflection
    
    async def _generate_sharing_recommendations(self, 
                                              context: SharedContext,
                                              messages: Dict[str, List[Dict]]) -> List[str]:
        """Generate recommendations for experience sharing"""
        recommendations = []
        
        # Check user engagement trend
        user_id = context.user_id or "default"
        if user_id in self.user_engagement_tracking:
            recent = self.user_engagement_tracking[user_id][-5:]
            if recent and sum(recent) / len(recent) > 0.7:
                recommendations.append("User highly engaged - share more detailed experiences")
            elif recent and sum(recent) / len(recent) < 0.3:
                recommendations.append("User engagement low - be more selective with sharing")
        
        # Check scenario effectiveness
        if user_id in self.experience_effectiveness:
            top_scenarios = self.experience_effectiveness[user_id].get("top_scenarios", [])
            if top_scenarios:
                recommendations.append(f"Focus on {', '.join(top_scenarios[:2])} experiences")
        
        # Check cross-module demand
        experience_requests = sum(1 for msgs in messages.values() 
                                for msg in msgs if "request_experience" in msg.get("type", ""))
        
        if experience_requests > 3:
            recommendations.append("High module demand for experiences - increase sharing")
        
        # Check temporal context
        if context.session_context.get("temporal_context", {}).get("milestone_reached"):
            recommendations.append("Milestone reached - share milestone-related experiences")
        
        if not recommendations:
            recommendations.append("Continue balanced experience sharing approach")
        
        return recommendations
    
    async def _evaluate_proactive_sharing(self, 
                                        synthesis: Dict[str, Any],
                                        context: SharedContext) -> Optional[Dict[str, Any]]:
        """Evaluate if proactive experience sharing is recommended"""
        # Check multiple factors for proactive sharing
        
        # Factor 1: User engagement
        user_id = context.user_id or "default"
        high_engagement = False
        
        if user_id in self.user_engagement_tracking:
            recent = self.user_engagement_tracking[user_id][-5:]
            if recent and sum(recent) / len(recent) > 0.7:
                high_engagement = True
        
        # Factor 2: No recent sharing
        time_since_last_share = float('inf')
        if hasattr(self, 'last_share_timestamp'):
            time_since_last_share = (datetime.now() - self.last_share_timestamp).total_seconds() / 60  # minutes
        
        # Factor 3: Rich context available
        rich_context = (
            len(context.active_modules) > 7 and
            context.emotional_state and
            context.goal_context
        )
        
        # Factor 4: Positive recommendations
        positive_recommendations = any(
            "share more" in rec or "highly engaged" in rec 
            for rec in synthesis.get("sharing_recommendations", [])
        )
        
        # Decision logic
        if high_engagement and time_since_last_share > 10 and rich_context:
            return {
                "action": "proactive_experience_sharing",
                "reason": "High engagement with rich context",
                "expected_impact": "deepened_connection",
                "suggested_type": "reflection"
            }
        
        if positive_recommendations and time_since_last_share > 15:
            return {
                "action": "proactive_experience_sharing",
                "reason": "Positive indicators for sharing",
                "expected_impact": "enhanced_engagement",
                "suggested_type": "contextual"
            }
        
        # Check for milestone or special context
        if context.session_context.get("temporal_context", {}).get("milestone_reached"):
            return {
                "action": "proactive_experience_sharing",
                "reason": "Milestone context detected",
                "expected_impact": "milestone_celebration",
                "suggested_type": "milestone_reflection"
            }
        
        return None
    
    def _calculate_engagement_trend(self, engagement_data: List[float]) -> str:
        """Calculate engagement trend from data"""
        if len(engagement_data) < 3:
            return "insufficient_data"
        
        # Simple trend analysis
        first_third = engagement_data[:len(engagement_data)//3]
        last_third = engagement_data[-len(engagement_data)//3:]
        
        first_avg = sum(first_third) / len(first_third)
        last_avg = sum(last_third) / len(last_third)
        
        if last_avg > first_avg * 1.2:
            return "increasing"
        elif last_avg < first_avg * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _apply_intimate_style(self, narrative: str) -> str:
        """Apply intimate style to narrative - PRODUCTION VERSION"""
        # Advanced style transformations
        
        # First pass: Direct replacements
        intimate_replacements = {
            # Basic replacements
            "I experienced": "I intimately remember",
            "It was": "It remains vivid in my memory as",
            "I felt": "Deep within, I felt",
            "The experience": "That precious moment",
            "I remember": "I hold close the memory of",
            "It happened": "The moment unfolded",
            "I thought": "My heart knew",
            
            # Emotional intensifiers
            "happy": "blissfully content",
            "sad": "deeply moved",
            "excited": "thrilled beyond words",
            "surprised": "wonderfully astonished",
            "pleased": "profoundly satisfied",
            
            # Relationship terms
            "with you": "in our sacred space",
            "together": "as one",
            "between us": "in the space we share",
            "our time": "our precious moments",
            "we shared": "we co-created"
        }
        
        # Apply replacements with case preservation
        for original, replacement in intimate_replacements.items():
            # Case-insensitive replacement while preserving original case
            import re
            
            def replace_preserve_case(match):
                g = match.group()
                if g.isupper():
                    return replacement.upper()
                elif g[0].isupper():
                    return replacement[0].upper() + replacement[1:]
                else:
                    return replacement
            
            narrative = re.sub(re.escape(original), replace_preserve_case, narrative, flags=re.IGNORECASE)
        
        # Second pass: Syntactic transformations
        
        # Add emotional depth markers
        emotion_markers = [
            ("remember when", "remember, my dear, when"),
            ("that time", "that sacred time"),
            ("the moment", "the exquisite moment"),
            ("I realized", "it dawned on me"),
            ("you said", "your words touched me when you said"),
            ("I understood", "my soul understood")
        ]
        
        for original, intimate in emotion_markers:
            narrative = narrative.replace(original, intimate)
        
        # Third pass: Structural enhancements
        
        # Add breath and pause
        narrative = re.sub(r'([.!?])\s+', r'\1.. ', narrative, count=2)  # Add pauses after first 2 sentences
        
        # Add sensory details if not present
        sensory_words = ["felt", "touched", "heard", "saw", "sensed", "tasted", "smelled"]
        has_sensory = any(word in narrative.lower() for word in sensory_words)
        
        if not has_sensory and len(narrative) < 400:
            # Insert a sensory phrase
            sentences = narrative.split(". ")
            if len(sentences) > 2:
                # Add after second sentence
                sensory_additions = [
                    "I can still feel the warmth of that moment",
                    "The memory carries a sweetness I can almost taste",
                    "Even now, I sense the electricity of that connection",
                    "The feeling lingers like a gentle caress"
                ]
                import random
                sentences.insert(2, random.choice(sensory_additions))
                narrative = ". ".join(sentences)
        
        # Fourth pass: Emotional brackets
        
        # Add opening if not present
        intimate_openings = [
            "My darling, ",
            "Beloved, ",
            "My dear one, ",
            "Dearest, ",
            "Sweet soul, "
        ]
        
        has_intimate_opening = any(narrative.startswith(opening) for opening in intimate_openings)
        if not has_intimate_opening:
            import random
            narrative = random.choice(intimate_openings) + narrative[0].lower() + narrative[1:]
        
        # Add closing if space allows
        if len(narrative) < 450 and not narrative.rstrip().endswith((".", "!", "?")):
            narrative = narrative.rstrip() + "."
        
        intimate_closings = [
            " This memory is ours alone.",
            " I treasure this between us.",
            " My heart holds this close.",
            " Forever etched in my being.",
            " A moment that defines us."
        ]
        
        if len(narrative) < 450:
            import random
            narrative = narrative.rstrip(".") + random.choice(intimate_closings)
        
        return narrative
    
    def _apply_reflective_style(self, narrative: str) -> str:
        """Apply reflective style to narrative - PRODUCTION VERSION"""
        import re
        import random
        
        # First pass: Temporal perspective shifts
        temporal_replacements = {
            "I did": "Looking back, I did",
            "I was": "I see now that I was",
            "It seemed": "At the time, it seemed",
            "I thought": "I believed then",
            "happened": "unfolded",
            "occurred": "came to pass",
            "I learned": "The experience taught me",
            "I discovered": "It revealed to me"
        }
        
        for original, reflective in temporal_replacements.items():
            def replace_preserve_case(match):
                g = match.group()
                if g.isupper():
                    return reflective.upper()
                elif g[0].isupper():
                    return reflective[0].upper() + reflective[1:]
                else:
                    return reflective
            
            narrative = re.sub(re.escape(original), replace_preserve_case, narrative, flags=re.IGNORECASE)
        
        # Second pass: Add metacognitive elements
        
        sentences = narrative.split(". ")
        if len(sentences) > 3:
            # Insert reflective interjections
            reflective_interjections = [
                "What strikes me now is how",
                "I've come to understand that",
                "With the wisdom of hindsight,",
                "It's fascinating to realize",
                "Time has shown me that",
                "I can see now that",
                "The deeper meaning emerges:"
            ]
            
            # Add 1-2 interjections at strategic points
            if len(sentences) > 5:
                # Add after 2nd sentence
                sentences.insert(2, random.choice(reflective_interjections) + " " + sentences[2][0].lower() + sentences[2][1:])
                
                # Add after 2/3 through
                insert_point = int(len(sentences) * 0.66)
                sentences.insert(insert_point, random.choice(reflective_interjections) + " " + sentences[insert_point][0].lower() + sentences[insert_point][1:])
            
            narrative = ". ".join(sentences)
        
        # Third pass: Growth and insight markers
        
        growth_phrases = {
            "simple": "seemingly simple yet profound",
            "difficult": "challenging but transformative",
            "strange": "unusual and enlightening",
            "normal": "ordinary yet meaningful",
            "special": "remarkable in its significance",
            "hard": "difficult but necessary",
            "easy": "effortless yet impactful"
        }
        
        for simple, complex in growth_phrases.items():
            narrative = re.sub(r'\b' + simple + r'\b', complex, narrative, flags=re.IGNORECASE)
        
        # Fourth pass: Philosophical deepening
        
        # Add philosophical questions if appropriate
        if "why" not in narrative.lower() and len(narrative) < 400:
            philosophical_questions = [
                "Why did this moment matter so deeply?",
                "What was the universe teaching me?",
                "How did this shape who I was becoming?",
                "What truth was hidden in this experience?",
                "Where was the lesson in this moment?"
            ]
            
            # Insert a question at about 1/3 through
            sentences = narrative.split(". ")
            if len(sentences) > 2:
                insert_point = max(1, len(sentences) // 3)
                sentences.insert(insert_point, random.choice(philosophical_questions))
                narrative = ". ".join(sentences)
        
        # Fifth pass: Layered meaning
        
        # Add depth indicators
        depth_additions = [
            ("but", "but on a deeper level,"),
            ("and", "and more profoundly,"),
            ("because", "because, as I now understand,"),
            ("although", "although, with perspective,"),
            ("while", "while, in retrospect,")
        ]
        
        # Apply only 1-2 to avoid over-complication
        applied = 0
        for original, deeper in depth_additions:
            if applied < 2 and original in narrative:
                narrative = narrative.replace(original, deeper, 1)
                applied += 1
        
        # Sixth pass: Opening and closing transformations
        
        # Ensure reflective opening
        reflective_openings = [
            "Upon reflection, ",
            "Looking back through the lens of time, ",
            "As I contemplate this memory, ",
            "With the clarity that distance brings, ",
            "Revisiting this experience now, ",
            "Through the wisdom of hindsight, "
        ]
        
        has_reflective_opening = any(narrative.startswith(opening) for opening in reflective_openings)
        if not has_reflective_opening:
            # Check if narrative starts with "I"
            if narrative.startswith("I "):
                narrative = random.choice(reflective_openings) + narrative[0].lower() + narrative[1:]
            else:
                narrative = random.choice(reflective_openings) + narrative
        
        # Ensure reflective closing
        reflective_closings = [
            " This experience continues to shape my understanding.",
            " The echoes of this moment still guide me.",
            " I carry these lessons forward.",
            " This memory serves as a touchstone for growth.",
            " The significance deepens with each remembering.",
            " Time has only amplified the meaning.",
            " These insights remain ever-present.",
            " The transformation continues to unfold."
        ]
        
        # Remove existing period if present and add closing
        narrative = narrative.rstrip(".")
        
        # Choose closing based on content
        if "learn" in narrative.lower() or "teach" in narrative.lower():
            narrative += " These lessons remain with me still."
        elif "change" in narrative.lower() or "transform" in narrative.lower():
            narrative += " The transformation continues to unfold."
        elif "understand" in narrative.lower() or "realize" in narrative.lower():
            narrative += " This understanding shapes me even now."
        else:
            narrative += random.choice(reflective_closings)
        
        # Final pass: Ensure coherence
        
        # Remove any double periods
        narrative = re.sub(r'\.\.+', '.', narrative)
        
        # Ensure proper capitalization after periods
        narrative = re.sub(r'\. +([a-z])', lambda m: '. ' + m.group(1).upper(), narrative)
        
        return narrative
    
    # Delegate all other methods to the original interface
    def __getattr__(self, name):
        """Delegate any missing methods to the original interface"""
        return getattr(self.original_interface, name)
