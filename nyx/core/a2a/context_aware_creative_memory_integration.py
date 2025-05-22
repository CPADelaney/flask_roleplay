# nyx/core/a2a/context_aware_creative_memory_integration.py

import logging
import random
import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareCreativeMemoryIntegration(ContextAwareModule):
    """
    Advanced CreativeMemoryIntegration with full context distribution capabilities
    """
    
    def __init__(self, original_integration):
        super().__init__("creative_memory_integration")
        self.original = original_integration
        self.context_subscriptions = [
            "memory_retrieval_complete", "emotional_state_update", "goal_context_available",
            "attention_updated", "relationship_state_change", "conversation_flow_analysis",
            "user_engagement_detected", "novelty_opportunity", "recognition_triggered",
            "mood_state_update", "cognitive_load_update", "social_context_update"
        ]
        
        # Inherit original context and state
        self.context = original_integration.context
        self.novelty_engine = original_integration.context.novelty_engine
        self.recognition_memory = original_integration.context.recognition_memory
        self.memory_core = original_integration.context.memory_core
        
        # Enhanced tracking for context-aware creativity
        self.creativity_state = {
            "current_wit_probability": 0.3,
            "current_insight_threshold": 0.7,
            "creativity_momentum": 0.5,
            "last_creative_output": None,
            "creative_cooldown": 0,
            "user_receptivity": 0.5,
            "contextual_appropriateness": 1.0
        }
        
        # Track creative patterns across conversations
        self.creative_patterns = defaultdict(lambda: {
            "wit_success_rate": 0.5,
            "insight_success_rate": 0.5,
            "preferred_wit_types": defaultdict(float),
            "preferred_insight_types": defaultdict(float),
            "engagement_correlation": 0.0
        })
    
    async def on_context_received(self, context: SharedContext):
        """Initialize creative processing for this context"""
        logger.debug(f"CreativeMemoryIntegration received context for user: {context.user_id}")
        
        # Assess creative potential for this context
        creative_assessment = await self._assess_contextual_creative_potential(context)
        
        # Initialize user-specific creative parameters
        user_pattern = self.creative_patterns[context.user_id]
        
        # Send initial creative context to other modules
        await self.send_context_update(
            update_type="creative_context_initialized",
            data={
                "wit_probability": creative_assessment["wit_probability"],
                "insight_potential": creative_assessment["insight_potential"],
                "creative_readiness": creative_assessment["overall_readiness"],
                "user_creative_preferences": {
                    "wit_types": dict(user_pattern["preferred_wit_types"]),
                    "insight_types": dict(user_pattern["preferred_insight_types"]),
                    "historical_success": {
                        "wit": user_pattern["wit_success_rate"],
                        "insight": user_pattern["insight_success_rate"]
                    }
                },
                "creativity_factors": creative_assessment["factors"]
            },
            priority=ContextPriority.NORMAL
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules affecting creativity"""
        
        if update.update_type == "memory_retrieval_complete":
            # Memories trigger creative opportunities
            memory_data = update.data
            memories = memory_data.get("retrieved_memories", [])
            
            if memories:
                creative_opportunity = await self._evaluate_memory_creative_potential(memories, update.source_module)
                
                if creative_opportunity["has_potential"]:
                    await self._process_creative_memory_opportunity(memories, creative_opportunity)
        
        elif update.update_type == "emotional_state_update":
            # Emotions strongly affect creativity
            emotional_data = update.data
            await self._adjust_creativity_from_emotion(emotional_data)
        
        elif update.update_type == "user_engagement_detected":
            # User engagement affects creative parameters
            engagement_data = update.data
            engagement_level = engagement_data.get("engagement_level", 0.5)
            engagement_type = engagement_data.get("engagement_type", "neutral")
            
            await self._process_engagement_feedback(engagement_level, engagement_type)
        
        elif update.update_type == "relationship_state_change":
            # Relationship affects creative boundaries
            relationship_data = update.data
            await self._adjust_creative_boundaries(relationship_data)
        
        elif update.update_type == "conversation_flow_analysis":
            # Conversation flow affects creative timing
            flow_data = update.data
            await self._optimize_creative_timing(flow_data)
        
        elif update.update_type == "cognitive_load_update":
            # High cognitive load reduces creative capacity
            load_data = update.data
            cognitive_load = load_data.get("load_level", 0.5)
            
            await self._adjust_creative_complexity(cognitive_load)
        
        elif update.update_type == "social_context_update":
            # Social context affects appropriateness
            social_data = update.data
            await self._update_social_appropriateness(social_data)
        
        elif update.update_type == "recognition_triggered":
            # Recognition events are prime creative opportunities
            recognition_data = update.data
            await self._process_recognition_creative_opportunity(recognition_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with creative context awareness"""
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Extract creative cues from input and context
        creative_cues = await self._extract_comprehensive_creative_cues(context, messages)
        
        # Determine if creativity is appropriate
        creativity_appropriate = await self._assess_creativity_appropriateness(context, messages)
        
        # Process through original system if appropriate
        creative_output = None
        if creativity_appropriate["is_appropriate"]:
            # Use original processing with enhanced parameters
            self.context.wit_generation_probability = self.creativity_state["current_wit_probability"]
            self.context.insight_threshold = self.creativity_state["current_insight_threshold"]
            
            result = await self.original.process_conversation_turn(
                context.user_input,
                context.session_context
            )
            
            if result.get("status") in ["wit_generated", "insight_generated", "blend_generated"]:
                creative_output = result
                
                # Track output for momentum and cooldown
                self.creativity_state["last_creative_output"] = datetime.datetime.now()
                self.creativity_state["creative_cooldown"] = 3  # Avoid rapid-fire creativity
                
                # Update momentum
                self.creativity_state["creativity_momentum"] = min(
                    1.0, 
                    self.creativity_state["creativity_momentum"] + 0.1
                )
        
        # Send update about creative processing
        await self.send_context_update(
            update_type="creative_processing_complete",
            data={
                "creative_output_generated": creative_output is not None,
                "output_type": creative_output.get("content_type") if creative_output else None,
                "creative_cues_found": len(creative_cues),
                "appropriateness_score": creativity_appropriate["score"],
                "creativity_momentum": self.creativity_state["creativity_momentum"]
            }
        )
        
        return {
            "creative_processing": True,
            "creative_output": creative_output,
            "creative_cues": creative_cues,
            "appropriateness_assessment": creativity_appropriate,
            "creativity_state": dict(self.creativity_state),
            "context_integrated": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze creative patterns and opportunities"""
        # Analyze conversation for creative patterns
        pattern_analysis = await self._analyze_conversation_creative_patterns(context)
        
        # Analyze user's creative preferences
        preference_analysis = await self._analyze_user_creative_preferences(context)
        
        # Identify missed creative opportunities
        missed_opportunities = await self._identify_missed_creative_opportunities(context)
        
        # Analyze creative effectiveness
        effectiveness_analysis = await self._analyze_creative_effectiveness(context)
        
        return {
            "pattern_analysis": pattern_analysis,
            "preference_analysis": preference_analysis,
            "missed_opportunities": missed_opportunities,
            "effectiveness_analysis": effectiveness_analysis,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize creative elements for response"""
        messages = await self.get_cross_module_messages()
        
        # Determine if response needs creative enhancement
        needs_creativity = await self._response_needs_creativity(context, messages)
        
        creative_synthesis = {
            "creative_enhancements": [],
            "wit_integration": None,
            "insight_offering": None,
            "creative_suggestions": [],
            "creativity_metadata": {}
        }
        
        if needs_creativity["needs_enhancement"]:
            # Generate appropriate creative elements
            if needs_creativity["wit_appropriate"]:
                wit = await self._generate_contextual_wit_for_response(context, messages)
                if wit:
                    creative_synthesis["wit_integration"] = wit
                    creative_synthesis["creative_enhancements"].append({
                        "type": "wit",
                        "content": wit["wit_text"],
                        "integration_point": wit.get("integration_point", "natural")
                    })
            
            if needs_creativity["insight_appropriate"]:
                insight = await self._generate_contextual_insight_for_response(context, messages)
                if insight:
                    creative_synthesis["insight_offering"] = insight
                    creative_synthesis["creative_enhancements"].append({
                        "type": "insight",
                        "content": insight["insight_text"],
                        "depth": insight.get("abstraction_level", 0.5)
                    })
            
            # Add creative suggestions for response style
            creative_synthesis["creative_suggestions"] = await self._generate_creative_style_suggestions(
                context, messages
            )
        
        # Add creativity metadata
        creative_synthesis["creativity_metadata"] = {
            "creativity_level": self._calculate_response_creativity_level(creative_synthesis),
            "user_receptivity": self.creativity_state["user_receptivity"],
            "contextual_appropriateness": self.creativity_state["contextual_appropriateness"],
            "creative_momentum": self.creativity_state["creativity_momentum"]
        }
        
        return creative_synthesis
    
    # ========================================================================================
    # DETAILED HELPER METHODS
    # ========================================================================================
    
    async def _assess_contextual_creative_potential(self, context: SharedContext) -> Dict[str, Any]:
        """Assess creative potential based on full context"""
        assessment = {
            "wit_probability": self.context.wit_generation_probability,
            "insight_potential": 0.5,
            "overall_readiness": 0.5,
            "factors": {}
        }
        
        # Emotional factors
        if context.emotional_state:
            emotional_creativity = await self._calculate_emotional_creativity_factor(context.emotional_state)
            assessment["factors"]["emotional"] = emotional_creativity
            
            # Positive emotions boost wit probability
            positive_emotions = ["Joy", "Excitement", "Curiosity", "Playfulness"]
            positive_strength = sum(
                context.emotional_state.get(e, 0) for e in positive_emotions
            ) / len(positive_emotions)
            
            assessment["wit_probability"] += positive_strength * 0.2
            
            # Curiosity boosts insight potential
            curiosity = context.emotional_state.get("Curiosity", 0)
            assessment["insight_potential"] += curiosity * 0.3
        
        # Goal context affects creativity
        if context.goal_context:
            goal_factor = await self._calculate_goal_creativity_factor(context.goal_context)
            assessment["factors"]["goal_alignment"] = goal_factor
            
            # Creative goals boost overall readiness
            creative_goals = [
                g for g in context.goal_context.get("active_goals", [])
                if any(word in g.get("description", "").lower() 
                      for word in ["creative", "novel", "explore", "discover"])
            ]
            if creative_goals:
                assessment["overall_readiness"] += 0.2
        
        # Relationship context affects creative boundaries
        if context.relationship_context:
            relationship_factor = await self._calculate_relationship_creativity_factor(
                context.relationship_context
            )
            assessment["factors"]["relationship"] = relationship_factor
            
            # Higher trust enables more creativity
            trust = context.relationship_context.get("trust", 0.5)
            assessment["wit_probability"] *= (0.7 + trust * 0.6)  # Scale from 0.7 to 1.3
        
        # Memory context provides creative material
        if context.memory_context:
            memory_richness = len(context.memory_context.get("relevant_memories", []))
            assessment["factors"]["memory_richness"] = min(1.0, memory_richness / 10)
            
            # Rich memories enable insights
            if memory_richness > 5:
                assessment["insight_potential"] += 0.2
        
        # Calculate overall readiness
        factor_values = list(assessment["factors"].values())
        if factor_values:
            assessment["overall_readiness"] = sum(factor_values) / len(factor_values)
        
        # Normalize values
        for key in ["wit_probability", "insight_potential", "overall_readiness"]:
            assessment[key] = max(0.0, min(1.0, assessment[key]))
        
        return assessment
    
    async def _calculate_emotional_creativity_factor(self, emotional_state: Dict[str, float]) -> float:
        """Calculate how emotions affect creativity"""
        if not emotional_state:
            return 0.5
        
        # Positive emotions enhance creativity
        creativity_enhancing = ["Joy", "Excitement", "Curiosity", "Playfulness", "Wonder"]
        enhancing_score = sum(emotional_state.get(e, 0) for e in creativity_enhancing)
        
        # Negative emotions can inhibit creativity
        creativity_inhibiting = ["Anxiety", "Fear", "Sadness", "Anger", "Frustration"]
        inhibiting_score = sum(emotional_state.get(e, 0) for e in creativity_inhibiting)
        
        # Calculate net effect
        creativity_factor = 0.5 + (enhancing_score * 0.3) - (inhibiting_score * 0.2)
        
        # However, mild negative emotions can sometimes enhance creativity
        if 0.2 < inhibiting_score < 0.5:
            creativity_factor += 0.1  # Mild tension can be creative
        
        return max(0.0, min(1.0, creativity_factor))
    
    async def _calculate_goal_creativity_factor(self, goal_context: Dict[str, Any]) -> float:
        """Calculate how goals affect creativity"""
        active_goals = goal_context.get("active_goals", [])
        
        if not active_goals:
            return 0.5
        
        # Check for creativity-compatible goals
        creative_keywords = ["explore", "discover", "create", "novel", "express", "imagine", "play"]
        analytical_keywords = ["analyze", "solve", "calculate", "optimize", "measure", "precise"]
        
        creative_goal_count = 0
        analytical_goal_count = 0
        
        for goal in active_goals:
            description = goal.get("description", "").lower()
            
            if any(keyword in description for keyword in creative_keywords):
                creative_goal_count += goal.get("priority", 0.5)
            if any(keyword in description for keyword in analytical_keywords):
                analytical_goal_count += goal.get("priority", 0.5)
        
        # Creative goals boost creativity, analytical goals reduce it
        factor = 0.5 + (creative_goal_count * 0.3) - (analytical_goal_count * 0.2)
        
        return max(0.0, min(1.0, factor))
    
    async def _calculate_relationship_creativity_factor(self, relationship_context: Dict[str, Any]) -> float:
        """Calculate how relationship affects creativity"""
        trust = relationship_context.get("trust", 0.5)
        intimacy = relationship_context.get("intimacy", 0.5)
        interaction_count = relationship_context.get("interaction_count", 0)
        
        # New relationships may be more formal
        if interaction_count < 5:
            base_factor = 0.3
        else:
            base_factor = 0.5
        
        # Trust enables creative expression
        trust_boost = trust * 0.3
        
        # Intimacy enables playfulness
        intimacy_boost = intimacy * 0.2
        
        return min(1.0, base_factor + trust_boost + intimacy_boost)
    
    async def _evaluate_memory_creative_potential(self, memories: List[Dict[str, Any]], 
                                                source_module: str) -> Dict[str, Any]:
        """Evaluate creative potential of retrieved memories"""
        if not memories:
            return {"has_potential": False}
        
        potential = {
            "has_potential": False,
            "wit_potential": 0.0,
            "insight_potential": 0.0,
            "best_technique": None,
            "memory_clusters": []
        }
        
        # Analyze memory characteristics
        memory_ages = []
        memory_types = defaultdict(int)
        emotional_memories = 0
        significant_memories = 0
        
        for memory in memories:
            # Track memory age
            if "timestamp" in memory.get("metadata", {}):
                memory_ages.append(memory["metadata"]["timestamp"])
            
            # Track types
            memory_types[memory.get("memory_type", "unknown")] += 1
            
            # Track emotional content
            if memory.get("emotional_context"):
                emotional_memories += 1
            
            # Track significance
            if memory.get("significance", 5) > 7:
                significant_memories += 1
        
        # Evaluate wit potential
        # Good for wit: mix of types, some emotional content, moderate significance
        if len(memory_types) > 1 and emotional_memories > 0:
            potential["wit_potential"] = min(1.0, 0.3 + len(memories) * 0.1 + emotional_memories * 0.15)
        
        # Evaluate insight potential
        # Good for insights: multiple significant memories, patterns across time
        if significant_memories >= 2 or len(memories) >= 5:
            potential["insight_potential"] = min(1.0, 0.4 + significant_memories * 0.2)
        
        # Determine best technique
        if potential["wit_potential"] > potential["insight_potential"]:
            potential["best_technique"] = "contextual_wit"
        elif potential["insight_potential"] > 0.5:
            potential["best_technique"] = "creative_insight"
        else:
            potential["best_technique"] = "memory_blend"
        
        # Cluster memories by theme
        potential["memory_clusters"] = await self._cluster_memories_by_theme(memories)
        
        potential["has_potential"] = (
            potential["wit_potential"] > 0.3 or 
            potential["insight_potential"] > 0.3
        )
        
        return potential
    
    async def _cluster_memories_by_theme(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster memories by thematic similarity"""
        clusters = []
        
        # Simple clustering by tags and keywords
        tag_clusters = defaultdict(list)
        
        for memory in memories:
            # Use tags for clustering
            tags = memory.get("tags", [])
            if tags:
                primary_tag = tags[0]  # Use first tag as primary cluster
                tag_clusters[primary_tag].append(memory)
            else:
                tag_clusters["untagged"].append(memory)
        
        # Convert to cluster format
        for tag, cluster_memories in tag_clusters.items():
            if len(cluster_memories) >= 2:  # Only keep clusters with multiple memories
                clusters.append({
                    "theme": tag,
                    "memories": cluster_memories,
                    "size": len(cluster_memories),
                    "coherence": 0.7  # Simple coherence estimate
                })
        
        return clusters
    
    async def _process_creative_memory_opportunity(self, memories: List[Dict[str, Any]], 
                                                 opportunity: Dict[str, Any]):
        """Process identified creative opportunity with memories"""
        technique = opportunity["best_technique"]
        
        # Check cooldown
        if self.creativity_state["creative_cooldown"] > 0:
            return
        
        # Check appropriateness
        if self.creativity_state["contextual_appropriateness"] < 0.5:
            return
        
        # Generate creative content based on technique
        creative_result = None
        
        if technique == "contextual_wit" and opportunity["wit_potential"] > 0.4:
            # Select best memory for wit
            wit_memory = await self._select_memory_for_wit(memories)
            if wit_memory:
                creative_result = await self.original.generate_contextual_wit(
                    "current context", wit_memory
                )
        
        elif technique == "creative_insight" and opportunity["insight_potential"] > 0.5:
            # Use memory clusters for insights
            if opportunity["memory_clusters"]:
                cluster = opportunity["memory_clusters"][0]  # Use largest cluster
                creative_result = await self.original.generate_creative_insight(
                    "current context", cluster["memories"]
                )
        
        elif technique == "memory_blend":
            # Blend memory with novelty
            blend_memory = random.choice(memories)
            creative_result = await self.original.blend_memory_with_novelty(
                "current context", blend_memory
            )
        
        if creative_result:
            # Send update about generated creativity
            await self.send_context_update(
                update_type="creative_content_from_memory",
                data={
                    "content_type": technique,
                    "content": creative_result,
                    "source_memory_count": len(memories),
                    "creative_confidence": opportunity.get(f"{technique}_potential", 0.5)
                },
                priority=ContextPriority.NORMAL
            )
    
    async def _select_memory_for_wit(self, memories: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select best memory for wit generation"""
        if not memories:
            return None
        
        # Score memories for wit potential
        scored_memories = []
        
        for memory in memories:
            score = 0.0
            
            # Emotional memories are good for wit
            if memory.get("emotional_context"):
                score += 0.3
            
            # Moderate significance is better than extreme
            significance = memory.get("significance", 5) / 10.0
            if 0.4 < significance < 0.7:
                score += 0.3
            else:
                score += 0.1
            
            # Recent memories may be more relevant
            if "timestamp" in memory.get("metadata", {}):
                # Simple recency check
                score += 0.2
            
            # Shorter memories are easier to reference wittily
            memory_length = len(memory.get("memory_text", ""))
            if memory_length < 100:
                score += 0.2
            
            scored_memories.append((memory, score))
        
        # Select highest scoring memory
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return scored_memories[0][0] if scored_memories else None
    
    async def _adjust_creativity_from_emotion(self, emotional_data: Dict[str, Any]):
        """Adjust creativity parameters based on emotional state"""
        emotional_state = emotional_data.get("emotional_state", {})
        arousal = emotional_data.get("arousal", 0.5)
        valence = emotional_data.get("valence", 0.0)
        
        # High positive emotions boost creativity
        if valence > 0.5 and arousal > 0.5:
            # Energetic positive state - increase wit probability
            self.creativity_state["current_wit_probability"] = min(
                0.7, 
                self.creativity_state["current_wit_probability"] * 1.3
            )
            
            # Lower insight threshold for sharing
            self.creativity_state["current_insight_threshold"] = max(
                0.5,
                self.creativity_state["current_insight_threshold"] * 0.8
            )
        
        # Low arousal positive is good for insights
        elif valence > 0.3 and arousal < 0.4:
            # Calm positive state - boost insight generation
            self.creativity_state["current_insight_threshold"] = max(
                0.4,
                self.creativity_state["current_insight_threshold"] * 0.7
            )
        
        # Negative emotions reduce wit but can enhance certain insights
        elif valence < -0.3:
            # Reduce wit probability
            self.creativity_state["current_wit_probability"] *= 0.7
            
            # But melancholy can lead to profound insights
            if emotional_state.get("Sadness", 0) > 0.5:
                self.creativity_state["current_insight_threshold"] *= 0.9
        
        # Update momentum based on emotional energy
        emotional_energy = arousal * abs(valence)
        self.creativity_state["creativity_momentum"] = (
            self.creativity_state["creativity_momentum"] * 0.7 + 
            emotional_energy * 0.3
        )
    
    async def _process_engagement_feedback(self, engagement_level: float, engagement_type: str):
        """Process user engagement to refine creative parameters"""
        # Update user receptivity
        self.creativity_state["user_receptivity"] = (
            self.creativity_state["user_receptivity"] * 0.7 + 
            engagement_level * 0.3
        )
        
        # Track success/failure of recent creativity
        if self.creativity_state["last_creative_output"]:
            time_since_output = (
                datetime.datetime.now() - self.creativity_state["last_creative_output"]
            ).total_seconds()
            
            # If engagement spike follows creative output, mark success
            if time_since_output < 30 and engagement_level > 0.7:
                # Update user patterns
                user_pattern = self.creative_patterns[self.current_context.user_id]
                
                # Determine what type of creativity was used
                if "wit" in str(self.creativity_state.get("last_output_type", "")):
                    user_pattern["wit_success_rate"] = (
                        user_pattern["wit_success_rate"] * 0.9 + 0.1
                    )
                elif "insight" in str(self.creativity_state.get("last_output_type", "")):
                    user_pattern["insight_success_rate"] = (
                        user_pattern["insight_success_rate"] * 0.9 + 0.1
                    )
        
        # Adjust parameters based on engagement type
        if engagement_type == "amused" or engagement_type == "delighted":
            # User enjoys humor - increase wit
            self.creativity_state["current_wit_probability"] = min(
                0.6,
                self.creativity_state["current_wit_probability"] * 1.2
            )
        elif engagement_type == "thoughtful" or engagement_type == "intrigued":
            # User appreciates insights
            self.creativity_state["current_insight_threshold"] = max(
                0.5,
                self.creativity_state["current_insight_threshold"] * 0.85
            )
        elif engagement_type == "confused" or engagement_type == "disconnected":
            # Reduce creative complexity
            self.creativity_state["current_wit_probability"] *= 0.8
            self.creativity_state["current_insight_threshold"] = min(
                0.8,
                self.creativity_state["current_insight_threshold"] * 1.1
            )
    
    async def _adjust_creative_boundaries(self, relationship_data: Dict[str, Any]):
        """Adjust creative expression based on relationship"""
        trust = relationship_data.get("trust", 0.5)
        intimacy = relationship_data.get("intimacy", 0.5)
        formality = relationship_data.get("formality", 0.5)
        interaction_count = relationship_data.get("interaction_count", 0)
        
        # High trust enables more creative risks
        if trust > 0.7:
            # Can be more playful and experimental
            self.context.wit_types.extend(["absurdist", "meta_humor"])
        elif trust < 0.3:
            # Be more conservative
            self.context.wit_types = ["analogy", "wordplay"]
        
        # Intimacy affects emotional creativity
        if intimacy > 0.6:
            # Can share deeper insights
            self.context.insight_types.extend(["vulnerability", "personal_growth"])
        
        # Formality affects wit style
        if formality > 0.7:
            # More sophisticated humor
            preferred_wit = ["wordplay", "literary_reference", "subtle_irony"]
            self.context.wit_types = [w for w in preferred_wit if w in self.context.wit_types]
        elif formality < 0.3:
            # More casual and playful
            self.context.wit_types.extend(["puns", "playful_teasing"])
        
        # Early interactions are more conservative
        if interaction_count < 10:
            self.creativity_state["current_wit_probability"] = min(
                0.4,
                self.creativity_state["current_wit_probability"]
            )
    
    async def _optimize_creative_timing(self, flow_data: Dict[str, Any]):
        """Optimize when to inject creativity based on conversation flow"""
        pace = flow_data.get("conversation_pace", "moderate")
        turn_count = flow_data.get("turn_count", 0)
        topic_shifts = flow_data.get("topic_shifts", 0)
        current_depth = flow_data.get("current_depth", "surface")
        
        # Reduce creativity if conversation is rapid
        if pace == "rapid":
            self.creativity_state["creative_cooldown"] = max(
                5,
                self.creativity_state["creative_cooldown"]
            )
        
        # Increase insight potential during deep conversations
        if current_depth == "deep":
            self.creativity_state["current_insight_threshold"] *= 0.8
        
        # After topic shifts, wait before being creative
        if topic_shifts > 0:
            self.creativity_state["creative_cooldown"] = 2
        
        # Build momentum in longer conversations
        if turn_count > 10:
            self.creativity_state["creativity_momentum"] = min(
                0.8,
                self.creativity_state["creativity_momentum"] + 0.05
            )
    
    async def _adjust_creative_complexity(self, cognitive_load: float):
        """Adjust creative complexity based on cognitive load"""
        # High cognitive load requires simpler creativity
        if cognitive_load > 0.7:
            # Prefer simple, clear wit over complex wordplay
            self.context.wit_types = ["simple_observation", "light_humor"]
            
            # Increase insight threshold to only share clearest insights
            self.creativity_state["current_insight_threshold"] = min(
                0.9,
                self.creativity_state["current_insight_threshold"] * 1.2
            )
        
        elif cognitive_load < 0.3:
            # Low load allows for more complex creativity
            self.context.wit_types = self.original.context.wit_types  # Full range
            
            # Can handle more abstract insights
            self.creativity_state["current_insight_threshold"] = max(
                0.5,
                self.creativity_state["current_insight_threshold"] * 0.9
            )
        
        # Adjust cooldown based on load
        if cognitive_load > 0.8:
            self.creativity_state["creative_cooldown"] = 5
    
    async def _update_social_appropriateness(self, social_data: Dict[str, Any]):
        """Update creative appropriateness based on social context"""
        context_type = social_data.get("context_type", "casual")
        audience_size = social_data.get("audience_size", 1)
        social_dynamics = social_data.get("dynamics", {})
        
        # Default appropriateness
        appropriateness = 1.0
        
        # Formal contexts reduce creativity
        if context_type in ["professional", "formal", "serious"]:
            appropriateness *= 0.5
            self.creativity_state["current_wit_probability"] *= 0.5
        
        # Large audiences require more universal creativity
        if audience_size > 3:
            appropriateness *= 0.8
            # Avoid inside jokes or highly contextual wit
            self.context.wit_types = ["universal_humor", "observational"]
        
        # Sensitive dynamics require care
        if social_dynamics.get("tension", 0) > 0.5:
            appropriateness *= 0.6
        
        if social_dynamics.get("vulnerability", 0) > 0.6:
            # Shift to supportive creativity
            self.context.insight_types = ["empathetic", "supportive", "validating"]
            appropriateness *= 0.9  # Still appropriate but different style
        
        self.creativity_state["contextual_appropriateness"] = appropriateness
    
    async def _process_recognition_creative_opportunity(self, recognition_data: Dict[str, Any]):
        """Process creative opportunities from recognition events"""
        recognized_pattern = recognition_data.get("pattern")
        confidence = recognition_data.get("confidence", 0.5)
        recognition_type = recognition_data.get("type", "general")
        
        # High-confidence recognitions are great for creativity
        if confidence > 0.7:
            # Boost creative probability temporarily
            self.creativity_state["current_wit_probability"] = min(
                0.8,
                self.creativity_state["current_wit_probability"] * 1.5
            )
            
            # Recognition callbacks are perfect for insights
            if recognition_type == "pattern" or recognition_type == "principle":
                self.creativity_state["current_insight_threshold"] *= 0.7
            
            # Send notification about opportunity
            await self.send_context_update(
                update_type="recognition_creative_opportunity",
                data={
                    "recognition_pattern": recognized_pattern,
                    "creative_potential": "high",
                    "suggested_approach": "pattern_based_insight" if recognition_type == "pattern" else "recognition_callback"
                },
                scope=ContextScope.GLOBAL
            )
    
    async def _extract_comprehensive_creative_cues(self, context: SharedContext, 
                                                 messages: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Extract all creative cues from context and messages"""
        creative_cues = []
        
        # Extract from user input
        input_cues = await self._extract_input_creative_cues(context.user_input)
        creative_cues.extend(input_cues)
        
        # Extract from emotional context
        if context.emotional_state:
            emotional_cues = await self._extract_emotional_creative_cues(context.emotional_state)
            creative_cues.extend(emotional_cues)
        
        # Extract from memories
        if context.memory_context:
            memory_cues = await self._extract_memory_creative_cues(context.memory_context)
            creative_cues.extend(memory_cues)
        
        # Extract from cross-module messages
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                if msg['type'] == 'creative_opportunity':
                    creative_cues.append({
                        "source": module_name,
                        "type": "explicit_opportunity",
                        "data": msg['data'],
                        "strength": msg['data'].get('strength', 0.5)
                    })
                elif msg['type'] == 'pattern_detected':
                    creative_cues.append({
                        "source": module_name,
                        "type": "pattern",
                        "pattern": msg['data'].get('pattern'),
                        "strength": 0.6
                    })
        
        # Score and sort cues
        for cue in creative_cues:
            cue['creative_potential'] = await self._score_creative_cue(cue, context)
        
        creative_cues.sort(key=lambda x: x['creative_potential'], reverse=True)
        
        return creative_cues
    
    async def _extract_input_creative_cues(self, user_input: str) -> List[Dict[str, Any]]:
        """Extract creative cues from user input"""
        cues = []
        input_lower = user_input.lower()
        
        # Humor indicators
        humor_words = ["funny", "joke", "laugh", "hilarious", "amusing", "witty", "humor"]
        if any(word in input_lower for word in humor_words):
            cues.append({
                "source": "user_input",
                "type": "humor_request",
                "explicit": True,
                "strength": 0.9
            })
        
        # Creativity indicators
        creative_words = ["creative", "imaginative", "unique", "different", "interesting"]
        if any(word in input_lower for word in creative_words):
            cues.append({
                "source": "user_input",
                "type": "creativity_request",
                "explicit": True,
                "strength": 0.8
            })
        
        # Insight indicators
        insight_words = ["why", "meaning", "understand", "insight", "deeper", "pattern"]
        if any(word in input_lower for word in insight_words):
            cues.append({
                "source": "user_input",
                "type": "insight_opportunity",
                "explicit": False,
                "strength": 0.6
            })
        
        # Playfulness indicators
        playful_patterns = ["what if", "imagine", "suppose", "wouldn't it be", "how about"]
        if any(pattern in input_lower for pattern in playful_patterns):
            cues.append({
                "source": "user_input",
                "type": "playful_invitation",
                "explicit": False,
                "strength": 0.7
            })
        
        return cues
    
    async def _extract_emotional_creative_cues(self, emotional_state: Dict[str, float]) -> List[Dict[str, Any]]:
        """Extract creative cues from emotional state"""
        cues = []
        
        # Joy and excitement are creative cues
        if emotional_state.get("Joy", 0) > 0.6:
            cues.append({
                "source": "emotional_state",
                "type": "positive_emotion",
                "emotion": "Joy",
                "strength": emotional_state["Joy"]
            })
        
        # Curiosity is a strong creative cue
        if emotional_state.get("Curiosity", 0) > 0.5:
            cues.append({
                "source": "emotional_state",
                "type": "curiosity_driven",
                "emotion": "Curiosity",
                "strength": emotional_state["Curiosity"] * 1.2  # Boost curiosity
            })
        
        # Playfulness directly triggers creativity
        if emotional_state.get("Playfulness", 0) > 0.4:
            cues.append({
                "source": "emotional_state",
                "type": "playful_mood",
                "emotion": "Playfulness",
                "strength": emotional_state["Playfulness"] * 1.5  # Strong boost
            })
        
        return cues
    
    async def _extract_memory_creative_cues(self, memory_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract creative cues from memory context"""
        cues = []
        memories = memory_context.get("relevant_memories", [])
        
        # Multiple memories suggest pattern
        if len(memories) > 3:
            cues.append({
                "source": "memory_context",
                "type": "memory_pattern",
                "memory_count": len(memories),
                "strength": min(1.0, len(memories) / 10)
            })
        
        # Emotional memories are creative fuel
        emotional_memories = [m for m in memories if m.get("emotional_context")]
        if emotional_memories:
            cues.append({
                "source": "memory_context",
                "type": "emotional_memory",
                "count": len(emotional_memories),
                "strength": min(1.0, len(emotional_memories) / 5)
            })
        
        # Old memories can trigger nostalgia-based creativity
        # This is simplified - would need proper date parsing
        old_memories = [m for m in memories if m.get("age_days", 0) > 30]
        if old_memories:
            cues.append({
                "source": "memory_context",
                "type": "nostalgic_memory",
                "count": len(old_memories),
                "strength": 0.6
            })
        
        return cues
    
    async def _score_creative_cue(self, cue: Dict[str, Any], context: SharedContext) -> float:
        """Score a creative cue for potential"""
        base_score = cue.get("strength", 0.5)
        
        # Explicit requests get priority
        if cue.get("explicit", False):
            base_score *= 1.5
        
        # Boost scores that align with user preferences
        user_pattern = self.creative_patterns.get(context.user_id, {})
        
        if cue["type"] == "humor_request" and user_pattern.get("wit_success_rate", 0.5) > 0.6:
            base_score *= 1.2
        elif cue["type"] == "insight_opportunity" and user_pattern.get("insight_success_rate", 0.5) > 0.6:
            base_score *= 1.2
        
        # Apply creativity momentum
        base_score *= (0.7 + self.creativity_state["creativity_momentum"] * 0.6)
        
        # Apply appropriateness filter
        base_score *= self.creativity_state["contextual_appropriateness"]
        
        return min(1.0, base_score)
    
    async def _assess_creativity_appropriateness(self, context: SharedContext, 
                                               messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Assess if creativity is appropriate in current context"""
        appropriateness = {
            "is_appropriate": True,
            "score": 1.0,
            "reasons": [],
            "warnings": []
        }
        
        # Check emotional appropriateness
        if context.emotional_state:
            # High negative emotions may make creativity inappropriate
            negative_emotions = ["Sadness", "Anger", "Fear", "Disgust"]
            negative_strength = sum(context.emotional_state.get(e, 0) for e in negative_emotions)
            
            if negative_strength > 2.0:  # Very negative
                appropriateness["score"] *= 0.3
                appropriateness["warnings"].append("High negative emotional state")
        
        # Check conversation seriousness
        serious_indicators = ["help", "problem", "urgent", "emergency", "serious", "important"]
        if any(word in context.user_input.lower() for word in serious_indicators):
            appropriateness["score"] *= 0.5
            appropriateness["warnings"].append("Serious conversation context")
        
        # Check cognitive load from messages
        for module_messages in messages.values():
            for msg in module_messages:
                if msg['type'] == 'cognitive_load_update':
                    load = msg['data'].get('load_level', 0.5)
                    if load > 0.8:
                        appropriateness["score"] *= 0.4
                        appropriateness["warnings"].append("High cognitive load")
        
        # Check cooldown
        if self.creativity_state["creative_cooldown"] > 0:
            appropriateness["score"] *= 0.2
            appropriateness["warnings"].append("Creative cooldown active")
            self.creativity_state["creative_cooldown"] -= 1
        
        # Check user receptivity
        if self.creativity_state["user_receptivity"] < 0.3:
            appropriateness["score"] *= 0.5
            appropriateness["warnings"].append("Low user receptivity")
        
        # Final decision
        appropriateness["is_appropriate"] = appropriateness["score"] > 0.4
        
        if appropriateness["is_appropriate"]:
            appropriateness["reasons"].append(f"Appropriateness score: {appropriateness['score']:.2f}")
        
        return appropriateness
    
    async def _analyze_conversation_creative_patterns(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze creative patterns in the conversation"""
        patterns = {
            "creative_density": 0.0,
            "wit_to_insight_ratio": 0.5,
            "creative_timing_pattern": "sparse",
            "user_response_pattern": "neutral",
            "effectiveness_trend": "stable"
        }
        
        # Analyze creative outputs in this conversation
        if hasattr(self, 'conversation_creative_history'):
            history = self.conversation_creative_history
            
            if history:
                # Calculate density
                conversation_length = getattr(self, 'conversation_turns', 1)
                patterns["creative_density"] = len(history) / max(1, conversation_length)
                
                # Calculate wit to insight ratio
                wit_count = sum(1 for h in history if h["type"] == "wit")
                insight_count = sum(1 for h in history if h["type"] == "insight")
                total = wit_count + insight_count
                
                if total > 0:
                    patterns["wit_to_insight_ratio"] = wit_count / total
                
                # Analyze timing
                if len(history) >= 3:
                    intervals = []
                    for i in range(1, len(history)):
                        interval = history[i]["turn"] - history[i-1]["turn"]
                        intervals.append(interval)
                    
                    avg_interval = sum(intervals) / len(intervals)
                    if avg_interval < 3:
                        patterns["creative_timing_pattern"] = "frequent"
                    elif avg_interval > 7:
                        patterns["creative_timing_pattern"] = "sparse"
                    else:
                        patterns["creative_timing_pattern"] = "moderate"
        
        return patterns
    
    async def _analyze_user_creative_preferences(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze user's creative preferences"""
        user_pattern = self.creative_patterns.get(context.user_id, {})
        
        # Determine preferred style
        preferred_style = "balanced"
        if user_pattern["wit_success_rate"] > user_pattern["insight_success_rate"] + 0.2:
            preferred_style = "wit_preferred"
        elif user_pattern["insight_success_rate"] > user_pattern["wit_success_rate"] + 0.2:
            preferred_style = "insight_preferred"
        
        # Analyze wit type preferences
        wit_preferences = dict(user_pattern["preferred_wit_types"])
        top_wit_types = sorted(wit_preferences.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Analyze insight type preferences
        insight_preferences = dict(user_pattern["preferred_insight_types"])
        top_insight_types = sorted(insight_preferences.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "preferred_style": preferred_style,
            "wit_success_rate": user_pattern["wit_success_rate"],
            "insight_success_rate": user_pattern["insight_success_rate"],
            "top_wit_types": top_wit_types,
            "top_insight_types": top_insight_types,
            "engagement_correlation": user_pattern["engagement_correlation"],
            "receptivity_score": self.creativity_state["user_receptivity"]
        }
    
    async def _identify_missed_creative_opportunities(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify missed opportunities for creativity"""
        missed = []
        
        # Check if there were memory patterns not used
        if context.memory_context:
            memories = context.memory_context.get("relevant_memories", [])
            if len(memories) > 5 and self.creativity_state.get("last_memory_creativity") != context.conversation_id:
                missed.append({
                    "type": "memory_pattern_unused",
                    "description": f"Had {len(memories)} memories but didn't generate insight",
                    "potential_value": 0.7
                })
        
        # Check if emotional state suggested creativity
        if context.emotional_state:
            positive_strength = sum(
                context.emotional_state.get(e, 0) 
                for e in ["Joy", "Excitement", "Playfulness", "Curiosity"]
            )
            if positive_strength > 1.5 and self.creativity_state["creative_cooldown"] > 0:
                missed.append({
                    "type": "positive_emotion_unused",
                    "description": "High positive emotion but creativity on cooldown",
                    "potential_value": 0.8
                })
        
        # Check if user explicitly requested creativity
        creativity_keywords = ["funny", "creative", "joke", "interesting", "unique"]
        if any(word in context.user_input.lower() for word in creativity_keywords):
            if not self.creativity_state.get("responded_to_request"):
                missed.append({
                    "type": "explicit_request_missed",
                    "description": "User requested creativity but none provided",
                    "potential_value": 0.9
                })
        
        return missed
    
    async def _analyze_creative_effectiveness(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze effectiveness of creative outputs"""
        effectiveness = {
            "overall_effectiveness": 0.5,
            "wit_effectiveness": 0.5,
            "insight_effectiveness": 0.5,
            "timing_effectiveness": 0.5,
            "relevance_effectiveness": 0.5
        }
        
        # Analyze based on user engagement patterns
        user_pattern = self.creative_patterns.get(context.user_id, {})
        
        effectiveness["wit_effectiveness"] = user_pattern.get("wit_success_rate", 0.5)
        effectiveness["insight_effectiveness"] = user_pattern.get("insight_success_rate", 0.5)
        
        # Timing effectiveness based on cooldown compliance
        if self.creativity_state["creative_cooldown"] == 0:
            effectiveness["timing_effectiveness"] = 0.8
        else:
            effectiveness["timing_effectiveness"] = 0.4
        
        # Relevance based on appropriateness
        effectiveness["relevance_effectiveness"] = self.creativity_state["contextual_appropriateness"]
        
        # Overall effectiveness
        effectiveness["overall_effectiveness"] = sum(
            effectiveness[k] for k in effectiveness if k != "overall_effectiveness"
        ) / 4
        
        return effectiveness
    
    async def _response_needs_creativity(self, context: SharedContext, 
                                       messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Determine if response needs creative enhancement"""
        needs = {
            "needs_enhancement": False,
            "wit_appropriate": False,
            "insight_appropriate": False,
            "reasons": []
        }
        
        # Check if conversation is getting stale
        conversation_energy = self.creativity_state.get("conversation_energy", 0.5)
        if conversation_energy < 0.3:
            needs["needs_enhancement"] = True
            needs["wit_appropriate"] = True
            needs["reasons"].append("Low conversation energy")
        
        # Check if user seems bored
        engagement_trending_down = False
        for module_messages in messages.values():
            for msg in module_messages:
                if msg['type'] == 'engagement_trend' and msg['data'].get('direction') == 'decreasing':
                    engagement_trending_down = True
                    break
        
        if engagement_trending_down:
            needs["needs_enhancement"] = True
            needs["wit_appropriate"] = True
            needs["reasons"].append("Decreasing engagement")
        
        # Check if there's a complex topic needing insight
        if len(context.user_input.split()) > 50 or "?" in context.user_input:
            needs["insight_appropriate"] = True
            needs["reasons"].append("Complex query benefits from insight")
        
        # Check if memories suggest creative opportunity
        if context.memory_context and len(context.memory_context.get("relevant_memories", [])) > 3:
            needs["insight_appropriate"] = True
            needs["reasons"].append("Multiple memories enable insight")
        
        # Check user preferences
        user_pattern = self.creative_patterns.get(context.user_id, {})
        if user_pattern.get("wit_success_rate", 0.5) > 0.7:
            needs["wit_appropriate"] = True
            needs["reasons"].append("User enjoys wit")
        
        # Final decision
        needs["needs_enhancement"] = needs["wit_appropriate"] or needs["insight_appropriate"]
        
        return needs
    
    async def _generate_contextual_wit_for_response(self, context: SharedContext, 
                                                  messages: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """Generate wit specifically for the response"""
        # Check if we have good material
        wit_material = await self._gather_wit_material(context, messages)
        
        if not wit_material:
            return None
        
        # Select wit type based on user preferences
        user_pattern = self.creative_patterns.get(context.user_id, {})
        preferred_types = dict(user_pattern.get("preferred_wit_types", {}))
        
        if preferred_types:
            wit_type = max(preferred_types.items(), key=lambda x: x[1])[0]
        else:
            wit_type = random.choice(self.context.wit_types)
        
        # Generate wit using material
        wit_text = await self._craft_wit(wit_material, wit_type, context)
        
        if wit_text:
            return {
                "wit_text": wit_text,
                "wit_type": wit_type,
                "confidence": self.creativity_state["user_receptivity"],
                "integration_point": "natural",  # How to integrate into response
                "timing": "responsive"  # When to deliver
            }
        
        return None
    
    async def _gather_wit_material(self, context: SharedContext, 
                                 messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Gather material for wit generation"""
        material = {
            "user_phrases": [],
            "interesting_words": [],
            "conceptual_elements": [],
            "emotional_elements": [],
            "contextual_elements": []
        }
        
        # Extract from user input
        words = context.user_input.split()
        material["user_phrases"] = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        material["interesting_words"] = [w for w in words if len(w) > 5]
        
        # Extract from emotional context
        if context.emotional_state:
            dominant = max(context.emotional_state.items(), key=lambda x: x[1])[0] if context.emotional_state else None
            if dominant:
                material["emotional_elements"].append(dominant)
        
        # Extract from memories
        if context.memory_context:
            for memory in context.memory_context.get("relevant_memories", [])[:3]:
                # Extract key phrases from memories
                memory_words = memory.get("memory_text", "").split()[:10]
                if memory_words:
                    material["contextual_elements"].append(" ".join(memory_words))
        
        # Check if we have enough material
        total_elements = sum(len(v) if isinstance(v, list) else 1 for v in material.values())
        
        return material if total_elements > 3 else None
    
    async def _craft_wit(self, material: Dict[str, Any], wit_type: str, 
                       context: SharedContext) -> Optional[str]:
        """Craft wit from material"""
        # This is simplified - in reality would use more sophisticated generation
        
        if wit_type == "wordplay":
            # Find words that can be played with
            interesting_words = material.get("interesting_words", [])
            if interesting_words:
                word = random.choice(interesting_words)
                return f"Speaking of '{word}', that reminds me of..."
        
        elif wit_type == "analogy":
            # Create analogy from elements
            if material["user_phrases"] and material["contextual_elements"]:
                user_element = random.choice(material["user_phrases"])
                context_element = random.choice(material["contextual_elements"])
                return f"That's like when {context_element}, but with {user_element}"
        
        elif wit_type == "callback":
            # Reference earlier content
            if material["contextual_elements"]:
                element = random.choice(material["contextual_elements"])
                return f"This brings us full circle to {element}"
        
        # Default fallback
        return None
    
    async def _generate_contextual_insight_for_response(self, context: SharedContext, 
                                                      messages: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """Generate insight specifically for the response"""
        # Gather insight material
        insight_material = await self._gather_insight_material(context, messages)
        
        if not insight_material["patterns"]:
            return None
        
        # Select insight type
        user_pattern = self.creative_patterns.get(context.user_id, {})
        preferred_types = dict(user_pattern.get("preferred_insight_types", {}))
        
        if preferred_types:
            insight_type = max(preferred_types.items(), key=lambda x: x[1])[0]
        else:
            insight_type = random.choice(self.context.insight_types)
        
        # Generate insight
        insight_text = await self._craft_insight(insight_material, insight_type, context)
        
        if insight_text:
            return {
                "insight_text": insight_text,
                "insight_type": insight_type,
                "abstraction_level": 0.7,  # How abstract the insight is
                "confidence": 0.8,
                "source": "pattern_recognition"
            }
        
        return None
    
    async def _gather_insight_material(self, context: SharedContext, 
                                     messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Gather material for insight generation"""
        material = {
            "patterns": [],
            "connections": [],
            "abstractions": [],
            "themes": []
        }
        
        # Look for patterns in memories
        if context.memory_context:
            memories = context.memory_context.get("relevant_memories", [])
            if len(memories) > 2:
                # Find common themes
                all_tags = []
                for memory in memories:
                    all_tags.extend(memory.get("tags", []))
                
                # Common tags indicate patterns
                tag_counts = defaultdict(int)
                for tag in all_tags:
                    tag_counts[tag] += 1
                
                patterns = [tag for tag, count in tag_counts.items() if count > 1]
                material["patterns"] = patterns
        
        # Look for conceptual connections
        if context.user_input:
            # Extract concepts (simplified)
            concepts = [w for w in context.user_input.split() if len(w) > 4]
            material["connections"] = concepts
        
        # Extract themes from conversation
        material["themes"] = ["growth", "understanding", "connection"]  # Simplified
        
        return material
    
    async def _craft_insight(self, material: Dict[str, Any], insight_type: str, 
                           context: SharedContext) -> Optional[str]:
        """Craft insight from material"""
        if insight_type == "pattern":
            if material["patterns"]:
                pattern = material["patterns"][0]
                return f"I notice a pattern here around {pattern} - it seems to be a recurring theme"
        
        elif insight_type == "principle":
            if material["connections"]:
                concept = material["connections"][0]
                return f"There's a principle at work here: when {concept} is involved, certain dynamics emerge"
        
        elif insight_type == "perspective":
            if material["themes"]:
                theme = random.choice(material["themes"])
                return f"From another perspective, this is really about {theme}"
        
        elif insight_type == "connection":
            if material["patterns"] and material["connections"]:
                pattern = material["patterns"][0]
                connection = material["connections"][0]
                return f"I see a connection between {pattern} and {connection}"
        
        return None
    
    async def _generate_creative_style_suggestions(self, context: SharedContext, 
                                                 messages: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Generate suggestions for creative response style"""
        suggestions = []
        
        # Base on emotional context
        if context.emotional_state:
            if context.emotional_state.get("Playfulness", 0) > 0.5:
                suggestions.append("playful_tone")
            if context.emotional_state.get("Curiosity", 0) > 0.6:
                suggestions.append("exploratory_style")
            if context.emotional_state.get("Joy", 0) > 0.7:
                suggestions.append("enthusiastic_delivery")
        
        # Base on user preferences
        user_pattern = self.creative_patterns.get(context.user_id, {})
        if user_pattern.get("wit_success_rate", 0.5) > 0.7:
            suggestions.append("humor_sprinkled")
        
        # Base on conversation depth
        if len(context.user_input.split()) > 30:
            suggestions.append("thoughtful_pacing")
        
        # Base on relationship
        if context.relationship_context:
            if context.relationship_context.get("intimacy", 0) > 0.7:
                suggestions.append("personal_anecdotes")
            if context.relationship_context.get("trust", 0) > 0.8:
                suggestions.append("vulnerable_sharing")
        
        return suggestions
    
    def _calculate_response_creativity_level(self, creative_synthesis: Dict[str, Any]) -> float:
        """Calculate overall creativity level of response"""
        level = 0.0
        
        # Count creative elements
        enhancements = creative_synthesis.get("creative_enhancements", [])
        level += len(enhancements) * 0.2
        
        # Wit adds creativity
        if creative_synthesis.get("wit_integration"):
            level += 0.3
        
        # Insights add depth
        if creative_synthesis.get("insight_offering"):
            level += 0.25
        
        # Style suggestions add flair
        suggestions = creative_synthesis.get("creative_suggestions", [])
        level += min(0.25, len(suggestions) * 0.05)
        
        return min(1.0, level)
    
    # Update cooldown when processing
    async def process_conversation_turn(self, conversation_text: str, 
                                      current_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Override to track cooldown"""
        # Decrement cooldown
        if self.creativity_state["creative_cooldown"] > 0:
            self.creativity_state["creative_cooldown"] -= 1
        
        # Track turn for history
        if not hasattr(self, 'conversation_turns'):
            self.conversation_turns = 0
        self.conversation_turns += 1
        
        # Process through original
        result = await self.original.process_conversation_turn(conversation_text, current_context)
        
        # Track creative output
        if result.get("status") in ["wit_generated", "insight_generated", "blend_generated"]:
            if not hasattr(self, 'conversation_creative_history'):
                self.conversation_creative_history = []
            
            self.conversation_creative_history.append({
                "turn": self.conversation_turns,
                "type": "wit" if "wit" in result["status"] else "insight",
                "output": result.get("generated_content")
            })
            
            # Update last output type for tracking
            self.creativity_state["last_output_type"] = result["content_type"]
        
        return result
    
    # Delegate all other methods to the original integration
    def __getattr__(self, name):
        """Delegate any missing methods to the original integration"""
        return getattr(self.original, name)
