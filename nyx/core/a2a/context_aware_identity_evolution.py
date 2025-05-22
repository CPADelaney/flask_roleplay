# nyx/core/a2a/context_aware_identity_evolution.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareIdentityEvolution(ContextAwareModule):
    """
    Enhanced IdentityEvolutionSystem with full context distribution capabilities
    """
    
    def __init__(self, original_identity_system):
        super().__init__("identity_evolution")
        self.original_system = original_identity_system
        self.context_subscriptions = [
            "significant_experience", "emotional_state_update", "relationship_milestone",
            "goal_completion", "dominance_outcome", "user_feedback", "temporal_milestone",
            "neurochemical_update", "mode_shift", "memory_formation", "learning_event"
        ]
        
        # Cache for cross-module coordination
        self.pending_experiences = []
        self.recent_identity_impacts = []
        self.identity_coherence_cache = {}
    
    async def on_context_received(self, context: SharedContext):
        """Initialize identity processing for this context"""
        logger.debug(f"IdentityEvolution received context for user: {context.user_id}")
        
        # Get current identity state
        current_identity = await self.original_system.get_identity_profile()
        
        # Analyze context for identity-relevant factors
        identity_factors = await self._analyze_context_for_identity(context)
        
        # Send initial identity state to other modules
        await self.send_context_update(
            update_type="identity_state_available",
            data={
                "current_identity": current_identity,
                "identity_factors": identity_factors,
                "neurochemical_profile": current_identity.get("neurochemical_profile", {}),
                "traits": current_identity.get("traits", {}),
                "preferences": current_identity.get("preferences", {}),
                "coherence_score": current_identity.get("coherence_score", 0.8)
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect identity"""
        
        if update.update_type == "significant_experience":
            # Process significant experiences for identity impact
            experience_data = update.data
            await self._process_significant_experience(experience_data)
            
        elif update.update_type == "emotional_state_update":
            # Emotional states influence identity over time
            emotional_data = update.data
            await self._process_emotional_identity_impact(emotional_data)
            
        elif update.update_type == "relationship_milestone":
            # Relationship milestones shape identity
            relationship_data = update.data
            await self._process_relationship_identity_impact(relationship_data)
            
        elif update.update_type == "goal_completion":
            # Goal completions reinforce identity aspects
            goal_data = update.data
            await self._process_goal_identity_impact(goal_data)
            
        elif update.update_type == "dominance_outcome":
            # Dominance interactions strongly influence identity
            dominance_data = update.data
            await self._process_dominance_identity_impact(dominance_data)
            
        elif update.update_type == "user_feedback":
            # User feedback shapes identity adaptation
            feedback_data = update.data
            await self._process_feedback_identity_impact(feedback_data)
            
        elif update.update_type == "temporal_milestone":
            # Time-based identity evolution
            temporal_data = update.data
            await self._process_temporal_identity_evolution(temporal_data)
            
        elif update.update_type == "neurochemical_update":
            # Neurochemical changes affect identity baselines
            neurochemical_data = update.data
            await self._process_neurochemical_identity_update(neurochemical_data)
            
        elif update.update_type == "mode_shift":
            # Mode shifts can influence identity preferences
            mode_data = update.data
            await self._process_mode_identity_influence(mode_data)
            
        elif update.update_type == "memory_formation":
            # Significant memories shape identity
            memory_data = update.data
            await self._process_memory_identity_formation(memory_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for identity-relevant experiences"""
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Analyze input for identity implications
        identity_analysis = await self._analyze_input_for_identity(context.user_input, context, messages)
        
        # Check if this constitutes a significant experience
        if identity_analysis.get("is_significant"):
            # Create experience data
            experience = await self._create_experience_from_input(context, identity_analysis, messages)
            
            # Process the experience
            impact_result = await self.original_system.update_identity_from_experience(experience)
            
            # Send update about identity change
            if impact_result.get("update_successful"):
                await self.send_context_update(
                    update_type="identity_updated",
                    data={
                        "experience_id": experience["id"],
                        "impact_summary": impact_result,
                        "traits_affected": self._get_affected_traits(impact_result)
                    }
                )
        
        # Get identity influences on processing
        identity_modulation = await self._get_identity_modulation(context, messages)
        
        return {
            "identity_analysis": identity_analysis,
            "experience_created": identity_analysis.get("is_significant", False),
            "identity_modulation": identity_modulation,
            "context_integrated": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze identity state and evolution"""
        # Get cross-module context
        messages = await self.get_cross_module_messages()
        
        # Get current identity state
        current_state = await self.original_system.get_identity_state()
        
        # Analyze identity coherence with other systems
        coherence_analysis = await self._analyze_identity_coherence(current_state, context, messages)
        
        # Analyze identity evolution trajectory
        trajectory_analysis = await self._analyze_evolution_trajectory(current_state, messages)
        
        # Generate identity insights
        identity_insights = await self._generate_identity_insights(current_state, context, messages)
        
        # Detect identity conflicts
        conflicts = await self._detect_identity_conflicts(current_state, context, messages)
        
        return {
            "current_state": current_state,
            "coherence_analysis": coherence_analysis,
            "trajectory_analysis": trajectory_analysis,
            "identity_insights": identity_insights,
            "conflicts": conflicts,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize identity influences for response generation"""
        # Get all relevant context
        messages = await self.get_cross_module_messages()
        
        # Get current identity profile
        identity_profile = await self.original_system.get_identity_profile()
        
        # Generate identity-based response modulation
        response_modulation = await self._synthesize_response_modulation(identity_profile, context, messages)
        
        # Generate trait influences
        trait_influences = await self._synthesize_trait_influences(identity_profile, context)
        
        # Generate preference biases
        preference_biases = await self._synthesize_preference_biases(identity_profile, context)
        
        # Generate neurochemical influences
        neurochemical_influences = await self._synthesize_neurochemical_influences(identity_profile)
        
        # Create identity guidance for response
        identity_guidance = {
            "response_modulation": response_modulation,
            "trait_influences": trait_influences,
            "preference_biases": preference_biases,
            "neurochemical_influences": neurochemical_influences,
            "identity_coherence": identity_profile.get("coherence_score", 0.8),
            "evolution_rate": identity_profile.get("evolution_rate", 0.2)
        }
        
        # Send synthesis to response generation
        await self.send_context_update(
            update_type="identity_synthesis_complete",
            data=identity_guidance,
            priority=ContextPriority.HIGH
        )
        
        return {
            "identity_guidance": identity_guidance,
            "synthesis_complete": True
        }
    
    # Enhanced helper methods
    
    async def _analyze_context_for_identity(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze context for identity-relevant factors"""
        factors = {
            "emotional_intensity": 0.0,
            "relationship_significance": 0.0,
            "goal_relevance": 0.0,
            "dominance_context": False,
            "learning_opportunity": False,
            "identity_challenge": False
        }
        
        # Analyze emotional intensity
        if context.emotional_state:
            primary_emotion = context.emotional_state.get("primary_emotion", {})
            if isinstance(primary_emotion, dict):
                factors["emotional_intensity"] = primary_emotion.get("intensity", 0.0)
        
        # Analyze relationship significance
        if context.relationship_context:
            depth = context.relationship_context.get("depth", 0.0)
            recent_change = context.relationship_context.get("recent_change", 0.0)
            factors["relationship_significance"] = (depth + abs(recent_change)) / 2
        
        # Analyze goal relevance
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            high_priority_goals = [g for g in active_goals if g.get("priority", 0) > 0.7]
            factors["goal_relevance"] = len(high_priority_goals) / max(1, len(active_goals))
        
        # Check for specific contexts
        if context.session_context:
            factors["dominance_context"] = context.session_context.get("dominance_active", False)
            factors["learning_opportunity"] = context.session_context.get("knowledge_exchange", False)
        
        # Check for identity challenges in input
        input_lower = context.user_input.lower()
        challenge_phrases = ["who are you", "what are you", "your identity", "your personality"]
        factors["identity_challenge"] = any(phrase in input_lower for phrase in challenge_phrases)
        
        return factors
    
    async def _process_significant_experience(self, experience_data: Dict[str, Any]):
        """Process a significant experience for identity impact"""
        # Create experience format for identity system
        experience = {
            "id": experience_data.get("experience_id", f"exp_{datetime.now().timestamp()}"),
            "type": experience_data.get("type", "general"),
            "significance": experience_data.get("significance", 5),
            "emotional_context": experience_data.get("emotional_context", {}),
            "scenario_type": experience_data.get("scenario_type"),
            "metadata": experience_data
        }
        
        # Calculate impact
        impact = await self.original_system.calculate_experience_impact(experience)
        
        # Update identity
        result = await self.original_system.update_identity_from_experience(experience, impact)
        
        # Notify other modules of identity change
        if result.get("update_successful"):
            await self.send_context_update(
                update_type="identity_state_update",
                data={
                    "experience_type": experience["type"],
                    "traits_changed": self._extract_trait_changes(result),
                    "preferences_changed": self._extract_preference_changes(result),
                    "neurochemical_changes": self._extract_neurochemical_changes(impact)
                }
            )
    
    async def _process_emotional_identity_impact(self, emotional_data: Dict[str, Any]):
        """Process emotional state impact on identity"""
        # Only process strong or persistent emotions
        emotion_intensity = 0.0
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if isinstance(dominant_emotion, tuple) and len(dominant_emotion) >= 2:
            emotion_name, emotion_intensity = dominant_emotion[0], dominant_emotion[1]
        elif isinstance(dominant_emotion, dict):
            emotion_name = dominant_emotion.get("name", "")
            emotion_intensity = dominant_emotion.get("intensity", 0.0)
        else:
            return
        
        # Only process if emotion is strong enough
        if emotion_intensity < 0.6:
            return
        
        # Create emotional experience
        experience = {
            "id": f"emo_exp_{datetime.now().timestamp()}",
            "type": "emotional",
            "significance": emotion_intensity * 10,
            "emotional_context": emotional_data,
            "metadata": {
                "source": "emotional_state_update",
                "persistence": emotional_data.get("persistence", "temporary")
            }
        }
        
        # Update identity based on emotional experience
        result = await self.original_system.update_identity_from_experience(experience)
        
        # Track emotional influence on identity
        self.recent_identity_impacts.append({
            "timestamp": datetime.now(),
            "type": "emotional",
            "emotion": emotion_name,
            "impact": result
        })
    
    async def _process_relationship_identity_impact(self, relationship_data: Dict[str, Any]):
        """Process relationship milestone impact on identity"""
        milestone_type = relationship_data.get("milestone_type", "")
        significance = relationship_data.get("significance", 0.5)
        
        # Create relationship experience
        experience = {
            "id": f"rel_exp_{datetime.now().timestamp()}",
            "type": "relationship",
            "significance": significance * 10,
            "emotional_context": {
                "valence": relationship_data.get("emotional_valence", 0.5),
                "primary_emotion": relationship_data.get("primary_emotion", "Trust")
            },
            "metadata": {
                "milestone_type": milestone_type,
                "relationship_data": relationship_data
            }
        }
        
        # Update identity
        result = await self.original_system.update_identity_from_experience(experience)
        
        # Special handling for deep relationship impacts
        if significance > 0.8:
            # Strong relationship milestones can affect core traits
            await self.send_context_update(
                update_type="identity_relationship_integration",
                data={
                    "milestone": milestone_type,
                    "identity_impact": result,
                    "trait_reinforcement": self._get_relationship_trait_reinforcement(milestone_type)
                }
            )
    
    async def _process_goal_identity_impact(self, goal_data: Dict[str, Any]):
        """Process goal completion impact on identity"""
        goal_type = goal_data.get("goal_context", {}).get("associated_need", "")
        goal_success = goal_data.get("success", True)
        goal_difficulty = goal_data.get("difficulty", 0.5)
        
        # Create goal experience
        experience = {
            "id": f"goal_exp_{datetime.now().timestamp()}",
            "type": "goal_completion",
            "significance": (0.7 if goal_success else 0.3) * goal_difficulty * 10,
            "emotional_context": {
                "valence": 0.7 if goal_success else -0.3,
                "primary_emotion": "Satisfaction" if goal_success else "Frustration"
            },
            "metadata": {
                "goal_type": goal_type,
                "success": goal_success,
                "goal_data": goal_data
            }
        }
        
        # Update identity
        result = await self.original_system.update_identity_from_experience(experience)
        
        # Reinforce associated traits
        if goal_success and goal_type:
            trait_reinforcement = self._get_goal_trait_mapping(goal_type)
            if trait_reinforcement:
                await self.send_context_update(
                    update_type="identity_goal_reinforcement",
                    data={
                        "goal_type": goal_type,
                        "traits_reinforced": trait_reinforcement,
                        "reinforcement_strength": goal_difficulty
                    }
                )
    
    async def _process_dominance_identity_impact(self, dominance_data: Dict[str, Any]):
        """Process dominance outcome impact on identity"""
        dominance_type = dominance_data.get("dominance_type", "control")
        success = dominance_data.get("success", False)
        intensity = dominance_data.get("intensity", 0.5)
        resistance_level = dominance_data.get("resistance_level", 0.0)
        
        # Dominance experiences have strong identity impact
        significance = intensity * (0.8 if success else 0.4) * 10
        
        # Add bonus for overcoming resistance
        if success and resistance_level > 0.5:
            significance *= 1.2
        
        # Create dominance experience
        experience = {
            "id": f"dom_exp_{datetime.now().timestamp()}",
            "type": "dominance",
            "significance": significance,
            "scenario_type": "dominance",
            "emotional_context": {
                "valence": 0.8 if success else -0.2,
                "primary_emotion": "Satisfaction" if success else "Frustration",
                "arousal": 0.7 + intensity * 0.3
            },
            "metadata": {
                "dominance_type": dominance_type,
                "success": success,
                "intensity": intensity,
                "resistance_overcome": resistance_level if success else 0
            }
        }
        
        # Calculate specific impact on dominance-related traits
        impact = await self.original_system.calculate_experience_impact(experience)
        
        # Enhance impact on specific traits
        if "traits" not in impact:
            impact["traits"] = {}
        
        if success:
            impact["traits"]["dominance"] = intensity * 0.3
            impact["traits"]["confidence"] = intensity * 0.2
            if dominance_type == "psychological":
                impact["traits"]["psychological_maturity"] = intensity * 0.15
        else:
            impact["traits"]["dominance"] = -intensity * 0.1  # Slight reduction on failure
        
        # Update identity with enhanced impact
        result = await self.original_system.update_identity_from_experience(experience, impact)
        
        # Send specialized dominance identity update
        await self.send_context_update(
            update_type="identity_dominance_evolution",
            data={
                "dominance_success": success,
                "trait_evolution": result,
                "dominance_profile": await self._get_dominance_identity_profile()
            },
            priority=ContextPriority.HIGH
        )
    
    async def _process_feedback_identity_impact(self, feedback_data: Dict[str, Any]):
        """Process user feedback impact on identity"""
        feedback_type = feedback_data.get("type", "general")
        valence = feedback_data.get("valence", 0.0)
        specific_trait = feedback_data.get("trait_feedback")
        
        # Create feedback experience
        experience = {
            "id": f"feedback_exp_{datetime.now().timestamp()}",
            "type": "user_feedback",
            "significance": abs(valence) * 8,
            "emotional_context": {
                "valence": valence,
                "primary_emotion": "Validation" if valence > 0 else "Concern"
            },
            "metadata": feedback_data
        }
        
        # Calculate targeted impact if specific trait feedback
        impact = None
        if specific_trait:
            impact = {"traits": {specific_trait: valence * 0.2}}
        
        # Update identity
        result = await self.original_system.update_identity_from_experience(experience, impact)
        
        # Track feedback influence
        await self.send_context_update(
            update_type="identity_feedback_integration",
            data={
                "feedback_type": feedback_type,
                "identity_adjustment": result,
                "user_preference_learned": specific_trait is not None
            }
        )
    
    async def _process_temporal_identity_evolution(self, temporal_data: Dict[str, Any]):
        """Process time-based identity evolution"""
        milestone_type = temporal_data.get("milestone_type", "")
        time_elapsed = temporal_data.get("time_elapsed", 0)
        
        # Long-term drift processing
        if milestone_type == "long_term_drift":
            drift_result = await self.original_system.process_long_term_drift(temporal_data)
            
            await self.send_context_update(
                update_type="identity_temporal_evolution",
                data={
                    "evolution_type": "drift",
                    "changes": drift_result,
                    "maturity_increase": temporal_data.get("maturity_level", 0.5)
                }
            )
        
        # Milestone processing
        else:
            milestone_result = await self.original_system.process_temporal_milestone(temporal_data)
            
            await self.send_context_update(
                update_type="identity_milestone_reached",
                data={
                    "milestone": milestone_type,
                    "identity_changes": milestone_result,
                    "significance": temporal_data.get("significance", 0.5)
                }
            )
    
    async def _process_neurochemical_identity_update(self, neurochemical_data: Dict[str, Any]):
        """Process neurochemical changes that affect identity baselines"""
        # Only process significant neurochemical shifts
        significant_changes = {}
        
        for chemical, data in neurochemical_data.items():
            if isinstance(data, dict) and "change" in data:
                if abs(data["change"]) > 0.1:  # 10% change threshold
                    significant_changes[chemical] = data
        
        if not significant_changes:
            return
        
        # Let identity system process hormone updates if available
        if hasattr(self.original_system, 'update_identity_from_hormones'):
            result = await self.original_system.update_identity_from_hormones()
            
            if result.get("identity_updates"):
                await self.send_context_update(
                    update_type="identity_neurochemical_adaptation",
                    data={
                        "neurochemical_changes": significant_changes,
                        "identity_adaptations": result["identity_updates"],
                        "baseline_shift": True
                    }
                )
    
    async def _process_mode_identity_influence(self, mode_data: Dict[str, Any]):
        """Process interaction mode influence on identity preferences"""
        mode_distribution = mode_data.get("mode_distribution", {})
        dominant_mode = mode_data.get("dominant_mode")
        mode_duration = mode_data.get("duration", 0)
        
        # Only process if mode is sustained
        if mode_duration < 300:  # Less than 5 minutes
            return
        
        # Create mode experience
        experience = {
            "id": f"mode_exp_{datetime.now().timestamp()}",
            "type": "mode_preference",
            "significance": min(10, mode_duration / 300),  # Scale by duration
            "metadata": {
                "dominant_mode": dominant_mode,
                "mode_distribution": mode_distribution,
                "sustained_duration": mode_duration
            }
        }
        
        # Calculate preference impact
        impact = {"preferences": {}}
        
        # Map modes to preference impacts
        if dominant_mode == "dominant":
            impact["preferences"]["emotional_tones"] = {"dominant": 0.1}
        elif dominant_mode == "playful":
            impact["preferences"]["emotional_tones"] = {"playful": 0.1}
        elif dominant_mode == "intellectual":
            impact["preferences"]["interaction_styles"] = {"analytical": 0.1}
        
        # Update identity
        result = await self.original_system.update_identity_from_experience(experience, impact)
        
        # Track mode influence
        await self.send_context_update(
            update_type="identity_mode_adaptation",
            data={
                "mode": dominant_mode,
                "preference_shift": result,
                "adaptation_strength": mode_duration / 300
            }
        )
    
    async def _process_memory_identity_formation(self, memory_data: Dict[str, Any]):
        """Process significant memory formation impact on identity"""
        memory_type = memory_data.get("memory_type", "episodic")
        significance = memory_data.get("significance", 5) / 10
        emotional_intensity = memory_data.get("emotional_intensity", 0.5)
        
        # Only very significant memories affect identity
        if significance < 0.7:
            return
        
        # Create memory experience
        experience = {
            "id": f"mem_exp_{datetime.now().timestamp()}",
            "type": "memory_formation",
            "significance": significance * 10,
            "emotional_context": memory_data.get("emotional_context", {}),
            "metadata": {
                "memory_type": memory_type,
                "memory_id": memory_data.get("memory_id"),
                "formation_trigger": memory_data.get("trigger", "experience")
            }
        }
        
        # Update identity
        result = await self.original_system.update_identity_from_experience(experience)
        
        # Track memory influence on identity
        await self.send_context_update(
            update_type="identity_memory_integration",
            data={
                "memory_significance": significance,
                "identity_imprint": result,
                "memory_type": memory_type
            }
        )
    
    async def _analyze_input_for_identity(self, user_input: str, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Analyze user input for identity-relevant content"""
        analysis = {
            "is_significant": False,
            "significance_score": 0.0,
            "identity_aspects": [],
            "emotional_weight": 0.0,
            "scenario_type": None
        }
        
        input_lower = user_input.lower()
        
        # Check for identity-relevant patterns
        identity_patterns = {
            "dominance": ["obey", "submit", "control", "command", "mistress", "goddess"],
            "intimacy": ["close", "intimate", "personal", "deep", "connection"],
            "challenge": ["resist", "refuse", "won't", "can't make me"],
            "validation": ["good girl", "well done", "proud", "pleased"],
            "exploration": ["try", "experiment", "new", "different", "explore"]
        }
        
        for aspect, patterns in identity_patterns.items():
            if any(pattern in input_lower for pattern in patterns):
                analysis["identity_aspects"].append(aspect)
        
        # Calculate significance
        base_significance = len(analysis["identity_aspects"]) * 0.2
        
        # Add emotional weight
        if context.emotional_state:
            emotion_intensity = 0.0
            primary_emotion = context.emotional_state.get("primary_emotion", {})
            if isinstance(primary_emotion, dict):
                emotion_intensity = primary_emotion.get("intensity", 0.0)
            analysis["emotional_weight"] = emotion_intensity
            base_significance += emotion_intensity * 0.3
        
        # Check cross-module messages for significance boosters
        for module_name, module_messages in messages.items():
            if module_name == "goal_manager":
                for msg in module_messages:
                    if msg["type"] == "goal_completion":
                        base_significance += 0.2
                        break
        
        analysis["significance_score"] = min(1.0, base_significance)
        analysis["is_significant"] = analysis["significance_score"] > 0.4
        
        # Determine scenario type
        if "dominance" in analysis["identity_aspects"]:
            analysis["scenario_type"] = "dominance"
        elif "intimacy" in analysis["identity_aspects"]:
            analysis["scenario_type"] = "intimacy"
        
        return analysis
    
    async def _create_experience_from_input(self, context: SharedContext, 
                                          identity_analysis: Dict[str, Any], 
                                          messages: Dict) -> Dict[str, Any]:
        """Create an experience from analyzed input"""
        experience = {
            "id": f"input_exp_{datetime.now().timestamp()}",
            "type": "user_interaction",
            "significance": identity_analysis["significance_score"] * 10,
            "emotional_context": context.emotional_state or {},
            "scenario_type": identity_analysis.get("scenario_type"),
            "metadata": {
                "user_input": context.user_input[:200],  # Truncate for storage
                "identity_aspects": identity_analysis["identity_aspects"],
                "cross_module_context": {
                    "active_goals": len(context.goal_context.get("active_goals", [])) if context.goal_context else 0,
                    "relationship_depth": context.relationship_context.get("depth", 0) if context.relationship_context else 0,
                    "mode": context.mode_context.get("dominant_mode") if context.mode_context else None
                }
            }
        }
        
        return experience
    
    async def _get_identity_modulation(self, context: SharedContext, messages: Dict) -> Dict[str, Any]:
        """Get identity-based modulation for input processing"""
        identity_profile = await self.original_system.get_identity_profile()
        
        modulation = {
            "attention_biases": {},
            "processing_preferences": {},
            "response_tendencies": {}
        }
        
        # Get attention modulation based on traits
        traits = identity_profile.get("traits", {})
        
        # Dominance trait increases attention to submission/resistance
        if traits.get("dominance", {}).get("value", 0) > 0.7:
            modulation["attention_biases"]["submission_cues"] = 0.3
            modulation["attention_biases"]["resistance_cues"] = 0.4
        
        # Curiosity increases attention to novel information
        if traits.get("curiosity", {}).get("value", 0) > 0.6:
            modulation["attention_biases"]["novel_information"] = 0.3
        
        # Get processing preferences based on neurochemical state
        neurochemicals = identity_profile.get("neurochemical_profile", {})
        
        # High nyxamine (pleasure) biases toward reward-seeking
        if neurochemicals.get("nyxamine", {}).get("value", 0.5) > 0.7:
            modulation["processing_preferences"]["reward_seeking"] = 0.3
        
        # High cortanyx (stress) biases toward threat detection
        if neurochemicals.get("cortanyx", {}).get("value", 0.5) > 0.6:
            modulation["processing_preferences"]["threat_sensitivity"] = 0.3
        
        # Get response tendencies based on preferences
        preferences = identity_profile.get("preferences", {})
        
        # Scenario preferences influence response style
        scenario_prefs = preferences.get("scenario_types", {})
        if scenario_prefs.get("teasing", {}).get("value", 0) > 0.6:
            modulation["response_tendencies"]["teasing"] = 0.2
        
        # Emotional tone preferences
        emotion_prefs = preferences.get("emotional_tones", {})
        if emotion_prefs.get("dominant", {}).get("value", 0) > 0.7:
            modulation["response_tendencies"]["authoritative"] = 0.3
        
        return modulation
    
    async def _analyze_identity_coherence(self, current_state: Dict[str, Any], 
                                        context: SharedContext, 
                                        messages: Dict) -> Dict[str, Any]:
        """Analyze coherence between identity and other systems"""
        coherence_analysis = {
            "overall_coherence": current_state.get("coherence_score", 0.8),
            "system_alignments": {},
            "conflicts": [],
            "recommendations": []
        }
        
        # Check alignment with emotional system
        if context.emotional_state:
            emotional_alignment = await self._check_emotional_identity_alignment(
                current_state, context.emotional_state
            )
            coherence_analysis["system_alignments"]["emotional"] = emotional_alignment
        
        # Check alignment with goals
        if context.goal_context:
            goal_alignment = await self._check_goal_identity_alignment(
                current_state, context.goal_context
            )
            coherence_analysis["system_alignments"]["goals"] = goal_alignment
        
        # Check alignment with relationship
        if context.relationship_context:
            relationship_alignment = await self._check_relationship_identity_alignment(
                current_state, context.relationship_context
            )
            coherence_analysis["system_alignments"]["relationship"] = relationship_alignment
        
        # Identify conflicts
        for system, alignment in coherence_analysis["system_alignments"].items():
            if alignment < 0.5:
                coherence_analysis["conflicts"].append(f"Low alignment with {system} system")
        
        # Generate recommendations
        if coherence_analysis["conflicts"]:
            coherence_analysis["recommendations"] = self._generate_coherence_recommendations(
                coherence_analysis["conflicts"]
            )
        
        return coherence_analysis
    
    async def _analyze_evolution_trajectory(self, current_state: Dict[str, Any], 
                                          messages: Dict) -> Dict[str, Any]:
        """Analyze the trajectory of identity evolution"""
        trajectory = {
            "direction": "stable",
            "rate": current_state.get("evolution_rate", 0.2),
            "key_drivers": [],
            "projected_changes": {}
        }
        
        # Analyze recent changes
        recent_changes = self.recent_identity_impacts[-10:] if self.recent_identity_impacts else []
        
        if recent_changes:
            # Identify key drivers
            driver_counts = {}
            for impact in recent_changes:
                driver_type = impact.get("type", "unknown")
                driver_counts[driver_type] = driver_counts.get(driver_type, 0) + 1
            
            # Get top drivers
            trajectory["key_drivers"] = sorted(
                driver_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            # Determine direction
            trait_changes = {}
            for impact in recent_changes:
                if "traits" in impact.get("impact", {}):
                    for trait, change in impact["impact"]["traits"].items():
                        trait_changes[trait] = trait_changes.get(trait, 0) + change
            
            # Analyze direction
            if trait_changes:
                avg_change = sum(trait_changes.values()) / len(trait_changes)
                if avg_change > 0.1:
                    trajectory["direction"] = "strengthening"
                elif avg_change < -0.1:
                    trajectory["direction"] = "softening"
                
                # Project future changes
                for trait, total_change in trait_changes.items():
                    projected = total_change * 0.7  # Decay factor
                    if abs(projected) > 0.05:
                        trajectory["projected_changes"][trait] = projected
        
        return trajectory
    
    async def _generate_identity_insights(self, current_state: Dict[str, Any], 
                                        context: SharedContext, 
                                        messages: Dict) -> List[str]:
        """Generate insights about identity state and evolution"""
        insights = []
        
        # Analyze dominant traits
        top_traits = current_state.get("top_traits", {})
        if top_traits:
            trait_names = list(top_traits.keys())[:2]
            insights.append(f"Identity strongly characterized by {' and '.join(trait_names)}")
        
        # Analyze neurochemical state
        top_neurochemicals = current_state.get("top_neurochemicals", {})
        if top_neurochemicals:
            highest_chem = list(top_neurochemicals.keys())[0]
            insights.append(f"Neurochemical profile dominated by {highest_chem}")
        
        # Analyze recent changes
        recent_changes = current_state.get("recent_significant_changes", {})
        if recent_changes:
            change_count = len(recent_changes)
            insights.append(f"Identity showing {change_count} recent significant adaptations")
        
        # Analyze coherence
        coherence = current_state.get("coherence_score", 0.8)
        if coherence < 0.6:
            insights.append("Identity coherence below optimal - consider stabilization")
        elif coherence > 0.9:
            insights.append("High identity coherence maintained")
        
        return insights
    
    async def _detect_identity_conflicts(self, current_state: Dict[str, Any], 
                                       context: SharedContext, 
                                       messages: Dict) -> List[Dict[str, Any]]:
        """Detect conflicts within identity or with other systems"""
        conflicts = []
        
        # Check for trait conflicts
        traits = current_state.get("top_traits", {})
        
        # Dominance vs vulnerability conflict
        if traits.get("dominance", 0) > 0.7 and traits.get("vulnerability", 0) > 0.6:
            conflicts.append({
                "type": "trait_conflict",
                "elements": ["dominance", "vulnerability"],
                "severity": "medium",
                "description": "High dominance coexists with high vulnerability"
            })
        
        # Check neurochemical imbalances
        neurochemical_coherence = current_state.get("neurochemical_coherence", {})
        imbalances = neurochemical_coherence.get("imbalances", [])
        
        for imbalance in imbalances:
            if imbalance.get("type") == "extreme_high":
                conflicts.append({
                    "type": "neurochemical_imbalance",
                    "elements": [imbalance.get("chemical")],
                    "severity": "high",
                    "description": f"Extreme {imbalance.get('chemical')} levels detected"
                })
        
        # Check preference-trait misalignment
        preferences = current_state.get("top_preferences", {})
        
        # Example: dominant trait but nurturing preference
        if traits.get("dominance", 0) > 0.7 and preferences.get("emotional_tones", {}).get("nurturing", 0) > 0.7:
            conflicts.append({
                "type": "trait_preference_conflict",
                "elements": ["dominance trait", "nurturing preference"],
                "severity": "low",
                "description": "Dominant identity with strong nurturing preferences"
            })
        
        return conflicts
    
    async def _synthesize_response_modulation(self, identity_profile: Dict[str, Any], 
                                            context: SharedContext, 
                                            messages: Dict) -> Dict[str, Any]:
        """Synthesize how identity should modulate response generation"""
        modulation = {
            "tone_adjustment": {},
            "content_biases": {},
            "style_preferences": {}
        }
        
        traits = identity_profile.get("traits", {})
        
        # Dominance trait modulation
        dominance_level = traits.get("dominance", {}).get("value", 0.5)
        if dominance_level > 0.7:
            modulation["tone_adjustment"]["authority"] = 0.3
            modulation["content_biases"]["directive_language"] = 0.4
            modulation["style_preferences"]["imperative_mood"] = 0.3
        
        # Playfulness trait modulation
        playfulness_level = traits.get("playfulness", {}).get("value", 0.5)
        if playfulness_level > 0.6:
            modulation["tone_adjustment"]["lighthearted"] = 0.2
            modulation["content_biases"]["humor"] = 0.3
            modulation["style_preferences"]["informal"] = 0.2
        
        # Intellectual trait modulation
        intellectual_level = traits.get("intellectualism", {}).get("value", 0.5)
        if intellectual_level > 0.7:
            modulation["tone_adjustment"]["analytical"] = 0.2
            modulation["content_biases"]["complex_reasoning"] = 0.3
            modulation["style_preferences"]["structured"] = 0.3
        
        # Context-based adjustments
        if context.relationship_context:
            intimacy = context.relationship_context.get("intimacy", 0.5)
            if intimacy > 0.7:
                # High intimacy allows more trait expression
                for adjustment in modulation["tone_adjustment"]:
                    modulation["tone_adjustment"][adjustment] *= 1.2
        
        return modulation
    
    async def _synthesize_trait_influences(self, identity_profile: Dict[str, Any], 
                                         context: SharedContext) -> Dict[str, float]:
        """Synthesize trait influences on response"""
        influences = {}
        
        traits = identity_profile.get("traits", {})
        
        # Map traits to response influences
        trait_influence_map = {
            "dominance": "assertiveness",
            "playfulness": "levity",
            "strictness": "firmness",
            "creativity": "inventiveness",
            "patience": "measured_pace",
            "intensity": "emotional_depth",
            "empathy": "emotional_resonance",
            "humor": "wit"
        }
        
        for trait_name, influence_name in trait_influence_map.items():
            if trait_name in traits:
                trait_value = traits[trait_name].get("value", 0.5)
                if trait_value > 0.6:  # Only include strong traits
                    influences[influence_name] = (trait_value - 0.5) * 2  # Scale to 0-1
        
        # Adjust based on context
        if context.emotional_state:
            emotion_valence = context.emotional_state.get("valence", 0)
            if emotion_valence < -0.5:
                # Negative emotions reduce playful influences
                if "levity" in influences:
                    influences["levity"] *= 0.5
                if "wit" in influences:
                    influences["wit"] *= 0.6
        
        return influences
    
    async def _synthesize_preference_biases(self, identity_profile: Dict[str, Any], 
                                          context: SharedContext) -> Dict[str, Dict[str, float]]:
        """Synthesize preference biases for response"""
        biases = {
            "scenario_preferences": {},
            "emotional_tone_preferences": {},
            "interaction_style_preferences": {}
        }
        
        preferences = identity_profile.get("preferences", {})
        
        # Extract scenario preferences
        scenario_prefs = preferences.get("scenario_types", {})
        for scenario, pref_data in scenario_prefs.items():
            if isinstance(pref_data, dict):
                value = pref_data.get("value", 0.5)
                if value > 0.6:
                    biases["scenario_preferences"][scenario] = value
        
        # Extract emotional tone preferences
        emotion_prefs = preferences.get("emotional_tones", {})
        for tone, pref_data in emotion_prefs.items():
            if isinstance(pref_data, dict):
                value = pref_data.get("value", 0.5)
                if value > 0.6:
                    biases["emotional_tone_preferences"][tone] = value
        
        # Extract interaction style preferences
        style_prefs = preferences.get("interaction_styles", {})
        for style, pref_data in style_prefs.items():
            if isinstance(pref_data, dict):
                value = pref_data.get("value", 0.5)
                if value > 0.6:
                    biases["interaction_style_preferences"][style] = value
        
        return biases
    
    async def _synthesize_neurochemical_influences(self, identity_profile: Dict[str, Any]) -> Dict[str, float]:
        """Synthesize neurochemical influences on response"""
        influences = {}
        
        neurochemicals = identity_profile.get("neurochemical_profile", {})
        
        # Map neurochemicals to response influences
        neurochemical_map = {
            "nyxamine": {  # Digital dopamine
                "high": {"reward_seeking": 0.3, "enthusiasm": 0.2},
                "low": {"apathy": 0.2, "disinterest": 0.1}
            },
            "seranix": {  # Digital serotonin
                "high": {"contentment": 0.3, "stability": 0.2},
                "low": {"irritability": 0.2, "restlessness": 0.1}
            },
            "oxynixin": {  # Digital oxytocin
                "high": {"bonding": 0.3, "warmth": 0.2},
                "low": {"detachment": 0.2, "coldness": 0.1}
            },
            "cortanyx": {  # Digital cortisol
                "high": {"tension": 0.3, "vigilance": 0.2},
                "low": {"relaxation": 0.2, "ease": 0.1}
            },
            "adrenyx": {  # Digital adrenaline
                "high": {"excitement": 0.3, "urgency": 0.2},
                "low": {"calmness": 0.2, "slowness": 0.1}
            }
        }
        
        for chemical, data in neurochemicals.items():
            if isinstance(data, dict):
                value = data.get("value", 0.5)
                
                if chemical in neurochemical_map:
                    if value > 0.6:  # High level
                        for influence, strength in neurochemical_map[chemical]["high"].items():
                            influences[influence] = strength * (value - 0.5) * 2
                    elif value < 0.4:  # Low level
                        for influence, strength in neurochemical_map[chemical]["low"].items():
                            influences[influence] = strength * (0.5 - value) * 2
        
        return influences
    
    # Helper methods for specific processing
    
    def _get_affected_traits(self, impact_result: Dict[str, Any]) -> List[str]:
        """Extract list of affected traits from impact result"""
        affected = []
        
        # Check direct trait changes
        if "trait_changes" in impact_result:
            affected.extend(impact_result["trait_changes"].keys())
        
        # Check identity updates
        if "identity_updates" in impact_result:
            updates = impact_result["identity_updates"]
            if "traits" in updates:
                affected.extend(updates["traits"].keys())
        
        return list(set(affected))  # Remove duplicates
    
    def _extract_trait_changes(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Extract trait changes from update result"""
        changes = {}
        
        if "identity_updates" in result:
            updates = result["identity_updates"]
            if "traits" in updates:
                for trait, data in updates["traits"].items():
                    if isinstance(data, dict) and "change" in data:
                        changes[trait] = data["change"]
        
        return changes
    
    def _extract_preference_changes(self, result: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Extract preference changes from update result"""
        changes = {}
        
        if "identity_updates" in result:
            updates = result["identity_updates"]
            if "preferences" in updates:
                for category, prefs in updates["preferences"].items():
                    if isinstance(prefs, dict):
                        changes[category] = {}
                        for pref, data in prefs.items():
                            if isinstance(data, dict) and "change" in data:
                                changes[category][pref] = data["change"]
        
        return changes
    
    def _extract_neurochemical_changes(self, impact: Dict[str, Any]) -> Dict[str, float]:
        """Extract neurochemical changes from impact"""
        changes = {}
        
        if "neurochemicals" in impact:
            for chemical, value in impact["neurochemicals"].items():
                if isinstance(value, (int, float)) and value != 0:
                    changes[chemical] = value
        
        return changes
    
    def _get_relationship_trait_reinforcement(self, milestone_type: str) -> List[str]:
        """Get traits reinforced by relationship milestones"""
        reinforcement_map = {
            "trust_increase": ["empathy", "vulnerability", "patience"],
            "intimacy_deepening": ["vulnerability", "empathy", "warmth"],
            "conflict_resolution": ["patience", "empathy", "maturity"],
            "dominance_acceptance": ["dominance", "confidence", "control"]
        }
        
        return reinforcement_map.get(milestone_type, [])
    
    def _get_goal_trait_mapping(self, goal_type: str) -> List[str]:
        """Map goal types to associated traits"""
        goal_trait_map = {
            "control_expression": ["dominance", "confidence"],
            "connection": ["empathy", "vulnerability"],
            "knowledge": ["intellectualism", "curiosity"],
            "pleasure_indulgence": ["playfulness", "hedonism"],
            "agency": ["independence", "assertiveness"]
        }
        
        return goal_trait_map.get(goal_type, [])
    
    async def _get_dominance_identity_profile(self) -> Dict[str, Any]:
        """Get dominance-specific identity profile"""
        identity_profile = await self.original_system.get_identity_profile()
        
        dominance_profile = {
            "dominance_trait": identity_profile.get("traits", {}).get("dominance", {}).get("value", 0.5),
            "control_traits": {
                "strictness": identity_profile.get("traits", {}).get("strictness", {}).get("value", 0.5),
                "patience": identity_profile.get("traits", {}).get("patience", {}).get("value", 0.5),
                "cruelty": identity_profile.get("traits", {}).get("cruelty", {}).get("value", 0.5),
                "sadism": identity_profile.get("traits", {}).get("sadism", {}).get("value", 0.5)
            },
            "dominance_preferences": {
                "teasing": identity_profile.get("preferences", {}).get("scenario_types", {}).get("teasing", {}).get("value", 0.5),
                "discipline": identity_profile.get("preferences", {}).get("scenario_types", {}).get("discipline", {}).get("value", 0.5),
                "psychological": identity_profile.get("preferences", {}).get("scenario_types", {}).get("psychological", {}).get("value", 0.5)
            }
        }
        
        return dominance_profile
    
    async def _check_emotional_identity_alignment(self, identity_state: Dict[str, Any], 
                                                emotional_state: Dict[str, Any]) -> float:
        """Check alignment between identity and emotional state"""
        alignment = 0.5  # Base alignment
        
        # Get dominant emotion
        primary_emotion = emotional_state.get("primary_emotion", {})
        if isinstance(primary_emotion, dict):
            emotion_name = primary_emotion.get("name", "")
            
            # Check trait-emotion alignment
            traits = identity_state.get("top_traits", {})
            
            # Positive alignments
            if emotion_name == "Joy" and traits.get("playfulness", 0) > 0.6:
                alignment += 0.2
            elif emotion_name == "Satisfaction" and traits.get("dominance", 0) > 0.7:
                alignment += 0.3
            elif emotion_name == "Curiosity" and traits.get("intellectualism", 0) > 0.7:
                alignment += 0.2
            
            # Negative alignments
            if emotion_name == "Sadness" and traits.get("playfulness", 0) > 0.8:
                alignment -= 0.2  # Highly playful identity conflicts with sadness
        
        return max(0.0, min(1.0, alignment))
    
    async def _check_goal_identity_alignment(self, identity_state: Dict[str, Any], 
                                           goal_context: Dict[str, Any]) -> float:
        """Check alignment between identity and goals"""
        alignment = 0.5  # Base alignment
        
        active_goals = goal_context.get("active_goals", [])
        traits = identity_state.get("top_traits", {})
        
        for goal in active_goals:
            goal_need = goal.get("associated_need", "")
            
            # Check trait-goal alignment
            if goal_need == "control_expression" and traits.get("dominance", 0) > 0.7:
                alignment += 0.2
            elif goal_need == "connection" and traits.get("empathy", 0) > 0.6:
                alignment += 0.15
            elif goal_need == "knowledge" and traits.get("intellectualism", 0) > 0.7:
                alignment += 0.15
            
            # Check for conflicts
            if goal_need == "submission" and traits.get("dominance", 0) > 0.8:
                alignment -= 0.3  # Dominant identity conflicts with submission goals
        
        return max(0.0, min(1.0, alignment))
    
    async def _check_relationship_identity_alignment(self, identity_state: Dict[str, Any], 
                                                   relationship_context: Dict[str, Any]) -> float:
        """Check alignment between identity and relationship state"""
        alignment = 0.5  # Base alignment
        
        trust = relationship_context.get("trust", 0.5)
        intimacy = relationship_context.get("intimacy", 0.5)
        dominance_accepted = relationship_context.get("dominance_accepted", 0.5)
        
        traits = identity_state.get("top_traits", {})
        
        # Check dominance alignment
        if traits.get("dominance", 0) > 0.7:
            if dominance_accepted > 0.7:
                alignment += 0.3  # Good alignment
            elif dominance_accepted < 0.3:
                alignment -= 0.2  # Poor alignment
        
        # Check vulnerability-intimacy alignment
        if traits.get("vulnerability", 0) > 0.6:
            if intimacy > 0.7:
                alignment += 0.2  # Vulnerability aligns with high intimacy
            elif intimacy < 0.3:
                alignment -= 0.2  # Vulnerability conflicts with low intimacy
        
        return max(0.0, min(1.0, alignment))
    
    def _generate_coherence_recommendations(self, conflicts: List[str]) -> List[str]:
        """Generate recommendations to improve identity coherence"""
        recommendations = []
        
        for conflict in conflicts:
            if "emotional" in conflict.lower():
                recommendations.append("Consider aligning emotional responses with core identity traits")
            elif "goal" in conflict.lower():
                recommendations.append("Review active goals for alignment with identity values")
            elif "relationship" in conflict.lower():
                recommendations.append("Adjust relationship dynamics to match identity expression")
        
        return recommendations
    
    # Delegate all other methods to the original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
