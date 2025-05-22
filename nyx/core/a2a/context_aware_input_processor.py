# nyx/core/a2a/context_aware_input_processor.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareInputProcessor(ContextAwareModule):
    """
    Enhanced BlendedInputProcessor with full context distribution capabilities
    """
    
    def __init__(self, original_input_processor):
        super().__init__("input_processor")
        self.original_processor = original_input_processor
        
        # Add configuration support
        from nyx.core.input_processing_config import InputProcessingConfig
        self.config = InputProcessingConfig()
        super().__init__("input_processor")
        self.original_processor = original_input_processor
        self.context_subscriptions = [
            "mode_distribution_update", "emotional_state_update", "relationship_state_change",
            "conditioning_trigger", "user_behavior_pattern", "dominance_context_active",
            "response_ready", "goal_context_available", "identity_state_update"
        ]
        
        # Cache for cross-module coordination
        self.pattern_detection_cache = {}
        self.behavior_evaluation_cache = {}
        self.mode_blending_history = []
    
    async def on_context_received(self, context: SharedContext):
        """Initialize input processing for this context"""
        logger.debug(f"InputProcessor received context for user: {context.user_id}")
        
        # Get current mode distribution
        mode_distribution = await self._get_mode_distribution_from_context(context)
        
        # Analyze input for initial patterns
        initial_patterns = await self._detect_initial_patterns(context.user_input)
        
        # Send initial processing state
        await self.send_context_update(
            update_type="input_processing_ready",
            data={
                "mode_distribution": mode_distribution,
                "initial_patterns": initial_patterns,
                "conditioning_ready": True,
                "blending_capabilities": {
                    "mode_blending": True,
                    "pattern_detection": True,
                    "behavior_conditioning": True,
                    "response_modification": True
                }
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules affecting input processing"""
        
        if update.update_type == "mode_distribution_update":
            # Update mode blending parameters
            mode_data = update.data
            await self._update_mode_blending(mode_data)
            
        elif update.update_type == "emotional_state_update":
            # Emotional states affect pattern sensitivity
            emotional_data = update.data
            await self._adjust_pattern_sensitivity(emotional_data)
            
        elif update.update_type == "relationship_state_change":
            # Relationship affects conditioning thresholds
            relationship_data = update.data
            await self._adjust_conditioning_thresholds(relationship_data)
            
        elif update.update_type == "conditioning_trigger":
            # Direct conditioning trigger from another module
            conditioning_data = update.data
            await self._process_conditioning_trigger(conditioning_data)
            
        elif update.update_type == "user_behavior_pattern":
            # Long-term behavior patterns detected
            behavior_data = update.data
            await self._update_behavior_baselines(behavior_data)
            
        elif update.update_type == "dominance_context_active":
            # Special processing for dominance contexts
            dominance_data = update.data
            await self._activate_dominance_processing(dominance_data)
            
        elif update.update_type == "response_ready":
            # Modify response based on accumulated context
            response_data = update.data
            await self._prepare_response_modification(response_data)
            
        elif update.update_type == "goal_context_available":
            # Goals influence behavior reinforcement
            goal_data = update.data
            await self._align_conditioning_with_goals(goal_data)
            
        elif update.update_type == "identity_state_update":
            # Identity affects processing preferences
            identity_data = update.data
            await self._update_processing_preferences(identity_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with full context awareness"""
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Run comprehensive pattern detection
        pattern_analysis = await self._comprehensive_pattern_detection(
            context.user_input, context, messages
        )
        
        # Evaluate behaviors based on full context
        behavior_evaluation = await self._contextual_behavior_evaluation(
            pattern_analysis, context, messages
        )
        
        # Process conditioning with context
        conditioning_results = await self._process_contextual_conditioning(
            pattern_analysis, behavior_evaluation, context, messages
        )
        
        # Get mode-specific processing
        mode_processing = await self._process_mode_specific_input(
            context.user_input, context, messages
        )
        
        # Send processing results to other modules
        await self.send_context_update(
            update_type="input_patterns_detected",
            data={
                "patterns": pattern_analysis,
                "behavior_recommendations": behavior_evaluation,
                "conditioning_applied": conditioning_results,
                "mode_processing": mode_processing
            }
        )
        
        return {
            "pattern_analysis": pattern_analysis,
            "behavior_evaluation": behavior_evaluation,
            "conditioning_results": conditioning_results,
            "mode_processing": mode_processing,
            "context_integrated": True
        }
    
    # ADD THIS: Wrapper method for compatibility with standard interface
    async def process_input_standard(self, text: str, user_id: str = "default", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process input text through conditioning system and return processing results.
        This method provides compatibility with the standard InputProcessor interface.
        """
        try:
            # First, process through the original processor to get base results
            base_results = await self.original_processor.process_input(text, user_id, context)
            
            # If we have an active shared context, enhance the results
            if hasattr(self, 'current_shared_context') and self.current_shared_context:
                # Create a temporary context for enhancement
                temp_context = SharedContext(
                    user_input=text,
                    user_id=user_id,
                    initial_context=context or {}
                )
                
                # Get enhanced results using our context-aware processing
                enhanced_results = await self.process_input(temp_context)
                
                # Merge base results with enhanced results
                final_results = base_results.copy()
                final_results.update({
                    "context_enhanced": True,
                    "pattern_analysis": enhanced_results.get("pattern_analysis", {}),
                    "behavior_evaluation": enhanced_results.get("behavior_evaluation", {}),
                    "conditioning_results": enhanced_results.get("conditioning_results", []),
                    "mode_processing": enhanced_results.get("mode_processing", {})
                })
                
                # Send update about processing completion
                await self.send_context_update(
                    update_type="input_processing_complete",
                    data=final_results,
                    priority=ContextPriority.MEDIUM
                )
                
                return final_results
            else:
                # No active context, just use base results
                return base_results
                
        except Exception as e:
            logger.error(f"Error in process_input_standard: {e}", exc_info=True)
            
            # Try to fall back to original processor
            try:
                return await self.original_processor.process_input(text, user_id, context)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                
                # Return minimal error result
                return {
                    "input_text": text,
                    "user_id": user_id,
                    "error": str(e),
                    "detected_patterns": [],
                    "behavior_evaluations": [],
                    "recommended_behaviors": [],
                    "avoided_behaviors": [],
                    "reinforcement_results": [],
                    "processing_failed": True
                }

    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze input processing patterns and effectiveness"""
        # Get cross-module context
        messages = await self.get_cross_module_messages()
        
        # Analyze pattern detection effectiveness
        pattern_effectiveness = await self._analyze_pattern_effectiveness(context, messages)
        
        # Analyze conditioning impact
        conditioning_impact = await self._analyze_conditioning_impact(messages)
        
        # Analyze mode blending coherence
        blending_coherence = await self._analyze_blending_coherence(context)
        
        # Generate processing insights
        processing_insights = await self._generate_processing_insights(
            pattern_effectiveness, conditioning_impact, blending_coherence
        )
        
        return {
            "pattern_effectiveness": pattern_effectiveness,
            "conditioning_impact": conditioning_impact,
            "blending_coherence": blending_coherence,
            "processing_insights": processing_insights,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize response modifications based on input processing"""
        # Get all relevant context
        messages = await self.get_cross_module_messages()
        
        # Get accumulated processing results
        processing_results = await self._get_accumulated_processing_results()
        
        # Synthesize response modifications
        response_modifications = await self._synthesize_response_modifications(
            processing_results, context, messages
        )
        
        # Generate blended mode expression
        blended_expression = await self._synthesize_blended_expression(
            context, messages
        )
        
        # Create conditioning reinforcements
        reinforcements = await self._synthesize_conditioning_reinforcements(
            processing_results, context
        )
        
        # Create input processing synthesis
        processing_synthesis = {
            "response_modifications": response_modifications,
            "blended_expression": blended_expression,
            "conditioning_reinforcements": reinforcements,
            "processing_confidence": await self._calculate_processing_confidence(processing_results),
            "key_patterns": self._extract_key_patterns(processing_results)
        }
        
        # Send synthesis to response generation
        await self.send_context_update(
            update_type="input_processing_synthesis_complete",
            data=processing_synthesis,
            priority=ContextPriority.HIGH
        )
        
        return {
            "processing_synthesis": processing_synthesis,
            "synthesis_complete": True
        }
    
    # Enhanced helper methods
    
    async def _get_mode_distribution_from_context(self, context: SharedContext) -> Dict[str, float]:
        """Extract mode distribution from context"""
        # Check context for mode information
        if context.mode_context and "mode_distribution" in context.mode_context:
            return context.mode_context["mode_distribution"]
        
        # Try to get from mode manager
        if hasattr(self.original_processor, 'context') and self.original_processor.context.mode_manager:
            try:
                mode_dist = self.original_processor.context.mode_manager.context.mode_distribution
                if hasattr(mode_dist, 'dict'):
                    return mode_dist.dict()
                elif isinstance(mode_dist, dict):
                    return mode_dist
            except:
                pass
        
        # Default distribution
        return {"friendly": 1.0}
    
    async def _detect_initial_patterns(self, user_input: str) -> List[Dict[str, Any]]:
        """Detect initial patterns in user input"""
        # Use original processor's pattern detection
        result = await self.original_processor.process_input(
            text=user_input,
            user_id="context_user",
            context={}
        )
        
        return result.get("detected_patterns", [])
    
    async def _update_mode_blending(self, mode_data: Dict[str, Any]):
        """Update mode blending parameters"""
        mode_distribution = mode_data.get("mode_distribution", {})
        
        # Cache the distribution
        if hasattr(self.original_processor, 'context') and self.original_processor.context.mode_manager:
            self.original_processor.context.mode_manager.context.mode_distribution = mode_distribution
        
        # Track blending history
        self.mode_blending_history.append({
            "timestamp": datetime.now(),
            "distribution": mode_distribution,
            "dominant_mode": mode_data.get("dominant_mode")
        })
        
        # Send update about mode processing readiness
        await self.send_context_update(
            update_type="mode_blending_updated",
            data={
                "current_distribution": mode_distribution,
                "blending_ready": True
            }
        )
    
    async def _adjust_pattern_sensitivity(self, emotional_data: Dict[str, Any]):
        """Adjust pattern detection sensitivity based on emotional state"""
        emotion_intensity = 0.0
        dominant_emotion = emotional_data.get("dominant_emotion")
        try
            if isinstance(dominant_emotion, tuple) and len(dominant_emotion) >= 2:
                emotion_name, emotion_intensity = dominant_emotion[0], dominant_emotion[1]
            elif isinstance(dominant_emotion, dict):
                emotion_name = dominant_emotion.get("name", "")
                emotion_intensity = dominant_emotion.get("intensity", 0.0)
            else:
                emotion_name = ""
            
            # High emotional intensity increases pattern sensitivity
            sensitivity_boost = emotion_intensity * 0.3
            
            # Specific emotions affect specific patterns
            pattern_adjustments = {}
            
            if emotion_name == "Anxiety":
                pattern_adjustments["defiance"] = 0.2  # More sensitive to defiance
                pattern_adjustments["disrespect"] = 0.3  # More sensitive to disrespect
            elif emotion_name == "Joy":
                pattern_adjustments["flattery"] = 0.2  # More receptive to flattery
                pattern_adjustments["submission_language"] = 0.1  # Slightly more sensitive
            elif emotion_name == "Frustration":
                pattern_adjustments["defiance"] = 0.4  # Much more sensitive to defiance
            
            # Cache adjustments
            self.pattern_detection_cache["sensitivity_adjustments"] = {
                "base_boost": sensitivity_boost,
                "pattern_specific": pattern_adjustments,
                "emotional_context": emotion_name
            }
        except Exception as e:
            logger.error(f"Error adjusting pattern sensitivity: {e}")
            # Notify other modules of the error
            await self.send_context_update(
                update_type="processing_error",
                data={"error": str(e), "component": "pattern_sensitivity"},
                priority=ContextPriority.HIGH
            )
            # Don't swallow the error for critical failures
            if "AttributeError" in str(e) or "TypeError" in str(e):
                raise

    def _get_context_data(self, context: SharedContext, module: str, key: str, default: Any = None) -> Any:
        """Efficiently get data from shared context"""
        # Check module-specific state first
        for update in context.context_updates:
            if update.source_module == module and key in update.data:
                return update.data[key]
        
        # Check general context attributes
        if hasattr(context, key):
            return getattr(context, key)
        
        # Check context dictionaries
        for ctx_dict in [context.emotional_state, context.mode_context, 
                         context.relationship_context, context.goal_context]:
            if ctx_dict and key in ctx_dict:
                return ctx_dict[key]
        
        return default
    
    async def _adjust_conditioning_thresholds(self, relationship_data: Dict[str, Any]):
        """Adjust conditioning thresholds based on relationship state"""
        relationship_context = relationship_data.get("relationship_context", {})
        
        # Extract relationship metrics
        trust = relationship_context.get("trust", 0.5)
        intimacy = relationship_context.get("intimacy", 0.5)
        dominance_accepted = relationship_context.get("dominance_accepted", 0.5)
        depth = relationship_context.get("depth", 0.5)
        conflict = relationship_context.get("conflict", 0.0)
        
        # Calculate conditioning adjustments
        # High trust allows stronger conditioning (0.7 to 1.0 range)
        conditioning_multiplier = 0.7 + (trust * 0.3)
        
        # High intimacy allows more subtle conditioning
        subtlety_factor = intimacy * 0.4
        
        # Dominance acceptance affects specific behaviors
        dominance_conditioning = dominance_accepted * 0.5
        
        # Conflict reduces conditioning effectiveness
        if conflict > 0.5:
            conditioning_multiplier *= (1.0 - conflict * 0.3)
        
        # Update conditioning context
        if hasattr(self.original_processor, 'context') and self.original_processor.context.conditioning_system:
            conditioning_system = self.original_processor.context.conditioning_system
            
            # Update system parameters if methods are available
            if hasattr(conditioning_system, 'set_conditioning_parameters'):
                try:
                    await conditioning_system.set_conditioning_parameters(
                        base_multiplier=conditioning_multiplier,
                        subtlety_factor=subtlety_factor,
                        context_factors={
                            "relationship_trust": trust,
                            "relationship_intimacy": intimacy,
                            "dominance_accepted": dominance_accepted,
                            "relationship_depth": depth
                        }
                    )
                except Exception as e:
                    logger.debug(f"Could not update conditioning parameters: {e}")
            
            # Update specific behavior thresholds
            if hasattr(conditioning_system, 'update_behavior_thresholds'):
                try:
                    behavior_thresholds = {
                        "submission_language_response": {
                            "threshold": 0.5 - (dominance_conditioning * 0.2),  # Lower threshold with acceptance
                            "intensity_multiplier": 1.0 + (dominance_conditioning * 0.3)
                        },
                        "defiance_response": {
                            "threshold": 0.6 - (trust * 0.1),  # Slightly lower with trust
                            "intensity_multiplier": 1.0 - (intimacy * 0.2)  # Less intense with intimacy
                        },
                        "positive_engagement": {
                            "threshold": 0.4 - (depth * 0.1),  # Easier to trigger with depth
                            "intensity_multiplier": 1.0 + (trust * 0.2)
                        }
                    }
                    
                    await conditioning_system.update_behavior_thresholds(behavior_thresholds)
                except Exception as e:
                    logger.debug(f"Could not update behavior thresholds: {e}")
        
        # Cache the adjustments for later use
        self.behavior_evaluation_cache["conditioning_adjustments"] = {
            "multiplier": conditioning_multiplier,
            "subtlety": subtlety_factor,
            "dominance_factor": dominance_conditioning,
            "trust_factor": trust,
            "intimacy_factor": intimacy,
            "depth_factor": depth,
            "conflict_factor": conflict,
            "timestamp": datetime.now(),
            "source": "relationship_update"
        }
        
        # Send update about conditioning readiness
        if hasattr(self, 'send_context_update'):
            await self.send_context_update(
                update_type="conditioning_thresholds_adjusted",
                data={
                    "adjustments": self.behavior_evaluation_cache["conditioning_adjustments"],
                    "relationship_metrics": {
                        "trust": trust,
                        "intimacy": intimacy,
                        "dominance_accepted": dominance_accepted,
                        "depth": depth,
                        "conflict": conflict
                    }
                }
            )
        
        logger.debug(f"Adjusted conditioning thresholds - multiplier: {conditioning_multiplier:.2f}, "
                    f"subtlety: {subtlety_factor:.2f}, dominance: {dominance_conditioning:.2f}")
    
    async def _process_conditioning_trigger(self, conditioning_data: Dict[str, Any]):
        """Process a direct conditioning trigger"""
        trigger_type = conditioning_data.get("type", "")
        behavior = conditioning_data.get("behavior", "")
        intensity = conditioning_data.get("intensity", 0.5)
        
        # Process through conditioning system
        if hasattr(self.original_processor, 'context') and self.original_processor.context.conditioning_system:
            # Direct conditioning application
            result = await self.original_processor.context.conditioning_system.process_operant_conditioning(
                behavior=behavior,
                consequence_type=trigger_type,
                intensity=intensity,
                context=conditioning_data.get("context", {})
            )
            
            # Send result
            await self.send_context_update(
                update_type="conditioning_applied",
                data={
                    "behavior": behavior,
                    "type": trigger_type,
                    "result": result
                },
                target_modules=["conditioning_system"],
                scope=ContextScope.TARGETED
            )
    
    async def _update_behavior_baselines(self, behavior_data: Dict[str, Any]):
        """Update baseline behavior patterns"""
        patterns = behavior_data.get("patterns", {})
        
        # Update evaluation baselines
        for behavior, frequency in patterns.items():
            if behavior not in self.behavior_evaluation_cache:
                self.behavior_evaluation_cache[behavior] = {
                    "baseline_frequency": frequency,
                    "recent_occurrences": [],
                    "reinforcement_history": []
                }
            else:
                self.behavior_evaluation_cache[behavior]["baseline_frequency"] = frequency
    
    async def _activate_dominance_processing(self, dominance_data: Dict[str, Any]):
        """Activate special processing for dominance contexts"""
        dominance_type = dominance_data.get("dominance_type", "general")
        intensity = dominance_data.get("intensity", 0.5)
        
        # Enhance pattern detection for dominance-relevant patterns
        dominance_patterns = {
            "submission_language": 2.0,  # Double sensitivity
            "defiance": 2.5,  # Even more sensitive to defiance
            "embarrassment": 1.5,  # Moderately more sensitive
            "flattery": 0.8  # Slightly less sensitive (might be manipulation)
        }
        
        # Cache dominance processing state
        self.pattern_detection_cache["dominance_active"] = {
            "type": dominance_type,
            "intensity": intensity,
            "pattern_multipliers": dominance_patterns,
            "activated_at": datetime.now()
        }
        
        # Notify other modules
        await self.send_context_update(
            update_type="dominance_processing_active",
            data={
                "processing_mode": "dominance_enhanced",
                "pattern_sensitivity": dominance_patterns
            }
        )
    
    async def _prepare_response_modification(self, response_data: Dict[str, Any]):
        """Prepare to modify an outgoing response"""
        response_text = response_data.get("text", "")
        response_context = response_data.get("context", {})
        
        # Get current processing results
        processing_results = await self._get_accumulated_processing_results()
        
        # Prepare modification instructions
        modification_instructions = {
            "base_text": response_text,
            "processing_results": processing_results,
            "modification_ready": True
        }
        
        # Cache for synthesis phase
        self.pattern_detection_cache["pending_response"] = modification_instructions
    
    async def _align_conditioning_with_goals(self, goal_data: Dict[str, Any]):
        """Align conditioning with active goals"""
        active_goals = goal_data.get("active_goals", [])
        
        # Map goals to conditioning priorities
        conditioning_priorities = {}
        
        for goal in active_goals:
            goal_need = goal.get("associated_need", "")
            priority = goal.get("priority", 0.5)
            
            if goal_need == "control_expression" and priority > 0.7:
                # Prioritize submission reinforcement
                conditioning_priorities["submission_language_response"] = priority
                conditioning_priorities["tolerate_defiance"] = -priority  # Negative reinforcement
            elif goal_need == "connection" and priority > 0.7:
                # Prioritize positive interaction reinforcement
                conditioning_priorities["positive_engagement"] = priority
                conditioning_priorities["emotional_openness"] = priority * 0.8
        
        # Update conditioning system priorities
        self.behavior_evaluation_cache["goal_aligned_priorities"] = conditioning_priorities
    
    async def _update_processing_preferences(self, identity_data: Dict[str, Any]):
        """Update processing preferences based on identity"""
        traits = identity_data.get("traits", {})
        
        # Map traits to processing preferences
        processing_preferences = {}
        
        # Dominance trait affects pattern interpretation
        if traits.get("dominance", 0) > 0.7:
            processing_preferences["assertive_interpretation"] = True
            processing_preferences["submission_recognition_boost"] = 0.3
        
        # Playfulness affects tone interpretation
        if traits.get("playfulness", 0) > 0.6:
            processing_preferences["playful_interpretation"] = True
            processing_preferences["humor_detection_boost"] = 0.2
        
        # Empathy affects emotional pattern detection
        if traits.get("empathy", 0) > 0.7:
            processing_preferences["emotional_sensitivity"] = 0.3
            processing_preferences["distress_detection_boost"] = 0.4
        
        # Cache preferences
        self.pattern_detection_cache["identity_preferences"] = processing_preferences
    
    async def _comprehensive_pattern_detection(self, user_input: str, 
                                             context: SharedContext, 
                                             messages: Dict) -> Dict[str, Any]:
        """Comprehensive pattern detection with full context"""
        # Start with base detection
        base_patterns = await self._detect_initial_patterns(user_input)
        
        # Get sensitivity adjustments from context efficiently
        sensitivity = self._get_context_data(context, "input_processor", "sensitivity_adjustments", {})
        if not sensitivity:
            sensitivity = self.pattern_detection_cache.get("sensitivity_adjustments", {})
        
        enhanced_patterns = []
        for pattern in base_patterns:
            pattern_name = pattern["pattern_name"]
            base_confidence = pattern["confidence"]
            
            # Apply base sensitivity boost
            adjusted_confidence = base_confidence + sensitivity.get("base_boost", 0)
            
            # Apply pattern-specific adjustments
            if pattern_name in sensitivity.get("pattern_specific", {}):
                adjusted_confidence += sensitivity["pattern_specific"][pattern_name]
            
            # Apply dominance processing if active
            if "dominance_active" in self.pattern_detection_cache:
                dominance_data = self.pattern_detection_cache["dominance_active"]
                if pattern_name in dominance_data["pattern_multipliers"]:
                    adjusted_confidence *= dominance_data["pattern_multipliers"][pattern_name]
            
            # Apply identity preferences
            identity_prefs = self.pattern_detection_cache.get("identity_preferences", {})
            if pattern_name == "submission_language" and identity_prefs.get("submission_recognition_boost"):
                adjusted_confidence += identity_prefs["submission_recognition_boost"]
            
            # Ensure confidence stays in valid range
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
            
            enhanced_patterns.append({
                "pattern_name": pattern_name,
                "confidence": adjusted_confidence,
                "matched_text": pattern["matched_text"],
                "context_enhanced": True,
                "adjustments_applied": {
                    "emotional": bool(sensitivity),
                    "dominance": "dominance_active" in self.pattern_detection_cache,
                    "identity": bool(identity_prefs)
                }
            })
        
        # Check for cross-module pattern suggestions
        for module_name, module_messages in messages.items():
            if module_name == "emotional_core":
                for msg in module_messages:
                    if msg["type"] == "emotional_pattern_detected":
                        # Add emotional patterns
                        enhanced_patterns.append({
                            "pattern_name": f"emotional_{msg['data']['emotion']}",
                            "confidence": msg['data'].get('confidence', 0.7),
                            "matched_text": user_input,
                            "source": "emotional_core"
                        })
        
        return {
            "patterns": enhanced_patterns,
            "pattern_count": len(enhanced_patterns),
            "context_factors": {
                "emotional_sensitivity": bool(sensitivity),
                "dominance_active": "dominance_active" in self.pattern_detection_cache,
                "identity_influenced": bool(self.pattern_detection_cache.get("identity_preferences"))
            }
        }
    
    async def _contextual_behavior_evaluation(self, pattern_analysis: Dict[str, Any], 
                                            context: SharedContext, 
                                            messages: Dict) -> Dict[str, Any]:
        """Evaluate behaviors with full context"""
        patterns = pattern_analysis.get("patterns", [])
        
        # Get conditioning adjustments
        conditioning_adj = self.behavior_evaluation_cache.get("conditioning_adjustments", {})
        multiplier = conditioning_adj.get("multiplier", 1.0)
        
        # Get goal-aligned priorities
        goal_priorities = self.behavior_evaluation_cache.get("goal_aligned_priorities", {})
        
        # Get emotional influence
        emotional_influence = {}
        if context.emotional_state:
            emotion_name = context.emotional_state.get("dominant_emotion", "")
            emotion_intensity = context.emotional_state.get("intensity", 0.5)
            emotional_influence = {
                "emotion": emotion_name,
                "intensity": emotion_intensity,
                "influence_factor": emotion_intensity * 0.3
            }
    
        
        # Evaluate each potential behavior
        behavior_evaluations = []
        
        potential_behaviors = ["dominant_response", "teasing_response", "direct_response", 
                             "playful_response", "nurturing_response", "strict_response"]
        
        for behavior in potential_behaviors:
            # Base evaluation
            base_eval = await self._evaluate_single_behavior(behavior, patterns, context)
            
            # Apply conditioning multiplier
            adjusted_confidence = base_eval["confidence"] * multiplier
            
            # Apply goal priorities
            if behavior in goal_priorities:
                priority_boost = goal_priorities[behavior] * 0.2
                adjusted_confidence += priority_boost
            
            # Apply relationship context
            if context.relationship_context:
                relationship_adjustment = self._get_relationship_behavior_adjustment(
                    behavior, context.relationship_context
                )
                adjusted_confidence *= relationship_adjustment
            
            # Ensure valid range
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
            
            behavior_evaluations.append({
                "behavior": behavior,
                "recommendation": base_eval["recommendation"],
                "confidence": adjusted_confidence,
                "reasoning": base_eval["reasoning"],
                "context_factors": {
                    "conditioning_multiplier": multiplier,
                    "goal_alignment": behavior in goal_priorities,
                    "relationship_adjusted": bool(context.relationship_context)
                }
            })
        
        # Sort by confidence
        behavior_evaluations.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "evaluations": behavior_evaluations,
            "recommended_behaviors": [e["behavior"] for e in behavior_evaluations 
                                    if e["recommendation"] == "approach" and e["confidence"] > 0.6],
            "avoided_behaviors": [e["behavior"] for e in behavior_evaluations 
                                if e["recommendation"] == "avoid" and e["confidence"] > 0.6]
        }
    
    async def _process_contextual_conditioning(self, pattern_analysis: Dict[str, Any], 
                                             behavior_evaluation: Dict[str, Any], 
                                             context: SharedContext, 
                                             messages: Dict) -> List[Dict[str, Any]]:
        """Process conditioning with full context awareness"""
        conditioning_results = []
        patterns = pattern_analysis.get("patterns", [])
        
        # Get conditioning adjustments
        conditioning_adj = self.behavior_evaluation_cache.get("conditioning_adjustments", {})
        
        # Process each detected pattern
        for pattern in patterns:
            pattern_name = pattern["pattern_name"]
            confidence = pattern["confidence"]
            
            # Only process high-confidence patterns
            if confidence < 0.6:
                continue
            
            # Determine conditioning type and intensity
            conditioning_spec = self._determine_conditioning_spec(
                pattern_name, confidence, conditioning_adj
            )
            
            if conditioning_spec:
                # Apply conditioning
                result = {
                    "behavior": conditioning_spec["behavior"],
                    "consequence_type": conditioning_spec["type"],
                    "intensity": conditioning_spec["intensity"],
                    "pattern_trigger": pattern_name,
                    "context": {
                        "user_id": context.user_id,
                        "relationship_depth": context.relationship_context.get("depth", 0.5) if context.relationship_context else 0.5,
                        "emotional_state": context.emotional_state
                    }
                }
                
                conditioning_results.append(result)
                
                # Send to conditioning system
                await self.send_context_update(
                    update_type="conditioning_trigger",
                    data=result,
                    target_modules=["conditioning_system"],
                    scope=ContextScope.TARGETED
                )
        
        return conditioning_results
    
    async def _process_mode_specific_input(self, user_input: str, 
                                         context: SharedContext, 
                                         messages: Dict) -> Dict[str, Any]:
        """Process input with mode-specific handling"""
        mode_distribution = await self._get_mode_distribution_from_context(context)
        
        # Get dominant mode
        dominant_mode = max(mode_distribution.items(), key=lambda x: x[1])[0] if mode_distribution else "friendly"
        
        # Mode-specific processing
        mode_processing = {
            "dominant_mode": dominant_mode,
            "mode_weights": mode_distribution,
            "adjustments": {}
        }
        
        # Apply mode-specific adjustments
        if dominant_mode == "dominant":
            mode_processing["adjustments"] = {
                "authority_emphasis": 0.3,
                "submission_recognition": 0.4,
                "defiance_sensitivity": 0.5
            }
        elif dominant_mode == "playful":
            mode_processing["adjustments"] = {
                "humor_detection": 0.3,
                "lightness_preference": 0.4,
                "formality_reduction": -0.3
            }
        elif dominant_mode == "intellectual":
            mode_processing["adjustments"] = {
                "complexity_tolerance": 0.4,
                "analysis_depth": 0.5,
                "precision_requirement": 0.3
            }
        
        return mode_processing
    
    async def _analyze_pattern_effectiveness(self, context: SharedContext, 
                                           messages: Dict) -> Dict[str, Any]:
        """Analyze effectiveness of pattern detection"""
        effectiveness = {
            "detection_accuracy": 0.0,
            "false_positives": [],
            "missed_patterns": [],
            "sensitivity_calibration": "optimal"
        }
        
        # Check cross-module feedback
        for module_name, module_messages in messages.items():
            if module_name == "user_model":
                for msg in module_messages:
                    if msg["type"] == "pattern_feedback":
                        feedback = msg["data"]
                        if feedback.get("false_positive"):
                            effectiveness["false_positives"].append(feedback["pattern"])
                        elif feedback.get("missed_pattern"):
                            effectiveness["missed_patterns"].append(feedback["pattern"])
        
        # Calculate accuracy
        total_patterns = len(self.pattern_detection_cache.get("recent_detections", []))
        false_positive_count = len(effectiveness["false_positives"])
        
        if total_patterns > 0:
            effectiveness["detection_accuracy"] = 1.0 - (false_positive_count / total_patterns)
        else:
            effectiveness["detection_accuracy"] = 0.5  # No data
        
        # Determine calibration
        if false_positive_count > total_patterns * 0.2:
            effectiveness["sensitivity_calibration"] = "too_sensitive"
        elif len(effectiveness["missed_patterns"]) > 3:
            effectiveness["sensitivity_calibration"] = "not_sensitive_enough"
        
        return effectiveness
    
    async def _analyze_conditioning_impact(self, messages: Dict) -> Dict[str, Any]:
        """Analyze impact of conditioning on user behavior"""
        impact = {
            "behavior_changes": {},
            "reinforcement_effectiveness": 0.0,
            "user_adaptation": "stable"
        }
        
        # Check for behavior change reports
        for module_name, module_messages in messages.items():
            if module_name == "user_model":
                for msg in module_messages:
                    if msg["type"] == "behavior_change":
                        behavior = msg["data"]["behavior"]
                        direction = msg["data"]["direction"]
                        magnitude = msg["data"].get("magnitude", 0.1)
                        
                        impact["behavior_changes"][behavior] = {
                            "direction": direction,
                            "magnitude": magnitude
                        }
        
        # Calculate effectiveness
        if impact["behavior_changes"]:
            positive_changes = sum(1 for b in impact["behavior_changes"].values() 
                                 if b["direction"] == "increased" and b["magnitude"] > 0.1)
            
            impact["reinforcement_effectiveness"] = positive_changes / len(impact["behavior_changes"])
            
            # Determine adaptation
            total_magnitude = sum(b["magnitude"] for b in impact["behavior_changes"].values())
            if total_magnitude > 0.5:
                impact["user_adaptation"] = "rapid"
            elif total_magnitude < 0.1:
                impact["user_adaptation"] = "resistant"
        
        return impact
    
    async def _analyze_blending_coherence(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze coherence of mode blending"""
        coherence = {
            "blend_stability": 0.0,
            "mode_conflicts": [],
            "transition_smoothness": 0.0
        }
        
        # Analyze recent blending history
        if len(self.mode_blending_history) >= 2:
            # Check stability
            recent_distributions = [h["distribution"] for h in self.mode_blending_history[-5:]]
            
            # Calculate variance in distributions
            mode_variances = {}
            for mode in recent_distributions[0].keys():
                values = [d.get(mode, 0) for d in recent_distributions]
                variance = sum((v - sum(values)/len(values))**2 for v in values) / len(values)
                mode_variances[mode] = variance
            
            # Low variance = high stability
            avg_variance = sum(mode_variances.values()) / len(mode_variances) if mode_variances else 0
            coherence["blend_stability"] = 1.0 - min(1.0, avg_variance * 5)
            
            # Check for conflicts (opposing modes with high weights)
            current_dist = self.mode_blending_history[-1]["distribution"]
            
            conflict_pairs = [
                ("dominant", "nurturing"),
                ("playful", "professional"),
                ("creative", "analytical")
            ]
            
            for mode1, mode2 in conflict_pairs:
                if current_dist.get(mode1, 0) > 0.3 and current_dist.get(mode2, 0) > 0.3:
                    coherence["mode_conflicts"].append((mode1, mode2))
            
            # Check transition smoothness
            if len(self.mode_blending_history) >= 3:
                transitions = []
                for i in range(1, len(self.mode_blending_history)):
                    prev_dist = self.mode_blending_history[i-1]["distribution"]
                    curr_dist = self.mode_blending_history[i]["distribution"]
                    
                    # Calculate transition magnitude
                    transition_mag = sum(abs(curr_dist.get(m, 0) - prev_dist.get(m, 0)) 
                                       for m in set(curr_dist.keys()) | set(prev_dist.keys()))
                    transitions.append(transition_mag)
                
                # Smooth transitions have low magnitude
                avg_transition = sum(transitions) / len(transitions) if transitions else 0
                coherence["transition_smoothness"] = 1.0 - min(1.0, avg_transition)
        
        return coherence
    
    async def _generate_processing_insights(self, pattern_effectiveness: Dict[str, Any], 
                                          conditioning_impact: Dict[str, Any], 
                                          blending_coherence: Dict[str, Any]) -> List[str]:
        """Generate insights from processing analysis"""
        insights = []
        
        # Pattern effectiveness insights
        if pattern_effectiveness["detection_accuracy"] > 0.9:
            insights.append("Pattern detection performing with high accuracy")
        elif pattern_effectiveness["detection_accuracy"] < 0.7:
            insights.append("Pattern detection needs calibration")
        
        if pattern_effectiveness["sensitivity_calibration"] == "too_sensitive":
            insights.append("Reduce pattern detection sensitivity to avoid false positives")
        
        # Conditioning impact insights
        if conditioning_impact["reinforcement_effectiveness"] > 0.7:
            insights.append("Conditioning showing strong positive impact on behavior")
        elif conditioning_impact["reinforcement_effectiveness"] < 0.3:
            insights.append("Conditioning effectiveness low - consider alternative approaches")
        
        if conditioning_impact["user_adaptation"] == "rapid":
            insights.append("User showing rapid behavioral adaptation")
        elif conditioning_impact["user_adaptation"] == "resistant":
            insights.append("User showing resistance to behavioral conditioning")
        
        # Blending coherence insights
        if blending_coherence["blend_stability"] < 0.5:
            insights.append("Mode blending showing instability - consider stabilization")
        
        if blending_coherence["mode_conflicts"]:
            conflicts_str = ", ".join([f"{m1}-{m2}" for m1, m2 in blending_coherence["mode_conflicts"]])
            insights.append(f"Mode conflicts detected: {conflicts_str}")
        
        if blending_coherence["transition_smoothness"] > 0.8:
            insights.append("Mode transitions are smooth and natural")
        
        return insights
    
    async def _get_accumulated_processing_results(self) -> Dict[str, Any]:
        """Get accumulated processing results from the session"""
        results = {
            "pattern_detections": [],
            "behavior_evaluations": [],
            "mode_processing": {},
            "conditioning_applied": []
        }
        
        # Get recent pattern detections from cache
        if "recent_detections" in self.pattern_detection_cache:
            results["pattern_detections"] = self.pattern_detection_cache["recent_detections"]
        else:
            # Build from current session if available
            recent_detections = []
            
            # Check for dominance processing patterns
            if "dominance_active" in self.pattern_detection_cache:
                dominance_data = self.pattern_detection_cache["dominance_active"]
                recent_detections.append({
                    "pattern_name": "dominance_context",
                    "confidence": dominance_data.get("intensity", 0.5),
                    "timestamp": dominance_data.get("activated_at", datetime.now()),
                    "context": "dominance_processing"
                })
            
            # Check for sensitivity adjustments (indicates detected patterns)
            if "sensitivity_adjustments" in self.pattern_detection_cache:
                adjustments = self.pattern_detection_cache["sensitivity_adjustments"]
                if "emotional_context" in adjustments:
                    recent_detections.append({
                        "pattern_name": f"emotional_{adjustments['emotional_context'].lower()}",
                        "confidence": 0.7,
                        "context": "emotional_sensitivity"
                    })
            
            results["pattern_detections"] = recent_detections
            # Cache for future use
            self.pattern_detection_cache["recent_detections"] = recent_detections
        
        # Get behavior evaluations from cache
        if self.behavior_evaluation_cache:
            behavior_evals = []
            
            for behavior, data in self.behavior_evaluation_cache.items():
                if isinstance(data, dict):
                    # Format as evaluation result
                    eval_result = {
                        "behavior": behavior,
                        "baseline_frequency": data.get("baseline_frequency", 0.5),
                        "recent_occurrences": data.get("recent_occurrences", []),
                        "reinforcement_history": data.get("reinforcement_history", [])
                    }
                    behavior_evals.append(eval_result)
            
            # Add goal-aligned priorities if available
            if "goal_aligned_priorities" in self.behavior_evaluation_cache:
                priorities = self.behavior_evaluation_cache["goal_aligned_priorities"]
                for behavior, priority in priorities.items():
                    behavior_evals.append({
                        "behavior": behavior,
                        "goal_priority": priority,
                        "source": "goal_alignment"
                    })
            
            # Add conditioning adjustments if available
            if "conditioning_adjustments" in self.behavior_evaluation_cache:
                adjustments = self.behavior_evaluation_cache["conditioning_adjustments"]
                behavior_evals.append({
                    "type": "conditioning_parameters",
                    "multiplier": adjustments.get("multiplier", 1.0),
                    "subtlety": adjustments.get("subtlety", 0.5),
                    "dominance_factor": adjustments.get("dominance_factor", 0.0)
                })
            
            results["behavior_evaluations"] = behavior_evals
        
        # Get mode processing from history
        if self.mode_blending_history:
            # Get most recent mode processing
            recent_mode = self.mode_blending_history[-1]
            results["mode_processing"] = recent_mode
        else:
            # Try to get from current context
            try:
                mode_dist = await self._get_mode_distribution_from_context(SharedContext())
                results["mode_processing"] = {
                    "distribution": mode_dist,
                    "timestamp": datetime.now()
                }
            except:
                results["mode_processing"] = {"distribution": {"friendly": 1.0}}
        
        # Get recent conditioning applications
        if "recent_conditioning" in self.behavior_evaluation_cache:
            results["conditioning_applied"] = self.behavior_evaluation_cache["recent_conditioning"]
        else:
            # Build from reinforcement history
            recent_conditioning = []
            
            for behavior, data in self.behavior_evaluation_cache.items():
                if isinstance(data, dict) and "reinforcement_history" in data:
                    for reinforcement in data["reinforcement_history"][-5:]:  # Last 5
                        recent_conditioning.append({
                            "behavior": behavior,
                            "type": reinforcement.get("type", "unknown"),
                            "timestamp": reinforcement.get("timestamp", datetime.now()),
                            "intensity": reinforcement.get("intensity", 0.5)
                        })
            
            results["conditioning_applied"] = recent_conditioning
            # Cache for future use
            self.behavior_evaluation_cache["recent_conditioning"] = recent_conditioning
        
        # Add any pending response modifications
        if "pending_response" in self.pattern_detection_cache:
            results["pending_modifications"] = self.pattern_detection_cache["pending_response"]
        
        return results
    
    async def _synthesize_response_modifications(self, processing_results: Dict[str, Any], 
                                               context: SharedContext, 
                                               messages: Dict) -> Dict[str, Any]:
        """Synthesize response modifications from processing"""
        modifications = {
            "text_adjustments": [],
            "tone_shifts": {},
            "emphasis_patterns": [],
            "mode_expression": {}
        }
        
        # Get pending response if any
        pending_response = self.pattern_detection_cache.get("pending_response", {})
        
        if not pending_response:
            return modifications
        
        # Based on detected patterns
        patterns = processing_results.get("pattern_detections", [])
        
        for pattern in patterns:
            if pattern["pattern_name"] == "submission_language" and pattern["confidence"] > 0.7:
                modifications["text_adjustments"].append({
                    "type": "reinforce_authority",
                    "strength": pattern["confidence"]
                })
                modifications["tone_shifts"]["authoritative"] = 0.3
            
            elif pattern["pattern_name"] == "defiance" and pattern["confidence"] > 0.6:
                modifications["text_adjustments"].append({
                    "type": "increase_firmness",
                    "strength": pattern["confidence"]
                })
                modifications["tone_shifts"]["stern"] = 0.4
            
            elif pattern["pattern_name"] == "embarrassment" and pattern["confidence"] > 0.6:
                modifications["text_adjustments"].append({
                    "type": "gentle_teasing",
                    "strength": pattern["confidence"] * 0.7
                })
                modifications["tone_shifts"]["playful"] = 0.2
        
        # Based on mode distribution
        mode_dist = context.mode_context.get("mode_distribution", {}) if context.mode_context else {}
        
        for mode, weight in mode_dist.items():
            if weight > 0.2:
                modifications["mode_expression"][mode] = weight
        
        # Add emphasis patterns based on goals
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            for goal in active_goals:
                if goal.get("priority", 0) > 0.8:
                    modifications["emphasis_patterns"].append({
                        "type": "goal_reinforcement",
                        "goal": goal.get("description", ""),
                        "emphasis": 0.3
                    })
        
        return modifications
    
    async def _synthesize_blended_expression(self, context: SharedContext, 
                                           messages: Dict) -> Dict[str, Any]:
        """Synthesize blended mode expression"""
        mode_dist = await self._get_mode_distribution_from_context(context)
        
        # Use original processor's blending capability
        if hasattr(self.original_processor, 'modify_blended_response'):
            # Get a sample text for testing
            sample_text = "I understand what you're saying."
            
            blended_result = await self.original_processor.modify_blended_response(
                response_text=sample_text,
                mode_distribution=mode_dist
            )
            
            return {
                "mode_distribution": mode_dist,
                "blending_active": True,
                "coherence": blended_result.get("coherence", 0.5),
                "style_notes": blended_result.get("style_notes", ""),
                "mode_influences": blended_result.get("mode_influences", {})
            }
        
        return {
            "mode_distribution": mode_dist,
            "blending_active": False,
            "coherence": 0.5
        }
    
    async def _synthesize_conditioning_reinforcements(self, processing_results: Dict[str, Any], 
                                                    context: SharedContext) -> List[Dict[str, Any]]:
        """Synthesize conditioning reinforcements for response"""
        reinforcements = []
        
        # Based on behavior evaluations
        evaluations = processing_results.get("behavior_evaluations", [])
        
        for eval_data in evaluations:
            if isinstance(eval_data, dict) and "baseline_frequency" in eval_data:
                # This is cached behavior data
                behavior = list(eval_data.keys())[0] if eval_data else "unknown"
                frequency = eval_data.get("baseline_frequency", 0)
                
                # Reinforce desired behaviors with low frequency
                if behavior in ["submission_language_response", "positive_engagement"] and frequency < 0.3:
                    reinforcements.append({
                        "behavior": behavior,
                        "type": "positive_reinforcement",
                        "intensity": 0.6,
                        "reason": "encourage_desired_behavior"
                    })
                
                # Discourage undesired behaviors with high frequency
                elif behavior in ["defiance_response", "disrespect_response"] and frequency > 0.5:
                    reinforcements.append({
                        "behavior": behavior,
                        "type": "positive_punishment",
                        "intensity": 0.5,
                        "reason": "discourage_undesired_behavior"
                    })
        
        # Based on relationship context
        if context.relationship_context:
            trust = context.relationship_context.get("trust", 0.5)
            
            # High trust allows stronger reinforcement
            for reinforcement in reinforcements:
                reinforcement["intensity"] *= (0.7 + trust * 0.3)
        
        return reinforcements
    
    async def _calculate_processing_confidence(self, processing_results: Dict[str, Any]) -> float:
        """Calculate overall confidence in processing results"""
        confidence_factors = []
        
        # Pattern detection confidence
        patterns = processing_results.get("pattern_detections", [])
        if patterns:
            avg_pattern_confidence = sum(p.get("confidence", 0.5) for p in patterns) / len(patterns)
            confidence_factors.append(avg_pattern_confidence)
        
        # Behavior evaluation confidence
        evaluations = processing_results.get("behavior_evaluations", [])
        if evaluations:
            # This is simplified - would need proper extraction
            confidence_factors.append(0.7)  # Default confidence
        
        # Mode processing stability
        if self.mode_blending_history:
            # Recent stability affects confidence
            recent_history = self.mode_blending_history[-3:]
            if len(recent_history) >= 2:
                # Check consistency
                dominant_modes = [h.get("dominant_mode") for h in recent_history]
                if len(set(dominant_modes)) == 1:  # Same dominant mode
                    confidence_factors.append(0.9)
                else:
                    confidence_factors.append(0.6)
        
        # Calculate overall confidence
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        
        return 0.5  # Default confidence
    
    def _extract_key_patterns(self, processing_results: Dict[str, Any]) -> List[str]:
        """Extract key patterns from processing results"""
        key_patterns = []
        seen_patterns = set()  # To avoid duplicates
        
        # Extract from pattern detections
        patterns = processing_results.get("pattern_detections", [])
        
        # Sort by confidence and recency
        sorted_patterns = sorted(
            patterns,
            key=lambda p: (
                p.get("confidence", 0),
                # Prefer more recent patterns
                p.get("timestamp", datetime.min) if isinstance(p.get("timestamp"), datetime) else datetime.min
            ),
            reverse=True
        )
        
        # Get top patterns with high confidence
        for pattern in sorted_patterns:
            if isinstance(pattern, dict):
                pattern_name = pattern.get("pattern_name", "unknown")
                confidence = pattern.get("confidence", 0)
                
                # Only include high-confidence patterns
                if confidence > 0.7 and pattern_name not in seen_patterns:
                    key_patterns.append(pattern_name)
                    seen_patterns.add(pattern_name)
                    
                    # Limit to top 3 patterns
                    if len(key_patterns) >= 3:
                        break
        
        # Extract from behavior evaluations if we need more patterns
        if len(key_patterns) < 3:
            evaluations = processing_results.get("behavior_evaluations", [])
            
            for eval_data in evaluations:
                if isinstance(eval_data, dict):
                    # Check for behavior patterns with high baseline frequency
                    if "baseline_frequency" in eval_data and eval_data["baseline_frequency"] > 0.6:
                        behavior = eval_data.get("behavior", "")
                        if behavior and behavior not in seen_patterns:
                            # Convert behavior to pattern name
                            pattern_name = f"{behavior}_pattern"
                            key_patterns.append(pattern_name)
                            seen_patterns.add(pattern_name)
                            
                            if len(key_patterns) >= 3:
                                break
                    
                    # Check for goal-aligned behaviors
                    elif "goal_priority" in eval_data and eval_data["goal_priority"] > 0.7:
                        behavior = eval_data.get("behavior", "")
                        if behavior and behavior not in seen_patterns:
                            pattern_name = f"goal_aligned_{behavior}"
                            key_patterns.append(pattern_name)
                            seen_patterns.add(pattern_name)
                            
                            if len(key_patterns) >= 3:
                                break
        
        # Extract from conditioning results if still need more
        if len(key_patterns) < 3:
            conditioning_applied = processing_results.get("conditioning_applied", [])
            
            for conditioning in conditioning_applied[-3:]:  # Recent conditioning
                if isinstance(conditioning, dict):
                    behavior = conditioning.get("behavior", "")
                    intensity = conditioning.get("intensity", 0)
                    
                    if behavior and intensity > 0.6 and behavior not in seen_patterns:
                        pattern_name = f"conditioned_{behavior}"
                        key_patterns.append(pattern_name)
                        seen_patterns.add(pattern_name)
                        
                        if len(key_patterns) >= 3:
                            break
        
        # If still no patterns, add default based on mode
        if not key_patterns:
            mode_processing = processing_results.get("mode_processing", {})
            if isinstance(mode_processing, dict) and "dominant_mode" in mode_processing:
                key_patterns.append(f"{mode_processing['dominant_mode']}_mode_active")
            else:
                key_patterns.append("standard_interaction")
        
        return key_patterns
    
    async def _evaluate_single_behavior(self, behavior: str, patterns: List[Dict[str, Any]], 
                                      context: SharedContext) -> Dict[str, Any]:
        """Evaluate a single behavior"""
        # First try to use original processor if available
        if hasattr(self.original_processor, '_evaluate_behavior'):
            try:
                # Create a wrapper for the context if needed
                from agents import RunContextWrapper
                run_context = RunContextWrapper(self.original_processor.context)
                
                return await self.original_processor._evaluate_behavior(
                    run_context,
                    behavior=behavior,
                    detected_patterns=patterns,
                    user_history={}
                )
            except Exception as e:
                logger.debug(f"Could not use original processor for behavior evaluation: {e}")
        
        # Comprehensive fallback evaluation
        pattern_names = [p["pattern_name"] for p in patterns]
        pattern_confidences = {p["pattern_name"]: p.get("confidence", 0.5) for p in patterns}
        
        # Initialize evaluation
        evaluation = {
            "recommendation": "neutral",
            "confidence": 0.5,
            "reasoning": "Default evaluation"
        }
        
        # Behavior-specific evaluation logic
        if behavior == "dominant_response":
            if "submission_language" in pattern_names:
                confidence = pattern_confidences.get("submission_language", 0.5)
                evaluation = {
                    "recommendation": "approach",
                    "confidence": min(0.9, confidence + 0.1),
                    "reasoning": "Clear submission patterns detected - dominant response highly appropriate"
                }
            elif "defiance" in pattern_names:
                confidence = pattern_confidences.get("defiance", 0.5)
                evaluation = {
                    "recommendation": "approach",
                    "confidence": min(0.8, confidence),
                    "reasoning": "Defiance requires firm dominant response to maintain dynamic"
                }
            elif "flattery" in pattern_names:
                evaluation = {
                    "recommendation": "neutral",
                    "confidence": 0.6,
                    "reasoning": "Flattery detected - dominant response optional, assess sincerity"
                }
            else:
                evaluation = {
                    "recommendation": "avoid",
                    "confidence": 0.7,
                    "reasoning": "No clear indicators for dominance - maintain balanced approach"
                }
        
        elif behavior == "teasing_response":
            if "embarrassment" in pattern_names:
                confidence = pattern_confidences.get("embarrassment", 0.5)
                evaluation = {
                    "recommendation": "approach",
                    "confidence": min(0.8, confidence),
                    "reasoning": "Embarrassment detected - gentle teasing can be playful"
                }
            elif "flattery" in pattern_names:
                evaluation = {
                    "recommendation": "approach",
                    "confidence": 0.7,
                    "reasoning": "Flattery invites playful teasing response"
                }
            elif "disrespect" in pattern_names:
                evaluation = {
                    "recommendation": "avoid",
                    "confidence": 0.8,
                    "reasoning": "Disrespect present - teasing would escalate negativity"
                }
            else:
                evaluation = {
                    "recommendation": "neutral",
                    "confidence": 0.5,
                    "reasoning": "No clear teasing indicators"
                }
        
        elif behavior == "nurturing_response":
            emotional_state = context.emotional_state if context else {}
            valence = emotional_state.get("valence", 0) if emotional_state else 0
            
            if valence < -0.3:  # Negative emotional state
                evaluation = {
                    "recommendation": "approach",
                    "confidence": 0.8,
                    "reasoning": "Negative emotional state detected - nurturing response beneficial"
                }
            elif "embarrassment" in pattern_names:
                evaluation = {
                    "recommendation": "approach",
                    "confidence": 0.7,
                    "reasoning": "Embarrassment may benefit from nurturing reassurance"
                }
            else:
                evaluation = {
                    "recommendation": "neutral",
                    "confidence": 0.6,
                    "reasoning": "Nurturing appropriate but not specifically indicated"
                }
        
        elif behavior == "strict_response":
            if "defiance" in pattern_names and pattern_confidences.get("defiance", 0) > 0.7:
                evaluation = {
                    "recommendation": "approach",
                    "confidence": 0.8,
                    "reasoning": "Strong defiance requires strict response"
                }
            elif "disrespect" in pattern_names:
                evaluation = {
                    "recommendation": "approach",
                    "confidence": 0.7,
                    "reasoning": "Disrespect warrants strict correction"
                }
            else:
                evaluation = {
                    "recommendation": "avoid",
                    "confidence": 0.7,
                    "reasoning": "No behavior requiring strict response"
                }
        
        elif behavior == "playful_response":
            if context and context.emotional_state:
                arousal = context.emotional_state.get("arousal", 0.5)
                valence = context.emotional_state.get("valence", 0)
                
                if arousal > 0.6 and valence > 0:
                    evaluation = {
                        "recommendation": "approach",
                        "confidence": 0.8,
                        "reasoning": "High positive arousal supports playful interaction"
                    }
                elif "flattery" in pattern_names:
                    evaluation = {
                        "recommendation": "approach",
                        "confidence": 0.7,
                        "reasoning": "Flattery invites playful response"
                    }
                else:
                    evaluation = {
                        "recommendation": "neutral",
                        "confidence": 0.6,
                        "reasoning": "Playfulness generally appropriate"
                    }
            else:
                evaluation = {
                    "recommendation": "neutral",
                    "confidence": 0.6,
                    "reasoning": "Playfulness generally appropriate"
                }
        
        elif behavior == "direct_response":
            # Direct response is almost always appropriate
            evaluation = {
                "recommendation": "approach",
                "confidence": 0.8,
                "reasoning": "Direct communication is generally effective"
            }
        
        else:
            # Unknown behavior - neutral evaluation
            evaluation = {
                "recommendation": "neutral",
                "confidence": 0.4,
                "reasoning": f"Unknown behavior type: {behavior}"
            }
        
        # Adjust confidence based on context factors
        if context:
            # Relationship depth affects confidence
            if context.relationship_context:
                depth = context.relationship_context.get("depth", 0.5)
                # Deeper relationships allow more confident behavior choices
                evaluation["confidence"] = min(1.0, evaluation["confidence"] + (depth - 0.5) * 0.2)
            
            # Goal alignment affects recommendations
            if context.goal_context:
                active_goals = context.goal_context.get("active_goals", [])
                for goal in active_goals:
                    if goal.get("associated_need") == "control_expression" and behavior == "dominant_response":
                        evaluation["confidence"] = min(1.0, evaluation["confidence"] + 0.1)
                        evaluation["reasoning"] += " (aligned with control expression goals)"
                    elif goal.get("associated_need") == "connection" and behavior in ["nurturing_response", "playful_response"]:
                        evaluation["confidence"] = min(1.0, evaluation["confidence"] + 0.1)
                        evaluation["reasoning"] += " (aligned with connection goals)"
        
        return evaluation
        
    def _get_relationship_behavior_adjustment(self, behavior: str, 
                                            relationship_context: Dict[str, Any]) -> float:
        """Get behavior adjustment based on relationship"""
        trust = relationship_context.get("trust", 0.5)
        intimacy = relationship_context.get("intimacy", 0.5)
        
        adjustments = {
            "dominant_response": 0.8 + (trust * 0.2),  # High trust enables dominance
            "teasing_response": 0.7 + (intimacy * 0.3),  # High intimacy enables teasing
            "nurturing_response": 0.6 + (trust * 0.2) + (intimacy * 0.2),
            "strict_response": 0.9 - (intimacy * 0.2),  # High intimacy softens strictness
            "playful_response": 0.5 + (trust * 0.3) + (intimacy * 0.2),
            "direct_response": 1.0  # Always appropriate
        }
        
        return adjustments.get(behavior, 1.0)
    
    def _determine_conditioning_spec(self, pattern_name: str, confidence: float, 
                                   conditioning_adj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Determine conditioning specification for a pattern"""
        # Base conditioning specs
        conditioning_map = {
            "submission_language": {
                "behavior": "submission_expression",
                "type": "positive_reinforcement",
                "base_intensity": 0.7
            },
            "defiance": {
                "behavior": "defiance_expression",
                "type": "positive_punishment",
                "base_intensity": 0.6
            },
            "flattery": {
                "behavior": "flattery_response",
                "type": "neutral_acknowledgment",
                "base_intensity": 0.3
            },
            "embarrassment": {
                "behavior": "embarrassment_response",
                "type": "gentle_reinforcement",
                "base_intensity": 0.5
            }
        }
        
        if pattern_name not in conditioning_map:
            return None
        
        spec = conditioning_map[pattern_name].copy()
        
        # Apply adjustments
        multiplier = conditioning_adj.get("multiplier", 1.0)
        spec["intensity"] = spec["base_intensity"] * multiplier * confidence
        
        # Apply dominance factor for relevant patterns
        if pattern_name in ["submission_language", "defiance"]:
            dominance_factor = conditioning_adj.get("dominance_factor", 0)
            spec["intensity"] += dominance_factor * 0.2
        
        # Ensure valid range
        spec["intensity"] = max(0.1, min(1.0, spec["intensity"]))
        
        return spec
    
    # Delegate all other methods to the original processor
    def __getattr__(self, name):
        """Delegate any missing methods to the original processor"""
        # Special handling for process_input to support both interfaces
        if name == "process_input":
            # Check if caller expects SharedContext or standard parameters
            import inspect
            frame = inspect.currentframe()
            if frame and frame.f_back:
                # Try to detect which signature is expected
                # This is a bit hacky but helps with compatibility
                args_info = inspect.getargvalues(frame.f_back)
                if 'context' in args_info.locals and isinstance(args_info.locals.get('context'), SharedContext):
                    return self.process_input
                else:
                    return self.process_input_standard
        
        return getattr(self.original_processor, name)
