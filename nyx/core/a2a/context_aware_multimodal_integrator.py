# nyx/core/a2a/context_aware_multimodal_integrator.py

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime


from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority
from nyx.core.multimodal_integrator import IntegratedPercept, SensoryInput, ExpectationSignal, Modality

logger = logging.getLogger(__name__)

class ContextAwareMultimodalIntegrator(ContextAwareModule):
    """
    Advanced MultimodalIntegrator with full context distribution capabilities
    """
    
    def __init__(self, original_integrator):
        super().__init__("multimodal_integrator")
        self.original_integrator = original_integrator
        self.context_subscriptions = [
            "attentional_focus_update", "emotional_state_update", 
            "memory_retrieval_complete", "sensory_expectation",
            "pattern_detected", "need_state_change",
            "goal_context_available", "environmental_change",
            "cross_modal_binding", "perception_request"
        ]
        
        # Track cross-modal patterns
        self.cross_modal_patterns: Dict[str, List[IntegratedPercept]] = {}
        self.active_expectations_by_source: Dict[str, List[ExpectationSignal]] = {}
        self.recent_percepts_by_modality: Dict[Modality, List[IntegratedPercept]] = {
            modality: [] for modality in Modality
        }
        self.perception_context_history: List[Dict[str, Any]] = []
    
    async def on_context_received(self, context: SharedContext):
        """Initialize multimodal processing for this context"""
        logger.debug(f"MultimodalIntegrator received context for user: {context.user_id}")
        
        # Analyze input for potential sensory content
        sensory_implications = await self._analyze_input_for_sensory_content(context.user_input)
        
        # Get current perceptual state
        perceptual_state = await self._get_current_perceptual_state()
        
        # Determine if we need enhanced sensory processing
        enhanced_processing_needed = await self._determine_enhanced_processing_needs(
            context, sensory_implications, perceptual_state
        )
        
        # Set up expectations based on context
        contextual_expectations = await self._generate_contextual_expectations(context)
        
        # Send initial multimodal context
        await self.send_context_update(
            update_type="multimodal_context_initialized",
            data={
                "sensory_implications": sensory_implications,
                "perceptual_state": perceptual_state,
                "enhanced_processing": enhanced_processing_needed,
                "active_modalities": list(perceptual_state["active_modalities"]),
                "contextual_expectations": len(contextual_expectations),
                "cross_modal_readiness": True
            },
            priority=ContextPriority.HIGH if enhanced_processing_needed else ContextPriority.MEDIUM
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules with sophisticated multimodal processing"""
        
        if update.update_type == "attentional_focus_update":
            # Adjust perceptual processing based on attention
            attention_data = update.data
            focused_elements = attention_data.get("focused_elements", [])
            
            # Update expectations based on attentional focus
            for element in focused_elements:
                if element.get("type") == "sensory":
                    modality = element.get("modality")
                    if modality:
                        await self._enhance_modality_processing(modality, element)
            
            # Notify about attention-driven perception changes
            await self.send_context_update(
                update_type="perception_focus_adjusted",
                data={
                    "focused_modalities": [e.get("modality") for e in focused_elements if e.get("type") == "sensory"],
                    "attention_driven": True
                }
            )
        
        elif update.update_type == "emotional_state_update":
            # Emotional states affect perception
            emotional_data = update.data
            dominant_emotion = emotional_data.get("dominant_emotion")
            
            if dominant_emotion:
                # Adjust perceptual biases based on emotion
                perceptual_biases = await self._calculate_emotional_perceptual_biases(dominant_emotion)
                
                # Apply biases to active expectations
                await self._apply_perceptual_biases(perceptual_biases)
        
        elif update.update_type == "memory_retrieval_complete":
            # Use retrieved memories to inform expectations
            memory_data = update.data
            memories = memory_data.get("memories", [])
            
            # Extract sensory memories
            sensory_memories = [m for m in memories if m.get("memory_type") == "sensory"]
            
            if sensory_memories:
                # Generate expectations from sensory memories
                memory_expectations = await self._generate_expectations_from_memories(sensory_memories)
                
                # Add to active expectations
                for expectation in memory_expectations:
                    await self.original_integrator.add_expectation(expectation)
        
        elif update.update_type == "sensory_expectation":
            # Direct sensory expectation from another module
            expectation_data = update.data
            
            # Create expectation signal
            expectation = ExpectationSignal(
                target_modality=Modality(expectation_data["modality"]),
                pattern=expectation_data["pattern"],
                strength=expectation_data.get("strength", 0.5),
                source=expectation_data.get("source", update.source_module),
                priority=expectation_data.get("priority", 0.5)
            )
            
            # Track by source
            source = update.source_module
            if source not in self.active_expectations_by_source:
                self.active_expectations_by_source[source] = []
            self.active_expectations_by_source[source].append(expectation)
            
            # Add to integrator
            await self.original_integrator.add_expectation(expectation)
        
        elif update.update_type == "pattern_detected":
            # Pattern detection can inform cross-modal binding
            pattern_data = update.data
            pattern_type = pattern_data.get("pattern_type")
            
            if pattern_type == "cross_modal":
                # Attempt cross-modal binding
                binding_result = await self._attempt_cross_modal_binding(pattern_data)
                
                if binding_result:
                    await self.send_context_update(
                        update_type="cross_modal_binding_detected",
                        data=binding_result,
                        priority=ContextPriority.HIGH
                    )
        
        elif update.update_type == "need_state_change":
            # Needs can drive perceptual attention
            needs_data = update.data
            high_priority_needs = needs_data.get("high_priority_needs", [])
            
            # Generate perceptual expectations for needs
            for need in high_priority_needs:
                expectations = await self._generate_need_based_expectations(need)
                for exp in expectations:
                    await self.original_integrator.add_expectation(exp)
        
        elif update.update_type == "environmental_change":
            # Environmental changes require perceptual update
            env_data = update.data
            
            # Trigger enhanced environmental scanning
            await self._perform_environmental_scan(env_data)
        
        elif update.update_type == "perception_request":
            # Explicit request for perception of specific content
            request_data = update.data
            modality = request_data.get("modality")
            content = request_data.get("content")
            
            if modality and content:
                # Process the requested perception
                percept = await self._process_requested_perception(modality, content, request_data)
                
                # Send result back
                await self.send_context_update(
                    update_type="perception_request_complete",
                    data={
                        "request_id": request_data.get("request_id"),
                        "percept": percept.dict() if percept else None
                    }
                )
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Advanced input processing with multimodal perception"""
        # Check for multimodal content in input
        multimodal_content = await self._extract_multimodal_content(context.user_input)
        
        percepts = []
        cross_modal_associations = []
        
        # Process each modality
        for modality, content in multimodal_content.items():
            # Get expectations for this modality
            expectations = await self._get_modality_expectations(modality)
            
            # Create sensory input
            sensory_input = SensoryInput(
                modality=modality,
                data=content,
                confidence=0.9,  # High confidence for direct input
                metadata={"source": "user_input", "context_id": context.request_id}
            )
            
            # Process through integrator
            percept = await self.original_integrator.process_sensory_input(
                sensory_input, 
                expectations
            )
            
            percepts.append(percept)
            
            # Track by modality
            self.recent_percepts_by_modality[modality].append(percept)
            if len(self.recent_percepts_by_modality[modality]) > 10:
                self.recent_percepts_by_modality[modality].pop(0)
            
            # Check for cross-modal associations
            associations = await self._find_cross_modal_associations(percept)
            cross_modal_associations.extend(associations)
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Analyze perception patterns
        perception_patterns = await self._analyze_perception_patterns(percepts)
        
        # Update perception context
        self.perception_context_history.append({
            "timestamp": datetime.now().isoformat(),
            "percepts": len(percepts),
            "modalities": list(multimodal_content.keys()),
            "patterns": perception_patterns
        })
        
        # Trim history
        if len(self.perception_context_history) > 50:
            self.perception_context_history.pop(0)
        
        # Send perception summary
        if percepts:
            await self.send_context_update(
                update_type="multimodal_perception_complete",
                data={
                    "percept_count": len(percepts),
                    "modalities_processed": list(multimodal_content.keys()),
                    "cross_modal_associations": len(cross_modal_associations),
                    "perception_patterns": perception_patterns,
                    "attention_weights": {p.modality.value: p.attention_weight for p in percepts}
                }
            )
        
        return {
            "multimodal_processed": True,
            "percepts": percepts,
            "cross_modal_associations": cross_modal_associations,
            "perception_patterns": perception_patterns,
            "cross_module_signals": len(messages),
            "advanced_processing": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Advanced multimodal analysis in context"""
        # Get current perceptual state
        perceptual_state = await self._get_comprehensive_perceptual_state()
        
        # Analyze cross-modal coherence
        cross_modal_coherence = await self._analyze_cross_modal_coherence(perceptual_state)
        
        # Identify perceptual gaps
        perceptual_gaps = await self._identify_perceptual_gaps(context, perceptual_state)
        
        # Analyze expectation accuracy
        expectation_analysis = await self._analyze_expectation_accuracy()
        
        # Identify dominant perceptual themes
        perceptual_themes = await self._identify_perceptual_themes()
        
        # Analyze attention distribution
        attention_analysis = await self._analyze_attention_distribution()
        
        return {
            "perceptual_state": perceptual_state,
            "cross_modal_coherence": cross_modal_coherence,
            "perceptual_gaps": perceptual_gaps,
            "expectation_accuracy": expectation_analysis,
            "perceptual_themes": perceptual_themes,
            "attention_analysis": attention_analysis,
            "analysis_complete": True,
            "advanced_analysis": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Advanced synthesis of multimodal perception for response"""
        # Get all relevant percepts
        relevant_percepts = await self._get_contextually_relevant_percepts(context)
        
        # Synthesize perceptual narrative
        perceptual_narrative = await self._synthesize_perceptual_narrative(relevant_percepts)
        
        # Generate perceptual insights
        perceptual_insights = await self._generate_perceptual_insights(relevant_percepts)
        
        # Identify salient perceptual elements for response
        salient_elements = await self._identify_salient_elements(relevant_percepts, context)
        
        # Check for cross-modal conflicts
        conflicts = await self._detect_perceptual_conflicts(relevant_percepts)
        
        # Generate response guidance based on perception
        response_guidance = {
            "perceptual_narrative": perceptual_narrative,
            "key_insights": perceptual_insights,
            "salient_elements": salient_elements,
            "avoid_conflicts": conflicts,
            "modality_emphasis": await self._determine_modality_emphasis(context),
            "suggested_sensory_language": await self._suggest_sensory_language(relevant_percepts)
        }
        
        # Send synthesis results
        await self.send_context_update(
            update_type="perceptual_synthesis_complete",
            data={
                "narrative_generated": bool(perceptual_narrative),
                "insights_count": len(perceptual_insights),
                "salient_elements": len(salient_elements),
                "conflicts_detected": len(conflicts) > 0
            }
        )
        
        return {
            "perceptual_influence": response_guidance,
            "relevant_percepts": len(relevant_percepts),
            "synthesis_complete": True,
            "advanced_synthesis": True
        }
    
    # ========================================================================================
    # ADVANCED HELPER METHODS
    # ========================================================================================
    
    async def _analyze_input_for_sensory_content(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input for references to sensory content"""
        input_lower = user_input.lower()
        
        sensory_implications = {
            "visual": any(word in input_lower for word in ["see", "look", "image", "picture", "color", "bright", "dark", "visual"]),
            "auditory": any(word in input_lower for word in ["hear", "sound", "music", "noise", "quiet", "loud", "listen", "audio"]),
            "tactile": any(word in input_lower for word in ["feel", "touch", "texture", "soft", "hard", "smooth", "rough", "warm", "cold"]),
            "olfactory": any(word in input_lower for word in ["smell", "scent", "odor", "fragrance", "aroma"]),
            "gustatory": any(word in input_lower for word in ["taste", "flavor", "sweet", "sour", "bitter", "salty", "savory"]),
            "multimodal": any(word in input_lower for word in ["experience", "sensation", "perceive", "sense"])
        }
        
        # Count active sensory references
        active_modalities = [k for k, v in sensory_implications.items() if v]
        
        return {
            "modality_references": sensory_implications,
            "active_modalities": active_modalities,
            "multimodal_content": len(active_modalities) > 1,
            "sensory_focus": len(active_modalities) > 0
        }
    
    async def _get_current_perceptual_state(self) -> Dict[str, Any]:
        """Get comprehensive current perceptual state"""
        # Get recent percepts from each modality
        active_modalities = set()
        total_percepts = 0
        
        for modality, percepts in self.recent_percepts_by_modality.items():
            if percepts:
                active_modalities.add(modality.value)
                total_percepts += len(percepts)
        
        # Get active expectations
        total_expectations = 0
        for expectations in self.active_expectations_by_source.values():
            total_expectations += len(expectations)
        
        # Calculate perceptual load
        perceptual_load = min(1.0, (total_percepts + total_expectations) / 20.0)
        
        return {
            "active_modalities": active_modalities,
            "total_recent_percepts": total_percepts,
            "active_expectations": total_expectations,
            "perceptual_load": perceptual_load,
            "expectation_sources": list(self.active_expectations_by_source.keys()),
            "cross_modal_patterns": len(self.cross_modal_patterns)
        }
    
    async def _determine_enhanced_processing_needs(self, context: SharedContext, 
                                                  sensory_implications: Dict[str, Any],
                                                  perceptual_state: Dict[str, Any]) -> bool:
        """Determine if enhanced sensory processing is needed"""
        # Check various factors
        factors = []
        
        # High sensory focus in input
        if sensory_implications.get("sensory_focus"):
            factors.append("sensory_input")
        
        # Multimodal content
        if sensory_implications.get("multimodal_content"):
            factors.append("multimodal")
        
        # High perceptual load
        if perceptual_state.get("perceptual_load", 0) > 0.7:
            factors.append("high_load")
        
        # Emotional intensity (affects perception)
        if context.emotional_state:
            emotion_intensity = 0
            for emotion, value in context.emotional_state.items():
                if isinstance(value, (int, float)):
                    emotion_intensity = max(emotion_intensity, value)
            if emotion_intensity > 0.7:
                factors.append("emotional_intensity")
        
        # Need at least 2 factors for enhanced processing
        return len(factors) >= 2
    
    async def _generate_contextual_expectations(self, context: SharedContext) -> List[ExpectationSignal]:
        """Generate expectations based on current context"""
        expectations = []
        
        # Emotional expectations
        if context.emotional_state:
            dominant_emotion = None
            max_value = 0
            for emotion, value in context.emotional_state.items():
                if isinstance(value, (int, float)) and value > max_value:
                    dominant_emotion = emotion
                    max_value = value
            
            if dominant_emotion:
                # Different emotions create different perceptual expectations
                emotion_expectations = {
                    "joy": [("visual", "bright colors"), ("auditory", "upbeat patterns")],
                    "sadness": [("visual", "muted tones"), ("auditory", "slow tempo")],
                    "fear": [("visual", "threats"), ("auditory", "sudden sounds")],
                    "curiosity": [("visual", "novel patterns"), ("multimodal", "interesting details")]
                }
                
                if dominant_emotion.lower() in emotion_expectations:
                    for modality, pattern in emotion_expectations[dominant_emotion.lower()]:
                        expectations.append(ExpectationSignal(
                            target_modality=Modality(modality) if modality != "multimodal" else Modality.TEXT,
                            pattern=pattern,
                            strength=max_value * 0.7,
                            source="emotional_context",
                            priority=0.6
                        ))
        
        # Memory-based expectations
        if context.memory_context:
            recent_memories = context.memory_context.get("recent_memories", [])
            for memory in recent_memories[:3]:  # Top 3 memories
                if "sensory" in memory.get("memory_type", ""):
                    # Extract sensory expectations from memory
                    memory_text = memory.get("memory_text", "")
                    if "visual" in memory_text.lower():
                        expectations.append(ExpectationSignal(
                            target_modality=Modality.IMAGE,
                            pattern=memory_text,
                            strength=0.5,
                            source="memory",
                            priority=0.5
                        ))
        
        return expectations
    
    async def _enhance_modality_processing(self, modality: str, attention_element: Dict[str, Any]):
        """Enhance processing for a specific modality based on attention"""
        # Increase expectation strength for attended modality
        modality_obj = Modality(modality)
        
        # Update expectations for this modality
        context = self.original_integrator.context
        if hasattr(context, "active_expectations"):
            for expectation in context.active_expectations:
                if expectation.target_modality == modality_obj:
                    # Boost strength based on attention
                    attention_weight = attention_element.get("weight", 0.5)
                    expectation.strength = min(1.0, expectation.strength * (1 + attention_weight))
                    expectation.priority = min(1.0, expectation.priority * (1 + attention_weight * 0.5))
    
    async def _calculate_emotional_perceptual_biases(self, dominant_emotion: Tuple[str, float]) -> Dict[str, float]:
        """Calculate how emotions bias perception"""
        emotion_name, strength = dominant_emotion
        
        # Define perceptual biases for emotions
        bias_profiles = {
            "Joy": {
                "positive_valence": 0.3,
                "bright_visual": 0.2,
                "harmonious_audio": 0.2,
                "pleasant_sensory": 0.3
            },
            "Fear": {
                "threat_detection": 0.4,
                "movement_sensitivity": 0.3,
                "sudden_change": 0.3,
                "negative_valence": 0.2
            },
            "Curiosity": {
                "novelty_detection": 0.4,
                "detail_attention": 0.3,
                "pattern_recognition": 0.3
            },
            "Sadness": {
                "negative_valence": 0.3,
                "reduced_attention": 0.2,
                "internal_focus": 0.3
            }
        }
        
        biases = bias_profiles.get(emotion_name, {})
        
        # Scale by emotion strength
        return {k: v * strength for k, v in biases.items()}
    
    async def _apply_perceptual_biases(self, biases: Dict[str, float]):
        """Apply perceptual biases to expectations"""
        # Modify active expectations based on biases
        context = self.original_integrator.context
        
        if not hasattr(context, "active_expectations"):
            return
        
        for expectation in context.active_expectations:
            # Apply relevant biases
            if "positive_valence" in biases and "positive" in str(expectation.pattern).lower():
                expectation.strength *= (1 + biases["positive_valence"])
            
            if "negative_valence" in biases and "negative" in str(expectation.pattern).lower():
                expectation.strength *= (1 + biases["negative_valence"])
            
            if "novelty_detection" in biases and expectation.source == "pattern":
                expectation.priority *= (1 + biases["novelty_detection"])
            
            # Ensure values stay in range
            expectation.strength = min(1.0, expectation.strength)
            expectation.priority = min(1.0, expectation.priority)
    
    async def _generate_expectations_from_memories(self, sensory_memories: List[Dict[str, Any]]) -> List[ExpectationSignal]:
        """Generate expectations from sensory memories"""
        expectations = []
        
        for memory in sensory_memories:
            memory_text = memory.get("memory_text", "")
            
            # Simple pattern extraction from memory text
            if "saw" in memory_text or "visual" in memory_text:
                expectations.append(ExpectationSignal(
                    target_modality=Modality.IMAGE,
                    pattern=memory_text,
                    strength=0.6,
                    source="sensory_memory",
                    priority=0.5
                ))
            
            if "heard" in memory_text or "sound" in memory_text:
                expectations.append(ExpectationSignal(
                    target_modality=Modality.AUDIO_MUSIC,
                    pattern=memory_text,
                    strength=0.6,
                    source="sensory_memory",
                    priority=0.5
                ))
        
        return expectations
    
    async def _attempt_cross_modal_binding(self, pattern_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Attempt to bind perceptions across modalities"""
        modalities_involved = pattern_data.get("modalities", [])
        
        if len(modalities_involved) < 2:
            return None
        
        # Get recent percepts from involved modalities
        related_percepts = []
        for modality in modalities_involved:
            if modality in self.recent_percepts_by_modality:
                recent = self.recent_percepts_by_modality[modality]
                if recent:
                    related_percepts.append(recent[-1])  # Most recent
        
        if len(related_percepts) < 2:
            return None
        
        # Check temporal proximity
        timestamps = [p.timestamp for p in related_percepts]
        time_diffs = []
        for i in range(1, len(timestamps)):
            t1 = datetime.fromisoformat(timestamps[i-1])
            t2 = datetime.fromisoformat(timestamps[i])
            time_diffs.append(abs((t2 - t1).total_seconds()))
        
        # If percepts are close in time, they might be related
        max_time_diff = max(time_diffs) if time_diffs else 0
        if max_time_diff < 5.0:  # Within 5 seconds
            binding_id = f"binding_{datetime.now().timestamp()}"
            
            # Store cross-modal pattern
            self.cross_modal_patterns[binding_id] = related_percepts
            
            return {
                "binding_id": binding_id,
                "modalities": modalities_involved,
                "percept_count": len(related_percepts),
                "temporal_proximity": max_time_diff,
                "binding_strength": 1.0 - (max_time_diff / 5.0)
            }
        
        return None
    
    async def _generate_need_based_expectations(self, need: str) -> List[ExpectationSignal]:
        """Generate perceptual expectations based on needs"""
        need_expectations = {
            "connection": [
                (Modality.AUDIO_SPEECH, "warm voice tones"),
                (Modality.TEXT, "caring language")
            ],
            "stimulation": [
                (Modality.IMAGE, "vibrant patterns"),
                (Modality.AUDIO_MUSIC, "dynamic rhythms")
            ],
            "safety": [
                (Modality.ENVIRONMENT, "stable patterns"),
                (Modality.AUDIO_SPEECH, "calm tones")
            ],
            "understanding": [
                (Modality.TEXT, "clear explanations"),
                (Modality.IMAGE, "informative visuals")
            ]
        }
        
        expectations = []
        if need in need_expectations:
            for modality, pattern in need_expectations[need]:
                expectations.append(ExpectationSignal(
                    target_modality=modality,
                    pattern=pattern,
                    strength=0.7,
                    source=f"need_{need}",
                    priority=0.7
                ))
        
        return expectations
    
    async def _perform_environmental_scan(self, env_data: Dict[str, Any]):
        """Perform enhanced environmental perception"""
        # Process environmental modalities
        env_modalities = [Modality.ENVIRONMENT, Modality.SYSTEM_SCREEN]
        
        for modality in env_modalities:
            # Create synthetic sensory input for environment
            sensory_input = SensoryInput(
                modality=modality,
                data=env_data,
                confidence=0.8,
                metadata={"source": "environmental_scan", "triggered_by": "environmental_change"}
            )
            
            # Process through integrator
            percept = await self.original_integrator.process_sensory_input(sensory_input)
            
            # Track percept
            self.recent_percepts_by_modality[modality].append(percept)
    
    async def _process_requested_perception(self, modality: str, content: Any, 
                                          request_data: Dict[str, Any]) -> Optional[IntegratedPercept]:
        """Process explicitly requested perception"""
        try:
            modality_obj = Modality(modality)
            
            # Create sensory input
            sensory_input = SensoryInput(
                modality=modality_obj,
                data=content,
                confidence=request_data.get("confidence", 0.9),
                metadata={
                    "source": "explicit_request",
                    "requester": request_data.get("requester", "unknown"),
                    "request_id": request_data.get("request_id")
                }
            )
            
            # Get any specific expectations
            expectations = []
            if "expectations" in request_data:
                for exp_data in request_data["expectations"]:
                    expectations.append(ExpectationSignal(**exp_data))
            
            # Process
            percept = await self.original_integrator.process_sensory_input(
                sensory_input,
                expectations if expectations else None
            )
            
            return percept
            
        except Exception as e:
            logger.error(f"Error processing requested perception: {e}")
            return None
    
    async def _extract_multimodal_content(self, user_input: str) -> Dict[Modality, Any]:
        """Extract multimodal content from user input"""
        content = {}
        
        # Text is always present
        content[Modality.TEXT] = user_input
        
        # Check for references to other modalities
        input_lower = user_input.lower()
        
        # Simple pattern matching for demonstration
        if "image:" in input_lower or "[image]" in input_lower:
            content[Modality.IMAGE] = {"description": "User-referenced image", "source": "input"}
        
        if "sound:" in input_lower or "[audio]" in input_lower:
            content[Modality.AUDIO_SPEECH] = {"description": "User-referenced audio", "source": "input"}
        
        return content
    
    async def _get_modality_expectations(self, modality: Modality) -> List[ExpectationSignal]:
        """Get current expectations for a specific modality"""
        expectations = []
        
        # Get from integrator context
        context = self.original_integrator.context
        if hasattr(context, "active_expectations"):
            for exp in context.active_expectations:
                if exp.target_modality == modality:
                    expectations.append(exp)
        
        return expectations
    
    async def _find_cross_modal_associations(self, percept: IntegratedPercept) -> List[Dict[str, Any]]:
        """Find associations between percepts across modalities"""
        associations = []
        
        # Check recent percepts in other modalities
        for modality, recent_percepts in self.recent_percepts_by_modality.items():
            if modality != percept.modality and recent_percepts:
                # Check most recent percept
                recent = recent_percepts[-1]
                
                # Simple temporal association
                time_diff = abs((datetime.fromisoformat(percept.timestamp) - 
                               datetime.fromisoformat(recent.timestamp)).total_seconds())
                
                if time_diff < 2.0:  # Within 2 seconds
                    associations.append({
                        "percept1": percept.modality.value,
                        "percept2": modality.value,
                        "temporal_proximity": time_diff,
                        "association_strength": 1.0 - (time_diff / 2.0)
                    })
        
        return associations
    
    async def _analyze_perception_patterns(self, percepts: List[IntegratedPercept]) -> Dict[str, Any]:
        """Analyze patterns in perception"""
        if not percepts:
            return {}
        
        patterns = {
            "dominant_modality": None,
            "attention_distribution": {},
            "confidence_levels": {},
            "top_down_influence": 0.0
        }
        
        # Count by modality
        modality_counts = {}
        attention_sums = {}
        confidence_sums = {}
        top_down_sum = 0.0
        
        for percept in percepts:
            modality = percept.modality.value
            
            # Count
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
            
            # Sum attention
            attention_sums[modality] = attention_sums.get(modality, 0) + percept.attention_weight
            
            # Sum confidence
            confidence_sums[modality] = confidence_sums.get(modality, 0) + percept.bottom_up_confidence
            
            # Sum top-down influence
            top_down_sum += percept.top_down_influence
        
        # Find dominant modality
        if modality_counts:
            patterns["dominant_modality"] = max(modality_counts, key=modality_counts.get)
        
        # Calculate averages
        for modality in modality_counts:
            count = modality_counts[modality]
            patterns["attention_distribution"][modality] = attention_sums[modality] / count
            patterns["confidence_levels"][modality] = confidence_sums[modality] / count
        
        # Average top-down influence
        patterns["top_down_influence"] = top_down_sum / len(percepts) if percepts else 0.0
        
        return patterns
    
    async def _get_comprehensive_perceptual_state(self) -> Dict[str, Any]:
        """Get detailed perceptual state for analysis"""
        state = await self._get_current_perceptual_state()
        
        # Add detailed modality information
        modality_details = {}
        for modality, percepts in self.recent_percepts_by_modality.items():
            if percepts:
                modality_details[modality.value] = {
                    "percept_count": len(percepts),
                    "avg_attention": sum(p.attention_weight for p in percepts) / len(percepts),
                    "avg_confidence": sum(p.bottom_up_confidence for p in percepts) / len(percepts),
                    "latest_timestamp": percepts[-1].timestamp
                }
        
        state["modality_details"] = modality_details
        
        # Add expectation details
        expectation_details = {}
        for source, expectations in self.active_expectations_by_source.items():
            expectation_details[source] = {
                "count": len(expectations),
                "modalities": list(set(e.target_modality.value for e in expectations)),
                "avg_strength": sum(e.strength for e in expectations) / len(expectations) if expectations else 0
            }
        
        state["expectation_details"] = expectation_details
        
        return state
    
    async def _analyze_cross_modal_coherence(self, perceptual_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coherence across modalities"""
        coherence_score = 0.0
        coherence_factors = []
        
        # Check temporal coherence
        modality_details = perceptual_state.get("modality_details", {})
        if len(modality_details) > 1:
            # Get timestamps
            timestamps = []
            for details in modality_details.values():
                if "latest_timestamp" in details:
                    timestamps.append(datetime.fromisoformat(details["latest_timestamp"]))
            
            if len(timestamps) > 1:
                # Calculate time spread
                time_spread = (max(timestamps) - min(timestamps)).total_seconds()
                if time_spread < 5.0:  # Within 5 seconds
                    coherence_score += 0.3
                    coherence_factors.append("temporal_alignment")
        
        # Check attention coherence
        attention_values = [d.get("avg_attention", 0) for d in modality_details.values()]
        if attention_values:
            attention_variance = sum((a - sum(attention_values)/len(attention_values))**2 for a in attention_values)
            if attention_variance < 0.1:  # Low variance = coherent attention
                coherence_score += 0.3
                coherence_factors.append("attention_alignment")
        
        # Check for cross-modal patterns
        if self.cross_modal_patterns:
            coherence_score += 0.2
            coherence_factors.append("cross_modal_binding")
        
        return {
            "coherence_score": min(1.0, coherence_score),
            "coherence_factors": coherence_factors,
            "cross_modal_patterns": len(self.cross_modal_patterns)
        }
    
    async def _identify_perceptual_gaps(self, context: SharedContext, 
                                       perceptual_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify gaps in perception"""
        gaps = []
        
        # Check for expected but missing modalities
        sensory_implications = await self._analyze_input_for_sensory_content(context.user_input)
        referenced_modalities = sensory_implications.get("modality_references", {})
        active_modalities = perceptual_state.get("active_modalities", set())
        
        for modality, referenced in referenced_modalities.items():
            if referenced and modality not in active_modalities:
                gaps.append({
                    "type": "missing_modality",
                    "modality": modality,
                    "reason": "referenced_but_not_perceived"
                })
        
        # Check for low attention modalities
        modality_details = perceptual_state.get("modality_details", {})
        for modality, details in modality_details.items():
            if details.get("avg_attention", 0) < 0.2:
                gaps.append({
                    "type": "low_attention",
                    "modality": modality,
                    "attention_level": details["avg_attention"]
                })
        
        return gaps
    
    async def _analyze_expectation_accuracy(self) -> Dict[str, Any]:
        """Analyze how well expectations matched perceptions"""
        # This would require tracking expectation fulfillment
        # For now, return a simplified analysis
        
        total_expectations = sum(len(exps) for exps in self.active_expectations_by_source.values())
        
        # Estimate fulfillment based on recent percepts
        recent_percept_count = sum(len(percepts) for percepts in self.recent_percepts_by_modality.values())
        
        if total_expectations > 0:
            fulfillment_ratio = min(1.0, recent_percept_count / total_expectations)
        else:
            fulfillment_ratio = 0.5  # Neutral if no expectations
        
        return {
            "total_expectations": total_expectations,
            "estimated_fulfillment": fulfillment_ratio,
            "accuracy_score": fulfillment_ratio,
            "expectation_sources": list(self.active_expectations_by_source.keys())
        }
    
    async def _identify_perceptual_themes(self) -> List[Dict[str, Any]]:
        """Identify dominant themes in perception"""
        themes = []
        
        # Analyze recent percepts for themes
        all_recent_percepts = []
        for percepts in self.recent_percepts_by_modality.values():
            all_recent_percepts.extend(percepts)
        
        if not all_recent_percepts:
            return themes
        
        # Group by attention weight
        high_attention = [p for p in all_recent_percepts if p.attention_weight > 0.7]
        if high_attention:
            themes.append({
                "theme": "high_attention_focus",
                "percept_count": len(high_attention),
                "modalities": list(set(p.modality.value for p in high_attention))
            })
        
        # Group by top-down influence
        high_top_down = [p for p in all_recent_percepts if p.top_down_influence > 0.5]
        if high_top_down:
            themes.append({
                "theme": "expectation_driven",
                "percept_count": len(high_top_down),
                "modalities": list(set(p.modality.value for p in high_top_down))
            })
        
        return themes
    
    async def _analyze_attention_distribution(self) -> Dict[str, Any]:
        """Analyze how attention is distributed across modalities"""
        distribution = {}
        total_attention = 0.0
        
        for modality, percepts in self.recent_percepts_by_modality.items():
            if percepts:
                modality_attention = sum(p.attention_weight for p in percepts)
                distribution[modality.value] = modality_attention
                total_attention += modality_attention
        
        # Normalize
        if total_attention > 0:
            for modality in distribution:
                distribution[modality] /= total_attention
        
        # Calculate entropy (diversity of attention)
        entropy = 0.0
        for value in distribution.values():
            if value > 0:
                entropy -= value * (value if value == 1 else value * (1 / value))
        
        return {
            "distribution": distribution,
            "total_attention": total_attention,
            "attention_entropy": entropy,
            "focused": entropy < 0.5  # Low entropy = focused attention
        }
    
    async def _get_contextually_relevant_percepts(self, context: SharedContext) -> List[IntegratedPercept]:
        """Get percepts relevant to current context"""
        relevant = []
        
        # Get all recent percepts
        all_percepts = []
        for percepts in self.recent_percepts_by_modality.values():
            all_percepts.extend(percepts)
        
        # Filter by relevance criteria
        for percept in all_percepts:
            relevance_score = 0.0
            
            # High attention weight
            if percept.attention_weight > 0.5:
                relevance_score += 0.3
            
            # Recent (within last minute)
            percept_time = datetime.fromisoformat(percept.timestamp)
            age = (datetime.now() - percept_time).total_seconds()
            if age < 60:
                relevance_score += 0.3 * (1 - age / 60)
            
            # High confidence
            if percept.bottom_up_confidence > 0.7:
                relevance_score += 0.2
            
            # Cross-modal binding
            for pattern_percepts in self.cross_modal_patterns.values():
                if percept in pattern_percepts:
                    relevance_score += 0.2
                    break
            
            if relevance_score > 0.4:
                relevant.append(percept)
        
        # Sort by relevance (attention weight as proxy)
        relevant.sort(key=lambda p: p.attention_weight, reverse=True)
        
        return relevant[:10]  # Top 10 most relevant
    
    async def _synthesize_perceptual_narrative(self, percepts: List[IntegratedPercept]) -> str:
        """Create a narrative from percepts"""
        if not percepts:
            return ""
        
        # Group by modality
        by_modality = {}
        for percept in percepts:
            modality = percept.modality.value
            if modality not in by_modality:
                by_modality[modality] = []
            by_modality[modality].append(percept)
        
        # Build narrative
        narrative_parts = []
        
        for modality, modality_percepts in by_modality.items():
            if modality_percepts:
                # Describe the modality experience
                attention_avg = sum(p.attention_weight for p in modality_percepts) / len(modality_percepts)
                
                if attention_avg > 0.7:
                    intensity = "vividly"
                elif attention_avg > 0.4:
                    intensity = "clearly"
                else:
                    intensity = "subtly"
                
                narrative_parts.append(f"{intensity} perceiving through {modality}")
        
        return ", ".join(narrative_parts) if narrative_parts else ""
    
    async def _generate_perceptual_insights(self, percepts: List[IntegratedPercept]) -> List[str]:
        """Generate insights from perceptual patterns"""
        insights = []
        
        if not percepts:
            return insights
        
        # Check for consistent high attention
        high_attention_count = sum(1 for p in percepts if p.attention_weight > 0.7)
        if high_attention_count > len(percepts) * 0.6:
            insights.append("Perceptual focus is highly concentrated")
        
        # Check for multimodal coherence
        modalities = set(p.modality.value for p in percepts)
        if len(modalities) > 2:
            insights.append("Rich multimodal experience detected")
        
        # Check for expectation influence
        high_top_down = sum(1 for p in percepts if p.top_down_influence > 0.5)
        if high_top_down > len(percepts) * 0.5:
            insights.append("Perceptions strongly influenced by expectations")
        
        return insights
    
    async def _identify_salient_elements(self, percepts: List[IntegratedPercept], 
                                        context: SharedContext) -> List[Dict[str, Any]]:
        """Identify most salient perceptual elements"""
        salient = []
        
        for percept in percepts[:5]:  # Top 5
            element = {
                "modality": percept.modality.value,
                "salience_score": percept.attention_weight,
                "confidence": percept.bottom_up_confidence,
                "content_summary": str(percept.content)[:100] if percept.content else ""
            }
            salient.append(element)
        
        return salient
    
    async def _detect_perceptual_conflicts(self, percepts: List[IntegratedPercept]) -> List[Dict[str, Any]]:
        """Detect conflicts between percepts"""
        conflicts = []
        
        # Simple conflict detection based on attention competition
        high_attention_percepts = [p for p in percepts if p.attention_weight > 0.7]
        
        if len(high_attention_percepts) > 1:
            # Multiple high-attention percepts might conflict
            for i, p1 in enumerate(high_attention_percepts):
                for p2 in high_attention_percepts[i+1:]:
                    if p1.modality != p2.modality:
                        conflicts.append({
                            "type": "attention_competition",
                            "modality1": p1.modality.value,
                            "modality2": p2.modality.value,
                            "resolution": "balance_attention"
                        })
        
        return conflicts
    
    async def _determine_modality_emphasis(self, context: SharedContext) -> Dict[str, float]:
        """Determine which modalities to emphasize in response"""
        emphasis = {}
        
        # Base emphasis on recent perception patterns
        for modality, percepts in self.recent_percepts_by_modality.items():
            if percepts:
                # Average attention weight as emphasis
                avg_attention = sum(p.attention_weight for p in percepts) / len(percepts)
                emphasis[modality.value] = avg_attention
        
        return emphasis
    
    async def _suggest_sensory_language(self, percepts: List[IntegratedPercept]) -> List[str]:
        """Suggest sensory language based on percepts"""
        suggestions = []
        
        # Map modalities to language suggestions
        modality_language = {
            Modality.TEXT: ["understand", "comprehend", "grasp"],
            Modality.IMAGE: ["see", "visualize", "picture"],
            Modality.AUDIO_MUSIC: ["hear", "listen", "resonate"],
            Modality.AUDIO_SPEECH: ["hear", "understand", "process"],
            Modality.TOUCH_EVENT: ["feel", "sense", "experience"],
            Modality.TASTE: ["taste", "savor", "experience"],
            Modality.SMELL: ["smell", "sense", "detect"]
        }
        
        # Get active modalities
        active_modalities = set(p.modality for p in percepts)
        
        for modality in active_modalities:
            if modality in modality_language:
                suggestions.extend(modality_language[modality])
        
        return list(set(suggestions))  # Remove duplicates
    
    # Delegate all other methods to the original integrator
    def __getattr__(self, name):
        """Delegate any missing methods to the original integrator"""
        return getattr(self.original_integrator, name)
