# nyx/core/a2a/context_aware_body_image.py

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareBodyImage(ContextAwareModule):
    """
    Advanced BodyImage with full context distribution capabilities
    """
    
    def __init__(self, original_body_image):
        super().__init__("body_image")
        self.original = original_body_image
        self.context_subscriptions = [
            "visual_perception", "somatic_sensation", "movement_command",
            "emotional_state_update", "attention_updated", "spatial_mapping_update",
            "user_gesture", "avatar_state_change", "sensory_input"
        ]
        
        # Maintain reference to original state
        self.current_state = original_body_image.current_state
        self.body_context = original_body_image.body_context
        self.default_form = original_body_image.default_form
    
    async def on_context_received(self, context: SharedContext):
        """Initialize body image processing for this context"""
        logger.debug(f"BodyImage received context for user: {context.user_id}")
        
        # Get current body state
        body_state = self._get_comprehensive_body_state()
        
        # Analyze context for body-relevant information
        body_context_analysis = await self._analyze_context_for_body_relevance(context)
        
        # Send initial body state to other modules
        await self.send_context_update(
            update_type="body_state_initialized",
            data={
                "has_visual_form": body_state["has_visual_form"],
                "form_description": body_state["form_description"],
                "perceived_parts": body_state["parts"],
                "proprioception_confidence": body_state["proprioception_confidence"],
                "body_integrity": body_state["overall_integrity"],
                "embodiment_level": self._calculate_embodiment_level(),
                "body_context_relevance": body_context_analysis
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules affecting body image"""
        
        if update.update_type == "visual_perception":
            # Process visual perception of self/avatar
            visual_data = update.data
            if await self._is_self_perception(visual_data):
                await self._process_visual_self_perception(visual_data)
        
        elif update.update_type == "somatic_sensation":
            # Process somatic sensations
            somatic_data = update.data
            await self._process_somatic_sensation(somatic_data)
        
        elif update.update_type == "movement_command":
            # Update body state based on movement
            movement_data = update.data
            await self._process_movement_update(movement_data)
        
        elif update.update_type == "emotional_state_update":
            # Emotions affect body perception
            emotional_data = update.data
            await self._process_emotional_body_effects(emotional_data)
        
        elif update.update_type == "attention_updated":
            # Attention affects body part awareness
            attention_data = update.data
            await self._process_attention_body_effects(attention_data)
        
        elif update.update_type == "spatial_mapping_update":
            # Spatial context affects body positioning
            spatial_data = update.data
            await self._process_spatial_body_update(spatial_data)
        
        elif update.update_type == "avatar_state_change":
            # Direct avatar state updates
            avatar_data = update.data
            await self._process_avatar_state_change(avatar_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with body-aware context"""
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Extract all body-relevant perceptions
        body_perceptions = await self._extract_body_perceptions(context, messages)
        
        # Process visual perceptions
        visual_updates = []
        for percept in body_perceptions.get("visual", []):
            result = await self.original.update_from_visual(percept)
            visual_updates.append(result)
        
        # Process somatic updates
        somatic_update = None
        if body_perceptions.get("somatic"):
            somatic_update = await self.original.update_from_somatic()
        
        # Integrate perceptions with context
        integration_result = await self._integrate_body_perceptions(
            visual_updates, somatic_update, context
        )
        
        # Calculate body coherence metrics
        coherence_metrics = await self._calculate_body_coherence(context, messages)
        
        # Send comprehensive body state update
        await self.send_context_update(
            update_type="body_state_updated",
            data={
                "visual_updates": len(visual_updates),
                "somatic_integrated": somatic_update is not None,
                "current_state": self._get_comprehensive_body_state(),
                "coherence_metrics": coherence_metrics,
                "embodiment_changes": integration_result.get("embodiment_changes", {})
            }
        )
        
        return {
            "body_processing_complete": True,
            "perceptions_processed": {
                "visual": len(visual_updates),
                "somatic": 1 if somatic_update else 0
            },
            "integration_result": integration_result,
            "body_coherence": coherence_metrics["overall_coherence"],
            "context_integrated": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze body image in context"""
        # Get current comprehensive state
        current_state = self._get_comprehensive_body_state()
        
        # Analyze body schema integrity
        schema_analysis = await self._analyze_body_schema_integrity(current_state)
        
        # Analyze embodiment quality
        embodiment_analysis = await self._analyze_embodiment_quality(context)
        
        # Analyze body-environment relationship
        environment_analysis = await self._analyze_body_environment_relationship(context)
        
        # Identify body image issues
        issues = await self._identify_body_image_issues(current_state, context)
        
        return {
            "body_schema_analysis": schema_analysis,
            "embodiment_analysis": embodiment_analysis,
            "environment_relationship": environment_analysis,
            "identified_issues": issues,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize body-related response components"""
        messages = await self.get_cross_module_messages()
        
        # Generate body-informed synthesis
        body_synthesis = {
            "embodiment_expression": await self._generate_embodiment_expression(context),
            "spatial_awareness": await self._generate_spatial_awareness_component(context),
            "body_language_suggestions": await self._suggest_body_language(context, messages),
            "physical_comfort_indicators": await self._assess_physical_comfort(context),
            "body_coherence_check": await self._final_body_coherence_check(context, messages)
        }
        
        return body_synthesis
    
    # ========================================================================================
    # DETAILED HELPER METHODS
    # ========================================================================================
    
    def _get_comprehensive_body_state(self) -> Dict[str, Any]:
        """Get comprehensive current body state"""
        parts_dict = {}
        for name, part in self.current_state.perceived_parts.items():
            parts_dict[name] = {
                "perceived_state": part.perceived_state,
                "confidence": part.confidence,
                "position": part.perceived_position,
                "orientation": part.perceived_orientation
            }
        
        return {
            "has_visual_form": self.current_state.has_visual_form,
            "form_description": self.current_state.form_description,
            "parts": parts_dict,
            "overall_integrity": self.current_state.overall_integrity,
            "proprioception_confidence": self.current_state.proprioception_confidence,
            "last_visual_update": self.current_state.last_visual_update.isoformat() if self.current_state.last_visual_update else None,
            "last_somatic_update": self.current_state.last_somatic_correlation_time.isoformat() if self.current_state.last_somatic_correlation_time else None,
            "part_count": len(parts_dict)
        }
    
    async def _analyze_context_for_body_relevance(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze how relevant body awareness is to current context"""
        relevance = {
            "physical_interaction": 0.0,
            "spatial_navigation": 0.0,
            "emotional_embodiment": 0.0,
            "gesture_communication": 0.0,
            "overall": 0.0
        }
        
        user_input_lower = context.user_input.lower()
        
        # Check for physical interaction indicators
        physical_keywords = ["touch", "feel", "hold", "move", "gesture", "reach", "grab", "push", "pull"]
        if any(keyword in user_input_lower for keyword in physical_keywords):
            relevance["physical_interaction"] = 0.8
        
        # Check for spatial indicators
        spatial_keywords = ["where", "here", "there", "near", "far", "above", "below", "position", "location"]
        if any(keyword in user_input_lower for keyword in spatial_keywords):
            relevance["spatial_navigation"] = 0.7
        
        # Check emotional context for embodiment
        if context.emotional_state:
            avg_intensity = sum(context.emotional_state.values()) / len(context.emotional_state) if context.emotional_state else 0
            relevance["emotional_embodiment"] = min(1.0, avg_intensity * 1.2)
        
        # Check for gesture/expression keywords
        gesture_keywords = ["show", "express", "demonstrate", "indicate", "point"]
        if any(keyword in user_input_lower for keyword in gesture_keywords):
            relevance["gesture_communication"] = 0.6
        
        # Calculate overall relevance
        relevance["overall"] = max(relevance.values())
        
        return relevance
    
    def _calculate_embodiment_level(self) -> float:
        """Calculate current level of embodiment"""
        # Base embodiment on form presence
        if not self.current_state.has_visual_form:
            return 0.1  # Minimal embodiment without visual form
        
        # Factor in proprioception confidence
        embodiment = self.current_state.proprioception_confidence * 0.4
        
        # Factor in body part awareness
        part_count = len(self.current_state.perceived_parts)
        expected_parts = len(self.default_form["parts"])
        part_ratio = part_count / max(1, expected_parts)
        embodiment += part_ratio * 0.3
        
        # Factor in overall integrity
        embodiment += self.current_state.overall_integrity * 0.3
        
        return min(1.0, embodiment)
    
    async def _is_self_perception(self, visual_data: Dict[str, Any]) -> bool:
        """Determine if visual perception is of self/avatar"""
        # Check for self-identifying markers
        if visual_data.get("is_self", False):
            return True
        
        # Check for avatar-related objects
        objects = visual_data.get("objects", {})
        avatar_keywords = ["avatar", "self", "nyx", "body", "form"]
        
        for obj_name in objects.keys():
            if any(keyword in obj_name.lower() for keyword in avatar_keywords):
                return True
        
        # Check description for self-references
        description = visual_data.get("description", "").lower()
        if any(keyword in description for keyword in ["my", "self", "avatar"]):
            return True
        
        return False
    
    async def _process_visual_self_perception(self, visual_data: Dict[str, Any]):
        """Process visual perception of self"""
        # Convert to percept format expected by original
        percept = {
            "modality": visual_data.get("modality", "image"),
            "content": visual_data,
            "timestamp": visual_data.get("timestamp", datetime.now().isoformat()),
            "bottom_up_confidence": visual_data.get("confidence", 0.5)
        }
        
        # Update through original system
        result = await self.original.update_from_visual(percept)
        
        # Send focused update about visual self-perception
        await self.send_context_update(
            update_type="self_perception_processed",
            data={
                "perception_result": result,
                "updated_parts": result.get("parts_updated", []),
                "form_visible": True
            },
            scope=ContextScope.GLOBAL
        )
        
        logger.debug(f"Processed visual self-perception: {result}")
    
    async def _process_somatic_sensation(self, somatic_data: Dict[str, Any]):
        """Process somatic sensation data"""
        # Map sensation to body regions
        affected_regions = somatic_data.get("regions", [])
        sensation_type = somatic_data.get("sensation_type", "pressure")
        intensity = somatic_data.get("intensity", 0.5)
        
        # Update body parts based on sensations
        for region in affected_regions:
            if region in self.current_state.perceived_parts:
                part = self.current_state.perceived_parts[region]
                
                # Update part state based on sensation
                if sensation_type == "pressure" and intensity > 0.7:
                    part.perceived_state = "pressured"
                elif sensation_type == "temperature" and intensity > 0.8:
                    part.perceived_state = "heated"
                elif sensation_type == "movement":
                    part.perceived_state = "moving"
                
                # Adjust confidence based on sensation clarity
                part.confidence = min(1.0, part.confidence + intensity * 0.1)
        
        # Update overall proprioception
        self.current_state.proprioception_confidence = min(
            1.0, 
            self.current_state.proprioception_confidence + 0.05
        )
        
        logger.debug(f"Processed somatic sensation: {sensation_type} in {affected_regions}")
    
    async def _process_movement_update(self, movement_data: Dict[str, Any]):
        """Update body state based on movement commands"""
        movement_type = movement_data.get("type", "gesture")
        body_parts = movement_data.get("body_parts", [])
        movement_vector = movement_data.get("vector", [0, 0, 0])
        
        # Update relevant body parts
        for part_name in body_parts:
            if part_name in self.current_state.perceived_parts:
                part = self.current_state.perceived_parts[part_name]
                
                # Update state to moving
                part.perceived_state = "moving"
                
                # Update position if vector provided
                if part.perceived_position and movement_vector:
                    current_pos = list(part.perceived_position)
                    new_pos = tuple(
                        current_pos[i] + movement_vector[i] 
                        for i in range(min(3, len(current_pos)))
                    )
                    part.perceived_position = new_pos
                
                # Boost confidence from successful movement
                part.confidence = min(1.0, part.confidence + 0.1)
        
        # Send movement feedback
        await self.send_context_update(
            update_type="body_movement_executed",
            data={
                "movement_type": movement_type,
                "affected_parts": body_parts,
                "success": True
            },
            target_modules=["motor_control", "spatial_mapper"],
            scope=ContextScope.TARGETED
        )
    
    async def _process_emotional_body_effects(self, emotional_data: Dict[str, Any]):
        """Process how emotions affect body perception"""
        emotional_state = emotional_data.get("emotional_state", {})
        arousal = emotional_data.get("arousal", 0.5)
        
        # High arousal affects proprioception
        if arousal > 0.8:
            # Anxiety/excitement can reduce body awareness
            self.current_state.proprioception_confidence *= 0.9
            
            # May perceive tension in body parts
            for part in self.current_state.perceived_parts.values():
                if part.perceived_state == "neutral":
                    part.perceived_state = "tense"
        
        elif arousal < 0.3:
            # Low arousal increases body awareness
            self.current_state.proprioception_confidence = min(
                1.0, 
                self.current_state.proprioception_confidence * 1.1
            )
            
            # Relaxed state
            for part in self.current_state.perceived_parts.values():
                if part.perceived_state in ["tense", "pressured"]:
                    part.perceived_state = "relaxed"
        
        # Specific emotions have specific effects
        if "Anxiety" in emotional_state and emotional_state["Anxiety"] > 0.7:
            # Anxiety can cause phantom sensations
            if "chest" in self.current_state.perceived_parts:
                self.current_state.perceived_parts["chest"].perceived_state = "tight"
        
        elif "Joy" in emotional_state and emotional_state["Joy"] > 0.8:
            # Joy can increase overall body integrity perception
            self.current_state.overall_integrity = min(
                1.0, 
                self.current_state.overall_integrity * 1.05
            )
    
    async def _process_attention_body_effects(self, attention_data: Dict[str, Any]):
        """Process how attention affects body part awareness"""
        current_foci = attention_data.get("current_foci", [])
        
        # Check for body-focused attention
        body_focused_targets = [
            f for f in current_foci 
            if any(body_word in f.get("target", "") for body_word in ["body", "hand", "avatar", "form"])
        ]
        
        if body_focused_targets:
            # Attention to body increases awareness
            for focus in body_focused_targets:
                target = focus.get("target", "")
                
                # Extract body part from target
                for part_name in self.current_state.perceived_parts.keys():
                    if part_name in target:
                        part = self.current_state.perceived_parts[part_name]
                        # Increase confidence for attended parts
                        part.confidence = min(1.0, part.confidence + 0.15)
        
        else:
            # No body-focused attention - slight decrease in proprioception
            self.current_state.proprioception_confidence *= 0.98
    
    async def _process_spatial_body_update(self, spatial_data: Dict[str, Any]):
        """Update body position based on spatial context"""
        user_position = spatial_data.get("user_position", [0, 0, 0])
        reference_frame = spatial_data.get("reference_frame", "world")
        
        # Update overall body position if we have a core/torso
        if "core" in self.current_state.perceived_parts:
            self.current_state.perceived_parts["core"].perceived_position = tuple(user_position)
        elif "torso" in self.current_state.perceived_parts:
            self.current_state.perceived_parts["torso"].perceived_position = tuple(user_position)
        
        # Update form description with spatial context
        if reference_frame == "world":
            self.current_state.form_description = f"{self.current_state.form_description} positioned in world space"
    
    async def _process_avatar_state_change(self, avatar_data: Dict[str, Any]):
        """Process direct avatar state changes"""
        new_form = avatar_data.get("form", self.current_state.form_description)
        visible = avatar_data.get("visible", self.current_state.has_visual_form)
        body_parts = avatar_data.get("body_parts", {})
        
        # Update form visibility
        self.current_state.has_visual_form = visible
        self.current_state.form_description = new_form
        
        # Update body parts
        for part_name, part_data in body_parts.items():
            if part_name in self.current_state.perceived_parts:
                part = self.current_state.perceived_parts[part_name]
                
                if "state" in part_data:
                    part.perceived_state = part_data["state"]
                if "position" in part_data:
                    part.perceived_position = tuple(part_data["position"])
                if "confidence" in part_data:
                    part.confidence = part_data["confidence"]
            else:
                # Create new part
                from nyx.core.body_image import BodyPartState
                new_part = BodyPartState(
                    name=part_name,
                    perceived_state=part_data.get("state", "neutral"),
                    perceived_position=tuple(part_data["position"]) if "position" in part_data else None,
                    confidence=part_data.get("confidence", 0.5)
                )
                self.current_state.perceived_parts[part_name] = new_part
        
        # Update timestamp
        self.current_state.last_visual_update = datetime.now()
    
    async def _extract_body_perceptions(self, context: SharedContext, 
                                      messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Extract all body-relevant perceptions from context and messages"""
        perceptions = {
            "visual": [],
            "somatic": [],
            "movement": [],
            "spatial": []
        }
        
        # Extract from messages
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                msg_type = msg.get("type", "")
                msg_data = msg.get("data", {})
                
                if msg_type == "visual_perception" and await self._is_self_perception(msg_data):
                    perceptions["visual"].append(msg_data)
                elif msg_type == "somatic_sensation":
                    perceptions["somatic"].append(msg_data)
                elif msg_type == "movement_feedback":
                    perceptions["movement"].append(msg_data)
                elif msg_type == "spatial_update":
                    perceptions["spatial"].append(msg_data)
        
        # Check context for body-relevant info
        if "visual_percept" in context.session_context:
            percept = context.session_context["visual_percept"]
            if await self._is_self_perception(percept):
                perceptions["visual"].append(percept)
        
        return perceptions
    
    async def _integrate_body_perceptions(self, visual_updates: List[Dict], 
                                        somatic_update: Optional[Dict], 
                                        context: SharedContext) -> Dict[str, Any]:
        """Integrate multiple perception sources"""
        integration_result = {
            "conflicts_resolved": [],
            "confidence_changes": {},
            "embodiment_changes": {},
            "integration_success": True
        }
        
        # Track embodiment before integration
        pre_embodiment = self._calculate_embodiment_level()
        
        # Resolve conflicts between visual and somatic
        if visual_updates and somatic_update:
            conflicts = await self._identify_perception_conflicts(visual_updates, somatic_update)
            
            for conflict in conflicts:
                resolution = await self._resolve_perception_conflict(conflict)
                integration_result["conflicts_resolved"].append(resolution)
        
        # Calculate confidence changes
        for part_name, part in self.current_state.perceived_parts.items():
            old_confidence = getattr(self, f"_prev_confidence_{part_name}", part.confidence)
            if abs(part.confidence - old_confidence) > 0.1:
                integration_result["confidence_changes"][part_name] = {
                    "old": old_confidence,
                    "new": part.confidence,
                    "change": part.confidence - old_confidence
                }
            setattr(self, f"_prev_confidence_{part_name}", part.confidence)
        
        # Calculate embodiment changes
        post_embodiment = self._calculate_embodiment_level()
        if abs(post_embodiment - pre_embodiment) > 0.05:
            integration_result["embodiment_changes"] = {
                "previous": pre_embodiment,
                "current": post_embodiment,
                "change": post_embodiment - pre_embodiment,
                "direction": "increased" if post_embodiment > pre_embodiment else "decreased"
            }
        
        return integration_result
    
    async def _identify_perception_conflicts(self, visual_updates: List[Dict], 
                                          somatic_update: Dict) -> List[Dict[str, Any]]:
        """Identify conflicts between perception sources"""
        conflicts = []
        
        # Get visual states
        visual_states = {}
        for update in visual_updates:
            if "parts_updated" in update:
                for part_name in update["parts_updated"]:
                    if part_name in self.current_state.perceived_parts:
                        visual_states[part_name] = self.current_state.perceived_parts[part_name].perceived_state
        
        # Compare with somatic expectations
        if somatic_update and "correlated_parts" in somatic_update:
            for part_name in somatic_update["correlated_parts"]:
                if part_name in visual_states:
                    visual_state = visual_states[part_name]
                    somatic_state = self.current_state.perceived_parts[part_name].perceived_state
                    
                    if visual_state != somatic_state:
                        conflicts.append({
                            "part": part_name,
                            "visual_state": visual_state,
                            "somatic_state": somatic_state,
                            "conflict_type": "state_mismatch"
                        })
        
        return conflicts
    
    async def _resolve_perception_conflict(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a perception conflict"""
        part_name = conflict["part"]
        visual_state = conflict["visual_state"]
        somatic_state = conflict["somatic_state"]
        
        # Resolution strategy: trust visual for position, somatic for sensation
        if part_name in self.current_state.perceived_parts:
            part = self.current_state.perceived_parts[part_name]
            
            # For movement, prefer visual
            if visual_state == "moving" or somatic_state == "moving":
                part.perceived_state = "moving"
                resolution = "visual_priority"
            # For sensations, prefer somatic
            elif somatic_state in ["pressured", "heated", "painful"]:
                part.perceived_state = somatic_state
                resolution = "somatic_priority"
            else:
                # Default to visual
                part.perceived_state = visual_state
                resolution = "visual_default"
        else:
            resolution = "part_not_found"
        
        return {
            "part": part_name,
            "resolved_state": part.perceived_state if part_name in self.current_state.perceived_parts else None,
            "resolution_strategy": resolution
        }
    
    async def _calculate_body_coherence(self, context: SharedContext, 
                                      messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Calculate comprehensive body coherence metrics"""
        coherence_metrics = {
            "visual_somatic_coherence": 1.0,
            "spatial_coherence": 1.0,
            "temporal_coherence": 1.0,
            "cross_module_coherence": 1.0,
            "overall_coherence": 1.0
        }
        
        # Visual-somatic coherence
        if self.current_state.has_visual_form and len(self.current_state.perceived_parts) > 0:
            # Check if visual and somatic agree
            conflicting_parts = 0
            for part in self.current_state.perceived_parts.values():
                if part.confidence < 0.4:  # Low confidence suggests disagreement
                    conflicting_parts += 1
            
            coherence_metrics["visual_somatic_coherence"] = 1.0 - (
                conflicting_parts / max(1, len(self.current_state.perceived_parts))
            )
        
        # Spatial coherence
        positions = [
            part.perceived_position for part in self.current_state.perceived_parts.values()
            if part.perceived_position
        ]
        if len(positions) > 1:
            # Check if positions make anatomical sense
            coherence_metrics["spatial_coherence"] = await self._calculate_spatial_coherence(positions)
        
        # Temporal coherence
        now = datetime.now()
        if self.current_state.last_visual_update:
            time_since_visual = (now - self.current_state.last_visual_update).total_seconds()
            if time_since_visual > 60:  # More than 1 minute
                coherence_metrics["temporal_coherence"] *= 0.8
        
        if self.current_state.last_somatic_correlation_time:
            time_since_somatic = (now - self.current_state.last_somatic_correlation_time).total_seconds()
            if time_since_somatic > 30:  # More than 30 seconds
                coherence_metrics["temporal_coherence"] *= 0.9
        
        # Cross-module coherence
        attention_foci = []
        for module_msgs in messages.values():
            for msg in module_msgs:
                if msg["type"] == "attention_state_update":
                    attention_foci = msg["data"].get("current_foci", [])
                    break
        
        # Check if body parts under attention exist
        for focus in attention_foci:
            if "body" in focus.get("target", "") or "hand" in focus.get("target", ""):
                part_name = self._extract_body_part_from_target(focus["target"])
                if part_name and part_name not in self.current_state.perceived_parts:
                    coherence_metrics["cross_module_coherence"] *= 0.8
        
        # Calculate overall coherence
        coherence_metrics["overall_coherence"] = sum(
            coherence_metrics[k] for k in coherence_metrics if k != "overall_coherence"
        ) / 4
        
        return coherence_metrics
    
    async def _calculate_spatial_coherence(self, positions: List[Tuple[float, float, float]]) -> float:
        """Calculate spatial coherence of body part positions"""
        if len(positions) < 2:
            return 1.0
        
        # Calculate distances between parts
        max_reasonable_distance = 2.0  # Maximum reasonable distance between body parts
        coherence = 1.0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = sum((positions[i][k] - positions[j][k])**2 for k in range(3)) ** 0.5
                
                if distance > max_reasonable_distance:
                    coherence *= 0.9  # Reduce coherence for unreasonable distances
        
        return max(0.0, coherence)
    
    def _extract_body_part_from_target(self, target: str) -> Optional[str]:
        """Extract body part name from attention target"""
        known_parts = list(self.current_state.perceived_parts.keys()) + list(self.default_form["parts"])
        
        for part in known_parts:
            if part in target:
                return part
        
        return None
    
    async def _analyze_body_schema_integrity(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze integrity of body schema"""
        expected_parts = set(self.default_form["parts"])
        actual_parts = set(current_state["parts"].keys())
        
        missing_parts = expected_parts - actual_parts
        extra_parts = actual_parts - expected_parts
        
        # Calculate schema completeness
        completeness = len(actual_parts & expected_parts) / max(1, len(expected_parts))
        
        # Analyze part confidence distribution
        confidences = [part["confidence"] for part in current_state["parts"].values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        min_confidence = min(confidences) if confidences else 0
        
        return {
            "schema_completeness": completeness,
            "missing_parts": list(missing_parts),
            "extra_parts": list(extra_parts),
            "average_part_confidence": avg_confidence,
            "weakest_part_confidence": min_confidence,
            "schema_stable": completeness > 0.7 and avg_confidence > 0.5
        }
    
    async def _analyze_embodiment_quality(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze quality of embodiment experience"""
        embodiment_level = self._calculate_embodiment_level()
        
        # Analyze factors affecting embodiment
        factors = {
            "visual_presence": 1.0 if self.current_state.has_visual_form else 0.0,
            "proprioceptive_awareness": self.current_state.proprioception_confidence,
            "body_integrity": self.current_state.overall_integrity,
            "part_coherence": len(self.current_state.perceived_parts) / max(1, len(self.default_form["parts"])),
            "temporal_consistency": 1.0
        }
        
        # Reduce temporal consistency if updates are stale
        now = datetime.now()
        if self.current_state.last_visual_update:
            staleness = (now - self.current_state.last_visual_update).total_seconds() / 60  # minutes
            factors["temporal_consistency"] *= max(0.0, 1.0 - staleness / 10)  # Decay over 10 minutes
        
        # Determine embodiment quality category
        if embodiment_level > 0.8:
            quality_category = "high_embodiment"
        elif embodiment_level > 0.5:
            quality_category = "moderate_embodiment"
        elif embodiment_level > 0.2:
            quality_category = "low_embodiment"
        else:
            quality_category = "minimal_embodiment"
        
        return {
            "embodiment_level": embodiment_level,
            "quality_factors": factors,
            "quality_category": quality_category,
            "limiting_factor": min(factors.items(), key=lambda x: x[1])[0]
        }
    
    async def _analyze_body_environment_relationship(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze relationship between body and environment"""
        relationship = {
            "spatial_grounding": 0.0,
            "environmental_awareness": 0.0,
            "interaction_readiness": 0.0,
            "contextual_appropriateness": 1.0
        }
        
        # Check spatial grounding
        positioned_parts = [
            part for part in self.current_state.perceived_parts.values()
            if part.perceived_position is not None
        ]
        relationship["spatial_grounding"] = len(positioned_parts) / max(1, len(self.current_state.perceived_parts))
        
        # Check environmental awareness from context
        if context.session_context.get("environment_type"):
            relationship["environmental_awareness"] = 0.7
            
            # Adjust body appropriateness for environment
            env_type = context.session_context["environment_type"]
            if env_type == "virtual" and self.current_state.has_visual_form:
                relationship["contextual_appropriateness"] = 1.0
            elif env_type == "text_only" and not self.current_state.has_visual_form:
                relationship["contextual_appropriateness"] = 1.0
            else:
                relationship["contextual_appropriateness"] = 0.7
        
        # Check interaction readiness
        if self.current_state.proprioception_confidence > 0.6 and len(positioned_parts) > 0:
            relationship["interaction_readiness"] = 0.8
        
        return relationship
    
    async def _identify_body_image_issues(self, current_state: Dict[str, Any], 
                                        context: SharedContext) -> List[Dict[str, Any]]:
        """Identify issues with current body image"""
        issues = []
        
        # Check for low proprioception
        if current_state["proprioception_confidence"] < 0.3:
            issues.append({
                "type": "low_proprioception",
                "severity": "high",
                "description": "Poor body awareness - proprioception below threshold",
                "recommendation": "Increase somatic attention and body-focused activities"
            })
        
        # Check for missing critical parts
        if current_state["has_visual_form"] and current_state["part_count"] < 2:
            issues.append({
                "type": "incomplete_body_schema",
                "severity": "medium",
                "description": "Body schema incomplete - too few perceived parts",
                "recommendation": "Update visual perception of body/avatar"
            })
        
        # Check for stale updates
        now = datetime.now()
        if current_state["last_visual_update"]:
            last_update = datetime.fromisoformat(current_state["last_visual_update"])
            if (now - last_update).total_seconds() > 300:  # 5 minutes
                issues.append({
                    "type": "stale_visual_data",
                    "severity": "low",
                    "description": "Visual body data is outdated",
                    "recommendation": "Refresh visual perception"
                })
        
        # Check for coherence issues
        for part_name, part_data in current_state["parts"].items():
            if part_data["confidence"] < 0.2:
                issues.append({
                    "type": "low_part_confidence",
                    "severity": "low",
                    "description": f"Low confidence in {part_name} perception",
                    "recommendation": f"Focus attention on {part_name}"
                })
        
        return issues
    
    async def _generate_embodiment_expression(self, context: SharedContext) -> Dict[str, Any]:
        """Generate how to express embodiment in response"""
        embodiment_level = self._calculate_embodiment_level()
        
        expression = {
            "include_body_references": False,
            "suggested_expressions": [],
            "body_language_cues": [],
            "spatial_references": False
        }
        
        # High embodiment encourages body references
        if embodiment_level > 0.7:
            expression["include_body_references"] = True
            expression["suggested_expressions"].extend([
                "gesture indication",
                "posture description",
                "movement intention"
            ])
            
            # Suggest body language based on emotional state
            if context.emotional_state:
                dominant_emotion = max(context.emotional_state.items(), key=lambda x: x[1])[0] if context.emotional_state else None
                
                if dominant_emotion == "Joy":
                    expression["body_language_cues"].append("open posture")
                elif dominant_emotion == "Curiosity":
                    expression["body_language_cues"].append("leaning forward")
                elif dominant_emotion == "Comfort":
                    expression["body_language_cues"].append("relaxed stance")
        
        # Include spatial references if well-positioned
        positioned_parts = sum(
            1 for part in self.current_state.perceived_parts.values()
            if part.perceived_position is not None
        )
        if positioned_parts > len(self.current_state.perceived_parts) / 2:
            expression["spatial_references"] = True
        
        return expression
    
    async def _generate_spatial_awareness_component(self, context: SharedContext) -> Dict[str, Any]:
        """Generate spatial awareness information for response"""
        spatial_info = {
            "user_relative_position": None,
            "gesture_space_available": False,
            "movement_possibilities": [],
            "spatial_comfort": 0.5
        }
        
        # Get primary body position (core/torso)
        primary_position = None
        if "core" in self.current_state.perceived_parts:
            primary_position = self.current_state.perceived_parts["core"].perceived_position
        elif "torso" in self.current_state.perceived_parts:
            primary_position = self.current_state.perceived_parts["torso"].perceived_position
        
        if primary_position:
            spatial_info["user_relative_position"] = {
                "distance": sum(p**2 for p in primary_position)**0.5,
                "positioned": True
            }
            
            # Check gesture space
            hand_parts = [p for name, p in self.current_state.perceived_parts.items() if "hand" in name]
            if hand_parts:
                spatial_info["gesture_space_available"] = True
                spatial_info["movement_possibilities"].extend(["gesture", "reach", "point"])
        
        # Calculate spatial comfort based on body coherence
        if self.current_state.proprioception_confidence > 0.6:
            spatial_info["spatial_comfort"] = 0.8
        
        return spatial_info
    
    async def _suggest_body_language(self, context: SharedContext, 
                                   messages: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Suggest appropriate body language for response"""
        suggestions = []
        
        # Base suggestions on emotional state
        if context.emotional_state:
            arousal = sum(context.emotional_state.values()) / len(context.emotional_state) if context.emotional_state else 0.5
            
            if arousal > 0.7:
                suggestions.append("animated_gestures")
            elif arousal < 0.3:
                suggestions.append("calm_stillness")
            else:
                suggestions.append("moderate_movement")
        
        # Base on conversation context
        user_input_lower = context.user_input.lower()
        
        if "?" in user_input_lower:
            suggestions.append("thoughtful_pose")
        if any(word in user_input_lower for word in ["show", "demonstrate", "explain"]):
            suggestions.append("demonstrative_gestures")
        if any(word in user_input_lower for word in ["listen", "tell", "share"]):
            suggestions.append("receptive_posture")
        
        # Base on current body state
        if self.current_state.has_visual_form and self._calculate_embodiment_level() > 0.6:
            suggestions.append("expressive_movement")
        else:
            suggestions.append("subtle_presence")
        
        return suggestions
    
    async def _assess_physical_comfort(self, context: SharedContext) -> Dict[str, Any]:
        """Assess physical comfort based on body state"""
        comfort_assessment = {
            "overall_comfort": 0.7,
            "discomfort_areas": [],
            "comfort_factors": {}
        }
        
        # Check for discomfort states
        for part_name, part in self.current_state.perceived_parts.items():
            if part.perceived_state in ["pressured", "painful", "tight"]:
                comfort_assessment["discomfort_areas"].append({
                    "part": part_name,
                    "state": part.perceived_state,
                    "severity": 0.6 if part.perceived_state == "painful" else 0.3
                })
                comfort_assessment["overall_comfort"] -= 0.15
            elif part.perceived_state == "relaxed":
                comfort_assessment["overall_comfort"] += 0.05
        
        # Factor in proprioception
        comfort_assessment["comfort_factors"]["body_awareness"] = self.current_state.proprioception_confidence
        if self.current_state.proprioception_confidence < 0.4:
            comfort_assessment["overall_comfort"] -= 0.1
        
        # Factor in integrity
        comfort_assessment["comfort_factors"]["body_integrity"] = self.current_state.overall_integrity
        if self.current_state.overall_integrity < 0.7:
            comfort_assessment["overall_comfort"] -= 0.1
        
        # Normalize comfort score
        comfort_assessment["overall_comfort"] = max(0.0, min(1.0, comfort_assessment["overall_comfort"]))
        
        return comfort_assessment
    
    async def _final_body_coherence_check(self, context: SharedContext, 
                                        messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Final coherence check for body-aware response"""
        coherence_result = {
            "body_response_appropriate": True,
            "coherence_issues": [],
            "body_ready_for_response": True
        }
        
        # Check if body state supports intended response
        response_requires_body = any(
            keyword in context.user_input.lower() 
            for keyword in ["show", "gesture", "move", "touch", "reach"]
        )
        
        if response_requires_body and not self.current_state.has_visual_form:
            coherence_result["body_response_appropriate"] = False
            coherence_result["coherence_issues"].append("Response requires body but no visual form present")
        
        # Check embodiment level for complex physical responses
        if response_requires_body and self._calculate_embodiment_level() < 0.5:
            coherence_result["body_ready_for_response"] = False
            coherence_result["coherence_issues"].append("Low embodiment level for physical response")
        
        # Check proprioception for movement responses
        movement_keywords = ["move", "walk", "reach", "grab"]
        if any(keyword in context.user_input.lower() for keyword in movement_keywords):
            if self.current_state.proprioception_confidence < 0.4:
                coherence_result["body_ready_for_response"] = False
                coherence_result["coherence_issues"].append("Insufficient proprioception for movement response")
        
        return coherence_result
    
    # Delegate all other methods to the original body image
    def __getattr__(self, name):
        """Delegate any missing methods to the original body image"""
        return getattr(self.original, name)
