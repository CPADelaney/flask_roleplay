# nyx/core/a2a/context_aware_game_vision.py

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time
import numpy as np
import asyncio

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareGameVision(ContextAwareModule):
    """
    Context-aware game vision system that coordinates visual analysis with other modules
    """
    
    def __init__(self, original_vision_system):
        super().__init__("game_vision")
        self.original_system = original_vision_system
        self.context_subscriptions = [
            "emotional_state_update", "memory_retrieval_complete", "goal_context_available",
            "cross_game_knowledge_available", "attention_focus", "mode_change",
            "user_input_processed", "speech_transcribed"
        ]
        
        # Visual analysis state
        self.current_frame_analysis = {}
        self.analysis_history = []
        self.attention_targets = []
        self.multi_modal_events = []
        
    async def on_context_received(self, context: SharedContext):
        """Initialize visual processing for this context"""
        logger.debug(f"GameVision received context for user: {context.user_id}")
        
        # Extract any visual processing hints from context
        task_purpose = context.task_purpose
        attention_hints = context.session_context.get("visual_focus", [])
        
        # If we have attention hints, use them
        if attention_hints:
            self.attention_targets = attention_hints
        
        # Send initial visual context if we have a current frame
        if hasattr(self.original_system, 'last_analysis') and self.original_system.last_analysis:
            await self.send_context_update(
                update_type="visual_context_available",
                data={
                    "current_analysis": self.original_system.last_analysis,
                    "game_identified": bool(self.original_system.current_game_id),
                    "tracking_active": True
                },
                priority=ContextPriority.HIGH
            )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect visual processing"""
        
        if update.update_type == "attention_focus":
            # Attention controller wants us to focus on something
            attention_data = update.data
            focus_targets = attention_data.get("focus_targets", [])
            
            # Update our attention targets
            self.attention_targets = focus_targets
            
            # Adjust visual processing priorities
            await self._adjust_visual_priorities(focus_targets)
        
        elif update.update_type == "emotional_state_update":
            # Emotional state might affect what we look for
            emotional_data = update.data
            dominant_emotion = emotional_data.get("dominant_emotion")
            
            if dominant_emotion:
                # Adjust visual analysis based on emotion
                await self._adjust_for_emotion(dominant_emotion)
        
        elif update.update_type == "memory_retrieval_complete":
            # Use memories to enhance visual recognition
            memory_data = update.data
            memories = memory_data.get("retrieved_memories", [])
            
            # Extract visual patterns from memories
            visual_patterns = await self._extract_visual_patterns(memories)
            
            if visual_patterns:
                # Use patterns to improve recognition
                await self._enhance_recognition_with_patterns(visual_patterns)
        
        elif update.update_type == "cross_game_knowledge_available":
            # Cross-game knowledge can help with visual recognition
            knowledge_data = update.data
            similar_games = knowledge_data.get("similar_games", [])
            
            # Load visual patterns from similar games
            if similar_games and hasattr(self.original_system, 'knowledge_base'):
                for game in similar_games[:2]:  # Top 2 similar games
                    game_id = game.get("game_id")
                    if game_id:
                        self.original_system.knowledge_base.load_game_data(game_id)
        
        elif update.update_type == "speech_transcribed":
            # Speech might give context for visual analysis
            speech_data = update.data
            text = speech_data.get("text", "")
            
            # Look for visual cues in speech
            visual_cues = self._extract_visual_cues_from_speech(text)
            
            if visual_cues:
                # Add to multi-modal events
                self.multi_modal_events.append({
                    "type": "speech_visual_correlation",
                    "speech": text,
                    "visual_cues": visual_cues,
                    "timestamp": time.time()
                })
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process visual input with context awareness"""
        # Check if we have a frame to process
        if not hasattr(self.original_system, 'last_frame') or self.original_system.last_frame is None:
            return {"visual_processing": False, "reason": "no_frame_available"}
        
        # Get current frame
        frame = self.original_system.last_frame
        
        # Analyze frame with context
        analysis = await self._analyze_frame_with_context(frame, context)
        
        # Update current analysis
        self.current_frame_analysis = analysis
        self.analysis_history.append(analysis)
        
        # Keep history limited
        if len(self.analysis_history) > 10:
            self.analysis_history.pop(0)
        
        # Send visual updates to other modules
        await self._send_visual_updates(analysis)
        
        return {
            "visual_processing": True,
            "frame_analyzed": True,
            "objects_detected": len(analysis.get("objects", [])),
            "current_location": analysis.get("location", {}).get("name"),
            "action_detected": analysis.get("action", {}).get("name")
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze visual patterns and trends"""
        messages = await self.get_cross_module_messages()
        
        # Analyze visual trends
        trend_analysis = await self._analyze_visual_trends()
        
        # Detect patterns across frames
        pattern_analysis = await self._detect_visual_patterns()
        
        # Analyze multi-modal correlations
        multi_modal_analysis = await self._analyze_multi_modal_events(messages)
        
        # Scene understanding
        scene_understanding = await self._deep_scene_analysis(context, messages)
        
        return {
            "visual_trends": trend_analysis,
            "detected_patterns": pattern_analysis,
            "multi_modal_insights": multi_modal_analysis,
            "scene_understanding": scene_understanding,
            "attention_effectiveness": await self._evaluate_attention_effectiveness()
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize visual information for response generation"""
        messages = await self.get_cross_module_messages()
        
        # Prepare visual context for response
        synthesis = {
            "visual_description": await self._generate_scene_description(),
            "notable_elements": await self._identify_notable_elements(),
            "visual_cues_for_response": await self._extract_response_cues(context),
            "spatial_context": await self._get_spatial_context(),
            "action_context": await self._get_action_context()
        }
        
        # Add game-specific visual insights
        if self.original_system.current_game_id:
            game_insights = await self._get_game_specific_insights()
            synthesis["game_visual_insights"] = game_insights
        
        # Add emotional visual cues
        emotional_visuals = await self._get_emotional_visual_cues(context.emotional_state)
        synthesis["emotional_visual_cues"] = emotional_visuals
        
        return synthesis
    
    # Helper methods
    
    async def _analyze_frame_with_context(self, frame: np.ndarray, context: SharedContext) -> Dict[str, Any]:
        """Analyze frame with full context awareness"""
        # Use original system's analysis as base
        if hasattr(self.original_system, 'analyze_frame'):
            base_analysis = await self.original_system.analyze_frame(frame)
        else:
            base_analysis = {}
        
        # Enhance with context
        enhanced_analysis = base_analysis.copy()
        
        # Apply attention focus if we have targets
        if self.attention_targets:
            focused_objects = await self._apply_attention_focus(
                base_analysis.get("objects", []),
                self.attention_targets
            )
            enhanced_analysis["focused_objects"] = focused_objects
        
        # Add context-aware scene interpretation
        scene_context = await self._interpret_scene_with_context(base_analysis, context)
        enhanced_analysis["scene_context"] = scene_context
        
        # Check for multi-modal events
        if self.multi_modal_events:
            recent_event = self.multi_modal_events[-1]
            if time.time() - recent_event["timestamp"] < 2.0:  # Within 2 seconds
                enhanced_analysis["multi_modal_correlation"] = recent_event
        
        return enhanced_analysis
    
    async def _send_visual_updates(self, analysis: Dict[str, Any]):
        """Send relevant visual updates to other modules"""
        # Game identification update
        game_info = analysis.get("game", {})
        if game_info.get("is_new") or game_info.get("confidence", 0) > 0.9:
            await self.send_context_update(
                update_type="game_identified",
                data={
                    "game_id": game_info.get("game_id"),
                    "game_name": game_info.get("game_name"),
                    "confidence": game_info.get("confidence", 0)
                },
                priority=ContextPriority.HIGH
            )
        
        # Location update
        location = analysis.get("location", {})
        if location.get("is_new") or location.get("confidence", 0) > 0.8:
            await self.send_context_update(
                update_type="location_change",
                data={
                    "location_id": location.get("id"),
                    "location_name": location.get("name"),
                    "location_type": location.get("type"),
                    "visit_count": location.get("visit_count", 1)
                }
            )
        
        # Action detection
        action = analysis.get("action", {})
        if action.get("name") and action.get("confidence", 0) > 0.7:
            await self.send_context_update(
                update_type="action_detected",
                data={
                    "action_id": action.get("id"),
                    "action_type": action.get("name"),
                    "action_confidence": action.get("confidence", 0),
                    "involves_player": action.get("involves_player", True)
                }
            )
        
        # Significant visual events
        events = analysis.get("events", [])
        for event in events:
            if event.get("significance", 0) > 6:
                await self.send_context_update(
                    update_type="significant_visual_event",
                    data=event,
                    priority=ContextPriority.HIGH
                )
    
    async def _adjust_visual_priorities(self, focus_targets: List[str]):
        """Adjust visual processing based on attention targets"""
        # In real implementation, this would adjust detection thresholds
        # and processing priorities for different visual elements
        logger.debug(f"Adjusting visual priorities for targets: {focus_targets}")
        
        # Store for use in frame analysis
        self.attention_targets = focus_targets
    
    async def _adjust_for_emotion(self, emotion_data: Tuple[str, float]):
        """Adjust visual processing based on emotional state"""
        emotion_name, intensity = emotion_data
        
        # Different emotions might make us look for different things
        if emotion_name == "Fear":
            # Look more carefully for threats
            self.attention_targets.append("threats")
            self.attention_targets.append("enemies")
        elif emotion_name == "Curiosity":
            # Look for interesting objects and areas
            self.attention_targets.append("interactables")
            self.attention_targets.append("unexplored_areas")
        elif emotion_name == "Joy":
            # Look for rewards and positive elements
            self.attention_targets.append("rewards")
            self.attention_targets.append("collectibles")
    
    async def _extract_visual_patterns(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract visual patterns from memories"""
        patterns = []
        
        for memory in memories:
            # Look for visual descriptions in memory
            memory_text = memory.get("text", "").lower()
            
            # Simple pattern extraction
            if "looks like" in memory_text or "appears to be" in memory_text:
                patterns.append({
                    "type": "visual_memory",
                    "description": memory_text,
                    "confidence": memory.get("confidence", 0.5)
                })
        
        return patterns
    
    async def _enhance_recognition_with_patterns(self, patterns: List[Dict[str, Any]]):
        """Use patterns to improve recognition accuracy"""
        # In real implementation, this would update recognition models
        # or adjust detection parameters
        logger.debug(f"Enhancing recognition with {len(patterns)} patterns")
    
    def _extract_visual_cues_from_speech(self, text: str) -> List[str]:
        """Extract visual cues mentioned in speech"""
        visual_keywords = [
            "look", "see", "watch", "notice", "appears", "seems",
            "red", "blue", "green", "bright", "dark", "large", "small",
            "left", "right", "up", "down", "behind", "front"
        ]
        
        cues = []
        text_lower = text.lower()
        
        for keyword in visual_keywords:
            if keyword in text_lower:
                # Extract phrase around keyword
                words = text_lower.split()
                if keyword in words:
                    idx = words.index(keyword)
                    start = max(0, idx - 2)
                    end = min(len(words), idx + 3)
                    phrase = " ".join(words[start:end])
                    cues.append(phrase)
        
        return cues
    
    async def _apply_attention_focus(self, objects: List[Dict[str, Any]], targets: List[str]) -> List[Dict[str, Any]]:
        """Apply attention focus to detected objects"""
        focused_objects = []
        
        for obj in objects:
            obj_class = obj.get("class", "").lower()
            obj_name = obj.get("name", "").lower()
            
            # Check if object matches any attention target
            for target in targets:
                target_lower = target.lower()
                if target_lower in obj_class or target_lower in obj_name:
                    # Boost confidence for focused objects
                    focused_obj = obj.copy()
                    focused_obj["attention_boost"] = 0.2
                    focused_obj["focused"] = True
                    focused_objects.append(focused_obj)
                    break
        
        return focused_objects
    
    async def _interpret_scene_with_context(self, analysis: Dict[str, Any], context: SharedContext) -> Dict[str, Any]:
        """Interpret the scene using full context"""
        interpretation = {
            "scene_type": "unknown",
            "activity_level": "normal",
            "mood": "neutral"
        }
        
        # Determine scene type based on detected elements
        objects = analysis.get("objects", [])
        location = analysis.get("location", {})
        
        # Check for combat scene
        if any(obj.get("class") in ["enemy", "weapon", "projectile"] for obj in objects):
            interpretation["scene_type"] = "combat"
            interpretation["activity_level"] = "high"
            interpretation["mood"] = "tense"
        
        # Check for dialog scene
        elif any(obj.get("class") == "npc" for obj in objects) and len(objects) < 5:
            interpretation["scene_type"] = "dialog"
            interpretation["activity_level"] = "low"
            interpretation["mood"] = "conversational"
        
        # Check for exploration
        elif location.get("is_new"):
            interpretation["scene_type"] = "exploration"
            interpretation["activity_level"] = "medium"
            interpretation["mood"] = "curious"
        
        # Add emotional context
        if context.emotional_state:
            dominant_emotion = max(context.emotional_state.items(), key=lambda x: x[1])[0]
            interpretation["emotional_influence"] = dominant_emotion.lower()
        
        return interpretation
    
    async def _analyze_visual_trends(self) -> Dict[str, Any]:
        """Analyze trends in visual data over time"""
        if len(self.analysis_history) < 2:
            return {"trend": "insufficient_data"}
        
        trends = {
            "object_stability": 0.0,
            "location_changes": 0,
            "action_frequency": {},
            "dominant_colors": []
        }
        
        # Calculate object stability (how consistent objects are between frames)
        if len(self.analysis_history) >= 2:
            prev_objects = set(obj.get("class") for obj in self.analysis_history[-2].get("objects", []))
            curr_objects = set(obj.get("class") for obj in self.analysis_history[-1].get("objects", []))
            
            if prev_objects or curr_objects:
                stability = len(prev_objects & curr_objects) / max(len(prev_objects | curr_objects), 1)
                trends["object_stability"] = stability
        
        # Count location changes
        locations = [h.get("location", {}).get("id") for h in self.analysis_history]
        trends["location_changes"] = len(set(locations)) - 1
        
        # Action frequency
        for analysis in self.analysis_history:
            action = analysis.get("action", {}).get("name")
            if action:
                trends["action_frequency"][action] = trends["action_frequency"].get(action, 0) + 1
        
        return trends
    
    async def _detect_visual_patterns(self) -> List[Dict[str, Any]]:
        """Detect patterns in visual data"""
        patterns = []
        
        # Look for repeating object configurations
        if len(self.analysis_history) >= 3:
            # Check for UI patterns (consistent UI elements)
            ui_elements = []
            for analysis in self.analysis_history[-3:]:
                ui = [obj for obj in analysis.get("objects", []) if "ui" in obj.get("class", "").lower()]
                ui_elements.append(ui)
            
            # If UI is consistent, it's likely a menu or HUD
            if all(len(ui) > 0 for ui in ui_elements):
                patterns.append({
                    "type": "consistent_ui",
                    "description": "Persistent UI elements detected",
                    "confidence": 0.9
                })
        
        # Look for action sequences
        recent_actions = [h.get("action", {}).get("name") for h in self.analysis_history[-5:]]
        recent_actions = [a for a in recent_actions if a]  # Filter None
        
        if len(recent_actions) >= 3:
            # Check for repetition
            if len(set(recent_actions)) == 1:
                patterns.append({
                    "type": "repeated_action",
                    "action": recent_actions[0],
                    "count": len(recent_actions),
                    "confidence": 0.8
                })
        
        return patterns
    
    async def _analyze_multi_modal_events(self, messages: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze correlations between visual and other modalities"""
        correlations = {
            "visual_audio_sync": [],
            "visual_speech_sync": [],
            "visual_emotional_sync": []
        }
        
        # Check recent multi-modal events
        recent_events = [e for e in self.multi_modal_events if time.time() - e["timestamp"] < 10]
        
        for event in recent_events:
            if event["type"] == "speech_visual_correlation":
                correlations["visual_speech_sync"].append({
                    "speech": event["speech"],
                    "visual_cues": event["visual_cues"],
                    "confidence": 0.7
                })
        
        # Check for visual-emotional correlations
        for module_name, module_messages in messages.items():
            if module_name == "emotional_core":
                for msg in module_messages:
                    if msg["type"] == "emotional_state_update":
                        # Check if visual scene matches emotion
                        emotion = msg["data"].get("dominant_emotion")
                        if emotion and self.current_frame_analysis:
                            scene_mood = self.current_frame_analysis.get("scene_context", {}).get("mood")
                            if self._emotion_matches_mood(emotion[0], scene_mood):
                                correlations["visual_emotional_sync"].append({
                                    "emotion": emotion[0],
                                    "scene_mood": scene_mood,
                                    "match": True
                                })
        
        return correlations
    
    async def _deep_scene_analysis(self, context: SharedContext, messages: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Perform deep analysis of the current scene"""
        if not self.current_frame_analysis:
            return {"status": "no_current_analysis"}
        
        analysis = {
            "scene_complexity": 0.0,
            "narrative_elements": [],
            "gameplay_stage": "unknown",
            "environmental_storytelling": []
        }
        
        # Calculate scene complexity
        objects = self.current_frame_analysis.get("objects", [])
        analysis["scene_complexity"] = min(1.0, len(objects) / 20.0)
        
        # Identify narrative elements
        for obj in objects:
            if obj.get("class") in ["npc", "character", "quest_giver"]:
                analysis["narrative_elements"].append({
                    "type": "character",
                    "name": obj.get("name", "Unknown")
                })
            elif obj.get("class") in ["note", "book", "terminal"]:
                analysis["narrative_elements"].append({
                    "type": "lore_object",
                    "name": obj.get("name", "Unknown")
                })
        
        # Determine gameplay stage
        action = self.current_frame_analysis.get("action", {})
        if action.get("name") == "combat":
            analysis["gameplay_stage"] = "combat"
        elif action.get("name") == "dialog":
            analysis["gameplay_stage"] = "narrative"
        elif action.get("name") == "exploration":
            analysis["gameplay_stage"] = "exploration"
        
        # Look for environmental storytelling
        location = self.current_frame_analysis.get("location", {})
        if location.get("name"):
            # Simple environmental analysis
            if "ruins" in location["name"].lower():
                analysis["environmental_storytelling"].append("Ancient ruins suggest past civilization")
            elif "battlefield" in location["name"].lower():
                analysis["environmental_storytelling"].append("Signs of past conflict")
        
        return analysis
    
    async def _evaluate_attention_effectiveness(self) -> Dict[str, Any]:
        """Evaluate how effective our attention focusing has been"""
        if not self.attention_targets or not self.analysis_history:
            return {"effectiveness": "not_evaluated"}
        
        # Check if we found what we were looking for
        found_targets = []
        missed_targets = []
        
        for target in self.attention_targets:
            found = False
            for analysis in self.analysis_history[-3:]:  # Check last 3 frames
                objects = analysis.get("objects", [])
                for obj in objects:
                    if target.lower() in obj.get("class", "").lower() or target.lower() in obj.get("name", "").lower():
                        found = True
                        break
                if found:
                    break
            
            if found:
                found_targets.append(target)
            else:
                missed_targets.append(target)
        
        effectiveness = len(found_targets) / max(len(self.attention_targets), 1)
        
        return {
            "effectiveness": effectiveness,
            "found_targets": found_targets,
            "missed_targets": missed_targets
        }
    
    async def _generate_scene_description(self) -> str:
        """Generate natural language description of current scene"""
        if not self.current_frame_analysis:
            return "No visual information available"
        
        # Get scene elements
        location = self.current_frame_analysis.get("location", {})
        objects = self.current_frame_analysis.get("objects", [])
        action = self.current_frame_analysis.get("action", {})
        
        # Build description
        parts = []
        
        if location.get("name"):
            parts.append(f"In {location['name']}")
        
        if objects:
            # Group objects by type
            object_types = {}
            for obj in objects:
                obj_type = obj.get("class", "object")
                if obj_type not in object_types:
                    object_types[obj_type] = 0
                object_types[obj_type] += 1
            
            # Describe main objects
            main_objects = []
            for obj_type, count in sorted(object_types.items(), key=lambda x: x[1], reverse=True)[:3]:
                if count == 1:
                    main_objects.append(f"a {obj_type}")
                else:
                    main_objects.append(f"{count} {obj_type}s")
            
            if main_objects:
                parts.append(f"I can see {', '.join(main_objects)}")
        
        if action.get("name"):
            parts.append(f"Currently {action['name']} is happening")
        
        return ". ".join(parts) if parts else "Observing the game environment"
    
    async def _identify_notable_elements(self) -> List[Dict[str, Any]]:
        """Identify visually notable elements worth mentioning"""
        notable = []
        
        if not self.current_frame_analysis:
            return notable
        
        objects = self.current_frame_analysis.get("objects", [])
        
        # Find high-confidence objects
        for obj in objects:
            if obj.get("confidence", 0) > 0.9:
                notable.append({
                    "type": "high_confidence_object",
                    "object": obj.get("name", obj.get("class")),
                    "reason": "clearly visible"
                })
        
        # Find focused objects
        focused = self.current_frame_analysis.get("focused_objects", [])
        for obj in focused[:2]:  # Top 2 focused objects
            notable.append({
                "type": "attention_target",
                "object": obj.get("name", obj.get("class")),
                "reason": "matches current focus"
            })
        
        # Find new objects (not in previous frame)
        if len(self.analysis_history) >= 2:
            prev_objects = set(obj.get("id") for obj in self.analysis_history[-2].get("objects", []))
            curr_objects = self.current_frame_analysis.get("objects", [])
            
            for obj in curr_objects:
                if obj.get("id") not in prev_objects:
                    notable.append({
                        "type": "new_object",
                        "object": obj.get("name", obj.get("class")),
                        "reason": "just appeared"
                    })
        
        return notable[:5]  # Limit to top 5
    
    async def _extract_response_cues(self, context: SharedContext) -> List[str]:
        """Extract visual cues that should influence response"""
        cues = []
        
        if not self.current_frame_analysis:
            return cues
        
        # Scene mood cues
        scene_context = self.current_frame_analysis.get("scene_context", {})
        mood = scene_context.get("mood")
        
        if mood == "tense":
            cues.append("maintain_alertness")
        elif mood == "conversational":
            cues.append("engage_socially")
        elif mood == "curious":
            cues.append("encourage_exploration")
        
        # Action cues
        action = self.current_frame_analysis.get("action", {})
        if action.get("name") == "combat":
            cues.append("provide_combat_support")
        elif action.get("name") == "puzzle":
            cues.append("offer_problem_solving")
        
        # Multi-modal cues
        if self.current_frame_analysis.get("multi_modal_correlation"):
            cues.append("reference_recent_dialog")
        
        return cues
    
    async def _get_spatial_context(self) -> Dict[str, Any]:
        """Get spatial context information"""
        spatial = {
            "current_location": None,
            "navigation_status": "stationary",
            "spatial_memory": []
        }
        
        if hasattr(self.original_system, 'spatial_memory'):
            spatial_memory = self.original_system.spatial_memory
            
            # Get current location
            current = spatial_memory.get_current_location()
            if current:
                spatial["current_location"] = current["name"]
            
            # Check if we're navigating
            if spatial_memory.current_path:
                spatial["navigation_status"] = "following_path"
                spatial["path_length"] = len(spatial_memory.current_path)
            
            # Get recent locations
            spatial["spatial_memory"] = list(spatial_memory.location_history)[-3:]
        
        return spatial
    
    async def _get_action_context(self) -> Dict[str, Any]:
        """Get action context information"""
        action_context = {
            "current_action": None,
            "action_sequence": [],
            "detected_pattern": None
        }
        
        if hasattr(self.original_system, 'action_recognition'):
            action_rec = self.original_system.action_recognition
            
            # Get current action
            if self.current_frame_analysis:
                current = self.current_frame_analysis.get("action", {})
                action_context["current_action"] = current.get("name")
            
            # Get action sequence
            sequence = action_rec.get_action_sequence(max_length=3)
            action_context["action_sequence"] = [a["name"] for a in sequence]
            
            # Check for patterns
            pattern = action_rec.detect_pattern()
            if pattern:
                action_context["detected_pattern"] = pattern["type"]
        
        return action_context
    
    async def _get_game_specific_insights(self) -> List[str]:
        """Get game-specific visual insights"""
        insights = []
        
        if not self.original_system.current_game_info:
            return insights
        
        game_name = self.original_system.current_game_info.get("name", "")
        genre = self.original_system.current_game_info.get("genre", [])
        
        # Genre-specific insights
        if "RPG" in genre:
            insights.append("Watch for quest markers and NPC indicators")
        elif "FPS" in genre:
            insights.append("Monitor crosshair and ammo indicators")
        elif "Strategy" in genre:
            insights.append("Observe unit positions and resource indicators")
        
        # Game-specific insights
        if hasattr(self.original_system, 'knowledge_base'):
            kb = self.original_system.knowledge_base
            game_elements = kb.get_game_elements(self.original_system.current_game_id)
            
            if game_elements:
                # Add insight about unique game elements
                unique_elements = list(game_elements.keys())[:2]
                if unique_elements:
                    insights.append(f"Look for {', '.join(unique_elements)}")
        
        return insights[:3]
    
    async def _get_emotional_visual_cues(self, emotional_state: Dict[str, float]) -> Dict[str, Any]:
        """Get visual cues based on emotional state"""
        cues = {
            "color_preference": "neutral",
            "focus_areas": [],
            "avoid_areas": []
        }
        
        if not emotional_state:
            return cues
        
        # Get dominant emotion
        dominant = max(emotional_state.items(), key=lambda x: x[1])
        emotion_name, intensity = dominant
        
        # Map emotions to visual preferences
        if emotion_name == "Joy":
            cues["color_preference"] = "bright"
            cues["focus_areas"] = ["rewards", "achievements"]
        elif emotion_name == "Fear":
            cues["color_preference"] = "muted"
            cues["focus_areas"] = ["exits", "safe_areas"]
            cues["avoid_areas"] = ["dark_areas", "enemies"]
        elif emotion_name == "Anger":
            cues["color_preference"] = "intense"
            cues["focus_areas"] = ["targets", "obstacles"]
        elif emotion_name == "Curiosity":
            cues["color_preference"] = "varied"
            cues["focus_areas"] = ["unexplored", "interactables"]
        
        return cues
    
    def _emotion_matches_mood(self, emotion: str, mood: str) -> bool:
        """Check if emotion matches scene mood"""
        emotion_mood_map = {
            "Joy": ["peaceful", "cheerful", "celebratory"],
            "Fear": ["tense", "dark", "threatening"],
            "Anger": ["tense", "combative", "aggressive"],
            "Sadness": ["melancholic", "quiet", "somber"],
            "Curiosity": ["mysterious", "curious", "exploratory"]
        }
        
        if emotion in emotion_mood_map:
            return mood in emotion_mood_map[emotion]
        
        return False
    
    # Delegate other methods to original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
