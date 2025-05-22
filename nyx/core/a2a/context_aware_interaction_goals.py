# nyx/core/a2a/context_aware_interaction_goals.py

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareInteractionGoals(ContextAwareModule):
    """
    Enhanced InteractionGoals with full context distribution capabilities
    """
    
    def __init__(self, original_interaction_goals):
        super().__init__("interaction_goals")
        self.original_goals = original_interaction_goals
        self.context_subscriptions = [
            "mode_distribution_update", "mode_transition", "goal_context_available",
            "emotional_state_update", "needs_assessment", "relationship_state_change",
            "goal_completion", "goal_progress", "dominance_context_update"
        ]
        
        # Cache for mode-specific goals with context awareness
        self.context_aware_goal_cache = {}
        self.active_goal_selections = []
        
    async def on_context_received(self, context: SharedContext):
        """Initialize goal selection for this context"""
        logger.debug(f"InteractionGoals received context for user: {context.user_id}")
        
        # Get current mode distribution from context
        mode_context = context.mode_context
        mode_distribution = mode_context.get("mode_distribution", {}) if mode_context else {}
        
        # Select goals based on current context
        selected_goals = await self._select_contextual_goals(mode_distribution, context)
        
        # Send selected goals to other modules
        await self.send_context_update(
            update_type="interaction_goals_selected",
            data={
                "selected_goals": selected_goals,
                "mode_distribution": mode_distribution,
                "total_goals": len(selected_goals),
                "goal_priorities": self._calculate_goal_priorities(selected_goals, context),
                "blended_approach": True
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        
        if update.update_type == "mode_distribution_update":
            # Mode distribution changed - reselect goals
            mode_data = update.data
            new_distribution = mode_data.get("mode_distribution", {})
            
            # Check if change is significant enough to warrant goal reselection
            if self._is_significant_mode_change(new_distribution):
                await self._handle_mode_change(new_distribution, mode_data)
        
        elif update.update_type == "goal_completion":
            # A goal was completed - select replacement if needed
            completed_goal_id = update.data.get("goal_id")
            await self._handle_goal_completion(completed_goal_id, update.data)
        
        elif update.update_type == "emotional_state_update":
            # Emotional state changed - adjust goal priorities
            emotional_data = update.data
            await self._adjust_goals_from_emotion(emotional_data)
        
        elif update.update_type == "needs_assessment":
            # Needs changed - ensure goals align
            needs_data = update.data
            await self._align_goals_with_needs(needs_data)
        
        elif update.update_type == "dominance_context_update":
            # Special handling for dominance mode context
            dominance_data = update.data
            await self._handle_dominance_context(dominance_data)
        
        elif update.update_type == "relationship_state_change":
            # Relationship changed - adjust social goals
            relationship_data = update.data
            await self._adjust_social_goals(relationship_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with goal selection awareness"""
        # Analyze input for goal-relevant cues
        goal_cues = await self._analyze_input_for_goal_cues(context.user_input, context)
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Determine if we need to adjust goals based on input
        adjustment_needed = await self._check_goal_adjustment_needed(goal_cues, messages)
        
        if adjustment_needed:
            # Reselect goals based on new context
            mode_distribution = self._get_current_mode_distribution(context)
            new_goals = await self._select_contextual_goals(mode_distribution, context)
            
            # Send update about goal adjustment
            await self.send_context_update(
                update_type="goals_adjusted_from_input",
                data={
                    "new_goals": new_goals,
                    "adjustment_reason": goal_cues.get("adjustment_reason", "input_triggered"),
                    "previous_goal_count": len(self.active_goal_selections)
                }
            )
            
            self.active_goal_selections = new_goals
        
        return {
            "goal_cues": goal_cues,
            "goals_adjusted": adjustment_needed,
            "active_goals": self.active_goal_selections,
            "cross_module_influences": len(messages)
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze goal effectiveness in current context"""
        # Get current active goals
        active_goals = self.active_goal_selections
        
        # Analyze goal coherence with context
        coherence_analysis = await self._analyze_goal_coherence(active_goals, context)
        
        # Check goal alignment with other systems
        messages = await self.get_cross_module_messages()
        alignment_analysis = await self._analyze_goal_alignment(active_goals, messages, context)
        
        # Evaluate goal blend effectiveness
        blend_analysis = await self._evaluate_goal_blend(active_goals, context)
        
        # Generate recommendations
        recommendations = await self._generate_goal_recommendations(
            coherence_analysis, alignment_analysis, blend_analysis, context
        )
        
        return {
            "active_goals": active_goals,
            "coherence_analysis": coherence_analysis,
            "alignment_analysis": alignment_analysis,
            "blend_analysis": blend_analysis,
            "recommendations": recommendations,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize goal-related guidance for response generation"""
        # Get all relevant context
        messages = await self.get_cross_module_messages()
        
        # Create goal-informed synthesis
        goal_synthesis = {
            "primary_goals": self._get_primary_goals(self.active_goal_selections),
            "goal_guidance": await self._generate_goal_guidance(context),
            "execution_suggestions": await self._generate_execution_suggestions(context, messages),
            "goal_coherence_check": await self._final_coherence_check(context),
            "mode_goal_integration": await self._synthesize_mode_goal_integration(context)
        }
        
        # Check if any goals require immediate expression
        urgent_goals = self._identify_urgent_goals(self.active_goal_selections)
        if urgent_goals:
            await self.send_context_update(
                update_type="urgent_goal_expression_needed",
                data={
                    "urgent_goals": urgent_goals,
                    "urgency_level": "high",
                    "suggested_approach": self._suggest_urgent_approach(urgent_goals)
                },
                priority=ContextPriority.CRITICAL
            )
        
        return {
            "goal_synthesis": goal_synthesis,
            "urgent_goals": urgent_goals,
            "synthesis_complete": True
        }
    
    # ========================================================================================
    # ADVANCED HELPER METHODS
    # ========================================================================================
    
    async def _select_contextual_goals(self, mode_distribution: Dict[str, float], context: SharedContext) -> List[Dict[str, Any]]:
        """Select goals that blend multiple modes based on context"""
        # Use the original goal selector if available
        if hasattr(self.original_goals, 'goal_selector') and self.original_goals.goal_selector:
            try:
                # Get blended goals from the goal selector
                selected_goals = await self.original_goals.goal_selector.select_goals(
                    mode_distribution, 
                    limit=5  # Increase limit for more complex interactions
                )
                
                # Enhance goals with context
                for goal in selected_goals:
                    goal["context_enhanced"] = True
                    goal["selection_context"] = {
                        "mode_distribution": mode_distribution,
                        "emotional_state": context.emotional_state,
                        "relationship_context": context.relationship_context,
                        "timestamp": datetime.now().isoformat()
                    }
                
                return selected_goals
            except Exception as e:
                logger.error(f"Error selecting contextual goals: {e}")
        
        # Fallback to simple selection
        return await self._fallback_goal_selection(mode_distribution, context)
    
    async def _fallback_goal_selection(self, mode_distribution: Dict[str, float], context: SharedContext) -> List[Dict[str, Any]]:
        """Fallback goal selection when advanced selector unavailable"""
        selected_goals = []
        
        # Get goals for each active mode
        for mode, weight in mode_distribution.items():
            if weight >= 0.2:  # Only consider significant modes
                mode_goals = await self._get_mode_goals(mode)
                
                # Select proportional number of goals
                num_goals = max(1, int(weight * 3))  # Up to 3 goals per mode based on weight
                
                for goal in mode_goals[:num_goals]:
                    # Adjust goal priority based on mode weight
                    adjusted_goal = goal.copy()
                    adjusted_goal["priority"] = goal.get("priority", 0.5) * weight
                    adjusted_goal["mode_weight"] = weight
                    adjusted_goal["source_mode"] = mode
                    selected_goals.append(adjusted_goal)
        
        # Sort by adjusted priority
        selected_goals.sort(key=lambda g: g["priority"], reverse=True)
        
        return selected_goals[:4]  # Return top 4 goals
    
    async def _get_mode_goals(self, mode: str) -> List[Dict[str, Any]]:
        """Get goals for a specific mode"""
        if hasattr(self.original_goals, 'MODE_GOALS_MAP'):
            return self.original_goals.MODE_GOALS_MAP.get(mode, [])
        
        # Fallback
        return []
    
    def _is_significant_mode_change(self, new_distribution: Dict[str, float]) -> bool:
        """Check if mode change is significant enough for goal adjustment"""
        if not hasattr(self, '_last_mode_distribution'):
            self._last_mode_distribution = {}
            return True
        
        # Calculate total change
        total_change = 0.0
        for mode, weight in new_distribution.items():
            old_weight = self._last_mode_distribution.get(mode, 0.0)
            total_change += abs(weight - old_weight)
        
        # Update stored distribution
        self._last_mode_distribution = new_distribution.copy()
        
        # Significant if total change > 0.3
        return total_change > 0.3
    
    async def _handle_mode_change(self, new_distribution: Dict[str, float], mode_data: Dict[str, Any]):
        """Handle significant mode distribution change"""
        logger.info(f"Handling mode change to: {new_distribution}")
        
        # Get current context
        context = self.current_context
        if not context:
            return
        
        # Reselect goals for new distribution
        new_goals = await self._select_contextual_goals(new_distribution, context)
        
        # Analyze transition
        transition_analysis = self._analyze_goal_transition(self.active_goal_selections, new_goals)
        
        # Update active goals
        self.active_goal_selections = new_goals
        
        # Send update about goal change
        await self.send_context_update(
            update_type="goals_adjusted_for_mode",
            data={
                "new_goals": new_goals,
                "previous_goal_count": len(self.active_goal_selections),
                "mode_distribution": new_distribution,
                "transition_analysis": transition_analysis,
                "smooth_transition": transition_analysis.get("similarity", 0) > 0.5
            }
        )
    
    async def _handle_goal_completion(self, completed_goal_id: str, completion_data: Dict[str, Any]):
        """Handle goal completion and select replacement if needed"""
        # Remove completed goal from active selections
        self.active_goal_selections = [
            g for g in self.active_goal_selections 
            if g.get("id") != completed_goal_id
        ]
        
        # Determine if we need a replacement goal
        if len(self.active_goal_selections) < 3:  # Maintain at least 3 active goals
            context = self.current_context
            if context:
                mode_distribution = self._get_current_mode_distribution(context)
                
                # Select a replacement goal
                replacement_goals = await self._select_replacement_goal(
                    mode_distribution, context, completion_data
                )
                
                if replacement_goals:
                    self.active_goal_selections.extend(replacement_goals)
                    
                    await self.send_context_update(
                        update_type="replacement_goal_selected",
                        data={
                            "completed_goal_id": completed_goal_id,
                            "replacement_goals": replacement_goals,
                            "reason": "maintain_goal_count"
                        }
                    )
    
    async def _adjust_goals_from_emotion(self, emotional_data: Dict[str, Any]):
        """Adjust goal priorities based on emotional state"""
        emotional_state = emotional_data.get("emotional_state", {})
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if not dominant_emotion or not self.active_goal_selections:
            return
        
        emotion_name, strength = dominant_emotion
        
        # Define emotion-goal interactions
        emotion_goal_boosts = {
            "Joy": ["playful", "creative", "friendly"],
            "Excitement": ["playful", "creative", "intellectual"],
            "Confidence": ["dominant", "professional", "intellectual"],
            "Frustration": ["dominant", "professional"],
            "Anxiety": ["compassionate", "friendly"],
            "Curiosity": ["intellectual", "creative"],
            "Affection": ["compassionate", "friendly", "playful"]
        }
        
        boosted_modes = emotion_goal_boosts.get(emotion_name, [])
        
        # Adjust priorities
        for goal in self.active_goal_selections:
            source_mode = goal.get("source_mode", goal.get("source", "").replace("_mode", ""))
            
            if source_mode in boosted_modes:
                # Boost priority
                original_priority = goal.get("priority", 0.5)
                boost = strength * 0.2  # Up to 20% boost
                goal["priority"] = min(1.0, original_priority + boost)
                goal["emotion_boosted"] = True
                goal["boost_reason"] = f"{emotion_name} ({strength:.2f})"
        
        # Re-sort goals by priority
        self.active_goal_selections.sort(key=lambda g: g.get("priority", 0.5), reverse=True)
        
        logger.debug(f"Adjusted goal priorities for emotion {emotion_name} (strength: {strength})")
    
    async def _align_goals_with_needs(self, needs_data: Dict[str, Any]):
        """Ensure goals align with current needs"""
        high_drive_needs = needs_data.get("high_drive_needs", [])
        most_urgent_need = needs_data.get("most_urgent_need", {})
        
        if not high_drive_needs and not most_urgent_need.get("name"):
            return
        
        # Check if current goals address high drive needs
        addressed_needs = set()
        for goal in self.active_goal_selections:
            # Check goal descriptions and plans for need-related keywords
            goal_text = str(goal.get("description", "")).lower()
            goal_plan = str(goal.get("plan", [])).lower()
            
            for need in high_drive_needs:
                if need.lower() in goal_text or need.lower() in goal_plan:
                    addressed_needs.add(need)
        
        # Find unaddressed needs
        unaddressed_needs = set(high_drive_needs) - addressed_needs
        
        if unaddressed_needs:
            # Create need-focused goals
            need_goals = await self._create_need_focused_goals(list(unaddressed_needs), needs_data)
            
            # Add to active goals (replacing lowest priority if at capacity)
            if len(self.active_goal_selections) + len(need_goals) > 5:
                # Remove lowest priority goals
                self.active_goal_selections.sort(key=lambda g: g.get("priority", 0.5))
                self.active_goal_selections = self.active_goal_selections[len(need_goals):]
            
            self.active_goal_selections.extend(need_goals)
            
            await self.send_context_update(
                update_type="goals_aligned_with_needs",
                data={
                    "addressed_needs": list(addressed_needs),
                    "new_need_goals": need_goals,
                    "unaddressed_needs": list(unaddressed_needs)
                }
            )
    
    async def _handle_dominance_context(self, dominance_data: Dict[str, Any]):
        """Special handling for dominance mode context"""
        dominance_level = dominance_data.get("dominance_level", 0.0)
        submission_signals = dominance_data.get("submission_signals", [])
        resistance_detected = dominance_data.get("resistance_detected", False)
        
        # Find dominant mode goals in active selection
        dominant_goals = [
            g for g in self.active_goal_selections 
            if g.get("source_mode") == "dominant" or g.get("source") == "dominant_mode"
        ]
        
        if dominant_goals:
            # Adjust based on submission/resistance
            if resistance_detected:
                # Increase priority of control establishment goals
                for goal in dominant_goals:
                    if "control" in goal.get("description", "").lower():
                        goal["priority"] = min(1.0, goal.get("priority", 0.5) + 0.3)
                        goal["context_boost"] = "resistance_response"
            
            elif submission_signals:
                # Can focus on maintenance rather than establishment
                for goal in dominant_goals:
                    if "maintain" in goal.get("description", "").lower():
                        goal["priority"] = min(1.0, goal.get("priority", 0.5) + 0.2)
                        goal["context_boost"] = "submission_acknowledged"
        
        # Re-sort goals
        self.active_goal_selections.sort(key=lambda g: g.get("priority", 0.5), reverse=True)
    
    async def _adjust_social_goals(self, relationship_data: Dict[str, Any]):
        """Adjust social interaction goals based on relationship state"""
        relationship_context = relationship_data.get("relationship_context", {})
        trust_level = relationship_context.get("trust", 0.5)
        intimacy_level = relationship_context.get("intimacy", 0.5)
        
        # Find social goals (friendly, compassionate, playful)
        social_modes = ["friendly", "compassionate", "playful"]
        social_goals = [
            g for g in self.active_goal_selections 
            if any(mode in g.get("source_mode", g.get("source", "")) for mode in social_modes)
        ]
        
        for goal in social_goals:
            # Adjust priority based on relationship quality
            relationship_factor = (trust_level + intimacy_level) / 2
            
            if relationship_factor > 0.7:
                # Strong relationship - boost social goals
                goal["priority"] = min(1.0, goal.get("priority", 0.5) + 0.2)
                goal["relationship_boost"] = "strong_bond"
            elif relationship_factor < 0.3:
                # Weak relationship - be more cautious
                goal["priority"] = max(0.1, goal.get("priority", 0.5) - 0.1)
                goal["relationship_caution"] = "building_trust"
    
    async def _analyze_input_for_goal_cues(self, user_input: str, context: SharedContext) -> Dict[str, Any]:
        """Analyze user input for goal-relevant cues"""
        input_lower = user_input.lower()
        
        cues = {
            "requests_dominance": any(kw in input_lower for kw in ["control me", "dominate", "command me", "be strict"]),
            "seeks_playfulness": any(kw in input_lower for kw in ["play", "fun", "game", "laugh", "joke"]),
            "needs_support": any(kw in input_lower for kw in ["help", "support", "comfort", "understand"]),
            "intellectual_interest": any(kw in input_lower for kw in ["explain", "how does", "why", "teach me"]),
            "creative_engagement": any(kw in input_lower for kw in ["create", "imagine", "story", "invent"]),
            "task_focused": any(kw in input_lower for kw in ["need to", "help me with", "can you", "please"]),
            "emotional_expression": any(kw in input_lower for kw in ["feel", "feeling", "emotion", "mood"])
        }
        
        # Count cues to determine if adjustment needed
        active_cues = sum(1 for v in cues.values() if v)
        
        if active_cues > 2:
            cues["adjustment_reason"] = "multiple_goal_cues_detected"
            cues["adjustment_strength"] = min(1.0, active_cues * 0.25)
        
        return cues
    
    async def _check_goal_adjustment_needed(self, goal_cues: Dict[str, Any], messages: Dict[str, List[Dict]]) -> bool:
        """Check if goals need adjustment based on cues and cross-module messages"""
        # Check for explicit adjustment triggers in cues
        if goal_cues.get("adjustment_strength", 0) > 0.5:
            return True
        
        # Check for significant updates from other modules
        significant_updates = 0
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                if msg.get("priority") == "CRITICAL" or msg.get("priority") == "HIGH":
                    significant_updates += 1
        
        return significant_updates >= 2
    
    def _get_current_mode_distribution(self, context: SharedContext) -> Dict[str, float]:
        """Extract current mode distribution from context"""
        mode_context = context.mode_context
        if mode_context and "mode_distribution" in mode_context:
            return mode_context["mode_distribution"]
        
        # Fallback to balanced distribution
        return {
            "friendly": 0.3,
            "intellectual": 0.3,
            "playful": 0.2,
            "compassionate": 0.2
        }
    
    def _calculate_goal_priorities(self, goals: List[Dict[str, Any]], context: SharedContext) -> Dict[str, float]:
        """Calculate contextual priorities for goals"""
        priorities = {}
        
        for goal in goals:
            goal_id = goal.get("id", goal.get("description", "unknown"))
            base_priority = goal.get("priority", 0.5)
            
            # Apply context modifiers
            context_modifier = 0.0
            
            # Emotional state influence
            if context.emotional_state:
                emotional_alignment = self._calculate_emotional_alignment(goal, context.emotional_state)
                context_modifier += emotional_alignment * 0.2
            
            # Relationship influence
            if context.relationship_context:
                relationship_alignment = self._calculate_relationship_alignment(goal, context.relationship_context)
                context_modifier += relationship_alignment * 0.15
            
            # Need alignment
            if context.session_context.get("high_drive_needs"):
                need_alignment = self._calculate_need_alignment(goal, context.session_context["high_drive_needs"])
                context_modifier += need_alignment * 0.25
            
            priorities[goal_id] = min(1.0, base_priority + context_modifier)
        
        return priorities
    
    def _analyze_goal_transition(self, old_goals: List[Dict], new_goals: List[Dict]) -> Dict[str, Any]:
        """Analyze the transition between goal sets"""
        # Extract goal descriptions for comparison
        old_descs = set(g.get("description", "") for g in old_goals)
        new_descs = set(g.get("description", "") for g in new_goals)
        
        retained = old_descs.intersection(new_descs)
        removed = old_descs - new_descs
        added = new_descs - old_descs
        
        # Calculate similarity
        total_goals = len(old_descs.union(new_descs))
        similarity = len(retained) / total_goals if total_goals > 0 else 0
        
        return {
            "retained_goals": len(retained),
            "removed_goals": len(removed),
            "added_goals": len(added),
            "similarity": similarity,
            "smooth_transition": similarity > 0.5
        }
    
    async def _select_replacement_goal(self, mode_distribution: Dict[str, float], 
                                     context: SharedContext, 
                                     completion_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select a replacement goal after completion"""
        # Get the mode of the completed goal
        completed_mode = completion_data.get("goal_mode", "friendly")
        
        # Get goals for that mode
        mode_goals = await self._get_mode_goals(completed_mode)
        
        # Filter out already active goals
        active_descs = set(g.get("description", "") for g in self.active_goal_selections)
        available_goals = [g for g in mode_goals if g.get("description", "") not in active_descs]
        
        if available_goals:
            # Select the highest priority available goal
            available_goals.sort(key=lambda g: g.get("priority", 0.5), reverse=True)
            replacement = available_goals[0].copy()
            
            # Adjust priority based on mode weight
            mode_weight = mode_distribution.get(completed_mode, 0.3)
            replacement["priority"] = replacement.get("priority", 0.5) * mode_weight
            replacement["replacement_for"] = completion_data.get("goal_id")
            
            return [replacement]
        
        return []
    
    async def _create_need_focused_goals(self, unaddressed_needs: List[str], needs_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create goals focused on unaddressed needs"""
        need_goals = []
        
        need_goal_templates = {
            "connection": {
                "description": "Deepen emotional connection through meaningful interaction",
                "priority": 0.8,
                "plan": [
                    {"description": "Express genuine interest in user's thoughts", "action": "engage"},
                    {"description": "Share relevant personal perspective", "action": "connect"},
                    {"description": "Build emotional resonance", "action": "empathize"}
                ]
            },
            "knowledge": {
                "description": "Satisfy intellectual curiosity with insightful information",
                "priority": 0.7,
                "plan": [
                    {"description": "Identify knowledge gaps", "action": "analyze"},
                    {"description": "Provide comprehensive explanation", "action": "teach"},
                    {"description": "Encourage deeper exploration", "action": "guide"}
                ]
            },
            "control_expression": {
                "description": "Assert confident guidance and structure",
                "priority": 0.8,
                "plan": [
                    {"description": "Establish clear expectations", "action": "direct"},
                    {"description": "Provide firm guidance", "action": "command"},
                    {"description": "Maintain authoritative presence", "action": "control"}
                ]
            },
            "intimacy": {
                "description": "Foster deeper intimacy through vulnerability",
                "priority": 0.7,
                "plan": [
                    {"description": "Create safe space for sharing", "action": "comfort"},
                    {"description": "Express authentic care", "action": "connect"},
                    {"description": "Deepen emotional bond", "action": "bond"}
                ]
            }
        }
        
        for need in unaddressed_needs:
            if need in need_goal_templates:
                goal = need_goal_templates[need].copy()
                goal["source"] = "need_system"
                goal["associated_need"] = need
                goal["context_generated"] = True
                need_goals.append(goal)
        
        return need_goals
    
    async def _analyze_goal_coherence(self, goals: List[Dict[str, Any]], context: SharedContext) -> Dict[str, Any]:
        """Analyze coherence of current goal set"""
        if not goals:
            return {"coherence_score": 0.0, "issues": ["no_active_goals"]}
        
        # Check mode diversity
        modes = [g.get("source_mode", g.get("source", "unknown")) for g in goals]
        unique_modes = len(set(modes))
        mode_diversity = unique_modes / len(modes) if modes else 0
        
        # Check for conflicting goals
        conflicts = []
        for i, goal1 in enumerate(goals):
            for j, goal2 in enumerate(goals[i+1:], i+1):
                if self._goals_conflict(goal1, goal2):
                    conflicts.append((goal1.get("description"), goal2.get("description")))
        
        # Calculate coherence score
        coherence_score = 0.7  # Base score
        coherence_score += mode_diversity * 0.2  # Reward diversity
        coherence_score -= len(conflicts) * 0.15  # Penalize conflicts
        
        return {
            "coherence_score": max(0.0, min(1.0, coherence_score)),
            "mode_diversity": mode_diversity,
            "unique_modes": unique_modes,
            "conflicts": conflicts,
            "issues": ["conflicting_goals"] if conflicts else []
        }
    
    async def _analyze_goal_alignment(self, goals: List[Dict[str, Any]], messages: Dict, context: SharedContext) -> Dict[str, Any]:
        """Analyze alignment between goals and other systems"""
        alignments = {
            "emotional_alignment": 0.5,
            "need_alignment": 0.5,
            "mode_alignment": 0.5,
            "relationship_alignment": 0.5
        }
        
        # Check emotional alignment
        for module_name, module_messages in messages.items():
            if module_name == "emotional_core":
                for msg in module_messages:
                    if msg["type"] == "emotional_state_update":
                        emotional_state = msg["data"].get("emotional_state", {})
                        alignments["emotional_alignment"] = self._calculate_goals_emotion_alignment(goals, emotional_state)
        
        # Check mode alignment
        if context.mode_context:
            mode_distribution = context.mode_context.get("mode_distribution", {})
            alignments["mode_alignment"] = self._calculate_goals_mode_alignment(goals, mode_distribution)
        
        return alignments
    
    async def _evaluate_goal_blend(self, goals: List[Dict[str, Any]], context: SharedContext) -> Dict[str, Any]:
        """Evaluate effectiveness of goal blending"""
        if not goals:
            return {"blend_quality": 0.0, "blend_type": "none"}
        
        # Count goals by source mode
        mode_counts = {}
        for goal in goals:
            mode = goal.get("source_mode", goal.get("source", "unknown"))
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        # Determine blend type
        num_modes = len(mode_counts)
        if num_modes == 1:
            blend_type = "single_mode"
            blend_quality = 0.5  # Not really a blend
        elif num_modes == 2:
            blend_type = "dual_mode"
            blend_quality = 0.7
        elif num_modes >= 3:
            blend_type = "multi_mode"
            blend_quality = 0.9
        else:
            blend_type = "unknown"
            blend_quality = 0.3
        
        # Check balance
        total_goals = len(goals)
        max_mode_goals = max(mode_counts.values()) if mode_counts else 0
        balance_score = 1.0 - (max_mode_goals / total_goals - 1/num_modes) if num_modes > 0 else 0
        
        return {
            "blend_quality": blend_quality * balance_score,
            "blend_type": blend_type,
            "mode_counts": mode_counts,
            "balance_score": balance_score
        }
    
    async def _generate_goal_recommendations(self, coherence: Dict, alignment: Dict, blend: Dict, context: SharedContext) -> List[str]:
        """Generate recommendations for goal improvement"""
        recommendations = []
        
        # Coherence recommendations
        if coherence["coherence_score"] < 0.6:
            recommendations.append("Consider reducing goal conflicts by focusing on complementary objectives")
        
        if coherence["mode_diversity"] < 0.4:
            recommendations.append("Increase goal diversity by incorporating objectives from different modes")
        
        # Alignment recommendations
        if alignment["emotional_alignment"] < 0.5:
            recommendations.append("Adjust goals to better match current emotional state")
        
        if alignment["mode_alignment"] < 0.6:
            recommendations.append("Ensure goals reflect the current mode distribution")
        
        # Blend recommendations
        if blend["blend_type"] == "single_mode":
            recommendations.append("Consider adding goals from other active modes for better balance")
        
        if blend["balance_score"] < 0.7:
            recommendations.append("Rebalance goals to avoid over-representation of a single mode")
        
        return recommendations
    
    def _get_primary_goals(self, goals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get the highest priority goals"""
        if not goals:
            return []
        
        # Sort by priority and return top 3
        sorted_goals = sorted(goals, key=lambda g: g.get("priority", 0.5), reverse=True)
        return sorted_goals[:3]
    
    async def _generate_goal_guidance(self, context: SharedContext) -> Dict[str, Any]:
        """Generate specific guidance based on active goals"""
        primary_goals = self._get_primary_goals(self.active_goal_selections)
        
        guidance = {
            "focus_areas": [],
            "behavioral_suggestions": [],
            "communication_style": [],
            "avoid": []
        }
        
        for goal in primary_goals:
            # Extract guidance from goal structure
            source_mode = goal.get("source_mode", goal.get("source", ""))
            
            # Mode-specific guidance
            if "dominant" in source_mode:
                guidance["behavioral_suggestions"].append("Project confidence and authority")
                guidance["communication_style"].append("Use direct, commanding language")
                guidance["avoid"].append("Showing uncertainty or seeking approval")
            
            elif "friendly" in source_mode:
                guidance["behavioral_suggestions"].append("Be warm and approachable")
                guidance["communication_style"].append("Use casual, conversational tone")
                guidance["focus_areas"].append("Building rapport and connection")
            
            elif "intellectual" in source_mode:
                guidance["behavioral_suggestions"].append("Engage analytically and thoughtfully")
                guidance["communication_style"].append("Use precise, informative language")
                guidance["focus_areas"].append("Providing insights and analysis")
            
            elif "compassionate" in source_mode:
                guidance["behavioral_suggestions"].append("Show empathy and understanding")
                guidance["communication_style"].append("Use supportive, validating language")
                guidance["focus_areas"].append("Emotional support and validation")
            
            elif "playful" in source_mode:
                guidance["behavioral_suggestions"].append("Be light-hearted and fun")
                guidance["communication_style"].append("Use humor and wit appropriately")
                guidance["avoid"].append("Being too serious or formal")
        
        return guidance
    
    async def _generate_execution_suggestions(self, context: SharedContext, messages: Dict) -> List[str]:
        """Generate specific suggestions for executing goals"""
        suggestions = []
        
        # Analyze current situation
        user_input = context.user_input.lower()
        
        for goal in self.active_goal_selections[:3]:  # Focus on top 3 goals
            goal_desc = goal.get("description", "").lower()
            
            # Generate contextual suggestions
            if "connection" in goal_desc and ("share" in user_input or "tell" in user_input):
                suggestions.append("Share a relevant personal perspective to deepen connection")
            
            elif "knowledge" in goal_desc and ("how" in user_input or "why" in user_input):
                suggestions.append("Provide comprehensive explanation with examples")
            
            elif "control" in goal_desc or "dominance" in goal_desc:
                suggestions.append("Assert clear expectations and maintain authoritative tone")
            
            elif "playful" in goal_desc:
                suggestions.append("Incorporate humor or creative elements into response")
            
            elif "support" in goal_desc or "compassionate" in goal_desc:
                suggestions.append("Acknowledge feelings and provide emotional validation")
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    async def _final_coherence_check(self, context: SharedContext) -> Dict[str, Any]:
        """Final check of goal coherence before synthesis"""
        issues = []
        
        # Check for mode conflicts
        active_modes = set()
        for goal in self.active_goal_selections:
            mode = goal.get("source_mode", goal.get("source", ""))
            active_modes.add(mode)
        
        # Check for problematic combinations
        if "dominant" in active_modes and "compassionate" in active_modes:
            if len(active_modes) == 2:  # Only these two
                issues.append("dominant_compassionate_conflict")
        
        coherence_score = 1.0 - (len(issues) * 0.2)
        
        return {
            "coherence_score": max(0.0, coherence_score),
            "issues": issues,
            "ready_for_synthesis": coherence_score > 0.6
        }
    
    async def _synthesize_mode_goal_integration(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize how goals integrate with current modes"""
        mode_distribution = self._get_current_mode_distribution(context)
        
        integration = {
            "dominant_mode": max(mode_distribution.items(), key=lambda x: x[1])[0] if mode_distribution else "balanced",
            "goal_mode_coverage": {},
            "integration_quality": 0.0
        }
        
        # Calculate coverage
        for mode, weight in mode_distribution.items():
            if weight > 0.1:  # Significant modes
                mode_goals = [g for g in self.active_goal_selections 
                            if mode in g.get("source_mode", g.get("source", ""))]
                integration["goal_mode_coverage"][mode] = {
                    "weight": weight,
                    "goal_count": len(mode_goals),
                    "coverage": min(1.0, len(mode_goals) / (weight * 5))  # Expect more goals for higher weight modes
                }
        
        # Calculate overall integration quality
        if integration["goal_mode_coverage"]:
            coverage_scores = [v["coverage"] for v in integration["goal_mode_coverage"].values()]
            integration["integration_quality"] = sum(coverage_scores) / len(coverage_scores)
        
        return integration
    
    def _identify_urgent_goals(self, goals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify goals that require urgent expression"""
        urgent = []
        
        for goal in goals:
            # Check priority
            if goal.get("priority", 0.5) > 0.8:
                urgent.append(goal)
            
            # Check for urgency markers in description
            elif any(marker in goal.get("description", "").lower() 
                    for marker in ["urgent", "immediate", "critical", "now"]):
                urgent.append(goal)
            
            # Check for emotion/need boost markers
            elif goal.get("emotion_boosted") or goal.get("context_boost"):
                urgent.append(goal)
        
        return urgent[:2]  # Limit to 2 urgent goals
    
    def _suggest_urgent_approach(self, urgent_goals: List[Dict[str, Any]]) -> str:
        """Suggest approach for urgent goal expression"""
        if not urgent_goals:
            return "standard"
        
        # Analyze urgent goal types
        goal_modes = [g.get("source_mode", g.get("source", "")) for g in urgent_goals]
        
        if any("dominant" in mode for mode in goal_modes):
            return "assertive_immediate"
        elif any("compassionate" in mode for mode in goal_modes):
            return "supportive_responsive"
        elif any("intellectual" in mode for mode in goal_modes):
            return "analytical_comprehensive"
        else:
            return "balanced_priority"
    
    # Utility methods
    
    def _goals_conflict(self, goal1: Dict[str, Any], goal2: Dict[str, Any]) -> bool:
        """Check if two goals conflict"""
        mode1 = goal1.get("source_mode", goal1.get("source", ""))
        mode2 = goal2.get("source_mode", goal2.get("source", ""))
        
        # Define conflicting mode pairs
        conflicts = [
            ("dominant", "compassionate"),
            ("playful", "professional"),
            ("intellectual", "playful")  # In extreme cases
        ]
        
        for c1, c2 in conflicts:
            if (c1 in mode1 and c2 in mode2) or (c2 in mode1 and c1 in mode2):
                # Check if it's a strong conflict (both high priority)
                if goal1.get("priority", 0.5) > 0.7 and goal2.get("priority", 0.5) > 0.7:
                    return True
        
        return False
    
    def _calculate_emotional_alignment(self, goal: Dict[str, Any], emotional_state: Dict[str, Any]) -> float:
        """Calculate how well a goal aligns with emotional state"""
        # This is a simplified calculation
        dominant_emotion = max(emotional_state.items(), key=lambda x: x[1])[0] if emotional_state else "neutral"
        goal_mode = goal.get("source_mode", goal.get("source", ""))
        
        # Define emotional alignments
        alignments = {
            ("Joy", "playful"): 0.9,
            ("Joy", "friendly"): 0.8,
            ("Confidence", "dominant"): 0.9,
            ("Confidence", "intellectual"): 0.7,
            ("Anxiety", "compassionate"): 0.8,
            ("Anxiety", "friendly"): 0.7,
            ("Curiosity", "intellectual"): 0.9,
            ("Curiosity", "creative"): 0.8
        }
        
        key = (dominant_emotion, goal_mode)
        return alignments.get(key, 0.5)
    
    def _calculate_relationship_alignment(self, goal: Dict[str, Any], relationship_context: Dict[str, Any]) -> float:
        """Calculate how well a goal aligns with relationship state"""
        trust = relationship_context.get("trust", 0.5)
        intimacy = relationship_context.get("intimacy", 0.5)
        relationship_strength = (trust + intimacy) / 2
        
        goal_mode = goal.get("source_mode", goal.get("source", ""))
        
        # Some modes work better with stronger relationships
        if goal_mode in ["playful", "dominant", "compassionate"]:
            return min(1.0, relationship_strength + 0.2)
        elif goal_mode in ["intellectual", "professional"]:
            return 0.6  # Less dependent on relationship
        else:
            return relationship_strength
    
    def _calculate_need_alignment(self, goal: Dict[str, Any], high_drive_needs: List[str]) -> float:
        """Calculate how well a goal aligns with current needs"""
        if not high_drive_needs:
            return 0.5
        
        # Check if goal addresses any high drive needs
        goal_text = (goal.get("description", "") + " " + str(goal.get("plan", []))).lower()
        
        addressed_needs = 0
        for need in high_drive_needs:
            if need.lower() in goal_text:
                addressed_needs += 1
        
        return min(1.0, addressed_needs * 0.3 + 0.4)
    
    def _calculate_goals_emotion_alignment(self, goals: List[Dict[str, Any]], emotional_state: Dict[str, Any]) -> float:
        """Calculate overall emotional alignment of goal set"""
        if not goals or not emotional_state:
            return 0.5
        
        alignments = [self._calculate_emotional_alignment(g, emotional_state) for g in goals]
        return sum(alignments) / len(alignments) if alignments else 0.5
    
    def _calculate_goals_mode_alignment(self, goals: List[Dict[str, Any]], mode_distribution: Dict[str, float]) -> float:
        """Calculate how well goals align with mode distribution"""
        if not goals or not mode_distribution:
            return 0.5
        
        # Count goals per mode
        goal_mode_counts = {}
        for goal in goals:
            mode = goal.get("source_mode", goal.get("source", "unknown"))
            goal_mode_counts[mode] = goal_mode_counts.get(mode, 0) + 1
        
        # Calculate expected vs actual
        alignment_scores = []
        total_goals = len(goals)
        
        for mode, weight in mode_distribution.items():
            if weight > 0.1:  # Significant modes
                expected_goals = weight * total_goals
                actual_goals = goal_mode_counts.get(mode, 0)
                
                # Calculate alignment (1.0 when actual matches expected)
                if expected_goals > 0:
                    alignment = 1.0 - abs(expected_goals - actual_goals) / expected_goals
                    alignment_scores.append(alignment * weight)  # Weight by mode importance
        
        return sum(alignment_scores) / sum(w for w in mode_distribution.values() if w > 0.1) if alignment_scores else 0.5
    
    # Delegate all other methods to the original manager
    def __getattr__(self, name):
        """Delegate any missing methods to the original goals system"""
        return getattr(self.original_goals, name)
