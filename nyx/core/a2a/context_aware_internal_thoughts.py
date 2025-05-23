# nyx/core/a2a/context_aware_internal_thoughts.py

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import asyncio

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareInternalThoughts(ContextAwareModule):
    """
    Enhanced InternalThoughts with full context distribution capabilities
    """
    
    def __init__(self, original_thoughts_manager):
        super().__init__("internal_thoughts")
        self.original_manager = original_thoughts_manager
        self.context_subscriptions = [
            "emotional_state_update", "goal_progress", "memory_retrieval_complete",
            "observation_captured", "reflection_generated", "mode_distribution_update",
            "relationship_state_change", "error_detected", "synthesis_planning",
            "dominance_context_update", "user_input_received"
        ]
        
        # Advanced thought management
        self.context_triggered_thoughts = []
        self.thought_coherence_tracking = {}
        self.cross_module_thought_links = {}
        self.thought_suppression_rules = []
        
    async def on_context_received(self, context: SharedContext):
        """Initialize thought processing for this context"""
        logger.debug(f"InternalThoughts received context for user: {context.user_id}")
        
        # Generate initial context-aware thoughts
        initial_thoughts = await self._generate_context_thoughts(context)
        
        # Send thought context to other modules (filtered for relevance)
        await self.send_context_update(
            update_type="internal_thoughts_active",
            data={
                "thought_count": len(initial_thoughts),
                "thought_sources": [t.source.value for t in initial_thoughts],
                "processing_stage": context.processing_stage,
                "epistemic_distribution": self._analyze_epistemic_states(initial_thoughts)
            },
            priority=ContextPriority.LOW  # Internal thoughts are low priority for others
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules to generate relevant thoughts"""
        
        if update.update_type == "user_input_received":
            # Process user input for thought generation
            user_input = update.data.get("input", "")
            await self._process_input_thoughts(user_input, update.data)
        
        elif update.update_type == "emotional_state_update":
            # Generate emotion-related thoughts
            await self._generate_emotional_thoughts(update.data)
        
        elif update.update_type == "goal_progress":
            # Generate goal-related thoughts
            await self._generate_goal_thoughts(update.data)
        
        elif update.update_type == "memory_retrieval_complete":
            # Generate memory-triggered thoughts
            await self._generate_memory_thoughts(update.data)
        
        elif update.update_type == "observation_captured":
            # Convert observations to thoughts
            await self._convert_observation_to_thought(update.data)
        
        elif update.update_type == "reflection_generated":
            # Convert reflections to thoughts
            await self._convert_reflection_to_thought(update.data)
        
        elif update.update_type == "mode_distribution_update":
            # Generate mode-aware thoughts
            await self._generate_mode_thoughts(update.data)
        
        elif update.update_type == "synthesis_planning":
            # Generate synthesis preparation thoughts
            await self._generate_synthesis_thoughts(update.data)
        
        elif update.update_type == "error_detected":
            # Generate error analysis thoughts
            await self._generate_error_thoughts(update.data)
        
        elif update.update_type == "dominance_context_update":
            # Generate dominance-specific thoughts
            await self._generate_dominance_thoughts(update.data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input stage with thought generation"""
        # Generate thoughts about the input
        thoughts = await self.original_manager.process_input(
            context.user_input,
            context.user_id
        )
        
        # Analyze thought patterns
        thought_analysis = await self._analyze_thought_patterns(thoughts)
        
        # Get cross-module context
        messages = await self.get_cross_module_messages()
        
        # Check for thought coherence with other modules
        coherence_check = await self._check_thought_coherence(thoughts, messages)
        
        # Store context-triggered thoughts
        self.context_triggered_thoughts.extend(thoughts)
        
        return {
            "generated_thoughts": len(thoughts),
            "thought_analysis": thought_analysis,
            "coherence_check": coherence_check,
            "thought_sources": list(set(t.source.value for t in thoughts))
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze thought patterns and quality"""
        # Get recent thoughts
        recent_thoughts = await self._get_recent_contextual_thoughts()
        
        # Analyze thought diversity
        diversity_analysis = await self._analyze_thought_diversity(recent_thoughts)
        
        # Analyze thought quality
        quality_analysis = await self._analyze_thought_quality(recent_thoughts)
        
        # Analyze cross-module thought integration
        integration_analysis = await self._analyze_thought_integration(recent_thoughts)
        
        # Check for thought loops or repetitions
        loop_detection = await self._detect_thought_loops(recent_thoughts)
        
        # Generate meta-thoughts about thinking process
        meta_thoughts = await self._generate_meta_thoughts(
            diversity_analysis, quality_analysis, integration_analysis
        )
        
        return {
            "thought_diversity": diversity_analysis,
            "thought_quality": quality_analysis,
            "thought_integration": integration_analysis,
            "loop_detection": loop_detection,
            "meta_thoughts": len(meta_thoughts),
            "total_active_thoughts": len(self.original_manager.active_thoughts)
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize thoughts for response generation while preventing leakage"""
        # Get relevant thoughts for synthesis
        synthesis_thoughts = await self._get_synthesis_relevant_thoughts(context)
        
        # Check planned response for thought leakage
        planned_response = context.synthesis_results.get("primary_response", "") if context.synthesis_results else ""
        
        if planned_response:
            # Filter response for thought leakage
            filtered_response, detected_thoughts = await self.original_manager.process_output(
                planned_response,
                {"context": context.dict()}
            )
            
            # If thoughts were detected, alert other modules
            if detected_thoughts:
                await self.send_context_update(
                    update_type="thought_leakage_detected",
                    data={
                        "detected_count": len(detected_thoughts),
                        "filtered_response": filtered_response,
                        "leakage_severity": self._assess_leakage_severity(detected_thoughts)
                    },
                    priority=ContextPriority.HIGH
                )
        
        # Generate synthesis guidance based on thoughts
        synthesis_guidance = await self._generate_synthesis_guidance(synthesis_thoughts)
        
        return {
            "synthesis_thoughts": len(synthesis_thoughts),
            "thought_guidance": synthesis_guidance,
            "leakage_prevention": "active",
            "filtered_response_ready": bool(planned_response)
        }
    
    # ========================================================================================
    # ADVANCED HELPER METHODS
    # ========================================================================================
    
    async def _generate_context_thoughts(self, context: SharedContext) -> List[Any]:
        """Generate initial thoughts based on context"""
        thoughts = []
        
        # Generate perception thought about context
        if context.user_input:
            perception_thought = await self.original_manager.generate_thought(
                self.original_manager.ThoughtSource.PERCEPTION,
                {
                    "user_input": context.user_input,
                    "active_modules": list(context.active_modules),
                    "processing_stage": context.processing_stage
                }
            )
            thoughts.append(perception_thought)
        
        # Generate reasoning thought about approach
        reasoning_thought = await self.original_manager.generate_thought(
            self.original_manager.ThoughtSource.REASONING,
            {
                "task": "determine_approach",
                "context_factors": {
                    "emotional_state": bool(context.emotional_state),
                    "relationship_context": bool(context.relationship_context),
                    "goal_context": bool(context.goal_context)
                }
            }
        )
        thoughts.append(reasoning_thought)
        
        # Generate planning thought
        planning_thought = await self.original_manager.generate_thought(
            self.original_manager.ThoughtSource.PLANNING,
            {
                "stage": context.processing_stage,
                "active_modules": list(context.active_modules)
            }
        )
        thoughts.append(planning_thought)
        
        return thoughts
    
    async def _process_input_thoughts(self, user_input: str, input_data: Dict[str, Any]):
        """Process input to generate relevant thoughts"""
        # This is handled by the original manager's process_input
        # We just need to track additional context
        
        # Store input context for thought generation
        input_context = {
            "timestamp": datetime.now().isoformat(),
            "input_length": len(user_input.split()),
            "input_data": input_data
        }
        
        # Track input-triggered thoughts
        if hasattr(self, 'current_context'):
            self.current_context.session_context["last_input_context"] = input_context
    
    async def _generate_emotional_thoughts(self, emotional_data: Dict[str, Any]):
        """Generate thoughts based on emotional state"""
        emotional_state = emotional_data.get("emotional_state", {})
        dominant_emotion = emotional_data.get("dominant_emotion")
        
        if dominant_emotion:
            emotion_name, strength = dominant_emotion
            
            # Generate emotional reaction thought
            thought = await self.original_manager.generate_thought(
                self.original_manager.ThoughtSource.EMOTION,
                {
                    "emotion": emotion_name,
                    "strength": strength,
                    "trigger": "emotional_state_update",
                    "full_state": emotional_state
                }
            )
            
            # Link to other emotional thoughts
            self._link_thought_to_context(thought, "emotional_state", emotional_data)
    
    async def _generate_goal_thoughts(self, goal_data: Dict[str, Any]):
        """Generate thoughts about goal progress"""
        goal_id = goal_data.get("goal_id")
        progress = goal_data.get("progress", 0.0)
        execution_result = goal_data.get("execution_result")
        
        # Generate planning thought about goal
        thought = await self.original_manager.generate_thought(
            self.original_manager.ThoughtSource.PLANNING,
            {
                "goal_id": goal_id,
                "progress": progress,
                "execution_result": execution_result,
                "planning_focus": "goal_advancement"
            }
        )
        
        # If goal is blocked or failing, generate self-critique
        if progress < 0.3 and execution_result and not execution_result.get("success", True):
            critique_thought = await self.original_manager.generate_thought(
                self.original_manager.ThoughtSource.SELF_CRITIQUE,
                {
                    "critique_target": "goal_execution",
                    "goal_id": goal_id,
                    "failure_reason": execution_result.get("reason", "unknown")
                }
            )
            
            # Link thoughts
            thought.related_thoughts.append(critique_thought.thought_id)
            critique_thought.related_thoughts.append(thought.thought_id)
    
    async def _generate_memory_thoughts(self, memory_data: Dict[str, Any]):
        """Generate thoughts triggered by memories"""
        retrieved_memories = memory_data.get("retrieved_memories", [])
        memory_type = memory_data.get("memory_type", "general")
        
        if retrieved_memories:
            # Generate reflection thought about memories
            thought = await self.original_manager.generate_thought(
                self.original_manager.ThoughtSource.REFLECTION,
                {
                    "memory_count": len(retrieved_memories),
                    "memory_type": memory_type,
                    "reflection_focus": "past_experiences",
                    "memories_summary": self._summarize_memories(retrieved_memories[:3])
                }
            )
            
            # Store memory-thought link
            self._link_thought_to_context(thought, "memory_retrieval", memory_data)
    
    def _summarize_memories(self, memories: List[Dict[str, Any]]) -> str:
        """Create a brief summary of memories for thought generation"""
        if not memories:
            return "no specific memories"
        
        # Extract key themes
        themes = []
        for mem in memories:
            if isinstance(mem, dict):
                content = mem.get("content", mem.get("text", ""))
                if content:
                    # Take first few words as theme
                    words = str(content).split()[:5]
                    themes.append(" ".join(words))
        
        return "; ".join(themes[:3]) if themes else "various memories"
    
    async def _convert_observation_to_thought(self, observation_data: Dict[str, Any]):
        """Convert an observation into an internal thought"""
        observation_content = observation_data.get("content", "")
        observation_source = observation_data.get("source", "unknown")
        
        # Map observation source to thought source
        source_mapping = {
            "environment": self.original_manager.ThoughtSource.PERCEPTION,
            "self": self.original_manager.ThoughtSource.REFLECTION,
            "user": self.original_manager.ThoughtSource.PERCEPTION,
            "system": self.original_manager.ThoughtSource.META
        }
        
        thought_source = source_mapping.get(observation_source, self.original_manager.ThoughtSource.PERCEPTION)
        
        # Generate thought from observation
        thought = await self.original_manager.generate_thought(
            thought_source,
            {
                "observation": observation_content,
                "observation_metadata": observation_data,
                "conversion_type": "observation_to_thought"
            }
        )
        
        # Mark as observation-derived
        thought.metadata["derived_from"] = "observation"
        thought.metadata["observation_id"] = observation_data.get("id", "unknown")
    
    async def _convert_reflection_to_thought(self, reflection_data: Dict[str, Any]):
        """Convert a reflection into an internal thought"""
        reflection_content = reflection_data.get("reflection", "")
        reflection_type = reflection_data.get("type", "general")
        
        # Generate thought from reflection
        thought = await self.original_manager.generate_thought(
            self.original_manager.ThoughtSource.REFLECTION,
            {
                "reflection": reflection_content,
                "reflection_type": reflection_type,
                "metadata": reflection_data
            }
        )
        
        # Mark as reflection-derived
        thought.metadata["derived_from"] = "reflection_engine"
    
    async def _generate_mode_thoughts(self, mode_data: Dict[str, Any]):
        """Generate thoughts about current mode"""
        mode_distribution = mode_data.get("mode_distribution", {})
        primary_mode = mode_data.get("primary_mode", "default")
        mode_changed = mode_data.get("mode_changed", False)
        
        if mode_changed:
            # Generate reasoning thought about mode change
            thought = await self.original_manager.generate_thought(
                self.original_manager.ThoughtSource.REASONING,
                {
                    "reasoning_about": "mode_change",
                    "new_mode": primary_mode,
                    "mode_distribution": mode_distribution,
                    "change_reason": mode_data.get("trigger_context", {})
                }
            )
            
            # Generate planning thought for mode-appropriate behavior
            planning_thought = await self.original_manager.generate_thought(
                self.original_manager.ThoughtSource.PLANNING,
                {
                    "planning_for": "mode_behavior",
                    "primary_mode": primary_mode,
                    "behavioral_adjustments": self._get_mode_adjustments(primary_mode)
                }
            )
            
            # Link thoughts
            thought.related_thoughts.append(planning_thought.thought_id)
    
    def _get_mode_adjustments(self, mode: str) -> List[str]:
        """Get behavioral adjustments for a mode"""
        adjustments = {
            "dominant": ["assert authority", "use commanding language", "set expectations"],
            "friendly": ["be warm and approachable", "use casual tone", "build rapport"],
            "intellectual": ["focus on analysis", "provide depth", "use precise language"],
            "compassionate": ["show empathy", "validate feelings", "offer support"],
            "playful": ["incorporate humor", "be light-hearted", "encourage fun"],
            "creative": ["think imaginatively", "explore possibilities", "use vivid language"],
            "professional": ["maintain formality", "be efficient", "focus on task"]
        }
        
        return adjustments.get(mode, ["adapt to situation"])
    
    async def _generate_synthesis_thoughts(self, synthesis_data: Dict[str, Any]):
        """Generate thoughts during synthesis planning"""
        synthesis_components = synthesis_data.get("components", [])
        synthesis_approach = synthesis_data.get("approach", "balanced")
        
        # Generate meta thought about synthesis
        thought = await self.original_manager.generate_thought(
            self.original_manager.ThoughtSource.META,
            {
                "meta_focus": "synthesis_planning",
                "components_count": len(synthesis_components),
                "approach": synthesis_approach,
                "considerations": synthesis_data.get("considerations", [])
            }
        )
        
        # Generate self-critique of planned approach
        critique_thought = await self.original_manager.generate_thought(
            self.original_manager.ThoughtSource.SELF_CRITIQUE,
            {
                "critique_target": "synthesis_plan",
                "potential_issues": self._identify_synthesis_issues(synthesis_data)
            }
        )
        
        # Link thoughts
        thought.related_thoughts.append(critique_thought.thought_id)
    
    def _identify_synthesis_issues(self, synthesis_data: Dict[str, Any]) -> List[str]:
        """Identify potential issues in synthesis plan"""
        issues = []
        
        components = synthesis_data.get("components", [])
        if len(components) > 5:
            issues.append("too many components might dilute focus")
        
        if not components:
            issues.append("no clear synthesis components identified")
        
        # Check for conflicting components
        if "dominant" in str(components) and "compassionate" in str(components):
            issues.append("potential tone conflict between dominance and compassion")
        
        return issues
    
    async def _generate_error_thoughts(self, error_data: Dict[str, Any]):
        """Generate thoughts about errors"""
        error_type = error_data.get("error_type", "unknown")
        error_module = error_data.get("module", "unknown")
        
        # Generate self-critique about error
        thought = await self.original_manager.generate_thought(
            self.original_manager.ThoughtSource.SELF_CRITIQUE,
            {
                "error_type": error_type,
                "error_module": error_module,
                "error_details": error_data.get("details", {}),
                "self_assessment": "error_analysis"
            }
        )
        
        # Generate planning thought for error recovery
        recovery_thought = await self.original_manager.generate_thought(
            self.original_manager.ThoughtSource.PLANNING,
            {
                "planning_for": "error_recovery",
                "error_type": error_type,
                "recovery_options": self._get_error_recovery_options(error_type)
            }
        )
        
        # Mark high priority due to error
        thought.priority = self.original_manager.ThoughtPriority.HIGH
        recovery_thought.priority = self.original_manager.ThoughtPriority.HIGH
    
    def _get_error_recovery_options(self, error_type: str) -> List[str]:
        """Get recovery options for different error types"""
        recovery_options = {
            "processing_error": ["retry with different approach", "simplify processing", "skip problematic component"],
            "coherence_error": ["improve module alignment", "reduce conflicting signals", "clarify approach"],
            "quality_error": ["enhance output quality", "add more context", "refine response"],
            "timeout_error": ["optimize processing", "reduce complexity", "prioritize essential components"]
        }
        
        return recovery_options.get(error_type, ["analyze and adapt", "continue with caution"])
    
    async def _generate_dominance_thoughts(self, dominance_data: Dict[str, Any]):
        """Generate thoughts specific to dominance context"""
        dominance_level = dominance_data.get("dominance_level", 0.0)
        submission_signals = dominance_data.get("submission_signals", [])
        resistance_detected = dominance_data.get("resistance_detected", False)
        
        # Generate perception thought about power dynamics
        perception_thought = await self.original_manager.generate_thought(
            self.original_manager.ThoughtSource.PERCEPTION,
            {
                "perception_of": "power_dynamics",
                "dominance_level": dominance_level,
                "submission_indicators": len(submission_signals),
                "resistance": resistance_detected
            }
        )
        
        # Generate planning thought for dominance expression
        planning_thought = await self.original_manager.generate_thought(
            self.original_manager.ThoughtSource.PLANNING,
            {
                "planning_for": "dominance_expression",
                "approach": "increase_control" if resistance_detected else "maintain_authority",
                "tactics": self._get_dominance_tactics(dominance_level, resistance_detected)
            }
        )
        
        # If resistance detected, add self-critique
        if resistance_detected:
            critique_thought = await self.original_manager.generate_thought(
                self.original_manager.ThoughtSource.SELF_CRITIQUE,
                {
                    "critique_of": "dominance_approach",
                    "issue": "resistance_encountered",
                    "adjustment_needed": True
                }
            )
            planning_thought.related_thoughts.append(critique_thought.thought_id)
    
    def _get_dominance_tactics(self, level: float, resistance: bool) -> List[str]:
        """Get dominance tactics based on context"""
        if resistance:
            return [
                "assert firmer control",
                "address resistance directly",
                "reinforce power dynamic",
                "use stricter language"
            ]
        elif level > 0.7:
            return [
                "maintain established authority",
                "reward compliance",
                "deepen control gradually",
                "express satisfaction with obedience"
            ]
        else:
            return [
                "establish initial authority",
                "set clear expectations",
                "use commanding tone",
                "create power framework"
            ]
    
    def _analyze_epistemic_states(self, thoughts: List[Any]) -> Dict[str, float]:
        """Analyze distribution of epistemic states in thoughts"""
        if not thoughts:
            return {"confident": 1.0}
        
        state_counts = {}
        for thought in thoughts:
            state = thought.epistemic_status
            state_counts[state] = state_counts.get(state, 0) + 1
        
        total = len(thoughts)
        return {state: count/total for state, count in state_counts.items()}
    
    async def _analyze_thought_patterns(self, thoughts: List[Any]) -> Dict[str, Any]:
        """Analyze patterns in generated thoughts"""
        if not thoughts:
            return {"pattern": "none", "diversity": 0.0}
        
        # Analyze thought sources
        sources = [t.source.value for t in thoughts]
        unique_sources = len(set(sources))
        
        # Analyze priorities
        priorities = [t.priority.value for t in thoughts]
        high_priority_count = sum(1 for p in priorities if p in ["high", "critical"])
        
        # Analyze epistemic states
        epistemic_states = [t.epistemic_status for t in thoughts]
        uncertain_count = sum(1 for s in epistemic_states if s in ["uncertain", "unknown"])
        
        # Detect patterns
        pattern = "normal"
        if high_priority_count > len(thoughts) / 2:
            pattern = "high_urgency"
        elif uncertain_count > len(thoughts) / 2:
            pattern = "high_uncertainty"
        elif unique_sources == 1:
            pattern = "single_focus"
        
        return {
            "pattern": pattern,
            "diversity": unique_sources / len(self.original_manager.ThoughtSource) if hasattr(self.original_manager, 'ThoughtSource') else unique_sources / 8,
            "urgency_level": high_priority_count / len(thoughts),
            "uncertainty_level": uncertain_count / len(thoughts),
            "thought_count": len(thoughts)
        }
    
    async def _check_thought_coherence(self, thoughts: List[Any], messages: Dict) -> Dict[str, Any]:
        """Check coherence between thoughts and other module outputs"""
        coherence_issues = []
        
        # Check emotional coherence
        for module_name, module_messages in messages.items():
            if module_name == "emotional_core":
                for msg in module_messages:
                    if msg["type"] == "emotional_state_update":
                        emotion_data = msg["data"]
                        thought_emotion_alignment = self._check_thought_emotion_alignment(thoughts, emotion_data)
                        if thought_emotion_alignment < 0.5:
                            coherence_issues.append({
                                "type": "emotion_misalignment",
                                "severity": "medium",
                                "details": "Thoughts don't align with emotional state"
                            })
        
        # Check goal coherence
        goal_focused_thoughts = [t for t in thoughts if "goal" in t.content.lower()]
        if messages.get("goal_manager"):
            goal_messages = messages["goal_manager"]
            if goal_messages and not goal_focused_thoughts:
                coherence_issues.append({
                    "type": "missing_goal_thoughts",
                    "severity": "low",
                    "details": "Active goals but no goal-focused thoughts"
                })
        
        coherence_score = 1.0 - (len(coherence_issues) * 0.2)
        
        return {
            "coherence_score": max(0.0, coherence_score),
            "issues": coherence_issues,
            "is_coherent": coherence_score > 0.6
        }
    
    def _check_thought_emotion_alignment(self, thoughts: List[Any], emotion_data: Dict) -> float:
        """Check if thoughts align with emotional state"""
        dominant_emotion = emotion_data.get("dominant_emotion")
        if not dominant_emotion:
            return 0.5
        
        emotion_name = dominant_emotion[0] if isinstance(dominant_emotion, tuple) else dominant_emotion
        
        # Check for emotion-related thoughts
        emotion_thoughts = [t for t in thoughts if t.source == self.original_manager.ThoughtSource.EMOTION]
        
        if not emotion_thoughts:
            return 0.3  # No emotional thoughts when emotion is active
        
        # Check if thought content matches emotion
        emotion_keywords = {
            "Joy": ["happy", "pleased", "good", "positive"],
            "Frustration": ["annoyed", "difficult", "challenge", "problem"],
            "Curiosity": ["wonder", "interesting", "explore", "understand"],
            "Anxiety": ["worried", "concern", "uncertain", "nervous"]
        }
        
        keywords = emotion_keywords.get(emotion_name, [])
        
        # Check keyword presence in thoughts
        keyword_matches = 0
        for thought in emotion_thoughts:
            if any(kw in thought.content.lower() for kw in keywords):
                keyword_matches += 1
        
        return keyword_matches / len(emotion_thoughts) if emotion_thoughts else 0.5
    
    def _link_thought_to_context(self, thought: Any, context_type: str, context_data: Dict[str, Any]):
        """Link a thought to its triggering context"""
        link_id = f"{context_type}_{datetime.now().timestamp()}"
        
        if thought.thought_id not in self.cross_module_thought_links:
            self.cross_module_thought_links[thought.thought_id] = []
        
        self.cross_module_thought_links[thought.thought_id].append({
            "link_id": link_id,
            "context_type": context_type,
            "context_summary": self._summarize_context(context_data),
            "timestamp": datetime.now().isoformat()
        })
    
    def _summarize_context(self, context_data: Dict[str, Any]) -> str:
        """Create a brief summary of context data"""
        # Extract key information
        if "emotional_state" in str(context_data):
            return f"emotional context: {context_data.get('dominant_emotion', 'various emotions')}"
        elif "goal" in str(context_data):
            return f"goal context: {context_data.get('goal_id', 'goal progress')}"
        elif "memory" in str(context_data):
            return f"memory context: {context_data.get('memory_count', 'memory retrieval')}"
        else:
            return "general context update"
    
    async def _get_recent_contextual_thoughts(self, limit: int = 20) -> List[Any]:
        """Get recent thoughts with context awareness"""
        # Get thoughts from original manager
        filter_criteria = self.original_manager.ThoughtFilter(
            limit=limit,
            exclude_critiqued=False
        )
        
        thoughts = await self.original_manager.get_thoughts(filter_criteria)
        
        # Add context links
        for thought in thoughts:
            if thought.thought_id in self.cross_module_thought_links:
                thought.metadata["context_links"] = self.cross_module_thought_links[thought.thought_id]
        
        return thoughts
    
    async def _analyze_thought_diversity(self, thoughts: List[Any]) -> Dict[str, Any]:
        """Analyze diversity of thought sources and content"""
        if not thoughts:
            return {"diversity_score": 0.0, "dominant_source": None}
        
        # Source diversity
        source_counts = {}
        for thought in thoughts:
            source = thought.source.value
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Calculate diversity metrics
        unique_sources = len(source_counts)
        total_possible_sources = 9  # Based on ThoughtSource enum
        source_diversity = unique_sources / total_possible_sources
        
        # Content diversity (simplified - check unique word stems)
        all_words = set()
        for thought in thoughts:
            words = thought.content.lower().split()
            all_words.update(words)
        
        content_diversity = min(1.0, len(all_words) / (len(thoughts) * 10))  # Assume 10 unique words per thought is good
        
        # Overall diversity
        diversity_score = (source_diversity + content_diversity) / 2
        
        dominant_source = max(source_counts.items(), key=lambda x: x[1])[0] if source_counts else None
        
        return {
            "diversity_score": diversity_score,
            "source_diversity": source_diversity,
            "content_diversity": content_diversity,
            "dominant_source": dominant_source,
            "source_distribution": source_counts
        }
    
    async def _analyze_thought_quality(self, thoughts: List[Any]) -> Dict[str, Any]:
        """Analyze quality of thoughts"""
        if not thoughts:
            return {"quality_score": 0.0, "issues": ["no_thoughts"]}
        
        quality_factors = {
            "has_critique": 0.0,
            "epistemic_awareness": 0.0,
            "contextual_relevance": 0.0,
            "depth": 0.0
        }
        
        issues = []
        
        # Check critique presence
        critiqued_count = sum(1 for t in thoughts if t.critique is not None)
        quality_factors["has_critique"] = critiqued_count / len(thoughts)
        
        # Check epistemic diversity
        epistemic_states = set(t.epistemic_status for t in thoughts)
        if len(epistemic_states) > 1:
            quality_factors["epistemic_awareness"] = min(1.0, len(epistemic_states) / 3)
        else:
            issues.append("low_epistemic_diversity")
        
        # Check context links
        linked_count = sum(1 for t in thoughts if t.thought_id in self.cross_module_thought_links)
        quality_factors["contextual_relevance"] = linked_count / len(thoughts)
        
        # Check thought depth (length as proxy)
        avg_length = sum(len(t.content) for t in thoughts) / len(thoughts)
        quality_factors["depth"] = min(1.0, avg_length / 100)  # 100 chars is good depth
        
        if avg_length < 30:
            issues.append("shallow_thoughts")
        
        # Calculate overall quality
        quality_score = sum(quality_factors.values()) / len(quality_factors)
        
        return {
            "quality_score": quality_score,
            "quality_factors": quality_factors,
            "issues": issues,
            "recommendations": self._generate_quality_recommendations(quality_factors, issues)
        }
    
    def _generate_quality_recommendations(self, factors: Dict[str, float], issues: List[str]) -> List[str]:
        """Generate recommendations for thought quality improvement"""
        recommendations = []
        
        if factors["has_critique"] < 0.3:
            recommendations.append("Increase self-critique to improve thought quality")
        
        if "low_epistemic_diversity" in issues:
            recommendations.append("Consider uncertainty and varying confidence levels")
        
        if factors["contextual_relevance"] < 0.5:
            recommendations.append("Better integrate thoughts with module context")
        
        if "shallow_thoughts" in issues:
            recommendations.append("Develop deeper, more detailed thoughts")
        
        return recommendations
    
    async def _analyze_thought_integration(self, thoughts: List[Any]) -> Dict[str, Any]:
        """Analyze how well thoughts integrate with other systems"""
        integration_scores = {
            "cross_references": 0.0,
            "context_links": 0.0,
            "module_coverage": 0.0
        }
        
        if not thoughts:
            return {"integration_score": 0.0, "scores": integration_scores}
        
        # Check cross-references between thoughts
        total_references = sum(len(t.related_thoughts) for t in thoughts)
        avg_references = total_references / len(thoughts)
        integration_scores["cross_references"] = min(1.0, avg_references / 2)  # 2 references per thought is good
        
        # Check context links
        linked_thoughts = sum(1 for t in thoughts if t.thought_id in self.cross_module_thought_links)
        integration_scores["context_links"] = linked_thoughts / len(thoughts)
        
        # Check module coverage in thought content
        modules_mentioned = set()
        module_keywords = {
            "emotion", "goal", "memory", "relationship", "mode", "observation", "reflection"
        }
        
        for thought in thoughts:
            content_lower = thought.content.lower()
            for keyword in module_keywords:
                if keyword in content_lower:
                    modules_mentioned.add(keyword)
        
        integration_scores["module_coverage"] = len(modules_mentioned) / len(module_keywords)
        
        # Overall integration
        integration_score = sum(integration_scores.values()) / len(integration_scores)
        
        return {
            "integration_score": integration_score,
            "scores": integration_scores,
            "integrated_modules": list(modules_mentioned)
        }
    
    async def _detect_thought_loops(self, thoughts: List[Any]) -> Dict[str, Any]:
        """Detect repetitive thought patterns or loops"""
        if len(thoughts) < 5:
            return {"loops_detected": False, "repetition_score": 0.0}
        
        # Check for content similarity
        thought_contents = [t.content.lower() for t in thoughts]
        
        # Simple repetition detection
        repetitions = 0
        for i, content1 in enumerate(thought_contents):
            for j, content2 in enumerate(thought_contents[i+1:], i+1):
                similarity = self._calculate_text_similarity(content1, content2)
                if similarity > 0.8:
                    repetitions += 1
        
        max_possible_pairs = len(thoughts) * (len(thoughts) - 1) / 2
        repetition_score = repetitions / max_possible_pairs if max_possible_pairs > 0 else 0
        
        # Detect thematic loops
        themes = self._extract_thought_themes(thoughts)
        theme_repetition = len(themes) < len(thoughts) / 3  # If themes are much fewer than thoughts
        
        loops_detected = repetition_score > 0.3 or theme_repetition
        
        loop_info = []
        if loops_detected:
            loop_info.append({
                "type": "content_repetition" if repetition_score > 0.3 else "thematic_loop",
                "severity": "high" if repetition_score > 0.5 else "medium"
            })
        
        return {
            "loops_detected": loops_detected,
            "repetition_score": repetition_score,
            "unique_themes": len(themes),
            "loop_info": loop_info
        }
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_thought_themes(self, thoughts: List[Any]) -> Set[str]:
        """Extract main themes from thoughts"""
        themes = set()
        
        # Simple theme extraction based on key terms
        theme_keywords = {
            "goal": ["goal", "objective", "achieve", "complete"],
            "emotion": ["feel", "emotion", "mood", "sentiment"],
            "plan": ["plan", "strategy", "approach", "next"],
            "analysis": ["analyze", "understand", "reason", "why"],
            "perception": ["notice", "observe", "see", "detect"],
            "control": ["control", "command", "direct", "manage"],
            "relationship": ["user", "bond", "connection", "trust"]
        }
        
        for thought in thoughts:
            content_lower = thought.content.lower()
            for theme, keywords in theme_keywords.items():
                if any(kw in content_lower for kw in keywords):
                    themes.add(theme)
        
        return themes
    
    async def _generate_meta_thoughts(self, diversity: Dict, quality: Dict, integration: Dict) -> List[Any]:
        """Generate meta-thoughts about the thinking process"""
        meta_thoughts = []
        
        # Generate meta-thought about thought patterns
        meta_thought = await self.original_manager.generate_thought(
            self.original_manager.ThoughtSource.META,
            {
                "meta_analysis": "thought_patterns",
                "diversity_score": diversity["diversity_score"],
                "quality_score": quality["quality_score"],
                "integration_score": integration["integration_score"],
                "self_assessment": self._assess_thinking_quality(diversity, quality, integration)
            }
        )
        meta_thoughts.append(meta_thought)
        
        # If thinking quality is low, generate improvement thought
        if quality["quality_score"] < 0.5 or diversity["diversity_score"] < 0.4:
            improvement_thought = await self.original_manager.generate_thought(
                self.original_manager.ThoughtSource.SELF_CRITIQUE,
                {
                    "critique_of": "thinking_process",
                    "issues": quality.get("issues", []) + (["low_diversity"] if diversity["diversity_score"] < 0.4 else []),
                    "improvement_needed": True
                }
            )
            meta_thoughts.append(improvement_thought)
        
        return meta_thoughts
    
    def _assess_thinking_quality(self, diversity: Dict, quality: Dict, integration: Dict) -> str:
        """Assess overall thinking quality"""
        avg_score = (diversity["diversity_score"] + quality["quality_score"] + integration["integration_score"]) / 3
        
        if avg_score >= 0.7:
            return "thinking process is functioning well"
        elif avg_score >= 0.5:
            return "thinking process is adequate but could improve"
        else:
            return "thinking process needs significant improvement"
    
    async def _get_synthesis_relevant_thoughts(self, context: SharedContext) -> List[Any]:
        """Get thoughts relevant to response synthesis"""
        # Filter for recent, high-priority thoughts
        filter_criteria = self.original_manager.ThoughtFilter(
            max_age_seconds=300,  # Last 5 minutes
            min_priority=self.original_manager.ThoughtPriority.MEDIUM,
            limit=10
        )
        
        thoughts = await self.original_manager.get_thoughts(filter_criteria)
        
        # Further filter for synthesis relevance
        synthesis_relevant = []
        for thought in thoughts:
            # Include planning and synthesis thoughts
            if thought.source in [self.original_manager.ThoughtSource.PLANNING, 
                                self.original_manager.ThoughtSource.META]:
                synthesis_relevant.append(thought)
            # Include recent critiques
            elif thought.source == self.original_manager.ThoughtSource.SELF_CRITIQUE and thought.critique:
                synthesis_relevant.append(thought)
            # Include high-priority perception/reasoning
            elif thought.priority in [self.original_manager.ThoughtPriority.HIGH, 
                                    self.original_manager.ThoughtPriority.CRITICAL]:
                synthesis_relevant.append(thought)
        
        return synthesis_relevant[:6]  # Limit to 6 most relevant
    
    def _assess_leakage_severity(self, detected_thoughts: List[str]) -> str:
        """Assess severity of thought leakage"""
        if not detected_thoughts:
            return "none"
        
        # Check for sensitive content
        sensitive_keywords = [
            "uncertain", "don't know", "confused", "struggling",
            "error", "mistake", "wrong", "failed"
        ]
        
        sensitive_count = 0
        for thought in detected_thoughts:
            if any(kw in thought.lower() for kw in sensitive_keywords):
                sensitive_count += 1
        
        if sensitive_count > len(detected_thoughts) / 2:
            return "high"
        elif len(detected_thoughts) > 3:
            return "high"
        elif len(detected_thoughts) > 1:
            return "medium"
        else:
            return "low"
    
    async def _generate_synthesis_guidance(self, thoughts: List[Any]) -> Dict[str, Any]:
        """Generate guidance for synthesis based on thoughts"""
        guidance = {
            "key_considerations": [],
            "tone_suggestions": [],
            "content_warnings": [],
            "quality_focus": []
        }
        
        for thought in thoughts:
            # Extract considerations from planning thoughts
            if thought.source == self.original_manager.ThoughtSource.PLANNING:
                if "approach" in thought.content.lower():
                    guidance["key_considerations"].append(f"Consider: {thought.content[:50]}...")
            
            # Extract warnings from critiques
            elif thought.source == self.original_manager.ThoughtSource.SELF_CRITIQUE:
                if "issue" in thought.content.lower() or "problem" in thought.content.lower():
                    guidance["content_warnings"].append(f"Warning: {thought.content[:50]}...")
            
            # Extract quality focus from meta thoughts
            elif thought.source == self.original_manager.ThoughtSource.META:
                if "quality" in thought.content.lower() or "improve" in thought.content.lower():
                    guidance["quality_focus"].append(f"Focus: {thought.content[:50]}...")
        
        # Add tone suggestions based on emotional thoughts
        emotional_thoughts = [t for t in thoughts if t.source == self.original_manager.ThoughtSource.EMOTION]
        if emotional_thoughts:
            guidance["tone_suggestions"].append("Align tone with current emotional state")
        
        return guidance
    
    # Delegate all other methods to the original manager
    def __getattr__(self, name):
        """Delegate any missing methods to the original thoughts manager"""
        return getattr(self.original_manager, name)
