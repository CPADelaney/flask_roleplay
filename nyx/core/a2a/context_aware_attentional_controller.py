# nyx/core/a2a/context_aware_attentional_controller.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareAttentionalController(ContextAwareModule):
    """
    Advanced AttentionalController with full context distribution capabilities
    """
    
    def __init__(self, original_controller):
        super().__init__("attentional_controller")
        self.original_controller = original_controller
        self.context_subscriptions = [
            "emotional_state_update", "goal_context_available", "memory_retrieval_complete",
            "urgent_need_expression", "body_state_changed", "relationship_milestone",
            "cognitive_load_update", "sensory_input_detected", "task_priority_change"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize attention processing for this context"""
        logger.debug(f"AttentionalController received context for user: {context.user_id}")
        
        # Analyze input for attention-worthy elements
        attention_analysis = await self._analyze_input_for_attention(context.user_input)
        
        # Get current attentional state
        current_state = await self._get_current_attention_state()
        
        # Calculate initial attention allocation
        initial_allocation = await self._calculate_initial_attention_allocation(context, attention_analysis)
        
        # Send comprehensive attention context to other modules
        await self.send_context_update(
            update_type="attention_context_available",
            data={
                "current_foci": current_state["current_foci"],
                "attentional_resources": current_state["resources"],
                "salient_items": attention_analysis["salient_items"],
                "attention_demands": attention_analysis["demands"],
                "initial_allocation": initial_allocation,
                "max_parallel_foci": self.original_controller.max_foci,
                "attention_strategy": self._determine_attention_strategy(context)
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect attention"""
        
        if update.update_type == "emotional_state_update":
            # Emotional state significantly affects attention
            emotional_data = update.data
            arousal = emotional_data.get("arousal", 0.5)
            valence = emotional_data.get("valence", 0.0)
            dominant_emotion = emotional_data.get("dominant_emotion")
            
            await self._process_emotional_attention_modulation(arousal, valence, dominant_emotion, emotional_data)
        
        elif update.update_type == "goal_context_available":
            # Goals direct attention strategically
            goal_data = update.data
            active_goals = goal_data.get("active_goals", [])
            goal_priorities = goal_data.get("goal_priorities", {})
            
            await self._process_goal_directed_attention(active_goals, goal_priorities)
        
        elif update.update_type == "memory_retrieval_complete":
            # Retrieved memories may demand attention
            memory_data = update.data
            memories = memory_data.get("retrieved_memories", [])
            
            await self._process_memory_triggered_attention(memories)
        
        elif update.update_type == "urgent_need_expression":
            # Urgent needs override current attention
            urgency_data = update.data
            urgency_score = urgency_data.get("urgency", 0.0)
            needs = urgency_data.get("needs_to_express", [])
            
            if urgency_score > 0.7:
                await self._emergency_attention_override(needs, urgency_score)
        
        elif update.update_type == "cognitive_load_update":
            # High cognitive load reduces attention capacity
            load_data = update.data
            cognitive_load = load_data.get("load_level", 0.5)
            
            await self._adjust_attention_capacity(cognitive_load)
        
        elif update.update_type == "sensory_input_detected":
            # New sensory input may capture attention
            sensory_data = update.data
            modality = sensory_data.get("modality")
            intensity = sensory_data.get("intensity", 0.5)
            
            if intensity > 0.6:
                await self._process_sensory_attention_capture(sensory_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with full context awareness"""
        # Get cross-module messages for comprehensive context
        messages = await self.get_cross_module_messages()
        
        # Extract all salient items from context and messages
        salient_items = await self._extract_comprehensive_salient_items(context, messages)
        
        # Calculate attention weights for all items
        attention_weights = await self._calculate_contextual_attention_weights(salient_items, context)
        
        # Determine control signals based on context
        control_signals = await self._generate_attention_control_signals(context, messages)
        
        # Update attention through original controller
        attention_result = await self.original_controller.update_attention(
            salient_items=salient_items,
            control_signals=control_signals
        )
        
        # Track attention metrics
        attention_metrics = await self._calculate_attention_metrics(attention_result)
        
        # Send detailed attention update
        await self.send_context_update(
            update_type="attention_state_update",
            data={
                "current_foci": [f.dict() for f in attention_result],
                "attention_shifts": attention_metrics["shifts"],
                "resource_utilization": attention_metrics["resource_usage"],
                "attention_stability": attention_metrics["stability"],
                "processing_efficiency": attention_metrics["efficiency"]
            }
        )
        
        return {
            "attention_updated": True,
            "foci_count": len(attention_result),
            "salient_items_processed": len(salient_items),
            "attention_weights": attention_weights,
            "control_signals_applied": len(control_signals),
            "cross_module_integration": len(messages)
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze attention patterns in context"""
        # Get current attention state
        current_state = await self._get_current_attention_state()
        
        # Analyze attention distribution
        distribution_analysis = await self._analyze_attention_distribution(current_state)
        
        # Analyze attention coherence with other systems
        coherence_analysis = await self._analyze_attention_coherence(context)
        
        # Identify attention bottlenecks
        bottlenecks = await self._identify_attention_bottlenecks(context)
        
        # Generate attention insights
        insights = await self._generate_attention_insights(context, current_state)
        
        return {
            "attention_distribution": distribution_analysis,
            "system_coherence": coherence_analysis,
            "identified_bottlenecks": bottlenecks,
            "attention_insights": insights,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize attention-related response components"""
        # Get all relevant context
        messages = await self.get_cross_module_messages()
        current_state = await self._get_current_attention_state()
        
        # Generate attention-informed synthesis
        attention_synthesis = {
            "attention_guidance": await self._generate_response_attention_guidance(context),
            "focus_emphasis": await self._determine_response_focus_points(context, current_state),
            "attention_metadata": await self._generate_attention_metadata(context),
            "processing_notes": await self._generate_processing_notes(messages),
            "attention_coherence_check": await self._final_coherence_check(context, messages)
        }
        
        return attention_synthesis
    
    # ========================================================================================
    # DETAILED HELPER METHODS
    # ========================================================================================
    
    async def _analyze_input_for_attention(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input for attention-worthy elements"""
        input_lower = user_input.lower()
        
        # Identify salient items
        salient_items = []
        
        # Check for questions (high attention priority)
        if any(q in input_lower for q in ['?', 'what', 'how', 'why', 'when', 'where', 'who']):
            salient_items.append({
                "target": "question_content",
                "type": "query",
                "priority": 0.9,
                "source": "linguistic_analysis"
            })
        
        # Check for emotional keywords
        emotional_keywords = {
            "urgent": 0.95, "emergency": 0.95, "help": 0.9,
            "important": 0.8, "critical": 0.85, "need": 0.7,
            "please": 0.6, "confused": 0.7, "worried": 0.75
        }
        
        for keyword, priority in emotional_keywords.items():
            if keyword in input_lower:
                salient_items.append({
                    "target": f"emotional_keyword_{keyword}",
                    "type": "emotional",
                    "priority": priority,
                    "source": "keyword_detection"
                })
        
        # Check for task indicators
        task_keywords = ["do", "create", "make", "build", "analyze", "explain", "show", "tell"]
        for keyword in task_keywords:
            if keyword in input_lower:
                salient_items.append({
                    "target": f"task_{keyword}",
                    "type": "task",
                    "priority": 0.7,
                    "source": "task_detection"
                })
        
        # Calculate attention demands
        demands = {
            "cognitive_load": len(user_input.split()) / 50.0,  # Normalize by typical length
            "emotional_intensity": len([s for s in salient_items if s["type"] == "emotional"]) * 0.2,
            "task_complexity": len([s for s in salient_items if s["type"] == "task"]) * 0.15,
            "urgency_level": max([s["priority"] for s in salient_items]) if salient_items else 0.5
        }
        
        return {
            "salient_items": salient_items,
            "demands": demands,
            "total_salience": len(salient_items),
            "primary_focus": salient_items[0]["target"] if salient_items else "general_input"
        }
    
    async def _get_current_attention_state(self) -> Dict[str, Any]:
        """Get comprehensive current attention state"""
        return {
            "current_foci": [
                {
                    "target": f.target,
                    "strength": f.strength,
                    "duration_ms": f.duration_ms,
                    "source": f.source,
                    "age_ms": (datetime.now() - datetime.fromisoformat(f.timestamp)).total_seconds() * 1000
                }
                for f in self.original_controller.current_foci
            ],
            "resources": {
                "available": self.original_controller.attentional_resources,
                "total_capacity": self.original_controller.total_attentional_capacity,
                "utilization": 1.0 - self.original_controller.attentional_resources
            },
            "inhibited_targets": list(self.original_controller.inhibited_targets.keys()),
            "active_foci_count": len(self.original_controller.current_foci),
            "max_foci": self.original_controller.max_foci
        }
    
    async def _calculate_initial_attention_allocation(self, context: SharedContext, 
                                                    attention_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate initial attention allocation based on context"""
        allocation = {}
        
        # Base allocation on salient items
        total_priority = sum(item["priority"] for item in attention_analysis["salient_items"])
        
        for item in attention_analysis["salient_items"]:
            normalized_priority = item["priority"] / total_priority if total_priority > 0 else 0.5
            allocation[item["target"]] = {
                "weight": normalized_priority,
                "type": item["type"],
                "duration_ms": 2000 if item["priority"] > 0.8 else 1000
            }
        
        # Adjust for emotional context
        if context.emotional_state:
            arousal = sum(context.emotional_state.values()) / len(context.emotional_state) if context.emotional_state else 0.5
            if arousal > 0.7:
                # High arousal - focus on fewer items
                sorted_targets = sorted(allocation.items(), key=lambda x: x[1]["weight"], reverse=True)
                for i, (target, alloc) in enumerate(sorted_targets):
                    if i < 2:  # Top 2 items
                        allocation[target]["weight"] *= 1.2
                    else:
                        allocation[target]["weight"] *= 0.8
        
        return allocation
    
    def _determine_attention_strategy(self, context: SharedContext) -> str:
        """Determine overall attention strategy based on context"""
        # Analyze context to determine strategy
        if context.session_context.get("task_purpose") == "emergency":
            return "focused_emergency"
        elif context.goal_context and len(context.goal_context.get("active_goals", [])) > 3:
            return "distributed_multitasking"
        elif context.emotional_state and max(context.emotional_state.values(), default=0) > 0.8:
            return "emotion_driven"
        else:
            return "balanced_exploration"
    
    async def _process_emotional_attention_modulation(self, arousal: float, valence: float, 
                                                    dominant_emotion: tuple, emotional_data: Dict[str, Any]):
        """Process how emotions modulate attention"""
        # High arousal narrows attention
        if arousal > 0.8:
            # Focus on fewer, more important items
            await self._narrow_attention_focus(factor=0.6)
            
            # Send update about narrowed focus
            await self.send_context_update(
                update_type="attention_narrowed",
                data={
                    "reason": "high_emotional_arousal",
                    "arousal_level": arousal,
                    "focus_factor": 0.6
                },
                scope=ContextScope.GLOBAL
            )
        
        elif arousal < 0.3:
            # Low arousal allows broader attention
            await self._broaden_attention_focus(factor=1.4)
        
        # Specific emotion effects
        if dominant_emotion:
            emotion_name, intensity = dominant_emotion
            
            if emotion_name == "Fear" and intensity > 0.6:
                # Fear biases attention toward threats
                await self._bias_attention_toward("threat_detection", bias_strength=0.3)
            elif emotion_name == "Curiosity" and intensity > 0.5:
                # Curiosity broadens exploratory attention
                await self._bias_attention_toward("novelty_exploration", bias_strength=0.25)
            elif emotion_name == "Love" and intensity > 0.7:
                # Love focuses attention on relationship elements
                await self._bias_attention_toward("relationship_focus", bias_strength=0.35)
    
    async def _narrow_attention_focus(self, factor: float = 0.7):
        """Narrow attention to fewer targets"""
        current_foci = self.original_controller.current_foci
        
        if len(current_foci) > 1:
            # Sort by strength and keep only top items
            sorted_foci = sorted(current_foci, key=lambda f: f.strength, reverse=True)
            keep_count = max(1, int(len(sorted_foci) * factor))
            
            # Inhibit lower priority items
            for i in range(keep_count, len(sorted_foci)):
                focus = sorted_foci[i]
                await self.original_controller._inhibit_attention(None, focus.target, 5000)
    
    async def _broaden_attention_focus(self, factor: float = 1.3):
        """Broaden attention to include more targets"""
        # Reduce inhibition times
        current_time = time.time()
        for target, expiry in list(self.original_controller.inhibited_targets.items()):
            new_expiry = current_time + (expiry - current_time) / factor
            self.original_controller.inhibited_targets[target] = new_expiry
    
    async def _bias_attention_toward(self, bias_type: str, bias_strength: float):
        """Bias attention toward specific types of targets"""
        # Update attention biases in original controller
        if hasattr(self.original_controller, 'attention_biases'):
            self.original_controller.attention_biases[bias_type] = bias_strength
    
    async def _process_goal_directed_attention(self, active_goals: List[Dict[str, Any]], 
                                             goal_priorities: Dict[str, float]):
        """Process goal-directed attention allocation"""
        # Create attention focuses for high-priority goals
        for goal in active_goals:
            goal_id = goal.get("id", "unknown")
            priority = goal_priorities.get(goal_id, goal.get("priority", 0.5))
            
            if priority > 0.7:
                # High priority goal deserves attention
                goal_target = f"goal_{goal_id}"
                strength = min(1.0, priority)
                
                await self.original_controller._focus_attention(
                    None, goal_target, strength, 3000, "goal_system"
                )
                
                # Send update about goal-focused attention
                await self.send_context_update(
                    update_type="goal_attention_allocated",
                    data={
                        "goal_id": goal_id,
                        "attention_strength": strength,
                        "goal_description": goal.get("description", "")
                    },
                    target_modules=["goal_manager"],
                    scope=ContextScope.TARGETED
                )
    
    async def _process_memory_triggered_attention(self, memories: List[Dict[str, Any]]):
        """Process attention triggered by retrieved memories"""
        # Highly significant memories capture attention
        for memory in memories:
            significance = memory.get("significance", 5) / 10.0
            
            if significance > 0.7:
                memory_id = memory.get("id", "unknown")
                memory_target = f"memory_{memory_id}"
                
                # Focus attention on significant memory
                await self.original_controller._focus_attention(
                    None, memory_target, significance * 0.8, 2000, "memory_system"
                )
                
                logger.debug(f"Memory {memory_id} captured attention (significance: {significance})")
    
    async def _emergency_attention_override(self, needs: List[Dict[str, Any]], urgency_score: float):
        """Override current attention for emergency needs"""
        logger.warning(f"Emergency attention override triggered (urgency: {urgency_score})")
        
        # Clear non-critical attention
        for focus in list(self.original_controller.current_foci):
            if focus.source != "emergency":
                await self.original_controller._inhibit_attention(None, focus.target, 10000)
        
        # Focus on urgent needs
        for need in needs[:self.original_controller.max_foci]:
            need_name = need.get("need", "unknown")
            target = f"urgent_need_{need_name}"
            
            await self.original_controller._focus_attention(
                None, target, urgency_score, 5000, "emergency"
            )
        
        # Send emergency attention notification
        await self.send_context_update(
            update_type="emergency_attention_active",
            data={
                "urgency_score": urgency_score,
                "focused_needs": [n.get("need") for n in needs[:self.original_controller.max_foci]],
                "attention_cleared": True
            },
            priority=ContextPriority.CRITICAL
        )
    
    async def _adjust_attention_capacity(self, cognitive_load: float):
        """Adjust attention capacity based on cognitive load"""
        # High cognitive load reduces available attention
        load_factor = 1.0 - (cognitive_load * 0.5)  # Max 50% reduction
        
        # Adjust resources
        self.original_controller.attentional_resources *= load_factor
        
        # Reduce max foci if load is very high
        if cognitive_load > 0.8:
            effective_max_foci = max(1, int(self.original_controller.max_foci * 0.7))
            
            # Remove excess foci if needed
            if len(self.original_controller.current_foci) > effective_max_foci:
                sorted_foci = sorted(
                    self.original_controller.current_foci, 
                    key=lambda f: f.strength
                )
                for focus in sorted_foci[:-effective_max_foci]:
                    await self.original_controller._inhibit_attention(None, focus.target, 3000)
    
    async def _process_sensory_attention_capture(self, sensory_data: Dict[str, Any]):
        """Process attention capture by sensory input"""
        modality = sensory_data.get("modality", "unknown")
        intensity = sensory_data.get("intensity", 0.5)
        
        # High intensity sensory input captures attention
        if intensity > 0.7:
            target = f"sensory_{modality}"
            strength = min(1.0, intensity)
            
            # Sensory capture is typically brief but strong
            await self.original_controller._focus_attention(
                None, target, strength, 1500, "sensory_capture"
            )
            
            logger.debug(f"Sensory attention capture: {modality} (intensity: {intensity})")
    
    async def _extract_comprehensive_salient_items(self, context: SharedContext, 
                                                  messages: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Extract all salient items from context and messages"""
        salient_items = []
        
        # Extract from user input
        input_analysis = await self._analyze_input_for_attention(context.user_input)
        salient_items.extend(input_analysis["salient_items"])
        
        # Extract from emotional context
        if context.emotional_state:
            for emotion, intensity in context.emotional_state.items():
                if intensity > 0.5:
                    salient_items.append({
                        "target": f"emotion_{emotion.lower()}",
                        "novelty": 0.3,
                        "intensity": intensity,
                        "emotional_impact": intensity,
                        "goal_relevance": 0.5,
                        "source": "emotional_state"
                    })
        
        # Extract from goal context
        if context.goal_context:
            for goal in context.goal_context.get("active_goals", [])[:3]:  # Top 3 goals
                salient_items.append({
                    "target": f"goal_{goal.get('id', 'unknown')}",
                    "novelty": 0.4,
                    "intensity": goal.get("priority", 0.5),
                    "goal_relevance": 1.0,
                    "source": "goal_system"
                })
        
        # Extract from cross-module messages
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                if msg['type'] == 'attention_request':
                    salient_items.append({
                        "target": msg['data'].get('target', f"{module_name}_request"),
                        "novelty": msg['data'].get('novelty', 0.5),
                        "intensity": msg['data'].get('intensity', 0.5),
                        "source": module_name
                    })
        
        # Sort by overall salience
        for item in salient_items:
            item['overall_salience'] = await self._calculate_item_salience(item, context)
        
        salient_items.sort(key=lambda x: x['overall_salience'], reverse=True)
        
        return salient_items
    
    async def _calculate_item_salience(self, item: Dict[str, Any], context: SharedContext) -> float:
        """Calculate overall salience score for an item"""
        # Base salience calculation
        novelty = item.get('novelty', 0.5)
        intensity = item.get('intensity', 0.5)
        emotional_impact = item.get('emotional_impact', 0.5)
        goal_relevance = item.get('goal_relevance', 0.5)
        
        # Get configured weights
        config = self.original_controller.saliency_config
        
        # Calculate weighted salience
        salience = (
            novelty * config.novelty_weight +
            intensity * config.intensity_weight +
            emotional_impact * config.emotional_weight +
            goal_relevance * config.goal_weight
        )
        
        # Apply source-based modulation
        source = item.get('source', 'unknown')
        if source == 'emergency':
            salience *= 1.5
        elif source == 'user_input':
            salience *= 1.2
        elif source == 'goal_system' and context.session_context.get('task_purpose') == 'goal_pursuit':
            salience *= 1.3
        
        return min(1.0, salience)
    
    async def _calculate_contextual_attention_weights(self, salient_items: List[Dict[str, Any]], 
                                                    context: SharedContext) -> Dict[str, float]:
        """Calculate attention weights for all salient items"""
        weights = {}
        
        # Get current attention state
        current_foci = {f.target: f.strength for f in self.original_controller.current_foci}
        
        for item in salient_items:
            target = item['target']
            base_weight = item['overall_salience']
            
            # Adjust for current focus (attention inertia)
            if target in current_foci:
                base_weight = base_weight * 0.7 + current_foci[target] * 0.3
            
            # Adjust for inhibition
            if target in self.original_controller.inhibited_targets:
                base_weight *= 0.2  # Strongly reduced but not zero
            
            weights[target] = base_weight
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    async def _generate_attention_control_signals(self, context: SharedContext, 
                                                messages: Dict[str, List[Dict[str, Any]]]) -> List[Any]:
        """Generate control signals for attention system"""
        from nyx.core.attentional_controller import AttentionalControl
        
        control_signals = []
        
        # Check for explicit attention requests in messages
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                if msg['type'] == 'request_attention_focus':
                    control_signals.append(AttentionalControl(
                        target=msg['data']['target'],
                        priority=msg['data'].get('priority', 0.5),
                        duration_ms=msg['data'].get('duration_ms', 2000),
                        source=module_name,
                        action="focus"
                    ))
                elif msg['type'] == 'request_attention_inhibit':
                    control_signals.append(AttentionalControl(
                        target=msg['data']['target'],
                        priority=0.0,
                        duration_ms=msg['data'].get('duration_ms', 5000),
                        source=module_name,
                        action="inhibit"
                    ))
        
        # Generate control signals based on context
        if context.session_context.get('attention_mode') == 'focused':
            # In focused mode, inhibit distractions
            for focus in self.original_controller.current_foci:
                if focus.strength < 0.5:
                    control_signals.append(AttentionalControl(
                        target=focus.target,
                        priority=0.0,
                        duration_ms=3000,
                        source="attention_mode",
                        action="inhibit"
                    ))
        
        return control_signals
    
    async def _calculate_attention_metrics(self, attention_result: List[Any]) -> Dict[str, Any]:
        """Calculate detailed attention metrics"""
        current_foci = attention_result
        previous_foci = getattr(self, '_previous_foci', [])
        
        # Calculate shifts
        current_targets = {f.target for f in current_foci}
        previous_targets = {f.target for f in previous_foci}
        
        new_targets = current_targets - previous_targets
        dropped_targets = previous_targets - current_targets
        maintained_targets = current_targets & previous_targets
        
        # Calculate resource usage
        total_strength = sum(f.strength for f in current_foci)
        resource_usage = min(1.0, total_strength / max(1, self.original_controller.max_foci))
        
        # Calculate stability (how much attention configuration changed)
        stability = len(maintained_targets) / max(1, len(current_targets | previous_targets))
        
        # Calculate efficiency (resource usage vs number of foci)
        efficiency = (len(current_foci) / max(1, self.original_controller.max_foci)) * (1.0 - resource_usage)
        
        # Store current for next comparison
        self._previous_foci = current_foci
        
        return {
            "shifts": {
                "new_targets": list(new_targets),
                "dropped_targets": list(dropped_targets),
                "maintained_targets": list(maintained_targets),
                "total_changes": len(new_targets) + len(dropped_targets)
            },
            "resource_usage": resource_usage,
            "stability": stability,
            "efficiency": efficiency,
            "focus_count": len(current_foci),
            "average_strength": total_strength / len(current_foci) if current_foci else 0
        }
    
    async def _analyze_attention_distribution(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how attention is distributed"""
        foci = current_state["current_foci"]
        
        if not foci:
            return {
                "distribution_type": "empty",
                "concentration": 0.0,
                "balance": 1.0
            }
        
        # Calculate concentration (how focused vs distributed)
        strengths = [f["strength"] for f in foci]
        max_strength = max(strengths)
        avg_strength = sum(strengths) / len(strengths)
        
        concentration = max_strength / avg_strength if avg_strength > 0 else 1.0
        
        # Calculate balance (how evenly distributed)
        if len(strengths) > 1:
            variance = sum((s - avg_strength) ** 2 for s in strengths) / len(strengths)
            balance = 1.0 / (1.0 + variance)
        else:
            balance = 1.0
        
        # Determine distribution type
        if concentration > 1.5:
            distribution_type = "highly_focused"
        elif concentration < 1.2 and balance > 0.7:
            distribution_type = "evenly_distributed"
        else:
            distribution_type = "mixed"
        
        return {
            "distribution_type": distribution_type,
            "concentration": concentration,
            "balance": balance,
            "focus_count": len(foci),
            "dominant_focus": foci[0]["target"] if foci else None
        }
    
    async def _analyze_attention_coherence(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze coherence between attention and other systems"""
        coherence_score = 1.0
        issues = []
        
        current_foci = {f.target for f in self.original_controller.current_foci}
        
        # Check goal coherence
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            high_priority_goals = [g for g in active_goals if g.get("priority", 0) > 0.7]
            
            for goal in high_priority_goals:
                goal_target = f"goal_{goal.get('id', 'unknown')}"
                if goal_target not in current_foci and len(current_foci) < self.original_controller.max_foci:
                    coherence_score -= 0.2
                    issues.append(f"High priority goal not in attention: {goal.get('description', 'unknown')}")
        
        # Check emotional coherence
        if context.emotional_state:
            arousal = sum(context.emotional_state.values()) / len(context.emotional_state) if context.emotional_state else 0.5
            focus_count = len(current_foci)
            
            if arousal > 0.8 and focus_count > 2:
                coherence_score -= 0.15
                issues.append("High arousal but attention too distributed")
            elif arousal < 0.3 and focus_count < 2:
                coherence_score -= 0.1
                issues.append("Low arousal but attention too narrow")
        
        return {
            "coherence_score": max(0.0, coherence_score),
            "coherence_issues": issues,
            "is_coherent": coherence_score > 0.7
        }
    
    async def _identify_attention_bottlenecks(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify bottlenecks in attention allocation"""
        bottlenecks = []
        
        # Check for resource exhaustion
        if self.original_controller.attentional_resources < 0.1:
            bottlenecks.append({
                "type": "resource_exhaustion",
                "severity": "high",
                "description": "Attentional resources nearly depleted",
                "recommendation": "Reduce focus targets or allow recovery time"
            })
        
        # Check for focus saturation
        if len(self.original_controller.current_foci) >= self.original_controller.max_foci:
            bottlenecks.append({
                "type": "focus_saturation",
                "severity": "medium",
                "description": "Maximum attention foci reached",
                "recommendation": "Prioritize and drop low-importance foci"
            })
        
        # Check for inhibition overload
        if len(self.original_controller.inhibited_targets) > 10:
            bottlenecks.append({
                "type": "inhibition_overload",
                "severity": "low",
                "description": "Many targets being actively inhibited",
                "recommendation": "Review inhibition necessity"
            })
        
        return bottlenecks
    
    async def _generate_attention_insights(self, context: SharedContext, 
                                         current_state: Dict[str, Any]) -> List[str]:
        """Generate insights about attention patterns"""
        insights = []
        
        # Analyze focus duration patterns
        if current_state["current_foci"]:
            avg_age = sum(f["age_ms"] for f in current_state["current_foci"]) / len(current_state["current_foci"])
            
            if avg_age > 10000:  # 10 seconds
                insights.append("Attention has been stable for an extended period")
            elif avg_age < 2000:  # 2 seconds
                insights.append("Rapid attention shifting detected")
        
        # Analyze resource utilization
        utilization = current_state["resources"]["utilization"]
        if utilization > 0.9:
            insights.append("Near maximum attention capacity utilization")
        elif utilization < 0.3:
            insights.append("Significant unused attention capacity available")
        
        # Context-specific insights
        if context.session_context.get("task_purpose") == "learning" and len(current_state["current_foci"]) > 3:
            insights.append("Consider narrowing focus for better learning retention")
        
        return insights
    
    async def _generate_response_attention_guidance(self, context: SharedContext) -> Dict[str, Any]:
        """Generate guidance for response generation based on attention"""
        current_foci = self.original_controller.current_foci
        
        # Identify primary focus for response
        primary_focus = None
        if current_foci:
            strongest = max(current_foci, key=lambda f: f.strength)
            primary_focus = {
                "target": strongest.target,
                "strength": strongest.strength,
                "type": self._identify_focus_type(strongest.target)
            }
        
        # Identify elements to emphasize
        emphasis_targets = []
        for focus in current_foci:
            if focus.strength > 0.6:
                emphasis_targets.append({
                    "target": focus.target,
                    "emphasis_level": focus.strength,
                    "reason": focus.source
                })
        
        return {
            "primary_focus": primary_focus,
            "emphasis_targets": emphasis_targets,
            "attention_distribution": self._describe_attention_pattern(current_foci),
            "recommended_response_style": self._recommend_response_style(context, current_foci)
        }
    
    def _identify_focus_type(self, target: str) -> str:
        """Identify the type of attention focus"""
        if target.startswith("goal_"):
            return "goal_directed"
        elif target.startswith("emotion_"):
            return "emotional"
        elif target.startswith("memory_"):
            return "memory_triggered"
        elif target.startswith("urgent_"):
            return "urgency_driven"
        elif target.startswith("sensory_"):
            return "sensory_captured"
        else:
            return "general"
    
    def _describe_attention_pattern(self, current_foci: List[Any]) -> str:
        """Describe the current attention pattern"""
        if not current_foci:
            return "unfocused"
        elif len(current_foci) == 1:
            return "single_focused"
        elif len(current_foci) <= 3:
            return "multi_focused"
        else:
            return "distributed"
    
    def _recommend_response_style(self, context: SharedContext, current_foci: List[Any]) -> str:
        """Recommend response style based on attention state"""
        if any(f.source == "emergency" for f in current_foci):
            return "direct_urgent"
        elif len(current_foci) == 1:
            return "focused_detailed"
        elif context.emotional_state and max(context.emotional_state.values(), default=0) > 0.7:
            return "emotionally_attuned"
        else:
            return "balanced_comprehensive"
    
    async def _determine_response_focus_points(self, context: SharedContext, 
                                             current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Determine key points to focus on in response"""
        focus_points = []
        
        # Extract main topics from current foci
        for focus in current_state["current_foci"]:
            if focus["strength"] > 0.5:
                focus_points.append({
                    "topic": self._extract_topic_from_target(focus["target"]),
                    "importance": focus["strength"],
                    "context": focus["source"]
                })
        
        # Add context-specific focus points
        if context.user_input and "?" in context.user_input:
            focus_points.append({
                "topic": "question_response",
                "importance": 0.9,
                "context": "user_query"
            })
        
        # Sort by importance
        focus_points.sort(key=lambda x: x["importance"], reverse=True)
        
        return focus_points[:5]  # Top 5 focus points
    
    def _extract_topic_from_target(self, target: str) -> str:
        """Extract readable topic from attention target"""
        if "_" in target:
            parts = target.split("_", 1)
            return parts[1] if len(parts) > 1 else target
        return target
    
    async def _generate_attention_metadata(self, context: SharedContext) -> Dict[str, Any]:
        """Generate metadata about attention state for response"""
        return {
            "attention_distribution": self._describe_attention_pattern(self.original_controller.current_foci),
            "resource_level": self.original_controller.attentional_resources,
            "focus_count": len(self.original_controller.current_foci),
            "inhibition_active": len(self.original_controller.inhibited_targets) > 0,
            "attention_strategy": self._determine_attention_strategy(context)
        }
    
    async def _generate_processing_notes(self, messages: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Generate notes about attention processing"""
        notes = []
        
        # Note significant attention events
        attention_requests = sum(1 for msgs in messages.values() 
                               for msg in msgs if 'attention' in msg.get('type', ''))
        
        if attention_requests > 3:
            notes.append(f"High cross-module attention demand ({attention_requests} requests)")
        
        # Note resource status
        if self.original_controller.attentional_resources < 0.2:
            notes.append("Low attentional resources - conservation mode active")
        
        # Note focus stability
        if hasattr(self, '_previous_foci'):
            current_targets = {f.target for f in self.original_controller.current_foci}
            previous_targets = {f.target for f in self._previous_foci}
            
            if current_targets == previous_targets:
                notes.append("Stable attention configuration maintained")
            elif len(current_targets - previous_targets) > 2:
                notes.append("Significant attention shift occurred")
        
        return notes
    
    async def _final_coherence_check(self, context: SharedContext, 
                                   messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Final check of attention coherence for response"""
        # Comprehensive coherence check
        coherence = await self._analyze_attention_coherence(context)
        
        # Check if attention supports response generation
        response_support = True
        support_issues = []
        
        # Ensure critical elements have attention
        if context.user_input and "?" in context.user_input:
            question_focused = any(
                "question" in f.target or "query" in f.target 
                for f in self.original_controller.current_foci
            )
            if not question_focused:
                response_support = False
                support_issues.append("User question not in attention focus")
        
        # Check for conflicting attention
        current_foci = self.original_controller.current_foci
        if any(f.source == "emergency" for f in current_foci) and len(current_foci) > 2:
            response_support = False
            support_issues.append("Emergency focus diluted by other targets")
        
        return {
            "overall_coherence": coherence["coherence_score"],
            "response_support": response_support,
            "support_issues": support_issues,
            "attention_ready": coherence["is_coherent"] and response_support
        }
    
    # Delegate all other methods to the original controller
    def __getattr__(self, name):
        """Delegate any missing methods to the original controller"""
        return getattr(self.original_controller, name)
