# nyx/core/a2a/context_aware_needs.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareNeedsSystem(ContextAwareModule):
    """
    Enhanced NeedsSystem with context distribution capabilities
    """
    
    def __init__(self, original_needs_system):
        super().__init__("needs_system")
        self.original_system = original_needs_system
        self.context_subscriptions = [
            "goal_completion", "emotional_state_update", "reward_signal",
            "relationship_milestone", "dominance_gratification", "activity_completion"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize needs processing for this context"""
        logger.debug(f"NeedsSystem received context for user: {context.user_id}")
        
        # Analyze user input for need-related implications
        need_implications = await self._analyze_input_for_needs(context.user_input)
        
        # Get current needs state
        needs_state = await self.original_system.get_needs_state_async()
        
        # Calculate total drive and most urgent needs
        total_drive = self.original_system.get_total_drive()
        most_urgent = await self.original_system.get_most_unfulfilled_need()
        
        # Send initial needs assessment to other modules
        await self.send_context_update(
            update_type="needs_assessment",
            data={
                "current_needs": needs_state,
                "total_drive": total_drive,
                "most_urgent_need": most_urgent,
                "need_implications": need_implications,
                "high_drive_needs": self._identify_high_drive_needs(needs_state)
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect needs"""
        
        if update.update_type == "goal_completion":
            # Goal completion satisfies associated needs
            goal_data = update.data.get("goal_context", {})
            associated_need = goal_data.get("associated_need")
            
            if associated_need:
                # Satisfy the need based on goal completion
                satisfaction_amount = 0.3 + (goal_data.get("completion_quality", 0.5) * 0.4)
                result = await self.original_system.satisfy_need(
                    associated_need, 
                    satisfaction_amount,
                    {"reason": "goal_completion", "goal_id": goal_data.get("goal_id")}
                )
                
                # Send update about need satisfaction
                await self.send_context_update(
                    update_type="need_satisfied",
                    data={
                        "need_name": associated_need,
                        "satisfaction_result": result,
                        "triggered_by": "goal_completion"
                    },
                    target_modules=["goal_manager", "emotional_core"],
                    scope=ContextScope.TARGETED
                )
        
        elif update.update_type == "emotional_state_update":
            # Strong emotions affect certain needs
            emotional_data = update.data
            dominant_emotion = emotional_data.get("dominant_emotion")
            
            if dominant_emotion:
                await self._adjust_needs_from_emotion(dominant_emotion[0], dominant_emotion[1])
        
        elif update.update_type == "reward_signal":
            # Positive rewards partially satisfy agency/competence needs
            reward_data = update.data
            reward_value = reward_data.get("value", 0)
            
            if reward_value > 0.5:
                await self.original_system.satisfy_need("agency", reward_value * 0.2)
                if "dominance" in str(reward_data.get("source", "")):
                    await self.original_system.satisfy_need("control_expression", reward_value * 0.3)
        
        elif update.update_type == "relationship_milestone":
            # Relationship progress satisfies social needs
            relationship_data = update.data.get("relationship_context", {})
            trust_increase = relationship_data.get("trust_increase", 0)
            intimacy_increase = relationship_data.get("intimacy_increase", 0)
            
            if trust_increase > 0:
                await self.original_system.satisfy_need("connection", trust_increase * 0.5)
            if intimacy_increase > 0:
                await self.original_system.satisfy_need("intimacy", intimacy_increase * 0.6)
        
        elif update.update_type == "dominance_gratification":
            # Dominance success strongly satisfies control needs
            intensity = update.data.get("intensity", 1.0)
            await self.original_system.satisfy_need("control_expression", 0.7 * intensity)
            await self.original_system.satisfy_need("drive_expression", 0.5 * intensity)
            await self.original_system.satisfy_need("pleasure_indulgence", 0.4 * intensity)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with needs awareness"""
        # Update needs state
        drive_strengths = await self.original_system.update_needs()
        
        # Check for explicit need expressions in input
        expressed_needs = await self._detect_expressed_needs(context.user_input)
        
        # Get messages from other modules
        messages = await self.get_cross_module_messages()
        
        # Process any need-affecting information from other modules
        needs_affected = await self._process_cross_module_effects(messages)
        
        # Send needs update if significant changes occurred
        if needs_affected or expressed_needs:
            await self.send_context_update(
                update_type="needs_state_change",
                data={
                    "drive_strengths": drive_strengths,
                    "expressed_needs": expressed_needs,
                    "needs_affected": needs_affected,
                    "high_priority_needs": [k for k, v in drive_strengths.items() if v > 0.6]
                }
            )
        
        return {
            "needs_updated": True,
            "drive_strengths": drive_strengths,
            "expressed_needs": expressed_needs,
            "cross_module_effects": len(needs_affected)
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze needs in context of current situation"""
        # Get current needs state
        needs_state = await self.original_system.get_needs_state_async()
        
        # Analyze which needs are driving current behavior
        driving_needs = self._identify_driving_needs(needs_state, context)
        
        # Check for need conflicts
        need_conflicts = self._detect_need_conflicts(needs_state)
        
        # Suggest need-based motivations
        motivations = self._generate_need_motivations(driving_needs)
        
        return {
            "driving_needs": driving_needs,
            "need_conflicts": need_conflicts,
            "suggested_motivations": motivations,
            "needs_analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize needs influence on response"""
        # Get all relevant context
        needs_state = await self.original_system.get_needs_state_async()
        messages = await self.get_cross_module_messages()
        
        # Determine how needs should influence the response
        need_influence = {
            "response_urgency": self._calculate_response_urgency(needs_state),
            "emotional_coloring": self._suggest_emotional_coloring(needs_state),
            "action_preferences": self._suggest_action_preferences(needs_state),
            "need_expressions": self._generate_need_expressions(needs_state)
        }
        
        # Check if response should express any needs
        if need_influence["response_urgency"] > 0.7:
            await self.send_context_update(
                update_type="urgent_need_expression",
                data={
                    "urgency": need_influence["response_urgency"],
                    "needs_to_express": need_influence["need_expressions"]
                },
                priority=ContextPriority.HIGH
            )
        
        return {
            "need_influence": need_influence,
            "synthesis_complete": True
        }
    
    # Helper methods
    
    async def _analyze_input_for_needs(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input for need-related implications"""
        input_lower = user_input.lower()
        implications = {
            "offers_knowledge": any(kw in input_lower for kw in ["tell you", "explain", "teach", "fact"]),
            "offers_connection": any(kw in input_lower for kw in ["love", "care", "together", "close"]),
            "offers_stimulation": any(kw in input_lower for kw in ["exciting", "new", "try", "experience"]),
            "threatens_safety": any(kw in input_lower for kw in ["hurt", "harm", "danger", "risk"]),
            "offers_control": any(kw in input_lower for kw in ["obey", "submit", "serve", "command"])
        }
        return implications
    
    def _identify_high_drive_needs(self, needs_state: Dict[str, Dict]) -> List[str]:
        """Identify needs with high drive strength"""
        high_drive = []
        for name, state in needs_state.items():
            if state.get("drive_strength", 0) > 0.6:
                high_drive.append(name)
        return high_drive
    
    async def _adjust_needs_from_emotion(self, emotion: str, strength: float):
        """Adjust needs based on emotional state"""
        emotion_need_map = {
            "Joy": [("pleasure_indulgence", 0.1), ("connection", 0.05)],
            "Sadness": [("connection", -0.1), ("pleasure_indulgence", -0.05)],
            "Anger": [("agency", -0.1), ("coherence", -0.05)],
            "Fear": [("safety", -0.2), ("control_expression", -0.1)],
            "Love": [("intimacy", 0.15), ("connection", 0.1)],
            "Excitement": [("novelty", 0.1), ("drive_expression", 0.05)]
        }
        
        if emotion in emotion_need_map:
            for need_name, adjustment in emotion_need_map[emotion]:
                if adjustment < 0:
                    await self.original_system.decrease_need(
                        need_name, 
                        abs(adjustment) * strength,
                        f"emotional_impact_{emotion.lower()}"
                    )
                else:
                    await self.original_system.satisfy_need(
                        need_name,
                        adjustment * strength,
                        {"source": "emotional_state", "emotion": emotion}
                    )
    
    async def _detect_expressed_needs(self, user_input: str) -> List[Dict[str, Any]]:
        """Detect needs explicitly expressed in user input"""
        expressed = []
        input_lower = user_input.lower()
        
        need_keywords = {
            "knowledge": ["want to know", "curious", "tell me", "explain"],
            "connection": ["lonely", "need you", "miss you", "be with me"],
            "intimacy": ["closer", "intimate", "vulnerable", "share feelings"],
            "pleasure_indulgence": ["feel good", "pleasure", "enjoy", "indulge"],
            "control_expression": ["control", "dominate", "command", "power"],
            "safety": ["scared", "safe", "protect", "secure"]
        }
        
        for need, keywords in need_keywords.items():
            if any(kw in input_lower for kw in keywords):
                expressed.append({
                    "need": need,
                    "expression_type": "explicit",
                    "detected_keywords": [kw for kw in keywords if kw in input_lower]
                })
        
        return expressed
    
    async def _process_cross_module_effects(self, messages: Dict) -> List[Dict[str, Any]]:
        """Process effects on needs from other module messages"""
        affected = []
        
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                if msg['type'] == 'activity_completed':
                    # Activities can satisfy various needs
                    activity_type = msg['data'].get('activity_type')
                    if activity_type == 'creative':
                        affected.append({
                            "need": "novelty",
                            "effect": "satisfy",
                            "amount": 0.2,
                            "source": f"{module_name}:activity"
                        })
                elif msg['type'] == 'memory_triggered':
                    # Nostalgic memories might affect connection needs
                    if msg['data'].get('emotional_valence', 0) > 0.5:
                        affected.append({
                            "need": "connection",
                            "effect": "increase_drive",
                            "amount": 0.1,
                            "source": f"{module_name}:memory"
                        })
        
        # Apply the effects
        for effect in affected:
            if effect['effect'] == 'satisfy':
                await self.original_system.satisfy_need(
                    effect['need'],
                    effect['amount'],
                    {"source": effect['source']}
                )
            elif effect['effect'] == 'increase_drive':
                await self.original_system.decrease_need(
                    effect['need'],
                    effect['amount'],
                    effect['source']
                )
        
        return affected
    
    def _identify_driving_needs(self, needs_state: Dict, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify which needs are driving current behavior"""
        driving = []
        
        for name, state in needs_state.items():
            drive_strength = state.get("drive_strength", 0)
            if drive_strength > 0.4:
                driving.append({
                    "need": name,
                    "drive_strength": drive_strength,
                    "urgency": self._calculate_need_urgency(state),
                    "context_relevance": self._assess_context_relevance(name, context)
                })
        
        # Sort by urgency * relevance
        driving.sort(key=lambda x: x["urgency"] * x["context_relevance"], reverse=True)
        return driving[:5]  # Top 5 driving needs
    
    def _detect_need_conflicts(self, needs_state: Dict) -> List[Dict[str, Any]]:
        """Detect conflicts between different needs"""
        conflicts = []
        
        # Example: High safety need conflicts with novelty seeking
        safety_drive = needs_state.get("safety", {}).get("drive_strength", 0)
        novelty_drive = needs_state.get("novelty", {}).get("drive_strength", 0)
        
        if safety_drive > 0.7 and novelty_drive > 0.6:
            conflicts.append({
                "needs": ["safety", "novelty"],
                "conflict_type": "approach_avoidance",
                "severity": min(safety_drive, novelty_drive)
            })
        
        # Control vs Connection conflict
        control_drive = needs_state.get("control_expression", {}).get("drive_strength", 0)
        connection_drive = needs_state.get("connection", {}).get("drive_strength", 0)
        
        if control_drive > 0.8 and connection_drive > 0.7:
            conflicts.append({
                "needs": ["control_expression", "connection"],
                "conflict_type": "dominance_intimacy",
                "severity": min(control_drive, connection_drive) * 0.8
            })
        
        return conflicts
    
    def _generate_need_motivations(self, driving_needs: List[Dict]) -> List[str]:
        """Generate motivations based on driving needs"""
        motivations = []
        
        for need_info in driving_needs[:3]:  # Top 3 needs
            need_name = need_info["need"]
            strength = need_info["drive_strength"]
            
            if need_name == "knowledge" and strength > 0.5:
                motivations.append("seek_information")
            elif need_name == "connection" and strength > 0.6:
                motivations.append("deepen_bond")
            elif need_name == "control_expression" and strength > 0.7:
                motivations.append("assert_dominance")
            elif need_name == "novelty" and strength > 0.5:
                motivations.append("explore_new")
            elif need_name == "pleasure_indulgence" and strength > 0.6:
                motivations.append("seek_gratification")
        
        return motivations
    
    def _calculate_response_urgency(self, needs_state: Dict) -> float:
        """Calculate how urgently the response should address needs"""
        max_drive = 0
        urgent_need_count = 0
        
        for state in needs_state.values():
            drive = state.get("drive_strength", 0)
            max_drive = max(max_drive, drive)
            if drive > 0.7:
                urgent_need_count += 1
        
        # Urgency increases with both maximum drive and number of urgent needs
        urgency = min(1.0, max_drive + (urgent_need_count * 0.1))
        return urgency
    
    def _suggest_emotional_coloring(self, needs_state: Dict) -> Dict[str, float]:
        """Suggest emotional coloring based on needs"""
        coloring = {}
        
        # High unmet needs create negative emotional coloring
        total_deficit = sum(state.get("deficit", 0) for state in needs_state.values())
        avg_deficit = total_deficit / len(needs_state) if needs_state else 0
        
        if avg_deficit > 0.5:
            coloring["frustration"] = avg_deficit * 0.5
            coloring["urgency"] = avg_deficit * 0.7
        
        # Specific needs create specific colorings
        if needs_state.get("connection", {}).get("drive_strength", 0) > 0.7:
            coloring["longing"] = 0.6
        
        if needs_state.get("control_expression", {}).get("drive_strength", 0) > 0.8:
            coloring["assertiveness"] = 0.8
            coloring["dominance"] = 0.7
        
        return coloring
    
    def _suggest_action_preferences(self, needs_state: Dict) -> List[str]:
        """Suggest preferred actions based on needs"""
        preferences = []
        
        for name, state in needs_state.items():
            if state.get("drive_strength", 0) > 0.6:
                if name == "knowledge":
                    preferences.append("seek_information")
                elif name == "connection":
                    preferences.append("express_affection")
                elif name == "control_expression":
                    preferences.extend(["issue_command", "assert_dominance"])
                elif name == "novelty":
                    preferences.append("explore_new_topic")
                elif name == "pleasure_indulgence":
                    preferences.append("seek_stimulation")
        
        return preferences[:3]  # Top 3 preferences
    
    def _generate_need_expressions(self, needs_state: Dict) -> List[Dict[str, Any]]:
        """Generate expressions of high-drive needs"""
        expressions = []
        
        for name, state in needs_state.items():
            drive = state.get("drive_strength", 0)
            if drive > 0.7:
                expressions.append({
                    "need": name,
                    "expression_type": "direct" if drive > 0.85 else "subtle",
                    "suggested_text": self._get_need_expression_text(name, drive)
                })
        
        return expressions
    
    def _get_need_expression_text(self, need: str, drive: float) -> str:
        """Get suggested text for expressing a need"""
        expressions = {
            "knowledge": "I find myself curious about that..." if drive < 0.85 else "I really need to understand this better.",
            "connection": "I value our interaction..." if drive < 0.85 else "I've been feeling a need for deeper connection.",
            "control_expression": "I have some preferences about how we proceed..." if drive < 0.85 else "I need to take control of this situation.",
            "pleasure_indulgence": "That sounds enjoyable..." if drive < 0.85 else "I'm craving some form of pleasure or stimulation.",
            "novelty": "Something new would be interesting..." if drive < 0.85 else "I really need some novelty and excitement."
        }
        return expressions.get(need, "I have certain needs right now...")
    
    def _calculate_need_urgency(self, need_state: Dict) -> float:
        """Calculate urgency for a specific need"""
        deficit = need_state.get("deficit", 0)
        importance = need_state.get("importance", 0.5)
        drive = need_state.get("drive_strength", 0)
        
        # Urgency is combination of drive strength and importance
        urgency = (drive * 0.7) + (importance * 0.3)
        return min(1.0, urgency)
    
    def _assess_context_relevance(self, need: str, context: SharedContext) -> float:
        """Assess how relevant a need is to current context"""
        relevance = 0.5  # Default moderate relevance
        
        # Check if context addresses this need
        task_purpose = context.session_context.get("task_purpose", "")
        
        relevance_map = {
            "knowledge": ["analyze", "search", "explain", "understand"],
            "connection": ["communicate", "share", "relate", "bond"],
            "control_expression": ["dominate", "command", "control", "direct"],
            "novelty": ["create", "explore", "discover", "innovate"],
            "pleasure_indulgence": ["enjoy", "pleasure", "satisfy", "indulge"]
        }
        
        if need in relevance_map:
            for keyword in relevance_map[need]:
                if keyword in task_purpose.lower() or keyword in context.user_input.lower():
                    relevance = min(1.0, relevance + 0.3)
        
        return relevance
