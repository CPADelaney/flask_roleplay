# nyx/core/a2a/context_aware_reflexive_system.py

import logging
from typing import Dict, List, Any, Optional
import time

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareReflexiveSystem(ContextAwareModule):
    """
    Enhanced ReflexiveSystem with full context distribution capabilities
    """
    
    def __init__(self, original_reflexive_system):
        super().__init__("reflexive_system")
        self.original_system = original_reflexive_system
        self.context_subscriptions = [
            "urgency_detected", "threat_detected", "pattern_recognized",
            "emotional_spike", "attention_demand", "mode_change",
            "rapid_response_needed", "goal_critical_action"
        ]
        
        # Context-based reflex modulation
        self.context_sensitivity = 1.0  # Multiplier for reflex sensitivity
        self.suppression_factors = {}  # Factors that suppress reflexes
        self.enhancement_factors = {}  # Factors that enhance reflexes
        self.context_overrides = {}  # Context-based pattern overrides
        
    async def on_context_received(self, context: SharedContext):
        """Initialize reflexive processing for this context"""
        logger.debug(f"ReflexiveSystem received context for user: {context.user_id}")
        
        # Analyze context for reflex modulation
        reflex_params = await self._analyze_context_for_reflexes(context)
        
        # Set initial reflex state based on context
        await self._configure_reflexes_for_context(context, reflex_params)
        
        # Send initial reflexive context
        await self.send_context_update(
            update_type="reflexive_system_ready",
            data={
                "active_patterns": len(self.original_system.reflex_patterns),
                "response_mode": self.original_system.response_mode,
                "context_sensitivity": self.context_sensitivity,
                "reflexes_configured": True
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates that affect reflexive responses"""
        
        if update.update_type == "urgency_detected":
            # High urgency requires faster reflexes
            urgency_data = update.data
            urgency_score = urgency_data.get("urgency_score", 0.5)
            
            if urgency_score > 0.8:
                # Switch to hyper mode
                self.original_system.set_response_mode("hyper")
                self.context_sensitivity = 1.5
                
                await self.send_context_update(
                    update_type="reflex_mode_change",
                    data={
                        "new_mode": "hyper",
                        "reason": "high_urgency",
                        "sensitivity": self.context_sensitivity
                    }
                )
                
        elif update.update_type == "threat_detected":
            # Threats trigger defensive reflexes
            threat_data = update.data
            threat_level = threat_data.get("threat_level", 0.5)
            threat_type = threat_data.get("threat_type", "unknown")
            
            # Enhance threat-related reflexes
            self.enhancement_factors[f"threat_{threat_type}"] = threat_level
            
            # Create temporary defensive reflex if needed
            await self._create_defensive_reflex(threat_type, threat_level)
            
        elif update.update_type == "emotional_spike":
            # Emotional spikes can trigger or suppress reflexes
            emotional_data = update.data
            emotion = emotional_data.get("emotion", "unknown")
            intensity = emotional_data.get("intensity", 0.5)
            
            # Fear enhances defensive reflexes
            if emotion == "Fear" and intensity > 0.7:
                self.enhancement_factors["defensive"] = intensity
            # Joy might suppress defensive reflexes
            elif emotion == "Joy" and intensity > 0.7:
                self.suppression_factors["defensive"] = intensity
                
        elif update.update_type == "attention_demand":
            # High attention demand might suppress non-critical reflexes
            attention_data = update.data
            demand_level = attention_data.get("demand_level", 0.5)
            
            if demand_level > 0.8:
                # Suppress non-critical reflexes
                self.suppression_factors["non_critical"] = demand_level
                
        elif update.update_type == "mode_change":
            # System mode changes affect reflex configuration
            mode_data = update.data
            new_mode = mode_data.get("new_mode", "normal")
            
            reflex_mode_mapping = {
                "focused": "normal",
                "relaxed": "relaxed",
                "alert": "hyper",
                "learning": "learning"
            }
            
            reflex_mode = reflex_mode_mapping.get(new_mode, "normal")
            self.original_system.set_response_mode(reflex_mode)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for potential reflex triggers"""
        messages = await self.get_cross_module_messages()
        
        # Check if input requires reflexive response
        reflex_analysis = await self._analyze_input_for_reflexes(context.user_input, context)
        
        reflex_triggered = False
        reflex_result = None
        
        if reflex_analysis["requires_reflex"]:
            # Prepare stimulus with context
            stimulus = await self._prepare_contextual_stimulus(context, reflex_analysis)
            
            # Apply context modulation
            stimulus = await self._apply_context_modulation(stimulus, messages)
            
            # Process through reflexive system
            start_time = time.time()
            reflex_result = await self.original_system.process_stimulus_fast(
                stimulus=stimulus,
                domain=reflex_analysis.get("domain"),
                context={"shared_context": context.session_context}
            )
            
            if reflex_result.get("success"):
                reflex_triggered = True
                reaction_time = (time.time() - start_time) * 1000
                
                # Send reflex trigger notification
                await self.send_context_update(
                    update_type="reflex_triggered",
                    data={
                        "pattern_name": reflex_result.get("pattern_name"),
                        "reaction_time_ms": reaction_time,
                        "stimulus_type": reflex_analysis.get("stimulus_type"),
                        "context_modulated": True
                    },
                    priority=ContextPriority.CRITICAL
                )
        
        return {
            "reflex_triggered": reflex_triggered,
            "reflex_result": reflex_result,
            "reflex_analysis": reflex_analysis,
            "context_sensitivity": self.context_sensitivity,
            "active_modulations": {
                "suppressions": list(self.suppression_factors.keys()),
                "enhancements": list(self.enhancement_factors.keys())
            }
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze reflexive patterns and performance"""
        messages = await self.get_cross_module_messages()
        
        # Analyze reflex usage patterns
        usage_analysis = await self._analyze_reflex_usage(context, messages)
        
        # Analyze context influences on reflexes
        context_analysis = await self._analyze_context_influences(messages)
        
        # Identify reflex optimization opportunities
        optimization_analysis = await self._analyze_optimization_opportunities(context)
        
        # Analyze reflex-deliberation balance
        balance_analysis = await self._analyze_reflex_deliberation_balance(context, messages)
        
        return {
            "usage_patterns": usage_analysis,
            "context_influences": context_analysis,
            "optimization_opportunities": optimization_analysis,
            "reflex_deliberation_balance": balance_analysis,
            "system_responsiveness": await self._calculate_system_responsiveness()
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize reflexive system state for response"""
        messages = await self.get_cross_module_messages()
        
        # Determine if reflexive state should influence response
        reflex_influence = await self._calculate_reflex_influence(context, messages)
        
        synthesis = {
            "reflex_readiness": await self._assess_reflex_readiness(),
            "recommended_mode": await self._recommend_response_mode(context, messages),
            "active_patterns": self._get_active_pattern_summary(),
            "performance_metrics": await self._get_performance_summary()
        }
        
        # If reflexes significantly influenced processing, note it
        if reflex_influence > 0.7:
            await self.send_context_update(
                update_type="reflex_influence_high",
                data={
                    "influence_level": reflex_influence,
                    "active_reflexes": len(self.original_system.reflex_patterns),
                    "avg_reaction_time": synthesis["performance_metrics"].get("avg_reaction_time", 0)
                }
            )
        
        return synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _analyze_context_for_reflexes(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze context to determine reflex parameters"""
        params = {
            "base_sensitivity": 1.0,
            "priority_domains": [],
            "suppression_domains": []
        }
        
        # Task purpose affects reflexes
        if context.task_purpose == "gaming":
            params["priority_domains"].append("gaming")
            params["base_sensitivity"] = 1.2
        elif context.task_purpose == "create":
            # Creative tasks might suppress reflexes
            params["suppression_domains"].append("conversation")
            params["base_sensitivity"] = 0.8
        
        # Emotional state affects reflexes
        if context.emotional_state:
            arousal = context.emotional_state.get("arousal", 0.5)
            # High arousal increases reflex sensitivity
            params["base_sensitivity"] *= (0.5 + arousal)
        
        # Active modules affect reflex priorities
        if "goal_manager" in context.active_modules:
            params["priority_domains"].append("decision")
        
        return params
    
    async def _configure_reflexes_for_context(self, context: SharedContext, params: Dict[str, Any]):
        """Configure reflexes based on context parameters"""
        # Set base sensitivity
        self.context_sensitivity = params["base_sensitivity"]
        
        # Adjust domain-specific reflexes
        for domain in params["priority_domains"]:
            if domain in self.original_system.domain_libraries:
                # Boost priority for this domain
                for pattern in self.original_system.domain_libraries[domain].values():
                    pattern.priority += 1
        
        for domain in params["suppression_domains"]:
            if domain in self.original_system.domain_libraries:
                # Reduce priority for this domain
                for pattern in self.original_system.domain_libraries[domain].values():
                    pattern.priority = max(0, pattern.priority - 1)
    
    async def _analyze_input_for_reflexes(self, user_input: str, context: SharedContext) -> Dict[str, Any]:
        """Analyze if input requires reflexive response"""
        analysis = {
            "requires_reflex": False,
            "stimulus_type": None,
            "domain": None,
            "urgency": 0.5
        }
        
        # Quick pattern checks
        input_lower = user_input.lower()
        
        # Emergency/urgent patterns
        urgent_patterns = ["help", "emergency", "urgent", "quickly", "now", "immediately"]
        if any(pattern in input_lower for pattern in urgent_patterns):
            analysis["requires_reflex"] = True
            analysis["stimulus_type"] = "urgent_request"
            analysis["urgency"] = 0.9
            return analysis
        
        # Threat patterns
        threat_patterns = ["attack", "danger", "threat", "hurt", "harm"]
        if any(pattern in input_lower for pattern in threat_patterns):
            analysis["requires_reflex"] = True
            analysis["stimulus_type"] = "threat"
            analysis["domain"] = "defensive"
            analysis["urgency"] = 0.8
            return analysis
        
        # Gaming patterns (if in gaming context)
        if context.task_purpose == "gaming":
            gaming_patterns = ["dodge", "block", "attack", "jump", "shoot"]
            if any(pattern in input_lower for pattern in gaming_patterns):
                analysis["requires_reflex"] = True
                analysis["stimulus_type"] = "game_action"
                analysis["domain"] = "gaming"
                analysis["urgency"] = 0.7
                return analysis
        
        # Simple yes/no decisions
        if input_lower in ["yes", "no", "ok", "okay", "sure", "nope"]:
            analysis["requires_reflex"] = True
            analysis["stimulus_type"] = "simple_response"
            analysis["domain"] = "conversation"
            analysis["urgency"] = 0.4
        
        return analysis
    
    async def _prepare_contextual_stimulus(self, context: SharedContext, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare stimulus with context information"""
        stimulus = {
            "input": context.user_input,
            "type": analysis["stimulus_type"],
            "urgency": analysis["urgency"],
            "user_id": context.user_id,
            "timestamp": context.created_at.isoformat()
        }
        
        # Add emotional context
        if context.emotional_state:
            stimulus["emotional_context"] = {
                "valence": context.emotional_state.get("valence", 0),
                "arousal": context.emotional_state.get("arousal", 0.5)
            }
        
        # Add goal context if relevant
        if context.goal_context and analysis["domain"] == "decision":
            active_goals = context.goal_context.get("active_goals", [])
            if active_goals:
                stimulus["goal_context"] = {
                    "primary_goal": active_goals[0].get("description", ""),
                    "goal_count": len(active_goals)
                }
        
        return stimulus
    
    async def _apply_context_modulation(self, stimulus: Dict[str, Any], messages: Dict) -> Dict[str, Any]:
        """Apply context-based modulation to stimulus"""
        # Apply sensitivity
        if "urgency" in stimulus:
            stimulus["urgency"] *= self.context_sensitivity
        
        # Apply enhancement factors
        for factor_name, factor_value in self.enhancement_factors.items():
            if factor_name in str(stimulus.get("type", "")):
                stimulus["urgency"] = min(1.0, stimulus.get("urgency", 0.5) * (1 + factor_value))
        
        # Apply suppression factors
        for factor_name, factor_value in self.suppression_factors.items():
            if factor_name == "non_critical" and stimulus.get("urgency", 0.5) < 0.7:
                stimulus["urgency"] *= (1 - factor_value * 0.5)
        
        return stimulus
    
    async def _create_defensive_reflex(self, threat_type: str, threat_level: float):
        """Create temporary defensive reflex for threat"""
        if threat_level < 0.6:
            return  # Not significant enough
        
        # Create pattern for this threat type
        pattern_data = {
            "threat_type": threat_type,
            "threat_indicators": {"min": threat_level - 0.1, "max": 1.0}
        }
        
        # Create defensive procedure name
        procedure_name = f"defend_against_{threat_type}"
        
        # Check if procedure exists
        procedures = await self.original_system.memory_manager.list_procedures()
        if procedure_name not in [p["name"] for p in procedures]:
            # Use generic defensive procedure
            procedure_name = "defensive_response"
            
            # If that doesn't exist either, skip
            if procedure_name not in [p["name"] for p in procedures]:
                return
        
        # Register temporary reflex
        reflex_name = f"temp_defend_{threat_type}_{int(time.time())}"
        
        await self.original_system.register_reflex(
            name=reflex_name,
            pattern_data=pattern_data,
            procedure_name=procedure_name,
            threshold=0.6,  # Lower threshold for defensive reflexes
            priority=5,  # High priority
            domain="defensive",
            context_template={"threat_type": threat_type, "defensive": True}
        )
        
        # Mark as temporary (would need to implement cleanup)
        self.context_overrides[reflex_name] = {
            "temporary": True,
            "created_at": time.time(),
            "ttl": 300  # 5 minutes
        }
    
    async def _calculate_system_responsiveness(self) -> float:
        """Calculate overall system responsiveness"""
        if hasattr(self.original_system, 'reaction_times') and self.original_system.reaction_times:
            avg_reaction = sum(self.original_system.reaction_times) / len(self.original_system.reaction_times)
            # Convert to responsiveness score (lower time = higher score)
            # 50ms = 1.0, 100ms = 0.5, 200ms = 0.25
            responsiveness = 50 / max(50, avg_reaction)
            return min(1.0, responsiveness)
        return 0.5
    
    def _get_active_pattern_summary(self) -> Dict[str, int]:
        """Get summary of active patterns by domain"""
        summary = {}
        for domain, patterns in self.original_system.domain_libraries.items():
            if patterns:
                summary[domain] = len(patterns)
        summary["total"] = len(self.original_system.reflex_patterns)
        return summary
    
    # Delegate other methods to original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
