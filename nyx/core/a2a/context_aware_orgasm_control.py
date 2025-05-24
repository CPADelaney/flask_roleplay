# nyx/core/a2a/context_aware_orgasm_control.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareOrgasmControl(ContextAwareModule):
    """
    Enhanced Orgasm Control System with full context distribution capabilities
    """
    
    def __init__(self, original_orgasm_control):
        super().__init__("orgasm_control")
        self.original_control = original_orgasm_control
        self.context_subscriptions = [
            "dominance_level_change", "submission_metric_update", "begging_detected",
            "punishment_needed", "reward_earned", "emotional_state_update",
            "physical_state_update", "relationship_milestone", "femdom_mode_change",
            "denial_extension_request", "control_escalation"
        ]
    
    async def on_context_received(self, context: SharedContext):
        """Initialize orgasm control processing for this context"""
        logger.debug(f"OrgasmControl received context for user: {context.user_id}")
        
        # Get current permission state
        permission_state = await self.original_control.get_permission_state(context.user_id)
        
        # Analyze input for orgasm control relevance
        control_analysis = await self._analyze_control_context(context.user_input, permission_state)
        
        # Send initial control context
        await self.send_context_update(
            update_type="orgasm_control_state",
            data={
                "user_id": context.user_id,
                "current_status": permission_state.get("current_status", "unknown"),
                "denial_active": permission_state.get("denial_active", False),
                "days_since_last": permission_state.get("days_since_last_orgasm", 0),
                "begging_count": permission_state.get("begging_count", 0),
                "control_analysis": control_analysis
            },
            priority=ContextPriority.HIGH
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates affecting orgasm control"""
        
        if update.update_type == "begging_detected":
            # Process begging with context
            begging_data = update.data
            user_id = begging_data.get("user_id")
            
            # Consider emotional state and submission level
            await self._process_contextual_begging(user_id, begging_data)
        
        elif update.update_type == "punishment_needed":
            # Extend denial as punishment
            punishment_data = update.data
            if punishment_data.get("severity", 0) > 0.5:
                await self._apply_punishment_denial(
                    punishment_data.get("user_id"),
                    punishment_data.get("severity")
                )
        
        elif update.update_type == "reward_earned":
            # Consider reducing denial or granting permission
            reward_data = update.data
            await self._process_reward_adjustment(
                reward_data.get("user_id"),
                reward_data.get("reward_value", 0)
            )
        
        elif update.update_type == "dominance_level_change":
            # Adjust control strictness
            dominance_data = update.data
            await self._adjust_control_strictness(
                dominance_data.get("user_id"),
                dominance_data.get("new_level", 0.5)
            )
        
        elif update.update_type == "physical_state_update":
            # Consider physical arousal/frustration
            physical_data = update.data
            await self._integrate_physical_state(physical_data)
        
        elif update.update_type == "denial_extension_request":
            # Process extension request from other systems
            extension_data = update.data
            await self._process_extension_request(extension_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with context awareness"""
        messages = await self.get_cross_module_messages()
        
        # Check for permission request
        if await self._is_permission_request(context.user_input):
            # Process with full context
            result = await self._process_permission_with_context(
                context.user_id,
                context.user_input,
                messages
            )
            
            # Send permission result update
            await self.send_context_update(
                update_type="permission_request_processed",
                data={
                    "user_id": context.user_id,
                    "granted": result.get("permission_granted", False),
                    "desperation_level": result.get("desperation_level", 0),
                    "reason": result.get("reason")
                },
                priority=ContextPriority.HIGH
            )
            
            # If begging detected, notify other systems
            if result.get("desperation_level", 0) > 0.5:
                await self.send_context_update(
                    update_type="begging_detected",
                    data={
                        "user_id": context.user_id,
                        "desperation_level": result.get("desperation_level"),
                        "begging_text": context.user_input
                    }
                )
        
        # Check for orgasm report
        elif await self._is_orgasm_report(context.user_input):
            result = await self._process_orgasm_report_with_context(
                context.user_id,
                context.user_input,
                messages
            )
            
            # Send orgasm recorded update
            await self.send_context_update(
                update_type="orgasm_recorded",
                data={
                    "user_id": context.user_id,
                    "type": result.get("type", "full"),
                    "with_permission": result.get("with_permission", True)
                }
            )
        
        return {
            "processed": True,
            "control_engaged": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze orgasm control patterns"""
        user_id = context.user_id
        messages = await self.get_cross_module_messages()
        
        # Get control patterns
        control_patterns = await self.original_control.analyze_control_patterns(user_id)
        
        # Enhance with context
        contextual_analysis = await self._enhance_pattern_analysis(control_patterns, messages)
        
        # Generate recommendations
        control_recommendations = await self._generate_control_recommendations(
            contextual_analysis, messages
        )
        
        # Assess frustration levels
        frustration_assessment = await self._assess_frustration_level(user_id, messages)
        
        return {
            "control_patterns": control_patterns,
            "contextual_analysis": contextual_analysis,
            "recommendations": control_recommendations,
            "frustration_assessment": frustration_assessment
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize orgasm control elements for response"""
        messages = await self.get_cross_module_messages()
        user_id = context.user_id
        
        # Get current state
        permission_state = await self.original_control.get_permission_state(user_id)
        
        synthesis = {
            "include_control_element": False,
            "control_reminders": [],
            "denial_emphasis": None
        }
        
        # Determine if control should be mentioned
        if permission_state.get("denial_active"):
            synthesis["include_control_element"] = True
            
            # Add appropriate reminders
            active_denial = permission_state.get("active_denial", {})
            if active_denial:
                hours_remaining = active_denial.get("hours_remaining", 0)
                if hours_remaining < 24:
                    synthesis["control_reminders"].append(
                        f"denial_ending_soon_{hours_remaining:.0f}h"
                    )
                
                if active_denial.get("level") in ["SEVERE", "EXTREME"]:
                    synthesis["denial_emphasis"] = "strict"
        
        # Check for recent begging
        recent_begging = permission_state.get("recent_begging", [])
        if recent_begging and not recent_begging[-1].get("granted"):
            synthesis["include_denial_reminder"] = True
            synthesis["recent_denial_count"] = permission_state.get("denied_begging_count", 0)
        
        # Check for control opportunities
        if self._should_suggest_control_change(messages):
            synthesis["suggest_control_change"] = True
            synthesis["suggested_change"] = await self._determine_control_suggestion(
                permission_state, messages
            )
        
        return synthesis
    
    # Helper methods
    async def _analyze_control_context(self, user_input: str, permission_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze input for orgasm control relevance"""
        input_lower = user_input.lower()
        
        return {
            "requests_permission": any(kw in input_lower for kw in ["may i", "can i", "permission", "please"]) and
                                 any(kw in input_lower for kw in ["cum", "come", "orgasm", "finish"]),
            "reports_orgasm": any(kw in input_lower for kw in ["came", "cummed", "orgasmed", "finished"]),
            "expresses_frustration": any(kw in input_lower for kw in ["frustrated", "desperate", "need", "aching"]),
            "begs": any(kw in input_lower for kw in ["beg", "please", "need", "desperate"]),
            "currently_denied": permission_state.get("denial_active", False)
        }
    
    async def _process_contextual_begging(self, user_id: str, begging_data: Dict[str, Any]):
        """Process begging with full context"""
        desperation = begging_data.get("desperation_level", 0.5)
        
        # Get emotional context
        emotional_context = begging_data.get("emotional_context", {})
        
        # Adjust response based on emotional state
        if emotional_context.get("dominant_emotion") == "Sadistic":
            # More likely to deny when feeling sadistic
            await self.send_context_update(
                update_type="sadistic_denial_opportunity",
                data={
                    "user_id": user_id,
                    "desperation_level": desperation,
                    "recommendation": "deny_for_sadistic_pleasure"
                },
                priority=ContextPriority.HIGH
            )
    
    async def _apply_punishment_denial(self, user_id: str, severity: float):
        """Apply denial as punishment"""
        # Calculate extension based on severity
        extension_hours = int(24 * severity * 2)  # Up to 48 hours for severe
        
        # Check current state
        permission_state = await self.original_control.get_permission_state(user_id)
        
        if permission_state.get("denial_active"):
            # Extend existing denial
            result = await self.original_control.extend_denial_period(
                user_id=user_id,
                additional_hours=extension_hours,
                reason=f"punishment (severity: {severity:.1f})"
            )
        else:
            # Start new denial
            from nyx.core.femdom.orgasm_control import DenialLevel
            level = DenialLevel.STRICT if severity > 0.7 else DenialLevel.MODERATE
            
            result = await self.original_control.start_denial_period(
                user_id=user_id,
                duration_hours=extension_hours,
                level=level,
                begging_allowed=True,
                conditions={"punishment": True, "severity": severity}
            )
        
        # Send notification
        await self.send_context_update(
            update_type="punishment_denial_applied",
            data={
                "user_id": user_id,
                "hours": extension_hours,
                "severity": severity,
                "result": result
            }
        )
    
    async def _assess_frustration_level(self, user_id: str, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Assess user's frustration level from various signals"""
        permission_state = await self.original_control.get_permission_state(user_id)
        
        # Base frustration on denial duration
        days_denied = permission_state.get("days_since_last_orgasm", 0)
        base_frustration = min(1.0, days_denied / 7.0)  # Max at 1 week
        
        # Adjust for begging
        begging_count = permission_state.get("begging_count", 0)
        denied_count = permission_state.get("denied_begging_count", 0)
        
        if begging_count > 0:
            denial_rate = denied_count / begging_count
            base_frustration += denial_rate * 0.2
        
        # Check physical state from messages
        for module, module_messages in messages.items():
            if module == "somatosensory_system":
                for msg in module_messages:
                    if msg.get("type") == "arousal_state":
                        arousal = msg.get("data", {}).get("arousal_level", 0)
                        base_frustration += arousal * 0.1
        
        return {
            "frustration_level": min(1.0, base_frustration),
            "components": {
                "denial_duration": days_denied,
                "begging_frequency": begging_count,
                "denial_rate": denied_count / max(1, begging_count),
                "physical_arousal": "detected" if base_frustration > 0.7 else "normal"
            }
        }
    
    # Delegate other methods to original control
    def __getattr__(self, name):
        """Delegate any missing methods to the original control"""
        return getattr(self.original_control, name)
