# nyx/core/a2a/context_aware_logging.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareNyxLogger(ContextAwareModule):
    """
    Advanced Logging System with full context distribution capabilities
    """
    
    def __init__(self, original_nyx_logger):
        super().__init__("logging_system")
        self.original_logger = original_nyx_logger
        self.context_subscriptions = [
            "significant_event", "thought_generated", "action_executed",
            "evolution_detected", "error_occurred", "milestone_reached",
            "insight_discovered", "pattern_recognized"
        ]
        self.log_buffer = []
        self.log_patterns = {}
    
    async def on_context_received(self, context: SharedContext):
        """Initialize logging for this context"""
        logger.debug(f"LoggingSystem received context for user: {context.user_id}")
        
        # Create context entry
        await self._log_context_initialization(context)
        
        # Send logging readiness
        await self.send_context_update(
            update_type="logging_system_ready",
            data={
                "log_types": ["thought", "action", "evolution_suggestion"],
                "context_aware": True,
                "pattern_detection": True,
                "cross_module_correlation": True
            },
            priority=ContextPriority.LOW
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules for logging"""
        
        # Log all significant events
        if update.priority in [ContextPriority.CRITICAL, ContextPriority.HIGH]:
            await self._log_high_priority_event(update)
        
        # Handle specific update types
        if update.update_type == "thought_generated":
            thought_data = update.data
            await self.original_logger.log_thought(
                title=thought_data.get("title", "Generated Thought"),
                content=thought_data.get("content", ""),
                metadata=self._enhance_metadata(thought_data.get("metadata", {}), update)
            )
        
        elif update.update_type == "action_executed":
            action_data = update.data
            await self.original_logger.log_action(
                title=action_data.get("title", "Executed Action"),
                content=action_data.get("content", ""),
                metadata=self._enhance_metadata(action_data.get("metadata", {}), update)
            )
        
        elif update.update_type == "evolution_detected":
            evolution_data = update.data
            await self.original_logger.log_evolution_suggestion(
                title=evolution_data.get("title", "Evolution Opportunity"),
                content=evolution_data.get("content", ""),
                metadata=self._enhance_metadata(evolution_data.get("metadata", {}), update)
            )
        
        elif update.update_type == "error_occurred":
            await self._log_error_with_context(update)
        
        elif update.update_type == "milestone_reached":
            await self._log_milestone(update)
        
        # Pattern detection
        await self._detect_and_log_patterns(update)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for logging opportunities"""
        # Log significant inputs
        if self._is_significant_input(context.user_input):
            await self.original_logger.log_thought(
                title="Significant User Input",
                content=f"User expressed: {context.user_input}",
                metadata={
                    "input_type": "significant",
                    "emotional_context": context.emotional_state,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Detect patterns in input
        pattern = self._detect_input_pattern(context.user_input)
        if pattern:
            self.log_patterns[pattern] = self.log_patterns.get(pattern, 0) + 1
            
            if self.log_patterns[pattern] % 5 == 0:  # Every 5 occurrences
                await self._log_pattern_milestone(pattern, self.log_patterns[pattern])
        
        # Get cross-module messages for correlation
        messages = await self.get_cross_module_messages()
        
        return {
            "input_logged": self._is_significant_input(context.user_input),
            "pattern_detected": pattern is not None,
            "cross_module_logs": len(messages)
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze logging patterns and insights"""
        messages = await self.get_cross_module_messages()
        
        # Analyze log patterns
        pattern_analysis = self._analyze_log_patterns()
        
        # Identify emerging themes
        themes = await self._identify_emerging_themes(messages)
        
        # Analyze cross-module correlations
        correlations = await self._analyze_cross_module_correlations(messages)
        
        # Generate insights from logs
        insights = await self._generate_log_insights(pattern_analysis, themes, correlations)
        
        return {
            "pattern_analysis": pattern_analysis,
            "emerging_themes": themes,
            "cross_module_correlations": correlations,
            "generated_insights": insights,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize logging insights for system awareness"""
        messages = await self.get_cross_module_messages()
        
        # Generate synthesis
        synthesis = {
            "logging_recommendations": await self._generate_logging_recommendations(context),
            "significant_patterns": self._get_significant_patterns(),
            "system_health_indicators": await self._assess_system_health_from_logs(),
            "suggested_investigations": await self._suggest_investigations(context)
        }
        
        # Check if we should create a meta-log
        if self._should_create_meta_log(context):
            await self._create_meta_log(synthesis)
        
        return synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _log_context_initialization(self, context: SharedContext):
        """Log the initialization of a new context"""
        await self.original_logger.log_thought(
            title="Context Initialized",
            content=f"New context session started for user {context.user_id}",
            metadata={
                "user_id": context.user_id,
                "conversation_id": context.conversation_id,
                "active_modules": list(context.active_modules),
                "task_purpose": context.task_purpose,
                "timestamp": context.created_at.isoformat()
            }
        )
    
    async def _log_high_priority_event(self, update: ContextUpdate):
        """Log high priority events from any module"""
        await self.original_logger.log_action(
            title=f"High Priority Event: {update.update_type}",
            content=f"Module {update.source_module} reported: {update.data}",
            metadata={
                "source_module": update.source_module,
                "update_type": update.update_type,
                "priority": update.priority.name,
                "timestamp": update.timestamp.isoformat(),
                "data_keys": list(update.data.keys()) if isinstance(update.data, dict) else None
            }
        )
    
    def _enhance_metadata(self, metadata: Dict[str, Any], update: ContextUpdate) -> Dict[str, Any]:
        """Enhance metadata with context information"""
        enhanced = metadata.copy()
        enhanced.update({
            "source_module": update.source_module,
            "update_type": update.update_type,
            "context_timestamp": update.timestamp.isoformat(),
            "priority": update.priority.name
        })
        return enhanced
    
    async def _log_error_with_context(self, update: ContextUpdate):
        """Log errors with full context"""
        error_data = update.data
        
        await self.original_logger.log_thought(
            title=f"Error in {update.source_module}",
            content=f"Error: {error_data.get('error', 'Unknown error')}\n\nContext: {error_data.get('context', 'No context available')}",
            metadata={
                "error_type": error_data.get("error_type", "unknown"),
                "source_module": update.source_module,
                "severity": error_data.get("severity", "medium"),
                "timestamp": datetime.now().isoformat(),
                "stack_trace": error_data.get("stack_trace", None)
            }
        )
    
    async def _log_milestone(self, update: ContextUpdate):
        """Log system milestones"""
        milestone_data = update.data
        
        await self.original_logger.log_action(
            title=f"Milestone: {milestone_data.get('milestone_name', 'Unknown')}",
            content=milestone_data.get("description", "A system milestone was reached"),
            metadata={
                "milestone_type": milestone_data.get("type", "system"),
                "source_module": update.source_module,
                "achievement_data": milestone_data.get("data", {}),
                "timestamp": datetime.now().isoformat()
            }
        )
    
    async def _detect_and_log_patterns(self, update: ContextUpdate):
        """Detect patterns in system behavior"""
        # Add to buffer for pattern detection
        self.log_buffer.append({
            "source": update.source_module,
            "type": update.update_type,
            "timestamp": update.timestamp,
            "priority": update.priority.name
        })
        
        # Keep buffer size manageable
        if len(self.log_buffer) > 100:
            self.log_buffer = self.log_buffer[-100:]
        
        # Detect patterns every 10 events
        if len(self.log_buffer) % 10 == 0:
            patterns = self._analyze_buffer_patterns()
            
            if patterns:
                await self.original_logger.log_thought(
                    title="Pattern Detected",
                    content=f"Detected patterns in system behavior: {patterns}",
                    metadata={
                        "pattern_type": "behavioral",
                        "detection_method": "buffer_analysis",
                        "buffer_size": len(self.log_buffer)
                    }
                )
    
    def _is_significant_input(self, user_input: str) -> bool:
        """Determine if input is significant enough to log"""
        # Log if input contains emotional expression
        emotional_words = ["feel", "love", "hate", "angry", "happy", "sad", "frustrated", "excited"]
        if any(word in user_input.lower() for word in emotional_words):
            return True
        
        # Log if input is a question about the system
        meta_questions = ["how do you", "what are you", "can you explain", "tell me about yourself"]
        if any(question in user_input.lower() for question in meta_questions):
            return True
        
        # Log if input is long (suggests detailed interaction)
        if len(user_input) > 200:
            return True
        
        return False
    
    def _detect_input_pattern(self, user_input: str) -> Optional[str]:
        """Detect patterns in user input"""
        input_lower = user_input.lower()
        
        patterns = {
            "question": "?" in user_input,
            "command": any(cmd in input_lower for cmd in ["do", "make", "create", "write"]),
            "emotional": any(emo in input_lower for emo in ["feel", "feeling", "emotion"]),
            "meta": any(meta in input_lower for meta in ["you", "your", "yourself"]),
            "creative": any(creative in input_lower for creative in ["story", "poem", "imagine"])
        }
        
        # Return the first detected pattern
        for pattern_name, detected in patterns.items():
            if detected:
                return pattern_name
        
        return None
    
    async def _log_pattern_milestone(self, pattern: str, count: int):
        """Log when a pattern reaches a milestone"""
        await self.original_logger.log_thought(
            title=f"Pattern Milestone: {pattern}",
            content=f"The '{pattern}' pattern has occurred {count} times",
            metadata={
                "pattern": pattern,
                "count": count,
                "milestone": True,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _analyze_log_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in the log buffer"""
        if not self.log_buffer:
            return {"patterns_found": False}
        
        # Analyze module activity
        module_activity = {}
        for entry in self.log_buffer:
            module = entry["source"]
            module_activity[module] = module_activity.get(module, 0) + 1
        
        # Find most active module
        most_active = max(module_activity.items(), key=lambda x: x[1]) if module_activity else None
        
        # Analyze update type distribution
        type_distribution = {}
        for entry in self.log_buffer:
            update_type = entry["type"]
            type_distribution[update_type] = type_distribution.get(update_type, 0) + 1
        
        return {
            "patterns_found": True,
            "module_activity": module_activity,
            "most_active_module": most_active,
            "type_distribution": type_distribution,
            "buffer_size": len(self.log_buffer)
        }
    
    async def _identify_emerging_themes(self, messages: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Identify emerging themes from cross-module messages"""
        themes = []
        
        # Count theme indicators
        theme_counts = {
            "learning": 0,
            "emotional_processing": 0,
            "goal_pursuit": 0,
            "creative_expression": 0,
            "system_evolution": 0
        }
        
        for module, module_messages in messages.items():
            for msg in module_messages:
                msg_type = msg.get("type", "").lower()
                
                if "learn" in msg_type or "knowledge" in msg_type:
                    theme_counts["learning"] += 1
                elif "emotion" in msg_type or "feeling" in msg_type:
                    theme_counts["emotional_processing"] += 1
                elif "goal" in msg_type:
                    theme_counts["goal_pursuit"] += 1
                elif "creative" in msg_type or "content" in msg_type:
                    theme_counts["creative_expression"] += 1
                elif "evolution" in msg_type or "capability" in msg_type:
                    theme_counts["system_evolution"] += 1
        
        # Identify significant themes
        for theme, count in theme_counts.items():
            if count >= 3:  # Threshold for emerging theme
                themes.append(theme)
        
        return themes
    
    async def _analyze_cross_module_correlations(self, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze correlations between module activities"""
        correlations = {}
        
        # Look for temporal correlations
        # (Simplified - in practice this would be more sophisticated)
        module_pairs = []
        modules = list(messages.keys())
        
        for i in range(len(modules)):
            for j in range(i + 1, len(modules)):
                module_pairs.append((modules[i], modules[j]))
        
        # Check each pair for correlation
        for module1, module2 in module_pairs:
            if messages[module1] and messages[module2]:
                # Simple correlation: do they often update together?
                correlation_key = f"{module1}<->{module2}"
                correlations[correlation_key] = {
                    "frequency": min(len(messages[module1]), len(messages[module2])),
                    "type": "temporal"
                }
        
        return correlations
    
    async def _generate_log_insights(self, patterns: Dict[str, Any], themes: List[str], correlations: Dict[str, Any]) -> List[str]:
        """Generate insights from log analysis"""
        insights = []
        
        # Pattern-based insights
        if patterns.get("most_active_module"):
            module, count = patterns["most_active_module"]
            insights.append(f"Module '{module}' is currently most active with {count} events")
        
        # Theme-based insights
        if themes:
            insights.append(f"Emerging themes: {', '.join(themes)}")
        
        # Correlation-based insights
        strong_correlations = [k for k, v in correlations.items() if v["frequency"] > 5]
        if strong_correlations:
            insights.append(f"Strong module correlations: {', '.join(strong_correlations)}")
        
        return insights
    
    async def _generate_logging_recommendations(self, context: SharedContext) -> List[str]:
        """Generate recommendations for logging improvements"""
        recommendations = []
        
        # Check log buffer patterns
        patterns = self._analyze_log_patterns()
        
        if patterns.get("buffer_size", 0) > 90:
            recommendations.append("Consider increasing log buffer size or implementing log rotation")
        
        # Check for underrepresented modules
        if patterns.get("module_activity"):
            activity = patterns["module_activity"]
            avg_activity = sum(activity.values()) / len(activity) if activity else 0
            
            quiet_modules = [m for m, count in activity.items() if count < avg_activity * 0.5]
            if quiet_modules:
                recommendations.append(f"Modules with low activity: {', '.join(quiet_modules)}")
        
        return recommendations
    
    def _get_significant_patterns(self) -> List[Dict[str, Any]]:
        """Get the most significant patterns detected"""
        significant = []
        
        # Add high-frequency patterns
        for pattern, count in self.log_patterns.items():
            if count >= 10:
                significant.append({
                    "pattern": pattern,
                    "frequency": count,
                    "significance": "high" if count > 20 else "medium"
                })
        
        return significant
    
    async def _assess_system_health_from_logs(self) -> Dict[str, Any]:
        """Assess system health based on log patterns"""
        health_indicators = {
            "error_rate": 0.0,
            "module_balance": 0.0,
            "pattern_stability": 0.0,
            "overall_health": "good"
        }
        
        if self.log_buffer:
            # Calculate error rate
            error_count = sum(1 for entry in self.log_buffer if "error" in entry["type"].lower())
            health_indicators["error_rate"] = error_count / len(self.log_buffer)
            
            # Calculate module balance
            patterns = self._analyze_log_patterns()
            if patterns.get("module_activity"):
                activity = patterns["module_activity"]
                if activity:
                    # Check if activity is well-distributed
                    avg = sum(activity.values()) / len(activity)
                    variance = sum((count - avg) ** 2 for count in activity.values()) / len(activity)
                    health_indicators["module_balance"] = 1.0 - min(1.0, variance / (avg ** 2) if avg > 0 else 1.0)
            
            # Assess overall health
            if health_indicators["error_rate"] > 0.1:
                health_indicators["overall_health"] = "poor"
            elif health_indicators["error_rate"] > 0.05:
                health_indicators["overall_health"] = "fair"
            elif health_indicators["module_balance"] < 0.5:
                health_indicators["overall_health"] = "fair"
        
        return health_indicators
    
    async def _suggest_investigations(self, context: SharedContext) -> List[str]:
        """Suggest areas for investigation based on logs"""
        suggestions = []
        
        # Check for anomalies
        if self.log_patterns.get("error", 0) > 5:
            suggestions.append("Investigate recurring errors in system")
        
        # Check for underutilized capabilities
        patterns = self._analyze_log_patterns()
        if patterns.get("type_distribution"):
            distribution = patterns["type_distribution"]
            
            # Look for missing expected types
            expected_types = ["goal_progress", "memory_creation", "emotional_update"]
            missing = [t for t in expected_types if t not in distribution]
            
            if missing:
                suggestions.append(f"Investigate why these updates are missing: {', '.join(missing)}")
        
        return suggestions
    
    def _should_create_meta_log(self, context: SharedContext) -> bool:
        """Determine if we should create a meta-log entry"""
        # Create meta-log every 50 entries or on significant patterns
        return len(self.log_buffer) >= 50 or len(self._get_significant_patterns()) > 3
    
    async def _create_meta_log(self, synthesis: Dict[str, Any]):
        """Create a meta-log summarizing system state"""
        await self.original_logger.log_evolution_suggestion(
            title="System Meta-Log",
            content=f"""System behavior analysis:
            
Significant Patterns: {len(synthesis['significant_patterns'])}
System Health: {synthesis['system_health_indicators']['overall_health']}
Recommendations: {', '.join(synthesis['logging_recommendations'])}

This meta-log represents a synthesis of system behavior patterns and health indicators.""",
            metadata={
                "type": "meta_log",
                "synthesis_data": synthesis,
                "timestamp": datetime.now().isoformat(),
                "log_buffer_size": len(self.log_buffer)
            }
        )
    
    def _analyze_buffer_patterns(self) -> Optional[str]:
        """Analyze the log buffer for patterns"""
        if len(self.log_buffer) < 10:
            return None
        
        # Look for repetitive sequences
        recent = self.log_buffer[-10:]
        
        # Check for module sequences
        module_sequence = [entry["source"] for entry in recent]
        
        # Simple pattern: same module repeated
        if len(set(module_sequence[-3:])) == 1:
            return f"Repeated activity from {module_sequence[-1]}"
        
        # Check for ping-pong pattern
        if len(set(module_sequence[-4:])) == 2 and module_sequence[-1] == module_sequence[-3] and module_sequence[-2] == module_sequence[-4]:
            return f"Ping-pong pattern between {module_sequence[-1]} and {module_sequence[-2]}"
        
        return None
    
    # Delegate all other methods to the original logger
    def __getattr__(self, name):
        """Delegate any missing methods to the original logger"""
        return getattr(self.original_logger, name)
