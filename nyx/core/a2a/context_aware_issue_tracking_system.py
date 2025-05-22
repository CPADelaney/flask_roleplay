# nyx/core/a2a/context_aware_issue_tracking_system.py

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareIssueTrackingSystem(ContextAwareModule):
    """
    Enhanced IssueTrackingSystem with full context distribution capabilities
    """
    
    def __init__(self, original_issue_tracker):
        super().__init__("issue_tracker")
        self.original_tracker = original_issue_tracker
        self.context_subscriptions = [
            "error_detected", "quality_issue_detected", "performance_issue",
            "missing_capability_identified", "improvement_suggestion", "user_feedback_negative",
            "module_conflict_detected", "goal_failure", "synthesis_quality_issues",
            "thought_loop_detected", "coherence_issue", "efficiency_bottleneck"
        ]
        
        # Advanced issue tracking
        self.context_issues_map = {}
        self.cross_module_patterns = {}
        self.issue_correlation_tracking = {}
        self.automatic_issue_detection = True
        
    async def on_context_received(self, context: SharedContext):
        """Initialize issue tracking for this context"""
        logger.debug(f"IssueTracker received context for processing stage: {context.processing_stage}")
        
        # Analyze context for potential issues
        initial_analysis = await self._analyze_context_for_issues(context)
        
        # Send issue tracking readiness
        await self.send_context_update(
            update_type="issue_tracking_active",
            data={
                "tracking_enabled": True,
                "auto_detection": self.automatic_issue_detection,
                "initial_issues_detected": len(initial_analysis),
                "monitoring_categories": list(self.original_tracker.db.categories.keys())
            },
            priority=ContextPriority.LOW
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates that may indicate issues"""
        
        issue_created = False
        
        if update.update_type == "error_detected":
            # Track error as issue
            issue_created = await self._track_error_issue(update.data, update.source_module)
        
        elif update.update_type == "quality_issue_detected":
            # Track quality issue
            issue_created = await self._track_quality_issue(update.data, update.source_module)
        
        elif update.update_type == "performance_issue":
            # Track performance issue
            issue_created = await self._track_performance_issue(update.data, update.source_module)
        
        elif update.update_type == "missing_capability_identified":
            # Track missing capability
            issue_created = await self._track_missing_capability(update.data, update.source_module)
        
        elif update.update_type == "improvement_suggestion":
            # Track improvement suggestion
            issue_created = await self._track_improvement_suggestion(update.data, update.source_module)
        
        elif update.update_type == "module_conflict_detected":
            # Track module conflict
            issue_created = await self._track_module_conflict(update.data, update.source_module)
        
        elif update.update_type == "goal_failure":
            # Track goal failure pattern
            issue_created = await self._track_goal_failure(update.data, update.source_module)
        
        elif update.update_type == "synthesis_quality_issues":
            # Track synthesis quality problems
            issue_created = await self._track_synthesis_issue(update.data, update.source_module)
        
        elif update.update_type == "thought_loop_detected":
            # Track thought loop issue
            issue_created = await self._track_thought_loop(update.data, update.source_module)
        
        elif update.update_type == "coherence_issue":
            # Track coherence problem
            issue_created = await self._track_coherence_issue(update.data, update.source_module)
        
        elif update.update_type == "efficiency_bottleneck":
            # Track efficiency bottleneck
            issue_created = await self._track_efficiency_issue(update.data, update.source_module)
        
        # If issue was created, notify relevant modules
        if issue_created:
            await self.send_context_update(
                update_type="new_issue_tracked",
                data={
                    "issue_type": update.update_type,
                    "source_module": update.source_module,
                    "severity": self._assess_issue_severity(update.data),
                    "timestamp": datetime.now().isoformat()
                },
                priority=ContextPriority.LOW
            )
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input stage - minimal issue tracking activity"""
        # Just monitor for any immediate issues in input
        input_issues = await self._check_input_issues(context)
        
        return {
            "input_issues_detected": len(input_issues),
            "monitoring_active": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze patterns across tracked issues"""
        # Get all recent issues
        recent_issues = self._get_recent_context_issues()
        
        # Analyze patterns
        pattern_analysis = await self._analyze_issue_patterns(recent_issues)
        
        # Analyze cross-module correlations
        correlation_analysis = await self._analyze_cross_module_correlations(recent_issues)
        
        # Get recommendations
        recommendations = await self._generate_issue_recommendations(
            pattern_analysis, correlation_analysis
        )
        
        # Check for critical patterns
        critical_patterns = await self._identify_critical_patterns(pattern_analysis)
        
        return {
            "total_recent_issues": len(recent_issues),
            "pattern_analysis": pattern_analysis,
            "correlation_analysis": correlation_analysis,
            "recommendations": recommendations,
            "critical_patterns": critical_patterns,
            "db_stats": self.original_tracker.db.get_stats().dict()
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize issue tracking insights"""
        # Get issue summary
        summary = await self.original_tracker.get_issue_summary(detailed=False)
        
        # Get priority issues that need attention
        priority_issues = await self._get_priority_issues()
        
        # Generate development insights
        dev_insights = await self._generate_development_insights()
        
        # Send critical issues alert if needed
        if priority_issues and len(priority_issues) > 3:
            await self.send_context_update(
                update_type="critical_issues_alert",
                data={
                    "priority_issue_count": len(priority_issues),
                    "top_issues": priority_issues[:3],
                    "action_required": "review_and_prioritize"
                },
                priority=ContextPriority.HIGH
            )
        
        return {
            "issue_summary": summary,
            "priority_issues": priority_issues,
            "development_insights": dev_insights,
            "synthesis_complete": True
        }
    
    # ========================================================================================
    # ADVANCED HELPER METHODS
    # ========================================================================================
    
    async def _analyze_context_for_issues(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Analyze initial context for potential issues"""
        issues = []
        
        # Check for missing critical context
        if not context.emotional_state:
            issues.append({
                "type": "missing_context",
                "component": "emotional_state",
                "severity": "low"
            })
        
        if not context.relationship_context:
            issues.append({
                "type": "missing_context",
                "component": "relationship_context",
                "severity": "low"
            })
        
        # Check for processing stage issues
        if context.processing_stage not in ["input", "analysis", "synthesis"]:
            issues.append({
                "type": "invalid_processing_stage",
                "stage": context.processing_stage,
                "severity": "medium"
            })
        
        # Check active modules
        if len(context.active_modules) < 3:
            issues.append({
                "type": "low_module_activation",
                "active_count": len(context.active_modules),
                "severity": "low"
            })
        
        return issues
    
    async def _track_error_issue(self, error_data: Dict[str, Any], source_module: str) -> bool:
        """Track an error as an issue"""
        error_type = error_data.get("error_type", "unknown")
        error_message = error_data.get("message", "No message provided")
        severity = error_data.get("severity", "medium")
        
        # Create issue description
        description = f"Error in {source_module}: {error_message}\n\n"
        description += f"Error Type: {error_type}\n"
        description += f"Severity: {severity}\n"
        
        if "stack_trace" in error_data:
            description += f"\nStack Trace:\n{error_data['stack_trace'][:500]}..."
        
        # Determine category
        category = "bug" if severity in ["high", "critical"] else "efficiency"
        
        # Create issue
        result = await self.original_tracker.add_issue_directly(
            title=f"{source_module} - {error_type}",
            description=description,
            category=category,
            priority=5 if severity == "critical" else 4 if severity == "high" else 3,
            tags=[source_module, "error", error_type, f"severity_{severity}"]
        )
        
        # Track in context map
        self._add_to_context_map(result.issue_id, "error", error_data, source_module)
        
        return result.success
    
    async def _track_quality_issue(self, quality_data: Dict[str, Any], source_module: str) -> bool:
        """Track a quality issue"""
        quality_score = quality_data.get("quality_score", 0.0)
        quality_type = quality_data.get("quality_type", "general")
        details = quality_data.get("details", {})
        
        # Create issue description
        description = f"Quality issue detected in {source_module}\n\n"
        description += f"Quality Score: {quality_score:.2f}\n"
        description += f"Quality Type: {quality_type}\n"
        
        if "suggestions" in quality_data:
            description += "\nSuggestions:\n"
            for suggestion in quality_data["suggestions"][:5]:
                description += f"- {suggestion}\n"
        
        # Add details
        if details:
            description += f"\nDetails:\n{str(details)[:300]}..."
        
        # Create issue
        result = await self.original_tracker.add_issue_directly(
            title=f"Quality Issue - {quality_type} in {source_module}",
            description=description,
            category="efficiency",
            priority=4 if quality_score < 0.3 else 3,
            tags=[source_module, "quality", quality_type, f"score_{int(quality_score*10)}"]
        )
        
        # Track in context map
        self._add_to_context_map(result.issue_id, "quality", quality_data, source_module)
        
        return result.success
    
    async def _track_performance_issue(self, perf_data: Dict[str, Any], source_module: str) -> bool:
        """Track a performance issue"""
        metric_name = perf_data.get("metric", "unknown")
        current_value = perf_data.get("current_value", 0.0)
        expected_value = perf_data.get("expected_value", 1.0)
        impact = perf_data.get("impact", "medium")
        
        # Create issue description
        description = f"Performance issue in {source_module}\n\n"
        description += f"Metric: {metric_name}\n"
        description += f"Current Value: {current_value:.3f}\n"
        description += f"Expected Value: {expected_value:.3f}\n"
        description += f"Impact: {impact}\n"
        
        if "bottleneck" in perf_data:
            description += f"\nBottleneck: {perf_data['bottleneck']}\n"
        
        # Create issue
        result = await self.original_tracker.add_issue_directly(
            title=f"Performance - {metric_name} in {source_module}",
            description=description,
            category="efficiency",
            priority=4 if impact == "high" else 3,
            tags=[source_module, "performance", metric_name, f"impact_{impact}"]
        )
        
        # Track in context map
        self._add_to_context_map(result.issue_id, "performance", perf_data, source_module)
        
        return result.success
    
    async def _track_missing_capability(self, capability_data: Dict[str, Any], source_module: str) -> bool:
        """Track a missing capability"""
        capability = capability_data.get("capability_description", "Unknown capability")
        required_for = capability_data.get("required_for", "Unknown purpose")
        priority_level = capability_data.get("priority", "medium")
        
        # Create issue description
        description = f"Missing capability identified by {source_module}\n\n"
        description += f"Capability: {capability}\n"
        description += f"Required For: {required_for}\n"
        description += f"Priority: {priority_level}\n"
        
        if "use_cases" in capability_data:
            description += "\nUse Cases:\n"
            for use_case in capability_data["use_cases"][:3]:
                description += f"- {use_case}\n"
        
        if "suggested_implementation" in capability_data:
            description += f"\nSuggested Implementation:\n{capability_data['suggested_implementation'][:500]}..."
        
        # Create issue
        result = await self.original_tracker.add_issue_directly(
            title=f"Missing Capability - {capability[:50]}...",
            description=description,
            category="enhancement",
            priority=5 if priority_level == "critical" else 4 if priority_level == "high" else 3,
            tags=[source_module, "missing_capability", "enhancement", f"priority_{priority_level}"]
        )
        
        # Track in context map
        self._add_to_context_map(result.issue_id, "capability", capability_data, source_module)
        
        return result.success
    
    async def _track_improvement_suggestion(self, suggestion_data: Dict[str, Any], source_module: str) -> bool:
        """Track an improvement suggestion"""
        suggestion = suggestion_data.get("suggestion", "No suggestion text")
        area = suggestion_data.get("area", "general")
        expected_benefit = suggestion_data.get("expected_benefit", "improvement")
        
        # Create issue description
        description = f"Improvement suggestion from {source_module}\n\n"
        description += f"Suggestion: {suggestion}\n"
        description += f"Area: {area}\n"
        description += f"Expected Benefit: {expected_benefit}\n"
        
        if "implementation_notes" in suggestion_data:
            description += f"\nImplementation Notes:\n{suggestion_data['implementation_notes']}"
        
        # Create issue
        result = await self.original_tracker.add_issue_directly(
            title=f"Suggestion - {area} - {suggestion[:40]}...",
            description=description,
            category="enhancement",
            priority=3,
            tags=[source_module, "suggestion", area, "improvement"]
        )
        
        # Track in context map
        self._add_to_context_map(result.issue_id, "suggestion", suggestion_data, source_module)
        
        return result.success
    
    async def _track_module_conflict(self, conflict_data: Dict[str, Any], source_module: str) -> bool:
        """Track a module conflict issue"""
        conflicting_modules = conflict_data.get("modules", [])
        conflict_type = conflict_data.get("conflict_type", "unknown")
        severity = conflict_data.get("severity", "medium")
        
        # Create issue description
        description = f"Module conflict detected by {source_module}\n\n"
        description += f"Conflicting Modules: {', '.join(conflicting_modules)}\n"
        description += f"Conflict Type: {conflict_type}\n"
        description += f"Severity: {severity}\n"
        
        if "details" in conflict_data:
            description += f"\nConflict Details:\n{conflict_data['details']}"
        
        if "resolution_suggestions" in conflict_data:
            description += "\n\nResolution Suggestions:\n"
            for suggestion in conflict_data["resolution_suggestions"]:
                description += f"- {suggestion}\n"
        
        # Create issue
        result = await self.original_tracker.add_issue_directly(
            title=f"Module Conflict - {' vs '.join(conflicting_modules[:2])}",
            description=description,
            category="bug" if severity == "high" else "efficiency",
            priority=4 if severity == "high" else 3,
            tags=[source_module, "module_conflict", conflict_type] + conflicting_modules
        )
        
        # Track conflict pattern
        self._track_conflict_pattern(conflicting_modules, conflict_type)
        
        # Track in context map
        self._add_to_context_map(result.issue_id, "conflict", conflict_data, source_module)
        
        return result.success
    
    async def _track_goal_failure(self, failure_data: Dict[str, Any], source_module: str) -> bool:
        """Track a goal failure pattern"""
        goal_id = failure_data.get("goal_id", "unknown")
        goal_description = failure_data.get("goal_description", "Unknown goal")
        failure_reason = failure_data.get("failure_reason", "Unknown reason")
        failure_count = failure_data.get("failure_count", 1)
        
        # Create issue description
        description = f"Goal failure pattern detected by {source_module}\n\n"
        description += f"Goal: {goal_description}\n"
        description += f"Goal ID: {goal_id}\n"
        description += f"Failure Reason: {failure_reason}\n"
        description += f"Failure Count: {failure_count}\n"
        
        if "context" in failure_data:
            description += f"\nContext:\n{str(failure_data['context'])[:300]}..."
        
        # Create issue
        result = await self.original_tracker.add_issue_directly(
            title=f"Goal Failure - {goal_description[:40]}...",
            description=description,
            category="efficiency",
            priority=4 if failure_count > 3 else 3,
            tags=[source_module, "goal_failure", "pattern", f"failures_{failure_count}"]
        )
        
        # Track in context map
        self._add_to_context_map(result.issue_id, "goal_failure", failure_data, source_module)
        
        return result.success
    
    async def _track_synthesis_issue(self, synthesis_data: Dict[str, Any], source_module: str) -> bool:
        """Track synthesis quality issue"""
        quality_score = synthesis_data.get("overall_score", 0.0)
        weak_areas = synthesis_data.get("weak_areas", [])
        critical = synthesis_data.get("critical", False)
        
        # Create issue description
        description = f"Synthesis quality issue detected by {source_module}\n\n"
        description += f"Overall Quality Score: {quality_score:.2f}\n"
        description += f"Critical Issue: {'Yes' if critical else 'No'}\n"
        
        if weak_areas:
            description += "\nWeak Areas:\n"
            for area in weak_areas:
                description += f"- {area}\n"
        
        if "suggestions" in synthesis_data:
            description += "\nImprovement Suggestions:\n"
            for suggestion in synthesis_data["suggestions"][:5]:
                description += f"- {suggestion}\n"
        
        # Create issue
        result = await self.original_tracker.add_issue_directly(
            title=f"Synthesis Quality - Score {quality_score:.1f}",
            description=description,
            category="quality" if critical else "efficiency",
            priority=5 if critical else 4 if quality_score < 0.5 else 3,
            tags=[source_module, "synthesis", "quality", f"score_{int(quality_score*10)}"]
        )
        
        # Track in context map
        self._add_to_context_map(result.issue_id, "synthesis", synthesis_data, source_module)
        
        return result.success
    
    async def _track_thought_loop(self, loop_data: Dict[str, Any], source_module: str) -> bool:
        """Track thought loop issue"""
        repetition_score = loop_data.get("repetition_score", 0.0)
        loop_type = loop_data.get("loop_type", "unknown")
        thought_count = loop_data.get("thought_count", 0)
        
        # Create issue description
        description = f"Thought loop detected by {source_module}\n\n"
        description += f"Loop Type: {loop_type}\n"
        description += f"Repetition Score: {repetition_score:.2f}\n"
        description += f"Affected Thoughts: {thought_count}\n"
        
        if "loop_pattern" in loop_data:
            description += f"\nLoop Pattern:\n{loop_data['loop_pattern'][:300]}..."
        
        # Create issue
        result = await self.original_tracker.add_issue_directly(
            title=f"Thought Loop - {loop_type}",
            description=description,
            category="efficiency",
            priority=3,
            tags=[source_module, "thought_loop", loop_type, "cognitive_pattern"]
        )
        
        # Track in context map
        self._add_to_context_map(result.issue_id, "thought_loop", loop_data, source_module)
        
        return result.success
    
    async def _track_coherence_issue(self, coherence_data: Dict[str, Any], source_module: str) -> bool:
        """Track coherence issue between modules"""
        coherence_score = coherence_data.get("coherence_score", 0.0)
        affected_modules = coherence_data.get("affected_modules", [])
        issue_type = coherence_data.get("issue_type", "general")
        
        # Create issue description
        description = f"Coherence issue detected by {source_module}\n\n"
        description += f"Coherence Score: {coherence_score:.2f}\n"
        description += f"Issue Type: {issue_type}\n"
        description += f"Affected Modules: {', '.join(affected_modules)}\n"
        
        if "conflicts" in coherence_data:
            description += "\nSpecific Conflicts:\n"
            for conflict in coherence_data["conflicts"][:3]:
                description += f"- {conflict}\n"
        
        # Create issue
        result = await self.original_tracker.add_issue_directly(
            title=f"Coherence Issue - {issue_type}",
            description=description,
            category="efficiency",
            priority=4 if coherence_score < 0.3 else 3,
            tags=[source_module, "coherence", issue_type] + affected_modules
        )
        
        # Track in context map
        self._add_to_context_map(result.issue_id, "coherence", coherence_data, source_module)
        
        return result.success
    
    async def _track_efficiency_issue(self, efficiency_data: Dict[str, Any], source_module: str) -> bool:
        """Track efficiency bottleneck"""
        bottleneck_type = efficiency_data.get("bottleneck_type", "unknown")
        impact = efficiency_data.get("impact", "medium")
        affected_operations = efficiency_data.get("affected_operations", [])
        
        # Create issue description
        description = f"Efficiency bottleneck detected by {source_module}\n\n"
        description += f"Bottleneck Type: {bottleneck_type}\n"
        description += f"Impact: {impact}\n"
        
        if affected_operations:
            description += "\nAffected Operations:\n"
            for op in affected_operations[:5]:
                description += f"- {op}\n"
        
        if "metrics" in efficiency_data:
            description += f"\nMetrics:\n{str(efficiency_data['metrics'])[:300]}..."
        
        # Create issue
        result = await self.original_tracker.add_issue_directly(
            title=f"Efficiency Bottleneck - {bottleneck_type}",
            description=description,
            category="efficiency",
            priority=4 if impact == "high" else 3,
            tags=[source_module, "efficiency", "bottleneck", bottleneck_type, f"impact_{impact}"]
        )
        
        # Track in context map
        self._add_to_context_map(result.issue_id, "efficiency", efficiency_data, source_module)
        
        return result.success
    
    def _assess_issue_severity(self, issue_data: Dict[str, Any]) -> str:
        """Assess severity of an issue"""
        # Check for explicit severity
        if "severity" in issue_data:
            return issue_data["severity"]
        
        # Check for critical indicators
        if any(key in issue_data for key in ["critical", "error", "failure"]):
            if issue_data.get("critical") or issue_data.get("error_type") == "fatal":
                return "critical"
        
        # Check scores
        if "score" in issue_data or "quality_score" in issue_data:
            score = issue_data.get("score", issue_data.get("quality_score", 1.0))
            if score < 0.3:
                return "high"
            elif score < 0.5:
                return "medium"
        
        # Check impact
        if issue_data.get("impact") == "high":
            return "high"
        
        return "medium"
    
    async def _check_input_issues(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Check for issues in input processing"""
        issues = []
        
        # Check input length
        if context.user_input:
            word_count = len(context.user_input.split())
            if word_count > 500:
                issues.append({
                    "type": "excessive_input_length",
                    "word_count": word_count,
                    "severity": "low"
                })
        
        # Check for missing user context
        if not context.user_id:
            issues.append({
                "type": "missing_user_id",
                "severity": "medium"
            })
        
        return issues
    
    def _get_recent_context_issues(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get issues tracked in recent context"""
        recent_issues = []
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        for issue_id, issue_data in self.context_issues_map.items():
            if issue_data["timestamp"] > cutoff_time:
                recent_issues.append({
                    "issue_id": issue_id,
                    "type": issue_data["type"],
                    "module": issue_data["module"],
                    "data": issue_data["data"],
                    "timestamp": issue_data["timestamp"]
                })
        
        return recent_issues
    
    def _add_to_context_map(self, issue_id: str, issue_type: str, data: Dict[str, Any], module: str):
        """Add issue to context tracking map"""
        if issue_id:
            self.context_issues_map[issue_id] = {
                "type": issue_type,
                "data": data,
                "module": module,
                "timestamp": datetime.now().timestamp()
            }
    
    def _track_conflict_pattern(self, modules: List[str], conflict_type: str):
        """Track patterns of module conflicts"""
        # Create conflict key
        conflict_key = tuple(sorted(modules))
        
        if conflict_key not in self.cross_module_patterns:
            self.cross_module_patterns[conflict_key] = {
                "count": 0,
                "types": {},
                "first_seen": datetime.now().isoformat(),
                "last_seen": None
            }
        
        pattern = self.cross_module_patterns[conflict_key]
        pattern["count"] += 1
        pattern["last_seen"] = datetime.now().isoformat()
        
        if conflict_type not in pattern["types"]:
            pattern["types"][conflict_type] = 0
        pattern["types"][conflict_type] += 1
    
    async def _analyze_issue_patterns(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in tracked issues"""
        if not issues:
            return {"pattern_count": 0, "patterns": []}
        
        # Count by type
        type_counts = {}
        for issue in issues:
            issue_type = issue["type"]
            type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
        
        # Count by module
        module_counts = {}
        for issue in issues:
            module = issue["module"]
            module_counts[module] = module_counts.get(module, 0) + 1
        
        # Identify patterns
        patterns = []
        
        # Recurring issue pattern
        for issue_type, count in type_counts.items():
            if count >= 3:
                patterns.append({
                    "pattern": "recurring_issue",
                    "issue_type": issue_type,
                    "count": count,
                    "severity": "high" if count > 5 else "medium"
                })
        
        # Module-specific pattern
        for module, count in module_counts.items():
            if count >= 5:
                patterns.append({
                    "pattern": "module_issues",
                    "module": module,
                    "count": count,
                    "severity": "high" if count > 10 else "medium"
                })
        
        # Time-based patterns (simplified)
        recent_count = len([i for i in issues if i["timestamp"] > datetime.now().timestamp() - 3600])
        if recent_count > 5:
            patterns.append({
                "pattern": "issue_spike",
                "recent_count": recent_count,
                "timeframe": "last_hour",
                "severity": "high"
            })
        
        return {
            "pattern_count": len(patterns),
            "patterns": patterns,
            "type_distribution": type_counts,
            "module_distribution": module_counts
        }
    
    async def _analyze_cross_module_correlations(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlations between issues across modules"""
        correlations = []
        
        # Check for module pairs with frequent issues
        module_pairs = {}
        
        # Build temporal correlation map
        time_window = 300  # 5 minutes
        for i, issue1 in enumerate(issues):
            for issue2 in issues[i+1:]:
                time_diff = abs(issue1["timestamp"] - issue2["timestamp"])
                if time_diff < time_window:
                    pair = tuple(sorted([issue1["module"], issue2["module"]]))
                    if pair[0] != pair[1]:  # Different modules
                        if pair not in module_pairs:
                            module_pairs[pair] = 0
                        module_pairs[pair] += 1
        
        # Identify significant correlations
        for pair, count in module_pairs.items():
            if count >= 3:
                correlations.append({
                    "modules": list(pair),
                    "correlation_count": count,
                    "correlation_type": "temporal",
                    "strength": "high" if count > 5 else "medium"
                })
        
        # Check conflict patterns
        for modules, pattern in self.cross_module_patterns.items():
            if pattern["count"] >= 3:
                correlations.append({
                    "modules": list(modules),
                    "correlation_count": pattern["count"],
                    "correlation_type": "conflict",
                    "conflict_types": pattern["types"],
                    "strength": "high" if pattern["count"] > 5 else "medium"
                })
        
        return {
            "correlation_count": len(correlations),
            "correlations": correlations,
            "has_significant_correlations": len(correlations) > 0
        }
    
    async def _generate_issue_recommendations(self, patterns: Dict, correlations: Dict) -> List[str]:
        """Generate recommendations based on issue analysis"""
        recommendations = []
        
        # Pattern-based recommendations
        for pattern in patterns.get("patterns", []):
            if pattern["pattern"] == "recurring_issue":
                recommendations.append(
                    f"Address recurring {pattern['issue_type']} issues (count: {pattern['count']})"
                )
            elif pattern["pattern"] == "module_issues":
                recommendations.append(
                    f"Review and refactor {pattern['module']} module (issue count: {pattern['count']})"
                )
            elif pattern["pattern"] == "issue_spike":
                recommendations.append(
                    f"Investigate recent issue spike ({pattern['recent_count']} issues in {pattern['timeframe']})"
                )
        
        # Correlation-based recommendations
        for correlation in correlations.get("correlations", []):
            if correlation["correlation_type"] == "conflict":
                recommendations.append(
                    f"Resolve conflicts between {' and '.join(correlation['modules'])} modules"
                )
            elif correlation["correlation_type"] == "temporal":
                recommendations.append(
                    f"Investigate coupling between {' and '.join(correlation['modules'])} modules"
                )
        
        # General recommendations based on stats
        if not recommendations:
            recommendations.append("Continue monitoring for issue patterns")
        
        return recommendations[:5]  # Top 5 recommendations
    
    async def _identify_critical_patterns(self, pattern_analysis: Dict) -> List[Dict[str, Any]]:
        """Identify critical patterns that need immediate attention"""
        critical_patterns = []
        
        for pattern in pattern_analysis.get("patterns", []):
            if pattern.get("severity") == "high":
                critical_patterns.append({
                    "pattern": pattern,
                    "urgency": "immediate",
                    "impact": "system_stability"
                })
        
        # Check for cascading issues
        type_dist = pattern_analysis.get("type_distribution", {})
        if type_dist.get("error", 0) > 5 and type_dist.get("quality", 0) > 5:
            critical_patterns.append({
                "pattern": {
                    "pattern": "cascading_issues",
                    "error_count": type_dist.get("error", 0),
                    "quality_count": type_dist.get("quality", 0)
                },
                "urgency": "high",
                "impact": "overall_performance"
            })
        
        return critical_patterns
    
    async def _get_priority_issues(self) -> List[Dict[str, Any]]:
        """Get high priority issues that need attention"""
        all_issues = self.original_tracker.db.get_all_issues()
        
        # Filter for open, high-priority issues
        priority_issues = []
        for issue in all_issues:
            if issue.status == "open" and issue.priority >= 4:
                # Add context if available
                context_data = self.context_issues_map.get(issue.id, {})
                
                priority_issues.append({
                    "id": issue.id,
                    "title": issue.title,
                    "category": issue.category,
                    "priority": issue.priority,
                    "created_at": issue.created_at,
                    "tags": issue.tags,
                    "has_context": bool(context_data),
                    "module": context_data.get("module", "unknown") if context_data else "unknown"
                })
        
        # Sort by priority and recency
        priority_issues.sort(key=lambda x: (x["priority"], x["created_at"]), reverse=True)
        
        return priority_issues
    
    async def _generate_development_insights(self) -> Dict[str, Any]:
        """Generate insights for development team"""
        stats = self.original_tracker.db.get_stats()
        
        insights = {
            "total_open_issues": stats.open_issues,
            "critical_issues": len([i for i in self.original_tracker.db.get_all_issues() 
                                  if i.status == "open" and i.priority == 5]),
            "top_categories": sorted(stats.by_category.items(), key=lambda x: x[1], reverse=True)[:3],
            "recent_patterns": [],
            "recommendations": []
        }
        
        # Add recent pattern insights
        recent_issues = self._get_recent_context_issues(hours=48)
        if recent_issues:
            pattern_analysis = await self._analyze_issue_patterns(recent_issues)
            insights["recent_patterns"] = pattern_analysis.get("patterns", [])[:3]
        
        # Generate specific recommendations
        if insights["critical_issues"] > 0:
            insights["recommendations"].append(f"Address {insights['critical_issues']} critical issues immediately")
        
        top_category = insights["top_categories"][0][0] if insights["top_categories"] else None
        if top_category:
            insights["recommendations"].append(f"Focus on {top_category} issues (most common category)")
        
        # Check for stale issues
        stale_count = len([i for i in self.original_tracker.db.get_all_issues() 
                          if i.status == "open" and 
                          (datetime.now() - datetime.fromisoformat(i.updated_at)).days > 7])
        
        if stale_count > 5:
            insights["recommendations"].append(f"Review {stale_count} stale issues (>7 days old)")
        
        return insights
    
    # Delegate all other methods to the original tracker
    def __getattr__(self, name):
        """Delegate any missing methods to the original issue tracker"""
        return getattr(self.original_tracker, name)
