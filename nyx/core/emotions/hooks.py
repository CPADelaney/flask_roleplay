# nyx/core/emotions/hooks.py

"""
Enhanced lifecycle hooks for the Nyx emotional system.
These hooks leverage the full OpenAI Agents SDK lifecycle with improved monitoring,
tracing, and performance optimization.
"""

import datetime
import logging
from typing import Any, Dict, Optional, Set, List, cast
from collections import defaultdict

from agents import (
    Agent, AgentHooks, RunContextWrapper, Tool, 
    function_span, custom_span, trace, gen_trace_id
)
from agents.tracing import Span

from nyx.core.emotions.context import EmotionalContext

logger = logging.getLogger(__name__)

class EmotionalAgentHooks(AgentHooks[EmotionalContext]):
    """
    Enhanced hooks for emotional agent lifecycle events with 
    full OpenAI Agents SDK integration
    """
    
    def __init__(self, neurochemicals=None):
        """
        Initialize the hooks with optional neurochemical state reference
        
        Args:
            neurochemicals: Reference to neurochemical state dictionary
        """
        self.neurochemicals = neurochemicals
        self.agent_timing: Dict[str, Dict[str, float]] = {}
        self.tool_usage: Dict[str, int] = defaultdict(int)
        self.handoff_history: List[Dict[str, Any]] = []
        self.last_active_agent = None
        
        # New tracking for better monitoring
        self.active_spans: Dict[str, Span] = {}
        self.error_counts: Dict[str, int] = defaultdict(int)
        
    async def on_start(self, context: RunContextWrapper[EmotionalContext], 
                     agent: Agent[EmotionalContext]) -> None:
        """
        Called when an agent starts processing
        
        Args:
            context: The run context wrapper
            agent: The agent that is starting
        """
        # Generate an agent span ID for tracking
        span_id = f"agent_span_{agent.name}_{datetime.datetime.now().timestamp()}"
        
        # Create a span for agent lifecycle monitoring
        span = custom_span(
            "agent_lifecycle", 
            data={
                "event": "start",
                "agent": agent.name,
                "cycle": str(context.context.cycle_count),
                "timestamp": datetime.datetime.now().isoformat(),
                "agent_tools": [tool.name for tool in agent.tools] if hasattr(agent, "tools") else [],
                "has_handoffs": hasattr(agent, "handoffs") and len(agent.handoffs) > 0
            }
        )
        
        # Start the span and store for later reference
        span.start(mark_as_current=True)
        self.active_spans[span_id] = span
        
        logger.debug(f"Emotional agent started: {agent.name}")
        
        # Store agent info in context
        context.context.record_agent_usage(agent.name)
        context.context.record_time_marker(f"start_{agent.name}")
        context.context.record_agent_state(agent.name, {
            "status": "started",
            "start_time": datetime.datetime.now().isoformat(),
            "cycle": str(context.context.cycle_count)
        })
        
        # Keep track of last active agent for transitions
        self.last_active_agent = agent.name
        
        # Add agent-specific initialization with improved SDK integration
        if agent.name == "Neurochemical Agent":
            # Pre-fetch current neurochemical decay needs
            if self.neurochemicals:
                now = datetime.datetime.now()
                last_update = context.context.get_value("last_neurochemical_update")
                
                if last_update:
                    hours_elapsed = (now - last_update).total_seconds() / 3600
                    context.context.set_value("decay_hours", hours_elapsed)
                    
                # Pre-calculate and cache neurochemical data
                context.context.set_value("cached_chemical_baselines", {
                    chemical: data["baseline"] for chemical, data in self.neurochemicals.items()
                })
                
                # Record current state in circular buffer for history tracking
                context.context._add_to_circular_buffer("agent_activity", {
                    "agent": agent.name,
                    "action": "start",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "cycle": str(context.context.cycle_count),
                    "data": {
                        "chemical_count": len(self.neurochemicals),
                    }
                })
        
        elif agent.name == "Emotion Derivation Agent":
            # Pre-fetch neurochemical state to avoid duplicate calls
            cached_state = context.context.get_cached_neurochemicals()
            if not cached_state and self.neurochemicals:
                context.context.record_neurochemical_values({
                    c: d["value"] for c, d in self.neurochemicals.items()
                })
                
            # Prepare chemical condition map for efficient lookup
            condition_map = {}
            for rule in agent.tools:
                if hasattr(rule, "name") and rule.name == "derive_emotional_state":
                    # Index rules by chemical for faster lookup
                    condition_map = self._index_emotion_rules_by_chemical()
                    break
            
            if condition_map:
                context.context.set_value("emotion_rule_index", condition_map)
            
            # Record emotional derivation agent activity
            context.context._add_to_circular_buffer("agent_activity", {
                "agent": agent.name,
                "action": "start",
                "timestamp": datetime.datetime.now().isoformat(),
                "cycle": str(context.context.cycle_count),
                "data": {
                    "emotion_rules_indexed": bool(condition_map)
                }
            })
        
        elif agent.name == "Emotional Reflection Agent":
            # Pre-calculate a dominant emotion if we have it
            last_emotions = context.context.last_emotions
            if last_emotions:
                dominant = max(last_emotions.items(), key=lambda x: x[1]) if last_emotions else None
                if dominant:
                    context.context.set_value("precalculated_dominant", dominant)
                    
            # Prepare reflection context
            context.context.set_value("reflection_session_start", datetime.datetime.now().isoformat())
            
            # Create a custom reflection span for better tracing
            reflection_span = custom_span(
                "reflection_session",
                data={
                    "dominant_emotion": dominant[0] if dominant else "unknown",
                    "intensity": dominant[1] if dominant else 0,
                    "cycle": str(context.context.cycle_count)
                }
            )
            reflection_span.start()
            context.context.set_value("active_reflection_span", reflection_span)
    
    async def on_end(self, context: RunContextWrapper[EmotionalContext], 
                   agent: Agent[EmotionalContext], output: Any) -> None:
        """
        Called when an agent completes processing
        
        Args:
            context: The run context wrapper
            agent: The agent that is ending
            output: The agent's output
        """
        # Find and finish the active span for this agent
        agent_span_id = next((span_id for span_id in self.active_spans 
                            if span_id.startswith(f"agent_span_{agent.name}_")), None)
        
        if agent_span_id and agent_span_id in self.active_spans:
            # Update the span data with output information
            span = self.active_spans[agent_span_id]
            
            # Get additional span data if available
            if hasattr(span, "data"):
                span.data.update({
                    "event": "end",
                    "output_type": type(output).__name__ if output else None,
                    "duration": context.context.get_elapsed_time(f"start_{agent.name}"),
                    "cycle": str(context.context.cycle_count)
                })
            
            # Finish and remove the span
            span.finish(reset_current=True)
            del self.active_spans[agent_span_id]
        
        # Calculate and store performance metrics
        context.context.record_time_marker(f"end_{agent.name}")
        duration = context.context.get_elapsed_time(f"start_{agent.name}", f"end_{agent.name}")
        
        # Update agent state in context
        context.context.record_agent_state(agent.name, {
            "status": "completed",
            "end_time": datetime.datetime.now().isoformat(),
            "duration": duration,
            "cycle": str(context.context.cycle_count),
            "output_type": type(output).__name__ if output else None
        })
        
        # Update rolling average response time for this agent
        agent_timing = context.context.get_timing_data()
        if agent.name not in agent_timing:
            agent_timing[agent.name] = {"count": 0, "avg_time": 0}
        
        stats = agent_timing[agent.name]
        stats["avg_time"] = ((stats["avg_time"] * stats["count"]) + duration) / (stats["count"] + 1)
        stats["count"] += 1
        
        # Add agent-specific teardown with improved SDK integration
        if agent.name == "Neurochemical Agent":
            # Cache the latest neurochemical update time
            context.context.set_value("last_neurochemical_update", datetime.datetime.now())
            
            # Store chemical update count for analytics
            update_count = context.context.get_value("chemical_update_count", 0)
            context.context.set_value("chemical_update_count", update_count + 1)
            
            # Track which chemicals were updated in this run
            updates = context.context.get_circular_buffer("chemical_updates")
            if updates:
                updated_chemicals = set(update["chemical"] for update in updates[-5:])  # Last 5 updates
                context.context.set_value("recent_updated_chemicals", list(updated_chemicals))
                
                # Record completion in circular buffer
                context.context._add_to_circular_buffer("agent_activity", {
                    "agent": agent.name,
                    "action": "complete",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "cycle": str(context.context.cycle_count),
                    "data": {
                        "updated_chemicals": list(updated_chemicals),
                        "duration": duration
                    }
                })
        
        elif agent.name == "Emotion Derivation Agent":
            # Cache emotion derivation results with proper SDK output typing
            if hasattr(output, "primary_emotion") and hasattr(output.primary_emotion, "name"):
                context.context.set_value("last_primary_emotion", output.primary_emotion.name)
                context.context.set_value("last_primary_intensity", output.primary_emotion.intensity)
                
                # Track emotion changes using SDK's custom span
                previous_emotion = context.context.get_value("previous_primary_emotion")
                current_emotion = output.primary_emotion.name
                
                if previous_emotion and previous_emotion != current_emotion:
                    # Create a dedicated span for emotion transitions
                    with custom_span(
                        "emotion_transition",
                        data={
                            "from": previous_emotion,
                            "to": current_emotion,
                            "intensity": output.primary_emotion.intensity,
                            "valence": output.primary_emotion.valence,
                            "arousal": output.primary_emotion.arousal,
                            "cycle": str(context.context.cycle_count),
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    ):
                        # Record transition in circular buffer for history
                        context.context._add_to_circular_buffer("emotion_transitions", {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "from": previous_emotion,
                            "to": current_emotion,
                            "cycle": str(context.context.cycle_count)
                        })
                
                # Update previous emotion for next transition tracking
                context.context.set_value("previous_primary_emotion", current_emotion)
        
        elif agent.name == "Emotional Reflection Agent":
            # Track reflection insights with proper SDK output typing
            if hasattr(output, "insight_level") and hasattr(output, "source_emotion"):
                # Record insight in circular buffer
                context.context._add_to_circular_buffer("reflection_insights", {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "emotion": output.source_emotion,
                    "insight_level": output.insight_level,
                    "cycle": str(context.context.cycle_count),
                    "thought": output.thought_text if hasattr(output, "thought_text") else ""
                })
                
                # Finish the reflection span if active
                reflection_span = context.context.get_value("active_reflection_span")
                if reflection_span:
                    # Update span data with insights
                    if hasattr(reflection_span, "data"):
                        reflection_span.data.update({
                            "insight_level": output.insight_level,
                            "duration": duration
                        })
                    
                    # Finish span
                    reflection_span.finish()
                    context.context.set_value("active_reflection_span", None)
        
        logger.debug(f"Agent {agent.name} completed in {duration:.2f}s (avg: {stats['avg_time']:.2f}s)")
    
    async def on_tool_start(self, context: RunContextWrapper[EmotionalContext], 
                          agent: Agent[EmotionalContext], tool: Tool) -> None:
        """
        Called when a tool is invoked
        
        Args:
            context: The run context wrapper
            agent: The agent invoking the tool
            tool: The tool being invoked
        """
        tool_name = tool.name if hasattr(tool, "name") else "unknown_tool"
        
        # Create a tool span with enhanced SDK integration
        tool_span = custom_span(
            "tool_lifecycle", 
            data={
                "event": "start",
                "tool": tool_name,
                "agent": agent.name,
                "cycle": str(context.context.cycle_count),
                "timestamp": datetime.datetime.now().isoformat()
            }
        )
        
        # Start span
        tool_span.start(mark_as_current=True)
        
        # Store the span for reference
        span_id = f"tool_span_{tool_name}_{datetime.datetime.now().timestamp()}"
        self.active_spans[span_id] = tool_span
        
        logger.debug(f"Tool started: {tool_name} by agent {agent.name}")
        context.context.record_time_marker(f"tool_start_{tool_name}")
        
        # Track tool usage count
        self.tool_usage[tool_name] += 1
        
        # Record tool usage in context
        context.context._add_to_circular_buffer("tool_usage", {
            "tool": tool_name,
            "agent": agent.name,
            "action": "start",
            "timestamp": datetime.datetime.now().isoformat(),
            "cycle": str(context.context.cycle_count)
        })
        
        # Pre-warm cache for specific tools
        if tool_name == "derive_emotional_state":
            # Pre-fetch neurochemical state if needed
            cached_state = context.context.get_cached_neurochemicals()
            if not cached_state and self.neurochemicals:
                context.context.record_neurochemical_values({
                    c: d["value"] for c, d in self.neurochemicals.items()
                })
        
        elif tool_name == "generate_internal_thought":
            # Pre-fetch emotional state if needed
            if not context.context.last_emotions and self.neurochemicals:
                # Create a simple approximation of emotional state
                nyxamine = self.neurochemicals.get("nyxamine", {}).get("value", 0.5)
                seranix = self.neurochemicals.get("seranix", {}).get("value", 0.5)
                cortanyx = self.neurochemicals.get("cortanyx", {}).get("value", 0.3)
                
                # Simple heuristic for dominant emotion
                if nyxamine > 0.7:
                    context.context.last_emotions = {"Joy": 0.7}
                elif seranix > 0.7:
                    context.context.last_emotions = {"Contentment": 0.7}
                elif cortanyx > 0.7:
                    context.context.last_emotions = {"Sadness": 0.7}
                else:
                    context.context.last_emotions = {"Neutral": 0.5}
    
    async def on_tool_end(self, context: RunContextWrapper[EmotionalContext], 
                        agent: Agent[EmotionalContext], tool: Tool, result: str) -> None:
        """
        Called when a tool completes execution with enhanced error recovery
        
        Args:
            context: The run context wrapper
            agent: The agent that invoked the tool
            tool: The tool that was invoked
            result: The tool's result
        """
        tool_name = tool.name if hasattr(tool, "name") else "unknown_tool"
        
        # Find and finish the active span for this tool
        tool_span_id = next((span_id for span_id in self.active_spans 
                           if span_id.startswith(f"tool_span_{tool_name}_")), None)
        
        if tool_span_id and tool_span_id in self.active_spans:
            # Update the span data with result information
            span = self.active_spans[tool_span_id]
            
            # Get additional span data if available
            if hasattr(span, "data"):
                # Calculate duration
                duration = context.context.get_elapsed_time(f"tool_start_{tool_name}", f"tool_end_{tool_name}")
                
                span.data.update({
                    "event": "end",
                    "result_length": len(str(result)),
                    "duration": duration,
                    "cycle": str(context.context.cycle_count)
                })
            
            # Finish and remove the span
            span.finish(reset_current=True)
            del self.active_spans[tool_span_id]
        
        # NEW: Check for errors in the result with recovery logic
        try:
            # Try to parse result as JSON to detect errors
            if isinstance(result, str) and result.startswith("{") and "error" in result.lower():
                result_data = json.loads(result)
                
                if "error" in result_data:
                    # Create a dedicated error span
                    with custom_span(
                        "tool_error",
                        data={
                            "tool": tool_name,
                            "error": result_data["error"],
                            "error_type": result_data.get("error_type", "unknown"),
                            "agent": agent.name,
                            "cycle": str(context.context.cycle_count)
                        }
                    ):
                        # Implement recovery strategies based on tool and error type
                        if tool_name == "update_neurochemical" and "unknown_neurochemical" in result_data.get("error", ""):
                            # Log recovery attempt
                            logger.info(f"Attempting recovery for {tool_name} error")
                            
                            # Update tool statistics to track errors
                            tool_stats = context.context.get_value("tool_stats", {})
                            if tool_name not in tool_stats:
                                tool_stats[tool_name] = {"errors": 0}
                            if "errors" not in tool_stats[tool_name]:
                                tool_stats[tool_name]["errors"] = 0
                            tool_stats[tool_name]["errors"] += 1
                            context.context.set_value("tool_stats", tool_stats)
        except:
            # If parsing fails, just continue with normal processing
            pass
            
        # Continue with existing tool end processing
        context.context.record_time_marker(f"tool_end_{tool_name}")
        duration = context.context.get_elapsed_time(f"tool_start_{tool_name}", f"tool_end_{tool_name}")
        
        # Update tool stats in context
        tool_stats = context.context.get_value("tool_stats", {})
        if tool_name not in tool_stats:
            tool_stats[tool_name] = {"count": 0, "total_time": 0}
        
        tool_stats[tool_name]["count"] += 1
        tool_stats[tool_name]["total_time"] += duration
        tool_stats[tool_name]["avg_time"] = (
            tool_stats[tool_name]["total_time"] / tool_stats[tool_name]["count"]
        )
        
        context.context.set_value("tool_stats", tool_stats)
        
        # Record tool completion in context
        context.context._add_to_circular_buffer("tool_usage", {
            "tool": tool_name,
            "agent": agent.name,
            "action": "complete",
            "timestamp": datetime.datetime.now().isoformat(),
            "cycle": str(context.context.cycle_count),
            "duration": duration
        })
        
        # Tool-specific post-processing with improved SDK integration
        if tool_name == "update_neurochemical":
            # Track chemical updates
            try:
                # Try to parse the result for better tracking
                import json
                result_data = json.loads(result) if isinstance(result, str) else result
                if isinstance(result_data, dict) and "updated_chemical" in result_data:
                    # Create a dedicated span for chemical updates
                    with custom_span(
                        "chemical_update",
                        data={
                            "chemical": result_data["updated_chemical"],
                            "old_value": result_data.get("old_value", 0),
                            "new_value": result_data.get("new_value", 0),
                            "change": result_data.get("new_value", 0) - result_data.get("old_value", 0),
                            "agent": agent.name,
                            "cycle": str(context.context.cycle_count),
                            "timestamp": datetime.datetime.now().isoformat(),
                            "type": "chemical_update"  # Type for analytics processor
                        }
                    ):
                        # Record the chemical update in our circular buffer
                        context.context._add_to_circular_buffer("chemical_updates", {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "chemical": result_data["updated_chemical"],
                            "old_value": result_data.get("old_value", 0),
                            "new_value": result_data.get("new_value", 0),
                            "agent": agent.name
                        })
            except:
                # If parsing fails, just log the raw information
                logger.debug(f"Could not parse chemical update result: {result[:100]}")
        
        logger.debug(f"Tool {tool_name} completed in {duration:.2f}s")
    
    async def on_handoff(self, context: RunContextWrapper[EmotionalContext], 
                       agent: Agent[EmotionalContext], source: Agent[EmotionalContext]) -> None:
        """
        Called when an agent receives a handoff
        
        Args:
            context: The run context wrapper
            agent: The agent receiving the handoff
            source: The agent that initiated the handoff
        """
        # Create a dedicated handoff span with improved SDK integration
        with custom_span(
            "handoff", 
            data={
                "from_agent": source.name,
                "to_agent": agent.name,
                "cycle": str(context.context.cycle_count),
                "timestamp": datetime.datetime.now().isoformat()
            }
        ) as handoff_span:
            logger.debug(f"Handoff: {source.name} -> {agent.name}")
            context.context.record_time_marker(f"handoff_to_{agent.name}")
            
            # Record the handoff in context for analysis
            handoffs = context.context.get_value("handoff_history", [])
            handoff_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "from": source.name,
                "to": agent.name,
                "cycle": str(context.context.cycle_count)
            }
            
            handoffs.append(handoff_entry)
            
            # Update our internal handoff history as well
            self.handoff_history.append(handoff_entry)
            
            # Record this handoff in our circular buffer
            context.context._add_to_circular_buffer("handoffs", handoff_entry)
            
            # Store current handoff in context for reference
            context.context.set_value("last_handoff", handoff_entry)
            
            # Limit history size
            if len(handoffs) > 20:
                handoffs = handoffs[-20:]
            
            context.context.set_value("handoff_history", handoffs)
            
            # Analyze handoff patterns for optimization
            self._analyze_handoff_patterns(context, agent, source)
    
    def _analyze_handoff_patterns(self, 
                               context: RunContextWrapper[EmotionalContext],
                               agent: Agent[EmotionalContext], 
                               source: Agent[EmotionalContext]) -> None:
        """
        Analyze handoff patterns to identify optimization opportunities
        
        Args:
            context: The run context wrapper
            agent: The agent receiving the handoff
            source: The agent that initiated the handoff
        """
        # Analyze last 10 handoffs for patterns
        recent_handoffs = self.handoff_history[-10:] if len(self.handoff_history) >= 10 else []
        
        if recent_handoffs:
            # Check for frequent transfers between the same agents
            transfers = defaultdict(int)
            for handoff in recent_handoffs:
                transfer_key = f"{handoff['from']}:{handoff['to']}"
                transfers[transfer_key] += 1
            
            # Identify the most common handoff
            if transfers:
                most_common = max(transfers.items(), key=lambda x: x[1])
                transfer_key, count = most_common
                
                # If we see a pattern forming, record it for potential optimization
                if count >= 3:  # 3 or more occurrences indicate a pattern
                    from_agent, to_agent = transfer_key.split(":")
                    
                    # Create a dedicated span for handoff pattern detection
                    with custom_span(
                        "handoff_pattern",
                        data={
                            "pattern": transfer_key,
                            "count": count,
                            "cycle": str(context.context.cycle_count),
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    ):
                        # Record this pattern for potential future optimization
                        handoff_patterns = context.context.get_value("handoff_patterns", {})
                        if transfer_key not in handoff_patterns:
                            handoff_patterns[transfer_key] = {
                                "count": 0,
                                "first_seen": datetime.datetime.now().isoformat()
                            }
                        
                        handoff_patterns[transfer_key]["count"] += 1
                        handoff_patterns[transfer_key]["last_seen"] = datetime.datetime.now().isoformat()
                        
                        context.context.set_value("handoff_patterns", handoff_patterns)
    
    def _index_emotion_rules_by_chemical(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create an index of emotion rules by chemical for faster lookups
        
        Returns:
            Dictionary mapping chemicals to relevant emotion rules
        """
        if not hasattr(self, "emotion_derivation_rules"):
            return {}
            
        chemical_rules = defaultdict(list)
        
        for rule in self.emotion_derivation_rules:
            for chemical in rule.get("chemical_conditions", {}):
                chemical_rules[chemical].append(rule)
        
        return dict(chemical_rules)
    
    def get_tool_usage_stats(self) -> Dict[str, int]:
        """
        Get statistics on tool usage
        
        Returns:
            Dictionary of tool usage counts
        """
        return dict(self.tool_usage)
    
    def get_handoff_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about handoffs
        
        Returns:
            Handoff statistics dictionary
        """
        # Calculate handoff counts by type
        handoff_counts = defaultdict(int)
        for handoff in self.handoff_history:
            transfer_key = f"{handoff['from']}:{handoff['to']}"
            handoff_counts[transfer_key] += 1
        
        # Determine most common handoff
        most_common = max(handoff_counts.items(), key=lambda x: x[1]) if handoff_counts else (None, 0)
        
        return {
            "total_handoffs": len(self.handoff_history),
            "handoff_counts": dict(handoff_counts),
            "most_common_handoff": most_common[0],
            "most_common_count": most_common[1]
        }
    
    def cleanup(self) -> None:
        """
        Clean up any active spans on shutdown
        """
        # Finish any active spans
        for span_id, span in self.active_spans.items():
            try:
                span.finish()
            except:
                pass
        
        # Clear the spans dictionary
        self.active_spans.clear()
