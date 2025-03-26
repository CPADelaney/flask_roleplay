# nyx/core/emotions/hooks.py

"""
Enhanced lifecycle hooks for the Nyx emotional system.
These hooks monitor agent lifecycle events with proper typing.
"""

import datetime
import logging
from typing import Any, Dict, Optional

from agents import Agent, AgentHooks, RunContextWrapper, FunctionTool

from nyx.core.emotions.context import EmotionalContext

logger = logging.getLogger(__name__)

class EmotionalAgentHooks(AgentHooks[EmotionalContext]):
    """Enhanced hooks for emotional agent lifecycle events with proper typing"""
    
    def __init__(self, neurochemicals=None):
        """
        Initialize the hooks with optional neurochemical state reference
        
        Args:
            neurochemicals: Reference to neurochemical state dictionary
        """
        self.neurochemicals = neurochemicals
        self.agent_timing: Dict[str, Dict[str, float]] = {}
    
    async def on_start(self, context: RunContextWrapper[EmotionalContext], 
                      agent: Agent[EmotionalContext]) -> None:
        """
        Called when an agent starts processing
        
        Args:
            context: The run context wrapper
            agent: The agent that is starting
        """
        logger.debug(f"Emotional agent started: {agent.name}")
        
        # Track which agents are being used
        context.context.record_agent_usage(agent.name)
        
        # Track start time for performance metrics
        context.context.record_time_marker(f"start_{agent.name}")
        
        # Add agent-specific initialization
        if agent.name == "Neurochemical Agent":
            # Pre-fetch current neurochemical decay needs
            if self.neurochemicals:
                now = datetime.datetime.now()
                last_update = context.context.get_value("last_neurochemical_update")
                
                if last_update:
                    hours_elapsed = (now - last_update).total_seconds() / 3600
                    context.context.set_value("decay_hours", hours_elapsed)
        
        elif agent.name == "Emotion Derivation Agent":
            # Pre-fetch neurochemical state to avoid duplicate calls
            if "cached_neurochemical_state" not in context.context.temp_data and self.neurochemicals:
                context.context.record_neurochemical_values({
                    c: d["value"] for c, d in self.neurochemicals.items()
                })
        
        elif agent.name == "Emotional Reflection Agent":
            # Pre-calculate a dominant emotion if we have it
            last_emotions = context.context.last_emotions
            if last_emotions:
                dominant = max(last_emotions.items(), key=lambda x: x[1]) if last_emotions else None
                if dominant:
                    context.context.set_value("precalculated_dominant", dominant)
    
    async def on_end(self, context: RunContextWrapper[EmotionalContext], 
                    agent: Agent[EmotionalContext], output: Any) -> None:
        """
        Called when an agent completes processing
        
        Args:
            context: The run context wrapper
            agent: The agent that is ending
            output: The agent's output
        """
        # Calculate and store performance metrics
        context.context.record_time_marker(f"end_{agent.name}")
        duration = context.context.get_elapsed_time(f"start_{agent.name}", f"end_{agent.name}")
        
        # Update rolling average response time for this agent
        agent_timing = context.context.get_timing_data()
        if agent.name not in agent_timing:
            agent_timing[agent.name] = {"count": 0, "avg_time": 0}
        
        stats = agent_timing[agent.name]
        stats["avg_time"] = ((stats["avg_time"] * stats["count"]) + duration) / (stats["count"] + 1)
        stats["count"] += 1
        
        # Add agent-specific teardown
        if agent.name == "Neurochemical Agent":
            # Cache the latest neurochemical update time
            context.context.set_value("last_neurochemical_update", datetime.datetime.now())
        
        logger.debug(f"Agent {agent.name} completed in {duration:.2f}s (avg: {stats['avg_time']:.2f}s)")
    
    async def on_tool_start(self, context: RunContextWrapper[EmotionalContext], 
                           agent: Agent[EmotionalContext], tool: FunctionTool) -> None:
        """
        Called when a tool is invoked
        
        Args:
            context: The run context wrapper
            agent: The agent invoking the tool
            tool: The tool being invoked
        """
        logger.debug(f"Tool started: {tool.name} by agent {agent.name}")
        context.context.record_time_marker(f"tool_start_{tool.name}")
    
    async def on_tool_end(self, context: RunContextWrapper[EmotionalContext], 
                         agent: Agent[EmotionalContext], tool: FunctionTool, result: str) -> None:
        """
        Called when a tool completes execution
        
        Args:
            context: The run context wrapper
            agent: The agent that invoked the tool
            tool: The tool that was invoked
            result: The tool's result
        """
        context.context.record_time_marker(f"tool_end_{tool.name}")
        duration = context.context.get_elapsed_time(f"tool_start_{tool.name}", f"tool_end_{tool.name}")
        
        # Update tool stats in context
        tool_stats = context.context.get_value("tool_stats", {})
        if tool.name not in tool_stats:
            tool_stats[tool.name] = {"count": 0, "total_time": 0}
        
        tool_stats[tool.name]["count"] += 1
        tool_stats[tool.name]["total_time"] += duration
        tool_stats[tool.name]["avg_time"] = (
            tool_stats[tool.name]["total_time"] / tool_stats[tool.name]["count"]
        )
        
        context.context.set_value("tool_stats", tool_stats)
        logger.debug(f"Tool {tool.name} completed in {duration:.2f}s")
    
    async def on_handoff(self, context: RunContextWrapper[EmotionalContext], 
                        agent: Agent[EmotionalContext], source: Agent[EmotionalContext]) -> None:
        """
        Called when an agent receives a handoff
        
        Args:
            context: The run context wrapper
            agent: The agent receiving the handoff
            source: The agent that initiated the handoff
        """
        logger.debug(f"Handoff: {source.name} -> {agent.name}")
        context.context.record_time_marker(f"handoff_to_{agent.name}")
        
        # Record the handoff in context for analysis
        handoffs = context.context.get_value("handoff_history", [])
        handoffs.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "from": source.name,
            "to": agent.name,
            "cycle": context.context.cycle_count
        })
        
        # Limit history size
        if len(handoffs) > 20:
            handoffs = handoffs[-20:]
        
        context.context.set_value("handoff_history", handoffs)
