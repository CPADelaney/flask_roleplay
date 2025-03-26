# nyx/core/emotions/context.py

"""
Enhanced context management for the Nyx emotional system.
Provides improved typing, helper methods, and serialization support
with better OpenAI Agents SDK integration.
"""

import datetime
import json
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, TypeVar, Generic, Set, Deque, Union

from pydantic import BaseModel, Field, root_validator, validator
from agents import RunContextWrapper, Agent

# Create a TypeVar for the EmotionalContext to be used in type hints
TEmotionalContext = TypeVar('TEmotionalContext', bound='EmotionalContext')
TAgent = TypeVar('TAgent', bound=Agent)

class EmotionalContext(BaseModel, Generic[TAgent]):
    """
    Enhanced context for emotional processing between agent runs with improved
    SDK integration, serialization support, and runtime optimizations.
    """
    cycle_count: int = Field(default=0, description="Current processing cycle count")
    last_emotions: Dict[str, float] = Field(default_factory=dict, description="Most recent emotion intensities")
    interaction_history: List[Dict[str, Any]] = Field(default_factory=list, max_length=20, 
                                                      description="Recent interaction data")
    temp_data: Dict[str, Any] = Field(default_factory=dict, exclude=True, 
                                      description="Temporary runtime data that won't be serialized")
    # Using a more efficient circular buffer for history tracking
    _circular_history: Dict[str, Deque] = Field(default_factory=lambda: defaultdict(lambda: deque(maxlen=20)), 
                                               exclude=True, description="Circular buffers for various history types")
    
    # New fields for improved SDK integration
    active_agent: Optional[str] = Field(default=None, description="Currently active agent name")
    agent_metadata: Dict[str, Dict[str, Any]] = Field(default_factory=dict, 
                                                      description="Metadata for each agent")
    trace_metadata: Dict[str, Any] = Field(default_factory=dict, 
                                          description="Metadata for tracing")
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True
        extra = "allow"
    
    # Serialization/deserialization support with enhanced validation
    def to_json(self) -> str:
        """
        Serialize context to JSON for persistence
        
        Returns:
            JSON string representation of the context
        """
        # Exclude temp_data and other non-serializable fields
        serializable_data = self.dict(exclude={"temp_data", "_circular_history"})
        return json.dumps(serializable_data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EmotionalContext':
        """
        Create context from JSON serialized data
        
        Args:
            json_str: JSON string representation of context
            
        Returns:
            Reconstructed context object
        """
        data = json.loads(json_str)
        return cls(**data)
    
    # Add helper methods directly to the context with improved implementations
    def record_emotion(self, emotion: str, intensity: float) -> None:
        """
        Record an emotion with its intensity
        
        Args:
            emotion: Name of the emotion
            intensity: Intensity value (0.0-1.0)
        """
        self.last_emotions[emotion] = intensity
        # Also add to history for trend analysis
        self._add_to_circular_buffer("emotion_history", {
            "emotion": emotion,
            "intensity": intensity,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def record_agent_state(self, agent_name: str, data: Dict[str, Any]) -> None:
        """
        Record agent state information
        
        Args:
            agent_name: Name of the agent
            data: State data to record
        """
        if agent_name not in self.agent_metadata:
            self.agent_metadata[agent_name] = {}
            
        # Update metadata with new information
        self.agent_metadata[agent_name].update(data)
        
        # Record the active agent
        self.active_agent = agent_name
        
        # Also record in circular buffer for history
        self._add_to_circular_buffer("agent_state_history", {
            "agent": agent_name,
            "data": data,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def add_interaction(self, data: Dict[str, Any]) -> None:
        """
        Add an interaction to history with automatic trimming using circular buffer
        
        Args:
            data: Interaction data to record
        """
        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = datetime.datetime.now().isoformat()
            
        # Add cycle count if not present
        if "cycle" not in data:
            data["cycle"] = self.cycle_count
            
        self.interaction_history.append(data)
        if len(self.interaction_history) > 20:
            self.interaction_history.pop(0)
    
    def _add_to_circular_buffer(self, buffer_name: str, item: Any) -> None:
        """
        Add item to a named circular buffer
        
        Args:
            buffer_name: Name of the buffer
            item: Item to add
        """
        self._circular_history[buffer_name].append(item)
    
    def get_circular_buffer(self, buffer_name: str) -> List[Any]:
        """
        Get contents of a named circular buffer as a list
        
        Args:
            buffer_name: Name of the buffer
            
        Returns:
            List of items in the buffer
        """
        return list(self._circular_history[buffer_name])
    
    def get_agent_usage(self) -> Dict[str, int]:
        """
        Get the agent usage statistics
        
        Returns:
            Dictionary of agent usage counts
        """
        if "agent_usage" not in self.temp_data:
            self.temp_data["agent_usage"] = defaultdict(int)
        return self.temp_data["agent_usage"]
    
    def record_agent_usage(self, agent_name: str) -> None:
        """
        Record usage of an agent
        
        Args:
            agent_name: Name of the agent used
        """
        agent_usage = self.get_agent_usage()
        agent_usage[agent_name] += 1
        
        # Also track when this agent was last used
        self.temp_data["last_agent_use"] = {
            "agent": agent_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "cycle": self.cycle_count
        }
    
    def get_timing_data(self) -> Dict[str, Dict[str, float]]:
        """
        Get agent timing data
        
        Returns:
            Dictionary of timing data by agent
        """
        if "agent_timing" not in self.temp_data:
            self.temp_data["agent_timing"] = {}
        return self.temp_data["agent_timing"]
    
    def set_value(self, key: str, value: Any) -> None:
        """
        Set a value in the temporary data store
        
        Args:
            key: Data key
            value: Data value
        """
        self.temp_data[key] = value
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the temporary data store
        
        Args:
            key: Data key
            default: Default value if key not found
            
        Returns:
            Retrieved value or default
        """
        return self.temp_data.get(key, default)
    
    def record_time_marker(self, marker_name: str) -> None:
        """
        Record a timestamp for performance measurement
        
        Args:
            marker_name: Name of the time marker
        """
        if "time_markers" not in self.temp_data:
            self.temp_data["time_markers"] = {}
        
        self.temp_data["time_markers"][marker_name] = datetime.datetime.now()
    
    def get_elapsed_time(self, start_marker: str, end_marker: Optional[str] = None) -> float:
        """
        Get elapsed time between markers in seconds
        
        Args:
            start_marker: Starting time marker name
            end_marker: Ending time marker name (current time if None)
            
        Returns:
            Elapsed time in seconds
        """
        markers = self.temp_data.get("time_markers", {})
        
        if start_marker not in markers:
            return 0.0
        
        start_time = markers[start_marker]
        
        if end_marker and end_marker in markers:
            end_time = markers[end_marker]
        else:
            end_time = datetime.datetime.now()
        
        return (end_time - start_time).total_seconds()
    
    def record_neurochemical_values(self, chemical_values: Dict[str, float]) -> None:
        """
        Record current neurochemical values for quick access with timestamp
        
        Args:
            chemical_values: Dictionary of neurochemical values
        """
        self.temp_data["cached_neurochemical_state"] = chemical_values
        self.temp_data["cached_time"] = datetime.datetime.now().timestamp()
        
        # Also add to history for trend analysis
        self._add_to_circular_buffer("neurochemical_history", {
            "values": dict(chemical_values),  # Create a copy
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def get_cached_neurochemicals(self, max_age_seconds: float = 1.0) -> Optional[Dict[str, float]]:
        """
        Get cached neurochemical values if not too old
        
        Args:
            max_age_seconds: Maximum age of cached values in seconds
            
        Returns:
            Dictionary of neurochemical values or None if too old
        """
        if "cached_neurochemical_state" not in self.temp_data:
            return None
        
        cached_time = self.temp_data.get("cached_time", 0)
        current_time = datetime.datetime.now().timestamp()
        
        # Use cached value if it's fresh (less than specified seconds old)
        if current_time - cached_time < max_age_seconds:
            return self.temp_data["cached_neurochemical_state"]
        
        return None
    
    def get_neurochemical_trends(self, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get neurochemical trends over time for analysis
        
        Args:
            limit: Maximum number of historical data points to consider
            
        Returns:
            Dictionary of neurochemical trends
        """
        history = self.get_circular_buffer("neurochemical_history")[-limit:]
        
        # Transform into per-chemical trends
        if not history:
            return {}
            
        trends = defaultdict(list)
        for entry in history:
            timestamp = entry["timestamp"]
            for chemical, value in entry["values"].items():
                trends[chemical].append({
                    "value": value,
                    "timestamp": timestamp
                })
                
        return dict(trends)
    
    def get_emotion_trends(self, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get emotion trends over time for analysis
        
        Args:
            limit: Maximum number of historical data points to consider
            
        Returns:
            Dictionary of emotion trends
        """
        history = self.get_circular_buffer("emotion_history")[-limit:]
        
        # Transform into per-emotion trends
        if not history:
            return {}
            
        trends = defaultdict(list)
        for entry in history:
            trends[entry["emotion"]].append({
                "intensity": entry["intensity"],
                "timestamp": entry["timestamp"]
            })
                
        return dict(trends)
    
    # New methods for SDK integration
    def prepare_for_agent(self, agent_type: str) -> Dict[str, Any]:
        """
        Prepare context-specific data for an agent type
        
        Args:
            agent_type: Type of agent to prepare for
            
        Returns:
            Dictionary of agent-specific context data
        """
        # Base context data for all agents
        context_data = {
            "cycle_count": self.cycle_count,
            "last_emotions": self.last_emotions
        }
        
        # Add agent-specific context
        if agent_type == "neurochemical":
            context_data.update({
                "cached_chemicals": self.get_cached_neurochemicals(),
                "decay_hours": self.get_value("decay_hours", 0.0)
            })
            
        elif agent_type == "emotion_derivation":
            # Include neurochemical data if available
            if "cached_neurochemical_state" in self.temp_data:
                context_data["chemicals"] = self.temp_data["cached_neurochemical_state"]
                
            # Include emotion rule index if available
            if "emotion_rule_index" in self.temp_data:
                context_data["rule_index"] = self.temp_data["emotion_rule_index"]
                
        elif agent_type == "reflection":
            # Include dominant emotion if available
            if self.last_emotions:
                dominant = max(self.last_emotions.items(), key=lambda x: x[1])
                context_data["dominant_emotion"] = dominant[0]
                context_data["intensity"] = dominant[1]
            
            # Include recent interactions for context
            if self.interaction_history:
                context_data["recent_interactions"] = self.interaction_history[-3:]
                
        elif agent_type == "learning":
            # Include pattern history
            pattern_history = self.get_value("pattern_history", [])
            if pattern_history:
                context_data["recent_patterns"] = pattern_history[-5:]
                
            # Include learning stats
            context_data["learning_stats"] = self.get_value("reward_learning_stats", {})
            
        elif agent_type == "orchestrator":
            # Include all relevant data for orchestration
            context_data["active_agent"] = self.active_agent
            context_data["last_handoff"] = self.get_value("last_handoff")
            context_data["current_emotional_state"] = self.get_value("current_emotional_state")
        
        return context_data
    
    def create_trace_metadata(self) -> Dict[str, Any]:
        """
        Create metadata for traces
        
        Returns:
            Dictionary of trace metadata
        """
        metadata = {
            "system": "nyx_emotional_core",
            "version": "1.0",
            "cycle": self.cycle_count,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add active agent if available
        if self.active_agent:
            metadata["active_agent"] = self.active_agent
            
        # Add dominant emotion if available
        if self.last_emotions:
            dominant = max(self.last_emotions.items(), key=lambda x: x[1])
            metadata["dominant_emotion"] = dominant[0]
            metadata["intensity"] = dominant[1]
            
        # Add conversation ID if available
        if "conversation_id" in self.temp_data:
            metadata["conversation_id"] = self.temp_data["conversation_id"]
            
        return metadata
# Adding enhanced methods to EmotionalContext

def prepare_agent_context(self, agent_type: str) -> Dict[str, Any]:
    """
    Prepare context data optimized for a specific agent type with
    enhanced SDK integration
    
    Args:
        agent_type: Type of agent to prepare for
        
    Returns:
        Dictionary of agent-specific context data
    """
    # Use the existing prepare_for_agent method as a base
    context_data = self.prepare_for_agent(agent_type)
    
    # Add SDK-specific optimizations
    context_data["_sdk_metadata"] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "cycle": self.cycle_count,
        "agent_type": agent_type
    }
    
    # Add trace context for better SDK integration
    context_data["trace_context"] = self.create_trace_metadata()
    
    # Add SDK-specific performance data
    perf_data = {}
    
    # Add agent timing data if available
    agent_timing = self.get_timing_data()
    if agent_type in agent_timing:
        perf_data["avg_response_time"] = agent_timing[agent_type].get("avg_time", 0)
        perf_data["response_count"] = agent_timing[agent_type].get("count", 0)
    
    # Add function timing data if available for this agent type
    function_timing = self.get_value("function_timing", {})
    if function_timing:
        perf_data["function_timing"] = {
            k: v for k, v in function_timing.items()
            if k.startswith(agent_type) or agent_type.lower() in k.lower()
        }
    
    # Add performance data to context if any was found
    if perf_data:
        context_data["_performance"] = perf_data
    
    return context_data

def create_trace_metadata(self) -> Dict[str, Any]:
    """
    Create metadata for traces with enhanced SDK integration
    
    Returns:
        Dictionary of trace metadata
    """
    metadata = {
        "system": "nyx_emotional_core",
        "version": "1.0",
        "cycle": self.cycle_count,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Add active agent if available
    if self.active_agent:
        metadata["active_agent"] = self.active_agent
        
    # Add dominant emotion if available with additional information
    if self.last_emotions:
        dominant = max(self.last_emotions.items(), key=lambda x: x[1])
        metadata["dominant_emotion"] = dominant[0]
        metadata["intensity"] = dominant[1]
        
        # Add all emotions above threshold for richer tracing
        significant_emotions = {
            emotion: intensity for emotion, intensity in self.last_emotions.items()
            if intensity > 0.3  # Only include emotions with significant intensity
        }
        if significant_emotions:
            metadata["emotions"] = significant_emotions
            
    # Add conversation ID if available
    if "conversation_id" in self.temp_data:
        metadata["conversation_id"] = self.temp_data["conversation_id"]
        
    # Add SDK integration timestamp
    metadata["sdk_timestamp"] = datetime.datetime.now().isoformat()
    
    return metadata

def get_recent_activity(self, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get recent activity across all circular buffers for
    improved SDK monitoring
    
    Args:
        limit: Maximum number of items to return from each buffer
        
    Returns:
        Dictionary with recent activity data
    """
    activity = {}
    
    # Get recent activity from circular buffers
    for buffer_name in self._circular_history.keys():
        buffer_data = self.get_circular_buffer(buffer_name)
        if buffer_data:
            activity[buffer_name] = buffer_data[-limit:]
    
    # Add standard fields
    activity["timestamp"] = datetime.datetime.now().isoformat()
    activity["cycle"] = self.cycle_count
    
    return activity
