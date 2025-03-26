# nyx/core/emotions/context.py

"""
Enhanced context management for the Nyx emotional system.
Provides improved typing, helper methods, and serialization support.
"""

import datetime
import json
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, TypeVar, Generic, Set, Deque

from pydantic import BaseModel, Field, root_validator

# Create a TypeVar for the EmotionalContext to be used in type hints
TEmotionalContext = TypeVar('TEmotionalContext', bound='EmotionalContext')

class EmotionalContext(BaseModel):
    """Enhanced context for emotional processing between agent runs with serialization support"""
    cycle_count: int = Field(default=0, description="Current processing cycle count")
    last_emotions: Dict[str, float] = Field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = Field(default_factory=list, max_length=20)
    temp_data: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    # Using a more efficient circular buffer for history tracking
    _circular_history: Dict[str, Deque] = Field(default_factory=lambda: defaultdict(lambda: deque(maxlen=20)), exclude=True)
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True
        extra = "allow"
    
    # Serialization/deserialization support
    def to_json(self) -> str:
        """Serialize context to JSON for persistence"""
        # Exclude temp_data and other non-serializable fields
        serializable_data = self.dict(exclude={"temp_data", "_circular_history"})
        return json.dumps(serializable_data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EmotionalContext':
        """Create context from JSON serialized data"""
        data = json.loads(json_str)
        return cls(**data)
    
    # Add helper methods directly to the context with improved implementations
    def record_emotion(self, emotion: str, intensity: float) -> None:
        """Record an emotion with its intensity"""
        self.last_emotions[emotion] = intensity
        # Also add to history for trend analysis
        self._add_to_circular_buffer("emotion_history", {
            "emotion": emotion,
            "intensity": intensity,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    def add_interaction(self, data: Dict[str, Any]) -> None:
        """Add an interaction to history with automatic trimming using circular buffer"""
        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = datetime.datetime.now().isoformat()
            
        self.interaction_history.append(data)
        if len(self.interaction_history) > 20:
            self.interaction_history.pop(0)
    
    def _add_to_circular_buffer(self, buffer_name: str, item: Any) -> None:
        """Add item to a named circular buffer"""
        self._circular_history[buffer_name].append(item)
    
    def get_circular_buffer(self, buffer_name: str) -> List[Any]:
        """Get contents of a named circular buffer as a list"""
        return list(self._circular_history[buffer_name])
    
    def get_agent_usage(self) -> Dict[str, int]:
        """Get the agent usage statistics"""
        if "agent_usage" not in self.temp_data:
            self.temp_data["agent_usage"] = defaultdict(int)
        return self.temp_data["agent_usage"]
    
    def record_agent_usage(self, agent_name: str) -> None:
        """Record usage of an agent"""
        agent_usage = self.get_agent_usage()
        agent_usage[agent_name] += 1
    
    def get_timing_data(self) -> Dict[str, Dict[str, float]]:
        """Get agent timing data"""
        if "agent_timing" not in self.temp_data:
            self.temp_data["agent_timing"] = {}
        return self.temp_data["agent_timing"]
    
    def set_value(self, key: str, value: Any) -> None:
        """Set a value in the temporary data store"""
        self.temp_data[key] = value
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a value from the temporary data store"""
        return self.temp_data.get(key, default)
    
    def record_time_marker(self, marker_name: str) -> None:
        """Record a timestamp for performance measurement"""
        if "time_markers" not in self.temp_data:
            self.temp_data["time_markers"] = {}
        
        self.temp_data["time_markers"][marker_name] = datetime.datetime.now()
    
    def get_elapsed_time(self, start_marker: str, end_marker: Optional[str] = None) -> float:
        """Get elapsed time between markers in seconds"""
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
        """Record current neurochemical values for quick access with timestamp"""
        self.temp_data["cached_neurochemical_state"] = chemical_values
        self.temp_data["cached_time"] = datetime.datetime.now().timestamp()
        
        # Also add to history for trend analysis
        self._add_to_circular_buffer("neurochemical_history", {
            "values": dict(chemical_values),  # Create a copy
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def get_cached_neurochemicals(self, max_age_seconds: float = 1.0) -> Optional[Dict[str, float]]:
        """Get cached neurochemical values if not too old"""
        if "cached_neurochemical_state" not in self.temp_data:
            return None
        
        cached_time = self.temp_data.get("cached_time", 0)
        current_time = datetime.datetime.now().timestamp()
        
        # Use cached value if it's fresh (less than specified seconds old)
        if current_time - cached_time < max_age_seconds:
            return self.temp_data["cached_neurochemical_state"]
        
        return None
    
    def get_neurochemical_trends(self, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Get neurochemical trends over time for analysis"""
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
        """Get emotion trends over time for analysis"""
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
