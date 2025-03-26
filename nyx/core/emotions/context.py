# nyx/core/emotions/context.py

"""
Context management for the Nyx emotional system.
Enhanced context class with better typing and helper methods.
"""

import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, TypeVar, Generic

from pydantic import BaseModel, Field

# Create a TypeVar for the EmotionalContext to be used in type hints
TEmotionalContext = TypeVar('TEmotionalContext', bound='EmotionalContext')

class EmotionalContext(BaseModel):
    """Enhanced context for emotional processing between agent runs"""
    cycle_count: int = Field(default=0, description="Current processing cycle count")
    last_emotions: Dict[str, float] = Field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = Field(default_factory=list, max_length=20)
    temp_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Add helper methods directly to the context
    def record_emotion(self, emotion: str, intensity: float) -> None:
        """Record an emotion with its intensity"""
        self.last_emotions[emotion] = intensity
        
    def add_interaction(self, data: Dict[str, Any]) -> None:
        """Add an interaction to history with automatic trimming"""
        self.interaction_history.append(data)
        if len(self.interaction_history) > 20:
            self.interaction_history.pop(0)
    
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
        """Record current neurochemical values for quick access"""
        self.temp_data["cached_neurochemical_state"] = chemical_values
        self.temp_data["cached_time"] = datetime.datetime.now().timestamp()
    
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
