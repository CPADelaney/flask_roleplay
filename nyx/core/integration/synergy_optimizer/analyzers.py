# nyx/core/integration/synergy_optimizer/analyzers.py

import asyncio
from typing import Dict, List, Any, Set
import datetime

async def analyze_event_flow(event_patterns):
    """
    Analyze event flow patterns to identify communication channels.
    
    Args:
        event_patterns: Dictionary of event patterns
        
    Returns:
        Analysis results
    """
    flow_data = {}
    
    for pattern_key, pattern in event_patterns.items():
        source, event_type = pattern_key.split(":", 1)
        
        # Calculate event frequency
        if pattern["recent_timestamps"]:
            timestamps = pattern["recent_timestamps"]
            if len(timestamps) >= 2:
                time_span = (timestamps[-1] - timestamps[0]).total_seconds() / 60  # minutes
                frequency = len(timestamps) / max(1, time_span)  # events per minute
            else:
                frequency = 0
        else:
            frequency = 0
            
        flow_data[pattern_key] = {
            "source": source,
            "event_type": event_type,
            "count": pattern["count"],
            "frequency": frequency
        }
    
    return flow_data

async def detect_missing_connections(module_interactions, integration_status):
    """
    Detect potential missing connections between modules.
    
    Args:
        module_interactions: Dictionary of module interactions
        integration_status: Current integration status
        
    Returns:
        List of potential missing connections
    """
    missing_connections = []
    
    # Get list of all active modules
    all_modules = set()
    bridge_details = integration_status.get("bridge_details", {})
    
    for bridge_name, details in bridge_details.items():
        if details.get("available", False):
            all_modules.add(bridge_name)
    
    # Add modules from interactions that might not be in bridge details
    for module in module_interactions:
        all_modules.add(module)
    
    # Identify potential connections based on module functionality
    module_pairs = [
        # Modules that might benefit from direct connection
        ("emotional_cognitive", "decision_action_coordinator"),
        ("prediction_imagination", "reward_learning"),
        ("memory_integration", "need_goal_action"),
        ("identity_imagination_emotional", "decision_action_coordinator"),
        # Add more pairs as patterns emerge
    ]
    
    for source, target in module_pairs:
        if source in all_modules and target in all_modules:
            # Check if already connected
            connected = False
            
            if source in module_interactions:
                if "targets" in module_interactions[source]:
                    if target in module_interactions[source]["targets"]:
                        connected = True
            
            if not connected:
                missing_connections.append({
                    "from_module": source,
                    "to_module": target,
                    "potential_benefit": "Improved coordination"
                })
    
    return missing_connections

async def identify_redundant_events(event_patterns):
    """
    Identify potentially redundant events in the system.
    
    Args:
        event_patterns: Dictionary of event patterns
        
    Returns:
        List of potentially redundant events
    """
    redundant_events = []
    
    # Group similar events
    event_by_type = {}
    
    for pattern_key, pattern in event_patterns.items():
        source, event_type = pattern_key.split(":", 1)
        
        if event_type not in event_by_type:
            event_by_type[event_type] = []
            
        event_by_type[event_type].append({
            "source": source,
            "count": pattern["count"],
            "pattern_key": pattern_key
        })
    
    # Look for similar events from different sources
    for event_type, sources in event_by_type.items():
        if len(sources) > 1:
            # Multiple sources for same event type - potential redundancy
            redundant_events.append({
                "event_type": event_type,
                "sources": [s["source"] for s in sources],
                "counts": [s["count"] for s in sources]
            })
    
    return redundant_events
