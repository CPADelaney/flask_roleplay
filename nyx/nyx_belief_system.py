# nyx/nyx_belief_system.py
"""
Minimal belief system implementation for Nyx.
This provides a foundation for Nyx's belief tracking and reasoning.
"""

import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class BeliefSystem:
    """
    Manages Nyx's beliefs about the world, NPCs, and the player.
    This is a minimal implementation that can be expanded later.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.beliefs = {}  # entity_id -> belief_type -> belief_data
        
    async def get_beliefs(self, entity_id: str) -> Dict[str, Any]:
        """Get all beliefs about an entity."""
        return self.beliefs.get(entity_id, {})
    
    async def update_belief(self, entity_id: str, belief_type: str, content: Dict[str, Any]):
        """Update a belief about an entity."""
        if entity_id not in self.beliefs:
            self.beliefs[entity_id] = {}
        
        self.beliefs[entity_id][belief_type] = {
            "content": content,
            "updated_at": datetime.now().isoformat(),
            "confidence": content.get("confidence", 0.7)
        }
        
        logger.info(f"Updated belief for {entity_id}: {belief_type}")
    
    async def query_beliefs(self, query: str) -> List[Dict[str, Any]]:
        """Query beliefs based on a search string."""
        results = []
        query_lower = query.lower()
        
        for entity_id, entity_beliefs in self.beliefs.items():
            for belief_type, belief_data in entity_beliefs.items():
                # Simple text search in belief content
                content_str = json.dumps(belief_data.get("content", {})).lower()
                if query_lower in content_str or query_lower in belief_type.lower():
                    results.append({
                        "entity_id": entity_id,
                        "belief_type": belief_type,
                        "belief_data": belief_data,
                        "relevance": 1.0 if query_lower in belief_type.lower() else 0.5
                    })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:10]  # Return top 10 results
