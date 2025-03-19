# nyx/eternal/context.py

from typing import Dict, Any, List, Optional
from datetime import datetime

class OpenAINyxContext:
    """Standalone context for OpenAI Agents SDK that doesn't import from existing Nyx code"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Core systems (will be set by coordinator)
        self.meta_learning = None
        self.dynamic_adaptation = None
        self.internal_feedback = None
        
        # Session state
        self.learning_cycle_count = 0
        self.current_strategy = None
        self.feature_importance = {}
        self.state_history = []
        
        # Interaction tracking
        self.last_interaction_time = datetime.now()
        self.conversation_history = []
        
        # System health
        self.system_health = "normal"
        self.errors = []
        
    def update_state(self, state_update: Dict[str, Any]):
        """Update context state"""
        for key, value in state_update.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Track state history
        self.state_history.append({
            "timestamp": datetime.now(),
            "update": state_update
        })
