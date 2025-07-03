# lore/core/context.py
from typing import Optional, Dict, Any, Union

class CanonicalContext:
    """
    Standardized context object for canonical operations.
    Ensures consistent interface regardless of how it's created.
    """
    
    def __init__(self, user_id: int, conversation_id: int, **kwargs):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Store any additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CanonicalContext':
        """Create context from dictionary."""
        user_id = data.get('user_id', 0)
        conversation_id = data.get('conversation_id', 0)
        return cls(user_id, conversation_id, **data)
    
    @classmethod
    def from_object(cls, obj: Any) -> 'CanonicalContext':
        """Create context from arbitrary object."""
        # Try various ways to extract user_id and conversation_id
        user_id = None
        conversation_id = None
        
        # Direct attributes
        if hasattr(obj, 'user_id'):
            user_id = obj.user_id
        if hasattr(obj, 'conversation_id'):
            conversation_id = obj.conversation_id
        
        # Context attribute
        if hasattr(obj, 'context'):
            if isinstance(obj.context, dict):
                user_id = user_id or obj.context.get('user_id')
                conversation_id = conversation_id or obj.context.get('conversation_id')
        
        # Dictionary-like
        if hasattr(obj, 'get'):
            user_id = user_id or obj.get('user_id')
            conversation_id = conversation_id or obj.get('conversation_id')
        
        # Use defaults if not found
        user_id = user_id if user_id is not None else 0
        conversation_id = conversation_id if conversation_id is not None else 0
        
        return cls(user_id, conversation_id)
    
    @classmethod
    def system_context(cls) -> 'CanonicalContext':
        """Create a system-level context for operations without user context."""
        return cls(0, 0, is_system=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'user_id': self.user_id,
            'conversation_id': self.conversation_id,
            **{k: v for k, v in self.__dict__.items() 
               if k not in ['user_id', 'conversation_id']}
        }
    
    def __repr__(self):
        return f"CanonicalContext(user_id={self.user_id}, conversation_id={self.conversation_id})"
