# nyx/core/context.py

import datetime
from typing import Dict, Any, Optional, Generic, TypeVar
from pydantic import BaseModel, Field

T = TypeVar('T')

class NyxBaseContext(BaseModel):
    """Base context class for all Nyx operations."""
    user_id: Optional[str] = None
    operation_id: str = Field(default_factory=lambda: f"op_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
    operation_type: str = "generic"
    operation_start_time: datetime.datetime = Field(default_factory=datetime.datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class NyxSystemContext(NyxBaseContext, Generic[T]):
    """System-specific context with generic system state."""
    system_name: str
    system_state: T
    execution_data: Dict[str, Any] = Field(default_factory=dict)
    
    def get_state(self) -> T:
        """Get the system state"""
        return self.system_state
