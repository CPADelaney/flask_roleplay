# nyx/core/integration/synergy_optimizer/models.py

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import datetime

class EventPattern(BaseModel):
    """Pattern of events flowing through the system."""
    source: str
    event_type: str
    count: int
    frequency: float  # Events per minute
    common_targets: List[str]
    
class ModuleInteraction(BaseModel):
    """Interaction between two modules."""
    source_module: str
    target_module: str
    event_types: List[str]
    strength: float  # Measure of connection strength
    
class SynergyRecommendation(BaseModel):
    """Recommendation for improving module synergy."""
    id: str
    type: str  # new_bridge, event_subscription, etc.
    description: str
    from_module: Optional[str]
    to_module: Optional[str]
    event_type: Optional[str]
    priority: float  # 0-1 importance score
    expected_impact: str
    created_at: datetime.datetime
    applied: bool = False
