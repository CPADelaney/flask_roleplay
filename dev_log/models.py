# dev_log/models.py

import datetime
import uuid
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

class DevLogEntry(BaseModel):
    """Base model for all developer log entries."""
    id: str
    timestamp: datetime.datetime
    log_type: str
    title: str
    content: str
    source_module: str
    metadata: Dict[str, Any] = {}
    severity: str = "info"  # info, warning, critical
    tags: List[str] = []

class SynergyRecommendation(DevLogEntry):
    """Synergy recommendation log entry."""
    recommendation_id: str
    from_module: Optional[str] = None
    to_module: Optional[str] = None
    priority: float
    expected_impact: str
    applied: bool = False
    applied_timestamp: Optional[datetime.datetime] = None
    applied_by: Optional[str] = None
    results: Optional[Dict[str, Any]] = None

class ModuleOptimization(DevLogEntry):
    """Module optimization log entry."""
    module_name: str
    optimization_type: str  # performance, code, architecture
    code_changes: Optional[Dict[str, Any]] = None
    performance_impact: Optional[Dict[str, Any]] = None

class SystemInsight(DevLogEntry):
    """System-wide insight log entry."""
    insight_type: str  # pattern, bottleneck, integration, security
    affected_modules: List[str] = []
    action_items: List[str] = []
