# dev_log/api.py

import asyncio
from typing import Dict, List, Any, Optional
import datetime
import uuid

from dev_log.storage import get_dev_log_storage
from dev_log.models import DevLogEntry, SynergyRecommendation, ModuleOptimization, SystemInsight

async def add_synergy_recommendation(
    title: str,
    content: str,
    from_module: Optional[str] = None,
    to_module: Optional[str] = None,
    priority: float = 0.5,
    expected_impact: str = "Improved integration",
    source_module: str = "synergy_optimizer",
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Add a synergy recommendation to the dev log.
    
    Args:
        title: Title of the recommendation
        content: Detailed description
        from_module: Source module
        to_module: Target module
        priority: Importance (0-1)
        expected_impact: Expected impact description
        source_module: Module that generated the recommendation
        metadata: Additional metadata
        
    Returns:
        ID of the created log entry
    """
    storage = get_dev_log_storage()
    
    recommendation_id = f"rec_{uuid.uuid4().hex[:8]}"
    
    recommendation = SynergyRecommendation(
        id=f"log_{uuid.uuid4().hex[:8]}",
        timestamp=datetime.datetime.now(),
        log_type="synergy_recommendation",
        title=title,
        content=content,
        source_module=source_module,
        metadata=metadata or {},
        tags=["synergy", "recommendation"],
        recommendation_id=recommendation_id,
        from_module=from_module,
        to_module=to_module,
        priority=priority,
        expected_impact=expected_impact,
        applied=False
    )
    
    log_id = await storage.add_log(recommendation)
    return log_id

async def add_module_optimization(
    title: str,
    content: str,
    module_name: str,
    optimization_type: str,
    source_module: str = "module_optimizer",
    code_changes: Optional[Dict[str, Any]] = None,
    performance_impact: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Add a module optimization to the dev log.
    
    Args:
        title: Title of the optimization
        content: Detailed description
        module_name: Name of the module
        optimization_type: Type of optimization
        source_module: Module that generated the optimization
        code_changes: Code changes details
        performance_impact: Performance impact details
        metadata: Additional metadata
        
    Returns:
        ID of the created log entry
    """
    storage = get_dev_log_storage()
    
    optimization = ModuleOptimization(
        id=f"log_{uuid.uuid4().hex[:8]}",
        timestamp=datetime.datetime.now(),
        log_type="module_optimization",
        title=title,
        content=content,
        source_module=source_module,
        metadata=metadata or {},
        tags=["optimization", optimization_type],
        module_name=module_name,
        optimization_type=optimization_type,
        code_changes=code_changes,
        performance_impact=performance_impact
    )
    
    log_id = await storage.add_log(optimization)
    return log_id

async def add_system_insight(
    title: str,
    content: str,
    insight_type: str,
    affected_modules: List[str] = [],
    action_items: List[str] = [],
    source_module: str = "meta_core",
    severity: str = "info",
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Add a system insight to the dev log.
    
    Args:
        title: Title of the insight
        content: Detailed description
        insight_type: Type of insight
        affected_modules: List of affected modules
        action_items: Suggested action items
        source_module: Module that generated the insight
        severity: Importance level
        metadata: Additional metadata
        
    Returns:
        ID of the created log entry
    """
    storage = get_dev_log_storage()
    
    insight = SystemInsight(
        id=f"log_{uuid.uuid4().hex[:8]}",
        timestamp=datetime.datetime.now(),
        log_type="system_insight",
        title=title,
        content=content,
        source_module=source_module,
        metadata=metadata or {},
        tags=["insight", insight_type],
        severity=severity,
        insight_type=insight_type,
        affected_modules=affected_modules,
        action_items=action_items
    )
    
    log_id = await storage.add_log(insight)
    return log_id

async def get_recent_recommendations(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent synergy recommendations."""
    storage = get_dev_log_storage()
    return await storage.get_logs(log_type="synergy_recommendation", limit=limit)

async def get_recommendation_stats() -> Dict[str, Any]:
    """Get statistics about synergy recommendations."""
    storage = get_dev_log_storage()
    return await storage.get_recommendation_stats()

async def update_recommendation_status(
    recommendation_id: str,
    applied: bool,
    applied_by: str = "developer",
    results: Optional[Dict[str, Any]] = None
) -> bool:
    """Update the status of a recommendation."""
    storage = get_dev_log_storage()
    return await storage.update_recommendation(recommendation_id, applied, applied_by, results)
