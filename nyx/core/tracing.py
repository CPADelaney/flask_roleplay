# nyx/core/tracing.py

import logging
from typing import Dict, Any, Optional
from agents import trace, trace_metadata, custom_span

logger = logging.getLogger(__name__)

class NyxTracing:
    """Centralized tracing system for Nyx."""
    
    @staticmethod
    def start_operation(
        workflow_name: str, 
        operation_type: str, 
        user_id: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Start a traced operation."""
        full_workflow_name = f"Nyx_{workflow_name}"
        meta = metadata or {}
        if user_id:
            meta["user_id"] = user_id
        meta["operation_type"] = operation_type
        
        # Create and return trace object
        return trace(
            workflow_name=full_workflow_name,
            trace_metadata=meta
        )
    
    @staticmethod
    def add_span(name: str, data: Dict[str, Any]):
        """Add a custom span to the current trace."""
        return custom_span(name=name, data=data)
    
    @staticmethod
    def add_metadata(metadata: Dict[str, Any]):
        """Add metadata to the current trace."""
        trace_metadata(metadata)
