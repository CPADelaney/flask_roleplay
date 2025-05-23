# nyx/core/a2a/causal_model_repository.py

import json
import asyncio
from typing import Dict, List, Any, Optional
from collections import defaultdict
import networkx as nx
import numpy as np

from .causal_model_types import NodeType, EdgeType, ModelMetadata

class CausalModelRepository:
    """Centralized repository for causal models"""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure one repository instance"""
        if cls._instance is None:
            cls._instance = super(CausalModelRepository, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize repository only once"""
        if self._initialized:
            return
            
        self.models = {}
        self.model_registry = {
            "by_domain": defaultdict(list),
            "by_tags": defaultdict(list),
            "by_confidence": defaultdict(list),
            "all_models": []
        }
        
        self._initialize_model_templates()
        self._initialized = True
    
    def _initialize_model_templates(self):
        """Initialize model templates"""
        # Add the comprehensive templates from the previous artifact
        self.model_templates = {
            # ... (insert model templates here)
        }
        
        # Register all models
        for template_id, template in self.model_templates.items():
            self._register_model(template_id, template)
    
    def _register_model(self, model_id: str, model: Dict[str, Any]):
        """Register a model in the searchable index"""
        if "metadata" not in model:
            return
            
        metadata = model["metadata"]
        if isinstance(metadata, dict):
            # Convert dict to ModelMetadata if needed
            metadata = ModelMetadata(**metadata)
        
        self.models[model_id] = model
        self.model_registry["all_models"].append(model_id)
        self.model_registry["by_domain"][metadata.domain].append(model_id)
        
        for tag in metadata.tags:
            self.model_registry["by_tags"][tag].append(model_id)
        
        confidence_bucket = f"{int(metadata.confidence_score * 10) / 10:.1f}"
        self.model_registry["by_confidence"][confidence_bucket].append(model_id)
    
    async def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a model by ID"""
        return self.models.get(model_id)
    
    async def find_models_by_domain(self, domain: str) -> List[str]:
        """Find all models in a domain"""
        return self.model_registry["by_domain"].get(domain, [])
    
    async def find_models_by_tags(self, tags: List[str]) -> List[str]:
        """Find models matching any of the given tags"""
        matching_models = set()
        for tag in tags:
            matching_models.update(self.model_registry["by_tags"].get(tag, []))
        return list(matching_models)
