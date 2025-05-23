# nyx/core/a2a/causal_model_types.py

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

class NodeType(Enum):
    """Types of nodes in causal models"""
    INPUT = "input"
    INTERMEDIATE = "intermediate"
    OUTCOME = "outcome"
    CONFOUNDER = "confounder"
    MEDIATOR = "mediator"
    MODERATOR = "moderator"
    LATENT = "latent"

class EdgeType(Enum):
    """Types of causal relationships"""
    DIRECT = "direct"
    INDIRECT = "indirect"
    BIDIRECTIONAL = "bidirectional"
    MODERATED = "moderated"
    TIME_DELAYED = "time_delayed"
    THRESHOLD = "threshold"
    NONLINEAR = "nonlinear"

@dataclass
class ModelMetadata:
    """Metadata for causal models"""
    model_id: str
    name: str
    domain: str
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    author: str = "system"
    validation_status: str = "validated"
    confidence_score: float = 0.8
    sample_size: Optional[int] = None
    data_sources: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
