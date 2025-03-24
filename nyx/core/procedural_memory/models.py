# nyx/core/procedural_memory/models.py

import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ActionTemplate(BaseModel):
    """Generic template for an action that can be mapped across domains"""
    action_type: str  # e.g., "aim", "shoot", "move", "sprint", "interact"
    intent: str  # Higher-level purpose, e.g., "target_acquisition", "locomotion"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    domain_mappings: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # domain -> specific implementations

class ChunkTemplate(BaseModel):
    """Generalizable template for a procedural chunk"""
    id: str
    name: str
    description: str
    actions: List[ActionTemplate] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)  # Domains where this chunk has been applied
    success_rate: Dict[str, float] = Field(default_factory=dict)  # domain -> success rate
    execution_count: Dict[str, int] = Field(default_factory=dict)  # domain -> count
    context_indicators: Dict[str, List[str]] = Field(default_factory=dict)  # domain -> [context keys]
    prerequisite_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # domain -> {state requirements}
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

class ContextPattern(BaseModel):
    """Pattern for recognizing a specific context"""
    id: str
    name: str
    domain: str
    indicators: Dict[str, Any] = Field(default_factory=dict)  # Key state variables and their values/ranges
    temporal_pattern: List[Dict[str, Any]] = Field(default_factory=list)  # Sequence of recent actions/states
    confidence_threshold: float = 0.7
    last_matched: Optional[str] = None
    match_count: int = 0

class ChunkPrediction(BaseModel):
    """Prediction for which chunk should be executed next"""
    chunk_id: str
    confidence: float
    context_match_score: float
    reasoning: List[str] = Field(default_factory=list)
    alternative_chunks: List[Dict[str, float]] = Field(default_factory=list)

class ControlMapping(BaseModel):
    """Mapping between control schemes across different domains"""
    source_domain: str
    target_domain: str
    action_type: str
    source_control: str
    target_control: str
    confidence: float = 1.0
    last_validated: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    validation_count: int = 0

class ProcedureTransferRecord(BaseModel):
    """Record of a procedure transfer between domains"""
    source_procedure_id: str
    source_domain: str
    target_procedure_id: str 
    target_domain: str
    transfer_date: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    adaptation_steps: List[Dict[str, Any]] = Field(default_factory=list)
    success_level: float = 0.0  # 0-1 rating of transfer success
    practice_needed: int = 0  # How many practice iterations needed post-transfer

class Procedure(BaseModel):
    """A procedure to be executed"""
    id: str
    name: str
    description: str
    domain: str
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    execution_count: int = 0
    successful_executions: int = 0
    average_execution_time: float = 0.0
    proficiency: float = 0.0
    chunked_steps: Dict[str, List[str]] = Field(default_factory=dict)
    is_chunked: bool = False
    chunk_contexts: Dict[str, str] = Field(default_factory=dict)  # Maps chunk_id -> context pattern id
    generalized_chunks: Dict[str, str] = Field(default_factory=dict)  # Maps chunk_id -> template_id
    context_history: List[Dict[str, Any]] = Field(default_factory=list)
    refinement_opportunities: List[Dict[str, Any]] = Field(default_factory=list)
    optimization_history: List[Dict[str, Any]] = Field(default_factory=list)
    max_history: int = 50
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_execution: Optional[str] = None
    
class StepResult(BaseModel):
    """Result from executing a step"""
    success: bool
    error: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float = 0.0

class ProcedureStats(BaseModel):
    """Statistics about a procedure"""
    procedure_name: str
    procedure_id: str
    proficiency: float
    level: str
    execution_count: int
    success_rate: float
    average_execution_time: float
    is_chunked: bool
    chunks_count: int
    steps_count: int
    last_execution: Optional[str] = None
    domain: str
    generalized_chunks: int = 0
    refinement_opportunities: int = 0

class TransferStats(BaseModel):
    """Statistics about procedure transfers"""
    total_transfers: int = 0
    successful_transfers: int = 0
    avg_success_level: float = 0.0
    avg_practice_needed: int = 0
    chunks_by_domain: Dict[str, int] = Field(default_factory=dict)
    recent_transfers: List[Dict[str, Any]] = Field(default_factory=list)
    templates_count: int = 0
    actions_count: int = 0
