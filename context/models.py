# context/models.py
"""
Pydantic models for context data using the OpenAI Agents SDK
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union


# Base metadata models
class BaseMetadata(BaseModel):
    """Base metadata model with common fields"""
    timestamp: Optional[str] = None
    source: Optional[str] = None
    version: Optional[int] = None
    
    class Config:
        extra = "forbid"  # Explicitly forbid extra fields


# Specific metadata models
class MemoryMetadata(BaseMetadata):
    """Metadata specific to memories"""
    npc_id: Optional[str] = None
    location_id: Optional[str] = None
    quest_id: Optional[str] = None
    emotion: Optional[str] = None
    context_type: Optional[str] = None
    related_memories: Optional[List[str]] = None
    
    # Consolidation metadata
    group_key: Optional[str] = None
    memory_count: Optional[int] = None
    time_span: Optional['TimeSpanMetadata'] = None


class TimeSpanMetadata(BaseModel):
    """Time span metadata for consolidated memories"""
    start: str
    end: str
    
    class Config:
        extra = "forbid"


class EntityMetadata(BaseMetadata):
    """Metadata specific to entities"""
    entity_class: Optional[str] = None
    relationships: Optional[List[str]] = None
    attributes: Optional[List[str]] = None
    status: Optional[str] = None
    tags: Optional[List[str]] = None


class NPCMetadata(EntityMetadata):
    """Metadata specific to NPCs"""
    personality: Optional[str] = None
    occupation: Optional[str] = None
    faction: Optional[str] = None
    mood: Optional[str] = None
    location: Optional[str] = None


class LocationMetadata(EntityMetadata):
    """Metadata specific to locations"""
    location_type: Optional[str] = None
    population: Optional[int] = None
    danger_level: Optional[int] = None
    resources: Optional[List[str]] = None
    quests_available: Optional[List[str]] = None


class LocationDetails(BaseModel):
    """Detailed location information"""
    location_id: Optional[str] = None
    location_name: str
    description: Optional[str] = None
    location_type: Optional[str] = None
    connected_locations: Optional[List[str]] = None
    metadata: Optional[LocationMetadata] = None
    
    class Config:
        extra = "forbid"


class NarrativeSummaryData(BaseModel):
    """Data structure for narrative summaries"""
    summary_text: str
    summary_level: int
    key_events: Optional[List[str]] = None
    important_npcs: Optional[List[str]] = None
    locations_visited: Optional[List[str]] = None
    time_period: Optional[TimeSpanMetadata] = None
    
    class Config:
        extra = "forbid"


class DeltaChange(BaseModel):
    """Represents a single delta change"""
    path: str
    operation: str  # "add", "remove", "replace"
    value: Optional[Union[str, int, float, bool, List[str]]] = None
    old_value: Optional[Union[str, int, float, bool, List[str]]] = None
    priority: int = 5
    timestamp: float
    
    class Config:
        extra = "forbid"


class DeltaChanges(BaseModel):
    """Container for delta changes"""
    changes: List[DeltaChange]
    from_version: int
    to_version: int
    
    class Config:
        extra = "forbid"


# Update existing models
class ContextRequest(BaseModel):
    """Request model for context operations"""
    user_id: int
    conversation_id: int
    input_text: str = ""
    location: Optional[str] = None
    context_budget: int = 4000
    use_vector_search: Optional[bool] = None
    use_delta: bool = True
    include_memories: bool = True
    include_npcs: bool = True
    include_location: bool = True
    include_quests: bool = True
    source_version: Optional[int] = None
    summary_level: Optional[int] = None


class Memory(BaseModel):
    """Memory data model"""
    memory_id: str
    content: str
    memory_type: str = "observation"
    created_at: str
    importance: float = 0.5
    access_count: int = 0
    last_accessed: str
    tags: List[str] = []
    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata)


class NPCData(BaseModel):
    """NPC data model"""
    npc_id: str
    npc_name: str
    dominance: Optional[float] = None
    cruelty: Optional[float] = None
    closeness: Optional[float] = None
    trust: Optional[float] = None
    respect: Optional[float] = None
    intensity: Optional[float] = None
    current_location: Optional[str] = None
    physical_description: Optional[str] = None
    relevance: Optional[float] = None
    metadata: Optional[NPCMetadata] = None


class LocationData(BaseModel):
    """Location data model"""
    location_id: Optional[str] = None
    location_name: str
    description: Optional[str] = None
    connected_locations: Optional[List[str]] = None
    relevance: Optional[float] = None
    metadata: Optional[LocationMetadata] = None


class QuestData(BaseModel):
    """Quest data model"""
    quest_id: str
    quest_name: str
    status: str
    progress_detail: Optional[str] = None
    quest_giver: Optional[str] = None
    reward: Optional[str] = None


class NarrativeSummary(BaseModel):
    """Narrative summary data model"""
    summary_id: str
    content: str
    narrative_type: str
    importance: float = 0.5
    relevance: Optional[float] = None
    data: Optional[NarrativeSummaryData] = None


class TokenUsage(BaseModel):
    """Token usage information"""
    player_stats: Optional[int] = None
    npcs: Optional[int] = None
    memories: Optional[int] = None
    location: Optional[int] = None
    quests: Optional[int] = None
    time: Optional[int] = None
    roleplay: Optional[int] = None
    narratives: Optional[int] = None
    summaries: Optional[int] = None
    other: Optional[int] = None


class ContextOutput(BaseModel):
    """Complete context output model"""
    npcs: List[NPCData] = []
    memories: List[Memory] = []
    location_details: Optional[LocationDetails] = None
    quests: List[QuestData] = []
    narrative_summaries: Optional[NarrativeSummaryData] = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    total_tokens: int = 0
    version: Optional[int] = None
    timestamp: str = ""
    retrieval_time: Optional[float] = None
    is_delta: bool = False
    delta_changes: Optional[DeltaChanges] = None


class MemorySearchRequest(BaseModel):
    """Request model for memory search"""
    user_id: int
    conversation_id: int
    query_text: str
    memory_types: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    limit: int = 5
    use_vector: bool = True


class VectorSearchRequest(BaseModel):
    """Request model for vector searches"""
    user_id: int
    conversation_id: int
    query_text: str
    entity_types: Optional[List[str]] = None
    top_k: int = 5
    hybrid_ranking: bool = True
    recency_weight: float = 0.3


class MemoryAddRequest(BaseModel):
    """Request model for adding a memory"""
    user_id: int
    conversation_id: int
    content: str
    memory_type: str = "observation"
    importance: Optional[float] = None
    tags: Optional[List[str]] = None
    metadata: Optional[MemoryMetadata] = None
    store_vector: bool = True
