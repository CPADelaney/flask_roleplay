# nyx/core/memory_core.py

import asyncio
import datetime
import json
import logging
import math
import uuid
import random
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Literal
from collections import defaultdict
import numpy as np
import hashlib
import copy 

from pydantic import BaseModel, Field
from agents import Agent, Runner, function_tool, RunContextWrapper, trace, custom_span

logger = logging.getLogger(__name__)

# ==================== Pydantic Models ====================

class EmotionalMemoryContext(BaseModel):
    """Emotional context for memories"""
    primary_emotion: str = "neutral"
    primary_intensity: float = 0.5
    secondary_emotions: Dict[str, float] = Field(default_factory=dict)
    valence: float = 0.0
    arousal: float = 0.0

class MemorySchema(BaseModel):
    """Schema that represents patterns Nyx has identified"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    confidence: float = 0.7
    category: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    example_memory_ids: List[str] = Field(default_factory=list)
    creation_date: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    usage_count: int = 0
    evolution_history: List[Dict[str, Any]] = Field(default_factory=list)

class MemoryMetadata(BaseModel):
    """Enhanced metadata including hierarchical information"""
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    entities: List[str] = Field(default_factory=list)
    emotional_context: Optional[EmotionalMemoryContext] = None
    schemas: List[Dict[str, Any]] = Field(default_factory=list)
    last_recalled: Optional[str] = None
    last_decay: Optional[str] = None
    decay_amount: Optional[float] = None
    consolidated_into: Optional[str] = None
    consolidation_date: Optional[str] = None
    original_significance: Optional[int] = None
    original_form: Optional[str] = None
    reconsolidation_history: List[Dict[str, Any]] = Field(default_factory=list)
    semantic_abstractions: List[str] = Field(default_factory=list)
    is_crystallized: bool = False
    crystallization_reason: Optional[str] = None
    crystallization_date: Optional[str] = None
    decay_resistance: float = 1.0
    memory_level: Literal['detail', 'summary', 'abstraction'] = 'detail'
    source_memory_ids: Optional[List[str]] = None
    fidelity: float = 1.0
    summary_of: Optional[str] = None

class Memory(BaseModel):
    """Core memory object"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_text: str
    memory_type: str = "observation"
    memory_scope: str = "game"
    significance: int = 5
    tags: List[str] = Field(default_factory=list)
    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata)
    embedding: List[float] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    times_recalled: int = 0
    is_archived: bool = False
    is_consolidated: bool = False
    relevance: float = 0.0

# ==================== Input/Output Models ====================

class MemoryCreateParams(BaseModel):
    """Input model for creating memories"""
    memory_text: str
    memory_type: Optional[str] = "observation"
    memory_scope: Optional[str] = "game"
    significance: Optional[int] = 5
    tags: Optional[List[str]] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    memory_level: Optional[Literal['detail', 'summary', 'abstraction']] = "detail"
    source_memory_ids: Optional[List[str]] = None
    fidelity: Optional[float] = 1.0
    summary_of: Optional[str] = None

class MemoryUpdateParams(BaseModel):
    """Input model for updating memories"""
    memory_id: str
    updates: Optional[Dict[str, Any]] = Field(default_factory=dict)

class MemoryQuery(BaseModel):
    """Input model for retrieving memories"""
    query: str
    memory_types: Optional[List[str]] = None
    scopes: Optional[List[str]] = None
    limit: Optional[int] = 10
    min_significance: Optional[int] = 3
    include_archived: Optional[bool] = False
    entities: Optional[List[str]] = None
    emotional_state: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    retrieval_level: Optional[Literal['detail', 'summary', 'abstraction', 'auto']] = "auto"
    min_fidelity: Optional[float] = 0.0

class MemoryRetrieveResult(BaseModel):
    """Structured result for retrieval"""
    memory_id: str
    memory_text: str
    memory_type: str
    significance: int
    relevance: float
    confidence_marker: Optional[str] = None
    memory_level: Literal['detail', 'summary', 'abstraction']
    source_memory_ids: Optional[List[str]] = None
    fidelity: float
    summary_of: Optional[str] = None

class MemoryMaintenanceResult(BaseModel):
    """Result of maintenance operations"""
    memories_decayed: int
    clusters_consolidated: int
    memories_archived: int

class NarrativeResult(BaseModel):
    """Result of narrative construction"""
    narrative: str
    confidence: float
    experience_count: int

# ==================== Memory Storage ====================

class MemoryStorage:
    """Enhanced in-memory storage for memories"""
    
    def __init__(self):
        self.memories: Dict[str, Memory] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.schemas: Dict[str, MemorySchema] = {}
        
        # Query cache
        self.query_cache: Dict[str, Tuple[datetime.datetime, List[str]]] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Indices
        self.type_index: Dict[str, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.scope_index: Dict[str, Set[str]] = defaultdict(set)
        self.level_index: Dict[str, Set[str]] = {
            "detail": set(),
            "summary": set(), 
            "abstraction": set()
        }
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
        self.emotional_index: Dict[str, Set[str]] = defaultdict(set)
        self.schema_index: Dict[str, Set[str]] = defaultdict(set)
        self.significance_index: Dict[int, Set[str]] = {i: set() for i in range(1, 11)}
        self.temporal_index: List[Tuple[datetime.datetime, str]] = []
        self.archived_memories: Set[str] = set()
        self.consolidated_memories: Set[str] = set()
        self.summary_links: Dict[str, List[str]] = defaultdict(list)
        
        # Configuration (matching old core)
        self.embed_dim = 1536  # Standard OpenAI embedding dimension
        self.decay_rate = 0.1
        self.consolidation_threshold = 0.85
        self.reconsolidation_probability = 0.3
        self.reconsolidation_strength = 0.1
        self.abstraction_threshold = 3
        self.max_memory_age_days = 90
        
        # Templates for conversational recall
        self.recall_templates = {
            "standard": [
                "That reminds me of {timeframe} when {brief_summary}... {detail}",
                "I recall {timeframe} when {brief_summary}. {reflection}",
            ],
            "positive": [
                "Mmm, I remember {timeframe} when {brief_summary}. {reflection}",
                "That brings back a delicious memory of {timeframe} when {brief_summary}... {detail}",
            ],
            "negative": [
                "I recall {timeframe} dealing with someone who {brief_summary}. {reflection}",
                "This reminds me of a frustrating time when {brief_summary}... {detail}",
            ],
            "intense": [
                "Mmm, that reminds me of an *intense* experience where {brief_summary}... {detail}",
                "I vividly remember when {brief_summary}. {reflection}",
            ],
            "teasing": [
                "Oh, this reminds me of {timeframe} when I teased someone until {brief_summary}... {reflection}",
                "There was this delicious time when I {brief_summary}... {detail}",
            ],
            "disciplinary": [
                "I remember having to discipline someone who {brief_summary}. {reflection}",
                "I once dealt with someone who needed strict handling when they {brief_summary}. {reflection}",
            ]
        }
        
        self.confidence_markers: Dict[Tuple[float, float], str] = {
            (0.8, 1.01): "vividly recall",
            (0.6, 0.8): "clearly remember",
            (0.4, 0.6): "remember",
            (0.2, 0.4): "think I recall",
            (0.0, 0.2): "vaguely remember"
        }
    
    def add(self, memory: Memory) -> str:
        """Add a memory and update all indices"""
        self.memories[memory.id] = memory
        
        # Update indices
        self.type_index[memory.memory_type].add(memory.id)
        self.scope_index[memory.memory_scope].add(memory.id)
        self.level_index[memory.metadata.memory_level].add(memory.id)
        self.significance_index[memory.significance].add(memory.id)
        
        for tag in memory.tags:
            self.tag_index[tag].add(memory.id)
        
        for entity in memory.metadata.entities:
            self.entity_index[entity].add(memory.id)
        
        if memory.metadata.emotional_context:
            emotion = memory.metadata.emotional_context.primary_emotion
            self.emotional_index[emotion].add(memory.id)
        
        for schema in memory.metadata.schemas:
            if schema_id := schema.get("schema_id"):
                self.schema_index[schema_id].add(memory.id)
        
        # Update temporal index
        dt = datetime.datetime.fromisoformat(memory.created_at)
        self.temporal_index.append((dt, memory.id))
        self.temporal_index.sort(key=lambda x: x[0])
        
        # Update summary links
        if memory.metadata.source_memory_ids:
            for source_id in memory.metadata.source_memory_ids:
                self.summary_links[source_id].append(memory.id)
        
        return memory.id
    
    def get(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID"""
        return self.memories.get(memory_id)
    
    def update(self, memory: "Memory") -> bool:   # quotes for forward ref
        """
        Update an existing memory without corrupting indices:
        1. snapshot the *old* object,
        2. remove indices that refer to the snapshot,
        3. write the new object,
        4. re‑index the fresh data.
        """
        if memory.id not in self.memories:
            return False

        old_snapshot = copy.deepcopy(self.memories[memory.id])
        self._remove_from_indices(old_snapshot)

        self.memories[memory.id] = memory
        self._add_to_indices(memory)
        return True
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory"""
        if memory_id not in self.memories:
            return False
            
        memory = self.memories[memory_id]
        self._remove_from_indices(memory)
        
        del self.memories[memory_id]
        if memory_id in self.embeddings:
            del self.embeddings[memory_id]
            
        return True
    
    def _remove_from_indices(self, memory: Memory):
        """Remove memory from all indices"""
        self.type_index[memory.memory_type].discard(memory.id)
        self.scope_index[memory.memory_scope].discard(memory.id)
        self.level_index[memory.metadata.memory_level].discard(memory.id)
        self.significance_index[memory.significance].discard(memory.id)
        
        for tag in memory.tags:
            self.tag_index[tag].discard(memory.id)
        
        for entity in memory.metadata.entities:
            self.entity_index[entity].discard(memory.id)
        
        if memory.metadata.emotional_context:
            emotion = memory.metadata.emotional_context.primary_emotion
            self.emotional_index[emotion].discard(memory.id)
        
        for schema in memory.metadata.schemas:
            if schema_id := schema.get("schema_id"):
                self.schema_index[schema_id].discard(memory.id)
        
        self.archived_memories.discard(memory.id)
        self.consolidated_memories.discard(memory.id)
    
    def _add_to_indices(self, memory: Memory):
        """Add memory to all indices"""
        self.type_index[memory.memory_type].add(memory.id)
        self.scope_index[memory.memory_scope].add(memory.id)
        self.level_index[memory.metadata.memory_level].add(memory.id)
        self.significance_index[memory.significance].add(memory.id)
        
        for tag in memory.tags:
            self.tag_index[tag].add(memory.id)
        
        for entity in memory.metadata.entities:
            self.entity_index[entity].add(memory.id)
        
        if memory.metadata.emotional_context:
            emotion = memory.metadata.emotional_context.primary_emotion
            self.emotional_index[emotion].add(memory.id)
        
        for schema in memory.metadata.schemas:
            if schema_id := schema.get("schema_id"):
                self.schema_index[schema_id].add(memory.id)
        
        if memory.is_archived:
            self.archived_memories.add(memory.id)
        if memory.is_consolidated:
            self.consolidated_memories.add(memory.id)

# ==================== Global Storage ====================

_storage_instances: Dict[Tuple[Optional[int], Optional[int]], MemoryStorage] = {}

def get_storage(user_id: Optional[int], conversation_id: Optional[int]) -> MemoryStorage:
    """Get or create storage for user/conversation"""
    key = (user_id, conversation_id)
    if key not in _storage_instances:
        _storage_instances[key] = MemoryStorage()
    return _storage_instances[key]

# ==================== Context ====================

class MemoryContext(BaseModel):
    """Context passed to memory agents"""
    user_id: Optional[int] = None
    conversation_id: Optional[int] = None
    
    @property
    def storage(self) -> MemoryStorage:
        """Get the storage instance for this context"""
        return get_storage(self.user_id, self.conversation_id)

# ==================== Default Context ====================
# For backward compatibility with direct imports

_default_context: Optional[MemoryContext] = None

def get_default_context() -> MemoryContext:
    """Get or create default context for standalone functions"""
    global _default_context
    if _default_context is None:
        _default_context = MemoryContext()
    return _default_context

# ==================== Utility Functions ====================

def _mk_cache_key(**parts) -> str:
    """
    Deterministic, order‑insensitive cache key based on JSON.
    Lists are converted to sorted tuples so ['a','b']==['b','a'].
    """
    def _norm(v):
        if isinstance(v, list):
            return tuple(sorted(v))
        return v
    clean = {k: _norm(v) for k, v in parts.items() if v is not None}
    return json.dumps(clean, sort_keys=True)

def _generate_embedding(text: str, embed_dim: int = 1536) -> List[float]:
    """
    Generate a pseudo‑embedding that is
    • deterministic for identical `text`
    • correct dimension
    • does NOT touch the process‑wide random seed.
    """
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    rng = random.Random(hash_val)          # local RNG instance
    return [rng.uniform(-1, 1) for _ in range(embed_dim)]

def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    
    if norm1 * norm2 == 0:
        return 0.0
        
    return dot_product / (norm1 * norm2)

def _calculate_emotional_relevance(
    memory_emotion: Optional[EmotionalMemoryContext],
    current_emotion: Optional[Dict[str, Any]]
) -> float:
    """Calculate emotional relevance boost"""
    if not memory_emotion or not current_emotion:
        return 0.0
    
    memory_primary = memory_emotion.primary_emotion
    current_primary = current_emotion.get("primary_emotion", "neutral")
    
    if memory_primary == current_primary:
        return 0.2
    
    # Check secondary emotions
    if current_primary in memory_emotion.secondary_emotions:
        return 0.1
    
    # Valence alignment
    memory_valence = memory_emotion.valence
    current_valence = current_emotion.get("valence", 0.0)
    
    if abs(memory_valence - current_valence) < 0.5:
        return 0.05
    
    return 0.0

def _get_confidence_marker(storage: MemoryStorage, relevance: float) -> str:
    """Get confidence marker for relevance"""
    for (min_val, max_val), marker in storage.confidence_markers.items():
        if min_val <= relevance < max_val:
            return marker
    return "remember"

def _get_timeframe_text(timestamp: str) -> str:
    """Get conversational timeframe text"""
    try:
        memory_time = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        days_ago = (datetime.datetime.now() - memory_time).days
        
        if days_ago < 1:
            return "earlier today"
        elif days_ago < 2:
            return "yesterday"
        elif days_ago < 7:
            return f"{days_ago} days ago"
        elif days_ago < 14:
            return "last week"
        elif days_ago < 30:
            return "a couple weeks ago"
        else:
            return "a while back"
    except:
        return "a while back"

def _alter_memory_text(text: str, strength: float) -> str:
    """Alter memory text for reconsolidation"""
    words = text.split()
    num_alterations = max(1, int(len(words) * strength))
    
    for _ in range(num_alterations):
        idx = random.randint(0, len(words) - 1)
        if len(words[idx]) <= 3:
            continue
        
        alteration = random.choice(["intensify", "weaken", "replace"])
        
        if alteration == "intensify":
            words[idx] = f"very {words[idx]}"
        elif alteration == "weaken":
            words[idx] = f"somewhat {words[idx]}"
        elif alteration == "replace" and words[idx].endswith("ed"):
            words[idx] = words[idx].replace("ed", "ing")
    
    return " ".join(words)

def _detect_memory_pattern(memories: List[Memory]) -> Optional[Dict[str, Any]]:
    """Detect patterns in memories"""
    if len(memories) < 3:
        return None
    
    # Count tag frequencies
    tag_counts = defaultdict(int)
    for memory in memories:
        for tag in memory.tags:
            tag_counts[tag] += 1
    
    if not tag_counts:
        return None
    
    # Find most common tag
    top_tag = max(tag_counts.items(), key=lambda x: x[1])[0]
    
    return {
        "name": f"{top_tag.capitalize()} pattern",
        "description": f"A recurring pattern involving {top_tag}",
        "category": "tag_pattern",
        "attributes": {
            "primary_tag": top_tag,
            "frequency": tag_counts[top_tag] / len(memories),
            "confidence": min(1.0, tag_counts[top_tag] / len(memories) + 0.3)
        }
    }

# ==================== Core Memory Functions ====================
# These are the actual implementations that both tools and standalone functions use

async def _create_memory_impl(
    storage: MemoryStorage,
    memory_text: str,
    memory_type: str = "observation",
    memory_scope: str = "game",
    significance: int = 5,
    tags: List[str] = None,
    emotional_context: Dict[str, Any] = None,
    memory_level: str = "detail",
    source_memory_ids: List[str] = None,
    entities: List[str] = None
) -> Dict[str, Any]:
    """Core implementation of create_memory"""
    # Create metadata
    metadata = MemoryMetadata(
        memory_level=memory_level,
        source_memory_ids=source_memory_ids,
        entities=entities or [],
        original_form=memory_text
    )
    
    if emotional_context:
        metadata.emotional_context = EmotionalMemoryContext(**emotional_context)
    
    # Create memory
    memory = Memory(
        memory_text=memory_text,
        memory_type=memory_type,
        memory_scope=memory_scope,
        significance=significance,
        tags=tags or [],
        metadata=metadata,
        embedding=_generate_embedding(memory_text, storage.embed_dim)
    )
    
    # Store
    memory_id = storage.add(memory)
    storage.embeddings[memory_id] = memory.embedding
    
    # Clear cache
    storage.query_cache.clear()
    
    # Check for pattern creation
    if memory_type == "observation" and tags:
        # Check if we should create a schema
        for tag in tags:
            tag_memories = [
                storage.get(mid) for mid in storage.tag_index.get(tag, set())
                if storage.get(mid)
            ]
            if len(tag_memories) >= 5:
                pattern = _detect_memory_pattern(tag_memories)
                if pattern and tag not in [s.name.lower() for s in storage.schemas.values()]:
                    schema = MemorySchema(
                        name=pattern["name"],
                        description=pattern["description"],
                        category=pattern["category"],
                        attributes=pattern["attributes"],
                        example_memory_ids=[m.id for m in tag_memories[:3]]
                    )
                    storage.schemas[schema.id] = schema
                    
                    # Link schema back to memories
                    for mem in tag_memories:
                        if not any(s.get("schema_id") == schema.id for s in mem.metadata.schemas):
                            mem.metadata.schemas.append({
                                "schema_id": schema.id,
                                "relevance": 1.0
                            })
                            storage.update(mem)
    
    return {
        "memory_id": memory_id,
        "memory": memory.dict()
    }

async def _search_memories_impl(
    storage: MemoryStorage,
    query: str,
    memory_types: List[str] = None,
    scopes: List[str] = None,
    limit: int = 10,
    min_significance: int = 3,
    include_archived: bool = False,
    tags: List[str] = None,
    entities: List[str] = None,
    emotional_state: Dict[str, Any] = None,
    retrieval_level: str = "auto",
    min_fidelity: float = 0.0
) -> List[Dict[str, Any]]:
    """Core implementation of search_memories"""
    # Check cache
    cache_key = _mk_cache_key(
        q=query,
        types=memory_types,
        scopes=scopes,
        limit=limit,
        min_sig=min_significance,
        tags=tags,
        entities=entities,
        level=retrieval_level
    )
    if cache_key in storage.query_cache:
        timestamp, memory_ids = storage.query_cache[cache_key]
        if (datetime.datetime.now() - timestamp).seconds < storage.cache_ttl:
            # Return cached results
            results = []
            for memory_id in memory_ids:
                memory = storage.get(memory_id)
                if memory:
                    memory.times_recalled += 1
                    memory.metadata.last_recalled = datetime.datetime.now().isoformat()
                    storage.update(memory)
                    result = memory.dict()
                    result["confidence_marker"] = _get_confidence_marker(storage, memory.relevance)
                    results.append(result)
            return results
    
    # Get candidates
    candidate_ids = set(storage.memories.keys())
    
    # Filter by type
    if memory_types:
        type_ids = set()
        for mt in memory_types:
            type_ids.update(storage.type_index.get(mt, set()))
        candidate_ids &= type_ids
    
    # Filter by scope
    if scopes:
        scope_ids = set()
        for s in scopes:
            scope_ids.update(storage.scope_index.get(s, set()))
        candidate_ids &= scope_ids
    
    # Filter by tags
    if tags:
        for tag in tags:
            candidate_ids &= storage.tag_index.get(tag, set())
    
    # Filter by entities
    if entities:
        entity_ids = set()
        for entity in entities:
            entity_ids.update(storage.entity_index.get(entity, set()))
        candidate_ids &= entity_ids
    
    # Filter by significance
    sig_ids = set()
    for sig in range(min_significance, 11):
        sig_ids.update(storage.significance_index.get(sig, set()))
    candidate_ids &= sig_ids
    
    # Filter archived
    if not include_archived:
        candidate_ids -= storage.archived_memories
    
    # Generate query embedding
    query_embedding = _generate_embedding(query, storage.embed_dim)
    
    # Score memories
    scored_memories = []
    
    for memory_id in candidate_ids:
        memory = storage.get(memory_id)
        if not memory:
            continue
        
        # Check level and fidelity
        if retrieval_level != "auto" and memory.metadata.memory_level != retrieval_level:
            continue
        
        if memory.metadata.fidelity < min_fidelity:
            continue
        
        # Calculate relevance
        if memory_id in storage.embeddings:
            relevance = _cosine_similarity(query_embedding, storage.embeddings[memory_id])
        else:
            relevance = 0.0
        
        # Apply boosts
        entity_boost = 0.0
        if entities and memory.metadata.entities:
            common = set(entities) & set(memory.metadata.entities)
            entity_boost = min(0.2, len(common) * 0.05)
        
        emotional_boost = _calculate_emotional_relevance(
            memory.metadata.emotional_context,
            emotional_state
        )
        
        schema_boost = min(0.1, len(memory.metadata.schemas) * 0.05)
        
        # Temporal boost
        days_old = (datetime.datetime.now() - 
                   datetime.datetime.fromisoformat(memory.created_at)).days
        temporal_boost = max(0.0, 0.15 * math.exp(-0.1 * days_old))
        
        # Level boost
        level_boost = 0.0
        if retrieval_level == "auto" and memory.metadata.memory_level == "detail":
            level_boost = 0.1
        
        # Fidelity factor
        fidelity_factor = 0.7 + (0.3 * memory.metadata.fidelity)
        
        # Final score
        final_relevance = min(1.0, (relevance + entity_boost + emotional_boost + 
                                   schema_boost + temporal_boost + level_boost) * 
                                   fidelity_factor)
        
        memory.relevance = final_relevance
        scored_memories.append(memory)
    
    # Sort by relevance
    scored_memories.sort(key=lambda m: m.relevance, reverse=True)
    
    # Apply reconsolidation to top results
    results = []
    result_ids = []
    
    for memory in scored_memories[:limit]:
        # Update recall
        memory.times_recalled += 1
        memory.metadata.last_recalled = datetime.datetime.now().isoformat()
        
        # Maybe reconsolidate
        if (random.random() < storage.reconsolidation_probability and
            memory.memory_type not in ["semantic", "consolidated"]):
            
            # Skip recent memories
            days_old = (datetime.datetime.now() - 
                       datetime.datetime.fromisoformat(memory.created_at)).days
            
            if days_old >= 7:
                # Add to history
                memory.metadata.reconsolidation_history.append({
                    "previous_text": memory.memory_text,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                # Keep only last 3
                memory.metadata.reconsolidation_history = memory.metadata.reconsolidation_history[-3:]
                
                # Alter text
                memory.memory_text = _alter_memory_text(
                    memory.memory_text,
                    storage.reconsolidation_strength
                )
                
                # Reduce fidelity
                memory.metadata.fidelity = max(0.3, 
                    memory.metadata.fidelity - (storage.reconsolidation_strength * 0.1))
                
                # Regenerate embedding for altered text
                memory.embedding = _generate_embedding(memory.memory_text, storage.embed_dim)
                storage.embeddings[memory.id] = memory.embedding
        
        storage.update(memory)
        
        # Add to results
        result = memory.dict()
        result["confidence_marker"] = _get_confidence_marker(storage, memory.relevance)
        results.append(result)
        result_ids.append(memory.id)
    
    # Cache results
    storage.query_cache[cache_key] = (datetime.datetime.now(), result_ids)
    
    return results

async def _update_memory_impl(
    storage: MemoryStorage,
    memory_id: str,
    updates: Dict[str, Any]
) -> Dict[str, Any]:
    """Core implementation of update_memory"""
    memory = storage.get(memory_id)
    if not memory:
        return {"success": False, "error": "Memory not found"}
    
    # Save old state for index updates
    old_source_ids = set(memory.metadata.source_memory_ids or [])
    
    # Apply updates
    for key, value in updates.items():
        if key == "metadata" and isinstance(value, dict):
            # Merge metadata
            current_meta = memory.metadata.dict()
            current_meta.update(value)
            memory.metadata = MemoryMetadata(**current_meta)
        elif hasattr(memory, key):
            setattr(memory, key, value)
    
    # Update embedding if text changed
    if "memory_text" in updates:
        memory.embedding = _generate_embedding(memory.memory_text, storage.embed_dim)
        storage.embeddings[memory.id] = memory.embedding
    
    # Update summary links if source_memory_ids changed
    new_source_ids = set(memory.metadata.source_memory_ids or [])
    if old_source_ids != new_source_ids:
        # Remove old links
        for old_id in old_source_ids - new_source_ids:
            if memory.id in storage.summary_links.get(old_id, []):
                storage.summary_links[old_id].remove(memory.id)
                if not storage.summary_links[old_id]:
                    del storage.summary_links[old_id]
        
        # Add new links
        for new_id in new_source_ids - old_source_ids:
            storage.summary_links[new_id].append(memory.id)
    
    storage.update(memory)
    
    # Clear cache
    storage.query_cache.clear()
    
    return {"success": True, "memory": memory.dict()}

async def _create_reflection_impl(
    storage: MemoryStorage,
    topic: str = None,
    memory_ids: List[str] = None
) -> Dict[str, Any]:
    """Core implementation of create_reflection"""
    # Get memories
    if memory_ids:
        memories = [storage.get(mid) for mid in memory_ids if storage.get(mid)]
    else:
        # Need to search for memories
        search_results = await _search_memories_impl(
            storage=storage,
            query=topic or "recent experiences",
            memory_types=["observation", "experience"],
            limit=5
        )
        memories = [Memory(**m) for m in search_results]
    
    if not memories:
        return {"error": "No memories to reflect on"}
    
    # Generate reflection (simplified)
    memory_texts = [m.memory_text for m in memories]
    reflection_text = f"Reflecting on {topic or 'experiences'}: {' '.join(memory_texts[:3])}"
    
    # Create reflection
    result = await _create_memory_impl(
        storage=storage,
        memory_text=reflection_text,
        memory_type="reflection",
        significance=7,
        tags=["reflection"] + ([topic] if topic else []),
        source_memory_ids=[m.id for m in memories]
    )
    
    return result

async def _apply_decay_impl(storage: MemoryStorage) -> Dict[str, Any]:
    """Core implementation of apply_decay"""
    decayed = 0
    archived = 0
    now = datetime.datetime.now()
    
    for memory in list(storage.memories.values()):
        if memory.is_archived or memory.memory_type != "observation":
            continue
        
        # Calculate age
        created = datetime.datetime.fromisoformat(memory.created_at)
        days_old = (now - created).days
        
        # Decay parameters
        base_rate = storage.decay_rate
        decay_rate = base_rate / memory.metadata.decay_resistance if memory.metadata.is_crystallized else base_rate
        min_fidelity = 0.85 if memory.metadata.is_crystallized else 0.30
        
        age_factor = min(1.0, days_old / 30.0)
        recall_factor = max(0.0, 1.0 - (memory.times_recalled / 10.0))
        
        decay_amount = decay_rate * age_factor * recall_factor
        
        if memory.times_recalled == 0 and days_old > 7:
            decay_amount *= 1.5
        
        if decay_amount < 0.05:
            continue
        
        # Apply decay
        new_significance = max(1, memory.significance - decay_amount)
        new_significance = int(round(min(10, new_significance)))   # clamp 1‑10, round
        new_fidelity    = max(min_fidelity,
                              memory.metadata.fidelity - (decay_amount * 0.2))
        
        if abs(new_significance - memory.significance) >= 0.5:
            memory.significance = int(new_significance)
            memory.metadata.fidelity = new_fidelity
            memory.metadata.last_decay = now.isoformat()
            memory.metadata.decay_amount = decay_amount
            
            if not memory.metadata.original_significance:
                memory.metadata.original_significance = memory.significance
            
            storage.update(memory)
            decayed += 1
        
        # Archive if too weak
        if new_significance < 2 and days_old > 30:
            memory.is_archived = True
            storage.update(memory)
            archived += 1
    
    # Clear cache after decay
    storage.query_cache.clear()
    
    return {
        "decayed_count": decayed,
        "archived_count": archived
    }

async def _consolidate_memories_impl(storage: MemoryStorage) -> Dict[str, Any]:
    """Core implementation of consolidate_memories"""
    # Find clusters
    clusters = []
    candidate_ids = set()
    
    for memory_type in ["observation", "experience"]:
        candidate_ids.update(storage.type_index.get(memory_type, set()))
    
    candidate_ids -= storage.consolidated_memories
    candidate_ids -= storage.archived_memories
    
    # Simple clustering
    unclustered = list(candidate_ids)
    
    while unclustered:
        seed_id = unclustered.pop(0)
        seed_embedding = storage.embeddings.get(seed_id)
        
        if not seed_embedding:
            continue
        
        cluster = [seed_id]
        
        i = 0
        while i < len(unclustered):
            memory_id = unclustered[i]
            embedding = storage.embeddings.get(memory_id)
            
            if embedding:
                similarity = _cosine_similarity(seed_embedding, embedding)
                if similarity > storage.consolidation_threshold:
                    cluster.append(memory_id)
                    unclustered.pop(i)
                    continue
            
            i += 1
        
        if len(cluster) >= 3:
            clusters.append(cluster)
    
    # Consolidate clusters
    consolidated_count = 0
    
    for cluster in clusters:
        memories = [storage.get(mid) for mid in cluster if storage.get(mid)]
        
        if len(memories) < 3:
            continue
        
        # Create consolidated memory
        texts = [m.memory_text for m in memories]
        consolidated_text = f"Pattern observed across {len(memories)} memories: {texts[0][:50]}..."
        
        avg_significance = sum(m.significance for m in memories) / len(memories)
        all_tags = set()
        for m in memories:
            all_tags.update(m.tags)
        
        # Create consolidated memory using implementation
        result = await _create_memory_impl(
            storage=storage,
            memory_text=consolidated_text,
            memory_type="consolidated",
            significance=min(10, int(avg_significance) + 1),
            tags=list(all_tags) + ["consolidated"],
            source_memory_ids=cluster
        )
        
        # Mark originals as consolidated
        for memory in memories:
            memory.is_consolidated = True
            memory.metadata.consolidated_into = result["memory_id"]
            memory.metadata.consolidation_date = datetime.datetime.now().isoformat()
            storage.update(memory)
        
        consolidated_count += 1
    
    # Clear cache after consolidation
    storage.query_cache.clear()
    
    return {
        "clusters_consolidated": consolidated_count,
        "memories_affected": sum(len(c) for c in clusters)
    }

async def _get_memory_stats_impl(storage: MemoryStorage) -> Dict[str, Any]:
    """Core implementation of get_memory_stats"""
    total = len(storage.memories)
    
    type_counts = {t: len(ids) for t, ids in storage.type_index.items()}
    scope_counts = {s: len(ids) for s, ids in storage.scope_index.items()}
    
    # Top tags
    tag_counts = [(tag, len(ids)) for tag, ids in storage.tag_index.items()]
    top_tags = dict(sorted(tag_counts, key=lambda x: x[1], reverse=True)[:10])
    
    # Schema stats
    schema_counts = defaultdict(int)
    for schema in storage.schemas.values():
        schema_counts[schema.category] += 1
    
    # Age stats
    if storage.temporal_index:
        oldest = (datetime.datetime.now() - storage.temporal_index[0][0]).days
        newest = (datetime.datetime.now() - storage.temporal_index[-1][0]).days
    else:
        oldest = newest = 0
    
    # Fidelity stats
    fidelities = [m.metadata.fidelity for m in storage.memories.values()]
    avg_fidelity = sum(fidelities) / len(fidelities) if fidelities else 1.0
    
    return {
        "total_memories": total,
        "type_counts": type_counts,
        "scope_counts": scope_counts,
        "top_tags": top_tags,
        "archived_count": len(storage.archived_memories),
        "consolidated_count": len(storage.consolidated_memories),
        "schema_counts": dict(schema_counts),
        "total_schemas": len(storage.schemas),
        "oldest_memory_days": oldest,
        "newest_memory_days": newest,
        "avg_fidelity": avg_fidelity
    }

async def _generate_conversational_recall_impl(
    storage: MemoryStorage,
    memory_id: str,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Core implementation of generate_conversational_recall"""
    memory = storage.get(memory_id)
    if not memory:
        return {"error": "Memory not found"}
    
    # Get template based on emotion
    emotional_tone = "standard"
    if memory.metadata.emotional_context:
        emotion = memory.metadata.emotional_context.primary_emotion
        if emotion in ["Joy", "Love"]:
            emotional_tone = "positive"
        elif emotion in ["Anger", "Fear"]:
            emotional_tone = "negative"
    
    # Get template
    templates = storage.recall_templates.get(emotional_tone, storage.recall_templates["standard"])
    template = random.choice(templates)
    
    # Generate recall
    timeframe = _get_timeframe_text(memory.created_at)
    brief_summary = memory.memory_text[:30]
    detail = memory.memory_text[30:60] if len(memory.memory_text) > 30 else "the details"
    reflection = "It was quite memorable"
    
    recall_text = template.format(
        timeframe=timeframe,
        brief_summary=brief_summary,
        detail=detail,
        reflection=reflection
    )
    
    # Add fidelity qualifier
    if memory.metadata.fidelity < 0.7:
        recall_text += " Though I'm not entirely certain about all the details."
    
    return {
        "recall_text": recall_text,
        "tone": emotional_tone,
        "confidence": memory.relevance * memory.metadata.fidelity
    }

# ==================== Memory Tools ====================

@function_tool
async def create_memory(
    ctx: RunContextWrapper[MemoryContext],
    memory_text: str,
    memory_type: str = "observation",
    memory_scope: str = "game",
    significance: int = 5,
    tags: List[str] = None,
    emotional_context: Dict[str, Any] = None,
    memory_level: str = "detail",
    source_memory_ids: List[str] = None,
    entities: List[str] = None
) -> Dict[str, Any]:
    """Create a new memory with full metadata"""
    with custom_span("create_memory", {"memory_type": memory_type, "memory_level": memory_level}):
        storage = ctx.context.storage
        return await _create_memory_impl(
            storage=storage,
            memory_text=memory_text,
            memory_type=memory_type,
            memory_scope=memory_scope,
            significance=significance,
            tags=tags,
            emotional_context=emotional_context,
            memory_level=memory_level,
            source_memory_ids=source_memory_ids,
            entities=entities
        )

@function_tool
async def search_memories(
    ctx: RunContextWrapper[MemoryContext],
    query: str,
    memory_types: List[str] = None,
    scopes: List[str] = None,
    limit: int = 10,
    min_significance: int = 3,
    include_archived: bool = False,
    tags: List[str] = None,
    entities: List[str] = None,
    emotional_state: Dict[str, Any] = None,
    retrieval_level: str = "auto",
    min_fidelity: float = 0.0
) -> List[Dict[str, Any]]:
    """Search memories with enhanced filtering and relevance"""
    with custom_span("search_memories", {"query": query, "limit": limit}):
        storage = ctx.context.storage
        return await _search_memories_impl(
            storage=storage,
            query=query,
            memory_types=memory_types,
            scopes=scopes,
            limit=limit,
            min_significance=min_significance,
            include_archived=include_archived,
            tags=tags,
            entities=entities,
            emotional_state=emotional_state,
            retrieval_level=retrieval_level,
            min_fidelity=min_fidelity
        )

@function_tool
async def update_memory(
    ctx: RunContextWrapper[MemoryContext],
    memory_id: str,
    updates: Dict[str, Any]
) -> Dict[str, Any]:
    """Update an existing memory"""
    with custom_span("update_memory", {"memory_id": memory_id}):
        return await _update_memory_impl(ctx.context.storage, memory_id, updates)

@function_tool
async def delete_memory(
    ctx: RunContextWrapper[MemoryContext],
    memory_id: str
) -> Dict[str, Any]:
    """Delete a memory"""
    storage = ctx.context.storage
    
    success = storage.delete(memory_id)
    
    # Clear cache
    if success:
        storage.query_cache.clear()
    
    return {"success": success}

@function_tool
async def get_memory(
    ctx: RunContextWrapper[MemoryContext],
    memory_id: str
) -> Optional[Dict[str, Any]]:
    """Get a specific memory by ID"""
    storage = ctx.context.storage
    
    memory = storage.get(memory_id)
    if memory:
        memory.times_recalled += 1
        memory.metadata.last_recalled = datetime.datetime.now().isoformat()
        storage.update(memory)
        
        result = memory.dict()
        result["confidence_marker"] = _get_confidence_marker(storage, memory.relevance or 1.0)
        return result
    
    return None

@function_tool
async def get_memory_details(
    ctx: RunContextWrapper[MemoryContext],
    memory_ids: List[str],
    min_fidelity: float = 0.0
) -> List[Dict[str, Any]]:
    """Get detailed information for specific memories"""
    storage = ctx.context.storage
    
    results = []
    for memory_id in memory_ids:
        memory = storage.get(memory_id)
        if not memory:
            continue
            
        if memory.metadata.memory_level != "detail" or memory.metadata.fidelity < min_fidelity:
            continue
        
        memory.times_recalled += 1
        memory.metadata.last_recalled = datetime.datetime.now().isoformat()
        storage.update(memory)
        
        result = memory.dict()
        result["relevance"] = 1.0
        result["confidence_marker"] = _get_confidence_marker(storage, 1.0)
        results.append(result)
    
    return results

@function_tool
async def retrieve_memories_with_formatting(
    ctx: RunContextWrapper[MemoryContext],
    query: str,
    memory_types: List[str] = None,
    limit: int = 10,
    min_significance: int = 3
) -> List[Dict[str, Any]]:
    """Retrieve memories with formatted results for agent consumption"""
    # Call implementation directly instead of search_memories tool
    memories = await _search_memories_impl(
        storage=ctx.context.storage,
        query=query,
        memory_types=memory_types,
        limit=limit,
        min_significance=min_significance
    )
    
    # Format for agent consumption
    formatted = []
    for memory in memories:
        formatted.append({
            "id": memory["id"],
            "text": memory["memory_text"],
            "type": memory["memory_type"],
            "significance": memory["significance"],
            "confidence": memory.get("confidence_marker", "remember"),
            "relevance": memory.get("relevance", 0.5),
            "tags": memory.get("tags", []),
            "fidelity": memory.get("metadata", {}).get("fidelity", 1.0)
        })
    
    return formatted

@function_tool
async def retrieve_relevant_experiences(
    ctx: RunContextWrapper[MemoryContext],
    query: str,
    scenario_type: str = "",
    emotional_state: Dict[str, Any] = None,
    entities: List[str] = None,
    limit: int = 3,
    min_relevance: float = 0.6
) -> List[Dict[str, Any]]:
    """Retrieve experiences relevant to current context"""
    # Add scenario tags
    tags = []
    if scenario_type:
        tags.append(scenario_type.lower())
    
    # Search using implementation
    experiences = await _search_memories_impl(
        storage=ctx.context.storage,
        query=query,
        memory_types=["experience"],
        tags=tags if tags else None,
        emotional_state=emotional_state,
        entities=entities,
        limit=limit * 2
    )
    
    # Filter and enhance
    results = []
    for exp in experiences:
        if exp.get("relevance", 0) >= min_relevance:
            # Add recall text using implementation
            recall_data = await _generate_conversational_recall_impl(
                ctx.context.storage,
                memory_id=exp["id"]
            )
            
            result = {
                "id": exp["id"],
                "content": exp["memory_text"],
                "relevance_score": exp.get("relevance", 0.5),
                "emotional_context": exp.get("metadata", {}).get("emotional_context", {}),
                "scenario_type": scenario_type or exp.get("tags", ["general"])[0],
                "confidence_marker": exp.get("confidence_marker", "remember"),
                "experiential_richness": min(1.0, exp.get("significance", 5) / 10.0),
                "fidelity": exp.get("metadata", {}).get("fidelity", 1.0),
                "schemas": exp.get("metadata", {}).get("schemas", []),
                "recall_text": recall_data.get("recall_text", "")
            }
            
            results.append(result)
            
            if len(results) >= limit:
                break
    
    return results

@function_tool
async def unarchive_memory(
    ctx: RunContextWrapper[MemoryContext],
    memory_id: str
) -> Dict[str, Any]:
    """Unarchive a memory"""
    storage = ctx.context.storage
    
    memory = storage.get(memory_id)
    if not memory:
        return {"success": False}
    
    memory.is_archived = False
    storage.update(memory)
    
    return {"success": True}

@function_tool
async def mark_as_consolidated(
    ctx: RunContextWrapper[MemoryContext],
    memory_id: str,
    consolidated_into: str
) -> Dict[str, Any]:
    """Mark a memory as consolidated into another"""
    # Use implementation directly
    return await _update_memory_impl(
        storage=ctx.context.storage,
        memory_id=memory_id,
        updates={
            "is_consolidated": True,
            "metadata": {
                "consolidated_into": consolidated_into,
                "consolidation_date": datetime.datetime.now().isoformat()
            }
        }
    )

@function_tool
async def create_semantic_memory(
    ctx: RunContextWrapper[MemoryContext],
    source_memory_ids: List[str],
    abstraction_type: str = "pattern"
) -> Optional[str]:
    """Create semantic memory from sources"""
    storage = ctx.context.storage
    
    memories = [storage.get(mid) for mid in source_memory_ids if storage.get(mid)]
    
    if len(memories) < storage.abstraction_threshold:
        return None
    
    # Generate semantic text
    texts = [m.memory_text for m in memories]
    
    if abstraction_type == "belief":
        semantic_text = f"Based on experiences, I believe: {texts[0][:50]}..."
    elif abstraction_type == "preference":
        semantic_text = f"I have developed a preference: {texts[0][:50]}..."
    else:
        semantic_text = f"Pattern recognized: {texts[0][:50]}..."
    
    # Create semantic memory using implementation
    result = await _create_memory_impl(
        storage=storage,
        memory_text=semantic_text,
        memory_type="semantic",
        memory_level="abstraction",
        significance=max(m.significance for m in memories) + 1,
        tags=["semantic", abstraction_type],
        source_memory_ids=source_memory_ids
    )
    
    # Update source memories
    for memory in memories:
        if result["memory_id"] not in memory.metadata.semantic_abstractions:
            memory.metadata.semantic_abstractions.append(result["memory_id"])
            storage.update(memory)
    
    return result["memory_id"]

@function_tool
async def detect_schema_from_memories(
    ctx: RunContextWrapper[MemoryContext],
    topic: str = None,
    min_memories: int = 3
) -> Optional[Dict[str, Any]]:
    """Detect potential schema from memories"""
    # Search for relevant memories using implementation
    memories = await _search_memories_impl(
        storage=ctx.context.storage,
        query=topic if topic else "important memory",
        limit=10
    )
    
    if len(memories) < min_memories:
        return None
    
    # Convert back to Memory objects
    memory_objects = []
    storage = ctx.context.storage
    for mem_dict in memories:
        if memory := storage.get(mem_dict["id"]):
            memory_objects.append(memory)
    
    # Detect pattern
    pattern = _detect_memory_pattern(memory_objects)
    
    if not pattern:
        return None
    
    # Create schema
    schema = MemorySchema(
        name=pattern["name"],
        description=pattern["description"],
        category=pattern["category"],
        attributes=pattern["attributes"],
        example_memory_ids=[m["id"] for m in memories[:min_memories]]
    )
    
    storage.schemas[schema.id] = schema
    
    # Update memories with schema
    for memory_id in schema.example_memory_ids:
        if memory := storage.get(memory_id):
            memory.metadata.schemas.append({
                "schema_id": schema.id,
                "relevance": 1.0
            })
            storage.update(memory)
    
    return {
        "schema_id": schema.id,
        "schema_name": schema.name,
        "description": schema.description,
        "memory_count": len(schema.example_memory_ids)
    }

@function_tool
async def reflect_on_memories(
    ctx: RunContextWrapper[MemoryContext],
    recent_only: bool = True,
    limit: int = 10
) -> Dict[str, Any]:
    """Periodic reflection to identify important memories"""
    storage = ctx.context.storage
    crystallized_count = 0
    
    if recent_only:
        # Get recent memories
        cutoff = datetime.datetime.now() - datetime.timedelta(days=1)
        memories_to_assess = []
        
        for dt, memory_id in reversed(storage.temporal_index):
            if dt < cutoff:
                break
            if memory := storage.get(memory_id):
                memories_to_assess.append(memory)
                if len(memories_to_assess) >= limit:
                    break
    else:
        # Get unassessed memories
        memories_to_assess = []
        for memory in storage.memories.values():
            if not memory.metadata.is_crystallized and not memory.is_archived:
                memories_to_assess.append(memory)
                if len(memories_to_assess) >= limit:
                    break
    
    # Assess each memory
    for memory in memories_to_assess:
        # Simple importance check
        importance_score = 0.5
        
        # Check for identity relevance
        if any(word in memory.memory_text.lower() 
               for word in ["i am", "my nature", "defines me"]):
            importance_score += 0.3
        
        # Check emotional significance
        if memory.metadata.emotional_context:
            importance_score += memory.metadata.emotional_context.primary_intensity * 0.2
        
        # Crystallize if important
        if importance_score > 0.7:
            memory.metadata.is_crystallized = True
            memory.metadata.crystallization_reason = "cognitive_importance"
            memory.metadata.crystallization_date = datetime.datetime.now().isoformat()
            memory.metadata.decay_resistance = 8.0
            memory.significance = max(memory.significance, 8)
            storage.update(memory)
            crystallized_count += 1
    
    return {
        "memories_assessed": len(memories_to_assess),
        "memories_crystallized": crystallized_count
    }

@function_tool
async def archive_memory(
    ctx: RunContextWrapper[MemoryContext],
    memory_id: str
) -> Dict[str, Any]:
    """Archive a memory"""
    storage = ctx.context.storage
    
    memory = storage.get(memory_id)
    if not memory:
        return {"success": False}
    
    memory.is_archived = True
    storage.update(memory)
    
    return {"success": True}

@function_tool
async def crystallize_memory(
    ctx: RunContextWrapper[MemoryContext],
    memory_id: str,
    reason: str = "automatic"
) -> Dict[str, Any]:
    """Crystallize a memory to prevent decay"""
    storage = ctx.context.storage
    
    memory = storage.get(memory_id)
    if not memory:
        return {"success": False}
    
    memory.metadata.is_crystallized = True
    memory.metadata.crystallization_reason = reason
    memory.metadata.crystallization_date = datetime.datetime.now().isoformat()
    memory.metadata.decay_resistance = 8.0 if reason == "cognitive_importance" else 5.0
    memory.significance = max(memory.significance, 8)
    
    storage.update(memory)
    
    return {"success": True, "memory": memory.dict()}

@function_tool
async def create_reflection(
    ctx: RunContextWrapper[MemoryContext],
    topic: str = None,
    memory_ids: List[str] = None
) -> Dict[str, Any]:
    """Create a reflection from memories"""
    return await _create_reflection_impl(ctx.context.storage, topic, memory_ids)

@function_tool
async def create_abstraction(
    ctx: RunContextWrapper[MemoryContext],
    memory_ids: List[str],
    pattern_type: str = "behavior"
) -> Dict[str, Any]:
    """Create abstraction from memories"""
    storage = ctx.context.storage
    
    memories = [storage.get(mid) for mid in memory_ids if storage.get(mid)]
    
    if len(memories) < 3:
        return {"error": "Need at least 3 memories for abstraction"}
    
    # Generate abstraction
    texts = [m.memory_text for m in memories]
    abstraction_text = f"Pattern observed in {pattern_type}: {' '.join(texts[:2])}"
    
    # Create abstraction memory using implementation
    result = await _create_memory_impl(
        storage=storage,
        memory_text=abstraction_text,
        memory_type="abstraction",
        memory_level="abstraction",
        significance=8,
        tags=["abstraction", pattern_type],
        source_memory_ids=memory_ids
    )
    
    return result

@function_tool
async def apply_decay(ctx: RunContextWrapper[MemoryContext]) -> Dict[str, Any]:
    """Apply memory decay"""
    with custom_span("apply_decay"):
        return await _apply_decay_impl(ctx.context.storage)

@function_tool
async def consolidate_memories(ctx: RunContextWrapper[MemoryContext]) -> Dict[str, Any]:
    """Consolidate similar memories"""
    with custom_span("consolidate_memories"):
        return await _consolidate_memories_impl(ctx.context.storage)

@function_tool
async def get_memory_stats(ctx: RunContextWrapper[MemoryContext]) -> Dict[str, Any]:
    """Get memory statistics"""
    return await _get_memory_stats_impl(ctx.context.storage)

@function_tool
async def generate_conversational_recall(
    ctx: RunContextWrapper[MemoryContext],
    memory_id: str,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Generate natural recall of a memory"""
    with custom_span("generate_conversational_recall", {"memory_id": memory_id}):
        return await _generate_conversational_recall_impl(ctx.context.storage, memory_id, context)

@function_tool
async def construct_narrative(
    ctx: RunContextWrapper[MemoryContext],
    topic: str,
    chronological: bool = True,
    limit: int = 5
) -> Dict[str, Any]:
    """Construct narrative from memories"""
    # Use implementation directly
    memories = await _search_memories_impl(
        storage=ctx.context.storage,
        query=topic,
        memory_types=["observation", "experience"],
        limit=limit
    )
    
    if not memories:
        return {
            "narrative": f"I don't have memories about {topic}",
            "confidence": 0.2,
            "experience_count": 0
        }
    
    # Sort if needed
    if chronological:
        memories.sort(key=lambda m: m.get("created_at", ""))
    
    # Build narrative
    narrative_parts = []
    for i, memory in enumerate(memories[:3]):
        timeframe = _get_timeframe_text(memory.get("created_at", ""))
        text = memory.get("memory_text", "")
        narrative_parts.append(f"{timeframe}, {text}")
    
    narrative = f"Regarding {topic}: " + " ".join(narrative_parts)
    
    # Calculate confidence
    avg_relevance = sum(m.get("relevance", 0.5) for m in memories) / len(memories)
    avg_fidelity = sum(m.get("metadata", {}).get("fidelity", 1.0) for m in memories) / len(memories)
    confidence = (0.4 + (len(memories) / 10) + (avg_relevance * 0.3)) * avg_fidelity
    
    return {
        "narrative": narrative,
        "confidence": min(1.0, confidence),
        "experience_count": len(memories)
    }

@function_tool
async def run_maintenance(
    ctx: RunContextWrapper[MemoryContext]
) -> Dict[str, Any]:
    """Run full memory maintenance cycle"""
    # Apply decay
    decay_result = await _apply_decay_impl(ctx.context.storage)
    
    # Consolidate memories
    consolidate_result = await _consolidate_memories_impl(ctx.context.storage)
    
    # Archive old memories
    storage = ctx.context.storage
    archive_count = 0
    
    for memory in list(storage.memories.values()):
        if memory.is_archived:
            continue
            
        if memory.memory_type not in ["observation", "experience"]:
            continue
        
        # Calculate age
        created = datetime.datetime.fromisoformat(memory.created_at)
        days_old = (datetime.datetime.now() - created).days
        
        # Archive very old memories with no recalls
        if days_old > storage.max_memory_age_days and memory.times_recalled == 0:
            memory.is_archived = True
            storage.update(memory)
            archive_count += 1
    
    return {
        "memories_decayed": decay_result["decayed_count"],
        "memories_archived": decay_result["archived_count"] + archive_count,
        "clusters_consolidated": consolidate_result["clusters_consolidated"]
    }

# ==================== Memory Agents ====================

def create_memory_agent() -> Agent:
    """Create the main memory management agent"""
    return Agent(
        name="Memory Manager",
        instructions="""You manage Nyx's complete memory system including:
        - Creating memories with proper metadata and emotional context
        - Managing memory decay and crystallization
        - Consolidating similar memories
        - Maintaining memory statistics
        
        Consider significance, emotional context, and relationships between memories.""",
        tools=[
            create_memory,
            search_memories,
            update_memory,
            delete_memory,
            archive_memory,
            unarchive_memory,
            crystallize_memory,
            mark_as_consolidated,
            apply_decay,
            consolidate_memories,
            get_memory_stats,
            run_maintenance
        ],
        model="gpt-4.1-nano",
    )

def create_retrieval_agent() -> Agent:
    """Create specialized retrieval agent"""
    return Agent(
        name="Memory Retrieval Specialist",
        instructions="""You specialize in finding and retrieving relevant memories.
        Consider:
        - Semantic similarity and contextual relevance
        - Emotional context alignment
        - Entity and schema relationships
        - Memory fidelity and confidence levels
        - Temporal relevance
        
        Apply reconsolidation when appropriate.""",
        tools=[
            search_memories, 
            get_memory,
            get_memory_details,
            retrieve_memories_with_formatting,
            retrieve_relevant_experiences,
            generate_conversational_recall
        ],
        model="gpt-4.1-nano",
    )

def create_reflection_agent() -> Agent:
    """Create specialized reflection agent"""
    return Agent(
        name="Reflection Specialist",
        instructions="""You create reflections, abstractions, and narratives from memories.
        You can:
        - Generate thoughtful reflections from experiences
        - Create abstractions that identify patterns
        - Construct coherent narratives
        - Detect schemas and recurring patterns
        - Create semantic memories from multiple sources""",
        tools=[
            search_memories,
            create_memory,
            create_reflection,
            create_abstraction,
            create_semantic_memory,
            construct_narrative,
            detect_schema_from_memories,
            reflect_on_memories
        ],
        model="gpt-4.1-nano",
    )

# ==================== Main Memory Core Class ====================

class MemoryCoreAgents:
    """Enhanced memory system with full functionality"""
    
    def __init__(self, user_id: Optional[int], conversation_id: Optional[int]):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.context = MemoryContext(user_id=user_id, conversation_id=conversation_id)
        
        # Create agents
        self.memory_agent = create_memory_agent()
        self.retrieval_agent = create_retrieval_agent()
        self.reflection_agent = create_reflection_agent()
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize the memory system"""
        self.initialized = True
        logger.info(f"Memory system initialized for user {self.user_id}")
    
    # ==================== Compatibility Methods ====================
    
    async def add_memory(self, **kwargs) -> str:
        """Add memory - compatibility wrapper"""
        # Extract metadata dict to individual params
        metadata = kwargs.pop('metadata', {})
        
        # Use the implementation directly
        result = await _create_memory_impl(
            storage=self.context.storage,
            memory_text=kwargs.get('memory_text', ''),
            memory_type=kwargs.get('memory_type', 'observation'),
            memory_scope=kwargs.get('memory_scope', 'game'),
            significance=kwargs.get('significance', 5),
            tags=kwargs.get('tags', []),
            emotional_context=metadata.get('emotional_context'),
            memory_level=metadata.get('memory_level', 'detail'),
            source_memory_ids=metadata.get('source_memory_ids'),
            entities=metadata.get('entities', [])
        )
        
        return result.get("memory_id", "")
    
    async def retrieve_memories(self, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve memories - compatibility wrapper"""
        return await _search_memories_impl(
            storage=self.context.storage,
            **kwargs
        )
    
    async def create_reflection(self, topic: str = None) -> Dict[str, Any]:
        """Create reflection - compatibility wrapper"""
        return await _create_reflection_impl(
            storage=self.context.storage,
            topic=topic
        )
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """Run maintenance - compatibility wrapper"""
        # Apply decay
        decay_result = await _apply_decay_impl(self.context.storage)
        
        # Consolidate memories
        consolidate_result = await _consolidate_memories_impl(self.context.storage)
        
        return {
            "memories_decayed": decay_result["decayed_count"],
            "memories_archived": decay_result["archived_count"],
            "clusters_consolidated": consolidate_result["clusters_consolidated"]
        }
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get stats - compatibility wrapper"""
        return await _get_memory_stats_impl(self.context.storage)
    
    async def construct_narrative(self, topic: str, chronological: bool = True,
                                limit: int = 5) -> Dict[str, Any]:
        """Construct narrative - compatibility wrapper"""
        memories = await _search_memories_impl(
            storage=self.context.storage,
            query=topic,
            memory_types=["observation", "experience"],
            limit=limit
        )
        
        if not memories:
            return {
                "narrative": f"I don't have memories about {topic}",
                "confidence": 0.2,
                "experience_count": 0
            }
        
        # Sort if needed
        if chronological:
            memories.sort(key=lambda m: m.get("created_at", ""))
        
        # Build narrative
        narrative_parts = []
        for i, memory in enumerate(memories[:3]):
            timeframe = _get_timeframe_text(memory.get("created_at", ""))
            text = memory.get("memory_text", "")
            narrative_parts.append(f"{timeframe}, {text}")
        
        narrative = f"Regarding {topic}: " + " ".join(narrative_parts)
        
        # Calculate confidence
        avg_relevance = sum(m.get("relevance", 0.5) for m in memories) / len(memories)
        avg_fidelity = sum(m.get("metadata", {}).get("fidelity", 1.0) for m in memories) / len(memories)
        confidence = (0.4 + (len(memories) / 10) + (avg_relevance * 0.3)) * avg_fidelity
        
        return {
            "narrative": narrative,
            "confidence": min(1.0, confidence),
            "experience_count": len(memories)
        }
    
    async def retrieve_experiences(self, query: str, scenario_type: str = "",
                                 limit: int = 3) -> List[Dict[str, Any]]:
        """Retrieve experiences - compatibility wrapper"""
        return await _search_memories_impl(
            storage=self.context.storage,
            query=query,
            memory_types=["experience"],
            limit=limit
        )
    
    async def load_recent_memories(self, memories_data: List[Dict[str, Any]]):
        """Load memories from checkpoint"""
        for mem_data in memories_data:
            # Extract all the data
            memory_text = mem_data.get("memory_text", "")
            memory_type = mem_data.get("memory_type", "observation")
            memory_scope = mem_data.get("memory_scope", "game")
            significance = mem_data.get("significance", 5)
            tags = mem_data.get("tags", [])
            metadata = mem_data.get("metadata", {})
            
            # If the memory has an ID and embedding, load it directly
            if "id" in mem_data and "embedding" in mem_data:
                memory = Memory(
                    id=mem_data["id"],
                    memory_text=memory_text,
                    memory_type=memory_type,
                    memory_scope=memory_scope,
                    significance=significance,
                    tags=tags,
                    metadata=MemoryMetadata(**metadata) if metadata else MemoryMetadata(),
                    embedding=mem_data.get("embedding", []),
                    created_at=mem_data.get("created_at", datetime.datetime.now().isoformat()),
                    times_recalled=mem_data.get("times_recalled", 0),
                    is_archived=mem_data.get("is_archived", False),
                    is_consolidated=mem_data.get("is_consolidated", False)
                )
                
                # Add directly to storage
                memory_id = memory.id
                self.context.storage.memories[memory_id] = memory
                if memory.embedding:
                    self.context.storage.embeddings[memory_id] = memory.embedding
                
                # Manually update indices (don't use add() to avoid duplication)
                storage = self.context.storage
                storage.type_index[memory.memory_type].add(memory_id)
                storage.scope_index[memory.memory_scope].add(memory_id)
                storage.level_index[memory.metadata.memory_level].add(memory_id)
                storage.significance_index[memory.significance].add(memory_id)
                
                for tag in memory.tags:
                    storage.tag_index[tag].add(memory_id)
                
                if memory.is_archived:
                    storage.archived_memories.add(memory_id)
                if memory.is_consolidated:
                    storage.consolidated_memories.add(memory_id)
            else:
                # Create new memory through normal process
                await self.add_memory(
                    memory_text=memory_text,
                    memory_type=memory_type,
                    memory_scope=memory_scope,
                    significance=significance,
                    tags=tags,
                    metadata=metadata
                )

# ==================== Brain Memory Core ====================

class BrainMemoryCore(MemoryCoreAgents):
    """Global memory core"""
    
    def __init__(self):
        super().__init__(user_id=None, conversation_id=None)
        self.omniscient = True

# ==================== Standalone Function Wrappers ====================
# These allow direct imports without needing to use the agent pattern

async def add_memory(
    memory_text: str,
    memory_type: str = "observation",
    memory_scope: str = "game",
    significance: int = 5,
    tags: List[str] = None,
    metadata: Dict[str, Any] = None,
    user_id: Optional[int] = None,
    conversation_id: Optional[int] = None
) -> str:
    """Standalone wrapper for add_memory that can be imported directly"""
    # Create a temporary context
    context = MemoryContext(user_id=user_id, conversation_id=conversation_id)
    storage = context.storage
    
    # Extract metadata fields
    meta = metadata or {}
    
    # Call the implementation directly
    result = await _create_memory_impl(
        storage=storage,
        memory_text=memory_text,
        memory_type=memory_type,
        memory_scope=memory_scope,
        significance=significance,
        tags=tags,
        emotional_context=meta.get('emotional_context'),
        memory_level=meta.get('memory_level', 'detail'),
        source_memory_ids=meta.get('source_memory_ids'),
        entities=meta.get('entities', [])
    )
    
    return result.get("memory_id", "")

async def retrieve_memories(
    query: str,
    memory_types: List[str] = None,
    scopes: List[str] = None,
    limit: int = 10,
    min_significance: int = 3,
    include_archived: bool = False,
    tags: List[str] = None,
    entities: List[str] = None,
    emotional_state: Dict[str, Any] = None,
    retrieval_level: str = "auto",
    min_fidelity: float = 0.0,
    user_id: Optional[int] = None,
    conversation_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Standalone wrapper for retrieve_memories"""
    context = MemoryContext(user_id=user_id, conversation_id=conversation_id)
    storage = context.storage
    
    return await _search_memories_impl(
        storage=storage,
        query=query,
        memory_types=memory_types,
        scopes=scopes,
        limit=limit,
        min_significance=min_significance,
        include_archived=include_archived,
        tags=tags,
        entities=entities,
        emotional_state=emotional_state,
        retrieval_level=retrieval_level,
        min_fidelity=min_fidelity
    )

# ==================== Module Exports ====================
# Make sure all commonly imported items are available

__all__ = [
    # Classes
    'MemoryCoreAgents',
    'BrainMemoryCore',
    'Memory',
    'MemoryMetadata',
    'EmotionalMemoryContext',
    'MemorySchema',
    'MemoryContext',
    'MemoryStorage',
    
    # Input/Output Models
    'MemoryCreateParams',
    'MemoryUpdateParams',
    'MemoryQuery',
    'MemoryRetrieveResult',
    'MemoryMaintenanceResult',
    'NarrativeResult',
    
    # Standalone functions
    'add_memory',
    'retrieve_memories',
    
    # Tool functions
    'create_memory',
    'search_memories',
    'update_memory',
    'delete_memory',
    'get_memory',
    'get_memory_details',
    'archive_memory',
    'unarchive_memory',
    'crystallize_memory',
    'mark_as_consolidated',
    'create_reflection',
    'create_abstraction',
    'create_semantic_memory',
    'apply_decay',
    'consolidate_memories',
    'get_memory_stats',
    'generate_conversational_recall',
    'construct_narrative',
    'run_maintenance',
    'retrieve_memories_with_formatting',
    'retrieve_relevant_experiences',
    'detect_schema_from_memories',
    'reflect_on_memories',
]
