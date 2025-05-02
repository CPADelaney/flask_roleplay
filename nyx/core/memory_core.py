# nyx/core/memory_core.py

import asyncio
import datetime
import json
import logging
import random
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Literal
from collections import defaultdict
import numpy as np
from pydantic import BaseModel, Field


# OpenAI Agents SDK imports
from agents import Agent, Runner, function_tool, handoff, trace, custom_span
from agents import RunContextWrapper, ModelSettings

logger = logging.getLogger(__name__)

# Pydantic models for memory operations
class EmotionalMemoryContext(BaseModel):
    """Emotional context for memories"""
    primary_emotion: str
    primary_intensity: float
    secondary_emotions: Dict[str, float] = Field(default_factory=dict)
    valence: float = 0.0
    arousal: float = 0.0

class MemorySchema(BaseModel):
    """Schema that represents patterns Nyx has identified in its experiences"""
    id: str
    name: str
    description: str
    confidence: float = 0.7
    category: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    example_memory_ids: List[str] = Field(default_factory=list)
    creation_date: str
    last_updated: str
    usage_count: int = 0
    evolution_history: List[Dict[str, Any]] = Field(default_factory=list)

class MemoryMetadata(BaseModel):
    """Enhanced metadata including hierarchical information."""
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

    # --- NEW HIERARCHICAL & FIDELITY FIELDS ---
    memory_level: Literal['detail', 'summary', 'abstraction'] = Field(
        default='detail',
        description="Level of detail/abstraction of this memory"
    )
    source_memory_ids: Optional[List[str]] = Field(
        default=None,
        description="IDs of source memories if this is a summary/abstraction"
    )
    fidelity: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the accuracy/reliability of this memory detail (0.0-1.0)"
    )
    summary_of: Optional[str] = Field(
        default=None,
        description="Brief description of what this summarizes, if applicable (e.g., 'User conversation on topic X')"
    )

class Memory(BaseModel):
    """Core memory object."""
    id: str
    memory_text: str
    memory_type: str # observation, reflection, experience, summary, abstraction, semantic, etc.
    memory_scope: str # game, user, global
    significance: int # 1-10 importance
    tags: List[str] = Field(default_factory=list)
    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata)
    embedding: List[float] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    times_recalled: int = 0
    is_archived: bool = False
    is_consolidated: bool = False # Kept for compatibility, indicates it was *used* in consolidation

class MemoryQuery(BaseModel):
    """Input model for retrieving memories."""
    query: str
    memory_types: List[str] = Field(default_factory=lambda: ["observation", "reflection", "abstraction", "experience", "summary", "semantic"])
    scopes: List[str] = Field(default_factory=lambda: ["game", "user", "global"])
    limit: int = 10 # Adjusted default limit
    min_significance: int = 3
    include_archived: bool = False
    entities: Optional[List[str]] = None
    emotional_state: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    # --- NEW PARAMS ---
    retrieval_level: Literal['detail', 'summary', 'abstraction', 'auto'] = 'auto'
    min_fidelity: Optional[float] = 0.0 # Default allows all fidelities initially

class MemoryCreateParams(BaseModel):
    """Input model for creating memories."""
    memory_text: str
    memory_type: str = "observation"
    memory_scope: str = "game"
    significance: int = 5
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict) # Base metadata passed in
    # --- NEW PARAMS ---
    memory_level: Literal['detail', 'summary', 'abstraction'] = 'detail'
    source_memory_ids: Optional[List[str]] = None
    fidelity: float = 1.0
    summary_of: Optional[str] = None # Optional description of what's summarized

class MemoryUpdateParams(BaseModel):
     """Input model for updating memories."""
     memory_id: str
     updates: Dict[str, Any] # Fields to update, can include nested metadata changes

class MemoryRetrieveResult(BaseModel):
    """Structured result for retrieval, including hierarchical info."""
    memory_id: str
    memory_text: str
    memory_type: str
    significance: int
    relevance: float
    confidence_marker: Optional[str] = None
    # --- NEW FIELDS ---
    memory_level: Literal['detail', 'summary', 'abstraction']
    source_memory_ids: Optional[List[str]] = None
    fidelity: float
    summary_of: Optional[str] = None

class MemoryMaintenanceResult(BaseModel):
    memories_decayed: int
    clusters_consolidated: int
    memories_archived: int

class NarrativeResult(BaseModel):
    narrative: str
    confidence: float
    experience_count: int

class MemoryCoreContext:
    """Context object for memory core operations (Enhanced)"""
    def __init__(self, user_id: Optional[int], conversation_id: Optional[int]):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.memories: Dict[str, Memory] = {} # Store Memory model instances
        self.memory_embeddings: Dict[str, List[float]] = {}
        self.type_index: Dict[str, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.scope_index: Dict[str, Set[str]] = defaultdict(set)
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
        self.emotional_index: Dict[str, Set[str]] = defaultdict(set)
        self.schemas: Dict[str, MemorySchema] = {}
        self.schema_index: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_index: List[Tuple[datetime.datetime, str]] = []
        self.significance_index: Dict[int, Set[str]] = defaultdict(set)
        self.archived_memories: Set[str] = set()
        self.consolidated_memories: Set[str] = set()

        # --- NEW INDEX & LINKS ---
        self.level_index: Dict[str, Set[str]] = defaultdict(set) # 'detail', 'summary', 'abstraction' -> {ids}
        self.summary_links: Dict[str, List[str]] = defaultdict(list) # detail_id -> [summary_ids] (optional reverse link)

        # --- Configuration ---
        self.embed_dim = 1536 # Example dimension
        self.max_memory_age_days = 90
        self.decay_rate = 0.1
        self.consolidation_threshold = 0.85
        self.reconsolidation_probability = 0.3 # Probability of reconsolidation on recall
        self.reconsolidation_strength = 0.1 # How much text changes during reconsolidation
        self.abstraction_threshold = 3
        self.query_cache: Dict[str, Tuple[datetime.datetime, List[str]]] = {} # query_key -> (timestamp, [memory_ids])
        self.cache_ttl = 300 # seconds
        self.initialized = False
        self.init_time = datetime.datetime.now()
        self.recall_templates: Dict[str, List[str]] = {
            # Basic recall templates
            "standard": [
                "That reminds me of {timeframe} when {brief_summary}... {detail}",
                "I recall {timeframe} when {brief_summary}. {reflection}",
                "This is similar to {timeframe} when {brief_summary}... {detail}"
            ],
            
            # For emotionally positive experiences
            "positive": [
                "Mmm, I remember {timeframe} when {brief_summary}. {reflection}",
                "That brings back a delicious memory of {timeframe} when {brief_summary}... {detail}",
                "I quite enjoyed {timeframe} when {brief_summary}. {reflection}"
            ],
            
            # For emotionally negative experiences
            "negative": [
                "I recall {timeframe} dealing with someone who {brief_summary}. {reflection}",
                "This reminds me of a frustrating time when {brief_summary}... {detail}",
                "I once had to handle someone who {brief_summary}. {reflection}"
            ],
            
            # For intense experiences
            "intense": [
                "Mmm, that reminds me of an *intense* experience where {brief_summary}... {detail}",
                "I vividly remember when {brief_summary}. {reflection}",
                "I'll never forget when {brief_summary}... {detail}"
            ],
            
            # For teasing experiences
            "teasing": [
                "Oh, this reminds me of {timeframe} when I teased someone until {brief_summary}... {reflection}",
                "I once had such fun teasing someone who {brief_summary}. {reflection}",
                "There was this delicious time when I {brief_summary}... {detail}"
            ],
            
            # For disciplinary experiences
            "disciplinary": [
                "I remember having to discipline someone who {brief_summary}. {reflection}",
                "This reminds me of {timeframe} when I had to correct someone who {brief_summary}... {detail}",
                "I once dealt with someone who needed strict handling when they {brief_summary}. {reflection}"
            ]
        }
        
        self.level_index: Dict[str, Set[str]] = {"detail": set(), "summary": set(), "abstraction": set()}
        
        # Confidence marker mapping for experience relevance scores
        self.confidence_markers: Dict[Tuple[float, float], str] = {
            (0.8, 1.01): "vividly recall", # Use 1.01 to include 1.0
            (0.6, 0.8): "clearly remember",
            (0.4, 0.6): "remember",
            (0.2, 0.4): "think I recall",
            (0.0, 0.2): "vaguely remember"
        }
    
    def initialize(self):
        if self.initialized: return
        # Initialize indices
        for level in ['detail', 'summary', 'abstraction']: self.level_index[level] = set()
        for i in range(1, 11): self.significance_index[i] = set()
        for t in ["observation", "reflection", "abstraction", "experience", "summary", "semantic"]: self.type_index[t]
        for s in ["game", "user", "global"]: self.scope_index[s]
        self.initialized = True
        logger.info(f"MemoryCoreContext initialized for user {self.user_id}")

# Utility functions for memory operations

async def _generate_embedding(ctx: RunContextWrapper[MemoryCoreContext], text: str) -> List[float]:
    """Generate an embedding vector for text"""
    with custom_span("generate_embedding", {"text_length": len(text)}):
        # Simplified embedding generation for this implementation
        # In a real implementation, this would call an embedding API
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        
        # Convert to a pseudo-embedding
        random.seed(hash_val)
        embed_dim = ctx.context.embed_dim
        return [random.uniform(-1, 1) for _ in range(embed_dim)]

def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    
    if norm1 * norm2 == 0:
        return 0.0
        
    return dot_product / (norm1 * norm2)

def _calculate_emotional_relevance(memory_emotion: Dict[str, Any], current_emotion: Dict[str, Any]) -> float:
    """Calculate emotional relevance boost based on emotional context"""
    # Get primary emotions
    memory_primary = memory_emotion.get("primary_emotion", "neutral")
    current_primary = current_emotion.get("primary_emotion", "neutral")
    
    # Primary emotion match gives highest boost
    if memory_primary == current_primary:
        return 0.2
    
    # Secondary emotion matches give smaller boost
    memory_secondary = memory_emotion.get("secondary_emotions", {})
    if current_primary in memory_secondary:
        return 0.1
    
    # Valence match (positive/negative alignment)
    memory_valence = memory_emotion.get("valence", 0.0)
    current_valence = current_emotion.get("valence", 0.0)
    
    # Similar valence gives small boost
    if abs(memory_valence - current_valence) < 0.5:
        return 0.05
    
    return 0.0

def _get_confidence_marker(ctx: RunContextWrapper[MemoryCoreContext], relevance: float) -> str:
    """Get confidence marker text based on relevance score"""
    for (min_val, max_val), marker in ctx.context.confidence_markers.items():
        if min_val <= relevance < max_val:
            return marker
    return "remember"  # Default

def _get_timeframe_text(timestamp: Optional[str]) -> str:
    """Get conversational timeframe text from timestamp"""
    if not timestamp:
        return "a while back"
        
    try:
        if isinstance(timestamp, str):
            memory_time = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        else:
            memory_time = timestamp
            
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
        elif days_ago < 60:
            return "a month ago"
        elif days_ago < 365:
            return f"{days_ago // 30} months ago"
        else:
            return "a while back"
            
    except Exception as e:
        logger.error(f"Error processing timestamp: {e}")
        return "a while back"

def _get_emotional_tone(emotional_context: Dict[str, Any]) -> str:
    """Determine the emotional tone for recall based on the experience's emotions"""
    if not emotional_context:
        return "standard"
        
    primary = emotional_context.get("primary_emotion", "neutral")
    intensity = emotional_context.get("primary_intensity", 0.5)
    valence = emotional_context.get("valence", 0.0)
    
    # High intensity experiences
    if intensity > 0.8:
        return "intense"
        
    # Positive emotions
    if valence > 0.3 or primary in ["Joy", "Anticipation", "Trust", "Love"]:
        return "positive"
        
    # Negative emotions
    if valence < -0.3 or primary in ["Anger", "Fear", "Disgust", "Sadness", "Frustration"]:
        return "negative"
        
    # Default to standard
    return "standard"

def _get_scenario_tone(scenario_type: str) -> Optional[str]:
    """Get tone based on scenario type"""
    scenario_type = scenario_type.lower()
    
    if scenario_type in ["teasing", "indulgent"]:
        return "teasing"
    elif scenario_type in ["discipline", "punishment", "training"]:
        return "disciplinary"
    elif scenario_type in ["dark", "fear"]:
        return "intense"
    
    # No specific tone for this scenario type
    return None

def _alter_memory_text(text: str, original_form: str, strength: float) -> str:
    """Alter memory text to simulate reconsolidation"""
    # Simple word-level alterations
    words = text.split()
    num_alterations = max(1, int(len(words) * strength))
    
    for _ in range(num_alterations):
        # Select a random word to alter
        idx = random.randint(0, len(words) - 1)
        
        # Skip short words
        if len(words[idx]) <= 3:
            continue
        
        # Choose an alteration
        alteration_type = random.choice(["intensify", "weaken", "replace", "qualifier"])
        
        if alteration_type == "intensify":
            words[idx] = f"very {words[idx]}"
        elif alteration_type == "weaken":
            words[idx] = f"somewhat {words[idx]}"
        elif alteration_type == "qualifier":
            words[idx] = f"{words[idx]} perhaps"
        elif alteration_type == "replace":
            # Replace with a similar word (simplified)
            if words[idx].endswith("ed"):
                words[idx] = words[idx].replace("ed", "ing")
            elif words[idx].endswith("ing"):
                words[idx] = words[idx].replace("ing", "ed")
            elif words[idx].endswith("ly"):
                words[idx] = words[idx].replace("ly", "")
            elif len(words[idx]) > 5:
                words[idx] = words[idx][:len(words[idx])-2]
    
    return " ".join(words)

def _calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts"""
    # Simple implementation - in production use embedding similarity
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union

async def _update_memory_recall(ctx: RunContextWrapper[MemoryCoreContext], memory_id: str):
    """Update recall count and timestamp for a memory"""
    memory_core = ctx.context
    if memory_id in memory_core.memories:
        memory = memory_core.memories[memory_id]
        memory["times_recalled"] = memory.get("times_recalled", 0) + 1
        memory["metadata"] = memory.get("metadata", {})
        memory["metadata"]["last_recalled"] = datetime.datetime.now().isoformat()

async def _generate_abstraction(ctx: RunContextWrapper[MemoryCoreContext], memories: List[Dict[str, Any]], abstraction_type: str) -> Optional[str]:
    """Generate an abstraction from a set of memories"""
    # Extract memory texts
    memory_texts = [memory["memory_text"] for memory in memories]
    
    # Simple abstract generation by template
    if abstraction_type == "pattern":
        return f"A pattern has emerged: {memory_texts[0][:50]}... (and {len(memory_texts)-1} similar memories)"
    elif abstraction_type == "summary":
        return f"Summary of {len(memory_texts)} related memories: {memory_texts[0][:30]}... and others with similar themes."
    elif abstraction_type == "belief":
        return f"Based on multiple experiences, I have developed the belief that: {memory_texts[0][:50]}..."
    else:
        return f"Abstract concept derived from {len(memory_texts)} memories with similar content."

async def _generate_consolidated_text(ctx: RunContextWrapper[MemoryCoreContext], memory_texts: List[str]) -> str:
    """Generate a consolidated memory text from multiple memory texts"""
    with custom_span("generate_consolidated_text", {"num_texts": len(memory_texts)}):
        # For real implementation, use LLM to generate a coherent consolidated memory
        # For this implementation, use a simple approach
        combined = " ".join(memory_texts)
        # Truncate if too long
        if len(combined) > 500:
            combined = combined[:497] + "..."
        
        return f"I've observed a pattern in several memories: {combined}"

async def _apply_reconsolidation(ctx: RunContextWrapper[MemoryCoreContext], memory: Dict[str, Any]) -> Dict[str, Any]:
    """Apply reconsolidation to a memory (simulate memory alteration on recall)"""
    memory_core = ctx.context
    memory_id = memory["id"]
    memory_text = memory["memory_text"]
    metadata = memory.get("metadata", {})
    
    # Skip recent memories
    timestamp_str = metadata.get("timestamp", datetime.datetime.now().isoformat())
    timestamp = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    if (datetime.datetime.now() - timestamp).days < 7:
        return memory
    
    # Get original form
    original_form = metadata.get("original_form", memory_text)
    
    # Initialize reconsolidation history if needed
    if "reconsolidation_history" not in metadata:
        metadata["reconsolidation_history"] = []
    
    # Add current version to history
    metadata["reconsolidation_history"].append({
        "previous_text": memory_text,
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    # Keep only last 3 versions to avoid metadata bloat
    if len(metadata["reconsolidation_history"]) > 3:
        metadata["reconsolidation_history"] = metadata["reconsolidation_history"][-3:]
    
    # Generate altered text
    altered_text = _alter_memory_text(
        memory_text,
        original_form,
        memory_core.reconsolidation_strength
    )
    
    # Update memory fidelity
    current_fidelity = metadata.get("fidelity", 1.0)
    new_fidelity = max(0.3, current_fidelity - (memory_core.reconsolidation_strength * 0.1))
    metadata["fidelity"] = new_fidelity
    
    # Update memory
    await update_memory(
        ctx,
        memory_id,
        {
            "memory_text": altered_text,
            "metadata": metadata
        }
    )
    
    # Return updated memory
    memory["memory_text"] = altered_text
    memory["metadata"] = metadata
    
    return memory

async def _apply_reconsolidation_to_results(ctx: RunContextWrapper[MemoryCoreContext], memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply reconsolidation to retrieved memories"""
    memory_core = ctx.context
    
    # Check each memory for potential reconsolidation
    for i, memory in enumerate(memories):
        # Skip memory types that shouldn't reconsolidate
        if memory["memory_type"] in ["semantic", "consolidated"]:
            continue
            
        # Apply reconsolidation with probability
        if random.random() < memory_core.reconsolidation_probability:
            memory = await _apply_reconsolidation(ctx, memory)
            memories[i] = memory
    
    return memories

async def _detect_memory_pattern(ctx: RunContextWrapper[MemoryCoreContext], memories: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Detect a pattern in a set of memories"""
    # Check if we have enough memories
    if len(memories) < 3:
        return None
    
    # Extract memory texts and tags
    memory_texts = [memory["memory_text"] for memory in memories]
    
    # Count tag frequencies
    tag_counts = {}
    for memory in memories:
        for tag in memory.get("tags", []):
            if tag not in tag_counts:
                tag_counts[tag] = 0
            tag_counts[tag] += 1
    
    # Top tags
    top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    
    # If no common tags, use simple text analysis
    if not top_tags:
        # Simplified - in production use more sophisticated NLP
        all_text = " ".join(memory_texts).lower()
        words = all_text.split()
        word_counts = {}
        
        for word in words:
            if len(word) > 4:  # Skip short words
                if word not in word_counts:
                    word_counts[word] = 0
                word_counts[word] += 1
        
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if not top_words:
            return None
        
        # Create a pattern from top words
        pattern_name = f"Pattern of {top_words[0][0]}"
        description = f"A recurring pattern involving {', '.join(w[0] for w in top_words)}"
        
        return {
            "name": pattern_name,
            "description": description,
            "category": "text_pattern",
            "attributes": {
                "key_elements": [w[0] for w in top_words],
                "confidence": 0.5
            }
        }
    
    # Use top tags for pattern
    pattern_tag = top_tags[0][0]
    pattern_name = f"{pattern_tag.capitalize()} pattern"
    description = f"A recurring pattern of experiences tagged with '{pattern_tag}'"
    
    # Additional tags for attributes
    attributes = {
        "primary_tag": pattern_tag,
        "frequency": top_tags[0][1] / len(memories),
        "confidence": min(1.0, top_tags[0][1] / len(memories) + 0.3)
    }
    
    if len(top_tags) > 1:
        attributes["secondary_tags"] = [t[0] for t in top_tags[1:3]]
    
    return {
        "name": pattern_name,
        "description": description,
        "category": "tag_pattern",
        "attributes": attributes
    }

async def _check_for_patterns(ctx: RunContextWrapper[MemoryCoreContext], tags: List[str]) -> None:
    """Check if there are patterns that should be identified as schemas"""
    memory_core = ctx.context
    
    # Skip if no tags
    if not tags:
        return
    
    # Check each tag for potential patterns
    for tag in tags:
        # Get memories with this tag
        tagged_memories = await ries(
            ctx,
            query="",
            memory_types=["observation", "experience"],
            limit=10,
            tags=[tag]
        )
        
        # Skip if not enough memories
        if len(tagged_memories) < 3:
            continue
        
        # Check if there might be a schema
        if tag not in [s.name.lower() for s in memory_core.schemas.values()]:
            # Create schema detection task
            asyncio.create_task(detect_schema_from_memories(ctx, tag))


# Memory operations function tools

@function_tool
async def add_memory(
    ctx: RunContextWrapper[MemoryCoreContext],
    params: MemoryCreateParams
) -> str:
    """
    Add a new memory to the system, including hierarchical details.
    Restored full indexing and pattern check call.
    """
    with custom_span("add_memory", {"memory_type": params.memory_type, "memory_level": params.memory_level}):
        memory_core = ctx.context
        if not memory_core.initialized: memory_core.initialize()

        memory_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()

        metadata = MemoryMetadata(
            timestamp=timestamp,
            fidelity=params.fidelity,
            memory_level=params.memory_level,
            source_memory_ids=params.source_memory_ids,
            summary_of=params.summary_of,
            original_form=params.memory_text, # Store original form on creation
            **params.metadata
        )
        if 'emotional_context' in params.metadata and isinstance(params.metadata['emotional_context'], dict):
            metadata.emotional_context = EmotionalMemoryContext(**params.metadata['emotional_context'])

        embedding = await _generate_embedding(ctx, params.memory_text)

        memory_data = Memory(
            id=memory_id, memory_text=params.memory_text, memory_type=params.memory_type,
            memory_scope=params.memory_scope, significance=params.significance,
            tags=params.tags or [], metadata=metadata, embedding=embedding,
            created_at=timestamp
        )

        memory_core.memories[memory_id] = memory_data
        memory_core.memory_embeddings[memory_id] = embedding

        # --- Update Indices ---
        memory_core.type_index[params.memory_type].add(memory_id)
        memory_core.scope_index[params.memory_scope].add(memory_id)
        memory_core.level_index[params.memory_level].add(memory_id)
        sig_level = min(10, max(1, int(params.significance)))
        memory_core.significance_index[sig_level].add(memory_id)
        for tag in params.tags or []: memory_core.tag_index[tag].add(memory_id)
        dt_timestamp = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        memory_core.temporal_index.append((dt_timestamp, memory_id))
        memory_core.temporal_index.sort(key=lambda x: x[0])
        for entity_id in metadata.entities: memory_core.entity_index[entity_id].add(memory_id)
        if metadata.emotional_context and metadata.emotional_context.primary_emotion:
             memory_core.emotional_index[metadata.emotional_context.primary_emotion].add(memory_id)
        for schema_ref in metadata.schemas:
             if schema_id := schema_ref.get("schema_id"): memory_core.schema_index[schema_id].add(memory_id)
        if params.memory_level in ['summary', 'abstraction'] and params.source_memory_ids:
            for source_id in params.source_memory_ids: memory_core.summary_links[source_id].append(memory_id)

        memory_core.query_cache = {}
        # --- Call _check_for_patterns (Restored) ---
        if params.memory_type == "observation" and (params.tags or []):
            asyncio.create_task(_check_for_patterns(ctx, params.tags))

        logger.debug(f"Added {params.memory_level} memory {memory_id} of type {params.memory_type}")
        return memory_id

@function_tool
async def retrieve_memories(
    ctx: RunContextWrapper[MemoryCoreContext],
    query: str,
    memory_types: Optional[List[str]] = None,
    limit: Optional[int] = None,
    min_significance: Optional[int] = None,
    include_archived: Optional[bool] = None,
    entities: Optional[List[str]] = None,
    emotional_state: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    retrieval_level: str = "auto",
    min_fidelity: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve memories based on query and filters, supporting hierarchical levels.
    Restored full boost logic. Returns raw memory dicts.
    """
    params = MemoryQuery(
        query=query,
        memory_types=memory_types or ["observation", "reflection", "abstraction", "experience", "summary", "semantic"],
        limit=limit or 10,  # Default value applied in code, not in schema
        min_significance=min_significance or 3,
        include_archived=include_archived or False,
        entities=entities,
        emotional_state=emotional_state,
        tags=tags,
        retrieval_level=retrieval_level,
        min_fidelity=min_fidelity or 0.0
    )

    with custom_span("retrieve_memories", {
        "query": params.query, "level": params.retrieval_level, "limit": params.limit, "fidelity": params.min_fidelity
    }):

        memory_core = ctx.context
        if not memory_core.initialized: memory_core.initialize()

        # Cache key generation (ensure stability)
        param_items = sorted([(k, v) for k, v in params.model_dump().items() if v is not None])
        cache_key_str = "&".join([f"{k}={json.dumps(v, sort_keys=True)}" for k, v in param_items])
        cache_key = f"retrieve:{hash(cache_key_str)}"

        if cache_key in memory_core.query_cache:
             cache_time, cached_ids = memory_core.query_cache[cache_key]
             if (datetime.datetime.now() - cache_time).total_seconds() < memory_core.cache_ttl:
                 results_models = [memory_core.memories[mid] for mid in cached_ids if mid in memory_core.memories]
                 # Apply reconsolidation on cache hit if desired
                 results_models = await _apply_reconsolidation_to_results(ctx, results_models)
                 # Convert Pydantic models back to dicts for return
                 results = [mem.model_dump() for mem in results_models]
                 # Add relevance back (or recalculate if needed, though cache implies stable relevance)
                 for i, mem_id in enumerate(cached_ids):
                      if i < len(results): # Ensure index exists
                           # Note: Relevance isn't stored in cache, might need recalculation or approximation
                           results[i]['relevance'] = 0.8 # Placeholder relevance for cached items
                 logger.debug(f"Cache hit for retrieve_memories: {params.query}")
                 return results

        query_embedding = await _generate_embedding(ctx, params.query)

        # 1. Initial Filtering (Optimized Set Operations)
        candidate_ids = set(memory_core.memories.keys())
        candidate_ids &= set().union(*(memory_core.type_index.get(mt, set()) for mt in params.memory_types))
        candidate_ids &= set().union(*(memory_core.scope_index.get(s, set()) for s in params.scopes))
        candidate_ids &= set().union(*(memory_core.significance_index.get(i, set()) for i in range(params.min_significance, 11)))
        if params.tags:
            tag_matches = set(candidate_ids) # Start with current candidates
            for i, tag in enumerate(params.tags):
                ids_for_tag = memory_core.tag_index.get(tag, set())
                if i == 0: tag_matches = ids_for_tag
                else: tag_matches &= ids_for_tag
            candidate_ids &= tag_matches
        if params.entities:
            # Option 1: Memory must match ANY entity (Union)
            # entity_matches = set().union(*(memory_core.entity_index.get(e, set()) for e in params.entities))
            # Option 2: Memory must match ALL entities (Intersection)
            entity_matches = set(candidate_ids) # Start with current
            for i, entity in enumerate(params.entities):
                ids_for_entity = memory_core.entity_index.get(entity, set())
                if i == 0: entity_matches = ids_for_entity
                else: entity_matches &= ids_for_entity
            candidate_ids &= entity_matches

        if not params.include_archived: candidate_ids -= memory_core.archived_memories

        # 3. Level and Fidelity Filtering
        final_candidate_ids = set()
        for mem_id in candidate_ids:
            memory = memory_core.memories.get(mem_id)
            if not memory: continue
            passes_level = (params.retrieval_level == 'auto' or memory.metadata.memory_level == params.retrieval_level)
            passes_fidelity = memory.metadata.fidelity >= params.min_fidelity
            if passes_level and passes_fidelity: final_candidate_ids.add(mem_id)

        # 4. Relevance Scoring & Sorting
        scored_candidates = []
        for memory_id in final_candidate_ids:
            embedding = memory_core.memory_embeddings.get(memory_id)
            if not embedding: continue
            relevance = _cosine_similarity(query_embedding, embedding)
            memory = memory_core.memories[memory_id]

            # --- Boosts (Restored Logic) ---
            entity_boost = 0.0
            if params.entities and memory.metadata.entities:
                 common_entities = set(params.entities) & set(memory.metadata.entities)
                 entity_boost = min(0.2, len(common_entities) * 0.05) # Cap boost

            emotional_boost = _calculate_emotional_relevance(memory.metadata.emotional_context, params.emotional_state or {})

            schema_boost = min(0.1, len(memory.metadata.schemas) * 0.05) if memory.metadata.schemas else 0.0

            dt_timestamp = datetime.datetime.fromisoformat(memory.metadata.timestamp.replace("Z", "+00:00"))
            days_old = max(0, (datetime.datetime.now() - dt_timestamp).days)
            temporal_boost = max(0.0, 0.15 * math.exp(-0.1 * days_old)) # Exponential decay boost for recency

            level_boost = 0.0
            if params.retrieval_level == 'auto' and memory.metadata.memory_level == 'detail': level_boost = 0.1
            elif params.retrieval_level == 'detail' and memory.metadata.memory_level == 'detail': level_boost = 0.15 # Stronger boost if detail explicitly requested
            elif params.retrieval_level == 'summary' and memory.metadata.memory_level == 'summary': level_boost = 0.1
            elif params.retrieval_level == 'abstraction' and memory.metadata.memory_level == 'abstraction': level_boost = 0.1

            # --- Final Score ---
            fidelity_factor = 0.7 + (0.3 * memory.metadata.fidelity) # Weighted fidelity influence
            final_relevance = min(1.0, (relevance + entity_boost + emotional_boost + schema_boost + temporal_boost + level_boost) * fidelity_factor)

            scored_candidates.append((memory_id, final_relevance))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # 5. Limit and Prepare Results
        top_memory_ids = [mid for mid, _ in scored_candidates[:params.limit]]
        results_models = [memory_core.memories[mid] for mid in top_memory_ids]

        # 6. Apply Reconsolidation (Moved outside core retrieval logic, apply if needed by caller)
        # results_models = await _apply_reconsolidation_to_results(ctx, results_models)

        # 7. Prepare results as dicts and add relevance
        results = []
        for i, mem_id in enumerate(top_memory_ids):
            relevance_score = scored_candidates[i][1]
            mem_dict = results_models[i].model_dump()
            mem_dict["relevance"] = relevance_score
            results.append(mem_dict)
            await _update_memory_recall(ctx, mem_id)

        # 8. Cache results (IDs only)
        memory_core.query_cache[cache_key] = (datetime.datetime.now(), top_memory_ids)
        logger.debug(f"Retrieved {len(results)} memories for query: {params.query}")

        return results

@function_tool
async def get_memory(
    ctx: RunContextWrapper[MemoryCoreContext], 
    memory_id: str
) -> Optional[Dict[str, Any]]:
    """
    Get a specific memory by ID
    
    Args:
        memory_id: The unique ID of the memory to retrieve
        
    Returns:
        Memory object or None if not found
    """
    memory_core = ctx.context
    if not memory_core.initialized:
        memory_core.initialize()
    
    if memory_id in memory_core.memories:
        memory = memory_core.memories[memory_id]
        memory_copy = memory.model_copy(deep=True)
        
        # Update recall count
        await _update_memory_recall(ctx, memory_id)
        
        # Apply reconsolidation if appropriate
        if random.random() < memory_core.reconsolidation_probability:
            memory_copy = await _apply_reconsolidation(ctx, memory_copy)
        
        return memory_copy.model_dump()
    
    return None

@function_tool
async def get_memory_details(
    ctx: RunContextWrapper[MemoryCoreContext],
    memory_ids: List[str],
    min_fidelity: float = 0.0, # Changed Optional[float] to float
) -> List[Dict[str, Any]]:
    """
    Retrieve full details for a specific list of memory IDs ('zoom-in').
    Filters for 'detail' level memories and minimum fidelity.
    """
    with custom_span("get_memory_details", {"num_ids": len(memory_ids), "min_fidelity": min_fidelity}):
        memory_core = ctx.context
        if not memory_core.initialized: memory_core.initialize()

        detailed_memories = []
        for mem_id in memory_ids:
            memory = memory_core.memories.get(mem_id)
            if memory:
                 # Ensure it's a detail memory and meets fidelity
                 if memory.metadata.memory_level == 'detail' and memory.metadata.fidelity >= min_fidelity:
                     # Apply reconsolidation *before* returning details? Decision point.
                     # Let's assume zoom-in implies trying to get the *current* state including reconsolidation effects.
                     memory_copy = memory.model_copy(deep=True) # Work on a copy
                     if random.random() < memory_core.reconsolidation_probability:
                         memory_copy = await _apply_reconsolidation(ctx, memory_copy)
                         # If _apply_reconsolidation saved changes, fine. If not, we return the altered copy.

                     mem_dict = memory_copy.model_dump()
                     mem_dict["relevance"] = 1.0 # Assume high relevance when directly requested
                     detailed_memories.append(mem_dict)
                     await _update_memory_recall(ctx, mem_id) # Mark original ID as recalled

        logger.debug(f"Retrieved details for {len(detailed_memories)} out of {len(memory_ids)} requested IDs.")
        return detailed_memories

@function_tool
async def update_memory(
    ctx: RunContextWrapper[MemoryCoreContext],
    params: MemoryUpdateParams
) -> bool:
    """
    Update an existing memory, handling index updates for hierarchical fields.
    Restored more robust index update logic.
    """
    memory_id = params.memory_id
    updates = params.updates
    with custom_span("update_memory", {"memory_id": memory_id}):
        memory_core = ctx.context
        if not memory_core.initialized: memory_core.initialize()

        if memory_id not in memory_core.memories:
            logger.warning(f"Attempted to update non-existent memory: {memory_id}")
            return False

        memory = memory_core.memories[memory_id]
        original_data = memory.model_copy(deep=True)

        # --- Apply Updates ---
        try:
            for key, value in updates.items():
                if key == "metadata" and isinstance(value, dict):
                    # Use Pydantic's update mechanism if available or manual merge
                    current_meta_dict = memory.metadata.model_dump()
                    # Merge carefully, potentially handling nested dicts if needed
                    updated_meta_dict = {**current_meta_dict, **value}
                    # Handle nested Pydantic models within metadata
                    if 'emotional_context' in updated_meta_dict and isinstance(updated_meta_dict['emotional_context'], dict):
                         # Validate and create/update the nested model
                        try:
                            updated_meta_dict['emotional_context'] = EmotionalMemoryContext(**updated_meta_dict['emotional_context'])
                        except Exception as pydantic_error:
                             logger.error(f"Pydantic validation error updating emotional_context for {memory_id}: {pydantic_error}")
                             # Decide how to handle - skip update, use original, etc.
                             updated_meta_dict['emotional_context'] = original_data.metadata.emotional_context # Revert on error

                    memory.metadata = MemoryMetadata(**updated_meta_dict)
                elif hasattr(memory, key):
                    setattr(memory, key, value)
        except Exception as e:
            logger.error(f"Error applying updates to memory {memory_id}: {e}", exc_info=True)
            memory_core.memories[memory_id] = original_data
            return False

        # --- Update Embedding ---
        if memory.memory_text != original_data.memory_text:
            memory.embedding = await _generate_embedding(ctx, memory.memory_text)
            memory_core.memory_embeddings[memory_id] = memory.embedding

        # --- Update Indices (using comparison with original_data) ---
        def update_index_set(index_dict, key, old_value, new_value):
            if old_value in index_dict: index_dict[old_value].discard(memory_id)
            if new_value: index_dict[new_value].add(memory_id)

        update_index_set(memory_core.type_index, 'memory_type', original_data.memory_type, memory.memory_type)
        update_index_set(memory_core.scope_index, 'memory_scope', original_data.memory_scope, memory.memory_scope)
        update_index_set(memory_core.level_index, 'memory_level', original_data.metadata.memory_level, memory.metadata.memory_level)

        old_sig_level = min(10, max(1, int(original_data.significance)))
        new_sig_level = min(10, max(1, int(memory.significance)))
        if old_sig_level != new_sig_level:
             memory_core.significance_index[old_sig_level].discard(memory_id)
             memory_core.significance_index[new_sig_level].add(memory_id)

        old_tags = set(original_data.tags)
        new_tags = set(memory.tags)
        for tag in old_tags - new_tags: memory_core.tag_index[tag].discard(memory_id)
        for tag in new_tags - old_tags: memory_core.tag_index[tag].add(memory_id)

        old_entities = set(original_data.metadata.entities)
        new_entities = set(memory.metadata.entities)
        for entity in old_entities - new_entities: memory_core.entity_index[entity].discard(memory_id)
        for entity in new_entities - old_entities: memory_core.entity_index[entity].add(memory_id)

        old_emotion = original_data.metadata.emotional_context.primary_emotion if original_data.metadata.emotional_context else None
        new_emotion = memory.metadata.emotional_context.primary_emotion if memory.metadata.emotional_context else None
        if old_emotion != new_emotion:
             if old_emotion: memory_core.emotional_index[old_emotion].discard(memory_id)
             if new_emotion: memory_core.emotional_index[new_emotion].add(memory_id)

        # Schema index update (more complex if relevance changes)
        old_schema_ids = {s.get("schema_id") for s in original_data.metadata.schemas if s.get("schema_id")}
        new_schema_ids = {s.get("schema_id") for s in memory.metadata.schemas if s.get("schema_id")}
        for schema_id in old_schema_ids - new_schema_ids: memory_core.schema_index[schema_id].discard(memory_id)
        for schema_id in new_schema_ids - old_schema_ids: memory_core.schema_index[schema_id].add(memory_id)

        # --- Update Archived/Consolidated Status ---
        if memory.is_archived != original_data.is_archived:
            if memory.is_archived: memory_core.archived_memories.add(memory_id)
            else: memory_core.archived_memories.discard(memory_id)
        if memory.is_consolidated != original_data.is_consolidated:
             if memory.is_consolidated: memory_core.consolidated_memories.add(memory_id)
             else: memory_core.consolidated_memories.discard(memory_id)

        # Update reverse summary links if structure changed
        old_sources = set(original_data.metadata.source_memory_ids or [])
        new_sources = set(memory.metadata.source_memory_ids or [])
        if memory.metadata.memory_level in ['summary', 'abstraction'] and old_sources != new_sources:
             for source_id in old_sources - new_sources:
                 if memory_id in memory_core.summary_links.get(source_id,[]):
                     memory_core.summary_links[source_id].remove(memory_id)
                     if not memory_core.summary_links[source_id]: del memory_core.summary_links[source_id]
             for source_id in new_sources - old_sources:
                 memory_core.summary_links[source_id].append(memory_id)

        memory_core.query_cache = {}
        logger.debug(f"Updated memory {memory_id}")
        return True

@function_tool
async def delete_memory(
    ctx: RunContextWrapper[MemoryCoreContext],
    memory_id: str
) -> bool:
    """
    Delete a memory from the system, cleaning up all indices.
    Restored full index cleanup.
    """
    with custom_span("delete_memory", {"memory_id": memory_id}):
        memory_core = ctx.context
        if not memory_core.initialized: memory_core.initialize()

        memory = memory_core.memories.pop(memory_id, None)
        if not memory:
            logger.warning(f"Attempted to delete non-existent memory: {memory_id}")
            return False

        memory_core.memory_embeddings.pop(memory_id, None)

        # --- Remove from all indices ---
        memory_core.type_index[memory.memory_type].discard(memory_id)
        memory_core.scope_index[memory.memory_scope].discard(memory_id)
        memory_core.level_index[memory.metadata.memory_level].discard(memory_id)
        sig_level = min(10, max(1, int(memory.significance)))
        memory_core.significance_index[sig_level].discard(memory_id)

        for tag in memory.tags: memory_core.tag_index[tag].discard(memory_id)
        for entity in memory.metadata.entities: memory_core.entity_index[entity].discard(memory_id)
        if memory.metadata.emotional_context and memory.metadata.emotional_context.primary_emotion:
             memory_core.emotional_index[memory.metadata.emotional_context.primary_emotion].discard(memory_id)
        for schema_ref in memory.metadata.schemas:
             if schema_id := schema_ref.get("schema_id"): memory_core.schema_index[schema_id].discard(memory_id)

        # Remove from temporal index (more efficient than rebuilding)
        memory_core.temporal_index = [(ts, mid) for ts, mid in memory_core.temporal_index if mid != memory_id]

        memory_core.archived_memories.discard(memory_id)
        memory_core.consolidated_memories.discard(memory_id)

        # Remove from summary links
        for source_id, summary_list in list(memory_core.summary_links.items()):
             if memory_id in summary_list:
                 summary_list.remove(memory_id)
                 if not summary_list: del memory_core.summary_links[source_id]
        memory_core.summary_links.pop(memory_id, None) # Remove if it was a source for others

        memory_core.query_cache = {}
        logger.debug(f"Deleted memory {memory_id}")
        return True

@function_tool
async def archive_memory(
    ctx: RunContextWrapper[MemoryCoreContext],
    memory_id: str
) -> bool:
    """
    Archive a memory
    
    Args:
        memory_id: ID of the memory to archive
        
    Returns:
        True if successful, False if memory not found
    """
    return await update_memory(ctx, memory_id, {"is_archived": True})

@function_tool
async def unarchive_memory(
    ctx: RunContextWrapper[MemoryCoreContext],
    memory_id: str
) -> bool:
    """
    Unarchive a memory
    
    Args:
        memory_id: ID of the memory to unarchive
        
    Returns:
        True if successful, False if memory not found
    """
    return await update_memory(ctx, memory_id, {"is_archived": False})

@function_tool
async def mark_as_consolidated(
    ctx: RunContextWrapper[MemoryCoreContext],
    memory_id: str, 
    consolidated_into: str
) -> bool:
    """
    Mark a memory as consolidated into another memory
    
    Args:
        memory_id: ID of the memory to mark
        consolidated_into: ID of the memory it was consolidated into
        
    Returns:
        True if successful, False if memory not found
    """
    return await update_memory(ctx, memory_id, {
        "is_consolidated": True,
        "metadata": {
            "consolidated_into": consolidated_into,
            "consolidation_date": datetime.datetime.now().isoformat()
        }
    })

@function_tool
async def retrieve_memories_with_formatting(
    ctx: RunContextWrapper[MemoryCoreContext],
    query: str,
    limit: Optional[int],                    # <- now the last non-default
    memory_types: Optional[List[str]] = None,
    scopes: Optional[List[str]] = None,
    min_significance: Optional[int] = None,
    include_archived: Optional[bool] = None,
    entities: Optional[List[str]] = None,
    emotional_state: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Enhanced retrieval with formatted results for agent consumption
    
    Args:
        query: Search query
        memory_types: Types of memories to include
        limit: Maximum number of results
    
    Returns:
        List of formatted memories with confidence markers
    """
    if limit is None:
        limit = 20
    with custom_span("retrieve_memories_with_formatting", {"query": query, "limit": limit}):
        memories = await retrieve_memories(
            ctx,
            query=query,
            memory_types=memory_types or ["observation", "reflection", "abstraction", "experience", "semantic"],
            limit=limit
        )
        
        # Format memories with confidence markers
        formatted_memories = []
        for memory in memories:
            confidence = memory.get("confidence_marker", "remember")
            formatted = {
                "id": memory["id"],
                "text": memory["memory_text"],
                "type": memory["memory_type"],
                "significance": memory["significance"],
                "confidence": confidence,
                "relevance": memory.get("relevance", 0.5),
                "tags": memory.get("tags", []),
                "fidelity": memory.get("metadata", {}).get("fidelity", 1.0)
            }
            formatted_memories.append(formatted)
        
        return formatted_memories

@function_tool
async def create_reflection_from_memories(
    ctx: RunContextWrapper[MemoryCoreContext],
    topic: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a reflection on a specific topic using memories
    
    Args:
        topic: Optional topic to reflect on
    
    Returns:
        Reflection result dictionary
    """
    with custom_span("create_reflection", {"topic": topic or "general"}):
        # Get relevant memories for reflection
        query = topic if topic else "important memories"
        memories = await retrieve_memories(
            ctx,
            query=query,
            memory_types=["observation", "experience"],
            limit=5
        )
        
        if not memories:
            return {
                "reflection": "I don't have enough memories to form a meaningful reflection yet.",
                "confidence": 0.3,
                "topic": topic,
                "reflection_id": None
            }
        
        # Generate reflection text using agent
        reflection_agent = Agent(
            name="Reflection Generator",
            instructions="""Generate a thoughtful reflection based on provided memories.
            The reflection should show insight and connect patterns across the memories."""
        )
        
        # Prepare memory texts for the agent
        memory_texts = [f"Memory: {m['memory_text']}" for m in memories]
        memory_context = "\n\n".join(memory_texts)
        
        topic_text = f" about {topic}" if topic else ""
        prompt = f"Based on these memories{topic_text}, create a thoughtful reflection:\n\n{memory_context}"
        
        result = await Runner.run(reflection_agent, prompt)
        reflection_text = result.final_output
        
        # Calculate a confidence score based on number of memories and their relevance
        avg_relevance = sum(m.get("relevance", 0.5) for m in memories) / len(memories)
        confidence = 0.4 + (len(memories) / 10) + (avg_relevance * 0.3)
        
        # Store reflection as a memory
        reflection_id = await add_memory(
            ctx,
            memory_text=reflection_text,
            memory_type="reflection",
            memory_scope="game",
            significance=6,
            tags=["reflection"] + ([topic] if topic else []),
            metadata={
                "confidence": confidence,
                "source_memory_ids": [m["id"] for m in memories],
                "timestamp": datetime.datetime.now().isoformat(),
                "fidelity": 0.8  # Reflections have slightly reduced fidelity
            }
        )
        
        return {
            "reflection": reflection_text,
            "confidence": min(1.0, confidence),
            "topic": topic,
            "reflection_id": reflection_id
        }

@function_tool
async def create_abstraction_from_memories(
    ctx: RunContextWrapper[MemoryCoreContext],
    memory_ids: List[str], 
    pattern_type: str = "behavior"
) -> Dict[str, Any]:
    """
    Create a higher-level abstraction from specific memories
    
    Args:
        memory_ids: IDs of memories to abstract from
        pattern_type: Type of pattern to identify
        
    Returns:
        Abstraction result dictionary
    """
    with custom_span("create_abstraction", {"pattern_type": pattern_type, "num_memories": len(memory_ids)}):
        # Retrieve the specified memories
        memories = []
        for memory_id in memory_ids:
            memory = await get_memory(ctx, memory_id)
            if memory:
                memories.append(memory)
        
        if not memories:
            return {
                "abstraction": "I couldn't find the specified memories to form an abstraction.",
                "confidence": 0.1,
                "pattern_type": pattern_type
            }
        
        # Generate abstraction using agent
        abstraction_agent = Agent(
            name="Abstraction Generator",
            instructions=f"""You create high-level abstractions that identify patterns in memories.
            Focus on identifying {pattern_type} patterns. Generate a clear abstraction that synthesizes
            the common elements across these memories."""
        )
        
        # Prepare memory texts for the agent
        memory_texts = [f"Memory {i+1}: {m['memory_text']}" for i, m in enumerate(memories)]
        memory_context = "\n\n".join(memory_texts)
        
        prompt = f"Identify a {pattern_type} pattern in these memories and create an abstraction:\n\n{memory_context}"
        
        result = await Runner.run(abstraction_agent, prompt)
        abstraction_text = result.final_output
        
        # Calculate confidence based on number of memories and their similarity
        confidence = 0.5 + (len(memories) / 10)
        
        # Simple pattern data
        pattern_data = {
            "pattern_type": pattern_type,
            "confidence": min(0.9, confidence),
            "sample_size": len(memories)
        }
        
        # Store abstraction as a memory
        abstraction_id = await add_memory(
            ctx,
            memory_text=abstraction_text,
            memory_type="abstraction",
            memory_scope="game",
            significance=7,
            tags=["abstraction", pattern_type],
            metadata={
                "pattern_data": pattern_data,
                "source_memory_ids": memory_ids,
                "timestamp": datetime.datetime.now().isoformat(),
                "fidelity": 0.75  # Abstractions have moderately reduced fidelity
            }
        )
        
        return {
            "abstraction": abstraction_text,
            "pattern_type": pattern_type,
            "confidence": pattern_data.get("confidence", 0.5),
            "abstraction_id": abstraction_id
        }

@function_tool
async def create_semantic_memory(
    ctx: RunContextWrapper[MemoryCoreContext],
    source_memory_ids: List[str], 
    abstraction_type: str = "pattern"
) -> Optional[str]:
    """
    Create a semantic memory (abstraction) from source memories
    
    Args:
        source_memory_ids: IDs of source memories
        abstraction_type: Type of abstraction to create
        
    Returns:
        Memory ID of created semantic memory or None
    """
    memory_core = ctx.context
    
    # Get the source memories
    source_memories = []
    for memory_id in source_memory_ids:
        memory = await get_memory(ctx, memory_id)
        if memory:
            source_memories.append(memory)
    
    if len(source_memories) < memory_core.abstraction_threshold:
        return None
    
    # Generate the abstraction text
    abstraction_text = await _generate_abstraction(ctx, source_memories, abstraction_type)
    
    if not abstraction_text:
        return None
    
    # Create the semantic memory
    memory_id = await add_memory(
        ctx,
        memory_text=abstraction_text,
        memory_type="semantic",
        memory_scope="personal",
        significance=max(m.get("significance", 5) for m in source_memories),
        tags=["semantic", "abstraction", abstraction_type],
        metadata={
            "timestamp": datetime.datetime.now().isoformat(),
            "source_memory_ids": source_memory_ids,
            "fidelity": 0.8  # Semantic memories have slightly reduced fidelity
        }
    )
    
    # Update source memories
    for memory in source_memories:
        metadata = memory.get("metadata", {})
        if "semantic_abstractions" not in metadata:
            metadata["semantic_abstractions"] = []
        
        metadata["semantic_abstractions"].append(memory_id)
        
        await update_memory(
            ctx,
            memory["id"], {"metadata": metadata}
        )
    
    return memory_id

@function_tool
async def apply_memory_decay(
    ctx: RunContextWrapper[MemoryCoreContext]
) -> Dict[str, Any]:
    """
    Apply decay to memories based on age, significance, and recall frequency
    
    Returns:
        Dictionary with decay statistics
    """
    with custom_span("apply_memory_decay"):
        memory_core = ctx.context
        if not memory_core.initialized:
            memory_core.initialize()
        
        decayed_count = 0
        archived_count = 0
        
        for memory_id, memory in memory_core.memories.items():
            # Skip non-episodic memories
            if memory["memory_type"] != "observation":
                continue
            
            # Skip already archived memories
            if memory_id in memory_core.archived_memories:
                continue

            # Check for crystallized memories
            metadata = memory.get("metadata", {})
            if metadata.get("is_crystallized", False):
                decay_resistance = metadata.get("decay_resistance", 5.0)  # Default high resistance
                # Still apply minimal decay but at greatly reduced rate
                decay_amount = memory_core.decay_rate / decay_resistance
                # Also maintain higher minimum fidelity
                min_fidelity = 0.85  # Crystallized memories stay vivid
            else:
                decay_amount = memory_core.decay_rate
                min_fidelity = 0.3  # Regular memories can fade more
            
            significance = memory["significance"]
            times_recalled = memory.get("times_recalled", 0)
            
            # Calculate age
            timestamp_str = memory.get("metadata", {}).get("timestamp", memory_core.init_time.isoformat())
            timestamp = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            days_old = (datetime.datetime.now() - timestamp).days
            
            # Calculate decay factors
            age_factor = min(1.0, days_old / 30.0)  # Older memories decay more
            recall_factor = max(0.0, 1.0 - (times_recalled / 10.0))  # Frequently recalled memories decay less
            
            # Calculate decay amount
            decay_amount = decay_amount * age_factor * recall_factor
            
            # Apply extra decay to memories never recalled
            if times_recalled == 0 and days_old > 7:
                decay_amount *= 1.5
            
            # Skip if decay is negligible
            if decay_amount < 0.05:
                continue
            
            # Store original significance if this is first decay
            if "original_significance" not in memory.get("metadata", {}):
                memory["metadata"]["original_significance"] = significance
            
            # Apply decay with minimum of 1
            new_significance = max(1, significance - decay_amount)
            
            # Apply decay to fidelity if present
            metadata = memory.get("metadata", {})
            if "fidelity" in metadata:
                fidelity = metadata["fidelity"]
                fidelity_decay = decay_amount * 0.2  # Fidelity decays more slowly
                metadata["fidelity"] = max(min_fidelity, fidelity - fidelity_decay)
            
            # Only update if change is significant
            if abs(new_significance - significance) >= 0.5:
                await update_memory(ctx, memory_id, {
                    "significance": new_significance,
                    "metadata": {
                        **memory.get("metadata", {}),
                        "last_decay": datetime.datetime.now().isoformat(),
                        "decay_amount": decay_amount
                    }
                })
                decayed_count += 1
            
            # Archive if significance is very low and memory is old
            if new_significance < 2 and days_old > 30:
                await archive_memory(ctx, memory_id)
                archived_count += 1
        
        return {
            "memories_decayed": decayed_count,
            "memories_archived": archived_count
        }

@function_tool
async def find_memory_clusters(
    ctx: RunContextWrapper[MemoryCoreContext]
) -> List[List[str]]:
    """
    Find clusters of similar memories based on embedding similarity
    
    Returns:
        List of memory ID clusters
    """
    memory_core = ctx.context
    if not memory_core.initialized:
        memory_core.initialize()
    
    with custom_span("find_memory_clusters"):
        # Get memories that are candidates for clustering
        candidate_ids = set()
        for memory_type in ["observation", "experience"]:
            if memory_type in memory_core.type_index:
                candidate_ids.update(memory_core.type_index[memory_type])
        
        # Remove already consolidated or archived memories
        candidate_ids = candidate_ids - memory_core.consolidated_memories - memory_core.archived_memories
        
        # Group memories into clusters
        clusters = []
        unclustered = list(candidate_ids)
        
        while unclustered:
            # Take the first memory as a seed
            seed_id = unclustered.pop(0)
            if seed_id not in memory_core.memory_embeddings:
                continue
                
            seed_embedding = memory_core.memory_embeddings[seed_id]
            current_cluster = [seed_id]
            
            # Find similar memories
            i = 0
            while i < len(unclustered):
                memory_id = unclustered[i]
                
                if memory_id not in memory_core.memory_embeddings:
                    i += 1
                    continue
                
                # Calculate similarity
                similarity = _cosine_similarity(seed_embedding, memory_core.memory_embeddings[memory_id])
                
                # Add to cluster if similar enough
                if similarity > memory_core.consolidation_threshold:
                    current_cluster.append(memory_id)
                    unclustered.pop(i)
                else:
                    i += 1
            
            # Only keep clusters with multiple memories
            if len(current_cluster) > 1:
                clusters.append(current_cluster)
        
        return clusters

@function_tool
async def consolidate_memory_clusters(
    ctx: RunContextWrapper[MemoryCoreContext]
) -> Dict[str, Any]:
    """
    Consolidate clusters of similar memories
    
    Returns:
        Dictionary with consolidation statistics
    """
    with custom_span("consolidate_memory_clusters"):
        memory_core = ctx.context
        if not memory_core.initialized:
            memory_core.initialize()
        
        # Find clusters
        clusters = await find_memory_clusters(ctx)
        
        consolidated_count = 0
        affected_memories = 0
        
        for cluster in clusters:
            # Skip small clusters
            if len(cluster) < 3:
                continue
            
            try:
                # Get memory texts
                memory_texts = []
                for memory_id in cluster:
                    if memory_id in memory_core.memories:
                        memory_texts.append(memory_core.memories[memory_id]["memory_text"])
                
                # Generate consolidated text
                consolidated_text = await _generate_consolidated_text(ctx, memory_texts)
                
                # Calculate average significance (weighted by individual memory significance)
                total_significance = sum(memory_core.memories[mid]["significance"] for mid in cluster if mid in memory_core.memories)
                avg_significance = total_significance / len(cluster)
                
                # Combine tags
                all_tags = set()
                for memory_id in cluster:
                    if memory_id in memory_core.memories:
                        all_tags.update(memory_core.memories[memory_id].get("tags", []))
                
                # Create consolidated memory
                consolidated_id = await add_memory(
                    ctx,
                    memory_text=consolidated_text,
                    memory_type="consolidated",
                    memory_scope="game",
                    significance=min(10, int(avg_significance) + 1),  # Slightly higher significance than average
                    tags=list(all_tags) + ["consolidated"],
                    metadata={
                        "source_memory_ids": cluster,
                        "consolidation_date": datetime.datetime.now().isoformat(),
                        "fidelity": 0.9  # High but not perfect fidelity for consolidated memories
                    }
                )
                
                # Mark original memories as consolidated
                for memory_id in cluster:
                    await mark_as_consolidated(ctx, memory_id, consolidated_id)
                
                consolidated_count += 1
                affected_memories += len(cluster)
                
            except Exception as e:
                logger.error(f"Error consolidating cluster: {e}")
        
        return {
            "clusters_consolidated": consolidated_count,
            "memories_affected": affected_memories
        }

@function_tool
async def run_maintenance(
    ctx: RunContextWrapper[MemoryCoreContext]
) -> Dict[str, Any]:
    """
    Run full memory maintenance
    
    Returns:
        Dictionary with maintenance statistics
    """
    with custom_span("run_maintenance"):
        memory_core = ctx.context
        if not memory_core.initialized:
            memory_core.initialize()
        
        # Apply memory decay
        decay_result = await apply_memory_decay(ctx)
        
        # Consolidate similar memories
        consolidation_result = await consolidate_memory_clusters(ctx)
        
        # Archive old memories with low recall
        archive_count = 0
        for memory_id, memory in memory_core.memories.items():
            # Skip already archived memories
            if memory_id in memory_core.archived_memories:
                continue
            
            # Skip non-observation memories
            if memory["memory_type"] not in ["observation", "experience"]:
                continue
            
            times_recalled = memory.get("times_recalled", 0)
            
            # Get age
            timestamp_str = memory.get("metadata", {}).get("timestamp", memory_core.init_time.isoformat())
            timestamp = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            days_old = (datetime.datetime.now() - timestamp).days
            
            # Archive very old memories that haven't been recalled
            if days_old > memory_core.max_memory_age_days and times_recalled == 0:
                await archive_memory(ctx, memory_id)
                archive_count += 1
        
        # Look for memory patterns and schema creation
        await _check_for_schema_creation(ctx)
        
        return {
            "memories_decayed": decay_result["memories_decayed"],
            "clusters_consolidated": consolidation_result["clusters_consolidated"],
            "memories_archived": decay_result["memories_archived"] + archive_count
        }

@function_tool
async def get_memory_stats(
    ctx: RunContextWrapper[MemoryCoreContext]
) -> Dict[str, Any]:
    """
    Get statistics about the memory system
    
    Returns:
        Dictionary with memory system statistics
    """
    with custom_span("get_memory_stats"):
        memory_core = ctx.context
        if not memory_core.initialized:
            memory_core.initialize()
        
        # Count memories by type
        type_counts = {}
        for memory_type, memory_ids in memory_core.type_index.items():
            type_counts[memory_type] = len(memory_ids)
        
        # Count by scope
        scope_counts = {}
        for scope, memory_ids in memory_core.scope_index.items():
            scope_counts[scope] = len(memory_ids)
        
        # Get top tags
        tag_counts = {}
        for tag, memory_ids in memory_core.tag_index.items():
            tag_counts[tag] = len(memory_ids)
        
        # Sort tags by count
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Count schemas by category
        schema_counts = {}
        for schema_id, schema in memory_core.schemas.items():
            category = schema.category
            if category not in schema_counts:
                schema_counts[category] = 0
            schema_counts[category] += 1
        
        # Get top schemas by usage
        schema_usage = []
        for schema_id, schema in memory_core.schemas.items():
            usage = len(memory_core.schema_index.get(schema_id, set()))
            schema_usage.append((schema_id, schema.name, usage))
        
        top_schemas = sorted(schema_usage, key=lambda x: x[2], reverse=True)[:5]
        
        # Get total memory count
        total_memories = len(memory_core.memories)
        
        # Get archive and consolidation counts
        archived_count = len(memory_core.archived_memories)
        consolidated_count = len(memory_core.consolidated_memories)
        
        # Get average significance
        if total_memories > 0:
            avg_significance = sum(m["significance"] for m in memory_core.memories.values()) / total_memories
        else:
            avg_significance = 0
        
        # Calculate memory age stats
        now = datetime.datetime.now()
        ages = []
        for memory in memory_core.memories.values():
            timestamp_str = memory.get("metadata", {}).get("timestamp", memory_core.init_time.isoformat())
            timestamp = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            age_days = (now - timestamp).days
            ages.append(age_days)
        
        oldest_memory = max(ages) if ages else 0
        newest_memory = min(ages) if ages else 0
        avg_age = sum(ages) / len(ages) if ages else 0
        
        # Calculate average fidelity
        fidelities = [m.get("metadata", {}).get("fidelity", 1.0) for m in memory_core.memories.values()]
        avg_fidelity = sum(fidelities) / len(fidelities) if fidelities else 1.0
        
        return {
            "total_memories": total_memories,
            "type_counts": type_counts,
            "scope_counts": scope_counts,
            "top_tags": dict(top_tags),
            "archived_count": archived_count,
            "consolidated_count": consolidated_count,
            "avg_significance": avg_significance,
            "oldest_memory_days": oldest_memory,
            "newest_memory_days": newest_memory,
            "avg_age_days": avg_age,
            "schema_counts": schema_counts,
            "top_schemas": [{"id": s[0], "name": s[1], "usage": s[2]} for s in top_schemas],
            "total_schemas": len(memory_core.schemas),
            "avg_fidelity": avg_fidelity
        }

@function_tool
async def construct_narrative_from_memories(
    ctx: RunContextWrapper[MemoryCoreContext],
    topic: str, 
    chronological: bool = True, 
    limit: int = 5
) -> Dict[str, Any]:
    """
    Construct a narrative from memories about a topic
    
    Args:
        topic: Topic for narrative construction
        chronological: Whether to maintain chronological order
        limit: Maximum number of memories to include
    
    Returns:
        Narrative result dictionary
    """
    with custom_span("construct_narrative", {"topic": topic, "chronological": chronological}):
        # Retrieve memories for the topic
        memories = await retrieve_memories(
            ctx,
            query=topic,
            memory_types=["observation", "experience"],
            limit=limit
        )
        
        if not memories:
            return {
                "narrative": f"I don't have enough memories about {topic} to construct a narrative.",
                "confidence": 0.2,
                "experience_count": 0
            }
        
        # Sort memories chronologically if requested
        if chronological:
            memories.sort(key=lambda m: m.get("metadata", {}).get("timestamp", ""))
        
        # Use Narrative Agent to create a coherent narrative
        narrative_agent = Agent(
            name="Narrative Constructor",
            instructions=f"""Create a coherent narrative based on memories about {topic}.
            The narrative should flow naturally and connect the experiences in a meaningful way.
            {"Maintain chronological order of events." if chronological else "Focus on thematic connections rather than chronology."}"""
        )
        
        # Prepare memories for the agent
        memory_texts = []
        for i, memory in enumerate(memories):
            timestamp = _get_timeframe_text(memory.get("metadata", {}).get("timestamp", ""))
            fidelity = memory.get("metadata", {}).get("fidelity", 1.0)
            fidelity_note = "" if fidelity > 0.9 else f" (confidence: {int(fidelity * 100)}%)"
            memory_texts.append(f"Memory {i+1} ({timestamp}{fidelity_note}): {memory['memory_text']}")
        
        memory_context = "\n\n".join(memory_texts)
        
        prompt = f"Construct a narrative about {topic} based on these memories:\n\n{memory_context}"
        
        result = await Runner.run(narrative_agent, prompt)
        narrative = result.final_output
        
        # Calculate confidence based on memory count and relevance
        avg_relevance = sum(m.get("relevance", 0.5) for m in memories) / len(memories)
        # Also factor in average fidelity
        avg_fidelity = sum(m.get("metadata", {}).get("fidelity", 1.0) for m in memories) / len(memories)
        confidence = (0.4 + (len(memories) / 10) + (avg_relevance * 0.3)) * avg_fidelity
        
        return {
            "narrative": narrative,
            "confidence": min(1.0, confidence),
            "experience_count": len(memories)
        }

@function_tool
async def retrieve_relevant_experiences(
    ctx: RunContextWrapper[MemoryCoreContext],
    query: str,
    scenario_type: Optional[str] = None,
    emotional_state: Optional[Dict[str, Any]] = None,
    entities: Optional[List[str]] = None,
    limit: Optional[int] = None,
    min_relevance: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve experiences relevant to the current conversation context
    
    Args:
        query: Search query or current topic
        scenario_type: Type of scenario (e.g., "teasing", "dark")
        emotional_state: Current emotional state
        entities: Entities involved in current context
        limit: Maximum number of experiences to return
        min_relevance: Minimum relevance score (0.0-1.0)
        
    Returns:
        List of relevant experiences with metadata
    """

    scenario_type  = scenario_type  or ""
    limit          = limit          or 3
    min_relevance  = min_relevance  if min_relevance is not None else 0.6
    with custom_span("retrieve_relevant_experiences", {"query": query, "scenario_type": scenario_type}):
        # Create context dictionary
        current_context = {
            "query": query,
            "scenario_type": scenario_type,
            "emotional_state": emotional_state or {},
            "entities": entities or []
        }
        
        # Retrieve experiences using the memory system
        experiences = await retrieve_memories(
            ctx,
            query=query,
            memory_types=["experience"],
            limit=limit * 2,  # Get more to filter by relevance
            include_archived=False,
            entities=entities,
            emotional_state=emotional_state
        )
        
        # Process top experiences
        results = []
        for memory in experiences:
            if memory.get("relevance", 0) >= min_relevance:
                # Get emotional context for this experience
                emotional_context = memory.get("metadata", {}).get("emotional_context", {})
                
                # Add confidence marker
                confidence_marker = _get_confidence_marker(ctx, memory.get("relevance", 0.5))
                
                # Get fidelity
                fidelity = memory.get("metadata", {}).get("fidelity", 1.0)
                
                # Format the experience
                experience = {
                    "id": memory["id"],
                    "content": memory["memory_text"],
                    "relevance_score": memory.get("relevance", 0.5),
                    "emotional_context": emotional_context,
                    "scenario_type": memory.get("tags", ["general"])[0] if memory.get("tags") else "general",
                    "confidence_marker": confidence_marker,
                    "experiential_richness": min(1.0, memory.get("significance", 5) / 10.0),
                    "fidelity": fidelity,
                    "schemas": memory.get("metadata", {}).get("schemas", [])
                }
                
                results.append(experience)
                
                if len(results) >= limit:
                    break
        
        return results

@function_tool
async def generate_conversational_recall(
    ctx: RunContextWrapper[MemoryCoreContext],
    experience: Dict[str, Any],
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate a natural, conversational recall of an experience.

    Args:
        experience: The experience to recall
        context: Current conversation context

    Returns:
        Conversational recall with reflection
    """
    memory_core = ctx.context
    with custom_span("generate_conversational_recall"):
        # Extract experience data
        content = experience.get("content", experience.get("memory_text", ""))
        emotional_context = experience.get("emotional_context", {})
        scenario_type = experience.get("scenario_type", "general")
        timestamp = experience.get(
            "timestamp",
            experience.get("metadata", {}).get("timestamp")
        )
        fidelity = experience.get(
            "fidelity",
            experience.get("metadata", {}).get("fidelity", 1.0)
        )

        # Get timeframe text
        timeframe = _get_timeframe_text(timestamp)

        # Determine tone for recall
        emotional_tone = _get_emotional_tone(emotional_context)
        scenario_tone = _get_scenario_tone(scenario_type)

        # Select tone (prioritize scenario tone if available)
        tone = scenario_tone or emotional_tone

        # Select a random template for this tone
        templates = memory_core.recall_templates.get(tone, memory_core.recall_templates["standard"])
        template = random.choice(templates)

        # Build the recall agent
        recall_agent = Agent(
            name="Experience Recall Agent",
            instructions=(
                f"Create a natural, conversational recall of this experience.\n"
                f"The recall should match a {tone} tone and feel authentic.\n"
                "Extract a brief summary, a relevant detail, and a thoughtful reflection from the experience."
            )
        )

        # Prepare the prompt
        prompt = (
            f'Given this experience: "{content}"\n\n'
            "Create three components for a conversational recall:\n"
            "1. A brief summary (15-20 words)\n"
            "2. A relevant detail (10-15 words)\n"
            "3. A thoughtful reflection (15-20 words)\n\n"
            "Format your response as:\n"
            "Brief Summary: [brief summary]\n"
            "Detail: [detail]\n"
            "Reflection: [reflection]"
        )

        # Run the agent
        result = await Runner.run(recall_agent, prompt)
        parts = result.final_output.strip().splitlines()

        # Default fallbacks
        brief_summary = content
        detail = "It was quite memorable."
        reflection = "I found it quite interesting to observe."

        # Parse the agent's output
        for part in parts:
            if part.startswith("Brief Summary:"):
                brief_summary = part.split("Brief Summary:", 1)[1].strip()
            elif part.startswith("Detail:"):
                detail = part.split("Detail:", 1)[1].strip()
            elif part.startswith("Reflection:"):
                reflection = part.split("Reflection:", 1)[1].strip()

        # Add qualifier if fidelity is low
        if fidelity < 0.7:
            reflection += " Though, I'm not entirely certain about all the details."

        # Fill in the selected template
        recall_text = template.format(
            timeframe=timeframe,
            brief_summary=brief_summary,
            detail=detail,
            reflection=reflection
        )

        return {
            "recall_text": recall_text,
            "tone": tone,
            "confidence": experience.get("relevance_score", 0.5) * fidelity
        }


@function_tool
async def detect_schema_from_memories(
    ctx: RunContextWrapper[MemoryCoreContext],
    topic: str = None, 
    min_memories: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Detect a potential schema from memories
    
    Args:
        topic: Optional topic to focus on
        min_memories: Minimum number of memories needed
        
    Returns:
        Detected schema information or None
    """
    memory_core = ctx.context
    
    # Retrieve relevant memories
    memories = await retrieve_memories(
        ctx,
        query=topic if topic else "important memory",
        limit=10
    )
    
    if len(memories) < min_memories:
        return None
    
    # Find patterns in the memories
    pattern = await _detect_memory_pattern(ctx, memories)
    
    if not pattern:
        return None
    
    # Create a schema from the pattern
    schema_id = str(uuid.uuid4())
    
    schema = MemorySchema(
        id=schema_id,
        name=pattern["name"],
        description=pattern["description"],
        category=pattern["category"],
        attributes=pattern["attributes"],
        example_memory_ids=[m["id"] for m in memories[:min_memories]],
        creation_date=datetime.datetime.now().isoformat(),
        last_updated=datetime.datetime.now().isoformat()
    )
    
    # Store the schema
    memory_core.schemas[schema_id] = schema
    
    # Update schema index
    for memory_id in schema.example_memory_ids:
        if schema_id not in memory_core.schema_index:
            memory_core.schema_index[schema_id] = set()
        memory_core.schema_index[schema_id].add(memory_id)
        
        # Update memory metadata
        memory = await get_memory(ctx, memory_id)
        if memory:
            metadata = memory.get("metadata", {})
            if "schemas" not in metadata:
                metadata["schemas"] = []
            
            metadata["schemas"].append({
                "schema_id": schema_id,
                "relevance": 1.0  # High relevance for examples
            })
            
            await update_memory(
                ctx, memory_id, {"metadata": metadata}
            )
    
    return {
        "schema_id": schema_id,
        "schema_name": pattern["name"],
        "description": pattern["description"],
        "memory_count": len(schema.example_memory_ids)
    }

async def _check_for_schema_creation(ctx: RunContextWrapper[MemoryCoreContext]) -> None:
    """Periodically check for new schemas that could be created"""
    memory_core = ctx.context
    
    # Get common tags
    tag_counts = {}
    for tag, memory_ids in memory_core.tag_index.items():
        tag_counts[tag] = len(memory_ids)
    
    # Find tags with many memories but no schema
    potential_schema_tags = []
    for tag, count in tag_counts.items():
        if count >= 5:  # At least 5 memories with this tag
            # Check if tag is not already a schema name
            if not any(s.name.lower() == tag.lower() for s in memory_core.schemas.values()):
                potential_schema_tags.append(tag)
    
    # Process top potential schema tags
    for tag in potential_schema_tags[:3]:  # Process up to 3 potential schemas
        await detect_schema_from_memories(ctx, tag)

@function_tool
async def crystallize_memory(
    ctx: RunContextWrapper[MemoryCoreContext],
    memory_id: str, 
    reason: str = "automatic", 
    importance_data: Dict = None
) -> bool:
    """
    Crystallize a memory to make it highly resistant to decay
    
    Args:
        memory_id: ID of the memory to crystallize
        reason: Reason for crystallization
        importance_data: Optional assessment data about the memory's importance
        
    Returns:
        True if successful, False if memory not found
    """
    memory = await get_memory(ctx, memory_id)
    if not memory:
        return False
    
    metadata = memory.get("metadata", {})
    
    # Set crystallization parameters based on reason
    if reason == "cognitive_importance":
        # Higher decay resistance for memories Nyx herself deems important
        decay_resistance = 8.0
        min_fidelity = 0.95
        
        # Store importance assessment data
        metadata["importance_assessment"] = importance_data
    else:
        # Standard crystallization values for automatic factors
        decay_resistance = 5.0
        min_fidelity = 0.9
    
    # Update metadata
    metadata["is_crystallized"] = True
    metadata["crystallization_reason"] = reason
    metadata["crystallization_date"] = datetime.datetime.now().isoformat()
    metadata["decay_resistance"] = decay_resistance
    
    # Ensure high fidelity
    metadata["fidelity"] = max(metadata.get("fidelity", 1.0), min_fidelity)
    
    # Update the memory
    success = await update_memory(ctx, memory_id, {
        "metadata": metadata,
        "significance": max(memory.get("significance", 5), 8)  # Ensure high significance
    })
    
    return success

@function_tool
async def assess_memory_importance(
    ctx: RunContextWrapper[MemoryCoreContext],
    memory_id: str
) -> Dict[str, Any]:
    """
    Assess if a memory should be marked as important based on cognitive criteria
    
    Args:
        memory_id: ID of the memory to assess
        
    Returns:
        Dictionary with importance assessment data
    """
    memory = await get_memory(ctx, memory_id)
    if not memory:
        return {"important": False}
    
    # Extract content and metadata
    content = memory["memory_text"]
    metadata = memory.get("metadata", {})
    
    # Factors that might make a memory important to Nyx
    importance_factors = {
        "identity_relevance": 0.0,
        "information_value": 0.0,
        "emotional_significance": 0.0,
        "uniqueness": 0.0,
        "utility": 0.0
    }
    
    # Assess identity relevance
    if any(word in content.lower() for word in ["i am", "my personality", "my nature", "defines me"]):
        importance_factors["identity_relevance"] = 0.8
    
    # Assess information value (unique/rare information)
    similar_memories = await retrieve_memories(ctx, query=content, limit=3)
    if len(similar_memories) <= 1 or all(m["id"] == memory_id for m in similar_memories):
        importance_factors["uniqueness"] = 0.7
    
    # Assess emotional significance
    emotional_context = metadata.get("emotional_context", {})
    primary_intensity = emotional_context.get("primary_intensity", 0.0)
    importance_factors["emotional_significance"] = primary_intensity
    
    # Utility for future interactions
    if any(word in content.lower() for word in ["remember this", "important", "crucial", "never forget"]):
        importance_factors["utility"] = 0.9
    
    # Calculate overall importance
    overall_importance = sum(importance_factors.values()) / len(importance_factors)
    should_crystallize = overall_importance > 0.6
    
    return {
        "important": should_crystallize,
        "importance_score": overall_importance,
        "factors": importance_factors
    }

@function_tool
async def reflect_on_memories(
    ctx: RunContextWrapper[MemoryCoreContext],
    recent_only: bool = True,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Periodic reflection to identify important memories
    
    Args:
        recent_only: Whether to focus only on recent memories
        limit: Maximum number of memories to assess
        
    Returns:
        Dictionary with reflection statistics
    """
    # Get memories to reflect on
    if recent_only:
        # Get memories from the last 24 hours
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        query = "timestamp:>" + yesterday.isoformat()
        memories = await retrieve_memories(
            ctx,
            query=query,
            limit=limit
        )
    else:
        # Get memories that haven't been reflected on yet
        memories = await retrieve_memories(
            ctx,
            query="importance:unassessed",
            limit=limit
        )
    
    crystallized_count = 0
    
    # Assess each memory
    for memory in memories:
        importance = await assess_memory_importance(ctx, memory["id"])
        
        if importance["important"]:
            # Crystallize this memory
            await crystallize_memory(
                ctx,
                memory_id=memory["id"],
                reason="cognitive_importance",
                importance_data=importance
            )
            crystallized_count += 1
            
            # Log the importance assessment
            logger.info(f"Crystallized memory {memory['id']} due to cognitive importance: {importance['importance_score']}")
    
    return {
        "memories_assessed": len(memories),
        "memories_crystallized": crystallized_count
    }

# Main Memory Core class that uses the Agents SDK
class MemoryCoreAgents:
    """
    Unified memory storage and retrieval system that integrates with OpenAI Agents SDK.
    This is a drop-in replacement for the original MemoryCore class.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        """Initialize the memory core with agents and context"""
        self.context = MemoryCoreContext(user_id, conversation_id)
        self.memory_retrieval_agent = self._create_retrieval_agent()
        self.memory_creation_agent = self._create_creation_agent()
        self.memory_maintenance_agent = self._create_maintenance_agent()
        self.experience_agent = self._create_experience_agent()
        self.narrative_agent = self._create_narrative_agent()
    
    def _create_retrieval_agent(self):
        """Create the memory retrieval agent"""
        return Agent(
            name="Memory Retrieval Agent",
            instructions="""You are a specialized memory retrieval agent for the Nyx AI system.
            Your job is to search the memory system using semantic relevance and return the most appropriate
            memories based on the current context. Focus on relevance, recency, and significance.""",
            tools=[
                retrieve_memories,
                get_memory_details,
                get_memory,
                retrieve_memories_with_formatting,
                retrieve_relevant_experiences
            ]
        )
    
    def _create_creation_agent(self):
        """Create the memory creation agent"""
        return Agent(
            name="Memory Creation Agent",
            instructions="""You are a specialized memory creation agent for the Nyx AI system.
            Your job is to create new memories, reflections, and abstractions based on experiences
            and observations. Make sure to properly categorize and tag memories.""",
            tools=[
                add_memory,
                update_memory,
                delete_memory,
                create_reflection_from_memories,
                create_abstraction_from_memories,
                create_semantic_memory
            ]
        )
    
    def _create_maintenance_agent(self):
        """Create the memory maintenance agent"""
        return Agent(
            name="Memory Maintenance Agent",
            instructions="""You are a specialized memory maintenance agent for the Nyx AI system.
            Your job is to manage the memory system by applying decay, consolidating similar memories,
            and archiving old memories. Focus on keeping the memory system efficient.""",
            tools=[
                run_maintenance,
                apply_memory_decay,
                consolidate_memory_clusters,
                archive_memory,
                unarchive_memory,
                get_memory_stats,
                detect_schema_from_memories
            ]
        )
    
    def _create_experience_agent(self):
        """Create the experience agent"""
        return Agent(
            name="Experience Agent",
            instructions="""You are a specialized experience agent for the Nyx AI system.
            Your job is to retrieve and format experiences for conversational recall.
            Focus on creating natural, engaging experience recalls.""",
            tools=[
                retrieve_relevant_experiences,
                generate_conversational_recall
            ]
        )
    
    def _create_narrative_agent(self):
        """Create the narrative agent"""
        return Agent(
            name="Narrative Agent",
            instructions="""You are a specialized narrative agent for the Nyx AI system.
            Your job is to construct cohesive narratives from memories and experiences.
            Focus on creating engaging, meaningful stories that connect memories.""",
            tools=[
                construct_narrative_from_memories
            ]
        )
    
    async def load_recent_memories(self, memories_data):
        """Compatibility method for checkpoint restoration"""
        for memory_data in memories_data:
            # Extract required fields with defaults
            memory_text = memory_data.get('memory_text', 'Restored memory')
            memory_type = memory_data.get('memory_type', 'observation')
            memory_scope = memory_data.get('memory_scope', 'game')
            significance = memory_data.get('significance', 5)
            tags = memory_data.get('tags', [])
            metadata = memory_data.get('metadata', {})
            
            # Add memory
            await self.add_memory(
                memory_text=memory_text,
                memory_type=memory_type,
                memory_scope=memory_scope,
                significance=significance,
                tags=tags,
                metadata=metadata
            )        
    
    async def initialize(self):
        """Initialize memory system"""
        with trace(workflow_name="Memory System Initialization"):
            self.context.initialize()
            logger.info("Memory Core initialized with agents.")
    
    async def add_memory(self, **kwargs):
        """Add a new memory"""
        result = await Runner.run(
            self.memory_creation_agent,
            f"Add a new memory with these parameters: {json.dumps(kwargs)}",
            context=self.context
        )
        
        # Parse memory ID from response
        return result.final_output
    
    async def retrieve_memories(self, **kwargs):
        """Retrieve memories matching a query"""
        result = await Runner.run(
            self.memory_retrieval_agent,
            f"Retrieve memories with these parameters: {json.dumps(kwargs)}",
            context=self.context
        )
        
        # Parse memories from result
        if isinstance(result.final_output, list):
            return result.final_output
        elif isinstance(result.final_output, dict) and "memories" in result.final_output:
            return result.final_output["memories"]
        else:
            return []
    
    async def create_reflection(self, topic=None):
        """Create a reflection on a topic"""
        prompt = f"Create a reflection" + (f" about topic: {topic}" if topic else "")
        
        result = await Runner.run(
            self.memory_creation_agent,
            prompt,
            context=self.context
        )
        
        return result.final_output
    
    async def retrieve_experiences(self, query, scenario_type="", limit=3):
        """Retrieve experiences matching a query"""
        prompt = f"Retrieve experiences related to '{query}'"
        if scenario_type:
            prompt += f" with scenario type '{scenario_type}'"
        prompt += f", limit {limit}"
        
        result = await Runner.run(
            self.experience_agent,
            prompt,
            context=self.context
        )
        
        if isinstance(result.final_output, list):
            return result.final_output
        elif isinstance(result.final_output, dict) and "experiences" in result.final_output:
            return result.final_output["experiences"]
        else:
            return []
    
    async def run_maintenance(self):
        """Run memory system maintenance"""
        result = await Runner.run(
            self.memory_maintenance_agent,
            "Run a complete memory maintenance cycle",
            context=self.context
        )
        
        return result.final_output
    
    async def construct_narrative(self, topic, chronological=True, limit=5):
        """Construct a narrative from memories"""
        prompt = f"Construct a narrative about '{topic}'"
        prompt += f", {'chronological' if chronological else 'non-chronological'}"
        prompt += f", using up to {limit} memories"
        
        result = await Runner.run(
            self.narrative_agent,
            prompt,
            context=self.context
        )
        
        return result.final_output
    
    async def get_memory_stats(self):
        """Get memory system statistics"""
        result = await Runner.run(
            self.memory_maintenance_agent,
            "Get comprehensive memory statistics",
            context=self.context
        )
        
        return result.final_output
        
class BrainMemoryCore(MemoryCoreAgents):
    """
    Special memory core for the ultimate 'Nyx Brain' -- one global memory.
    Ignores user/conversation in all operations; always accesses the full store.
    """

    def __init__(self):
        # No user_id/conversation_id needed for this global entity
        super().__init__(user_id=None, conversation_id=None)
        self.omniscient = True

    async def retrieve_memories(self, **kwargs):
        """
        Retrieve memories across ALL users/conversations. No filtering.
        Usage: await nyx_brain.retrieve_memories(query="love", limit=3)
        """
        query_text = kwargs.get('query', '')
        limit = kwargs.get('limit', 5)
        sql = """
            SELECT * FROM unified_memories
            WHERE memory_text ILIKE $1
            ORDER BY timestamp DESC
            LIMIT $2
        """
        async with get_db_connection_context() as conn:
            rows = await conn.fetch(sql, f'%{query_text}%', limit)
            return [dict(row) for row in rows]

    async def add_memory(self, **kwargs):
        """
        Add a memory, globally.
        Usage: await nyx_brain.add_memory(memory_text="I discovered recursion.", memory_type="reflection", significance=6)
        """
        sql = """
            INSERT INTO unified_memories (entity_type, entity_id, user_id, conversation_id, memory_text, memory_type, significance, tags, timestamp)
            VALUES ($1, $2, 0, 0, $3, $4, $5, $6, NOW())
            RETURNING *
        """
        # Required/optional fields
        entity_type = kwargs.get('entity_type', 'nyx')
        entity_id = kwargs.get('entity_id', 0)
        memory_text = kwargs.get('memory_text', '...')
        memory_type = kwargs.get('memory_type', 'reflection')
        significance = kwargs.get('significance', 5)
        tags = kwargs.get('tags', [])

        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                sql, entity_type, entity_id, memory_text, memory_type, significance, tags
            )
            return dict(row) if row else None
