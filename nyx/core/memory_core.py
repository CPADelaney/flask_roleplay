# nyx/core/memory_system.py

import asyncio
import datetime
import json
import logging
import random
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import numpy as np
from pydantic import BaseModel, Field

# OpenAI Agents SDK imports
from agents import Agent, Runner, function_tool
from agents.tracing import custom_span, trace

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
    source_memory_ids: Optional[List[str]] = None
    fidelity: float = 1.0  # How accurate the memory is (decays over time)
    original_form: Optional[str] = None  # Original text before reconsolidation
    reconsolidation_history: List[Dict[str, Any]] = Field(default_factory=list)
    semantic_abstractions: List[str] = Field(default_factory=list)

    # Fields from enhanced memory system
    fidelity: float = 1.0
    original_form: Optional[str] = None
    reconsolidation_history: List[Dict[str, Any]] = Field(default_factory=list)
    schemas: List[Dict[str, Any]] = Field(default_factory=list)
    semantic_abstractions: List[str] = Field(default_factory=list)
    
    # New fields for crystallization
    is_crystallized: bool = False
    crystallization_reason: Optional[str] = None
    crystallization_date: Optional[str] = None
    decay_resistance: float = 1.0

class Memory(BaseModel):
    id: str
    memory_text: str
    memory_type: str
    memory_scope: str
    significance: int
    tags: List[str] = Field(default_factory=list)
    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata)
    embedding: List[float] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    times_recalled: int = 0
    is_archived: bool = False
    is_consolidated: bool = False

class MemoryQuery(BaseModel):
    query: str
    memory_types: List[str] = Field(default_factory=lambda: ["observation", "reflection", "abstraction", "experience"])
    scopes: List[str] = Field(default_factory=lambda: ["game", "user", "global"])
    limit: int = 5
    min_significance: int = 3
    include_archived: bool = False
    entities: Optional[List[str]] = None
    emotional_state: Optional[Dict[str, Any]] = None

class MemoryCreateParams(BaseModel):
    memory_text: str
    memory_type: str = "observation"
    memory_scope: str = "game"
    significance: int = 5
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MemoryRetrieveResult(BaseModel):
    memory_id: str
    memory_text: str
    memory_type: str
    significance: int
    relevance: float
    confidence_marker: Optional[str] = None

class MemoryMaintenanceResult(BaseModel):
    memories_decayed: int
    clusters_consolidated: int
    memories_archived: int

class NarrativeResult(BaseModel):
    narrative: str
    confidence: float
    experience_count: int

class MemoryCore:
    """
    Unified memory storage and retrieval system that consolidates all memory-related functionality.
    Handles storage, retrieval, embedding, decay, consolidation, and archival of memories.
    
    Integrated with OpenAI Agents SDK for agent-based operations.
    Enhanced with schema detection, memory reconsolidation, and semantic memory processing.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Memory storage containers
        self.memories = {}  # Main memory storage: memory_id -> memory_data
        self.memory_embeddings = {}  # memory_id -> embedding vector
        
        # Indices for efficient retrieval
        self.type_index = {}  # memory_type -> [memory_ids]
        self.tag_index = {}  # tag -> [memory_ids]
        self.scope_index = {}  # memory_scope -> [memory_ids]
        self.entity_index = {}  # entity_id -> [memory_ids]
        self.emotional_index = {}  # emotion -> [memory_ids]
        
        # Schema registry and indexing
        self.schemas = {}  # schema_id -> schema
        self.schema_index = {}  # schema_id -> [memory_ids]
        
        # Temporal index for chronological retrieval
        self.temporal_index = []  # [(timestamp, memory_id)]
        
        # Significance tracking for memory importance
        self.significance_index = {}  # significance_level -> [memory_ids]
        
        # Archive status
        self.archived_memories = set()  # Set of archived memory_ids
        self.consolidated_memories = set()  # Set of memory_ids that have been consolidated
        
        # Configuration
        self.embed_dim = 1536  # Default embedding dimension
        self.max_memory_age_days = 90  # Default max memory age before archival consideration
        self.decay_rate = 0.1  # Default decay rate per day
        self.consolidation_threshold = 0.85  # Similarity threshold for memory consolidation
        self.reconsolidation_probability = 0.3  # Probability of reconsolidation on recall
        self.reconsolidation_strength = 0.1  # Strength of alterations during reconsolidation
        self.abstraction_threshold = 3  # Memories needed for semantic abstraction
        
        # Caches for performance
        self.query_cache = {}  # query -> (timestamp, results)
        self.cache_ttl = 300  # Cache TTL in seconds
        
        # Initialized flag
        self.initialized = False
        
        # Initialization timestamp
        self.init_time = datetime.datetime.now()
        
        # Template patterns for experience recall formatting
        self.recall_templates = {
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
        
        # Confidence marker mapping for experience relevance scores
        self.confidence_markers = {
            (0.8, 1.0): "vividly recall",
            (0.6, 0.8): "clearly remember",
            (0.4, 0.6): "remember",
            (0.2, 0.4): "think I recall",
            (0.0, 0.2): "vaguely remember"
        }
        
        # Initialize the agent instances
        self._init_agents()
    
    def _init_agents(self):
        """Initialize the memory system's agents"""
        # These will be lazy-loaded when needed
        self._memory_retrieval_agent = None
        self._memory_creation_agent = None
        self._memory_maintenance_agent = None
        self._narrative_agent = None
    
    def get_memory_retrieval_agent(self) -> Agent:
        """Get or create the memory retrieval agent"""
        if not self._memory_retrieval_agent:
            self._memory_retrieval_agent = Agent(
                name="Memory Retrieval Agent",
                instructions="""You are a specialized memory retrieval agent for the Nyx AI system.
                Your job is to search the memory system using semantic relevance and return the most appropriate
                memories based on the current context. Focus on relevance and significance.""",
                tools=[
                    function_tool(self.retrieve_memories),
                    function_tool(self.get_memory),
                    function_tool(self.retrieve_memories_with_formatting),
                    function_tool(self.retrieve_relevant_experiences)
                ]
            )
        return self._memory_retrieval_agent
    
    def get_memory_creation_agent(self) -> Agent:
        """Get or create the memory creation agent"""
        if not self._memory_creation_agent:
            self._memory_creation_agent = Agent(
                name="Memory Creation Agent",
                instructions="""You are a specialized memory creation agent for the Nyx AI system.
                Your job is to create new memories, reflections, and abstractions based on experiences
                and observations. Make sure to properly categorize and tag memories.""",
                tools=[
                    function_tool(self.add_memory),
                    function_tool(self.update_memory),
                    function_tool(self.create_reflection_from_memories),
                    function_tool(self.create_abstraction_from_memories),
                    function_tool(self.create_semantic_memory)  # Added semantic memory tool
                ]
            )
        return self._memory_creation_agent
    
    def get_memory_maintenance_agent(self) -> Agent:
        """Get or create the memory maintenance agent"""
        if not self._memory_maintenance_agent:
            self._memory_maintenance_agent = Agent(
                name="Memory Maintenance Agent",
                instructions="""You are a specialized memory maintenance agent for the Nyx AI system.
                Your job is to manage the memory system by applying decay, consolidating similar memories,
                and archiving old memories. Focus on keeping the memory system efficient.""",
                tools=[
                    function_tool(self.run_maintenance),
                    function_tool(self.apply_memory_decay),
                    function_tool(self.consolidate_memory_clusters),
                    function_tool(self.archive_memory),
                    function_tool(self.unarchive_memory),
                    function_tool(self.get_memory_stats),
                    function_tool(self.detect_schema_from_memories)  # Added schema detection tool
                ]
            )
        return self._memory_maintenance_agent
    
    def get_narrative_agent(self) -> Agent:
        """Get or create the narrative agent"""
        if not self._narrative_agent:
            self._narrative_agent = Agent(
                name="Narrative Agent",
                instructions="""You are a specialized narrative agent for the Nyx AI system.
                Your job is to create narratives, storytelling, and conversational recalls based on
                memories and experiences. Focus on creating engaging and natural narratives.""",
                tools=[
                    function_tool(self.construct_narrative_from_memories),
                    function_tool(self.generate_conversational_recall)
                ]
            )
        return self._narrative_agent
    
    async def initialize(self):
        """Initialize memory system and load existing memories"""
        if self.initialized:
            return
        
        with trace(workflow_name="Memory System Initialization"):
            logger.info(f"Initializing memory system for user {self.user_id}, conversation {self.conversation_id}")
            
            # This is where you would load memories from a database
            # For this implementation, we'll just initialize empty structures
            
            # Initialize indices
            self.type_index = {
                "observation": set(),
                "reflection": set(),
                "abstraction": set(),
                "experience": set(),
                "consolidated": set(),
                "semantic": set()  # Added semantic memory type
            }
            
            self.scope_index = {
                "game": set(),
                "user": set(),
                "global": set()
            }
            
            # Initialize significance index
            self.significance_index = {level: set() for level in range(1, 11)}
            
            self.initialized = True
            logger.info("Memory system initialized")
    
    @function_tool
    async def add_memory(self,
                        memory_text: str,
                        memory_type: str = "observation",
                        memory_scope: str = "game",
                        significance: int = 5,
                        tags: List[str] = None,
                        metadata: Dict[str, Any] = None) -> str:
        """
        Add a new memory to the system
        
        Args:
            memory_text: Text content of the memory
            memory_type: Type of memory (observation, reflection, abstraction, experience, semantic)
            memory_scope: Scope of memory (game, user, global)
            significance: Importance level (1-10)
            tags: Optional tags for categorization
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        with custom_span("add_memory", {"memory_type": memory_type, "memory_scope": memory_scope}):
            if not self.initialized:
                await self.initialize()
            
            # Generate memory ID
            memory_id = str(uuid.uuid4())
            
            # Normalize tags
            if tags is None:
                tags = []
            
            # Initialize metadata
            if metadata is None:
                metadata = {}
            
            # Set timestamp if not provided
            if "timestamp" not in metadata:
                metadata["timestamp"] = datetime.datetime.now().isoformat()
                
            # Set fidelity default if not provided
            if "fidelity" not in metadata:
                metadata["fidelity"] = 1.0
                
            # Set original form if not already present
            if "original_form" not in metadata:
                metadata["original_form"] = memory_text
            
            # Generate embedding
            embedding = await self._generate_embedding(memory_text)
            
            # Create memory object
            memory = {
                "id": memory_id,
                "memory_text": memory_text,
                "memory_type": memory_type,
                "memory_scope": memory_scope,
                "significance": significance,
                "tags": tags,
                "metadata": metadata,
                "embedding": embedding,
                "created_at": datetime.datetime.now().isoformat(),
                "times_recalled": 0,
                "is_archived": False,
                "is_consolidated": False
            }
            
            # Store memory
            self.memories[memory_id] = memory
            self.memory_embeddings[memory_id] = embedding
            
            # Update indices
            if memory_type in self.type_index:
                self.type_index[memory_type].add(memory_id)
            else:
                self.type_index[memory_type] = {memory_id}
            
            if memory_scope in self.scope_index:
                self.scope_index[memory_scope].add(memory_id)
            else:
                self.scope_index[memory_scope] = {memory_id}
            
            # Update tag index
            for tag in tags:
                if tag in self.tag_index:
                    self.tag_index[tag].add(memory_id)
                else:
                    self.tag_index[tag] = {memory_id}
            
            # Update temporal index
            timestamp = datetime.datetime.fromisoformat(metadata["timestamp"].replace("Z", "+00:00"))
            self.temporal_index.append((timestamp, memory_id))
            self.temporal_index.sort(key=lambda x: x[0])  # Keep sorted by timestamp
            
            # Update significance index
            sig_level = min(10, max(1, int(significance)))
            self.significance_index[sig_level].add(memory_id)
            
            # Update entity index if entities are specified
            entities = metadata.get("entities", [])
            for entity_id in entities:
                if entity_id in self.entity_index:
                    self.entity_index[entity_id].add(memory_id)
                else:
                    self.entity_index[entity_id] = {memory_id}
            
            # Update emotional index if emotional context is specified
            if "emotional_context" in metadata:
                emotional_context = metadata["emotional_context"]
                primary_emotion = emotional_context.get("primary_emotion")
                if primary_emotion:
                    if primary_emotion in self.emotional_index:
                        self.emotional_index[primary_emotion].add(memory_id)
                    else:
                        self.emotional_index[primary_emotion] = {memory_id}
            
            # Update schema index if schemas are specified
            if "schemas" in metadata:
                for schema_ref in metadata["schemas"]:
                    schema_id = schema_ref.get("schema_id")
                    if schema_id:
                        if schema_id in self.schema_index:
                            self.schema_index[schema_id].add(memory_id)
                        else:
                            self.schema_index[schema_id] = {memory_id}
            
            # Clear query cache since memory store has changed
            self.query_cache = {}
            
            # Check for potential patterns after adding memory
            if memory_type == "observation" and len(tags) > 0:
                # Only check patterns for certain types of memories
                asyncio.create_task(self._check_for_patterns(tags))
            
            logger.debug(f"Added memory {memory_id} of type {memory_type}")
            return memory_id
    
    @function_tool
    async def retrieve_memories(self,
                              query: str,
                              memory_types: List[str] = None,
                              scopes: List[str] = None,
                              limit: int = 5,
                              min_significance: int = 3,
                              include_archived: bool = False,
                              entities: List[str] = None,
                              emotional_state: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on query and filters
        
        Args:
            query: Search query text
            memory_types: Types of memories to retrieve
            scopes: Memory scopes to search in
            limit: Maximum number of memories to return
            min_significance: Minimum significance level
            include_archived: Whether to include archived memories
            entities: Entities to filter by
            emotional_state: Emotional state for relevance boosting
            
        Returns:
            List of matching memories with relevance scores
        """
        with custom_span("retrieve_memories", {
            "query": query, 
            "memory_types": str(memory_types), 
            "limit": limit
        }):
            if not self.initialized:
                await self.initialize()
            
            # Set defaults
            memory_types = memory_types or ["observation", "reflection", "abstraction", "experience", "semantic"]
            scopes = scopes or ["game", "user", "global"]
            context = {
                "include_archived": include_archived,
                "entities": entities or [],
                "emotional_state": emotional_state or {}
            }
            
            # Check cache first
            cache_key = f"{query}_{','.join(memory_types)}_{','.join(scopes)}_{limit}_{min_significance}_{include_archived}"
            if cache_key in self.query_cache:
                cache_time, cache_results = self.query_cache[cache_key]
                cache_age = (datetime.datetime.now() - cache_time).total_seconds()
                if cache_age < self.cache_ttl:
                    # Return cached results
                    results = [self.memories[mid] for mid in cache_results if mid in self.memories]
                    
                    # Apply reconsolidation if appropriate
                    results = await self._apply_reconsolidation_to_results(results)
                    
                    return results
            
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            
            # Get IDs of memories matching type and scope filters
            type_matches = set()
            for memory_type in memory_types:
                if memory_type in self.type_index:
                    type_matches.update(self.type_index[memory_type])
            
            scope_matches = set()
            for scope in scopes:
                if scope in self.scope_index:
                    scope_matches.update(self.scope_index[scope])
            
            significance_matches = set()
            for sig_level in range(min_significance, 11):
                significance_matches.update(self.significance_index[sig_level])
            
            # Find intersection of all filters
            candidate_ids = type_matches.intersection(scope_matches, significance_matches)
            
            # Remove archived memories unless explicitly requested
            if not context.get("include_archived", False):
                candidate_ids = candidate_ids - self.archived_memories
            
            # Calculate relevance for candidates
            scored_candidates = []
            for memory_id in candidate_ids:
                if memory_id in self.memory_embeddings:
                    embedding = self.memory_embeddings[memory_id]
                    relevance = self._cosine_similarity(query_embedding, embedding)
                    
                    # Apply entity relevance boost if context includes entities
                    entity_boost = 0.0
                    if "entities" in context and self.memories[memory_id].get("metadata", {}).get("entities"):
                        memory_entities = self.memories[memory_id]["metadata"]["entities"]
                        context_entities = context["entities"]
                        matching_entities = set(memory_entities).intersection(set(context_entities))
                        entity_boost = len(matching_entities) * 0.05  # 5% boost per matching entity
                    
                    # Apply emotional relevance boost if context includes emotional state
                    emotional_boost = 0.0
                    if "emotional_state" in context and self.memories[memory_id].get("metadata", {}).get("emotional_context"):
                        memory_emotion = self.memories[memory_id]["metadata"]["emotional_context"]
                        context_emotion = context["emotional_state"]
                        emotional_boost = self._calculate_emotional_relevance(memory_emotion, context_emotion)
                    
                    # Apply schema relevance boost if memory has schemas
                    schema_boost = 0.0
                    if self.memories[memory_id].get("metadata", {}).get("schemas"):
                        schema_boost = 0.05  # 5% boost for schematized memories
                    
                    # Apply temporal recency boost
                    temporal_boost = 0.0
                    memory_timestamp = datetime.datetime.fromisoformat(
                        self.memories[memory_id].get("metadata", {}).get("timestamp", self.init_time.isoformat()).replace("Z", "+00:00")
                    )
                    days_old = (datetime.datetime.now() - memory_timestamp).days
                    if days_old < 7:  # Boost recent memories
                        temporal_boost = 0.1 * (1 - (days_old / 7))
                    
                    # Apply fidelity factor - memories with higher fidelity are more reliable
                    fidelity = self.memories[memory_id].get("metadata", {}).get("fidelity", 1.0)
                    fidelity_factor = 0.8 + (0.2 * fidelity)  # Scale from 0.8 to 1.0
                    
                    # Calculate final relevance score
                    final_relevance = min(1.0, (relevance + entity_boost + emotional_boost + 
                                               schema_boost + temporal_boost) * fidelity_factor)
                    
                    scored_candidates.append((memory_id, final_relevance))
            
            # Sort by relevance
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Get top results
            top_memory_ids = [mid for mid, _ in scored_candidates[:limit]]
            
            # Prepare results with relevance scores
            results = []
            for memory_id, relevance in scored_candidates[:limit]:
                if memory_id in self.memories:
                    memory = self.memories[memory_id].copy()
                    memory["relevance"] = relevance
                    
                    # Add confidence marker for experiences
                    if memory["memory_type"] == "experience":
                        memory["confidence_marker"] = self._get_confidence_marker(relevance)
                    
                    # Update recall count
                    await self._update_memory_recall(memory_id)
                    
                    results.append(memory)
            
            # Apply reconsolidation to results if appropriate
            results = await self._apply_reconsolidation_to_results(results)
            
            # Cache results
            self.query_cache[cache_key] = (datetime.datetime.now(), top_memory_ids)
            
            return results
    
    async def _apply_reconsolidation_to_results(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply reconsolidation to retrieved memories"""
        # Check each memory for potential reconsolidation
        for i, memory in enumerate(memories):
            # Skip memory types that shouldn't reconsolidate
            if memory["memory_type"] in ["semantic", "consolidated"]:
                continue
                
            # Apply reconsolidation with probability
            if random.random() < self.reconsolidation_probability:
                memory = await self._apply_reconsolidation(memory)
                memories[i] = memory
        
        return memories
    
    async def _apply_reconsolidation(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Apply reconsolidation to a memory (simulate memory alteration on recall)"""
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
        altered_text = self._alter_memory_text(
            memory_text,
            original_form,
            self.reconsolidation_strength
        )
        
        # Update memory fidelity
        current_fidelity = metadata.get("fidelity", 1.0)
        new_fidelity = max(0.3, current_fidelity - (self.reconsolidation_strength * 0.1))
        metadata["fidelity"] = new_fidelity
        
        # Update memory
        await self.update_memory(
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
    
    def _alter_memory_text(self, text: str, original_form: str, strength: float) -> str:
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
    
    @function_tool
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID"""
        if not self.initialized:
            await self.initialize()
        
        if memory_id in self.memories:
            memory = self.memories[memory_id].copy()
            
            # Update recall count
            await self._update_memory_recall(memory_id)
            
            # Apply reconsolidation if appropriate
            if random.random() < self.reconsolidation_probability:
                memory = await self._apply_reconsolidation(memory)
            
            return memory
        
        return None
    
    @function_tool
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing memory"""
        with custom_span("update_memory", {"memory_id": memory_id}):
            if not self.initialized:
                await self.initialize()
            
            if memory_id not in self.memories:
                return False
            
            # Get existing memory
            memory = self.memories[memory_id]
            
            # Track indices to update
            index_updates = {
                "type": False,
                "scope": False,
                "tags": False,
                "significance": False,
                "entities": False,
                "emotional": False,
                "schemas": False  # Added schema index tracking
            }
            
            # Update fields
            for key, value in updates.items():
                if key in memory:
                    # Check if indices need updating
                    if key == "memory_type" and value != memory["memory_type"]:
                        index_updates["type"] = True
                    elif key == "memory_scope" and value != memory["memory_scope"]:
                        index_updates["scope"] = True
                    elif key == "tags" and set(value) != set(memory["tags"]):
                        index_updates["tags"] = True
                    elif key == "significance" and value != memory["significance"]:
                        index_updates["significance"] = True
                    elif key == "metadata":
                        # Check for entity changes
                        old_entities = memory.get("metadata", {}).get("entities", [])
                        new_entities = value.get("entities", old_entities)
                        if set(new_entities) != set(old_entities):
                            index_updates["entities"] = True
                        
                        # Check for emotional context changes
                        old_emotion = memory.get("metadata", {}).get("emotional_context", {}).get("primary_emotion")
                        new_emotion = value.get("emotional_context", {}).get("primary_emotion", old_emotion)
                        if new_emotion != old_emotion:
                            index_updates["emotional"] = True
                            
                        # Check for schema changes
                        old_schemas = memory.get("metadata", {}).get("schemas", [])
                        new_schemas = value.get("schemas", old_schemas)
                        if len(old_schemas) != len(new_schemas) or set(s.get("schema_id", "") for s in old_schemas) != set(s.get("schema_id", "") for s in new_schemas):
                            index_updates["schemas"] = True
                    
                    # Update the field
                    memory[key] = value
            
            # Update embedding if text changed
            if "memory_text" in updates:
                memory["embedding"] = await self._generate_embedding(updates["memory_text"])
                self.memory_embeddings[memory_id] = memory["embedding"]
            
            # Perform index updates
            if index_updates["type"]:
                self._update_type_index(memory_id, memory)
            
            if index_updates["scope"]:
                self._update_scope_index(memory_id, memory)
            
            if index_updates["tags"]:
                self._update_tag_index(memory_id, memory)
            
            if index_updates["significance"]:
                self._update_significance_index(memory_id, memory)
            
            if index_updates["entities"]:
                self._update_entity_index(memory_id, memory)
            
            if index_updates["emotional"]:
                self._update_emotional_index(memory_id, memory)
                
            if index_updates["schemas"]:
                self._update_schema_index(memory_id, memory)
            
            # Update archived status if specified
            if "is_archived" in updates:
                if updates["is_archived"] and memory_id not in self.archived_memories:
                    self.archived_memories.add(memory_id)
                elif not updates["is_archived"] and memory_id in self.archived_memories:
                    self.archived_memories.remove(memory_id)
            
            # Update consolidated status if specified
            if "is_consolidated" in updates:
                if updates["is_consolidated"] and memory_id not in self.consolidated_memories:
                    self.consolidated_memories.add(memory_id)
                elif not updates["is_consolidated"] and memory_id in self.consolidated_memories:
                    self.consolidated_memories.remove(memory_id)
            
            # Clear query cache as memory store has changed
            self.query_cache = {}
            
            logger.debug(f"Updated memory {memory_id}")
            return True
    
    def _update_type_index(self, memory_id: str, memory: Dict[str, Any]):
        """Update the type index for a memory"""
        # Get old type from existing indices
        old_type = None
        for t, ids in self.type_index.items():
            if memory_id in ids:
                old_type = t
                break
        
        # Remove from old type index
        if old_type and old_type in self.type_index and memory_id in self.type_index[old_type]:
            self.type_index[old_type].remove(memory_id)
        
        # Add to new type index
        new_type = memory["memory_type"]
        if new_type in self.type_index:
            self.type_index[new_type].add(memory_id)
        else:
            self.type_index[new_type] = {memory_id}
    
    def _update_scope_index(self, memory_id: str, memory: Dict[str, Any]):
        """Update the scope index for a memory"""
        # Get old scope from existing indices
        old_scope = None
        for s, ids in self.scope_index.items():
            if memory_id in ids:
                old_scope = s
                break
        
        # Remove from old scope index
        if old_scope and old_scope in self.scope_index and memory_id in self.scope_index[old_scope]:
            self.scope_index[old_scope].remove(memory_id)
        
        # Add to new scope index
        new_scope = memory["memory_scope"]
        if new_scope in self.scope_index:
            self.scope_index[new_scope].add(memory_id)
        else:
            self.scope_index[new_scope] = {memory_id}
    
    def _update_tag_index(self, memory_id: str, memory: Dict[str, Any]):
        """Update the tag index for a memory"""
        # Get old tags from memory
        old_tags = []
        for tag, ids in self.tag_index.items():
            if memory_id in ids:
                old_tags.append(tag)
        
        # Remove from old tag indices
        for tag in old_tags:
            if tag in self.tag_index and memory_id in self.tag_index[tag]:
                self.tag_index[tag].remove(memory_id)
        
        # Add to new tag indices
        new_tags = memory["tags"]
        for tag in new_tags:
            if tag in self.tag_index:
                self.tag_index[tag].add(memory_id)
            else:
                self.tag_index[tag] = {memory_id}
    
    def _update_significance_index(self, memory_id: str, memory: Dict[str, Any]):
        """Update the significance index for a memory"""
        # Get old significance from existing indices
        old_sig = None
        for sig, ids in self.significance_index.items():
            if memory_id in ids:
                old_sig = sig
                break
        
        # Remove from old significance index
        if old_sig and old_sig in self.significance_index and memory_id in self.significance_index[old_sig]:
            self.significance_index[old_sig].remove(memory_id)
        
        # Add to new significance index
        new_sig = min(10, max(1, int(memory["significance"])))
        if new_sig in self.significance_index:
            self.significance_index[new_sig].add(memory_id)
        else:
            self.significance_index[new_sig] = {memory_id}
    
    def _update_entity_index(self, memory_id: str, memory: Dict[str, Any]):
        """Update the entity index for a memory"""
        # Get old entities from memory
        old_entities = []
        for entity, ids in self.entity_index.items():
            if memory_id in ids:
                old_entities.append(entity)
        
        # Remove from old entity indices
        for entity_id in old_entities:
            if entity_id in self.entity_index and memory_id in self.entity_index[entity_id]:
                self.entity_index[entity_id].remove(memory_id)
        
        # Add to new entity indices
        new_entities = memory.get("metadata", {}).get("entities", [])
        for entity_id in new_entities:
            if entity_id in self.entity_index:
                self.entity_index[entity_id].add(memory_id)
            else:
                self.entity_index[entity_id] = {memory_id}
    
    def _update_emotional_index(self, memory_id: str, memory: Dict[str, Any]):
        """Update the emotional index for a memory"""
        # Get old emotion from existing indices
        old_emotion = None
        for emotion, ids in self.emotional_index.items():
            if memory_id in ids:
                old_emotion = emotion
                break
        
        # Remove from old emotional index
        if old_emotion and old_emotion in self.emotional_index and memory_id in self.emotional_index[old_emotion]:
            self.emotional_index[old_emotion].remove(memory_id)
        
        # Add to new emotional index
        new_emotion = memory.get("metadata", {}).get("emotional_context", {}).get("primary_emotion")
        if new_emotion:
            if new_emotion in self.emotional_index:
                self.emotional_index[new_emotion].add(memory_id)
            else:
                self.emotional_index[new_emotion] = {memory_id}
    
    def _update_schema_index(self, memory_id: str, memory: Dict[str, Any]):
        """Update the schema index for a memory"""
        # Get old schemas from existing indices
        old_schemas = []
        for schema_id, ids in self.schema_index.items():
            if memory_id in ids:
                old_schemas.append(schema_id)
        
        # Remove from old schema indices
        for schema_id in old_schemas:
            if schema_id in self.schema_index and memory_id in self.schema_index[schema_id]:
                self.schema_index[schema_id].remove(memory_id)
        
        # Add to new schema indices
        for schema_ref in memory.get("metadata", {}).get("schemas", []):
            schema_id = schema_ref.get("schema_id")
            if schema_id:
                if schema_id in self.schema_index:
                    self.schema_index[schema_id].add(memory_id)
                else:
                    self.schema_index[schema_id] = {memory_id}
    
    @function_tool
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from the system"""
        with custom_span("delete_memory", {"memory_id": memory_id}):
            if not self.initialized:
                await self.initialize()
            
            if memory_id not in self.memories:
                return False
            
            # Get memory for index cleanup
            memory = self.memories[memory_id]
            
            # Remove from type index
            memory_type = memory["memory_type"]
            if memory_type in self.type_index and memory_id in self.type_index[memory_type]:
                self.type_index[memory_type].remove(memory_id)
            
            # Remove from scope index
            memory_scope = memory["memory_scope"]
            if memory_scope in self.scope_index and memory_id in self.scope_index[memory_scope]:
                self.scope_index[memory_scope].remove(memory_id)
            
            # Remove from tag index
            for tag in memory["tags"]:
                if tag in self.tag_index and memory_id in self.tag_index[tag]:
                    self.tag_index[tag].remove(memory_id)
            
            # Remove from significance index
            sig_level = min(10, max(1, int(memory["significance"])))
            if sig_level in self.significance_index and memory_id in self.significance_index[sig_level]:
                self.significance_index[sig_level].remove(memory_id)
            
            # Remove from entity index
            entities = memory.get("metadata", {}).get("entities", [])
            for entity_id in entities:
                if entity_id in self.entity_index and memory_id in self.entity_index[entity_id]:
                    self.entity_index[entity_id].remove(memory_id)
            
            # Remove from emotional index
            emotion = memory.get("metadata", {}).get("emotional_context", {}).get("primary_emotion")
            if emotion and emotion in self.emotional_index and memory_id in self.emotional_index[emotion]:
                self.emotional_index[emotion].remove(memory_id)
                
            # Remove from schema index
            for schema_ref in memory.get("metadata", {}).get("schemas", []):
                schema_id = schema_ref.get("schema_id")
                if schema_id and schema_id in self.schema_index and memory_id in self.schema_index[schema_id]:
                    self.schema_index[schema_id].remove(memory_id)
            
            # Remove from temporal index
            self.temporal_index = [(ts, mid) for ts, mid in self.temporal_index if mid != memory_id]
            
            # Remove from archived or consolidated sets if present
            if memory_id in self.archived_memories:
                self.archived_memories.remove(memory_id)
            if memory_id in self.consolidated_memories:
                self.consolidated_memories.remove(memory_id)
            
            # Delete memory and embedding
            del self.memories[memory_id]
            if memory_id in self.memory_embeddings:
                del self.memory_embeddings[memory_id]
            
            # Clear query cache as memory store has changed
            self.query_cache = {}
            
            logger.debug(f"Deleted memory {memory_id}")
            return True
    
    @function_tool
    async def archive_memory(self, memory_id: str) -> bool:
        """Archive a memory"""
        return await self.update_memory(memory_id, {"is_archived": True})
    
    @function_tool
    async def unarchive_memory(self, memory_id: str) -> bool:
        """Unarchive a memory"""
        return await self.update_memory(memory_id, {"is_archived": False})
    
    @function_tool
    async def mark_as_consolidated(self, memory_id: str, consolidated_into: str) -> bool:
        """Mark a memory as consolidated into another memory"""
        return await self.update_memory(memory_id, {
            "is_consolidated": True,
            "metadata": {
                "consolidated_into": consolidated_into,
                "consolidation_date": datetime.datetime.now().isoformat()
            }
        })
    
    async def _update_memory_recall(self, memory_id: str):
        """Update recall count and timestamp for a memory"""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory["times_recalled"] = memory.get("times_recalled", 0) + 1
            memory["metadata"] = memory.get("metadata", {})
            memory["metadata"]["last_recalled"] = datetime.datetime.now().isoformat()
    
    @function_tool
    async def apply_memory_decay(self) -> Dict[str, Any]:
        """Apply decay to memories based on age, significance, and recall frequency"""
        with custom_span("apply_memory_decay"):
            if not self.initialized:
                await self.initialize()
            
            decayed_count = 0
            archived_count = 0
            
            for memory_id, memory in self.memories.items():
                # Skip non-episodic memories
                if memory["memory_type"] != "observation":
                    continue
                
                # Skip already archived memories
                if memory_id in self.archived_memories:
                    continue

                # NEW: Skip or reduce decay for crystallized memories
                metadata = memory.get("metadata", {})
                if metadata.get("is_crystallized", False):
                    decay_resistance = metadata.get("decay_resistance", 5.0)  # Default high resistance
                    # Still apply minimal decay but at greatly reduced rate
                    decay_amount = decay_amount / decay_resistance
                    # Also maintain higher minimum fidelity
                    min_fidelity = 0.85  # Crystallized memories stay vivid
                else:
                    min_fidelity = 0.3  # Regular memories can fade more
                
                significance = memory["significance"]
                times_recalled = memory.get("times_recalled", 0)
                
                # Calculate age
                timestamp_str = memory.get("metadata", {}).get("timestamp", self.init_time.isoformat())
                timestamp = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                days_old = (datetime.datetime.now() - timestamp).days
                
                # Calculate decay factors
                age_factor = min(1.0, days_old / 30.0)  # Older memories decay more
                recall_factor = max(0.0, 1.0 - (times_recalled / 10.0))  # Frequently recalled memories decay less
                
                # Calculate decay amount
                decay_amount = self.decay_rate * age_factor * recall_factor
                
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
                    await self.update_memory(memory_id, {
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
                    await self.archive_memory(memory_id)
                    archived_count += 1
            
            return {
                "memories_decayed": decayed_count,
                "memories_archived": archived_count
            }

    async def check_for_crystallization(self, memory_id: str) -> bool:
        """Check if a memory meets criteria for crystallization"""
        memory = await self.get_memory(memory_id)
        if not memory:
            return False
            
        metadata = memory.get("metadata", {})
        
        # Already crystallized
        if metadata.get("is_crystallized", False):
            return True
            
        # Check criteria
        
        # 1. Emotional significance
        emotional_context = metadata.get("emotional_context", {})
        emotional_intensity = emotional_context.get("primary_intensity", 0.0)
        
        # 2. Repeated access
        access_count = memory.get("times_recalled", 0)
        
        # 3. Associated with identity
        identity_related = any(tag in memory.get("tags", []) for tag in 
                              ["identity", "self", "personal", "formative", "family"])
        
        # 4. Extended periods of high significance
        high_significance_duration = metadata.get("high_significance_duration", 0)
        
        # Determine crystallization
        if ((emotional_intensity > 0.8 and access_count > 5) or
            (identity_related and access_count > 3) or
            (high_significance_duration > 30) or  # 30 days of high significance
            (access_count > 10 and memory.get("significance", 0) > 8)):
            
            # Crystallize the memory
            return True
            
        return False

    async def assess_memory_importance(self, memory_id: str) -> Dict[str, Any]:
        """Assess if a memory should be marked as important based on cognitive criteria"""
        memory = await self.get_memory(memory_id)
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
        similar_memories = await self.retrieve_memories(query=content, limit=3)
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

    async def reflect_on_memories(self, recent_only=True, limit=10):
        """Periodic reflection to identify important memories"""
        # Get memories to reflect on
        if recent_only:
            # Get memories from the last 24 hours
            yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
            memories = await self.retrieve_memories_by_timeframe(
                start_time=yesterday.isoformat(),
                limit=limit
            )
        else:
            # Get memories that haven't been reflected on yet
            memories = await self.retrieve_memories(
                query="importance:unassessed",
                limit=limit
            )
        
        crystallized_count = 0
        
        # Assess each memory
        for memory in memories:
            importance = await self.assess_memory_importance(memory["id"])
            
            if importance["important"]:
                # Crystallize this memory
                await self.crystallize_memory(
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

    async def crystallize_memory(self, memory_id: str, reason: str = "automatic", importance_data: Dict = None) -> bool:
        """Crystallize a memory to make it highly resistant to decay"""
        memory = await self.get_memory(memory_id)
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
        success = await self.update_memory(memory_id, {
            "metadata": metadata,
            "significance": max(memory.get("significance", 5), 8)  # Ensure high significance
        })
        
        return success
    
    async def find_memory_clusters(self) -> List[List[str]]:
        """Find clusters of similar memories based on embedding similarity"""
        if not self.initialized:
            await self.initialize()
        
        with custom_span("find_memory_clusters"):
            # Get memories that are candidates for clustering
            candidate_ids = set()
            for memory_type in ["observation", "experience"]:
                if memory_type in self.type_index:
                    candidate_ids.update(self.type_index[memory_type])
            
            # Remove already consolidated or archived memories
            candidate_ids = candidate_ids - self.consolidated_memories - self.archived_memories
            
            # Group memories into clusters
            clusters = []
            unclustered = list(candidate_ids)
            
            while unclustered:
                # Take the first memory as a seed
                seed_id = unclustered.pop(0)
                if seed_id not in self.memory_embeddings:
                    continue
                    
                seed_embedding = self.memory_embeddings[seed_id]
                current_cluster = [seed_id]
                
                # Find similar memories
                i = 0
                while i < len(unclustered):
                    memory_id = unclustered[i]
                    
                    if memory_id not in self.memory_embeddings:
                        i += 1
                        continue
                    
                    # Calculate similarity
                    similarity = self._cosine_similarity(seed_embedding, self.memory_embeddings[memory_id])
                    
                    # Add to cluster if similar enough
                    if similarity > self.consolidation_threshold:
                        current_cluster.append(memory_id)
                        unclustered.pop(i)
                    else:
                        i += 1
                
                # Only keep clusters with multiple memories
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
            
            return clusters
    
    @function_tool
    async def consolidate_memory_clusters(self) -> Dict[str, Any]:
        """Consolidate clusters of similar memories"""
        with custom_span("consolidate_memory_clusters"):
            if not self.initialized:
                await self.initialize()
            
            # Find clusters
            clusters = await self.find_memory_clusters()
            
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
                        if memory_id in self.memories:
                            memory_texts.append(self.memories[memory_id]["memory_text"])
                    
                    # Generate consolidated text
                    consolidated_text = await self._generate_consolidated_text(memory_texts)
                    
                    # Calculate average significance (weighted by individual memory significance)
                    total_significance = sum(self.memories[mid]["significance"] for mid in cluster if mid in self.memories)
                    avg_significance = total_significance / len(cluster)
                    
                    # Combine tags
                    all_tags = set()
                    for memory_id in cluster:
                        if memory_id in self.memories:
                            all_tags.update(self.memories[memory_id].get("tags", []))
                    
                    # Create consolidated memory
                    consolidated_id = await self.add_memory(
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
                        await self.mark_as_consolidated(memory_id, consolidated_id)
                    
                    consolidated_count += 1
                    affected_memories += len(cluster)
                    
                except Exception as e:
                    logger.error(f"Error consolidating cluster: {e}")
            
            return {
                "clusters_consolidated": consolidated_count,
                "memories_affected": affected_memories
            }
    
    @function_tool
    async def run_maintenance(self) -> MemoryMaintenanceResult:
        """Run full memory maintenance"""
        with custom_span("run_maintenance"):
            if not self.initialized:
                await self.initialize()
            
            # Apply memory decay
            decay_result = await self.apply_memory_decay()
            
            # Consolidate similar memories
            consolidation_result = await self.consolidate_memory_clusters()
            
            # Archive old memories with low recall
            archive_count = 0
            for memory_id, memory in self.memories.items():
                # Skip already archived memories
                if memory_id in self.archived_memories:
                    continue
                
                # Skip non-observation memories
                if memory["memory_type"] not in ["observation", "experience"]:
                    continue
                
                times_recalled = memory.get("times_recalled", 0)
                
                # Get age
                timestamp_str = memory.get("metadata", {}).get("timestamp", self.init_time.isoformat())
                timestamp = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                days_old = (datetime.datetime.now() - timestamp).days
                
                # Archive very old memories that haven't been recalled
                if days_old > self.max_memory_age_days and times_recalled == 0:
                    await self.archive_memory(memory_id)
                    archive_count += 1
            
            # Look for memory patterns and schema creation
            await self._check_for_schema_creation()
            
            return MemoryMaintenanceResult(
                memories_decayed=decay_result["memories_decayed"],
                clusters_consolidated=consolidation_result["clusters_consolidated"],
                memories_archived=decay_result["memories_archived"] + archive_count
            )
    
    @function_tool
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system"""
        with custom_span("get_memory_stats"):
            if not self.initialized:
                await self.initialize()
            
            # Count memories by type
            type_counts = {}
            for memory_type, memory_ids in self.type_index.items():
                type_counts[memory_type] = len(memory_ids)
            
            # Count by scope
            scope_counts = {}
            for scope, memory_ids in self.scope_index.items():
                scope_counts[scope] = len(memory_ids)
            
            # Get top tags
            tag_counts = {}
            for tag, memory_ids in self.tag_index.items():
                tag_counts[tag] = len(memory_ids)
            
            # Sort tags by count
            top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Count schemas by category
            schema_counts = {}
            for schema_id, schema in self.schemas.items():
                category = schema.category
                if category not in schema_counts:
                    schema_counts[category] = 0
                schema_counts[category] += 1
            
            # Get top schemas by usage
            schema_usage = []
            for schema_id, schema in self.schemas.items():
                usage = len(self.schema_index.get(schema_id, set()))
                schema_usage.append((schema_id, schema.name, usage))
            
            top_schemas = sorted(schema_usage, key=lambda x: x[2], reverse=True)[:5]
            
            # Get total memory count
            total_memories = len(self.memories)
            
            # Get archive and consolidation counts
            archived_count = len(self.archived_memories)
            consolidated_count = len(self.consolidated_memories)
            
            # Get average significance
            if total_memories > 0:
                avg_significance = sum(m["significance"] for m in self.memories.values()) / total_memories
            else:
                avg_significance = 0
            
            # Calculate memory age stats
            now = datetime.datetime.now()
            ages = []
            for memory in self.memories.values():
                timestamp_str = memory.get("metadata", {}).get("timestamp", self.init_time.isoformat())
                timestamp = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                age_days = (now - timestamp).days
                ages.append(age_days)
            
            oldest_memory = max(ages) if ages else 0
            newest_memory = min(ages) if ages else 0
            avg_age = sum(ages) / len(ages) if ages else 0
            
            # Calculate average fidelity
            fidelities = [m.get("metadata", {}).get("fidelity", 1.0) for m in self.memories.values()]
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
                "total_schemas": len(self.schemas),
                "avg_fidelity": avg_fidelity
            }
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 * norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def _calculate_emotional_relevance(self, memory_emotion: Dict[str, Any], current_emotion: Dict[str, Any]) -> float:
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
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text
        
        In a real implementation, this would call an embedding API
        like OpenAI's text-embedding-ada-002. For this implementation,
        we'll return a simple hash-based pseudo-embedding.
        """
        # For tracing the embedding generation
        with custom_span("generate_embedding", {"text_length": len(text)}):
            # This is just a placeholder - in a real implementation you'd call an embedding API
            # For demo purposes, we'll use a simple hash-based approach
            
            # Hash the text to a fixed-length number
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            
            # Convert to a pseudo-embedding (this is NOT a proper embedding, just a placeholder)
            random.seed(hash_val)
            return [random.uniform(-1, 1) for _ in range(self.embed_dim)]
    
    async def _generate_consolidated_text(self, memory_texts: List[str]) -> str:
        """Generate a consolidated memory text from multiple memory texts
        
        In a real implementation, this would use an LLM to generate
        a coherent consolidated memory.
        """
        with custom_span("generate_consolidated_text", {"num_texts": len(memory_texts)}):
            # This is just a placeholder - in a real implementation you'd call an LLM
            
            # For demo purposes, create a simple summary
            combined = " ".join(memory_texts)
            # Truncate if too long
            if len(combined) > 500:
                combined = combined[:497] + "..."
            
            return f"I've observed a pattern in several memories: {combined}"
    
    @function_tool
    def export_memories(self, format_type: str = "json") -> str:
        """Export memories to the specified format"""
        if format_type.lower() == "json":
            return json.dumps({
                "memories": list(self.memories.values()),
                "stats": {
                    "total": len(self.memories),
                    "archived": len(self.archived_memories),
                    "consolidated": len(self.consolidated_memories)
                },
                "schemas": list(self.schemas.values()),
                "export_date": datetime.datetime.now().isoformat()
            }, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    @function_tool
    def import_memories(self, data: str, format_type: str = "json") -> Dict[str, Any]:
        """Import memories from the specified format"""
        with custom_span("import_memories"):
            if format_type.lower() == "json":
                import_data = json.loads(data)
                imported_count = 0
                schema_count = 0
                
                # Import schemas first
                for schema in import_data.get("schemas", []):
                    schema_id = schema.get("id", str(uuid.uuid4()))
                    self.schemas[schema_id] = schema
                    schema_count += 1
                
                # Import memories
                for memory in import_data.get("memories", []):
                    memory_id = memory.get("id", str(uuid.uuid4()))
                    self.memories[memory_id] = memory
                    
                    # Update indices
                    memory_type = memory.get("memory_type", "observation")
                    if memory_type in self.type_index:
                        self.type_index[memory_type].add(memory_id)
                    else:
                        self.type_index[memory_type] = {memory_id}
                    
                    # Update other indices
                    memory_scope = memory.get("memory_scope", "game")
                    if memory_scope in self.scope_index:
                        self.scope_index[memory_scope].add(memory_id)
                    else:
                        self.scope_index[memory_scope] = {memory_id}
                    
                    # Update embedding index
                    if "embedding" in memory:
                        self.memory_embeddings[memory_id] = memory["embedding"]
                    
                    # Update archived status
                    if memory.get("is_archived", False):
                        self.archived_memories.add(memory_id)
                    
                    # Update consolidated status
                    if memory.get("is_consolidated", False):
                        self.consolidated_memories.add(memory_id)
                    
                    # Update schema index
                    for schema_ref in memory.get("metadata", {}).get("schemas", []):
                        schema_id = schema_ref.get("schema_id")
                        if schema_id:
                            if schema_id in self.schema_index:
                                self.schema_index[schema_id].add(memory_id)
                            else:
                                self.schema_index[schema_id] = {memory_id}
                    
                    imported_count += 1
                
                # Clear cache
                self.query_cache = {}
                
                return {
                    "memories_imported": imported_count,
                    "schemas_imported": schema_count
                }
            else:
                raise ValueError(f"Unsupported import format: {format_type}")

    # Experience-related methods
    
    def _get_confidence_marker(self, relevance: float) -> str:
        """Get confidence marker text based on relevance score"""
        for (min_val, max_val), marker in self.confidence_markers.items():
            if min_val <= relevance < max_val:
                return marker
        return "remember"  # Default
    
    def _get_timeframe_text(self, timestamp: Optional[str]) -> str:
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
    
    def _get_emotional_tone(self, emotional_context: Dict[str, Any]) -> str:
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
    
    def _get_scenario_tone(self, scenario_type: str) -> Optional[str]:
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
    
    # Schema methods
    
    @function_tool
    async def create_schema_from_memories(self,
                                        schema_name: str,
                                        description: str,
                                        category: str,
                                        memory_ids: List[str],
                                        attributes: Dict[str, Any] = None) -> str:
        """
        Create a schema based on a set of memories
        
        Args:
            schema_name: Name of the schema
            description: Description of the schema
            category: Category for the schema
            memory_ids: IDs of memories that exemplify this schema
            attributes: Optional attributes for the schema
            
        Returns:
            Schema ID
        """
        # Generate a unique ID
        schema_id = f"schema_{len(self.schemas) + 1}_{int(datetime.datetime.now().timestamp())}"
        
        # Create the schema
        schema = MemorySchema(
            id=schema_id,
            name=schema_name,
            description=description,
            category=category,
            attributes=attributes or {},
            example_memory_ids=memory_ids,
            creation_date=datetime.datetime.now().isoformat(),
            last_updated=datetime.datetime.now().isoformat()
        )
        
        # Store the schema
        self.schemas[schema_id] = schema
        
        # Update schema index
        for memory_id in memory_ids:
            if schema_id not in self.schema_index:
                self.schema_index[schema_id] = set()
            self.schema_index[schema_id].add(memory_id)
            
            # Update memory metadata
            memory = await self.get_memory(memory_id)
            if memory:
                metadata = memory.get("metadata", {})
                if "schemas" not in metadata:
                    metadata["schemas"] = []
                
                metadata["schemas"].append({
                    "schema_id": schema_id,
                    "relevance": 1.0  # High relevance for examples
                })
                
                await self.update_memory(
                    memory_id, {"metadata": metadata}
                )
        
        logger.info(f"Created schema {schema_id}: {schema_name}")
        return schema_id
    
    @function_tool
    async def detect_schema_from_memories(self, topic: str = None, min_memories: int = 3) -> Optional[Dict[str, Any]]:
        """
        Detect a potential schema from memories
        
        Args:
            topic: Optional topic to focus on
            min_memories: Minimum number of memories needed
            
        Returns:
            Detected schema information or None
        """
        # Retrieve relevant memories
        memories = await self.retrieve_memories(
            query=topic if topic else "important memory",
            limit=10
        )
        
        if len(memories) < min_memories:
            return None
        
        # Find patterns in the memories
        pattern = await self._detect_memory_pattern(memories)
        
        if not pattern:
            return None
        
        # Create a schema from the pattern
        schema_id = await self.create_schema_from_memories(
            schema_name=pattern["name"],
            description=pattern["description"],
            category=pattern["category"],
            memory_ids=[m["id"] for m in memories[:min_memories]],
            attributes=pattern["attributes"]
        )
        
        return {
            "schema_id": schema_id,
            "schema_name": pattern["name"],
            "description": pattern["description"],
            "memory_count": len(memories[:min_memories])
        }
    
    async def _detect_memory_pattern(self, memories: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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
    
    async def _check_for_patterns(self, tags: List[str]) -> None:
        """Check if there are patterns that should be identified as schemas"""
        # Skip if no tags
        if not tags:
            return
        
        # Check each tag for potential patterns
        for tag in tags:
            # Get memories with this tag
            tagged_memories = await self.retrieve_memories(
                query="",
                memory_types=["observation", "experience"],
                tags=[tag],
                limit=10
            )
            
            # Skip if not enough memories
            if len(tagged_memories) < 3:
                continue
            
            # Check if there might be a schema
            if tag not in [s.name.lower() for s in self.schemas.values()]:
                # Create schema detection task
                asyncio.create_task(self.detect_schema_from_memories(tag))
    
    async def _check_for_schema_creation(self) -> None:
        """Periodically check for new schemas that could be created"""
        # Get common tags
        tag_counts = {}
        for tag, memory_ids in self.tag_index.items():
            tag_counts[tag] = len(memory_ids)
        
        # Find tags with many memories but no schema
        potential_schema_tags = []
        for tag, count in tag_counts.items():
            if count >= 5:  # At least 5 memories with this tag
                # Check if tag is not already a schema name
                if not any(s.name.lower() == tag.lower() for s in self.schemas.values()):
                    potential_schema_tags.append(tag)
        
        # Process top potential schema tags
        for tag in potential_schema_tags[:3]:  # Process up to 3 potential schemas
            await self.detect_schema_from_memories(tag)
    
    # Semantic Memory methods
    
    @function_tool
    async def create_semantic_memory(self, source_memory_ids: List[str], abstraction_type: str = "pattern") -> Optional[str]:
        """
        Create a semantic memory (abstraction) from source memories
        
        Args:
            source_memory_ids: IDs of source memories
            abstraction_type: Type of abstraction to create
            
        Returns:
            Memory ID of created semantic memory or None
        """
        # Get the source memories
        source_memories = []
        for memory_id in source_memory_ids:
            memory = await self.get_memory(memory_id)
            if memory:
                source_memories.append(memory)
        
        if len(source_memories) < self.abstraction_threshold:
            return None
        
        # Generate the abstraction text
        abstraction_text = await self._generate_abstraction(source_memories, abstraction_type)
        
        if not abstraction_text:
            return None
        
        # Create the semantic memory
        memory_id = await self.add_memory(
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
            
            await self.update_memory(
                memory["id"], {"metadata": metadata}
            )
        
        return memory_id
    
    async def _generate_abstraction(self, memories: List[Dict[str, Any]], abstraction_type: str) -> Optional[str]:
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
    
    async def _memories_warrant_abstraction(self, memories: List[Dict[str, Any]]) -> bool:
        """Determine if a set of memories warrants creating an abstraction"""
        # Check if enough memories
        if len(memories) < self.abstraction_threshold:
            return False
        
        # Check recency - at least one recent memory
        recent_count = 0
        for memory in memories:
            metadata = memory.get("metadata", {})
            timestamp_str = metadata.get("timestamp", datetime.datetime.now().isoformat())
            timestamp = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            
            if (datetime.datetime.now() - timestamp).days < 14:  # Within last 2 weeks
                recent_count += 1
        
        if recent_count == 0:
            return False
        
        # Check similarity between memories
        similarity_pairs = 0
        comparison_count = 0
        
        for i in range(len(memories)):
            for j in range(i+1, len(memories)):
                comparison_count += 1
                
                # Calculate similarity between memory texts
                similarity = self._calculate_text_similarity(
                    memories[i]["memory_text"], 
                    memories[j]["memory_text"]
                )
                
                if similarity > 0.3:  # At least some similarity
                    similarity_pairs += 1
        
        # If enough pairs are similar
        if comparison_count > 0 and similarity_pairs / comparison_count > 0.3:
            return True
        
        return False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
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

    # API Functions
    
    @function_tool
    async def retrieve_memories_with_formatting(self, query: str, memory_types: List[str] = None, 
                                              limit: int = 5) -> List[Dict[str, Any]]:
        """
        Enhanced retrieval with formatted results for agent consumption.
        
        Args:
            query: Search query
            memory_types: Types of memories to include
            limit: Maximum number of results
        
        Returns:
            List of formatted memories with confidence markers
        """
        with custom_span("retrieve_memories_with_formatting", {"query": query, "limit": limit}):
            memories = await self.retrieve_memories(
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
    async def create_reflection_from_memories(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a reflection on a specific topic using memories.
        
        Args:
            topic: Optional topic to reflect on
        
        Returns:
            Reflection result dictionary
        """
        with custom_span("create_reflection", {"topic": topic or "general"}):
            # Get relevant memories for reflection
            query = topic if topic else "important memories"
            memories = await self.retrieve_memories(
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
            reflection_id = await self.add_memory(
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
    async def create_abstraction_from_memories(self, memory_ids: List[str], 
                                             pattern_type: str = "behavior") -> Dict[str, Any]:
        """
        Create a higher-level abstraction from specific memories.
        
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
                memory = await self.get_memory(memory_id)
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
            abstraction_id = await self.add_memory(
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
    async def construct_narrative_from_memories(self, topic: str, chronological: bool = True, 
                                              limit: int = 5) -> NarrativeResult:
        """
        Construct a narrative from memories about a topic.
        
        Args:
            topic: Topic for narrative construction
            chronological: Whether to maintain chronological order
            limit: Maximum number of memories to include
        
        Returns:
            Narrative result dictionary
        """
        with custom_span("construct_narrative", {"topic": topic, "chronological": chronological}):
            # Retrieve memories for the topic
            memories = await self.retrieve_memories(
                query=topic,
                memory_types=["observation", "experience"],
                limit=limit
            )
            
            if not memories:
                return NarrativeResult(
                    narrative=f"I don't have enough memories about {topic} to construct a narrative.",
                    confidence=0.2,
                    experience_count=0
                )
            
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
                timestamp = self._get_timeframe_text(memory.get("metadata", {}).get("timestamp", ""))
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
            
            return NarrativeResult(
                narrative=narrative,
                confidence=min(1.0, confidence),
                experience_count=len(memories)
            )
    
    # Experience retrieval methods
    
    @function_tool
    async def retrieve_relevant_experiences(self, 
                                          query: str,
                                          scenario_type: str = "",
                                          emotional_state: Dict[str, Any] = None,
                                          entities: List[str] = None,
                                          limit: int = 3,
                                          min_relevance: float = 0.6) -> List[Dict[str, Any]]:
        """
        Retrieve experiences relevant to the current conversation context.
        
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
        with custom_span("retrieve_relevant_experiences", {"query": query, "scenario_type": scenario_type}):
            # Create context dictionary
            current_context = {
                "query": query,
                "scenario_type": scenario_type,
                "emotional_state": emotional_state or {},
                "entities": entities or []
            }
            
            # Retrieve experiences using the memory system
            experiences = await self.retrieve_memories(
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
                    confidence_marker = self._get_confidence_marker(memory.get("relevance", 0.5))
                    
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
    async def generate_conversational_recall(self, 
                                           experience: Dict[str, Any],
                                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a natural, conversational recall of an experience.
        
        Args:
            experience: The experience to recall
            context: Current conversation context
            
        Returns:
            Conversational recall with reflection
        """
        with custom_span("generate_conversational_recall"):
            # Extract experience data
            content = experience.get("content", experience.get("memory_text", ""))
            emotional_context = experience.get("emotional_context", {})
            scenario_type = experience.get("scenario_type", "general")
            timestamp = experience.get("timestamp", experience.get("metadata", {}).get("timestamp"))
            fidelity = experience.get("fidelity", experience.get("metadata", {}).get("fidelity", 1.0))
            
            # Get timeframe text
            timeframe = self._get_timeframe_text(timestamp)
            
            # Determine tone for recall
            emotional_tone = self._get_emotional_tone(emotional_context)
            scenario_tone = self._get_scenario_tone(scenario_type)
            
            # Select tone (prioritize scenario tone if available)
            tone = scenario_tone or emotional_tone
            
            # Get templates for this tone
            templates = self.recall_templates.get(tone, self.recall_templates["standard"])
            
            # Select a random template
            template = random.choice(templates)
            
            # Generate components for template filling using a specialized agent
            recall_agent = Agent(
                name="Experience Recall Agent",
                instructions=f"""Create a natural, conversational recall of this experience.
                The recall should match a {tone} tone and feel authentic.
                Extract a brief summary, a relevant detail, and a thoughtful reflection from the experience."""
            )
            
            # Prepare prompt
            prompt = f"""Given this experience: "{content}"
            
            Create three components for a conversational recall:
            1. A brief summary (15-20 words)
            2. A relevant detail (10-15 words)
            3. A thoughtful reflection (15-20 words)
            
            Format your response as:
            Brief Summary: [brief summary]
            Detail: [detail]
            Reflection: [reflection]"""
            
            # Get result from agent
            result = await Runner.run(recall_agent, prompt)
            parts = result.final_output.strip().split("\n")
            
            brief_summary = content
            detail = "It was quite memorable."
            reflection = "I found it quite interesting to observe."
            
            # Parse the result
            for part in parts:
                if part.startswith("Brief Summary:"):
                    brief_summary = part.replace("Brief Summary:", "").strip()
                elif part.startswith("Detail:"):
                    detail = part.replace("Detail:", "").strip()
                elif part.startswith("Reflection:"):
                    reflection = part.replace("Reflection:", "").strip()
            
            # Add fidelity qualifier if memory is less reliable
            if fidelity < 0.7:
                reflection += f" Though, I'm not entirely certain about all the details."
            
            # Fill in the template
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
