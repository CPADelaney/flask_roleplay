# context/semantic_context.py

import numpy as np
import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Callable, Set, Union
from datetime import datetime, timedelta
import asyncpg
import tiktoken

# Try to import embedding libraries, with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False

try:
    import openai
    HAVE_OPENAI = True
except ImportError:
    HAVE_OPENAI = False

# Configure logging
logger = logging.getLogger(__name__)

class EmbeddingProvider:
    """Abstract base class for embedding providers"""
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        raise NotImplementedError("Subclasses must implement get_embedding")
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = await self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        a = np.array(embedding1)
        b = np.array(embedding2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Embedding provider using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not HAVE_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers is required for SentenceTransformerEmbedding")
        self.model = SentenceTransformer(model_name)
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding using sentence-transformers"""
        # Run in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, lambda: self.model.encode(text))
        return embedding.tolist()


class OpenAIEmbedding(EmbeddingProvider):
    """Embedding provider using OpenAI API"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        if not HAVE_OPENAI:
            raise ImportError("openai is required for OpenAIEmbedding")
        self.client = openai.AsyncClient(api_key=api_key)
        self.model = model
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding using OpenAI API"""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts more efficiently"""
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [data.embedding for data in response.data]


class LocalEmbedding(EmbeddingProvider):
    """Simple local embedding provider using word hashing (fallback)"""
    
    def __init__(self, dimensions: int = 512):
        self.dimensions = dimensions
        self.word_vectors = {}
    
    def _hash_word(self, word: str) -> List[float]:
        """Create a pseudorandom vector for a word"""
        np.random.seed(hash(word) % 2**32)
        return np.random.normal(0, 1, self.dimensions).tolist()
    
    def _get_word_vector(self, word: str) -> List[float]:
        """Get or create vector for a word"""
        if word not in self.word_vectors:
            self.word_vectors[word] = self._hash_word(word)
        return self.word_vectors[word]
    
    async def get_embedding(self, text: str) -> List[float]:
        """Create a simple embedding by averaging word vectors"""
        words = text.lower().split()
        if not words:
            return [0.0] * self.dimensions
        
        # Get vectors for each word
        vectors = [self._get_word_vector(word) for word in words]
        
        # Average the vectors
        embedding = np.mean(vectors, axis=0)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding.tolist()


class TokenCounter:
    """Utility to count tokens for context budgeting"""
    
    def __init__(self, model_name: str = "gpt-4"):
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except:
            # Fallback to cl100k_base encoding (used by newer models)
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        return len(self.encoding.encode(text))
    
    def count_tokens_from_object(self, obj: Any) -> int:
        """Count tokens in a JSON-serializable object"""
        if obj is None:
            return 0
        elif isinstance(obj, (int, float, bool)):
            return 1
        elif isinstance(obj, str):
            return self.count_tokens(obj)
        elif isinstance(obj, dict):
            # Count keys and values
            return sum(
                self.count_tokens(str(k)) + self.count_tokens_from_object(v)
                for k, v in obj.items()
            )
        elif isinstance(obj, (list, tuple)):
            # Count all elements
            return sum(self.count_tokens_from_object(item) for item in obj)
        else:
            # Try to convert to string
            return self.count_tokens(str(obj))


class ContextItem:
    """Represents a piece of context with its metadata and embeddings"""
    
    def __init__(
        self,
        item_id: str,
        content: Dict[str, Any],
        item_type: str,
        timestamp: Optional[datetime] = None,
        embedding: Optional[List[float]] = None,
        importance: float = 1.0,
        last_referenced: Optional[datetime] = None,
        reference_count: int = 0,
        tags: Optional[List[str]] = None
    ):
        self.item_id = item_id
        self.content = content
        self.item_type = item_type
        self.timestamp = timestamp or datetime.now()
        self.embedding = embedding
        self.importance = importance  # 0.0 to 1.0
        self.last_referenced = last_referenced or self.timestamp
        self.reference_count = reference_count
        self.tags = tags or []
        self.token_count = None  # Calculated later
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "item_id": self.item_id,
            "content": self.content,
            "item_type": self.item_type,
            "timestamp": self.timestamp.isoformat(),
            "embedding": self.embedding,
            "importance": self.importance,
            "last_referenced": self.last_referenced.isoformat(),
            "reference_count": self.reference_count,
            "tags": self.tags,
            "token_count": self.token_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextItem':
        """Create from dictionary"""
        timestamp = datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"]
        last_referenced = datetime.fromisoformat(data["last_referenced"]) if isinstance(data["last_referenced"], str) else data["last_referenced"]
        
        item = cls(
            item_id=data["item_id"],
            content=data["content"],
            item_type=data["item_type"],
            timestamp=timestamp,
            embedding=data.get("embedding"),
            importance=data.get("importance", 1.0),
            last_referenced=last_referenced,
            reference_count=data.get("reference_count", 0),
            tags=data.get("tags", [])
        )
        item.token_count = data.get("token_count")
        return item
    
    def reference(self) -> None:
        """Record a reference to this item"""
        self.last_referenced = datetime.now()
        self.reference_count += 1
    
    def get_recency_score(self) -> float:
        """
        Calculate how recent this item is (0.0 to 1.0)
        0.0 = very old, 1.0 = very recent
        """
        now = datetime.now()
        age_days = (now - self.timestamp).total_days if hasattr((now - self.timestamp), 'total_days') else (now - self.timestamp).days
        
        # Items from the past day get a high score
        if age_days < 1:
            return 1.0
        # Score decreases over 30 days
        elif age_days < 30:
            return 1.0 - (age_days / 30)
        else:
            return 0.0
    
    def get_relevance_score(self, query_embedding: List[float]) -> float:
        """
        Calculate semantic similarity to a query (0.0 to 1.0)
        0.0 = not relevant, 1.0 = highly relevant
        """
        if not self.embedding or not query_embedding:
            return 0.0
            
        # Calculate cosine similarity
        a = np.array(self.embedding)
        b = np.array(query_embedding)
        
        # Return 0 for zero vectors
        if np.all(a == 0) or np.all(b == 0):
            return 0.0
            
        similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Map from -1..1 to 0..1 (though in practice cosine sim is usually 0..1)
        return (similarity + 1) / 2
    
    def get_importance_score(self) -> float:
        """
        Get the importance score (0.0 to 1.0)
        """
        return self.importance
    
    def get_usage_score(self) -> float:
        """
        Calculate usage-based score (0.0 to 1.0)
        Higher score for more frequently referenced items
        """
        # Normalize reference count
        if self.reference_count == 0:
            return 0.0
        # Scale with diminishing returns
        return min(1.0, 0.5 + 0.5 * (1 - 1 / (1 + self.reference_count / 5)))
    
    def get_composite_score(self, query_embedding: Optional[List[float]] = None) -> float:
        """
        Calculate a composite score based on all factors
        
        Args:
            query_embedding: Optional query embedding for relevance calculation
            
        Returns:
            Composite score (0.0 to 1.0)
        """
        # Calculate individual scores
        recency = self.get_recency_score()
        importance = self.get_importance_score()
        usage = self.get_usage_score()
        
        # Calculate relevance if query is provided
        relevance = 0.5  # Default
        if query_embedding and self.embedding:
            relevance = self.get_relevance_score(query_embedding)
        
        # Weighted average (could be adjusted)
        weights = {
            'relevance': 0.40,  # Semantic relevance is most important
            'recency': 0.25,    # Recent items are important
            'importance': 0.20, # Intrinsic importance
            'usage': 0.15       # Usage patterns
        }
        
        score = (
            weights['relevance'] * relevance +
            weights['recency'] * recency +
            weights['importance'] * importance +
            weights['usage'] * usage
        )
        
        return score


class SemanticContextDatabase:
    """
    Semantic Context Database for efficient context retrieval
    with vector embeddings and adaptive prioritization.
    """
    
    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        token_counter: Optional[TokenCounter] = None,
        db_connection_string: Optional[str] = None
    ):
        # Set up embedding provider
        self.embedding_provider = embedding_provider or LocalEmbedding()
        
        # Set up token counter
        self.token_counter = token_counter or TokenCounter()
        
        # Set up storage
        self.db_connection_string = db_connection_string
        self.items: Dict[str, ContextItem] = {}
        self.item_embeddings: Dict[str, List[float]] = {}
        
        # Query caching
        self.query_cache: Dict[str, Tuple[List[ContextItem], datetime]] = {}
        self.cache_ttl = timedelta(minutes=5)
        
        # Index structures
        self.type_index: Dict[str, Set[str]] = {}
        self.tag_index: Dict[str, Set[str]] = {}
        self.timestamp_index: List[Tuple[datetime, str]] = []
        self.reference_index: List[Tuple[int, str]] = []
    
    async def initialize(self) -> None:
        """Initialize database connection if using PostgreSQL"""
        if self.db_connection_string:
            self.pool = await asyncpg.create_pool(self.db_connection_string)
            
            # Create tables if they don't exist
            async with self.pool.acquire() as conn:
                await conn.execute('''
                CREATE TABLE IF NOT EXISTS context_items (
                    item_id TEXT PRIMARY KEY,
                    content JSONB NOT NULL,
                    item_type TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    embedding JSONB,
                    importance REAL NOT NULL,
                    last_referenced TIMESTAMP NOT NULL,
                    reference_count INTEGER NOT NULL,
                    tags JSONB NOT NULL,
                    token_count INTEGER
                );
                
                CREATE INDEX IF NOT EXISTS context_items_type_idx ON context_items (item_type);
                CREATE INDEX IF NOT EXISTS context_items_timestamp_idx ON context_items (timestamp);
                CREATE INDEX IF NOT EXISTS context_items_last_referenced_idx ON context_items (last_referenced);
                ''')
    
    async def close(self) -> None:
        """Close database connection"""
        if hasattr(self, 'pool'):
            await self.pool.close()
    
    async def add_item(
        self,
        item_id: str,
        content: Dict[str, Any],
        item_type: str,
        timestamp: Optional[datetime] = None,
        importance: float = 1.0,
        tags: Optional[List[str]] = None,
        generate_embedding: bool = True
    ) -> ContextItem:
        """
        Add an item to the context database
        
        Args:
            item_id: Unique ID for the item
            content: The content dictionary
            item_type: Type of context item (npc, location, event, etc.)
            timestamp: When this item was created/updated
            importance: Importance score (0.0 to 1.0)
            tags: List of tags for filtering
            generate_embedding: Whether to generate an embedding
            
        Returns:
            The created ContextItem
        """
        # Create a summary text for embedding
        embedding = None
        if generate_embedding:
            summary_text = self._create_summary_text(content, item_type)
            embedding = await self.embedding_provider.get_embedding(summary_text)
        
        # Create the item
        item = ContextItem(
            item_id=item_id,
            content=content,
            item_type=item_type,
            timestamp=timestamp or datetime.now(),
            embedding=embedding,
            importance=importance,
            tags=tags or []
        )
        
        # Count tokens
        if self.token_counter:
            item.token_count = self.token_counter.count_tokens_from_object(content)
        
        # Store the item
        if self.db_connection_string:
            await self._store_item_in_db(item)
        else:
            # In-memory storage
            self.items[item_id] = item
            
            # Update indexes
            self._index_item(item)
        
        # Invalidate query cache
        self.query_cache = {}
        
        return item
    
    async def _store_item_in_db(self, item: ContextItem) -> None:
        """Store an item in the database"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
            INSERT INTO context_items (
                item_id, content, item_type, timestamp, embedding,
                importance, last_referenced, reference_count, tags, token_count
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (item_id) DO UPDATE SET
                content = $2,
                item_type = $3,
                timestamp = $4,
                embedding = $5,
                importance = $6,
                last_referenced = $7,
                reference_count = $8,
                tags = $9,
                token_count = $10
            ''', 
            item.item_id,
            json.dumps(item.content),
            item.item_type,
            item.timestamp,
            json.dumps(item.embedding) if item.embedding else None,
            item.importance,
            item.last_referenced,
            item.reference_count,
            json.dumps(item.tags),
            item.token_count
            )
    
    def _index_item(self, item: ContextItem) -> None:
        """Add an item to the in-memory indexes"""
        # Type index
        if item.item_type not in self.type_index:
            self.type_index[item.item_type] = set()
        self.type_index[item.item_type].add(item.item_id)
        
        # Tag index
        for tag in item.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(item.item_id)
        
        # Timestamp index (sorted by timestamp)
        self.timestamp_index.append((item.timestamp, item.item_id))
        self.timestamp_index.sort(key=lambda x: x[0], reverse=True)
        
        # Reference index (sorted by reference count)
        self.reference_index.append((item.reference_count, item.item_id))
        self.reference_index.sort(key=lambda x: x[0], reverse=True)
    
    def _create_summary_text(self, content: Dict[str, Any], item_type: str) -> str:
        """Create a summary text for embedding generation"""
        # Create a text representation based on the item type
        summary_parts = []
        
        # Add type-specific information
        summary_parts.append(f"Type: {item_type}")
        
        # Extract key content based on type
        if item_type == "npc":
            if "npc_name" in content:
                summary_parts.append(f"Name: {content['npc_name']}")
            if "physical_description" in content:
                summary_parts.append(f"Description: {content['physical_description']}")
            for field in ["personality_traits", "hobbies", "likes", "dislikes"]:
                if field in content and content[field]:
                    items = content[field]
                    if isinstance(items, list):
                        items_text = ", ".join(str(item) for item in items)
                    else:
                        items_text = str(items)
                    summary_parts.append(f"{field.replace('_', ' ').title()}: {items_text}")
            
        elif item_type == "location":
            if "location_name" in content:
                summary_parts.append(f"Name: {content['location_name']}")
            if "description" in content:
                summary_parts.append(f"Description: {content['description']}")
            
        elif item_type == "event":
            if "event_name" in content:
                summary_parts.append(f"Event: {content['event_name']}")
            if "description" in content:
                summary_parts.append(f"Description: {content['description']}")
            if "location" in content:
                summary_parts.append(f"Location: {content['location']}")
            
        elif item_type == "memory":
            if "memory_text" in content:
                summary_parts.append(f"Memory: {content['memory_text']}")
            if "emotion" in content:
                summary_parts.append(f"Emotion: {content['emotion']}")
            
        elif item_type == "quest":
            if "quest_name" in content:
                summary_parts.append(f"Quest: {content['quest_name']}")
            if "description" in content:
                summary_parts.append(f"Description: {content['description']}")
            if "status" in content:
                summary_parts.append(f"Status: {content['status']}")
            
        # For conflicts
        elif item_type == "conflict":
            if "conflict_name" in content:
                summary_parts.append(f"Conflict: {content['conflict_name']}")
            if "description" in content:
                summary_parts.append(f"Description: {content['description']}")
            if "faction_a_name" in content and "faction_b_name" in content:
                summary_parts.append(f"Factions: {content['faction_a_name']} vs {content['faction_b_name']}")
        
        # Default case: include important fields
        else:
            important_fields = ["name", "title", "description", "text", "summary"]
            for field in important_fields:
                if field in content:
                    summary_parts.append(f"{field.title()}: {content[field]}")
        
        # Include any tags
        if "tags" in content and content["tags"]:
            tags = content["tags"]
            if isinstance(tags, list):
                tags_text = ", ".join(str(tag) for tag in tags)
            else:
                tags_text = str(tags)
            summary_parts.append(f"Tags: {tags_text}")
        
        return "\n".join(summary_parts)
    
    async def update_item(
        self,
        item_id: str,
        content: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        regenerate_embedding: bool = True
    ) -> Optional[ContextItem]:
        """
        Update an existing item in the context database
        
        Args:
            item_id: ID of the item to update
            content: New content (if changed)
            importance: New importance score (if changed)
            tags: New tags (if changed)
            regenerate_embedding: Whether to regenerate embedding if content changed
            
        Returns:
            Updated item or None if not found
        """
        # Get the existing item
        item = await self.get_item(item_id)
        if not item:
            return None
        
        # Update content if provided
        if content is not None:
            item.content = content
            
            # Regenerate embedding if needed
            if regenerate_embedding:
                summary_text = self._create_summary_text(content, item.item_type)
                item.embedding = await self.embedding_provider.get_embedding(summary_text)
            
            # Recount tokens
            if self.token_counter:
                item.token_count = self.token_counter.count_tokens_from_object(content)
        
        # Update importance if provided
        if importance is not None:
            item.importance = importance
        
        # Update tags if provided
        if tags is not None:
            item.tags = tags
        
        # Update timestamp
        item.timestamp = datetime.now()
        
        # Store the updated item
        if self.db_connection_string:
            await self._store_item_in_db(item)
        else:
            # In-memory storage
            self.items[item_id] = item
            
            # Update indexes
            self._update_indexes(item)
        
        # Invalidate query cache
        self.query_cache = {}
        
        return item
    
    def _update_indexes(self, item: ContextItem) -> None:
        """Update in-memory indexes for an item"""
        # Type index should already have this item
        
        # Update tag index
        for tag in list(self.tag_index.keys()):
            if tag in item.tags:
                if item.item_id not in self.tag_index[tag]:
                    self.tag_index[tag].add(item.item_id)
            else:
                # Remove from tag if no longer associated
                if item.item_id in self.tag_index[tag]:
                    self.tag_index[tag].remove(item.item_id)
        
        # Update timestamp index
        self.timestamp_index = [(ts, id) for ts, id in self.timestamp_index if id != item.item_id]
        self.timestamp_index.append((item.timestamp, item.item_id))
        self.timestamp_index.sort(key=lambda x: x[0], reverse=True)
        
        # Update reference index
        self.reference_index = [(rc, id) for rc, id in self.reference_index if id != item.item_id]
        self.reference_index.append((item.reference_count, item.item_id))
        self.reference_index.sort(key=lambda x: x[0], reverse=True)
    
    async def get_item(self, item_id: str) -> Optional[ContextItem]:
        """Get an item by ID"""
        if self.db_connection_string:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow('''
                SELECT 
                    item_id, content, item_type, timestamp, embedding,
                    importance, last_referenced, reference_count, tags, token_count
                FROM context_items
                WHERE item_id = $1
                ''', item_id)
                
                if row:
                    return ContextItem.from_dict({
                        "item_id": row["item_id"],
                        "content": json.loads(row["content"]),
                        "item_type": row["item_type"],
                        "timestamp": row["timestamp"],
                        "embedding": json.loads(row["embedding"]) if row["embedding"] else None,
                        "importance": row["importance"],
                        "last_referenced": row["last_referenced"],
                        "reference_count": row["reference_count"],
                        "tags": json.loads(row["tags"]),
                        "token_count": row["token_count"]
                    })
                return None
        else:
            return self.items.get(item_id)
    
    async def remove_item(self, item_id: str) -> bool:
        """Remove an item from the context database"""
        if self.db_connection_string:
            async with self.pool.acquire() as conn:
                result = await conn.execute('''
                DELETE FROM context_items
                WHERE item_id = $1
                ''', item_id)
                
                success = "DELETE 1" in result
        else:
            if item_id in self.items:
                item = self.items[item_id]
                
                # Remove from type index
                if item.item_type in self.type_index and item_id in self.type_index[item.item_type]:
                    self.type_index[item.item_type].remove(item_id)
                
                # Remove from tag index
                for tag in item.tags:
                    if tag in self.tag_index and item_id in self.tag_index[tag]:
                        self.tag_index[tag].remove(item_id)
                
                # Remove from timestamp index
                self.timestamp_index = [(ts, id) for ts, id in self.timestamp_index if id != item_id]
                
                # Remove from reference index
                self.reference_index = [(rc, id) for rc, id in self.reference_index if id != item_id]
                
                # Remove from items
                del self.items[item_id]
                success = True
            else:
                success = False
        
        # Invalidate query cache
        self.query_cache = {}
        
        return success
    
    async def reference_item(self, item_id: str) -> bool:
        """
        Record a reference to an item
        
        Args:
            item_id: ID of the referenced item
            
        Returns:
            Whether the item was found and referenced
        """
        item = await self.get_item(item_id)
        if not item:
            return False
        
        # Update reference stats
        item.reference()
        
        # Store the updated item
        if self.db_connection_string:
            async with self.pool.acquire() as conn:
                await conn.execute('''
                UPDATE context_items
                SET last_referenced = $1, reference_count = $2
                WHERE item_id = $3
                ''', item.last_referenced, item.reference_count, item_id)
        else:
            # Update reference index
            self.reference_index = [(rc, id) for rc, id in self.reference_index if id != item_id]
            self.reference_index.append((item.reference_count, item_id))
            self.reference_index.sort(key=lambda x: x[0], reverse=True)
        
        return True
    
    async def search(
        self,
        query: str,
        item_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        max_tokens: Optional[int] = None,
        min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search for context items using semantic search
        
        Args:
            query: Search query text
            item_types: Optional list of item types to filter
            tags: Optional list of tags to filter
            limit: Maximum number of results
            max_tokens: Maximum total tokens to return
            min_score: Minimum relevance score (0.0 to 1.0)
            
        Returns:
            List of matching items with scores
        """
        # Check cache first
        cache_key = f"{query}::{':'.join(item_types or [])}::{':'.join(tags or [])}::{limit}::{max_tokens}"
        if cache_key in self.query_cache:
            results, timestamp = self.query_cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return [
                    {
                        "item": item.content,
                        "item_id": item.item_id,
                        "item_type": item.item_type,
                        "score": score,
                        "tokens": item.token_count or 0
                    } 
                    for item, score in results
                ]
        
        # Generate query embedding
        query_embedding = await self.embedding_provider.get_embedding(query)
        
        # Get candidate items
        candidate_items = await self._get_candidate_items(item_types, tags)
        
        # Score candidates
        scored_items = []
        for item in candidate_items:
            score = item.get_composite_score(query_embedding)
            if score >= min_score:
                scored_items.append((item, score))
        
        # Sort by score
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Apply token limit if specified
        if max_tokens is not None and self.token_counter:
            results = []
            token_count = 0
            
            for item, score in scored_items:
                item_tokens = item.token_count or self.token_counter.count_tokens_from_object(item.content)
                
                if token_count + item_tokens <= max_tokens:
                    results.append((item, score))
                    token_count += item_tokens
                    
                    # Stop if we've reached the limit
                    if len(results) >= limit:
                        break
                # Consider adding smaller items even if a large one doesn't fit
                elif len(results) < limit:
                    continue
                else:
                    break
        else:
            # Just apply the count limit
            results = scored_items[:limit]
        
        # Update cache
        self.query_cache[cache_key] = (results, datetime.now())
        
        # Format results
        return [
            {
                "item": item.content,
                "item_id": item.item_id,
                "item_type": item.item_type,
                "score": score,
                "tokens": item.token_count or 0
            } 
            for item, score in results
        ]
    
    async def _get_candidate_items(
        self,
        item_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> List[ContextItem]:
        """Get candidate items for search based on filters"""
        if self.db_connection_string:
            # Build query conditions
            conditions = []
            params = []
            
            if item_types:
                placeholders = ','.join(f'${i+1}' for i in range(len(item_types)))
                conditions.append(f"item_type IN ({placeholders})")
                params.extend(item_types)
            
            # Handle tags with array containment
            if tags:
                for tag in tags:
                    param_index = len(params) + 1
                    conditions.append(f"tags @> $::{param_index}::jsonb")
                    params.append(json.dumps([tag]))
            
            # Build the query
            query = '''
            SELECT 
                item_id, content, item_type, timestamp, embedding,
                importance, last_referenced, reference_count, tags, token_count
            FROM context_items
            '''
            
            if conditions:
                query += f" WHERE {' AND '.join(conditions)}"
            
            # Execute query
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                return [ContextItem.from_dict({
                    "item_id": row["item_id"],
                    "content": json.loads(row["content"]),
                    "item_type": row["item_type"],
                    "timestamp": row["timestamp"],
                    "embedding": json.loads(row["embedding"]) if row["embedding"] else None,
                    "importance": row["importance"],
                    "last_referenced": row["last_referenced"],
                    "reference_count": row["reference_count"],
                    "tags": json.loads(row["tags"]),
                    "token_count": row["token_count"]
                }) for row in rows]
        else:
            # In-memory filtering
            candidate_ids = set()
            
            if item_types:
                # Get IDs matching any of the types
                type_ids = set()
                for item_type in item_types:
                    if item_type in self.type_index:
                        type_ids.update(self.type_index[item_type])
                
                if not candidate_ids:
                    candidate_ids = type_ids
                else:
                    candidate_ids &= type_ids
            else:
                # All items are candidates
                candidate_ids = set(self.items.keys())
            
            if tags:
                # Get IDs matching all tags
                tag_ids = None
                for tag in tags:
                    ids = self.tag_index.get(tag, set())
                    if tag_ids is None:
                        tag_ids = ids
                    else:
                        tag_ids &= ids
                
                if tag_ids is not None:
                    candidate_ids &= tag_ids
            
            # Get the actual items
            return [self.items[item_id] for item_id in candidate_ids]
    
    async def get_recent_items(
        self,
        item_types: Optional[List[str]] = None,
        days: int = 7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent context items
        
        Args:
            item_types: Optional list of item types to filter
            days: Number of days to look back
            limit: Maximum number of results
            
        Returns:
            List of recent items
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        if self.db_connection_string:
            # Build query conditions
            conditions = ["timestamp >= $1"]
            params = [cutoff_date]
            
            if item_types:
                placeholders = ','.join(f'${i+2}' for i in range(len(item_types)))
                conditions.append(f"item_type IN ({placeholders})")
                params.extend(item_types)
            
            # Build the query
            query = '''
            SELECT 
                item_id, content, item_type, timestamp, embedding,
                importance, last_referenced, reference_count, tags, token_count
            FROM context_items
            WHERE {}
            ORDER BY timestamp DESC
            LIMIT ${}
            '''.format(' AND '.join(conditions), len(params) + 1)
            
            params.append(limit)
            
            # Execute query
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                return [{
                    "item": json.loads(row["content"]),
                    "item_id": row["item_id"],
                    "item_type": row["item_type"],
                    "timestamp": row["timestamp"].isoformat(),
                    "tokens": row["token_count"] or 0
                } for row in rows]
        else:
            # In-memory filtering
            candidate_ids = set()
            
            # Filter by type first if specified
            if item_types:
                for item_type in item_types:
                    if item_type in self.type_index:
                        candidate_ids.update(self.type_index[item_type])
            else:
                candidate_ids = set(self.items.keys())
            
            # Get the recent items from the timestamp index
            recent_items = []
            for timestamp, item_id in self.timestamp_index:
                if timestamp >= cutoff_date and item_id in candidate_ids:
                    item = self.items[item_id]
                    recent_items.append({
                        "item": item.content,
                        "item_id": item.item_id,
                        "item_type": item.item_type,
                        "timestamp": item.timestamp.isoformat(),
                        "tokens": item.token_count or 0
                    })
                    
                    if len(recent_items) >= limit:
                        break
            
            return recent_items
    
    async def get_frequently_referenced_items(
        self,
        item_types: Optional[List[str]] = None,
        min_references: int = 2,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get frequently referenced context items
        
        Args:
            item_types: Optional list of item types to filter
            min_references: Minimum number of references
            limit: Maximum number of results
            
        Returns:
            List of frequently referenced items
        """
        if self.db_connection_string:
            # Build query conditions
            conditions = ["reference_count >= $1"]
            params = [min_references]
            
            if item_types:
                placeholders = ','.join(f'${i+2}' for i in range(len(item_types)))
                conditions.append(f"item_type IN ({placeholders})")
                params.extend(item_types)
            
            # Build the query
            query = '''
            SELECT 
                item_id, content, item_type, timestamp, embedding,
                importance, last_referenced, reference_count, tags, token_count
            FROM context_items
            WHERE {}
            ORDER BY reference_count DESC, last_referenced DESC
            LIMIT ${}
            '''.format(' AND '.join(conditions), len(params) + 1)
            
            params.append(limit)
            
            # Execute query
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                return [{
                    "item": json.loads(row["content"]),
                    "item_id": row["item_id"],
                    "item_type": row["item_type"],
                    "reference_count": row["reference_count"],
                    "last_referenced": row["last_referenced"].isoformat(),
                    "tokens": row["token_count"] or 0
                } for row in rows]
        else:
            # In-memory filtering
            candidate_ids = set()
            
            # Filter by type first if specified
            if item_types:
                for item_type in item_types:
                    if item_type in self.type_index:
                        candidate_ids.update(self.type_index[item_type])
            else:
                candidate_ids = set(self.items.keys())
            
            # Get the frequently referenced items
            freq_items = []
            for ref_count, item_id in self.reference_index:
                if ref_count >= min_references and item_id in candidate_ids:
                    item = self.items[item_id]
                    freq_items.append({
                        "item": item.content,
                        "item_id": item.item_id,
                        "item_type": item.item_type,
                        "reference_count": item.reference_count,
                        "last_referenced": item.last_referenced.isoformat(),
                        "tokens": item.token_count or 0
                    })
                    
                    if len(freq_items) >= limit:
                        break
            
            return freq_items
    
    async def get_optimal_context(
        self,
        query: str,
        context_budget: int = 4000,
        item_types: Optional[Dict[str, int]] = None,
        tags: Optional[List[str]] = None,
        include_recent: bool = True,
        include_referenced: bool = True,
        min_score: float = 0.3
    ) -> Dict[str, Any]:
        """
        Get an optimal context tailored to the query within token budget
        
        Args:
            query: Query text to contextualize
            context_budget: Maximum tokens for the context
            item_types: Dict mapping item types to their limits
            tags: Optional list of tags to filter
            include_recent: Whether to include recent items
            include_referenced: Whether to include frequently referenced items
            min_score: Minimum relevance score
            
        Returns:
            Dictionary with optimized context
        """
        if not self.token_counter:
            raise ValueError("Token counter is required for optimal context retrieval")
        
        # Default item types if not provided
        if item_types is None:
            item_types = {
                "npc": 5,
                "location": 3,
                "quest": 5,
                "memory": 5,
                "conflict": 3,
                "event": 3
            }
        
        # Generate query embedding once
        query_embedding = await self.embedding_provider.get_embedding(query)
        
        # Get candidates for each item type
        candidates_by_type = {}
        token_usage = 0
        selected_items = []
        
        # Helper to add a selected item
        def add_selected_item(item, score, item_type):
            nonlocal token_usage
            selected_items.append({
                "item": item.content,
                "item_id": item.item_id,
                "item_type": item.item_type,
                "score": score,
                "tokens": item.token_count or 0
            })
            token_usage += item.token_count or 0
        
        # 1. Get relevant items for each type
        for item_type, limit in item_types.items():
            # Skip if we've exhausted the budget
            remaining_budget = context_budget - token_usage
            if remaining_budget <= 0:
                break
                
            type_limit = min(limit, 20)  # Get more candidates than needed
            
            # Get candidates for this type
            results = await self.search(
                query=query,
                item_types=[item_type],
                tags=tags,
                limit=type_limit,
                min_score=min_score
            )
            
            # Add to candidates
            candidates_by_type[item_type] = [
                (await self.get_item(result["item_id"]), result["score"])
                for result in results
            ]
        
        # 2. Select highest scoring items within budget
        for item_type, limit in item_types.items():
            # Skip if we've exhausted the budget
            remaining_budget = context_budget - token_usage
            if remaining_budget <= 0:
                break
                
            # Get candidates for this type
            candidates = candidates_by_type.get(item_type, [])
            
            # Sort by score
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Select top items within budget
            selected_count = 0
            for item, score in candidates:
                item_tokens = item.token_count or 0
                
                # Skip if item is too large
                if item_tokens > remaining_budget:
                    continue
                
                # Add item
                add_selected_item(item, score, item_type)
                selected_count += 1
                
                # Update remaining budget
                remaining_budget = context_budget - token_usage
                
                # Break if we've selected enough
                if selected_count >= limit or remaining_budget <= 0:
                    break
        
        # 3. Add recent items if requested
        if include_recent and token_usage < context_budget:
            recent_limit = min(5, (context_budget - token_usage) // 200)
            if recent_limit > 0:
                recent_items = await self.get_recent_items(days=1, limit=recent_limit)
                
                for result in recent_items:
                    # Skip if already selected
                    if any(s["item_id"] == result["item_id"] for s in selected_items):
                        continue
                        
                    item = await self.get_item(result["item_id"])
                    if item:
                        # Get score
                        score = item.get_composite_score(query_embedding)
                        
                        # Skip if below threshold
                        if score < min_score:
                            continue
                            
                        # Check budget
                        item_tokens = item.token_count or 0
                        if token_usage + item_tokens <= context_budget:
                            add_selected_item(item, score, item.item_type)
        
        # 4. Add frequently referenced items if requested
        if include_referenced and token_usage < context_budget:
            ref_limit = min(3, (context_budget - token_usage) // 200)
            if ref_limit > 0:
                ref_items = await self.get_frequently_referenced_items(min_references=3, limit=ref_limit)
                
                for result in ref_items:
                    # Skip if already selected
                    if any(s["item_id"] == result["item_id"] for s in selected_items):
                        continue
                        
                    item = await self.get_item(result["item_id"])
                    if item:
                        # Get score
                        score = item.get_composite_score(query_embedding)
                        
                        # Skip if below threshold
                        if score < min_score:
                            continue
                            
                        # Check budget
                        item_tokens = item.token_count or 0
                        if token_usage + item_tokens <= context_budget:
                            add_selected_item(item, score, item.item_type)
        
        # Sort by score
        selected_items.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "context_items": selected_items,
            "token_usage": token_usage,
            "context_budget": context_budget,
            "remaining_budget": context_budget - token_usage,
            "query": query
        }


class RPGContextManager:
    """Context manager for RPG-specific context retrieval"""
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        db_connection_string: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Set up embedding provider
        if openai_api_key and HAVE_OPENAI:
            self.embedding_provider = OpenAIEmbedding(openai_api_key)
        elif HAVE_SENTENCE_TRANSFORMERS:
            self.embedding_provider = SentenceTransformerEmbedding()
        else:
            self.embedding_provider = LocalEmbedding()
        
        # Set up token counter
        self.token_counter = TokenCounter()
        
        # Set up context database
        self.context_db = SemanticContextDatabase(
            embedding_provider=self.embedding_provider,
            token_counter=self.token_counter,
            db_connection_string=db_connection_string
        )
    
    async def initialize(self):
        """Initialize the context manager"""
        await self.context_db.initialize()
    
    async def close(self):
        """Close the context manager"""
        await self.context_db.close()
    
    async def add_npc_to_context(self, npc_data: Dict[str, Any], importance: float = 1.0) -> str:
        """Add an NPC to the context database"""
        npc_id = str(npc_data.get("npc_id", 0))
        item_id = f"npc:{self.user_id}:{self.conversation_id}:{npc_id}"
        
        await self.context_db.add_item(
            item_id=item_id,
            content=npc_data,
            item_type="npc",
            importance=importance,
            tags=["npc", npc_data.get("npc_name", "unknown")]
        )
        
        return item_id
    
    async def add_location_to_context(self, location_data: Dict[str, Any], importance: float = 1.0) -> str:
        """Add a location to the context database"""
        location_id = str(location_data.get("location_id", 0))
        item_id = f"location:{self.user_id}:{self.conversation_id}:{location_id}"
        
        await self.context_db.add_item(
            item_id=item_id,
            content=location_data,
            item_type="location",
            importance=importance,
            tags=["location", location_data.get("location_name", "unknown")]
        )
        
        return item_id
    
    async def add_memory_to_context(self, memory_data: Dict[str, Any], importance: float = 1.0) -> str:
        """Add a memory to the context database"""
        memory_id = str(memory_data.get("id", 0))
        item_id = f"memory:{self.user_id}:{self.conversation_id}:{memory_id}"
        
        # Generate tags based on content
        tags = ["memory"]
        if "type" in memory_data:
            tags.append(memory_data["type"])
        
        await self.context_db.add_item(
            item_id=item_id,
            content=memory_data,
            item_type="memory",
            importance=importance,
            tags=tags
        )
        
        return item_id
    
    async def add_conflict_to_context(self, conflict_data: Dict[str, Any], importance: float = 1.0) -> str:
        """Add a conflict to the context database"""
        conflict_id = str(conflict_data.get("conflict_id", 0))
        item_id = f"conflict:{self.user_id}:{self.conversation_id}:{conflict_id}"
        
        # Generate tags based on content
        tags = ["conflict", conflict_data.get("conflict_type", "unknown")]
        if "faction_a_name" in conflict_data:
            tags.append(conflict_data["faction_a_name"])
        if "faction_b_name" in conflict_data:
            tags.append(conflict_data["faction_b_name"])
        
        await self.context_db.add_item(
            item_id=item_id,
            content=conflict_data,
            item_type="conflict",
            importance=importance,
            tags=tags
        )
        
        return item_id
    
    async def add_quest_to_context(self, quest_data: Dict[str, Any], importance: float = 1.0) -> str:
        """Add a quest to the context database"""
        quest_id = str(quest_data.get("quest_id", 0))
        item_id = f"quest:{self.user_id}:{self.conversation_id}:{quest_id}"
        
        # Generate tags based on content
        tags = ["quest", quest_data.get("status", "unknown")]
        
        await self.context_db.add_item(
            item_id=item_id,
            content=quest_data,
            item_type="quest",
            importance=importance,
            tags=tags
        )
        
        return item_id
    
    async def get_relevant_npcs(self, input_text: str, location: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Get NPCs relevant to the input text and/or location"""
        # Create a combined query
        query = input_text
        if location:
            query += f" Location: {location}"
        
        # Search for relevant NPCs
        results = await self.context_db.search(
            query=query,
            item_types=["npc"],
            limit=limit
        )
        
        return [result["item"] for result in results]
    
    async def get_relevant_memories(self, input_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get memories relevant to the input text"""
        results = await self.context_db.search(
            query=input_text,
            item_types=["memory"],
            limit=limit
        )
        
        # Record references to these memories
        for result in results:
            await self.context_db.reference_item(result["item_id"])
        
        return [result["item"] for result in results]
    
    async def get_active_conflicts(self, input_text: Optional[str] = None, limit: int = 3) -> List[Dict[str, Any]]:
        """Get active conflicts, optionally filtered by relevance to input text"""
        if input_text:
            # Search by relevance
            results = await self.context_db.search(
                query=input_text,
                item_types=["conflict"],
                tags=["active"],
                limit=limit
            )
            
            return [result["item"] for result in results]
        else:
            # Get all active conflicts
            results = await self.context_db.search(
                query="active conflict",
                item_types=["conflict"],
                tags=["active"],
                limit=limit
            )
            
            return [result["item"] for result in results]
    
    async def get_context_for_agent(
        self,
        agent_type: str,
        input_text: str,
        location: Optional[str] = None,
        context_budget: int = 4000
    ) -> Dict[str, Any]:
        """
        Get optimized context for a specific agent
        
        Args:
            agent_type: Type of agent (narrator, npc_handler, etc.)
            input_text: Current user input
            location: Current location
            context_budget: Maximum tokens for context
            
        Returns:
            Optimized context for the agent
        """
        # Set up item type limits based on agent type
        if agent_type == "narrator":
            item_types = {
                "npc": 5,
                "location": 2,
                "memory": 5,
                "conflict": 3,
                "quest": 3,
                "event": 2
            }
        elif agent_type == "npc_handler":
            item_types = {
                "npc": 10,
                "location": 1,
                "memory": 3,
                "conflict": 0,
                "quest": 0,
                "event": 0
            }
        elif agent_type == "conflict_analyst":
            item_types = {
                "npc": 3,
                "location": 1,
                "memory": 2,
                "conflict": 10,
                "quest": 2,
                "event": 0
            }
        else:  # Default balanced context
            item_types = {
                "npc": 4,
                "location": 2,
                "memory": 4,
                "conflict": 3,
                "quest": 3,
                "event": 2
            }
        
        # Create a combined query
        query = input_text
        if location:
            query += f" Location: {location}"
        
        # Get optimal context
        context = await self.context_db.get_optimal_context(
            query=query,
            context_budget=context_budget,
            item_types=item_types,
            include_recent=True,
            include_referenced=True
        )
        
        # Format context by type
        formatted_context = {
            "npcs": [],
            "locations": [],
            "memories": [],
            "conflicts": [],
            "quests": [],
            "events": []
        }
        
        for item in context["context_items"]:
            item_type = item["item_type"]
            
            if item_type == "npc":
                formatted_context["npcs"].append(item["item"])
            elif item_type == "location":
                formatted_context["locations"].append(item["item"])
            elif item_type == "memory":
                formatted_context["memories"].append(item["item"])
            elif item_type == "conflict":
                formatted_context["conflicts"].append(item["item"])
            elif item_type == "quest":
                formatted_context["quests"].append(item["item"])
            elif item_type == "event":
                formatted_context["events"].append(item["item"])
        
        return {
            "context": formatted_context,
            "token_usage": context["token_usage"],
            "context_budget": context["context_budget"],
            "input_text": input_text,
            "location": location
        }

# Example usage

async def example_usage():
    """Example usage of the RPG context manager"""
    # Initialize context manager
    rpg_context = RPGContextManager(
        user_id=123,
        conversation_id=456,
        db_connection_string=None  # Use in-memory storage for example
    )
    
    await rpg_context.initialize()
    
    try:
        # Add NPCs
        await rpg_context.add_npc_to_context({
            "npc_id": 1,
            "npc_name": "Mistress Victoria",
            "dominance": 95,
            "cruelty": 60,
            "physical_description": "Tall woman with piercing blue eyes and elegant posture.",
            "personality_traits": ["commanding", "intelligent", "manipulative", "composed"],
            "current_location": "Mansion"
        }, importance=0.9)
        
        await rpg_context.add_npc_to_context({
            "npc_id": 2,
            "npc_name": "Dr. Elena",
            "dominance": 85,
            "cruelty": 40,
            "physical_description": "Middle-aged woman with glasses and a perpetual knowing smile.",
            "personality_traits": ["analytical", "patient", "methodical", "caring"],
            "current_location": "Clinic"
        }, importance=0.8)
        
        # Add locations
        await rpg_context.add_location_to_context({
            "location_id": 1,
            "location_name": "Mansion",
            "description": "A grand Victorian mansion on the outskirts of town.",
            "connected_locations": ["Town", "Garden"]
        }, importance=0.8)
        
        await rpg_context.add_location_to_context({
            "location_id": 2,
            "location_name": "Clinic",
            "description": "A modern medical clinic with a private research wing.",
            "connected_locations": ["Town"]
        }, importance=0.7)
        
        # Add memories
        await rpg_context.add_memory_to_context({
            "id": 1,
            "memory_text": "Victoria smiled at me in a way that made me feel both comforted and uneasy.",
            "type": "observation",
            "importance": "high"
        }, importance=0.9)
        
        # Add conflicts
        await rpg_context.add_conflict_to_context({
            "conflict_id": 1,
            "conflict_name": "Town Council Dispute",
            "conflict_type": "standard",
            "description": "A dispute over new regulations affecting local businesses.",
            "faction_a_name": "Business Owners",
            "faction_b_name": "Town Council",
            "phase": "active",
            "progress": 35
        }, importance=0.7)
        
        # Add quests
        await rpg_context.add_quest_to_context({
            "quest_id": 1,
            "quest_name": "Victoria's Request",
            "status": "active",
            "description": "Victoria has asked you to retrieve a special item from the town library.",
            "quest_giver": "Mistress Victoria"
        }, importance=0.8)
        
        # Get context for different agent types
        input_text = "I want to talk to Victoria about the item she requested."
        location = "Mansion"
        
        narrator_ctx = await rpg_context.get_context_for_agent(
            agent_type="narrator",
            input_text=input_text,
            location=location
        )
        
        npc_ctx = await rpg_context.get_context_for_agent(
            agent_type="npc_handler",
            input_text=input_text,
            location=location
        )
        
        # Get relevant NPCs for the input
        relevant_npcs = await rpg_context.get_relevant_npcs(input_text, location)
        
        # Get relevant memories
        relevant_memories = await rpg_context.get_relevant_memories(input_text)
        
        # Get active conflicts
        active_conflicts = await rpg_context.get_active_conflicts()
        
        print(f"Found {len(relevant_npcs)} relevant NPCs")
        print(f"Found {len(relevant_memories)} relevant memories")
        print(f"Found {len(active_conflicts)} active conflicts")
        print(f"Narrator context uses {narrator_ctx['token_usage']} tokens")
        print(f"NPC handler context uses {npc_ctx['token_usage']} tokens")
        
    finally:
        await rpg_context.close()

# Run the example
if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
